"""End-to-end smoke test for `ppo_update` after the dreamer4-aligned rewrite.

Exercises the PMPO surrogate + reverse-KL + distributional value head: builds
a tiny synthetic batch that mirrors the keys `_stack_trajectories` produces,
runs one PPO update, and asserts the returned metrics are finite. Also keeps
a regression guard on the log_prob recompute invariant (ratio ≈ 1 on epoch 0).
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from torch.distributions import Beta, Categorical, kl_divergence

from owars.policies.config import OrbitPolicyConfig
from owars.policies.features import EncodedObs, MAX_FLEETS, MAX_PLANETS
from owars.policies.model import HLGaussLoss, OrbitPolicy
from owars.policies.sampling import sample_batch_with_records
from owars.training.ppo import (
    _conditional_action_entropy,
    _conditional_fraction_kl,
    _fixed_minibatches,
    ppo_update,
    value_only_update,
)


def _toy_batch(model: OrbitPolicy, B: int, P: int = MAX_PLANETS, F: int = MAX_FLEETS) -> dict[str, torch.Tensor]:
    """Forward the model once on synthetic obs, sample, and pack a PPO batch.

    We use the *real* sampler to get a `fraction` whose log_prob exactly
    matches what `ppo_update` will recompute — i.e. ratio ≈ 1 at step 0,
    which is the only invariant we want to assert at this scale.
    """
    torch.manual_seed(0)
    planet_feats = torch.randn(B, P, 19) * 0.3
    planet_mask = torch.zeros(B, P, dtype=torch.bool)
    planet_mask[:, :8] = True
    planet_owned = torch.zeros(B, P, dtype=torch.bool)
    planet_owned[:, :3] = True
    planet_ids = torch.full((B, P), -1, dtype=torch.long)
    planet_ids[:, :8] = torch.arange(8)
    planet_garrison = torch.zeros(B, P)
    planet_garrison[:, :8] = 50
    fleet_feats = torch.zeros(B, F, 20)
    fleet_mask = torch.zeros(B, F, dtype=torch.bool)
    feats = EncodedObs(
        planet_feats=planet_feats, planet_mask=planet_mask,
        planet_owned_mask=planet_owned, planet_ids=planet_ids,
        planet_garrison=planet_garrison,
        fleet_feats=fleet_feats, fleet_mask=fleet_mask,
    )
    with torch.no_grad():
        out = model(feats)
    # Build a SampleRecord per env via the real batched sampler so the
    # `fraction` sample and `log_prob` are mutually consistent with the
    # model's current heads.
    from owars.game.observation import Observation
    from owars.game.types import Planet
    obs = [
        Observation(
            player=0, step=0,
            planets=[Planet(i, 0, 0.0, 0.0, 1.0, 10, 1) for i in range(8)],
            fleets=[], angular_velocity=0.04, initial_planets=[],
            comet_planet_ids=set(), comets=[], remaining_overage_time=60.0,
        ) for _ in range(B)
    ]
    _, records = sample_batch_with_records(out, obs, deterministic=False)
    target_idx = torch.stack([r.target_idx for r in records])
    fraction = torch.stack([r.fraction for r in records])
    log_prob = torch.stack([r.log_prob for r in records])
    old_target_logits = torch.stack([r.target_logits for r in records])
    old_fraction_alpha = torch.stack([r.fraction_alpha for r in records])
    old_fraction_beta = torch.stack([r.fraction_beta for r in records])

    return {
        "planet_feats": planet_feats,
        "planet_mask": planet_mask,
        "planet_owned_mask": planet_owned,
        "planet_ids": planet_ids,
        "planet_garrison": planet_garrison,
        "fleet_feats": fleet_feats,
        "fleet_mask": fleet_mask,
        "target_idx": target_idx,
        "fraction": fraction,
        "old_log_prob": log_prob,
        "owned_mask": planet_owned,
        "advantage": torch.randn(B),
        # Returns stay inside the default value-head support [-2, 2].
        "return": torch.randn(B).clamp(-1.0, 1.0),
        "old_target_logits": old_target_logits,
        "old_fraction_alpha": old_fraction_alpha,
        "old_fraction_beta": old_fraction_beta,
    }


def test_ppo_update_runs_and_returns_finite_metrics():
    cfg = OrbitPolicyConfig(dim=32, ff_dim=64, depth=2, n_heads=2)
    model = OrbitPolicy(cfg)
    batch = _toy_batch(model, B=8)
    optim = torch.optim.AdamW(model.parameters(), lr=3e-4)

    log = ppo_update(
        model, optim, batch,
        value_coef=0.5, entropy_coef=0.01, pmpo_kl_coef=0.3,
        pmpo_pos_to_neg_weight=0.5, pmpo_reverse_kl=True,
        epochs=2, minibatch_size=4, grad_clip=0.5,
    )

    for name in (
        "policy_loss",
        "value_loss",
        "entropy",
        "approx_kl",
        "pmpo_kl",
        "pos_frac",
        "target_entropy",
        "fraction_entropy",
        "move_prob",
        "target_confidence",
        "fraction_alpha_mean",
        "fraction_alpha_max",
        "fraction_beta_mean",
        "fraction_beta_max",
        "pmpo_target_kl",
        "pmpo_fraction_kl",
    ):
        v = getattr(log, name)
        assert math.isfinite(v), f"{name}={v!r}"
    # Reported number is the running mean across all (epoch, minibatch)
    # updates, so we just sanity-check non-negativity here.
    assert log.pmpo_kl >= 0.0, log.pmpo_kl
    assert 0.0 <= log.pos_frac <= 1.0, log.pos_frac
    assert log.value_loss >= 0.0, log.value_loss


def test_log_prob_recompute_matches_sample_time():
    """Re-evaluating log_prob at the recorded `fraction` with the same
    parameters (no gradient step yet) must reproduce the recorded `log_prob`
    — this is the invariant that makes PPO's importance ratio ≈ 1 on epoch 0.
    """
    cfg = OrbitPolicyConfig(dim=32, ff_dim=64, depth=2, n_heads=2)
    model = OrbitPolicy(cfg)
    batch = _toy_batch(model, B=4)

    # Re-run forward (no-grad) and recompute log_prob exactly the way
    # ppo_update does — without taking any optimizer step.
    from owars.policies.features import EncodedObs
    from torch.distributions import Beta
    feats = EncodedObs(
        planet_feats=batch["planet_feats"], planet_mask=batch["planet_mask"],
        planet_owned_mask=batch["planet_owned_mask"], planet_ids=batch["planet_ids"],
        planet_garrison=batch["planet_garrison"],
        fleet_feats=batch["fleet_feats"], fleet_mask=batch["fleet_mask"],
    )
    with torch.no_grad():
        out = model(feats)
    target = batch["target_idx"]
    p = out.target_logits.shape[1]
    target_log_probs = torch.log_softmax(out.target_logits, dim=-1)
    target_lp = target_log_probs.gather(-1, target.unsqueeze(-1)).squeeze(-1)
    frac_lp = Beta(out.fraction_alpha, out.fraction_beta).log_prob(batch["fraction"])
    move_mask = (target != p).float()
    chosen = target_lp + move_mask * frac_lp

    owned = batch["owned_mask"].float()
    diff = (chosen - batch["old_log_prob"]) * owned
    # Should be exactly zero up to numerical noise — same params, same fraction.
    assert diff.abs().max().item() < 1e-4, diff.abs().max().item()


def test_conditional_fraction_kl_vanishes_when_noop_is_certain():
    source_target_probs = torch.tensor(
        [
            [0.0, 0.0, 1.0],
            [0.25, 0.25, 0.50],
        ],
        dtype=torch.float32,
    )
    frac_kl = torch.tensor([7.0, 7.0])

    weighted = _conditional_fraction_kl(frac_kl, source_target_probs, noop_idx=2)

    assert torch.allclose(weighted, torch.tensor([0.0, 3.5]))


def test_conditional_entropy_weights_fraction_by_current_move_probability():
    # Uniform categorical over target0, target1, no-op -> H=log(3), P(move)=2/3.
    target_log_probs = torch.log_softmax(torch.zeros(1, 3), dim=-1)
    beta_entropy = torch.tensor([1.5])

    entropy = _conditional_action_entropy(target_log_probs, beta_entropy, noop_idx=2)

    assert torch.allclose(entropy, torch.tensor([math.log(3.0) + 1.0]), atol=1e-6)


def test_fixed_minibatches_zero_weight_padding_rows():
    torch.manual_seed(0)
    batches = _fixed_minibatches(3, 5, torch.device("cpu"))

    assert len(batches) == 1
    mb, weight = batches[0]
    assert mb.shape == (5,)
    assert weight.shape == (5,)
    assert weight.sum().item() == 3.0

    counts = torch.zeros(3)
    counts.scatter_add_(0, mb, weight)
    assert torch.equal(counts, torch.ones(3))


def test_fixed_minibatches_tail_padding_does_not_reweight_head_rows():
    torch.manual_seed(0)
    batches = _fixed_minibatches(7, 5, torch.device("cpu"))

    assert [mb.shape[0] for mb, _ in batches] == [5, 5]

    counts = torch.zeros(7)
    for mb, weight in batches:
        counts.scatter_add_(0, mb, weight)
    assert torch.equal(counts, torch.ones(7))


class _FixedPolicy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dummy = torch.nn.Parameter(torch.zeros(()))
        self.value_encoder = HLGaussLoss(min_value=-1.0, max_value=1.0, num_bins=5)
        self.new_target_logits = torch.log(torch.tensor([[[0.20, 0.80]]]))
        self.new_alpha = torch.tensor([[4.0]])
        self.new_beta = torch.tensor([[3.0]])

    def forward(self, feats):
        b = feats.planet_feats.shape[0]
        return SimpleNamespace(
            target_logits=(
                self.new_target_logits.expand(b, -1, -1).to(feats.planet_feats.device)
                + self.dummy * 0.0
            ),
            fraction_alpha=(
                self.new_alpha.expand(b, -1).to(feats.planet_feats.device)
                + self.dummy * 0.0
            ),
            fraction_beta=(
                self.new_beta.expand(b, -1).to(feats.planet_feats.device)
                + self.dummy * 0.0
            ),
            value_logits=(
                torch.zeros(b, 5, device=feats.planet_feats.device)
                + self.dummy * 0.0
            ),
        )


def test_forward_pmpo_kl_weights_fraction_by_new_move_probability():
    model = _FixedPolicy()
    old_target_probs = torch.tensor([[[0.60, 0.40]]])
    old_alpha = torch.tensor([[2.0]])
    old_beta = torch.tensor([[5.0]])
    batch = {
        "planet_feats": torch.zeros(1, 1, 19),
        "planet_mask": torch.ones(1, 1, dtype=torch.bool),
        "planet_owned_mask": torch.ones(1, 1, dtype=torch.bool),
        "planet_ids": torch.zeros(1, 1, dtype=torch.long),
        "planet_garrison": torch.ones(1, 1),
        "fleet_feats": torch.zeros(1, 1, 20),
        "fleet_mask": torch.zeros(1, 1, dtype=torch.bool),
        "target_idx": torch.ones(1, 1, dtype=torch.long),
        "fraction": torch.full((1, 1), 0.5),
        "old_log_prob": old_target_probs.log()
        .gather(-1, torch.ones(1, 1, 1, dtype=torch.long))
        .squeeze(-1),
        "owned_mask": torch.ones(1, 1, dtype=torch.bool),
        "advantage": torch.zeros(1),
        "return": torch.zeros(1),
        "old_target_logits": old_target_probs.log(),
        "old_fraction_alpha": old_alpha,
        "old_fraction_beta": old_beta,
    }
    optim = torch.optim.AdamW(model.parameters(), lr=0.0)

    log = ppo_update(
        model, optim, batch,
        value_coef=0.0, entropy_coef=0.0, pmpo_kl_coef=1.0,
        pmpo_pos_to_neg_weight=0.5, pmpo_reverse_kl=False,
        epochs=1, minibatch_size=1, grad_clip=1.0,
    )

    new_target_probs = model.new_target_logits.exp()
    expected_target = kl_divergence(
        Categorical(probs=new_target_probs[0, 0]),
        Categorical(probs=old_target_probs[0, 0]),
    )
    expected_fraction = (1.0 - new_target_probs[0, 0, 1]) * kl_divergence(
        Beta(model.new_alpha[0, 0], model.new_beta[0, 0]),
        Beta(old_alpha[0, 0], old_beta[0, 0]),
    )
    assert math.isclose(
        log.pmpo_kl,
        float(expected_target + expected_fraction),
        rel_tol=1e-6,
    )


class _ValueOnlyPolicy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dummy = torch.nn.Parameter(torch.zeros(()))
        self.value_encoder = HLGaussLoss(min_value=-1.0, max_value=1.0, num_bins=5)

    def forward(self, feats):
        x = feats.planet_feats[:, 0, 0]
        value_logits = torch.stack((x, -x, x * 0.0, x * 0.5, -x * 0.5), dim=-1)
        return SimpleNamespace(value_logits=value_logits + self.dummy * 0.0)


def test_value_only_update_ignores_zero_weight_padding_rows():
    model = _ValueOnlyPolicy()
    batch = {
        "planet_feats": torch.zeros(3, 1, 19),
        "planet_mask": torch.ones(3, 1, dtype=torch.bool),
        "planet_owned_mask": torch.ones(3, 1, dtype=torch.bool),
        "planet_ids": torch.zeros(3, 1, dtype=torch.long),
        "planet_garrison": torch.ones(3, 1),
        "fleet_feats": torch.zeros(3, 1, 20),
        "fleet_mask": torch.zeros(3, 1, dtype=torch.bool),
        "return": torch.tensor([-1.0, 0.0, 1.0]),
    }
    batch["planet_feats"][:, 0, 0] = torch.tensor([0.0, 1.0, 2.0])

    with torch.no_grad():
        logits = model(EncodedObs(
            planet_feats=batch["planet_feats"],
            planet_mask=batch["planet_mask"],
            planet_owned_mask=batch["planet_owned_mask"],
            planet_ids=batch["planet_ids"],
            planet_garrison=batch["planet_garrison"],
            fleet_feats=batch["fleet_feats"],
            fleet_mask=batch["fleet_mask"],
        )).value_logits
        target_probs = model.value_encoder.target_probs(batch["return"])
        expected = float((-(target_probs * F.log_softmax(logits, dim=-1)).sum(dim=-1)).mean())

    optim = torch.optim.AdamW(model.parameters(), lr=0.0)
    got = value_only_update(model, optim, batch, epochs=1, minibatch_size=5, grad_clip=1.0)

    assert math.isclose(got, expected, rel_tol=1e-6)
