"""End-to-end smoke tests for `ppo_update`.

Builds a tiny synthetic batch that mirrors the keys `_stack_trajectories`
produces, runs one PPO update, and asserts the returned metrics are finite.
Also keeps a regression guard on the log_prob recompute invariant
(ratio ≈ 1 on epoch 0).
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import torch
import torch.nn.functional as nn_functional

from owars.policies.config import OrbitPolicyConfig
from owars.policies.features import MAX_FLEETS, MAX_PLANETS, EncodedObs
from owars.policies.model import HLGaussLoss, OrbitPolicy
from owars.policies.sampling import sample_batch_with_records
from owars.training.ppo import (
    _conditional_action_entropy,
    _fixed_minibatches,
    _backward_actor_critic_with_separate_clips,
    _minibatch_loss_scale,
    _squashed_normal_log_prob,
    ppo_update,
    value_only_update,
)


def _toy_batch(
    model: OrbitPolicy,
    batch_size: int,
    num_planets: int = MAX_PLANETS,
    num_fleets: int = MAX_FLEETS,
) -> dict[str, torch.Tensor]:
    """Forward the model once on synthetic obs, sample, and pack a PPO batch.

    We use the *real* sampler to get a `fraction` whose log_prob exactly
    matches what `ppo_update` will recompute — i.e. ratio ≈ 1 at step 0,
    which is the only invariant we want to assert at this scale.
    """
    torch.manual_seed(0)
    planet_feats = torch.randn(batch_size, num_planets, 19) * 0.3
    planet_mask = torch.zeros(batch_size, num_planets, dtype=torch.bool)
    planet_mask[:, :8] = True
    planet_owned = torch.zeros(batch_size, num_planets, dtype=torch.bool)
    planet_owned[:, :3] = True
    planet_ids = torch.full((batch_size, num_planets), -1, dtype=torch.long)
    planet_ids[:, :8] = torch.arange(8)
    planet_garrison = torch.zeros(batch_size, num_planets)
    planet_garrison[:, :8] = 50
    fleet_feats = torch.zeros(batch_size, num_fleets, 20)
    fleet_mask = torch.zeros(batch_size, num_fleets, dtype=torch.bool)
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
        ) for _ in range(batch_size)
    ]
    _, records = sample_batch_with_records(out, obs, deterministic=False)
    launch = torch.stack([r.launch for r in records])
    target_idx = torch.stack([r.target_idx for r in records])
    fraction = torch.stack([r.fraction for r in records])
    log_prob = torch.stack([r.log_prob for r in records])
    target_legal_mask = torch.stack([r.target_legal_mask for r in records])

    return {
        "planet_feats": planet_feats,
        "planet_mask": planet_mask,
        "planet_owned_mask": planet_owned,
        "planet_ids": planet_ids,
        "planet_garrison": planet_garrison,
        "fleet_feats": fleet_feats,
        "fleet_mask": fleet_mask,
        "launch": launch,
        "target_idx": target_idx,
        "fraction": fraction,
        "old_log_prob": log_prob,
        "owned_mask": planet_owned,
        "advantage": torch.randn(batch_size),
        # Returns stay inside the default value-head support [-2, 2].
        "return": torch.randn(batch_size).clamp(-1.0, 1.0),
        "target_legal_mask": target_legal_mask,
    }


def test_ppo_update_runs_and_returns_finite_metrics():
    cfg = OrbitPolicyConfig(dim=32, ff_dim=64, depth=2, n_heads=2)
    model = OrbitPolicy(cfg)
    batch = _toy_batch(model, batch_size=8)
    optim = torch.optim.AdamW(model.parameters(), lr=3e-4)

    log = ppo_update(
        model, optim, batch,
        value_coef=0.5,
        target_entropy_coef=0.01,
        fraction_entropy_coef=0.0,
        norm_advantage=True,
        spo_eps_low=0.2,
        spo_eps_high=0.28,
        epochs=2, minibatch_size=4, grad_clip=0.5,
    )

    for name in (
        "policy_loss",
        "value_loss",
        "entropy",
        "approx_kl",
        "spo_penalty",
        "pos_frac",
        "target_entropy",
        "fraction_entropy",
        "move_prob",
        "target_confidence",
        "fraction_mean_mean",
        "fraction_mean_abs_max",
        "fraction_log_std_mean",
        "fraction_log_std_min",
        "fraction_log_std_max",
        "deterministic_fraction_mean",
    ):
        v = getattr(log, name)
        assert math.isfinite(v), f"{name}={v!r}"
    # Reported number is the running mean across all (epoch, minibatch)
    # updates, so we just sanity-check non-negativity here.
    assert log.spo_penalty >= 0.0, log.spo_penalty
    assert 0.0 <= log.pos_frac <= 1.0, log.pos_frac
    assert log.value_loss >= 0.0, log.value_loss


def test_log_prob_recompute_matches_sample_time():
    """Re-evaluating log_prob at the recorded `fraction` with the same
    parameters (no gradient step yet) must reproduce the recorded `log_prob`
    — this is the invariant that makes PPO's importance ratio ≈ 1 on epoch 0.
    """
    cfg = OrbitPolicyConfig(dim=32, ff_dim=64, depth=2, n_heads=2)
    model = OrbitPolicy(cfg)
    batch = _toy_batch(model, batch_size=4)

    # Re-run forward (no-grad) and recompute log_prob exactly the way
    # ppo_update does — without taking any optimizer step.
    feats = EncodedObs(
        planet_feats=batch["planet_feats"], planet_mask=batch["planet_mask"],
        planet_owned_mask=batch["planet_owned_mask"], planet_ids=batch["planet_ids"],
        planet_garrison=batch["planet_garrison"],
        fleet_feats=batch["fleet_feats"], fleet_mask=batch["fleet_mask"],
    )
    with torch.no_grad():
        out = model(feats)
    launch = batch["launch"]
    target = batch["target_idx"]
    target_legal_mask = batch["target_legal_mask"]
    target_logits = out.target_logits.masked_fill(~target_legal_mask, float("-inf"))
    finite = torch.isfinite(target_logits).any(dim=-1, keepdim=True)
    target_logits = torch.where(finite, target_logits, torch.zeros_like(target_logits))
    launch_logits = out.launch_logits.masked_fill(~target_legal_mask.any(dim=-1), -20.0)
    target_log_probs = torch.log_softmax(target_logits, dim=-1)
    target_lp = target_log_probs.gather(-1, target.unsqueeze(-1)).squeeze(-1)
    frac_lp = _squashed_normal_log_prob(
        out.fraction_mean, out.fraction_log_std, batch["fraction"]
    )
    launch_lp = -nn_functional.binary_cross_entropy_with_logits(
        launch_logits, launch, reduction="none"
    )
    chosen = launch_lp + launch * (target_lp + frac_lp)

    owned = batch["owned_mask"].float()
    diff = (chosen - batch["old_log_prob"]) * owned
    # Should be exactly zero up to numerical noise — same params, same fraction.
    assert diff.abs().max().item() < 1e-4, diff.abs().max().item()


def test_conditional_entropy_weights_fraction_by_current_move_probability():
    # P(launch)=0.25, uniform categorical over two targets -> H=log(2).
    launch_logits = torch.logit(torch.tensor([0.25]))
    target_log_probs = torch.log_softmax(torch.zeros(1, 2), dim=-1)
    fraction_entropy = torch.tensor([1.5])

    entropy = _conditional_action_entropy(
        launch_logits, target_log_probs, fraction_entropy
    )

    expected_launch_entropy = -(0.25 * math.log(0.25) + 0.75 * math.log(0.75))
    expected = expected_launch_entropy + 0.25 * (math.log(2.0) + 1.5)
    assert torch.allclose(entropy, torch.tensor([expected]), atol=1e-6)


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


def test_minibatch_loss_scale_downweights_padded_tail_step():
    assert torch.allclose(_minibatch_loss_scale(torch.ones(5)), torch.tensor(1.0))
    weight = torch.tensor([1.0, 1.0, 0.0, 0.0, 0.0])
    assert torch.allclose(_minibatch_loss_scale(weight), torch.tensor(0.4))


def test_policy_value_grad_clip_separates_shared_actor_and_critic_grads():
    class Toy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.launch_head = torch.nn.Linear(1, 1, bias=False)
            self.value_head = torch.nn.Linear(1, 1, bias=False)
            self.shared = torch.nn.Linear(1, 1, bias=False)

    model = Toy()
    actor_loss = 3.0 * model.launch_head.weight.sum() + 4.0 * model.shared.weight.sum()
    critic_loss = 30.0 * model.value_head.weight.sum() + 40.0 * model.shared.weight.sum()

    actor_norm, critic_norm = _backward_actor_critic_with_separate_clips(
        model,
        actor_loss,
        critic_loss,
        1.0,
    )

    assert torch.allclose(actor_norm, torch.tensor(5.0))
    assert torch.allclose(critic_norm, torch.tensor(50.0))
    assert torch.allclose(model.launch_head.weight.grad, torch.tensor([[0.6]]))
    assert torch.allclose(model.value_head.weight.grad, torch.tensor([[0.6]]))
    assert torch.allclose(model.shared.weight.grad, torch.tensor([[1.6]]))


class _FixedPolicy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dummy = torch.nn.Parameter(torch.zeros(()))
        self.value_encoder = HLGaussLoss(min_value=-1.0, max_value=1.0, num_bins=5)
        self.new_launch_logits = torch.logit(torch.tensor([[0.25, 0.25]]))
        self.new_target_logits = torch.log(
            torch.tensor([[[0.20, 0.80], [0.50, 0.50]]])
        )
        self.new_fraction_mean = torch.tensor([[0.0, 0.0]])
        self.new_fraction_log_std = torch.tensor([[-0.25, -0.25]])

    def forward(self, feats, *, detach_actor: bool = False):
        b = feats.planet_feats.shape[0]
        return SimpleNamespace(
            target_logits=(
                self.new_target_logits.expand(b, -1, -1).to(feats.planet_feats.device)
                + self.dummy * 0.0
            ),
            launch_logits=(
                self.new_launch_logits.expand(b, -1).to(feats.planet_feats.device)
                + self.dummy * 0.0
            ),
            fraction_mean=(
                self.new_fraction_mean.expand(b, -1).to(feats.planet_feats.device)
                + self.dummy * 0.0
            ),
            fraction_log_std=(
                self.new_fraction_log_std.expand(b, -1).to(feats.planet_feats.device)
                + self.dummy * 0.0
            ),
            value_logits=(
                torch.zeros(b, 5, device=feats.planet_feats.device)
                + self.dummy * 0.0
            ),
        )


def _fixed_policy_batch(
    *,
    launch: float,
    advantage: float,
    old_log_prob: torch.Tensor,
    owned_mask: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    launch_t = torch.tensor([[launch, launch]])
    if owned_mask is None:
        owned_mask = torch.tensor([[True, False]])
    if old_log_prob.shape[1] == 1:
        old_log_prob = torch.cat((old_log_prob, torch.zeros(1, 1)), dim=1)
    return {
        "planet_feats": torch.zeros(1, 2, 19),
        "planet_mask": torch.ones(1, 2, dtype=torch.bool),
        "planet_owned_mask": owned_mask,
        "planet_ids": torch.arange(2).reshape(1, 2),
        "planet_garrison": torch.ones(1, 2),
        "fleet_feats": torch.zeros(1, 1, 20),
        "fleet_mask": torch.zeros(1, 1, dtype=torch.bool),
        "launch": launch_t,
        "target_idx": torch.ones(1, 2, dtype=torch.long),
        "fraction": torch.full((1, 2), 0.5),
        "old_log_prob": old_log_prob,
        "owned_mask": owned_mask,
        "advantage": torch.tensor([advantage]),
        "return": torch.zeros(1),
        "target_legal_mask": torch.ones(1, 2, 2, dtype=torch.bool),
    }


def test_spo_asym_policy_loss_uses_high_eps_when_drift_agrees_with_advantage():
    model = _FixedPolicy()
    launch_lp = torch.log(model.new_launch_logits.sigmoid()[0, 0])
    target_lp = torch.log(model.new_target_logits.exp()[0, 0, 1])
    frac_lp = _squashed_normal_log_prob(
        model.new_fraction_mean[0, 0],
        model.new_fraction_log_std[0, 0],
        torch.tensor(0.5),
    )
    new_log_prob = (launch_lp + target_lp + frac_lp).reshape(1, 1)
    old_log_prob = new_log_prob - math.log(1.5)
    batch = _fixed_policy_batch(
        launch=1.0,
        advantage=1.0,
        old_log_prob=old_log_prob,
    )
    optim = torch.optim.AdamW(model.parameters(), lr=0.0)

    log = ppo_update(
        model, optim, batch,
        value_coef=0.0,
        target_entropy_coef=0.0,
        fraction_entropy_coef=0.0,
        norm_advantage=False,
        spo_eps_low=0.2,
        spo_eps_high=0.28,
        epochs=1, minibatch_size=1, grad_clip=1.0,
    )

    expected_penalty = 0.5**2 / (2.0 * 0.28)
    expected_loss = -(1.5 - expected_penalty)
    assert math.isclose(log.policy_loss, expected_loss, rel_tol=1e-6)
    assert math.isclose(log.spo_penalty, expected_penalty, rel_tol=1e-6)
    expected_kl = (1.5 - 1.0) - math.log(1.5)
    assert math.isclose(log.approx_kl, expected_kl, rel_tol=1e-6)


def test_approx_kl_sums_owned_planet_log_probs_cleanrl_style():
    model = _FixedPolicy()
    launch_lp = torch.log(model.new_launch_logits.sigmoid())
    target_lp = torch.log(torch.tensor([[0.80, 0.50]]))
    frac_lp = _squashed_normal_log_prob(
        model.new_fraction_mean,
        model.new_fraction_log_std,
        torch.full((1, 2), 0.5),
    )
    new_log_prob = launch_lp + target_lp + frac_lp
    old_log_prob = new_log_prob - torch.log(torch.tensor([[1.5, 1.2]]))
    batch = _fixed_policy_batch(
        launch=1.0,
        advantage=1.0,
        old_log_prob=old_log_prob,
        owned_mask=torch.tensor([[True, True]]),
    )
    optim = torch.optim.AdamW(model.parameters(), lr=0.0)

    log = ppo_update(
        model, optim, batch,
        value_coef=0.0,
        target_entropy_coef=0.0,
        fraction_entropy_coef=0.0,
        norm_advantage=False,
        spo_eps_low=0.2,
        spo_eps_high=0.28,
        epochs=1, minibatch_size=1, grad_clip=1.0,
    )

    joint_ratio = 1.5 * 1.2
    expected_kl = (joint_ratio - 1.0) - math.log(joint_ratio)
    assert math.isclose(log.approx_kl, expected_kl, rel_tol=1e-6)


def test_approx_kl_reports_latest_minibatch_not_epoch_mean():
    model = _FixedPolicy()
    launch_lp = torch.log(model.new_launch_logits.sigmoid()[0, 0])
    target_lp = torch.log(model.new_target_logits.exp()[0, 0, 1])
    frac_lp = _squashed_normal_log_prob(
        model.new_fraction_mean[0, 0],
        model.new_fraction_log_std[0, 0],
        torch.tensor(0.5),
    )
    new_log_prob = launch_lp + target_lp + frac_lp
    ratios = torch.tensor([1.2, 1.8])
    batch = {
        "planet_feats": torch.zeros(2, 2, 19),
        "planet_mask": torch.ones(2, 2, dtype=torch.bool),
        "planet_owned_mask": torch.tensor([[True, False], [True, False]]),
        "planet_ids": torch.arange(2).expand(2, -1),
        "planet_garrison": torch.ones(2, 2),
        "fleet_feats": torch.zeros(2, 1, 20),
        "fleet_mask": torch.zeros(2, 1, dtype=torch.bool),
        "launch": torch.tensor([[1.0, 0.0], [1.0, 0.0]]),
        "target_idx": torch.ones(2, 2, dtype=torch.long),
        "fraction": torch.full((2, 2), 0.5),
        "old_log_prob": torch.stack(
            (
                torch.tensor([new_log_prob - torch.log(ratios[0]), 0.0]),
                torch.tensor([new_log_prob - torch.log(ratios[1]), 0.0]),
            )
        ),
        "owned_mask": torch.tensor([[True, False], [True, False]]),
        "advantage": torch.ones(2),
        "return": torch.zeros(2),
        "target_legal_mask": torch.ones(2, 2, 2, dtype=torch.bool),
    }
    torch.manual_seed(7)
    expected_order = torch.randperm(2)
    expected_ratio = float(ratios[int(expected_order[-1])])
    torch.manual_seed(7)
    optim = torch.optim.AdamW(model.parameters(), lr=0.0)

    log = ppo_update(
        model, optim, batch,
        value_coef=0.0,
        target_entropy_coef=0.0,
        fraction_entropy_coef=0.0,
        norm_advantage=False,
        spo_eps_low=0.2,
        spo_eps_high=0.28,
        epochs=1, minibatch_size=1, grad_clip=1.0,
    )

    latest_kl = (expected_ratio - 1.0) - math.log(expected_ratio)
    mean_kl = torch.mean((ratios - 1.0) - torch.log(ratios)).item()
    assert math.isclose(log.approx_kl, latest_kl, rel_tol=1e-6, abs_tol=1e-7)
    assert not math.isclose(log.approx_kl, mean_kl, rel_tol=1e-6)


def test_spo_asym_policy_loss_uses_low_eps_when_drift_opposes_advantage():
    model = _FixedPolicy()
    launch_lp = torch.log(model.new_launch_logits.sigmoid()[0, 0])
    target_lp = torch.log(model.new_target_logits.exp()[0, 0, 1])
    frac_lp = _squashed_normal_log_prob(
        model.new_fraction_mean[0, 0],
        model.new_fraction_log_std[0, 0],
        torch.tensor(0.5),
    )
    new_log_prob = (launch_lp + target_lp + frac_lp).reshape(1, 1)
    old_log_prob = new_log_prob + math.log(2.0)
    batch = _fixed_policy_batch(
        launch=1.0,
        advantage=1.0,
        old_log_prob=old_log_prob,
    )
    optim = torch.optim.AdamW(model.parameters(), lr=0.0)

    log = ppo_update(
        model, optim, batch,
        value_coef=0.0,
        target_entropy_coef=0.0,
        fraction_entropy_coef=0.0,
        norm_advantage=False,
        spo_eps_low=0.2,
        spo_eps_high=0.28,
        epochs=1, minibatch_size=1, grad_clip=1.0,
    )

    expected_penalty = 0.5**2 / (2.0 * 0.2)
    expected_loss = -(0.5 - expected_penalty)
    assert math.isclose(log.policy_loss, expected_loss, rel_tol=1e-6)
    assert math.isclose(log.spo_penalty, expected_penalty, rel_tol=1e-6)


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
        expected = float(
            (-(target_probs * nn_functional.log_softmax(logits, dim=-1)).sum(dim=-1)).mean()
        )

    optim = torch.optim.AdamW(model.parameters(), lr=0.0)
    got = value_only_update(model, optim, batch, epochs=1, minibatch_size=5, grad_clip=1.0)

    assert math.isclose(got, expected, rel_tol=1e-6)
