"""End-to-end smoke tests for `ppo_update`.

Builds a tiny synthetic batch that mirrors the keys `_stack_trajectories`
produces, runs one PPO update, and asserts the returned metrics are finite.
Also keeps a regression guard on the log_prob recompute invariant
(ratio ≈ 1 on epoch 0).
"""

from __future__ import annotations

import math
import numpy as np
from types import SimpleNamespace

import torch
import torch.nn.functional as nn_functional

from owars.game.observation import Observation
from owars.game.types import Planet
from owars.policies.config import OrbitPolicyConfig
from owars.policies.features import EncodedObs, encode_observations
from owars.policies.model import HLGaussLoss, OrbitPolicy
from owars.policies.sampling import (
    _categorical_action_log_probs,
    _threshold_normal_launch_log_prob,
    _threshold_normal_launch_prob,
    sample_batch_with_records,
)
from owars.training.ppo import (
    _beta_log_prob,
    _conditional_action_entropy,
    _fixed_minibatches,
    _fixed_minibatches_by_count,
    _backward_actor_critic_with_group_clips,
    _distributional_value_loss,
    _minibatch_loss_scale,
    _rank_gaussian_advantage,
    compute_gae,
    ppo_update,
    value_only_update,
)
from owars.training import train as train_mod


# A non-degenerate 8-planet position. Player 0 owns the first three; the rest
# are enemy/neutral so owned planets have legal targets. Every planet sits in
# the left half of the board (x <= 30), and the sun spans x in [40, 60], so
# every source->target straight line clears the sun and launches actually fire
# — without that, the launch/target/fraction log-probs collapse to zero and the
# recompute invariant below is vacuously satisfied.
_TOY_PLANETS = [
    # (id, owner, x, y, radius, ships, production)
    Planet(0, 0, 8.0, 10.0, 2.1, 80, 3),   # owned
    Planet(1, 0, 8.0, 40.0, 2.1, 80, 3),   # owned
    Planet(2, 0, 8.0, 70.0, 2.1, 80, 3),   # owned
    Planet(3, 1, 8.0, 95.0, 1.7, 30, 2),   # enemy
    Planet(4, 1, 30.0, 10.0, 1.7, 30, 2),  # enemy
    Planet(5, -1, 30.0, 40.0, 1.0, 20, 1),  # neutral
    Planet(6, -1, 30.0, 70.0, 1.0, 20, 1),  # neutral
    Planet(7, 1, 30.0, 95.0, 1.7, 30, 2),  # enemy
]


def _toy_batch(
    model: OrbitPolicy,
    batch_size: int,
) -> dict[str, torch.Tensor]:
    """Forward the model once on synthetic obs, sample, and pack a PPO batch.

    The obs is encoded with the *real* feature encoder so the model features and
    the sampler's legality view are mutually consistent, and the *real* sampler
    produces a `launch`/`target_idx`/`fraction`/`log_prob` tuple that
    `ppo_update` must reproduce exactly on epoch 0 (importance ratio ≈ 1).
    """
    torch.manual_seed(0)
    obs = [
        Observation(
            player=0, step=0,
            planets=list(_TOY_PLANETS),
            fleets=[], angular_velocity=0.0, initial_planets=[],
            comet_planet_ids=set(), comets=[], remaining_overage_time=60.0,
        ) for _ in range(batch_size)
    ]
    feats = encode_observations(obs)
    with torch.no_grad():
        out = model(feats)
    _, records = sample_batch_with_records(
        out,
        obs,
        deterministic=False,
    )
    launch = torch.stack([r.launch for r in records])
    target_idx = torch.stack([r.target_idx for r in records])
    fraction = torch.stack([r.fraction for r in records])
    log_prob = torch.stack([r.log_prob for r in records])
    target_legal_mask = torch.stack([r.target_legal_mask for r in records])

    return {
        "planet_feats": feats.planet_feats,
        "planet_mask": feats.planet_mask,
        "planet_owned_mask": feats.planet_owned_mask,
        "planet_ids": feats.planet_ids,
        "planet_garrison": feats.planet_garrison,
        "fleet_feats": feats.fleet_feats,
        "fleet_mask": feats.fleet_mask,
        "launch": launch,
        "target_idx": target_idx,
        "fraction": fraction,
        "old_log_prob": log_prob,
        "owned_mask": feats.planet_owned_mask,
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
        advantage_transform="rankgauss",
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
        "fraction_alpha_mean",
        "fraction_beta_mean",
        "fraction_concentration_mean",
        "fraction_concentration_max",
        "fraction_skew_abs_mean",
        "deterministic_fraction_mean",
    ):
        v = getattr(log, name)
        assert math.isfinite(v), f"{name}={v!r}"
    # Reported number is the running mean across all (epoch, minibatch)
    # updates, so we just sanity-check non-negativity here.
    assert log.spo_penalty >= 0.0, log.spo_penalty
    assert 0.0 <= log.pos_frac <= 1.0, log.pos_frac
    assert log.value_loss >= 0.0, log.value_loss


def test_pretrain_value_batch_uses_configured_lambda_return(monkeypatch):
    monkeypatch.setattr(train_mod, "_stack_encoded", lambda _trajs: {})
    rewards = np.asarray([1.0, -0.5, 2.0], dtype=np.float32)
    values = np.asarray([0.25, -0.1, 0.4], dtype=np.float32)
    traj = SimpleNamespace(
        reward=rewards.tolist(),
        value=[torch.tensor(v) for v in values],
    )

    batch = train_mod._pretrain_value_batch(
        [traj],
        gamma=0.9,
        gae_lambda=0.5,
    )

    _adv, expected = compute_gae(rewards, values, gamma=0.9, lam=0.5)
    assert torch.allclose(batch["return"], torch.from_numpy(expected))


def test_stack_trajectories_can_decouple_policy_and_value_lambdas(monkeypatch):
    monkeypatch.setattr(train_mod, "_stack_encoded", lambda _trajs: {})
    rewards = np.asarray([0.0, 0.0, 1.0], dtype=np.float32)
    values = np.asarray([0.2, 0.1, -0.3], dtype=np.float32)
    traj = SimpleNamespace(
        reward=rewards.tolist(),
        value=[torch.tensor(v) for v in values],
        launch=[torch.zeros(1) for _ in rewards],
        target_idx=[torch.zeros(1, dtype=torch.long) for _ in rewards],
        fraction=[torch.full((1,), 0.5) for _ in rewards],
        log_prob=[torch.zeros(1) for _ in rewards],
        owned_mask=[torch.ones(1, dtype=torch.bool) for _ in rewards],
        target_legal_mask=[torch.ones(1, 1, dtype=torch.bool) for _ in rewards],
    )

    batch = train_mod._stack_trajectories(
        [traj],
        gamma=1.0,
        gae_lambda=0.5,
        value_gae_lambda=1.0,
    )

    expected_adv, _policy_return = compute_gae(rewards, values, gamma=1.0, lam=0.5)
    _value_adv, expected_return = compute_gae(rewards, values, gamma=1.0, lam=1.0)
    assert torch.allclose(batch["advantage"], torch.from_numpy(expected_adv))
    assert torch.allclose(batch["return"], torch.from_numpy(expected_return))


def test_stack_trajectories_builds_masked_critic_mtp_targets(monkeypatch):
    monkeypatch.setattr(train_mod, "_stack_encoded", lambda _trajs: {})
    rewards = np.asarray([1.0, 2.0, 3.0], dtype=np.float32)
    values = np.zeros_like(rewards)
    traj = SimpleNamespace(
        reward=rewards.tolist(),
        value=[torch.tensor(v) for v in values],
        launch=[torch.zeros(1) for _ in rewards],
        target_idx=[torch.zeros(1, dtype=torch.long) for _ in rewards],
        fraction=[torch.full((1,), 0.5) for _ in rewards],
        log_prob=[torch.zeros(1) for _ in rewards],
        owned_mask=[torch.ones(1, dtype=torch.bool) for _ in rewards],
        target_legal_mask=[torch.ones(1, 1, dtype=torch.bool) for _ in rewards],
    )

    batch = train_mod._stack_trajectories(
        [traj],
        gamma=1.0,
        gae_lambda=1.0,
        critic_mtp_horizon=4,
    )

    assert torch.equal(
        batch["return_mtp_mask"],
        torch.tensor(
            [
                [True, True, True, False],
                [True, True, False, False],
                [True, False, False, False],
            ]
        ),
    )
    assert torch.allclose(
        batch["return_mtp"],
        torch.tensor(
            [
                [6.0, 5.0, 3.0, 0.0],
                [5.0, 3.0, 0.0, 0.0],
                [3.0, 0.0, 0.0, 0.0],
            ]
        ),
    )


def test_distributional_value_loss_sums_valid_mtp_horizons():
    encoder = HLGaussLoss(min_value=-1.0, max_value=1.0, num_bins=5)
    logits = torch.zeros(2, 3, 5)
    returns = torch.tensor([[0.0, 0.5, 1.0], [-0.5, 0.25, 0.75]])
    mask = torch.tensor([[True, True, False], [True, False, False]])
    row_weight = torch.ones(2)

    got = _distributional_value_loss(encoder, logits, returns, row_weight, mask)
    target_probs = encoder.target_probs(returns)
    ce = -(target_probs * nn_functional.log_softmax(logits, dim=-1)).sum(dim=-1)
    expected = torch.tensor([(ce[0, :2].sum() + ce[1, 0]) / 2.0])

    assert torch.allclose(got, expected.squeeze(0))


def test_distributional_value_loss_legacy_logits_use_horizon_zero_only():
    encoder = HLGaussLoss(min_value=-1.0, max_value=1.0, num_bins=5)
    logits = torch.zeros(2, 5)
    returns = torch.tensor([[0.0, 1.0], [0.5, -1.0]])
    mask = torch.tensor([[True, True], [False, True]])
    row_weight = torch.ones(2)

    got = _distributional_value_loss(encoder, logits, returns, row_weight, mask)
    target_probs = encoder.target_probs(returns[:, 0])
    ce = -(target_probs * nn_functional.log_softmax(logits, dim=-1)).sum(dim=-1)
    expected = ce[0]

    assert torch.allclose(got, expected)


def test_ppo_update_minibatch_count_runs_exact_count():
    cfg = OrbitPolicyConfig(dim=32, ff_dim=64, depth=2, n_heads=2)
    model = OrbitPolicy(cfg)
    batch = _toy_batch(model, batch_size=12)
    optim = torch.optim.AdamW(model.parameters(), lr=3e-4)

    log = ppo_update(
        model, optim, batch,
        value_coef=0.5,
        target_entropy_coef=0.01,
        fraction_entropy_coef=0.0,
        norm_advantage=True,
        advantage_transform="rankgauss",
        spo_eps_low=0.2,
        spo_eps_high=0.28,
        epochs=1, minibatch_size=2, minibatch_count=3, grad_clip=0.5,
    )

    assert math.isfinite(log.policy_loss)
    assert math.isfinite(log.value_loss)
    assert log.actor_grad_norm >= 0.0
    assert log.critic_grad_norm >= 0.0
    assert log.actor_shared_grad_norm >= 0.0
    assert log.critic_shared_grad_norm >= 0.0
    assert log.shared_grad_norm >= 0.0
    assert log.actor_shared_raw_grad_norm >= log.actor_shared_grad_norm
    assert log.critic_shared_raw_grad_norm >= log.critic_shared_grad_norm
    assert 0.0 <= log.actor_clip_scale <= 1.0
    assert 0.0 <= log.critic_clip_scale <= 1.0
    assert 0.0 <= log.actor_clip_frac <= 1.0
    assert 0.0 <= log.critic_clip_frac <= 1.0


def test_fixed_minibatches_by_count_cover_rows_once_with_equal_shapes():
    torch.manual_seed(3)
    batches = _fixed_minibatches_by_count(10, 4, torch.device("cpu"))

    assert len(batches) == 4
    assert {tuple(mb.shape) for mb, _ in batches} == {(3,)}
    assert {tuple(weight.shape) for _, weight in batches} == {(3,)}

    real_rows = []
    for mb, weight in batches:
        real_rows.extend(mb[weight.bool()].tolist())
    assert sorted(real_rows) == list(range(10))
    assert sum(float(weight.sum()) for _, weight in batches) == 10.0


def test_log_prob_recompute_matches_sample_time():
    """Re-evaluating log_prob at the recorded action with the same parameters
    (no gradient step yet) must reproduce the recorded `log_prob` — this is the
    invariant that makes PPO's importance ratio ≈ 1 on epoch 0.

    The recompute mirrors the *categorical* action path in `_PPOMinibatchKernel`
    — the one `OrbitPolicy` actually drives, since it always emits
    `action_logit_softcap`: a single softmax over `[noop, target_0..target_P]`,
    plus the Beta fraction term on launched rows.
    """
    # Seed before init so the weights (and hence which planets launch) are
    # deterministic regardless of test order; `_toy_batch` reseeds the sampler.
    torch.manual_seed(0)
    cfg = OrbitPolicyConfig(dim=32, ff_dim=64, depth=2, n_heads=2)
    model = OrbitPolicy(cfg)
    batch = _toy_batch(model, batch_size=4)

    owned = batch["owned_mask"]
    launch_f = batch["launch"].float().clamp(0.0, 1.0)
    # Guard against the batch silently degenerating to "nobody launches", which
    # would collapse every action onto the no-op column and never exercise the
    # target-gather, making the invariant vacuous.
    assert launch_f[owned].sum().item() > 0, "expected at least one owned launch"

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
    assert out.action_logit_softcap is not None

    p = out.target_logits.shape[1]
    target = batch["target_idx"].clamp(0, p - 1)
    target_logits = out.target_logits.masked_fill(
        ~batch["target_legal_mask"], float("-inf")
    )
    action_log_probs = _categorical_action_log_probs(
        out.launch_logits,
        target_logits,
        out.action_logit_softcap,
    )
    action_idx = torch.where(launch_f > 0.5, target + 1, torch.zeros_like(target))
    action_lp = action_log_probs.gather(-1, action_idx.unsqueeze(-1)).squeeze(-1)
    frac_lp = _beta_log_prob(out.fraction_alpha, out.fraction_beta, batch["fraction"])
    chosen = action_lp + launch_f * frac_lp

    diff = (chosen - batch["old_log_prob"]) * owned.float()
    # Should be exactly zero up to numerical noise — same params, same action.
    assert diff.abs().max().item() < 1e-4, diff.abs().max().item()


def test_categorical_flat_logits_balance_noop_against_target_group():
    noop_logits = torch.zeros(1, 2)
    target_logits = torch.full((1, 2, 5), float("-inf"))
    target_logits[0, 0, :4] = 0.0
    target_logits[0, 1, :2] = 0.0
    log_probs = _categorical_action_log_probs(
        noop_logits,
        target_logits,
        action_logit_softcap=8.0,
    )

    probs = log_probs.exp()
    assert torch.allclose(probs[0, 0, 0], torch.tensor(0.5), atol=1e-6)
    assert torch.allclose(probs[0, 1, 0], torch.tensor(0.5), atol=1e-6)
    assert torch.allclose(probs[0, 0, 1:5], torch.full((4,), 0.125), atol=1e-6)
    assert torch.allclose(probs[0, 1, 1:3], torch.full((2,), 0.25), atol=1e-6)
    assert torch.allclose(probs[0, 0, 1:].sum(), torch.tensor(0.5), atol=1e-6)
    assert torch.allclose(probs[0, 1, 1:].sum(), torch.tensor(0.5), atol=1e-6)


def test_categorical_softcap_preserves_balanced_group_semantics():
    noop_logits = torch.zeros(1, 1)
    target_logits = torch.zeros(1, 1, 3)
    log_probs = _categorical_action_log_probs(
        noop_logits,
        target_logits,
        action_logit_softcap=8.0,
    )

    noop_prob = log_probs.exp()[0, 0, 0]
    assert torch.allclose(noop_prob, torch.tensor(0.5), atol=1e-6)


def test_threshold_normal_launch_log_std_controls_exploration():
    mean = torch.tensor([-1.0, -1.0, 1.0, 1.0])
    low_std = torch.full_like(mean, -5.0)
    high_std = torch.full_like(mean, 2.0)
    launch = torch.tensor([0.0, 1.0, 1.0, 0.0])

    low_lp = _threshold_normal_launch_log_prob(mean, low_std, launch)
    high_lp = _threshold_normal_launch_log_prob(mean, high_std, launch)

    assert low_lp[0] > high_lp[0]   # confident no-launch when mean < 0
    assert low_lp[2] > high_lp[2]   # confident launch when mean > 0
    assert high_lp[1] > low_lp[1]   # high std explores against mean sign
    assert high_lp[3] > low_lp[3]


def test_threshold_normal_launch_prob_floor_bounds_extreme_probabilities():
    mean = torch.tensor([-100.0, 0.0, 100.0])
    prob = _threshold_normal_launch_prob(mean, None, prob_floor=0.05)

    assert torch.allclose(prob, torch.tensor([0.05, 0.5, 0.95]), atol=1e-6)

    launch_lp = _threshold_normal_launch_log_prob(
        mean,
        None,
        torch.tensor([1.0, 1.0, 0.0]),
        prob_floor=0.05,
    )
    assert torch.isfinite(launch_lp).all()
    assert torch.allclose(launch_lp[0], torch.log(torch.tensor(0.05)), atol=1e-6)
    assert torch.allclose(launch_lp[2], torch.log(torch.tensor(0.05)), atol=1e-6)


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


def test_rank_gaussian_advantage_maps_full_batch_ranks_to_normal_quantiles():
    adv = torch.tensor([10.0, -1.0, 3.0, 0.0])
    got = _rank_gaussian_advantage(adv)
    ranks = adv.argsort().argsort().float()
    expected = math.sqrt(2.0) * torch.erfinv(2.0 * ((ranks + 0.5) / 4.0) - 1.0)

    assert torch.allclose(got, expected)
    assert torch.equal(got.argsort(), adv.argsort())


def test_policy_value_grad_clip_clips_combined_head_and_shared_flows_then_sums_shared():
    class Toy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.target_noop_key = torch.nn.Linear(1, 1, bias=False)
            self.value_head = torch.nn.Linear(1, 1, bias=False)
            self.shared = torch.nn.Linear(1, 1, bias=False)

    model = Toy()
    actor_loss = (
        3.0 * model.target_noop_key.weight.sum()
        + 4.0 * model.shared.weight.sum()
    )
    critic_loss = 30.0 * model.value_head.weight.sum() + 40.0 * model.shared.weight.sum()

    (
        actor_norm,
        critic_norm,
        actor_shared_norm,
        critic_shared_norm,
        shared_norm,
        actor_shared_raw_norm,
        critic_shared_raw_norm,
        actor_clip_scale,
        critic_clip_scale,
        actor_clip_frac,
        critic_clip_frac,
    ) = _backward_actor_critic_with_group_clips(model, actor_loss, critic_loss, 1.0)

    assert torch.allclose(actor_norm, torch.tensor(5.0))
    assert torch.allclose(critic_norm, torch.tensor(50.0))
    assert torch.allclose(actor_shared_raw_norm, torch.tensor(4.0))
    assert torch.allclose(critic_shared_raw_norm, torch.tensor(40.0))
    assert torch.allclose(actor_shared_norm, torch.tensor(4.0 / (5.0 + 1e-6)))
    assert torch.allclose(critic_shared_norm, torch.tensor(40.0 / (50.0 + 1e-6)))
    assert torch.allclose(actor_clip_scale, torch.tensor(1.0 / (5.0 + 1e-6)))
    assert torch.allclose(critic_clip_scale, torch.tensor(1.0 / (50.0 + 1e-6)))
    assert torch.allclose(actor_clip_frac, torch.tensor(1.0))
    assert torch.allclose(critic_clip_frac, torch.tensor(1.0))
    assert torch.allclose(shared_norm, torch.tensor(1.6), atol=1e-6)
    assert torch.allclose(
        model.target_noop_key.weight.grad,
        torch.tensor([[3.0 / (5.0 + 1e-6)]]),
    )
    assert torch.allclose(model.value_head.weight.grad, torch.tensor([[30.0 / (50.0 + 1e-6)]]))
    assert torch.allclose(model.shared.weight.grad, torch.tensor([[1.6]]), atol=1e-6)


def test_policy_value_grad_clip_retains_shared_forward_graph():
    class Toy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.trunk = torch.nn.Linear(1, 1, bias=False)
            self.target_noop_key = torch.nn.Linear(1, 1, bias=False)
            self.value_head = torch.nn.Linear(1, 1, bias=False)

    model = Toy()
    with torch.no_grad():
        model.trunk.weight.fill_(1.0)
        model.target_noop_key.weight.fill_(1.0)
        model.value_head.weight.fill_(1.0)

    h = model.trunk(torch.tensor([[2.0]]))
    actor_loss = model.target_noop_key(h).sum()
    critic_loss = 10.0 * model.value_head(h).sum()

    (
        actor_norm,
        critic_norm,
        actor_shared_norm,
        critic_shared_norm,
        shared_norm,
        actor_shared_raw_norm,
        critic_shared_raw_norm,
        actor_clip_scale,
        critic_clip_scale,
        actor_clip_frac,
        critic_clip_frac,
    ) = _backward_actor_critic_with_group_clips(model, actor_loss, critic_loss, 1.0)

    sqrt2 = math.sqrt(2.0)
    assert torch.allclose(actor_norm, torch.tensor(2.0 * sqrt2))
    assert torch.allclose(critic_norm, torch.tensor(20.0 * sqrt2))
    assert torch.allclose(actor_shared_raw_norm, torch.tensor(2.0))
    assert torch.allclose(critic_shared_raw_norm, torch.tensor(20.0))
    actor_scale = 1.0 / (2.0 * sqrt2 + 1e-6)
    critic_scale = 1.0 / (20.0 * sqrt2 + 1e-6)
    expected_shared = 2.0 * actor_scale + 20.0 * critic_scale
    assert torch.allclose(actor_shared_norm, torch.tensor(2.0 * actor_scale))
    assert torch.allclose(critic_shared_norm, torch.tensor(20.0 * critic_scale))
    assert torch.allclose(actor_clip_scale, torch.tensor(actor_scale))
    assert torch.allclose(critic_clip_scale, torch.tensor(critic_scale))
    assert torch.allclose(actor_clip_frac, torch.tensor(1.0))
    assert torch.allclose(critic_clip_frac, torch.tensor(1.0))
    assert torch.allclose(shared_norm, torch.tensor(expected_shared))
    assert torch.allclose(model.trunk.weight.grad, torch.tensor([[expected_shared]]))


class _FixedPolicy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dummy = torch.nn.Parameter(torch.zeros(()))
        self.value_encoder = HLGaussLoss(min_value=-1.0, max_value=1.0, num_bins=5)
        self.new_launch_logits = torch.logit(torch.tensor([[0.25, 0.25]]))
        self.new_target_logits = torch.log(
            torch.tensor([[[0.20, 0.80], [0.50, 0.50]]])
        )
        self.new_fraction_alpha = torch.tensor([[2.0, 2.0]])
        self.new_fraction_beta = torch.tensor([[2.0, 2.0]])

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
            fraction_alpha=(
                self.new_fraction_alpha.expand(b, -1).to(feats.planet_feats.device)
                + self.dummy * 0.0
            ),
            fraction_beta=(
                self.new_fraction_beta.expand(b, -1).to(feats.planet_feats.device)
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
    frac_lp = _beta_log_prob(
        model.new_fraction_alpha[0, 0],
        model.new_fraction_beta[0, 0],
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
        advantage_transform="none",
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
    frac_lp = _beta_log_prob(
        model.new_fraction_alpha,
        model.new_fraction_beta,
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
        advantage_transform="none",
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
    frac_lp = _beta_log_prob(
        model.new_fraction_alpha[0, 0],
        model.new_fraction_beta[0, 0],
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
        advantage_transform="none",
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
    frac_lp = _beta_log_prob(
        model.new_fraction_alpha[0, 0],
        model.new_fraction_beta[0, 0],
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
        advantage_transform="none",
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
