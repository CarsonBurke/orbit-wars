"""End-to-end smoke tests for `ppo_update`.

Builds a tiny synthetic batch that mirrors the keys `_stack_trajectories`
produces, runs one PPO update, and asserts the returned metrics are finite.
Also keeps a regression guard on the log_prob recompute invariant
(ratio ≈ 1 on epoch 0).
"""

from __future__ import annotations

import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
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
from owars.training import train as train_mod
from owars.training.config import RunConfig
from owars.training.ppo import (
    _backward_actor_critic_with_group_clips,
    _batch_normalize_advantage,
    _beta_log_prob,
    _conditional_action_entropy,
    _distributional_value_loss,
    _fixed_minibatches,
    _fixed_minibatches_by_count,
    _minibatch_loss_scale,
    _rank_gaussian_advantage,
    _source_capacity_for_minibatch,
    _source_planet_bucket,
    _stage_target_legal_mask,
    compute_gae,
    compute_old_log_probs,
    compute_old_log_probs_and_values,
    ppo_update,
    value_only_update,
)
from owars.training.rollout import TrajectoryRecordRef

# A non-degenerate 8-planet position. Player 0 owns the first three; the rest
# are enemy/neutral so owned planets have legal targets. Every planet sits in
# the left half of the board (x <= 30), and the sun spans x in [40, 60], so
# every source->target straight line clears the sun and launches actually fire
# — without that, the launch/target/fraction log-probs collapse to zero and the
# recompute invariant below is vacuously satisfied.
_TOY_PLANETS = [
    # (id, owner, x, y, radius, ships, production)
    Planet(0, 0, 8.0, 10.0, 2.1, 80, 3),  # owned
    Planet(1, 0, 8.0, 40.0, 2.1, 80, 3),  # owned
    Planet(2, 0, 8.0, 70.0, 2.1, 80, 3),  # owned
    Planet(3, 1, 8.0, 95.0, 1.7, 30, 2),  # enemy
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
            player=0,
            step=0,
            planets=list(_TOY_PLANETS),
            fleets=[],
            angular_velocity=0.0,
            initial_planets=[],
            comet_planet_ids=set(),
            comets=[],
            remaining_overage_time=60.0,
        )
        for _ in range(batch_size)
    ]
    feats = encode_observations(
        obs,
        include_fleet_targets=model.cfg.encoder_backend == "destination_conditioned",
    )
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
        "global_feats": feats.global_feats,
        "planet_feats": feats.planet_feats,
        "planet_mask": feats.planet_mask,
        "planet_owned_mask": feats.planet_owned_mask,
        "planet_ids": feats.planet_ids,
        "planet_garrison": feats.planet_garrison,
        "fleet_feats": feats.fleet_feats,
        "fleet_mask": feats.fleet_mask,
        "fleet_target_planet_idx": feats.fleet_target_planet_idx,
        "launch": launch,
        "target_idx": target_idx,
        "fraction": fraction,
        "old_log_prob": log_prob,
        "old_log_prob_computed": torch.tensor(True),
        "owned_mask": feats.planet_owned_mask,
        "advantage": torch.randn(batch_size),
        # Returns stay inside the default value-head support [-2, 2].
        "return": torch.randn(batch_size).clamp(-1.0, 1.0),
        "target_legal_mask": target_legal_mask,
    }


def _with_source_actor_keys(batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    out = dict(batch)
    source_rows, source_cols = torch.nonzero(
        batch["owned_mask"].bool(),
        as_tuple=True,
    )
    counts = torch.bincount(source_rows, minlength=int(batch["owned_mask"].shape[0]))
    out["actor_source_row_idx"] = source_rows.long()
    out["actor_source_col_idx"] = source_cols.long()
    out["actor_source_row_offsets"] = torch.cat(
        (torch.zeros(1, dtype=torch.long), counts.cumsum(0).long()),
    )
    out["actor_launch"] = batch["launch"][source_rows, source_cols].float()
    out["actor_raw_launch"] = out["actor_launch"].clone()
    out["actor_target_idx"] = batch["target_idx"][source_rows, source_cols].long()
    out["actor_fraction"] = batch["fraction"][source_rows, source_cols].float()
    out["actor_target_legal_mask"] = batch["target_legal_mask"][
        source_rows,
        source_cols,
    ].bool()
    return out


def test_ppo_update_runs_and_returns_finite_metrics():
    cfg = OrbitPolicyConfig(dim=32, ff_dim=64, depth=2, n_heads=2)
    model = OrbitPolicy(cfg)
    batch = _toy_batch(model, batch_size=8)
    optim = torch.optim.AdamW(model.parameters(), lr=3e-4)

    log = ppo_update(
        model,
        optim,
        batch,
        value_coef=0.5,
        target_entropy_coef=0.01,
        fraction_entropy_coef=0.0,
        norm_advantage=True,
        advantage_transform="rankgauss",
        clip_coef=0.2,
        clip_coef_high=0.28,
        epochs=2,
        minibatch_size=4,
        grad_clip=0.5,
    )

    for name in (
        "policy_loss",
        "value_loss",
        "entropy",
        "approx_kl",
        "ratio_clip_frac_high",
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
    # updates, so we just sanity-check the [0, 1] range here.
    assert 0.0 <= log.ratio_clip_frac_high <= 1.0, log.ratio_clip_frac_high
    assert 0.0 <= log.pos_frac <= 1.0, log.pos_frac
    assert log.value_loss >= 0.0, log.value_loss


def test_ppo_update_runs_with_source_major_actor_records():
    cfg = OrbitPolicyConfig(dim=32, ff_dim=64, depth=2, n_heads=2)
    model = OrbitPolicy(cfg)
    batch = _with_source_actor_keys(_toy_batch(model, batch_size=8))
    optim = torch.optim.AdamW(model.parameters(), lr=3e-4)

    log = ppo_update(
        model,
        optim,
        batch,
        value_coef=0.5,
        target_entropy_coef=0.0,
        fraction_entropy_coef=0.0,
        norm_advantage=True,
        advantage_transform="rankgauss",
        clip_coef=0.2,
        clip_coef_high=0.28,
        epochs=1,
        minibatch_size=4,
        grad_clip=0.5,
    )

    assert math.isfinite(log.policy_loss)
    assert math.isfinite(log.value_loss)
    assert math.isfinite(log.approx_kl)
    assert 0.0 <= log.ratio_clip_frac_high <= 1.0


def test_batch_normalize_advantage_weighted():
    # Per-row advantages weighted by the active kernel's per-row mass; a row with
    # zero weight (e.g. no launch sources) must not influence the statistic.
    adv = torch.tensor([1.0, 2.0, 3.0, 100.0])
    w = torch.tensor([2.0, 1.0, 3.0, 0.0])
    out = _batch_normalize_advantage(adv, w)

    mean = (adv * w).sum() / w.sum()
    std = (((adv - mean).square() * w).sum() / w.sum()).sqrt()
    expected = (adv - mean) / (std + 1e-8)
    assert torch.allclose(out, expected, atol=1e-6)
    # Weighted mean removed -> weighted mean of the output is ~0.
    assert abs(float((out * w).sum() / w.sum())) < 1e-5


def test_batch_normalize_advantage_degenerate_falls_back():
    adv = torch.tensor([1.0, 2.0, 3.0])
    # Zero total weight (no owned planets / no sources) -> unchanged.
    assert torch.allclose(_batch_normalize_advantage(adv, torch.zeros(3)), adv.float())
    # Zero-variance advantage -> unchanged.
    flat = torch.tensor([5.0, 5.0, 5.0])
    assert torch.allclose(
        _batch_normalize_advantage(flat, torch.ones(3)),
        flat,
    )


def test_ppo_update_batch_scope_advnorm_runs_on_both_paths():
    cfg = OrbitPolicyConfig(dim=32, ff_dim=64, depth=2, n_heads=2)
    for with_source in (False, True):
        model = OrbitPolicy(cfg)
        batch = _toy_batch(model, batch_size=8)
        if with_source:
            batch = _with_source_actor_keys(batch)
        optim = torch.optim.AdamW(model.parameters(), lr=3e-4)
        log = ppo_update(
            model,
            optim,
            batch,
            value_coef=0.5,
            target_entropy_coef=0.0,
            fraction_entropy_coef=0.0,
            norm_advantage=True,
            norm_advantage_scope="batch",
            advantage_transform="none",
            clip_coef=0.2,
            clip_coef_high=0.28,
            epochs=2,
            minibatch_size=4,
            grad_clip=0.5,
        )
        assert math.isfinite(log.policy_loss), with_source
        assert math.isfinite(log.value_loss), with_source
        assert math.isfinite(log.approx_kl), with_source


@pytest.mark.parametrize("with_source", [False, True])
def test_batch_scope_matches_minibatch_scope_for_single_full_minibatch(with_source):
    # With one minibatch spanning the whole rollout, the kernel's per-minibatch
    # z-score (minibatch scope) sees exactly the whole-batch statistic that batch
    # scope applies pre-loop, so the actor objective must coincide. This must hold
    # on BOTH actor paths: the dense kernel weights by owned-planet count and the
    # source-major kernel by launch-source count, and batch scope mirrors each.
    cfg = OrbitPolicyConfig(dim=32, ff_dim=64, depth=2, n_heads=2)
    base = OrbitPolicy(cfg)
    batch = _toy_batch(base, batch_size=8)
    if with_source:
        batch = _with_source_actor_keys(batch)

    def run(scope: str) -> float:
        model = OrbitPolicy(cfg)
        model.load_state_dict(base.state_dict())
        optim = torch.optim.AdamW(model.parameters(), lr=3e-4)
        log = ppo_update(
            model,
            optim,
            dict(batch),
            value_coef=0.5,
            target_entropy_coef=0.0,
            fraction_entropy_coef=0.0,
            norm_advantage=True,
            norm_advantage_scope=scope,
            advantage_transform="none",
            clip_coef=0.2,
            clip_coef_high=0.28,
            epochs=1,
            minibatch_size=8,
            grad_clip=0.5,
        )
        return log.policy_loss

    assert run("batch") == pytest.approx(run("minibatch"), rel=1e-4, abs=1e-5)


def test_batch_scope_source_path_weights_by_launch_source_count_not_owned_count():
    # Regression guard for the source-major path: batch scope must weight rows by
    # launch-source count (the source kernel's source_w), NOT total owned-planet
    # count. Drop one source from row 0 so its source count (2) differs from its
    # owned count (3); a self-consistent batch where the two weightings diverge.
    # With correct source-count weighting the pre-loop z-score's centering still
    # cancels against the source-weighted loss mean (epoch-0 policy_loss ~ 0 and
    # equals minibatch scope); owned-count weighting would break that cancellation.
    cfg = OrbitPolicyConfig(dim=32, ff_dim=64, depth=2, n_heads=2)
    base = OrbitPolicy(cfg)
    batch = _with_source_actor_keys(_toy_batch(base, batch_size=8))
    keep = torch.arange(1, batch["actor_source_row_idx"].numel())
    rows = batch["actor_source_row_idx"][keep]
    for key in (
        "actor_source_row_idx",
        "actor_source_col_idx",
        "actor_launch",
        "actor_raw_launch",
        "actor_target_idx",
        "actor_fraction",
        "actor_target_legal_mask",
    ):
        batch[key] = batch[key][keep]
    counts = torch.bincount(rows, minlength=int(batch["owned_mask"].shape[0]))
    batch["actor_source_row_offsets"] = torch.cat(
        (torch.zeros(1, dtype=torch.long), counts.cumsum(0).long()),
    )
    # Precondition: the two candidate weightings genuinely disagree on row 0.
    assert int(counts[0]) < int(batch["owned_mask"][0].sum())

    def run(scope: str) -> float:
        model = OrbitPolicy(cfg)
        model.load_state_dict(base.state_dict())
        optim = torch.optim.AdamW(model.parameters(), lr=3e-4)
        log = ppo_update(
            model,
            optim,
            dict(batch),
            value_coef=0.5,
            target_entropy_coef=0.0,
            fraction_entropy_coef=0.0,
            norm_advantage=True,
            norm_advantage_scope=scope,
            advantage_transform="none",
            clip_coef=0.2,
            clip_coef_high=0.28,
            epochs=1,
            minibatch_size=8,
            grad_clip=0.5,
        )
        return log.policy_loss

    assert run("batch") == pytest.approx(run("minibatch"), rel=1e-4, abs=1e-5)


def test_source_major_ppo_capacity_uses_log_spaced_source_buckets():
    assert _source_planet_bucket(1, 64) == 8
    assert _source_planet_bucket(9, 64) == 12
    assert _source_planet_bucket(23, 64) == 24
    assert _source_planet_bucket(33, 64) == 48
    assert _source_planet_bucket(80, 64) == 64
    assert _source_capacity_for_minibatch(1024, 64, 23) == 1024 * 24
    assert _source_capacity_for_minibatch(1024, 24, 23) == 1024 * 24
    assert _source_capacity_for_minibatch(0, 0, 0) == 1


def test_source_major_ppo_metrics_match_dense_path_with_stale_actor_old_log_prob():
    cfg = OrbitPolicyConfig(dim=32, ff_dim=64, depth=2, n_heads=2)
    base = OrbitPolicy(cfg)
    dense_model = OrbitPolicy(cfg)
    source_model = OrbitPolicy(cfg)
    dense_model.load_state_dict(base.state_dict())
    source_model.load_state_dict(base.state_dict())
    batch = _toy_batch(base, batch_size=8)
    source_batch = _with_source_actor_keys(batch)
    source_batch["actor_old_log_prob"] = torch.zeros_like(source_batch["actor_launch"]) + 123.0
    dense_optim = torch.optim.AdamW(dense_model.parameters(), lr=0.0)
    source_optim = torch.optim.AdamW(source_model.parameters(), lr=0.0)
    kwargs = dict(
        value_coef=0.5,
        target_entropy_coef=0.0,
        fraction_entropy_coef=0.0,
        norm_advantage=True,
        advantage_transform="rankgauss",
        clip_coef=0.2,
        clip_coef_high=0.28,
        epochs=1,
        minibatch_size=4,
        grad_clip=0.5,
    )

    dense_log = ppo_update(dense_model, dense_optim, batch, **kwargs)
    source_log = ppo_update(source_model, source_optim, source_batch, **kwargs)

    for name in (
        "policy_loss",
        "value_loss",
        "entropy",
        "approx_kl",
        "ratio_clip_frac_high",
        "target_entropy",
        "fraction_entropy",
        "move_prob",
        "target_confidence",
        "owned_planets_mean",
        "executed_launch_frac",
        "turn_no_action_frac",
        "legal_target_count_mean",
    ):
        assert getattr(source_log, name) == pytest.approx(getattr(dense_log, name), abs=1e-5)


def test_ppo_update_reports_entropy_diagnostics_when_coef_zero():
    cfg = OrbitPolicyConfig(dim=32, ff_dim=64, depth=2, n_heads=2)
    model = OrbitPolicy(cfg)
    batch = _toy_batch(model, batch_size=8)
    optim = torch.optim.AdamW(model.parameters(), lr=3e-4)

    log = ppo_update(
        model,
        optim,
        batch,
        value_coef=0.5,
        target_entropy_coef=0.0,
        fraction_entropy_coef=0.0,
        norm_advantage=True,
        advantage_transform="rankgauss",
        clip_coef=0.2,
        clip_coef_high=0.28,
        epochs=1,
        minibatch_size=4,
        grad_clip=0.5,
    )

    assert math.isfinite(log.entropy)
    assert math.isfinite(log.target_entropy)
    assert math.isfinite(log.fraction_entropy)
    assert log.entropy > 0.0
    assert log.target_entropy > 0.0


def test_deferred_old_log_probs_match_sampler_records():
    cfg = OrbitPolicyConfig(dim=32, ff_dim=64, depth=2, n_heads=2)
    model = OrbitPolicy(cfg)
    batch = _toy_batch(model, batch_size=8)

    old_log_prob = compute_old_log_probs(
        model,
        batch,
        minibatch_size=4,
        compile_mode=None,
    )

    assert torch.allclose(old_log_prob, batch["old_log_prob"], atol=1e-5, rtol=1e-5)


def test_compute_old_log_probs_uses_eval_mode_and_restores_training_state():
    cfg = OrbitPolicyConfig(
        dim=32,
        ff_dim=64,
        depth=2,
        n_heads=2,
        dropout=0.9,
        value_num_bins=21,
    )
    model = OrbitPolicy(cfg)
    batch = _toy_batch(model, batch_size=4)
    model.train()

    torch.manual_seed(1)
    first = compute_old_log_probs(model, batch, minibatch_size=2)
    torch.manual_seed(2)
    second = compute_old_log_probs(model, batch, minibatch_size=2)

    assert model.training
    torch.testing.assert_close(first, second, atol=0.0, rtol=0.0)


def test_trim_ppo_batch_fleet_width_keeps_destination_sidecar_aligned():
    batch = {
        "fleet_feats": torch.zeros(2, 513, 20),
        "fleet_mask": torch.zeros(2, 513, dtype=torch.bool),
        "fleet_target_planet_idx": torch.full((2, 513), -1, dtype=torch.long),
    }
    batch["fleet_mask"][0, 3] = True
    batch["fleet_mask"][1, 18] = True
    batch["fleet_target_planet_idx"][0, 3] = 2
    batch["fleet_target_planet_idx"][1, 18] = 5

    trimmed = train_mod._trim_ppo_batch_fleet_width(batch)

    assert trimmed["fleet_feats"].shape == (2, 64, 20)
    assert trimmed["fleet_mask"].shape == (2, 64)
    assert trimmed["fleet_target_planet_idx"].shape == (2, 64)
    assert int(trimmed["fleet_target_planet_idx"][0, 3]) == 2
    assert int(trimmed["fleet_target_planet_idx"][1, 18]) == 5


def test_trim_ppo_batch_fleet_width_can_pad_for_compile_bucket():
    batch = {
        "fleet_feats": torch.zeros(3, 25, 20),
        "fleet_mask": torch.zeros(3, 25, dtype=torch.bool),
        "fleet_target_planet_idx": torch.full((3, 25), -1, dtype=torch.long),
    }
    batch["fleet_mask"][1, 24] = True
    batch["fleet_target_planet_idx"][1, 24] = 7

    padded = train_mod._trim_ppo_batch_fleet_width(batch, pad_to_bucket=True)

    assert padded["fleet_feats"].shape[1] == 64
    assert padded["fleet_mask"][1, 24]
    assert int(padded["fleet_target_planet_idx"][1, 24]) == 7
    assert not bool(padded["fleet_mask"][:, 25:].any())
    assert torch.all(padded["fleet_target_planet_idx"][:, 25:] == -1)


def test_train_num_players_helper_uses_configured_mix():
    cfg = RunConfig.from_dict({"game": {"num_players": 2, "train_num_players": [2, 4]}})

    assert train_mod._train_num_players(cfg) == (2, 4)


def test_format_episode_counts_keeps_sniper_update_game_count_with_even_mix():
    counts = train_mod._format_episode_counts(
        128,
        (2, 4),
        train_mod.random.Random(0),
    )

    assert counts[2] == 64
    assert counts[4] == 64
    assert sum(counts.values()) == 128


def test_format_episode_counts_distributes_odd_remainder():
    counts = train_mod._format_episode_counts(
        129,
        (2, 4),
        train_mod.random.Random(0),
    )

    assert sorted(counts.values()) == [64, 65]
    assert sum(counts.values()) == 129


def test_training_vec_counts_split_mixed_formats_without_multiplying_envs():
    cfg = RunConfig.from_dict(
        {
            "game": {"train_num_players": [2, 4]},
            "rollout": {"num_envs": 128},
            "ppo": {"pretrain_updates": 0},
        }
    )

    assert train_mod._training_vec_counts(cfg) == {2: 64, 4: 64}


def test_training_vec_counts_keep_full_pretrain_vec_when_needed():
    cfg = RunConfig.from_dict(
        {
            "game": {"num_players": 2, "train_num_players": [2, 4]},
            "rollout": {"num_envs": 128},
            "ppo": {"pretrain_updates": 1},
        }
    )

    assert train_mod._training_vec_counts(cfg)[2] == 128
    assert train_mod._training_vec_counts(cfg)[4] == 64


def test_policy_compile_rows_follow_expected_current_model_seats():
    cfg = RunConfig.from_dict(
        {
            "opponents": {
                "mode": "no_builtins",
                "current_learner_prob": 0.4,
            },
        }
    )

    assert train_mod._policy_compile_rows_for_rollout(
        cfg,
        num_envs=128,
        num_players=2,
    ) == 256
    assert train_mod._policy_compile_rows_for_rollout(
        cfg,
        num_envs=128,
        num_players=4,
    ) == 384
    assert train_mod._policy_compile_rows_for_rollout(
        cfg,
        num_envs=64,
        num_players=4,
    ) == 192


def test_policy_compile_rows_for_sampled_rollout_counts_actual_current_slots():
    current = train_mod.OpponentSlot(name=train_mod.LEARNER_NAME, agent=None)
    snapshot = train_mod.OpponentSlot(name="snapshot_1", agent=None)

    assert (
        train_mod._policy_compile_rows_for_sampled_rollout(
            [[snapshot] for _ in range(128)],
            learner_seats=[0] * 128,
            num_players=2,
        )
        == 128
    )
    assert (
        train_mod._policy_compile_rows_for_sampled_rollout(
            [[current] if i % 2 == 0 else [snapshot] for i in range(128)],
            learner_seats=[0] * 128,
            num_players=2,
        )
        == 192
    )
    assert (
        train_mod._policy_compile_rows_for_sampled_rollout(
            [
                [current, snapshot, current],
                [snapshot, current, snapshot],
            ],
            learner_seats=[0, 2],
            num_players=4,
            bucket_multiple=4,
        )
        == 8
    )


def test_ppo_minibatch_size_caps_only_high_fleet_bucket():
    cfg = RunConfig.from_dict({"optim": {"minibatch_size": 4096}})

    assert train_mod._ppo_minibatch_size_for_fleet_width(cfg, 1024) == 4096
    assert train_mod._ppo_minibatch_size_for_fleet_width(cfg, 2048) == 2048


def test_ppo_minibatch_size_respects_smaller_configured_size():
    cfg = RunConfig.from_dict({"optim": {"minibatch_size": 1024}})

    assert train_mod._ppo_minibatch_size_for_fleet_width(cfg, 2048) == 1024


def test_destination_conditioned_value_only_update_after_fleet_trim_smoke():
    cfg = OrbitPolicyConfig(
        dim=32,
        ff_dim=64,
        depth=1,
        n_heads=2,
        encoder_backend="destination_conditioned",
    )
    model = OrbitPolicy(cfg)
    batch = _toy_batch(model, batch_size=4)
    batch["fleet_feats"] = torch.zeros(4, 32, 20)
    batch["fleet_mask"] = torch.zeros(4, 32, dtype=torch.bool)
    batch["fleet_target_planet_idx"] = torch.full((4, 32), -1, dtype=torch.long)
    batch["fleet_mask"][:, 18] = True
    batch["fleet_target_planet_idx"][:, 18] = 0
    batch = train_mod._trim_ppo_batch_fleet_width(batch)
    optim = torch.optim.AdamW(model.parameters(), lr=0.0)

    got = value_only_update(model, optim, batch, epochs=1, minibatch_size=2, grad_clip=1.0)

    assert math.isfinite(got)


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


def test_discounted_return_normalizer_matches_cleanrl_reward_scale():
    norm = train_mod.DiscountedReturnNormalizer(gamma=0.5, clip=2.0, epsilon=1e-8)
    episodes = [
        np.asarray([1.0, 2.0, 10.0], dtype=np.float32),
        np.asarray([1.0], dtype=np.float32),
    ]

    mean = 0.0
    var = 1.0
    count = 1.0e-4
    expected_episodes = [np.empty_like(rewards, dtype=np.float32) for rewards in episodes]
    running = [0.0 for _ in episodes]
    clip_count = 0
    reward_count = 0
    for step in range(max(len(rewards) for rewards in episodes)):
        active = [
            episode_idx for episode_idx, rewards in enumerate(episodes) if step < len(rewards)
        ]
        rewards = np.asarray(
            [episodes[episode_idx][step] for episode_idx in active],
            dtype=np.float32,
        )
        for episode_idx, reward in zip(active, rewards, strict=True):
            running[episode_idx] = running[episode_idx] * 0.5 + float(reward)
        running_step = np.asarray([running[episode_idx] for episode_idx in active])
        batch_mean = float(running_step.mean())
        batch_var = float(running_step.var())
        batch_count = float(running_step.size)
        delta = batch_mean - mean
        total = count + batch_count
        m2 = var * count + batch_var * batch_count + delta * delta * count * batch_count / total
        mean += delta * batch_count / total
        var = m2 / total
        count = total
        scaled = rewards / math.sqrt(var + 1e-8)
        clip_count += int(np.count_nonzero(np.abs(scaled) > 2.0))
        reward_count += int(scaled.size)
        clipped = np.clip(scaled, -2.0, 2.0)
        for local_idx, episode_idx in enumerate(active):
            expected_episodes[episode_idx][step] = clipped[local_idx]

    got_first, got_second = norm.normalize_episodes(episodes)

    assert np.allclose(got_first, expected_episodes[0])
    assert np.allclose(got_second, expected_episodes[1])
    assert math.isclose(norm.mean, mean)
    assert math.isclose(norm.var, var)
    assert math.isclose(norm.batch_clip_frac, clip_count / reward_count)


def test_stack_trajectories_normalizes_rewards_before_gae(monkeypatch):
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
    norm = train_mod.DiscountedReturnNormalizer(gamma=1.0, clip=None)
    expected_rewards = norm.normalize_episode(rewards)
    _adv, expected_return = compute_gae(
        expected_rewards,
        values,
        gamma=1.0,
        lam=1.0,
    )

    batch = train_mod._stack_trajectories(
        [traj],
        gamma=1.0,
        gae_lambda=1.0,
        critic_mtp_horizon=2,
        reward_normalizer=train_mod.DiscountedReturnNormalizer(gamma=1.0, clip=None),
    )

    assert torch.allclose(batch["return"], torch.from_numpy(expected_return))
    assert torch.allclose(batch["return_mtp"][0], torch.tensor(expected_return[:2]))


def test_save_ppo_checkpoint_includes_critic_return_normalizer(tmp_path: Path):
    cfg = OrbitPolicyConfig(dim=32, ff_dim=64, depth=2, n_heads=2)
    model = OrbitPolicy(cfg)
    norm = train_mod.DiscountedReturnNormalizer(gamma=0.9, clip=3.0)
    norm.normalize_episode(np.asarray([1.0, 2.0], dtype=np.float32))
    path = tmp_path / "ckpt.pt"

    train_mod._save_ppo_checkpoint(model, path, norm)
    payload = torch.load(path, map_location="cpu", weights_only=False)

    assert "critic_return_normalizer" in payload
    restored = train_mod.DiscountedReturnNormalizer(gamma=0.9, clip=3.0)
    restored.load_state_dict(payload["critic_return_normalizer"])
    assert restored.gamma == norm.gamma
    assert restored.clip == norm.clip
    assert math.isclose(restored.mean, norm.mean)
    assert math.isclose(restored.var, norm.var)
    assert math.isclose(restored.count, norm.count)


def test_restore_reward_normalizer_from_checkpoint_rejects_raw_critic_checkpoint():
    norm = train_mod.DiscountedReturnNormalizer(gamma=0.9)

    with pytest.raises(ValueError, match="critic_return_normalizer"):
        train_mod._restore_reward_normalizer_from_checkpoint(
            norm,
            {"model": {}},
            "old.pt",
        )


def test_restore_reward_normalizer_from_checkpoint_loads_state():
    source = train_mod.DiscountedReturnNormalizer(gamma=0.9, clip=3.0)
    source.normalize_episode(np.asarray([1.0, 2.0], dtype=np.float32))
    restored = train_mod.DiscountedReturnNormalizer(gamma=0.9, clip=3.0)

    train_mod._restore_reward_normalizer_from_checkpoint(
        restored,
        {"critic_return_normalizer": source.state_dict()},
        "new.pt",
    )

    assert restored.gamma == source.gamma
    assert restored.clip == source.clip
    assert math.isclose(restored.mean, source.mean)


def test_restore_reward_normalizer_from_checkpoint_rejects_config_mismatch():
    source = train_mod.DiscountedReturnNormalizer(gamma=0.9, clip=3.0)
    restored = train_mod.DiscountedReturnNormalizer(gamma=1.0, clip=3.0)

    with pytest.raises(ValueError, match="gamma"):
        train_mod._restore_reward_normalizer_from_checkpoint(
            restored,
            {"critic_return_normalizer": source.state_dict()},
            "mismatch.pt",
        )


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


def test_stack_trajectories_can_defer_old_log_prob(monkeypatch):
    monkeypatch.setattr(train_mod, "_stack_encoded", lambda _trajs: {})
    rewards = np.asarray([0.0, 1.0], dtype=np.float32)
    traj = SimpleNamespace(
        reward=rewards.tolist(),
        value=[torch.tensor(0.0) for _ in rewards],
        launch=[torch.ones(2) for _ in rewards],
        target_idx=[torch.zeros(2, dtype=torch.long) for _ in rewards],
        fraction=[torch.full((2,), 0.5) for _ in rewards],
        log_prob=[],
        owned_mask=[torch.ones(2, dtype=torch.bool) for _ in rewards],
        target_legal_mask=[torch.ones(2, 2, dtype=torch.bool) for _ in rewards],
    )

    batch = train_mod._stack_trajectories(
        [traj],
        gamma=1.0,
        gae_lambda=1.0,
        include_old_log_prob=False,
    )

    assert torch.equal(batch["old_log_prob"], torch.zeros_like(batch["launch"]))

    fallback = train_mod._stack_trajectories(
        [traj],
        gamma=1.0,
        gae_lambda=1.0,
        include_old_log_prob=True,
    )

    assert torch.equal(fallback["old_log_prob"], torch.zeros_like(fallback["launch"]))
    assert not bool(fallback["old_log_prob_computed"])


def test_refresh_batch_advantages_uses_recomputed_values(monkeypatch):
    monkeypatch.setattr(train_mod, "_stack_encoded", lambda _trajs: {})
    rewards = np.asarray([1.0, -0.25, 0.5], dtype=np.float32)
    traj = SimpleNamespace(
        reward=rewards.tolist(),
        value=[torch.tensor(0.0) for _ in rewards],
        launch=[torch.ones(2) for _ in rewards],
        target_idx=[torch.zeros(2, dtype=torch.long) for _ in rewards],
        fraction=[torch.full((2,), 0.5) for _ in rewards],
        log_prob=[],
        owned_mask=[torch.ones(2, dtype=torch.bool) for _ in rewards],
        target_legal_mask=[torch.ones(2, 2, dtype=torch.bool) for _ in rewards],
    )
    batch = train_mod._stack_trajectories(
        [traj],
        gamma=0.9,
        gae_lambda=0.8,
        value_gae_lambda=1.0,
        include_old_log_prob=False,
    )
    recomputed_values = torch.tensor([0.25, -0.1, 0.4])

    train_mod._refresh_batch_advantages_from_values(
        batch,
        recomputed_values,
        gamma=0.9,
        gae_lambda=0.8,
        value_gae_lambda=1.0,
        critic_mtp_horizon=2,
    )

    expected_adv, _policy_return = compute_gae(
        rewards,
        recomputed_values.numpy(),
        gamma=0.9,
        lam=0.8,
    )
    _value_adv, expected_return = compute_gae(
        rewards,
        recomputed_values.numpy(),
        gamma=0.9,
        lam=1.0,
    )
    torch.testing.assert_close(batch["advantage"], torch.from_numpy(expected_adv))
    torch.testing.assert_close(batch["return"], torch.from_numpy(expected_return))
    torch.testing.assert_close(batch["value"], recomputed_values)
    assert bool(batch["values_computed"])


def test_refresh_batch_advantages_does_not_update_reward_normalizer(monkeypatch):
    monkeypatch.setattr(train_mod, "_stack_encoded", lambda _trajs: {})
    rewards = np.asarray([1.0, 2.0, -0.5], dtype=np.float32)
    traj = SimpleNamespace(
        reward=rewards.tolist(),
        value=[torch.tensor(0.0) for _ in rewards],
        launch=[torch.ones(1) for _ in rewards],
        target_idx=[torch.zeros(1, dtype=torch.long) for _ in rewards],
        fraction=[torch.full((1,), 0.5) for _ in rewards],
        log_prob=[],
        owned_mask=[torch.ones(1, dtype=torch.bool) for _ in rewards],
        target_legal_mask=[torch.ones(1, 1, dtype=torch.bool) for _ in rewards],
    )
    normalizer = train_mod.DiscountedReturnNormalizer(gamma=0.9, clip=None)
    batch = train_mod._stack_trajectories(
        [traj],
        gamma=0.9,
        gae_lambda=0.8,
        reward_normalizer=normalizer,
        include_old_log_prob=False,
    )
    state = normalizer.state_dict().copy()

    train_mod._refresh_batch_advantages_from_values(
        batch,
        torch.tensor([0.25, -0.1, 0.4]),
        gamma=0.9,
        gae_lambda=0.8,
        value_gae_lambda=None,
        critic_mtp_horizon=1,
    )

    assert normalizer.state_dict() == state


def test_compute_old_log_probs_and_values_matches_separate_old_log_prob():
    cfg = OrbitPolicyConfig(dim=32, ff_dim=64, depth=2, n_heads=2)
    model = OrbitPolicy(cfg)
    batch = _toy_batch(model, batch_size=4)

    old_log_prob, values = compute_old_log_probs_and_values(
        model,
        batch,
        minibatch_size=2,
        compile_mode=None,
    )
    expected_old_log_prob = compute_old_log_probs(
        model,
        batch,
        minibatch_size=2,
        compile_mode=None,
    )

    torch.testing.assert_close(old_log_prob, expected_old_log_prob)
    assert values.shape == (batch["planet_feats"].shape[0],)
    assert torch.isfinite(values).all()


def test_stack_trajectories_chunked_records_match_row_records():
    planets = 3

    def chunk(offset: int, rows: int, fleets: int) -> dict[str, torch.Tensor | None]:
        row_base = torch.arange(offset, offset + rows, dtype=torch.float32)
        return {
            "global_feats": row_base[:, None].repeat(1, 2),
            "planet_feats": (row_base[:, None, None] + torch.arange(planets * 4).view(1, planets, 4)),
            "planet_mask": torch.ones(rows, planets, dtype=torch.bool),
            "planet_owned_mask": torch.tensor(
                [[True, False, True], [True, True, False], [False, True, True]][:rows],
            ),
            "planet_ids": torch.arange(planets, dtype=torch.long).repeat(rows, 1) + offset,
            "planet_garrison": row_base[:, None] + torch.arange(planets, dtype=torch.float32),
            "fleet_feats": row_base[:, None, None] + torch.arange(max(fleets, 1) * 5).view(
                1,
                max(fleets, 1),
                5,
            )[:, :fleets],
            "fleet_mask": torch.ones(rows, fleets, dtype=torch.bool),
            "fleet_target_planet_idx": torch.arange(fleets, dtype=torch.long).repeat(rows, 1),
            "planet_inbound_feats": (
                row_base[:, None, None] + torch.arange(planets * 3).view(1, planets, 3)
            ),
            "launch": row_base[:, None] + torch.arange(planets, dtype=torch.float32),
            "target_idx": torch.arange(planets, dtype=torch.long).repeat(rows, 1),
            "fraction": torch.full((rows, planets), 0.5) + row_base[:, None] * 0.01,
            "log_prob": torch.full((rows, planets), -0.25) - row_base[:, None] * 0.01,
            "value": row_base * 0.1,
            "owned_mask": torch.tensor(
                [[True, False, True], [True, True, False], [False, True, True]][:rows],
            ),
            "target_legal_mask": torch.ones(rows, planets, planets, dtype=torch.bool),
            "old_log_prob_computed": True,
        }

    chunks = [chunk(10, rows=2, fleets=0), chunk(20, rows=1, fleets=2)]
    layout = [[(chunks[0], 1), (chunks[1], 0)], [(chunks[0], 0)]]
    rewards = [[1.0, -0.5], [0.25]]

    def encoded_from(chunk_: dict[str, torch.Tensor | None], row: int) -> EncodedObs:
        return EncodedObs(
            planet_feats=chunk_["planet_feats"][row],
            planet_mask=chunk_["planet_mask"][row],
            planet_owned_mask=chunk_["planet_owned_mask"][row],
            planet_ids=chunk_["planet_ids"][row],
            planet_garrison=chunk_["planet_garrison"][row],
            fleet_feats=chunk_["fleet_feats"][row],
            fleet_mask=chunk_["fleet_mask"][row],
            global_feats=chunk_["global_feats"][row],
            fleet_target_planet_idx=chunk_["fleet_target_planet_idx"][row],
            planet_inbound_feats=chunk_["planet_inbound_feats"][row],
        )

    row_trajs = []
    chunk_trajs = []
    for rows_, rewards_ in zip(layout, rewards, strict=True):
        row_trajs.append(
            SimpleNamespace(
                encoded=[encoded_from(chunk_, row) for chunk_, row in rows_],
                launch=[chunk_["launch"][row] for chunk_, row in rows_],
                target_idx=[chunk_["target_idx"][row] for chunk_, row in rows_],
                fraction=[chunk_["fraction"][row] for chunk_, row in rows_],
                log_prob=[chunk_["log_prob"][row] for chunk_, row in rows_],
                value=[chunk_["value"][row] for chunk_, row in rows_],
                reward=list(rewards_),
                owned_mask=[chunk_["owned_mask"][row] for chunk_, row in rows_],
                target_legal_mask=[chunk_["target_legal_mask"][row] for chunk_, row in rows_],
            )
        )
        chunk_trajs.append(
            SimpleNamespace(
                encoded=[],
                launch=[],
                target_idx=[],
                fraction=[],
                log_prob=[],
                value=[],
                reward=list(rewards_),
                owned_mask=[],
                target_legal_mask=[],
                record_refs=[
                    TrajectoryRecordRef(chunk_, row) for chunk_, row in rows_
                ],
            )
        )

    row_batch = train_mod._stack_trajectories(
        row_trajs,
        gamma=0.9,
        gae_lambda=0.8,
        critic_mtp_horizon=3,
    )
    chunk_batch = train_mod._stack_trajectories(
        chunk_trajs,
        gamma=0.9,
        gae_lambda=0.8,
        critic_mtp_horizon=3,
    )

    assert row_batch.keys() == chunk_batch.keys()
    for key, expected in row_batch.items():
        got = chunk_batch[key]
        if expected is None:
            assert got is None
        elif expected.dtype.is_floating_point:
            torch.testing.assert_close(got, expected)
        else:
            assert torch.equal(got, expected), key

    deferred = train_mod._stack_trajectories(
        chunk_trajs,
        gamma=0.9,
        gae_lambda=0.8,
        include_old_log_prob=False,
    )
    assert torch.equal(deferred["old_log_prob"], torch.zeros_like(deferred["launch"]))

    fallback_chunks = [
        {**chunk_, "log_prob": None, "old_log_prob_computed": False}
        for chunk_ in chunks
    ]
    fallback_by_id = {
        id(chunk_): fallback_chunk
        for chunk_, fallback_chunk in zip(chunks, fallback_chunks, strict=True)
    }
    fallback_trajs = []
    for rows_, rewards_ in zip(layout, rewards, strict=True):
        fallback_trajs.append(
            SimpleNamespace(
                encoded=[],
                launch=[],
                target_idx=[],
                fraction=[],
                log_prob=[],
                value=[],
                reward=list(rewards_),
                owned_mask=[],
                target_legal_mask=[],
                record_refs=[
                    TrajectoryRecordRef(fallback_by_id[id(chunk_)], row)
                    for chunk_, row in rows_
                ],
            )
        )
    fallback = train_mod._stack_trajectories(
        fallback_trajs,
        gamma=0.9,
        gae_lambda=0.8,
        include_old_log_prob=True,
    )
    assert torch.equal(fallback["old_log_prob"], torch.zeros_like(fallback["launch"]))
    assert not bool(fallback["old_log_prob_computed"])


def test_stack_trajectories_preserves_compact_target_legality():
    planets = 3
    chunk = {
        "global_feats": torch.zeros(2, 2),
        "planet_feats": torch.zeros(2, planets, 4),
        "planet_mask": torch.ones(2, planets, dtype=torch.bool),
        "planet_owned_mask": torch.tensor(
            [[True, False, True], [False, True, True]],
        ),
        "planet_ids": torch.arange(planets, dtype=torch.long).repeat(2, 1),
        "planet_garrison": torch.zeros(2, planets),
        "fleet_feats": torch.zeros(2, 0, 5),
        "fleet_mask": torch.zeros(2, 0, dtype=torch.bool),
        "fleet_target_planet_idx": torch.zeros(2, 0, dtype=torch.long),
        "planet_inbound_feats": torch.zeros(2, planets, 3),
        "launch": torch.zeros(2, planets),
        "target_idx": torch.zeros(2, planets, dtype=torch.long),
        "fraction": torch.full((2, planets), 0.5),
        "log_prob": torch.zeros(2, planets),
        "value": torch.zeros(2),
        "target_legal_mask": None,
        "target_legal_row_idx": torch.tensor([0, 0, 1, 1]),
        "target_legal_source_idx": torch.tensor([0, 2, 1, 2]),
        "target_legal_source_mask": torch.tensor(
            [
                [False, True, True],
                [True, True, False],
                [True, False, True],
                [False, True, True],
            ],
        ),
        "old_log_prob_computed": True,
    }
    traj = SimpleNamespace(
        encoded=[],
        launch=[],
        target_idx=[],
        fraction=[],
        log_prob=[],
        value=[],
        reward=[0.0, 0.0],
        owned_mask=[],
        target_legal_mask=[],
        record_refs=[TrajectoryRecordRef(chunk, 1), TrajectoryRecordRef(chunk, 0)],
    )

    batch = train_mod._stack_trajectories(
        [traj],
        gamma=1.0,
        gae_lambda=1.0,
    )

    assert batch["target_legal_mask"] is None
    assert torch.equal(
        batch["target_legal_source_owned_mask"],
        torch.tensor([[False, True, True], [True, False, True]]),
    )
    assert torch.equal(batch["target_legal_row_offsets"], torch.tensor([0, 2, 4]))
    dense = _stage_target_legal_mask(
        batch,
        torch.tensor([0, 1]),
        torch.device("cpu"),
    )
    expected = torch.ones(2, planets, planets, dtype=torch.bool)
    expected[0, 1] = torch.tensor([True, False, True])
    expected[0, 2] = torch.tensor([False, True, True])
    expected[1, 0] = torch.tensor([False, True, True])
    expected[1, 2] = torch.tensor([True, True, False])
    assert torch.equal(dense, expected)


def test_compact_target_legality_keeps_owned_sources_without_rows_illegal():
    batch = {
        "target_legal_mask": None,
        "target_legal_source_owned_mask": torch.tensor([[True, True, False]]),
        "target_legal_row_idx": torch.tensor([0]),
        "target_legal_source_idx": torch.tensor([0]),
        "target_legal_source_mask": torch.tensor([[False, True, True]]),
    }

    dense = _stage_target_legal_mask(
        batch,
        torch.tensor([0]),
        torch.device("cpu"),
    )

    expected = torch.ones(1, 3, 3, dtype=torch.bool)
    expected[0, 0] = torch.tensor([False, True, True])
    expected[0, 1] = False
    assert torch.equal(dense, expected)


def test_stack_trajectories_preserves_source_actor_records():
    planets = 3
    chunk = {
        "global_feats": torch.zeros(2, 2),
        "planet_feats": torch.zeros(2, planets, 4),
        "planet_mask": torch.ones(2, planets, dtype=torch.bool),
        "planet_owned_mask": torch.tensor(
            [[True, False, True], [False, True, False]],
        ),
        "planet_ids": torch.arange(planets, dtype=torch.long).repeat(2, 1),
        "planet_garrison": torch.zeros(2, planets),
        "fleet_feats": torch.zeros(2, 0, 5),
        "fleet_mask": torch.zeros(2, 0, dtype=torch.bool),
        "fleet_target_planet_idx": torch.zeros(2, 0, dtype=torch.long),
        "planet_inbound_feats": torch.zeros(2, planets, 3),
        "launch": torch.zeros(2, planets),
        "target_idx": torch.zeros(2, planets, dtype=torch.long),
        "fraction": torch.full((2, planets), 0.5),
        "log_prob": torch.zeros(2, planets),
        "value": torch.zeros(2),
        "target_legal_mask": torch.ones(2, planets, planets, dtype=torch.bool),
        "old_log_prob_computed": True,
        "source_row_idx": torch.tensor([0, 0, 1]),
        "source_col_idx": torch.tensor([0, 2, 1]),
        "source_launch": torch.tensor([1.0, 0.0, 1.0]),
        "source_raw_launch": torch.tensor([1.0, 0.0, 1.0]),
        "source_target_idx": torch.tensor([2, 1, 0]),
        "source_fraction": torch.tensor([0.25, 0.5, 0.75]),
        "source_log_prob": torch.tensor([-0.1, -0.2, -0.3]),
        "source_target_legal_mask": torch.tensor(
            [
                [False, True, True],
                [True, False, True],
                [True, True, False],
            ],
        ),
    }
    traj = SimpleNamespace(
        encoded=[],
        launch=[],
        target_idx=[],
        fraction=[],
        log_prob=[],
        value=[],
        reward=[0.0, 0.0],
        owned_mask=[],
        target_legal_mask=[],
        record_refs=[TrajectoryRecordRef(chunk, 1), TrajectoryRecordRef(chunk, 0)],
    )

    batch = train_mod._stack_trajectories([traj], gamma=1.0, gae_lambda=1.0)

    assert torch.equal(batch["actor_source_row_idx"], torch.tensor([0, 1, 1]))
    assert torch.equal(batch["actor_source_col_idx"], torch.tensor([1, 0, 2]))
    assert torch.equal(batch["actor_source_row_offsets"], torch.tensor([0, 1, 3]))
    torch.testing.assert_close(batch["actor_launch"], torch.tensor([1.0, 1.0, 0.0]))
    torch.testing.assert_close(batch["actor_fraction"], torch.tensor([0.75, 0.25, 0.5]))
    torch.testing.assert_close(
        batch["actor_old_log_prob"],
        torch.tensor([-0.3, -0.1, -0.2]),
    )
    assert torch.equal(
        batch["actor_target_legal_mask"],
        torch.tensor(
            [
                [True, True, False],
                [False, True, True],
                [True, False, True],
            ],
        ),
    )


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
        model,
        optim,
        batch,
        value_coef=0.5,
        target_entropy_coef=0.01,
        fraction_entropy_coef=0.0,
        norm_advantage=True,
        advantage_transform="rankgauss",
        clip_coef=0.2,
        clip_coef_high=0.28,
        epochs=1,
        minibatch_size=2,
        minibatch_count=3,
        grad_clip=0.5,
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


def test_ppo_update_minibatch_count_above_batch_size_runs_real_steps_only():
    cfg = OrbitPolicyConfig(dim=32, ff_dim=64, depth=2, n_heads=2)
    model = OrbitPolicy(cfg)
    batch = _toy_batch(model, batch_size=3)
    optim = torch.optim.AdamW(model.parameters(), lr=0.0)

    log = ppo_update(
        model,
        optim,
        batch,
        value_coef=0.5,
        target_entropy_coef=0.01,
        fraction_entropy_coef=0.0,
        norm_advantage=True,
        advantage_transform="rankgauss",
        clip_coef=0.2,
        clip_coef_high=0.28,
        epochs=1,
        minibatch_size=2,
        minibatch_count=8,
        grad_clip=0.5,
    )

    assert math.isfinite(log.policy_loss)
    assert log.epochs_run == 1.0


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


def test_fixed_minibatches_by_count_caps_count_to_real_rows():
    torch.manual_seed(3)
    batches = _fixed_minibatches_by_count(3, 8, torch.device("cpu"))

    assert len(batches) == 3
    assert all(float(weight.sum()) > 0.0 for _mb, weight in batches)
    real_rows = []
    for mb, weight in batches:
        real_rows.extend(mb[weight.bool()].tolist())
    assert sorted(real_rows) == [0, 1, 2]


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
        planet_feats=batch["planet_feats"],
        planet_mask=batch["planet_mask"],
        planet_owned_mask=batch["planet_owned_mask"],
        planet_ids=batch["planet_ids"],
        planet_garrison=batch["planet_garrison"],
        fleet_feats=batch["fleet_feats"],
        fleet_mask=batch["fleet_mask"],
        global_feats=batch.get("global_feats"),
    )
    with torch.no_grad():
        out = model(feats)
    assert out.action_logit_softcap is not None

    p = out.target_logits.shape[1]
    target = batch["target_idx"].clamp(0, p - 1)
    target_logits = out.target_logits.masked_fill(~batch["target_legal_mask"], float("-inf"))
    action_log_probs = _categorical_action_log_probs(
        out.launch_logits,
        target_logits,
        out.action_logit_softcap,
    )
    action_idx = torch.where(launch_f > 0.5, target + 1, torch.zeros_like(target))
    action_lp = action_log_probs.gather(-1, action_idx.unsqueeze(-1)).squeeze(-1)
    frac_lp = _beta_log_prob(out.fraction_alpha, out.fraction_beta, batch["fraction"])
    chosen = action_lp + torch.where(
        launch_f > 0.5,
        frac_lp,
        torch.zeros_like(frac_lp),
    )

    diff = (chosen - batch["old_log_prob"]) * owned.float()
    # Should be exactly zero up to numerical noise — same params, same action.
    assert diff.abs().max().item() < 1e-4, diff.abs().max().item()


def test_ppo_update_rejects_deferred_placeholder_old_log_probs():
    torch.manual_seed(0)
    cfg = OrbitPolicyConfig(dim=32, ff_dim=64, depth=2, n_heads=2)
    model = OrbitPolicy(cfg)
    batch = _toy_batch(model, batch_size=4)
    batch["old_log_prob"] = torch.zeros_like(batch["old_log_prob"])
    batch["old_log_prob_computed"] = torch.tensor(False)
    optim = torch.optim.AdamW(model.parameters(), lr=3e-4)

    with pytest.raises(ValueError, match="requires real old_log_prob"):
        ppo_update(
            model,
            optim,
            batch,
            value_coef=0.5,
            target_entropy_coef=0.01,
            fraction_entropy_coef=0.0,
            norm_advantage=True,
            advantage_transform="rankgauss",
            clip_coef=0.2,
            clip_coef_high=0.28,
            epochs=1,
            minibatch_size=2,
            grad_clip=0.5,
        )


def test_ppo_update_sanitizes_masked_deferred_log_ratio():
    torch.manual_seed(0)
    cfg = OrbitPolicyConfig(dim=32, ff_dim=64, depth=2, n_heads=2)
    model = OrbitPolicy(cfg)
    batch = _toy_batch(model, batch_size=4)
    launched = (batch["owned_mask"] & (batch["launch"] > 0.5)).nonzero(as_tuple=False)
    assert launched.numel() > 0
    for row, source in launched.tolist():
        target = int(batch["target_idx"][row, source].clamp_min(0).item())
        batch["target_legal_mask"][row, source, target] = False
        batch["old_log_prob"][row, source] = float("-inf")
    batch["old_log_prob_computed"] = torch.tensor(True)
    optim = torch.optim.AdamW(model.parameters(), lr=3e-4)

    log = ppo_update(
        model,
        optim,
        batch,
        value_coef=0.5,
        target_entropy_coef=0.01,
        fraction_entropy_coef=0.0,
        norm_advantage=True,
        advantage_transform="rankgauss",
        clip_coef=0.2,
        clip_coef_high=0.28,
        epochs=1,
        minibatch_size=2,
        grad_clip=0.5,
    )

    assert math.isfinite(log.policy_loss)


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

    assert low_lp[0] > high_lp[0]  # confident no-launch when mean < 0
    assert low_lp[2] > high_lp[2]  # confident launch when mean > 0
    assert high_lp[1] > low_lp[1]  # high std explores against mean sign
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

    entropy = _conditional_action_entropy(launch_logits, target_log_probs, fraction_entropy)

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
    actor_loss = 3.0 * model.target_noop_key.weight.sum() + 4.0 * model.shared.weight.sum()
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
        self.new_target_logits = torch.log(torch.tensor([[[0.20, 0.80], [0.50, 0.50]]]))
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
            value_logits=(torch.zeros(b, 5, device=feats.planet_feats.device) + self.dummy * 0.0),
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
        "old_log_prob_computed": torch.tensor(True),
        "owned_mask": owned_mask,
        "advantage": torch.tensor([advantage]),
        "return": torch.zeros(1),
        "target_legal_mask": torch.ones(1, 2, 2, dtype=torch.bool),
    }


def test_clip_higher_clamps_ratio_above_upper_bound_with_positive_advantage():
    # ratio = 1.5 exceeds the looser upper bound 1 + clip_coef_high = 1.28, so
    # with a positive advantage the pessimistic max picks the clamped surrogate.
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
        model,
        optim,
        batch,
        value_coef=0.0,
        target_entropy_coef=0.0,
        fraction_entropy_coef=0.0,
        norm_advantage=False,
        advantage_transform="none",
        clip_coef=0.2,
        clip_coef_high=0.28,
        epochs=1,
        minibatch_size=1,
        grad_clip=1.0,
    )

    # max(-A*ratio, -A*clamp) = max(-1.5, -1.28) = -1.28 = -(1 + clip_coef_high).
    expected_loss = -(1.0 + 0.28)
    assert math.isclose(log.policy_loss, expected_loss, rel_tol=1e-6)
    assert math.isclose(log.ratio_clip_frac_high, 1.0, rel_tol=1e-6)
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
        model,
        optim,
        batch,
        value_coef=0.0,
        target_entropy_coef=0.0,
        fraction_entropy_coef=0.0,
        norm_advantage=False,
        advantage_transform="none",
        clip_coef=0.2,
        clip_coef_high=0.28,
        epochs=1,
        minibatch_size=1,
        grad_clip=1.0,
    )

    joint_ratio = 1.5 * 1.2
    expected_kl = (joint_ratio - 1.0) - math.log(joint_ratio)
    expected_per_planet_kl = (((1.5 - 1.0) - math.log(1.5)) + ((1.2 - 1.0) - math.log(1.2))) / 2.0
    assert math.isclose(log.approx_kl, expected_kl, rel_tol=1e-6)
    assert math.isclose(log.per_planet_approx_kl, expected_per_planet_kl, rel_tol=1e-6)
    assert not math.isclose(log.per_planet_approx_kl, log.approx_kl, rel_tol=1e-6)


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
        "old_log_prob_computed": torch.tensor(True),
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
        model,
        optim,
        batch,
        value_coef=0.0,
        target_entropy_coef=0.0,
        fraction_entropy_coef=0.0,
        norm_advantage=False,
        advantage_transform="none",
        clip_coef=0.2,
        clip_coef_high=0.28,
        epochs=1,
        minibatch_size=1,
        grad_clip=1.0,
    )

    latest_kl = (expected_ratio - 1.0) - math.log(expected_ratio)
    mean_kl = torch.mean((ratios - 1.0) - torch.log(ratios)).item()
    assert math.isclose(log.approx_kl, latest_kl, rel_tol=1e-6, abs_tol=1e-7)
    assert math.isclose(log.per_planet_approx_kl, latest_kl, rel_tol=1e-6, abs_tol=1e-7)
    assert not math.isclose(log.approx_kl, mean_kl, rel_tol=1e-6)
    assert not math.isclose(log.per_planet_approx_kl, mean_kl, rel_tol=1e-6)


def test_clip_clamps_ratio_below_lower_bound_with_negative_advantage():
    # ratio = 0.5 is below the lower bound 1 - clip_coef = 0.8. With a negative
    # advantage the pessimistic max picks the clamped surrogate (the lower bound
    # binds), and the looser-upper-bound counter stays zero.
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
        advantage=-1.0,
        old_log_prob=old_log_prob,
    )
    optim = torch.optim.AdamW(model.parameters(), lr=0.0)

    log = ppo_update(
        model,
        optim,
        batch,
        value_coef=0.0,
        target_entropy_coef=0.0,
        fraction_entropy_coef=0.0,
        norm_advantage=False,
        advantage_transform="none",
        clip_coef=0.2,
        clip_coef_high=0.28,
        epochs=1,
        minibatch_size=1,
        grad_clip=1.0,
    )

    # max(-A*ratio, -A*clamp) = max(0.5, 0.8) = 0.8 = (1 - clip_coef).
    expected_loss = 1.0 - 0.2
    assert math.isclose(log.policy_loss, expected_loss, rel_tol=1e-6)
    assert math.isclose(log.ratio_clip_frac, 1.0, rel_tol=1e-6)
    assert math.isclose(log.ratio_clip_frac_high, 0.0, abs_tol=1e-9)


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
        logits = model(
            EncodedObs(
                planet_feats=batch["planet_feats"],
                planet_mask=batch["planet_mask"],
                planet_owned_mask=batch["planet_owned_mask"],
                planet_ids=batch["planet_ids"],
                planet_garrison=batch["planet_garrison"],
                fleet_feats=batch["fleet_feats"],
                fleet_mask=batch["fleet_mask"],
            )
        ).value_logits
        target_probs = model.value_encoder.target_probs(batch["return"])
        expected = float(
            (-(target_probs * nn_functional.log_softmax(logits, dim=-1)).sum(dim=-1)).mean()
        )

    optim = torch.optim.AdamW(model.parameters(), lr=0.0)
    got = value_only_update(model, optim, batch, epochs=1, minibatch_size=5, grad_clip=1.0)

    assert math.isclose(got, expected, rel_tol=1e-6)


def test_value_only_update_normalizes_after_each_optimizer_step(monkeypatch):
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
    calls = []
    monkeypatch.setattr(
        "owars.training.ppo.normalize_matrices",
        lambda seen_model: calls.append(seen_model),
    )

    optim = torch.optim.AdamW(model.parameters(), lr=0.0)
    value_only_update(model, optim, batch, epochs=1, minibatch_size=2, grad_clip=1.0)

    assert calls == [model, model]


def test_pretrain_value_rounds_episode_batches_up_and_uses_behavior(monkeypatch):
    cfg = RunConfig.from_dict(
        {
            "ppo": {"pretrain_updates": 1, "pretrain_episodes": 129},
            "rollout": {"num_envs": 128},
        }
    )
    behavior_seen = []

    def behavior(_obs):
        return []

    monkeypatch.setitem(train_mod.BUILTIN, "unit_behavior", behavior)
    cfg.ppo.pretrain_behavior = "unit_behavior"

    def fake_rollout(*_args, **kwargs):
        behavior_seen.append(kwargs.get("learner_action_agent"))
        return [SimpleNamespace(reward=[0.0], value=[torch.tensor(0.0)])]

    monkeypatch.setattr(train_mod, "rollout_episodes_batched", fake_rollout)
    monkeypatch.setattr(
        train_mod,
        "_pretrain_value_batch",
        lambda *_args, **_kwargs: {
            "planet_feats": torch.zeros(1, 1, 19),
            "planet_mask": torch.ones(1, 1, dtype=torch.bool),
            "planet_owned_mask": torch.ones(1, 1, dtype=torch.bool),
            "planet_ids": torch.zeros(1, 1, dtype=torch.long),
            "planet_garrison": torch.ones(1, 1),
            "fleet_feats": torch.zeros(1, 1, 20),
            "fleet_mask": torch.zeros(1, 1, dtype=torch.bool),
            "return": torch.zeros(1),
        },
    )
    monkeypatch.setattr(train_mod, "value_only_update", lambda *_args, **_kwargs: 0.0)
    monkeypatch.setattr(
        train_mod,
        "_slice_encoded_obs_to_device",
        lambda batch, _mb, _device: SimpleNamespace(planet_feats=batch["planet_feats"]),
    )

    class Logger:
        def scalars(self, *_args, **_kwargs):
            return None

    class PretrainModel:
        def __call__(self, feats):
            return SimpleNamespace(value=torch.zeros(feats.planet_feats.shape[0]))

    train_mod.pretrain_value(
        cfg,
        PretrainModel(),
        torch.optim.AdamW([torch.nn.Parameter(torch.zeros(()))]),
        Logger(),
        torch.device("cpu"),
        SimpleNamespace(),
    )

    assert behavior_seen == [behavior, behavior]


def test_ppo_update_runs_all_configured_epochs():
    cfg = OrbitPolicyConfig(dim=32, ff_dim=64, depth=2, n_heads=2)
    model = OrbitPolicy(cfg)
    batch = _toy_batch(model, batch_size=8)
    optim = torch.optim.AdamW(model.parameters(), lr=3e-4)
    log = ppo_update(
        model,
        optim,
        batch,
        value_coef=0.5,
        target_entropy_coef=0.01,
        fraction_entropy_coef=0.0,
        norm_advantage=True,
        advantage_transform="rankgauss",
        clip_coef=0.2,
        clip_coef_high=0.28,
        epochs=3,
        minibatch_size=4,
        grad_clip=0.5,
    )
    assert log.epochs_run == 3.0


def test_ppo_update_runs_all_epochs_when_kl_is_large():
    cfg = OrbitPolicyConfig(dim=32, ff_dim=64, depth=2, n_heads=2)
    model = OrbitPolicy(cfg)
    batch = _toy_batch(model, batch_size=8)
    batch["old_log_prob"] = batch["old_log_prob"] - 5.0
    optim = torch.optim.AdamW(model.parameters(), lr=0.0)

    log = ppo_update(
        model,
        optim,
        batch,
        value_coef=0.5,
        target_entropy_coef=0.01,
        fraction_entropy_coef=0.0,
        norm_advantage=True,
        advantage_transform="rankgauss",
        clip_coef=0.2,
        clip_coef_high=0.28,
        epochs=3,
        minibatch_size=4,
        grad_clip=0.5,
    )

    assert log.approx_kl > 1.0
    assert log.epochs_run == 3.0
