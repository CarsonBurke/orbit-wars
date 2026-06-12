"""Unit tests for the SAC replay buffer + async minibatch prefetcher.

These exercise the CPU path only (no GPU in CI): `sample` gathers the right
shapes/dtypes and `_ReplayPrefetcher` degrades to a plain synchronous sample
that still yields independent uniform-random minibatches. The CUDA side-stream
overlap path can't be validated here and is covered by manual GPU testing.
"""

from __future__ import annotations

import torch

from owars.policies.config import OrbitPolicyConfig
from owars.policies.features import MAX_PLANETS, EncodedObs
from owars.policies.sac_model import SACActor
from owars.policies.sac_sampling import get_sac_heads_kernel, run_sac_heads
from owars.training.config import RunConfig
from owars.training.sac import (
    ReplayBuffer,
    _ReplayPrefetcher,
    _builtin_opponent_slate,
)

PLANET_FEATURES = 19
FLEET_FEATURES = 20


def _toy_state(
    seed: int,
    fleet_count: int = 3,
    *,
    include_fleet_targets: bool = False,
) -> EncodedObs:
    g = torch.Generator().manual_seed(seed)
    fleet_target_planet_idx = None
    if include_fleet_targets:
        fleet_target_planet_idx = torch.full((fleet_count,), -1, dtype=torch.int64)
        if fleet_count:
            fleet_target_planet_idx[0] = seed % MAX_PLANETS
    return EncodedObs(
        planet_feats=torch.randn(MAX_PLANETS, PLANET_FEATURES, generator=g),
        planet_mask=torch.ones(MAX_PLANETS, dtype=torch.bool),
        planet_owned_mask=torch.zeros(MAX_PLANETS, dtype=torch.bool),
        planet_ids=torch.arange(MAX_PLANETS, dtype=torch.int64),
        planet_garrison=torch.rand(MAX_PLANETS, generator=g),
        fleet_feats=torch.randn(fleet_count, FLEET_FEATURES, generator=g),
        fleet_mask=torch.zeros(fleet_count, dtype=torch.bool),
        fleet_target_planet_idx=fleet_target_planet_idx,
    )


def _fill(buf: ReplayBuffer, n: int) -> None:
    for i in range(n):
        buf.add(
            feats=_toy_state(i, fleet_count=3 + (i % 5)),
            launch=torch.zeros(MAX_PLANETS),
            target_idx=torch.zeros(MAX_PLANETS, dtype=torch.int64),
            fraction=torch.rand(MAX_PLANETS),
            target_legal_mask=torch.zeros(
                MAX_PLANETS, MAX_PLANETS, dtype=torch.bool
            ),
            next_target_legal_mask=torch.zeros(
                MAX_PLANETS, MAX_PLANETS, dtype=torch.bool
            ),
            reward=float(i),
            done=bool(i % 7 == 0),
            next_feats=_toy_state(i + 1000, fleet_count=4 + (i % 5)),
            time=float(i) / 500.0,
            next_time=float(i + 1) / 500.0,
        )


def _new_buffer(capacity: int = 64) -> ReplayBuffer:
    return ReplayBuffer(
        capacity,
        planet_features=PLANET_FEATURES,
        fleet_features=FLEET_FEATURES,
        device="cpu",
    )


def test_sample_shapes_and_dtypes() -> None:
    buf = _new_buffer()
    _fill(buf, 64)
    batch = buf.sample(8, device="cpu")

    assert batch.feats.planet_feats.shape == (8, MAX_PLANETS, PLANET_FEATURES)
    assert batch.next_feats.fleet_feats.shape == (8, 64, FLEET_FEATURES)
    assert batch.target_legal_mask.shape == (8, MAX_PLANETS, MAX_PLANETS)
    assert batch.reward.shape == (8,)
    assert batch.feats.planet_ids.dtype == torch.int64
    assert batch.feats.planet_mask.dtype == torch.bool
    assert batch.reward.dtype == torch.float32
    assert torch.isfinite(batch.feats.planet_feats).all()
    assert batch.feats.fleet_target_planet_idx is None


def test_sample_preserves_dynamic_fleet_targets() -> None:
    buf = _new_buffer()
    buf.add(
        feats=_toy_state(1, fleet_count=5, include_fleet_targets=True),
        launch=torch.zeros(MAX_PLANETS),
        target_idx=torch.zeros(MAX_PLANETS, dtype=torch.int64),
        fraction=torch.rand(MAX_PLANETS),
        target_legal_mask=torch.zeros(MAX_PLANETS, MAX_PLANETS, dtype=torch.bool),
        next_target_legal_mask=torch.zeros(MAX_PLANETS, MAX_PLANETS, dtype=torch.bool),
        reward=1.0,
        done=False,
        next_feats=_toy_state(2, fleet_count=9, include_fleet_targets=True),
        time=0.1,
        next_time=0.2,
    )

    batch = buf.sample(1, device="cpu")

    assert batch.feats.fleet_feats.shape[0] == 1
    assert batch.feats.fleet_feats.shape[1] >= 9
    assert batch.feats.fleet_feats.shape[2] == FLEET_FEATURES
    assert batch.feats.fleet_target_planet_idx is not None
    assert batch.next_feats.fleet_target_planet_idx is not None
    assert int(batch.feats.fleet_target_planet_idx[0, 0]) == 1
    assert batch.feats.fleet_target_planet_idx[0, 5:].eq(-1).all()
    assert int(batch.next_feats.fleet_target_planet_idx[0, 0]) == 2


def test_sample_rejects_oversized_batch() -> None:
    buf = _new_buffer()
    _fill(buf, 4)
    try:
        buf.sample(8, device="cpu")
    except ValueError:
        pass
    else:  # pragma: no cover - guard
        raise AssertionError("expected ValueError for batch_size > size")


def test_non_blocking_is_noop_on_cpu() -> None:
    # On CPU the pinned/async path is skipped; the result must equal the plain
    # gather for the same indices (we can't fix indices, so just check it runs
    # and produces CPU tensors of the right shape).
    buf = _new_buffer()
    _fill(buf, 32)
    batch = buf.sample(8, device="cpu", non_blocking=True)
    assert batch.feats.planet_feats.device.type == "cpu"
    assert batch.feats.planet_feats.shape == (8, MAX_PLANETS, PLANET_FEATURES)


def test_prefetcher_cpu_fallback_yields_independent_batches() -> None:
    buf = _new_buffer()
    _fill(buf, 64)
    pf = _ReplayPrefetcher(buf, batch_size=8, device=torch.device("cpu"))
    # CPU device → no side stream, synchronous passthrough.
    assert pf._stream is None

    torch.manual_seed(0)
    b1 = pf.next()
    b2 = pf.next()
    assert b1.reward.shape == (8,)
    assert b2.reward.shape == (8,)
    # Two independent uniform draws should (almost surely) differ.
    assert not torch.equal(b1.reward, b2.reward)


def test_sac_destination_backend_heads_accept_fleet_targets() -> None:
    cfg = OrbitPolicyConfig(
        encoder_backend="destination_conditioned",
        dim=32,
        ff_dim=64,
        depth=1,
        n_heads=4,
        n_kv_heads=1,
    )
    actor = SACActor(cfg)
    state = _toy_state(3, fleet_count=4, include_fleet_targets=True)
    state.fleet_mask[:] = True

    feats = EncodedObs(
        planet_feats=state.planet_feats.unsqueeze(0),
        planet_mask=state.planet_mask.unsqueeze(0),
        planet_owned_mask=state.planet_owned_mask.unsqueeze(0),
        planet_ids=state.planet_ids.unsqueeze(0),
        planet_garrison=state.planet_garrison.unsqueeze(0),
        fleet_feats=state.fleet_feats.unsqueeze(0),
        fleet_mask=state.fleet_mask.unsqueeze(0),
        fleet_target_planet_idx=state.fleet_target_planet_idx.unsqueeze(0),
    )
    kernel = get_sac_heads_kernel(actor, device=torch.device("cpu"), compile_mode=None)
    out = run_sac_heads(
        kernel,
        feats,
        device=torch.device("cpu"),
        time_feat=torch.zeros(1),
    )

    assert out.launch_logits.shape == (1, MAX_PLANETS)
    assert out.target_logits.shape == (1, MAX_PLANETS, MAX_PLANETS)


def test_fixed_mode_overrides_sac_builtin_slate() -> None:
    cfg = RunConfig.from_dict(
        {
            "opponents": {
                "mode": "fixed",
                "fixed_opponents": ["sniper"],
            },
            "sac": {
                "builtin_opponents": ["random", "heuristic"],
                "builtin_prob": 0.25,
            },
        }
    )

    names, prob = _builtin_opponent_slate(cfg)
    assert names == ["sniper"]
    assert prob == 1.0
