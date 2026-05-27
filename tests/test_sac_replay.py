"""Unit tests for the SAC replay buffer + async minibatch prefetcher.

These exercise the CPU path only (no GPU in CI): `sample` gathers the right
shapes/dtypes and `_ReplayPrefetcher` degrades to a plain synchronous sample
that still yields independent uniform-random minibatches. The CUDA side-stream
overlap path can't be validated here and is covered by manual GPU testing.
"""

from __future__ import annotations

import torch

from owars.policies.features import MAX_FLEETS, MAX_PLANETS, EncodedObs
from owars.training.sac import ReplayBuffer, _ReplayPrefetcher

PLANET_FEATURES = 19
FLEET_FEATURES = 20


def _toy_state(seed: int) -> EncodedObs:
    g = torch.Generator().manual_seed(seed)
    return EncodedObs(
        planet_feats=torch.randn(MAX_PLANETS, PLANET_FEATURES, generator=g),
        planet_mask=torch.ones(MAX_PLANETS, dtype=torch.bool),
        planet_owned_mask=torch.zeros(MAX_PLANETS, dtype=torch.bool),
        planet_ids=torch.arange(MAX_PLANETS, dtype=torch.int64),
        planet_garrison=torch.rand(MAX_PLANETS, generator=g),
        fleet_feats=torch.randn(MAX_FLEETS, FLEET_FEATURES, generator=g),
        fleet_mask=torch.zeros(MAX_FLEETS, dtype=torch.bool),
    )


def _fill(buf: ReplayBuffer, n: int) -> None:
    for i in range(n):
        buf.add(
            feats=_toy_state(i),
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
            next_feats=_toy_state(i + 1000),
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
    assert batch.next_feats.fleet_feats.shape == (8, MAX_FLEETS, FLEET_FEATURES)
    assert batch.target_legal_mask.shape == (8, MAX_PLANETS, MAX_PLANETS)
    assert batch.reward.shape == (8,)
    assert batch.feats.planet_ids.dtype == torch.int64
    assert batch.feats.planet_mask.dtype == torch.bool
    assert batch.reward.dtype == torch.float32
    assert torch.isfinite(batch.feats.planet_feats).all()


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
