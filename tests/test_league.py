"""OpponentPool: top-K-by-Elo eviction, 80/20 sampling, empty-pool fallback."""

from __future__ import annotations

import random
from pathlib import Path

import torch

from owars.policies.config import OrbitPolicyConfig
from owars.policies.model import OrbitPolicy
from owars.training.elo import EloTracker
from owars.training.league import LEARNER_NAME, OpponentPool


def _tiny_model() -> OrbitPolicy:
    return OrbitPolicy(OrbitPolicyConfig(dim=16, ff_dim=32, depth=1, n_heads=2))


def test_empty_pool_samples_only_self():
    """No snapshots in the pool → every slot must be the learner."""
    elo = EloTracker()
    pool = OpponentPool(elo=elo, top_k=8, self_play_prob=0.0, rng=random.Random(0))
    model = _tiny_model()
    slots = pool.sample(8)
    assert all(s.name == LEARNER_NAME for s in slots)


def test_self_play_prob_split(tmp_path: Path):
    """With one snapshot in the pool and self_play_prob=0.5, the slot label
    distribution should be roughly half-and-half."""
    elo = EloTracker()
    pool = OpponentPool(elo=elo, top_k=4, self_play_prob=0.5, rng=random.Random(0))
    model = _tiny_model()
    pool.add_snapshot("a", model, tmp_path / "a.pt")

    n = 2000
    slots = pool.sample(n)
    self_count = sum(1 for s in slots if s.name == LEARNER_NAME)
    # Tolerance: ±5%.
    assert 0.45 * n <= self_count <= 0.55 * n


def test_top_k_eviction_keeps_highest_elo(tmp_path: Path):
    """When the pool fills, the *lowest-Elo* snapshot must be evicted."""
    elo = EloTracker(initial_rating=1500.0)
    pool = OpponentPool(elo=elo, top_k=2, self_play_prob=0.0, rng=random.Random(0))
    model = _tiny_model()

    # Insert three snapshots, manually setting their seed Elo so we know
    # which one should die.
    pool.add_snapshot("low", model, tmp_path / "low.pt", seed_rating=1400.0)
    pool.add_snapshot("mid", model, tmp_path / "mid.pt", seed_rating=1500.0)
    pool.add_snapshot("high", model, tmp_path / "high.pt", seed_rating=1700.0)

    alive = set(pool.snapshot_names())
    assert alive == {"frozen:high", "frozen:mid"}
    assert "frozen:low" not in alive


def test_sampling_only_picks_alive_snapshots(tmp_path: Path):
    """After eviction, sampled snapshot names must be from the live set."""
    elo = EloTracker()
    pool = OpponentPool(elo=elo, top_k=2, self_play_prob=0.0, rng=random.Random(0))
    model = _tiny_model()
    pool.add_snapshot("low", model, tmp_path / "low.pt", seed_rating=1300.0)
    pool.add_snapshot("hi1", model, tmp_path / "hi1.pt", seed_rating=1700.0)
    pool.add_snapshot("hi2", model, tmp_path / "hi2.pt", seed_rating=1700.0)

    alive = set(pool.snapshot_names())
    slots = pool.sample(50)
    for s in slots:
        if s.name == LEARNER_NAME:
            continue
        assert s.name in alive


def test_eviction_unlinks_checkpoint_file(tmp_path: Path):
    """Evicted snapshots must delete their on-disk checkpoint, otherwise
    runs with `snapshot_every: 1` accumulate ~1 file per update forever."""
    elo = EloTracker(initial_rating=1500.0)
    pool = OpponentPool(elo=elo, top_k=2, self_play_prob=0.0, rng=random.Random(0))
    model = _tiny_model()

    low = tmp_path / "low.pt"
    mid = tmp_path / "mid.pt"
    high = tmp_path / "high.pt"
    pool.add_snapshot("low", model, low, seed_rating=1400.0)
    pool.add_snapshot("mid", model, mid, seed_rating=1500.0)
    pool.add_snapshot("high", model, high, seed_rating=1700.0)

    assert not low.exists()
    assert mid.exists()
    assert high.exists()


def test_new_snapshot_inherits_learner_rating(tmp_path: Path):
    elo = EloTracker(initial_rating=1500.0)
    elo.set(LEARNER_NAME, 1700.0)
    pool = OpponentPool(elo=elo, top_k=4, self_play_prob=0.0, rng=random.Random(0))
    pool.add_snapshot("a", _tiny_model(), tmp_path / "a.pt")
    assert elo.get("frozen:a") == 1700.0
