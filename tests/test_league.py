"""OpponentPool: top-K-by-Elo eviction, 80/20 sampling, empty-pool fallback."""

from __future__ import annotations

import random
from pathlib import Path

import torch

from owars.agents.sniper import sniper_v18_agent
from owars.policies.config import OrbitPolicyConfig
from owars.policies.model import OrbitPolicy, normalize_matrices
from owars.training.elo import EloTracker
from owars.training.league import (
    LEARNER_NAME,
    FixedOpponentPool,
    NoBuiltinTrainingPool,
    OpponentPool,
    _copy_model_without_compile_caches,
)


def _tiny_model() -> OrbitPolicy:
    return OrbitPolicy(OrbitPolicyConfig(dim=16, ff_dim=32, depth=1, n_heads=2))


def test_snapshot_copy_strips_normalization_target_cache():
    model = _tiny_model()
    normalize_matrices(model)

    snap = _copy_model_without_compile_caches(model)

    assert "_owars_normalize_targets" not in snap.__dict__

    normalize_matrices(snap)
    snap_modules = set(snap.modules())
    snap_targets = snap.__dict__["_owars_normalize_targets"]

    assert snap_targets
    assert all(module in snap_modules for module in snap_targets)
    assert all(module not in model.modules() for module in snap_targets)


def test_empty_pool_samples_only_self():
    """No snapshots in the pool → every slot must be the learner."""
    elo = EloTracker()
    pool = OpponentPool(elo=elo, top_k=8, self_play_prob=0.0, rng=random.Random(0))
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


def test_opponent_pool_snapshot_persists_checkpoint_extra(tmp_path: Path):
    elo = EloTracker()
    pool = OpponentPool(elo=elo, top_k=4, self_play_prob=0.0, rng=random.Random(0))
    path = tmp_path / "a.pt"

    pool.add_snapshot(
        "a",
        _tiny_model(),
        path,
        checkpoint_extra={"critic_return_normalizer": {"mean": 1.25}},
    )

    payload = torch.load(path, map_location="cpu", weights_only=False)
    assert payload["critic_return_normalizer"] == {"mean": 1.25}


def test_no_builtin_snapshot_persists_checkpoint_extra(tmp_path: Path):
    pool = NoBuiltinTrainingPool(rng=random.Random(0))
    path = tmp_path / "a.pt"

    pool.add_snapshot(
        "a",
        _tiny_model(),
        path,
        checkpoint_extra={"critic_return_normalizer": {"mean": 1.25}},
    )

    payload = torch.load(path, map_location="cpu", weights_only=False)
    assert payload["critic_return_normalizer"] == {"mean": 1.25}


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


def test_fixed_opponent_pool_samples_static_builtin():
    # "sniper" is the default and resolves to sniper_v18_agent.
    pool = FixedOpponentPool(["sniper"], rng=random.Random(0))
    slots = pool.sample(8)
    assert all(s.name == "sniper" for s in slots)
    assert all(s.agent is sniper_v18_agent for s in slots)
    assert pool.snapshot_names() == []


def test_fixed_opponent_pool_rejects_unknown_builtin():
    try:
        FixedOpponentPool(["not_real"], rng=random.Random(0))
    except ValueError as exc:
        assert "unknown fixed opponents" in str(exc)
    else:
        raise AssertionError("unknown fixed opponent should raise")


def test_no_builtin_pool_empty_samples_current_learner():
    pool = NoBuiltinTrainingPool(
        current_learner_prob=0.0,
        active_pool_prob=0.5,
        historical_archive_prob=0.5,
        rng=random.Random(0),
    )

    slots = pool.sample(16)

    assert all(slot.name == LEARNER_NAME for slot in slots)
    assert all(slot.agent is None for slot in slots)


def test_no_builtin_pool_redistributes_when_historical_empty(tmp_path: Path):
    pool = NoBuiltinTrainingPool(
        active_pool_size=4,
        current_learner_prob=0.4,
        active_pool_prob=0.3,
        historical_archive_prob=0.3,
        rng=random.Random(0),
    )
    pool.add_snapshot("a", _tiny_model(), tmp_path / "a.pt", created_update=0)

    slots = pool.sample(4000)
    current = sum(1 for slot in slots if slot.name == LEARNER_NAME)
    active = sum(1 for slot in slots if slot.name == "frozen:a")

    # Historical is empty, so 40/30/30 normalizes to 40/30 over current+active.
    assert 0.54 * len(slots) <= current <= 0.60 * len(slots)
    assert 0.40 * len(slots) <= active <= 0.46 * len(slots)


def test_no_builtin_pool_samples_40_30_30_when_all_sources_exist(tmp_path: Path):
    pool = NoBuiltinTrainingPool(
        active_pool_size=1,
        historical_training_archive_size=4,
        min_games_before_eviction=0,
        current_learner_prob=0.4,
        active_pool_prob=0.3,
        historical_archive_prob=0.3,
        rng=random.Random(1),
    )
    model = _tiny_model()
    pool.add_snapshot("hist", model, tmp_path / "hist.pt", created_update=0)
    pool.set_current_update(1)
    pool.add_snapshot("active", model, tmp_path / "active.pt", created_update=1)

    assert pool.active_snapshot_names() == ["frozen:active"]
    assert pool.historical_snapshot_names()

    slots = pool.sample(6000)
    current = sum(1 for slot in slots if slot.name == LEARNER_NAME)
    active = sum(1 for slot in slots if slot.name in pool.active_snapshot_names())
    historical = sum(1 for slot in slots if slot.name in pool.historical_snapshot_names())

    assert 0.37 * len(slots) <= current <= 0.43 * len(slots)
    assert 0.27 * len(slots) <= active <= 0.33 * len(slots)
    assert 0.27 * len(slots) <= historical <= 0.33 * len(slots)


def test_no_builtin_pool_updates_active_stats_only_vs_current(tmp_path: Path):
    pool = NoBuiltinTrainingPool(rng=random.Random(0))
    name = pool.add_snapshot("a", _tiny_model(), tmp_path / "a.pt")

    pool.record_game([(name, 10.0), ("frozen:other", 1.0)])
    assert pool.snapshot_stats(name).games_vs_current == 0

    pool.record_game([(LEARNER_NAME, 5.0), (name, 7.0)])
    stats = pool.snapshot_stats(name)
    assert stats.games_vs_current == 1
    assert stats.wins_vs_current == 1
    assert stats.mean_margin_vs_current == 2.0

    pool.record_game([(LEARNER_NAME, 9.0), ("frozen:not_active", 20.0)])
    assert stats.games_vs_current == 1


def test_no_builtin_active_retention_uses_utility_not_top_k_elo(tmp_path: Path):
    pool = NoBuiltinTrainingPool(
        active_pool_size=2,
        min_games_before_eviction=0,
        recency_half_life_updates=1000.0,
        rng=random.Random(0),
    )
    model = _tiny_model()
    near = pool.add_snapshot("near_50", model, tmp_path / "near.pt", created_update=0)
    weak = pool.add_snapshot("weak", model, tmp_path / "weak.pt", created_update=0)

    for _ in range(20):
        pool.record_result_vs_current(near, learner_score=10.0, snapshot_score=10.0)
        pool.record_result_vs_current(weak, learner_score=20.0, snapshot_score=1.0)

    pool.add_snapshot("new", model, tmp_path / "new.pt", created_update=0)

    alive = set(pool.active_snapshot_names())
    assert near in alive
    assert "frozen:new" in alive
    assert weak not in alive


def test_no_builtin_active_min_exposure_evicts_oldest_underexposed(tmp_path: Path):
    pool = NoBuiltinTrainingPool(
        active_pool_size=2,
        min_games_before_eviction=16,
        rng=random.Random(0),
    )
    model = _tiny_model()
    pool.set_current_update(0)
    oldest = pool.add_snapshot("oldest", model, tmp_path / "oldest.pt")
    pool.set_current_update(1)
    pool.add_snapshot("middle", model, tmp_path / "middle.pt")
    pool.set_current_update(2)
    pool.add_snapshot("newest", model, tmp_path / "newest.pt")

    assert oldest not in set(pool.active_snapshot_names())
    assert set(pool.active_snapshot_names()) == {"frozen:middle", "frozen:newest"}


def test_no_builtin_historical_archive_uses_log_eviction_and_notable_buckets(
    tmp_path: Path,
):
    pool = NoBuiltinTrainingPool(
        active_pool_size=1,
        historical_training_archive_size=8,
        recent_eviction_archive_size=2,
        notable_archive_size=2,
        min_games_before_eviction=0,
        rng=random.Random(0),
    )
    model = _tiny_model()
    pool.set_current_update(0)
    first = pool.add_snapshot("u0", model, tmp_path / "u0.pt", notable=True)
    pool.set_current_update(1)
    pool.add_snapshot("u1", model, tmp_path / "u1.pt")
    pool.set_current_update(2)
    pool.add_snapshot("u2", model, tmp_path / "u2.pt")
    pool.set_current_update(4)
    pool.add_snapshot("u4", model, tmp_path / "u4.pt")
    pool.set_current_update(8)
    pool.rebuild_historical_archive()

    assert first in pool.historical_snapshot_names("notable")
    assert pool.historical_snapshot_names("recent_eviction")
    assert pool.historical_snapshot_names("log")
    assert set(pool.historical_snapshot_names()).issuperset(
        pool.historical_snapshot_names("recent_eviction")
    )


def test_no_builtin_log_archive_retains_log_spaced_candidates(tmp_path: Path):
    pool = NoBuiltinTrainingPool(
        active_pool_size=1,
        historical_training_archive_size=64,
        recent_eviction_archive_size=0,
        notable_archive_size=0,
        min_games_before_eviction=0,
        rng=random.Random(0),
    )
    model = _tiny_model()
    for update in range(65):
        pool.set_current_update(update)
        pool.add_snapshot(f"u{update}", model, tmp_path / f"u{update}.pt")

    historical = pool.historical_snapshot_names("log")
    ages = sorted(
        pool.current_update - pool.snapshot_stats(name).created_update for name in historical
    )

    assert len(historical) <= 7
    assert ages == [1, 2, 4, 8, 16, 32, 64]


def test_no_builtin_historical_archive_size_is_logarithmic(tmp_path: Path):
    pool = NoBuiltinTrainingPool(
        active_pool_size=1,
        historical_training_archive_size=128,
        min_games_before_eviction=0,
        rng=random.Random(0),
    )
    model = _tiny_model()
    for update in range(80):
        pool.set_current_update(update)
        pool.add_snapshot(f"u{update}", model, tmp_path / f"u{update}.pt")

    historical = pool.historical_snapshot_names()

    assert len(historical) <= 16


def test_no_builtin_recent_eviction_bucket_keeps_newest_representative(
    tmp_path: Path,
):
    pool = NoBuiltinTrainingPool(
        active_pool_size=1,
        historical_training_archive_size=16,
        recent_eviction_archive_size=3,
        notable_archive_size=0,
        min_games_before_eviction=0,
        rng=random.Random(0),
    )
    model = _tiny_model()
    for update in range(8):
        pool.set_current_update(update)
        pool.add_snapshot(f"u{update}", model, tmp_path / f"u{update}.pt")

    recent = pool.historical_snapshot_names("recent_eviction")
    evicted_updates = [pool.snapshot_stats(name).evicted_update for name in recent]

    assert max(update for update in evicted_updates if update is not None) == 7
    assert len(recent) <= 3


def test_no_builtin_recent_eviction_age_handles_update_zero(tmp_path: Path):
    pool = NoBuiltinTrainingPool(
        active_pool_size=1,
        historical_training_archive_size=16,
        recent_eviction_archive_size=16,
        notable_archive_size=0,
        min_games_before_eviction=0,
        rng=random.Random(0),
    )
    model = _tiny_model()
    pool.set_current_update(0)
    pool.add_snapshot("u0a", model, tmp_path / "u0a.pt")
    pool.add_snapshot("u0b", model, tmp_path / "u0b.pt")
    evicted_at_zero = [
        name for name in pool.all_snapshot_names() if pool.snapshot_stats(name).evicted_update == 0
    ]
    pool.set_current_update(8)
    pool.rebuild_historical_archive()

    assert len(evicted_at_zero) == 1
    assert evicted_at_zero[0] in pool.historical_snapshot_names("recent_eviction")


def test_no_builtin_evicted_historical_snapshot_uses_lazy_agent(tmp_path: Path):
    pool = NoBuiltinTrainingPool(
        active_pool_size=1,
        historical_training_archive_size=8,
        recent_eviction_archive_size=0,
        notable_archive_size=0,
        min_games_before_eviction=0,
        current_learner_prob=0.0,
        active_pool_prob=0.0,
        historical_archive_prob=1.0,
        rng=random.Random(0),
    )
    model = _tiny_model()
    pool.add_snapshot("u0", model, tmp_path / "u0.pt")
    pool.set_current_update(1)
    pool.add_snapshot("u1", model, tmp_path / "u1.pt")

    historical = pool.historical_snapshot_names("log")
    assert historical == ["frozen:u0"]

    slot = pool.sample(1)[0]
    assert type(slot.agent).__name__ == "LazyLearnedAgent"
    assert slot.agent.model is not None


def test_no_builtin_lazy_snapshot_retains_compile_mode(tmp_path: Path):
    pool = NoBuiltinTrainingPool(
        active_pool_size=1,
        historical_training_archive_size=8,
        recent_eviction_archive_size=0,
        notable_archive_size=0,
        min_games_before_eviction=0,
        current_learner_prob=0.0,
        active_pool_prob=0.0,
        historical_archive_prob=1.0,
        compile_mode="reduce-overhead",
        rng=random.Random(0),
    )
    model = _tiny_model()
    pool.add_snapshot("u0", model, tmp_path / "u0.pt")
    pool.set_current_update(1)
    pool.add_snapshot("u1", model, tmp_path / "u1.pt")

    slot = pool.sample(1)[0]

    assert type(slot.agent).__name__ == "LazyLearnedAgent"
    assert slot.agent.compile_mode == "reduce-overhead"


def test_no_builtin_historical_sampling_uses_per_update_panel(tmp_path: Path):
    pool = NoBuiltinTrainingPool(
        active_pool_size=1,
        historical_training_archive_size=16,
        historical_sample_panel_size=2,
        recent_eviction_archive_size=0,
        notable_archive_size=0,
        min_games_before_eviction=0,
        current_learner_prob=0.0,
        active_pool_prob=0.0,
        historical_archive_prob=1.0,
        rng=random.Random(2),
    )
    model = _tiny_model()
    for update in range(8):
        pool.set_current_update(update)
        pool.add_snapshot(f"u{update}", model, tmp_path / f"u{update}.pt")

    slots = pool.sample(200, current_update=8)
    assert len({slot.name for slot in slots}) <= 2

    pool.set_current_update(9)
    slots_after_update = pool.sample(200, current_update=9)
    assert len({slot.name for slot in slots_after_update}) <= 2
    assert pool.current_update == 9


def test_no_builtin_panel_caps_active_snapshot_identities(tmp_path: Path):
    pool = NoBuiltinTrainingPool(
        active_pool_size=8,
        active_sample_panel_size=2,
        current_learner_prob=0.0,
        active_pool_prob=1.0,
        historical_archive_prob=0.0,
        rng=random.Random(7),
    )
    model = _tiny_model()
    for idx in range(6):
        pool.add_snapshot(f"a{idx}", model, tmp_path / f"a{idx}.pt")

    panel = pool.sample_panel()
    slots = pool.sample(200, panel=panel)

    assert len(panel.active) == 2
    assert {slot.name for slot in slots} <= set(panel.active)


def test_no_builtin_sample_panel_is_stable_within_update(tmp_path: Path):
    pool = NoBuiltinTrainingPool(
        active_pool_size=8,
        active_sample_panel_size=2,
        historical_training_archive_size=8,
        historical_sample_panel_size=2,
        min_games_before_eviction=0,
        rng=random.Random(11),
    )
    model = _tiny_model()
    for update in range(6):
        pool.set_current_update(update)
        pool.add_snapshot(f"u{update}", model, tmp_path / f"u{update}.pt")

    pool.set_current_update(6)
    panel = pool.sample_panel()

    assert pool.sample_panel() == panel
    assert pool.sample_panel(current_update=6) == panel

    pool.set_current_update(7)
    assert pool.sample_panel() == pool.sample_panel()


def test_no_builtin_panel_keeps_source_probabilities(tmp_path: Path):
    pool = NoBuiltinTrainingPool(
        active_pool_size=1,
        active_sample_panel_size=1,
        historical_training_archive_size=8,
        historical_sample_panel_size=1,
        min_games_before_eviction=0,
        current_learner_prob=0.4,
        active_pool_prob=0.3,
        historical_archive_prob=0.3,
        rng=random.Random(8),
    )
    model = _tiny_model()
    pool.add_snapshot("hist", model, tmp_path / "hist.pt", created_update=0)
    pool.set_current_update(1)
    pool.add_snapshot("active", model, tmp_path / "active.pt", created_update=1)
    panel = pool.sample_panel()

    slots = pool.sample(6000, panel=panel)
    current = sum(1 for slot in slots if slot.name == LEARNER_NAME)
    active = sum(1 for slot in slots if slot.name in panel.active)
    historical = sum(1 for slot in slots if slot.name in panel.historical)

    assert 0.37 * len(slots) <= current <= 0.43 * len(slots)
    assert 0.27 * len(slots) <= active <= 0.33 * len(slots)
    assert 0.27 * len(slots) <= historical <= 0.33 * len(slots)


def test_no_builtin_historical_panel_samples_weighted_buckets_when_capped(
    tmp_path: Path,
):
    pool = NoBuiltinTrainingPool(
        active_pool_size=1,
        historical_training_archive_size=12,
        historical_sample_panel_size=2,
        recent_eviction_archive_size=4,
        notable_archive_size=4,
        min_games_before_eviction=0,
        rng=random.Random(9),
    )
    model = _tiny_model()
    pool.set_current_update(0)
    notable = pool.add_snapshot(
        "notable",
        model,
        tmp_path / "notable.pt",
        notable=True,
    )
    pool.set_current_update(1)
    pool.add_snapshot("recent", model, tmp_path / "recent.pt")
    pool.set_current_update(2)
    pool.add_snapshot("active", model, tmp_path / "active.pt")
    pool.set_current_update(4)
    pool.rebuild_historical_archive()

    assert notable in pool.historical_snapshot_names("notable")
    assert pool.historical_snapshot_names("recent_eviction")
    assert pool.historical_snapshot_names("log")

    seen_buckets: set[str] = set()
    for update in range(4, 204):
        pool.set_current_update(update)
        panel = pool.sample_panel()
        for bucket in ("log", "recent_eviction", "notable"):
            if set(panel.historical) & set(pool.historical_snapshot_names(bucket)):
                seen_buckets.add(bucket)

    assert seen_buckets == {"log", "recent_eviction", "notable"}


def test_no_builtin_sample_filters_stale_panel(tmp_path: Path):
    pool = NoBuiltinTrainingPool(
        active_pool_size=1,
        historical_training_archive_size=4,
        min_games_before_eviction=0,
        current_learner_prob=0.0,
        active_pool_prob=0.0,
        historical_archive_prob=1.0,
        rng=random.Random(10),
    )
    model = _tiny_model()
    pool.add_snapshot("old", model, tmp_path / "old.pt", created_update=0)
    panel = pool.sample_panel()
    pool.set_current_update(1)
    pool.add_snapshot("new", model, tmp_path / "new.pt", created_update=1)
    for update in range(2, 10):
        pool.set_current_update(update)
        pool.add_snapshot(f"u{update}", model, tmp_path / f"u{update}.pt")

    slots = pool.sample(4, panel=panel)

    assert len(slots) == 4
    assert all(
        slot.name == LEARNER_NAME or slot.name in pool.all_snapshot_names() for slot in slots
    )


def test_no_builtin_sample_current_update_rotates_panel_clock(tmp_path: Path):
    pool = NoBuiltinTrainingPool(
        active_pool_size=1,
        historical_training_archive_size=12,
        historical_sample_panel_size=2,
        recent_eviction_archive_size=0,
        notable_archive_size=0,
        min_games_before_eviction=0,
        current_learner_prob=0.0,
        active_pool_prob=0.0,
        historical_archive_prob=1.0,
        rng=random.Random(5),
    )
    model = _tiny_model()
    for update in range(5):
        pool.set_current_update(update)
        pool.add_snapshot(f"u{update}", model, tmp_path / f"u{update}.pt")

    pool.sample(10, current_update=10)

    assert pool.current_update == 10
    assert len({slot.name for slot in pool.sample(100, current_update=10)}) <= 2


def test_no_builtin_sample_diversifies_all_same_non_current_when_possible(
    tmp_path: Path,
):
    pool = NoBuiltinTrainingPool(
        active_pool_size=4,
        current_learner_prob=0.0,
        active_pool_prob=1.0,
        historical_archive_prob=0.0,
        rng=random.Random(4),
    )
    model = _tiny_model()
    pool.add_snapshot("a", model, tmp_path / "a.pt")
    pool.add_snapshot("b", model, tmp_path / "b.pt")

    for _ in range(100):
        slots = pool.sample(3)
        assert len({slot.name for slot in slots}) > 1


def test_no_builtin_sample_diversifies_within_sampled_source_when_possible(
    tmp_path: Path,
):
    pool = NoBuiltinTrainingPool(
        active_pool_size=1,
        historical_training_archive_size=8,
        recent_eviction_archive_size=0,
        notable_archive_size=0,
        min_games_before_eviction=0,
        current_learner_prob=0.0,
        active_pool_prob=0.0,
        historical_archive_prob=1.0,
        rng=random.Random(3),
    )
    model = _tiny_model()
    for update in range(3):
        pool.set_current_update(update)
        pool.add_snapshot(f"u{update}", model, tmp_path / f"u{update}.pt")

    active_name = pool.active_snapshot_names()[0]
    historical = set(pool.historical_snapshot_names())
    assert len(historical) >= 2

    for _ in range(200):
        slots = pool.sample(3)
        names = {slot.name for slot in slots}
        assert active_name not in names
        assert names <= historical


def test_no_builtin_diversify_does_not_cross_sources(tmp_path: Path):
    pool = NoBuiltinTrainingPool(
        active_pool_size=2,
        historical_training_archive_size=8,
        recent_eviction_archive_size=0,
        notable_archive_size=0,
        min_games_before_eviction=0,
        current_learner_prob=0.0,
        active_pool_prob=0.0,
        historical_archive_prob=1.0,
        rng=random.Random(6),
    )
    model = _tiny_model()
    pool.add_snapshot("active", model, tmp_path / "active.pt")
    pool.set_current_update(1)
    pool.add_snapshot(
        "historical",
        model,
        tmp_path / "historical.pt",
        created_update=0,
        enter_active=False,
    )

    slots = pool.sample(3)

    assert [slot.name for slot in slots] == ["frozen:historical"] * 3
