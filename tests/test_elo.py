"""Elo arithmetic + multi-player pairwise updates."""

from __future__ import annotations

from owars.training.elo import EloTracker, expected_score


def test_expected_score_symmetry():
    # Equal ratings → 0.5/0.5.
    assert abs(expected_score(1500, 1500) - 0.5) < 1e-9
    # 400-point gap → ~0.909 / 0.091.
    e = expected_score(1900, 1500)
    assert abs(e - (1 - expected_score(1500, 1900))) < 1e-9
    assert e > 0.9


def test_update_pair_zero_sum():
    elo = EloTracker(initial_rating=1500.0, k_factor=32.0)
    elo.update_pair("a", "b", score_a=1.0)
    delta_a = elo.get("a") - 1500.0
    delta_b = elo.get("b") - 1500.0
    assert abs(delta_a + delta_b) < 1e-9  # zero-sum
    assert delta_a == 16.0  # K * (1 - 0.5)


def test_update_pair_skips_self():
    elo = EloTracker()
    elo.update_pair("a", "a", score_a=1.0)
    # No-op: rating unchanged, games_played unchanged.
    assert elo.get("a") == elo.initial_rating
    assert elo.games_played["a"] == 0


def test_multi_player_pairwise():
    """4-player FFA: rank order should produce monotone Elo deltas."""
    elo = EloTracker(initial_rating=1500.0, k_factor=32.0)
    # Player a wins outright, then b > c > d.
    elo.update_from_game([("a", 4.0), ("b", 3.0), ("c", 2.0), ("d", 1.0)])
    ra, rb, rc, rd = elo.get("a"), elo.get("b"), elo.get("c"), elo.get("d")
    assert ra > rb > rc > rd
    # Zero-sum across the table.
    assert abs((ra + rb + rc + rd) - 4 * 1500.0) < 1e-6


def test_self_play_seats_share_identity():
    """When 'learner' fills two seats and beats two snapshots, learner is
    one identity (its score is the mean of its two seats), so the Elo
    update fires once per (learner, snapshot) pair — not per seat. The
    bound on ΔR per game is K, not 2K.
    """
    elo = EloTracker(initial_rating=1500.0, k_factor=32.0)
    elo.update_from_game(
        [("learner", 4.0), ("learner", 3.0), ("frozen:A", 2.0), ("frozen:B", 1.0)]
    )
    # 3 identities → per-pair K = 16. Two pairs involve "learner", so
    # max possible Δlearner = 2 * 16 = 32 (the per-game K-budget).
    delta = elo.get("learner") - 1500.0
    assert 0 < delta <= 32.0
    # And learner ranks above both snapshots.
    assert elo.get("learner") > elo.get("frozen:A") > elo.get("frozen:B")


def test_all_same_identity_noop():
    elo = EloTracker()
    before = elo.get("learner")
    elo.update_from_game([("learner", 1.0), ("learner", 0.0)])
    assert elo.get("learner") == before  # no distinct opponent → no update
