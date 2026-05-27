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


def test_ffa_uses_pregame_ratings():
    """Regression: every pair in a >2-identity game must score against the
    PRE-game ratings. Updating in place would let a later pair read a rating an
    earlier pair already moved this game. Only visible when the starting ratings
    differ (equal ratings give E=0.5 everywhere, hiding the bug — which is why
    `test_multi_player_pairwise` at equal ratings doesn't catch it).
    """
    k = 32.0
    pre = {"a": 1600.0, "b": 1500.0, "c": 1400.0}
    scores = {"a": 3.0, "b": 2.0, "c": 1.0}
    elo = EloTracker(initial_rating=1500.0, k_factor=k)
    for name, rating in pre.items():
        elo.set(name, rating)
    elo.update_from_game([(n, scores[n]) for n in pre])

    # Reference: every pair scored from the FROZEN pre-game ratings, deltas
    # accumulated and applied after — what an order-independent update yields.
    per_pair_k = k / (len(pre) - 1)  # K / (M − 1)
    expected = dict(pre)
    names = list(pre)
    for i, x in enumerate(names):
        for y in names[i + 1 :]:
            sx = 1.0 if scores[x] > scores[y] else (0.0 if scores[x] < scores[y] else 0.5)
            d = per_pair_k * (sx - expected_score(pre[x], pre[y]))
            expected[x] += d
            expected[y] -= d

    for n in names:
        assert abs(elo.get(n) - expected[n]) < 1e-9, (n, elo.get(n), expected[n])
    assert abs(sum(elo.get(n) for n in names) - sum(pre.values())) < 1e-6  # zero-sum
    # Each identity played M−1 pairs, so games_played reflects every pairing.
    assert all(elo.games_played[n] == len(pre) - 1 for n in names)
