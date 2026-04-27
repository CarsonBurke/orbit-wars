"""Elo ratings for the self-play league.

We track a single rating per *identity* — the live learner ("learner") and
each frozen snapshot ("frozen:<label>"). Snapshots don't learn but their
ratings drift as new games are played; this is what lets us cull weak
snapshots (UCB-based eviction in `OpponentPool`).

For 2-player matches this is textbook Elo. For FFA we aggregate scores by
identity (when "self" fills multiple seats they share an identity) and
update each pair of distinct identities once, with `K / (N − 1)` per pair
so the per-game K-budget on any one rating is bounded.

The pool evicts by `ucb(name) = rating + 2 * K / sqrt(games_played)` rather
than raw rating: a freshly-added snapshot whose initial rating happens to
dip on its first game shouldn't be culled before it's had a fair sample.
Eviction-by-lowest-UCB means we cut a snapshot only when we're *confident*
it's weak — high uncertainty (low n) inflates UCB and protects newcomers,
which is the opposite of what an LCB-based rule would do (LCB punishes
fresh snapshots since their interval extends far below the mean). Cheap
stand-in for Glicko-2's rating-deviation without the full RD bookkeeping.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field


def expected_score(r_a: float, r_b: float) -> float:
    """E_a = 1 / (1 + 10^((R_b − R_a) / 400)). Standard Elo."""
    return 1.0 / (1.0 + 10.0 ** ((r_b - r_a) / 400.0))


@dataclass
class EloTracker:
    """In-memory rating book.

    `ratings` is `name → rating`. `games_played` is `name → int` for
    optional K-decay later (not used by default).
    """

    initial_rating: float = 1500.0
    k_factor: float = 32.0
    ratings: dict[str, float] = field(default_factory=dict)
    games_played: dict[str, int] = field(default_factory=lambda: defaultdict(int))

    def get(self, name: str) -> float:
        return self.ratings.get(name, self.initial_rating)

    def ucb(self, name: str, z: float = 2.0) -> float:
        """Upper confidence bound on the rating: `μ + z·K/√n`.

        With `z=2.0` and the default K=32, a snapshot at rating 1500 with
        zero games has ucb ≈ 1564; after 64 games it tightens to ≈ 1508.
        `OpponentPool` evicts by *lowest* UCB so few-games snapshots get a
        protective uncertainty buffer — only snapshots that are well-
        measured *and* low-rated drop below newcomers and get culled.
        """
        n = max(1, self.games_played.get(name, 0))
        return self.get(name) + z * self.k_factor / math.sqrt(n)

    def set(self, name: str, rating: float) -> None:
        self.ratings[name] = rating

    def ensure(self, name: str) -> None:
        if name not in self.ratings:
            self.ratings[name] = self.initial_rating

    def update_pair(self, a: str, b: str, score_a: float, k: float | None = None) -> None:
        """Single Elo update. `score_a ∈ {0, 0.5, 1}`."""
        if a == b:
            return
        self.ensure(a)
        self.ensure(b)
        k = self.k_factor if k is None else k
        ea = expected_score(self.ratings[a], self.ratings[b])
        delta = k * (score_a - ea)
        self.ratings[a] += delta
        self.ratings[b] -= delta
        self.games_played[a] += 1
        self.games_played[b] += 1

    def update_from_game(self, seats: list[tuple[str, float]]) -> None:
        """One Elo update per *pair of distinct identities* in the game.

        `seats` is a list of `(identity_name, final_score)` — one entry per
        seat. When a single identity fills multiple seats (self-play in
        FFA), its final score is the *mean* of those seats' scores. Pairs
        of the same identity are skipped (self can't beat itself).

        K per pair is scaled to `K / (M − 1)` where M is the number of
        distinct identities, so any one rating moves at most ~K per game.
        """
        if len(seats) < 2:
            return
        by_identity: dict[str, list[float]] = defaultdict(list)
        for name, score in seats:
            by_identity[name].append(float(score))
        identities = list(by_identity.keys())
        if len(identities) < 2:
            return  # everyone was the same identity (e.g., all self)
        avg = {n: sum(s) / len(s) for n, s in by_identity.items()}
        per_pair_k = self.k_factor / max(1, len(identities) - 1)
        for i, a in enumerate(identities):
            for b in identities[i + 1 :]:
                if avg[a] > avg[b]:
                    sa = 1.0
                elif avg[a] < avg[b]:
                    sa = 0.0
                else:
                    sa = 0.5
                self.update_pair(a, b, sa, k=per_pair_k)

    def snapshot_dict(self) -> dict[str, float]:
        """Plain dict for logging / persistence."""
        return dict(self.ratings)
