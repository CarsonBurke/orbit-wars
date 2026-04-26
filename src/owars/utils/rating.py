"""Lightweight Glicko-style rating updates for offline ladder evaluation.

Matches the *flavor* of Kaggle's matchmaking: each agent has a Gaussian
N(μ, σ²); after each game we shift μ towards the result and shrink σ. We use
this to track relative agent strength during local self-play, *not* to
predict the public leaderboard. The Kaggle leaderboard uses its own server
ratings and that is the only score that matters.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class Rating:
    mu: float = 600.0
    sigma: float = 200.0  # broad initial uncertainty


def _g(sigma: float, q: float = 1.0 / 173.7178) -> float:
    return 1.0 / math.sqrt(1.0 + 3.0 * (q * sigma) ** 2 / math.pi**2)


def _expected(mu_a: float, mu_b: float, sigma_b: float, q: float = 1.0 / 173.7178) -> float:
    return 1.0 / (1.0 + 10.0 ** (-_g(sigma_b, q) * (mu_a - mu_b) / 400.0))


def update_ratings(a: Rating, b: Rating, score_a: float) -> tuple[Rating, Rating]:
    """Update `a` vs `b` after a game with `score_a` ∈ {1, 0.5, 0}.

    `score_a == 1` means `a` won, `0` means `b` won, `0.5` is a draw. Returns
    fresh copies (so callers can keep history).
    """
    q = 1.0 / 173.7178
    score_b = 1.0 - score_a

    # Update a from b's perspective; then b from a's.
    e_ab = _expected(a.mu, b.mu, b.sigma, q)
    e_ba = _expected(b.mu, a.mu, a.sigma, q)
    g_b = _g(b.sigma, q)
    g_a = _g(a.sigma, q)

    d2_a = 1.0 / (q * q * g_b * g_b * e_ab * (1.0 - e_ab) + 1e-9)
    d2_b = 1.0 / (q * q * g_a * g_a * e_ba * (1.0 - e_ba) + 1e-9)

    new_a_mu = a.mu + (q / (1.0 / a.sigma**2 + 1.0 / d2_a)) * g_b * (score_a - e_ab)
    new_b_mu = b.mu + (q / (1.0 / b.sigma**2 + 1.0 / d2_b)) * g_a * (score_b - e_ba)
    new_a_sigma = math.sqrt(1.0 / (1.0 / a.sigma**2 + 1.0 / d2_a))
    new_b_sigma = math.sqrt(1.0 / (1.0 / b.sigma**2 + 1.0 / d2_b))
    return (
        Rating(mu=new_a_mu, sigma=max(30.0, new_a_sigma)),
        Rating(mu=new_b_mu, sigma=max(30.0, new_b_sigma)),
    )
