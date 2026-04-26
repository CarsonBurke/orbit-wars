"""Nearest-planet sniper — the starter agent shipped with the competition.

Reproduced here so it can be used as a calibration baseline both during
training (as an opponent) and during evaluation (does our learned policy
beat the canonical first-pass agent?).
"""

from __future__ import annotations

from typing import Any

from ..game import angle_to, distance, parse_observation


def sniper_agent(obs: Any) -> list[list]:
    o = parse_observation(obs)
    targets = o.enemy_planets() + o.neutral_planets()
    if not targets:
        return []

    moves: list[list] = []
    for mine in o.my_planets():
        nearest = min(targets, key=lambda t: distance(mine.x, mine.y, t.x, t.y))
        ships_needed = nearest.ships + 1
        if mine.ships >= ships_needed:
            moves.append([mine.id, angle_to(mine.x, mine.y, nearest.x, nearest.y), ships_needed])
    return moves
