"""Nearest-planet sniper baseline.

This started as the competition starter bot, but our local training baseline
leads orbiting targets so it is a useful fixed opponent instead of mostly
teaching the learner to exploit missed shots.
"""

from __future__ import annotations

from typing import Any

from ..game import distance, parse_observation
from ..policies.sampling import (
    _lead_solution,
    _route_blockers_from_rows,
    _route_clear_to_solution,
)


def sniper_agent(obs: Any) -> list[list]:
    o = parse_observation(obs)
    targets = o.enemy_planets() + o.neutral_planets()
    if not targets:
        return []
    blockers = _route_blockers_from_rows(o.planets, o.angular_velocity, o.comet_planet_ids)

    moves: list[list] = []
    for mine in o.my_planets():
        for target in sorted(targets, key=lambda t: distance(mine.x, mine.y, t.x, t.y)):
            ships_needed = target.ships + 1
            if mine.ships < ships_needed:
                continue
            solution = _lead_solution(
                mine.x,
                mine.y,
                mine.radius,
                target.x,
                target.y,
                target.radius,
                o.angular_velocity,
                ships_needed,
                target_is_comet=int(target.id) in o.comet_planet_ids,
            )
            if solution is None:
                continue
            if not _route_clear_to_solution(
                mine.id,
                target.id,
                mine.x,
                mine.y,
                mine.radius,
                solution,
                ships_needed,
                blockers,
                o.angular_velocity,
            ):
                continue
            moves.append([mine.id, solution.angle, ships_needed])
            break
    return moves
