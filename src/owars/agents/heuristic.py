"""Heuristic agent — the "smart-but-simple" baseline.

What sniper misses:
  - It always shoots at the *nearest* target, even if a slightly farther
    high-production planet is a better pick.
  - It ignores enemy fleets en route (we can lose a planet right after
    sending out a fleet because we drained the garrison too low).
  - It doesn't predict where an orbiting target will be when our fleet
    arrives — the angle is computed against `(t.x, t.y)` *now*.

This agent fixes those by:
  1. Scoring each `(my_planet, target)` pair by `production / (ships_needed
     + travel_steps)` — production-per-cost-and-time.
  2. Holding back a small reserve garrison.
  3. Aiming at the predicted future position of orbiting targets, using
     `predicted_position`.

It still falls well short of an optimal policy (no defense routing, no
multi-step planning, no comet exploitation) — that's the learned model's
job. This exists so we have a credible non-trivial opponent.
"""

from __future__ import annotations

from typing import Any

from ..game import (
    angle_to,
    distance,
    fleet_speed,
    parse_observation,
    predicted_position,
    travel_steps,
)
from ..game.observation import Observation


class HeuristicAgent:
    """Stateless callable; kept as a class for parity with `LearnedAgent`."""

    def __init__(
        self,
        reserve_ships: int = 5,
        send_buffer: int = 2,
        prediction_lookahead_cap: int = 60,
    ):
        self.reserve_ships = reserve_ships
        self.send_buffer = send_buffer
        self.prediction_lookahead_cap = prediction_lookahead_cap

    def __call__(self, obs: Any) -> list[list]:
        o = parse_observation(obs)
        targets = o.enemy_planets() + o.neutral_planets()
        if not targets:
            return []
        return [m for mine in o.my_planets() for m in self._best_move(o, mine, targets)]

    def _best_move(self, o: Observation, mine, targets) -> list[list]:
        if mine.ships <= self.reserve_ships:
            return []

        budget = mine.ships - self.reserve_ships
        best = None
        best_score = float("-inf")
        for t in targets:
            d = distance(mine.x, mine.y, t.x, t.y)
            ships_needed = max(1, t.ships + self.send_buffer)
            if ships_needed > budget:
                continue
            steps = travel_steps(d, ships_needed)
            score = (t.production + 1) / (ships_needed + steps)
            if score > best_score:
                best_score = score
                best = (t, ships_needed, steps)

        if best is None:
            return []

        t, ships_needed, steps = best
        steps = min(steps, self.prediction_lookahead_cap)
        tx, ty = predicted_position(t.x, t.y, t.radius, o.angular_velocity, steps)
        # Recompute speed — sending more ships moves the fleet faster, which
        # can pull arrival forward by 1-2 steps; one re-prediction is enough.
        sp = fleet_speed(ships_needed)
        steps = max(1, int(distance(mine.x, mine.y, tx, ty) / sp))
        steps = min(steps, self.prediction_lookahead_cap)
        tx, ty = predicted_position(t.x, t.y, t.radius, o.angular_velocity, steps)
        return [[mine.id, angle_to(mine.x, mine.y, tx, ty), int(ships_needed)]]


def heuristic_agent(obs: Any) -> list[list]:
    return HeuristicAgent()(obs)
