"""Fleet speed and travel-time helpers.

The fleet speed formula is:

    speed = 1.0 + (max_speed - 1.0) * (log(ships) / log(1000))**1.5

Larger fleets travel faster. `1` ship goes 1 unit/turn; `~1000` ships hit the
cap (default 6.0). Useful for sizing decisions: a marginal ship of garrison
held back is "free" production but a marginal ship in a fleet doesn't speed
the fleet up much above ~500 ships.
"""

from __future__ import annotations

import math

from .types import MAX_SHIP_SPEED


def fleet_speed(ships: int, max_speed: float = MAX_SHIP_SPEED) -> float:
    if ships <= 1:
        return 1.0
    frac = math.log(ships) / math.log(1000.0)
    return min(max_speed, 1.0 + (max_speed - 1.0) * (frac**1.5))


def travel_steps(distance: float, ships: int, max_speed: float = MAX_SHIP_SPEED) -> int:
    """Whole turns needed to cover `distance` in a straight line.

    Approximation: ignores the planet/comet that may move *during* travel.
    Good enough for sizing decisions; for precise interception use
    `predicted_position` and binary-search the arrival step.
    """
    sp = fleet_speed(ships, max_speed=max_speed)
    if sp <= 0.0:
        return 10**9
    return max(1, math.ceil(distance / sp))
