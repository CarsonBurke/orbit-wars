"""Game-state value types and shared constants.

Mirrors the named tuples exported by `kaggle_environments.envs.orbit_wars.orbit_wars`
so we can use them without depending on the kaggle-environments install at import
time. The runtime agent (`submission/main.py`) is free to import the upstream
tuples directly.
"""

from __future__ import annotations

from typing import NamedTuple

# Board geometry — confirmed from the official "How to Play" page.
BOARD_SIZE: float = 100.0
CENTER: tuple[float, float] = (50.0, 50.0)
SUN_RADIUS: float = 10.0

# A planet rotates iff `orbital_radius + planet_radius < ROTATION_RADIUS_LIMIT`.
ROTATION_RADIUS_LIMIT: float = 50.0

# Default per-turn fleet speed cap. Configurable by the host via `shipSpeed`.
MAX_SHIP_SPEED: float = 6.0

# Production is integer in [1, 5]; planet radius = 1 + ln(production).
MAX_PRODUCTION: int = 5

# Comets spawn in groups of 4 at these step indices (one per quadrant).
COMET_SPAWN_STEPS: tuple[int, ...] = (50, 150, 250, 350, 450)

# Player IDs. -1 is neutral; players are 0..3.
NEUTRAL: int = -1
NUM_PLAYERS_2P: int = 2
NUM_PLAYERS_4P: int = 4


class Planet(NamedTuple):
    """`[id, owner, x, y, radius, ships, production]` — matches the obs layout."""

    id: int
    owner: int
    x: float
    y: float
    radius: float
    ships: int
    production: int


class Fleet(NamedTuple):
    """`[id, owner, x, y, angle, from_planet_id, ships]` — matches the obs layout."""

    id: int
    owner: int
    x: float
    y: float
    angle: float
    from_planet_id: int
    ships: int


class Move(NamedTuple):
    """`[from_planet_id, angle, num_ships]` — the action format."""

    from_planet_id: int
    angle: float
    num_ships: int

    def as_list(self) -> list:
        return [self.from_planet_id, float(self.angle), int(self.num_ships)]
