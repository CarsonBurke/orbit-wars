"""Vector math + collision and orbit prediction helpers.

The official scorer is the simulator itself, so geometry mistakes here cost
games. Keep these primitives well-tested in `tests/test_geometry.py`.
"""

from __future__ import annotations

import math

from .types import CENTER, ROTATION_RADIUS_LIMIT


def distance(ax: float, ay: float, bx: float, by: float) -> float:
    return math.hypot(ax - bx, ay - by)


def angle_to(from_x: float, from_y: float, to_x: float, to_y: float) -> float:
    """Angle in radians from `(from)` to `(to)`. Matches `math.atan2(dy, dx)`."""
    return math.atan2(to_y - from_y, to_x - from_x)


def wrap_angle(theta: float) -> float:
    """Wrap an angle to (-pi, pi]."""
    return (theta + math.pi) % (2.0 * math.pi) - math.pi


def line_circle_intersects(
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    cx: float,
    cy: float,
    r: float,
) -> bool:
    """Does the segment (x0,y0)→(x1,y1) come within `r` of (cx,cy)?

    Used for the same continuous collision check the simulator does — fleets
    are destroyed if any point of their step crosses the sun, and combat is
    triggered if a fleet's path comes within a planet's radius.
    """
    dx = x1 - x0
    dy = y1 - y0
    fx = x0 - cx
    fy = y0 - cy
    a = dx * dx + dy * dy
    if a == 0.0:
        return (fx * fx + fy * fy) <= r * r
    b = 2.0 * (fx * dx + fy * dy)
    c = fx * fx + fy * fy - r * r
    disc = b * b - 4.0 * a * c
    if disc < 0.0:
        return False
    disc = math.sqrt(disc)
    t1 = (-b - disc) / (2.0 * a)
    t2 = (-b + disc) / (2.0 * a)
    return (0.0 <= t1 <= 1.0) or (0.0 <= t2 <= 1.0) or (t1 < 0.0 and t2 > 1.0)


def predicted_position(
    initial_x: float,
    initial_y: float,
    radius: float,
    angular_velocity: float,
    steps: int,
    cx: float = CENTER[0],
    cy: float = CENTER[1],
) -> tuple[float, float]:
    """Where will an orbiting planet be `steps` turns from now?

    A planet orbits iff `orbital_radius + planet_radius < ROTATION_RADIUS_LIMIT`;
    static planets are returned unchanged. We compute orbital radius from the
    *initial* position (which is what the obs gives us).
    """
    dx = initial_x - cx
    dy = initial_y - cy
    orbital_radius = math.hypot(dx, dy)
    if orbital_radius + radius >= ROTATION_RADIUS_LIMIT:
        return initial_x, initial_y
    theta0 = math.atan2(dy, dx)
    theta = theta0 + angular_velocity * steps
    return cx + orbital_radius * math.cos(theta), cy + orbital_radius * math.sin(theta)
