import math

from owars.game.geometry import (
    angle_to,
    distance,
    line_circle_intersects,
    predicted_position,
    wrap_angle,
)
from owars.game.types import CENTER, ROTATION_RADIUS_LIMIT, SUN_RADIUS


def test_distance_symmetric():
    assert distance(0.0, 0.0, 3.0, 4.0) == 5.0
    assert distance(3.0, 4.0, 0.0, 0.0) == 5.0


def test_angle_to_quadrants():
    assert math.isclose(angle_to(0, 0, 1, 0), 0.0, abs_tol=1e-9)
    assert math.isclose(angle_to(0, 0, 0, 1), math.pi / 2, abs_tol=1e-9)
    assert math.isclose(angle_to(0, 0, -1, 0), math.pi, abs_tol=1e-9)
    assert math.isclose(angle_to(0, 0, 0, -1), -math.pi / 2, abs_tol=1e-9)


def test_wrap_angle_range():
    for a in (-7.0, -math.pi, 0.0, math.pi, 7.0):
        w = wrap_angle(a)
        assert -math.pi - 1e-9 <= w <= math.pi + 1e-9


def test_line_circle_segment_through_sun():
    cx, cy = CENTER
    # A path that crosses the sun.
    assert line_circle_intersects(0.0, 50.0, 100.0, 50.0, cx, cy, SUN_RADIUS)


def test_line_circle_segment_misses_sun():
    cx, cy = CENTER
    # A path well above the sun.
    assert not line_circle_intersects(0.0, 5.0, 100.0, 5.0, cx, cy, SUN_RADIUS)


def test_static_planet_does_not_rotate():
    cx, cy = CENTER
    # `orbital_radius + planet_radius >= ROTATION_RADIUS_LIMIT` ⇒ static.
    far_radius = ROTATION_RADIUS_LIMIT  # well outside the rotation zone
    x, y = predicted_position(cx + far_radius, cy, 1.0, 0.05, steps=100)
    assert math.isclose(x, cx + far_radius, abs_tol=1e-9)
    assert math.isclose(y, cy, abs_tol=1e-9)


def test_orbiting_planet_returns_after_full_rotation():
    cx, cy = CENTER
    omega = 0.04
    initial = (cx + 20.0, cy)
    # Steps to full revolution.
    n = int(round(2 * math.pi / omega))
    x, y = predicted_position(*initial, 1.0, omega, steps=n)
    # The integer step count introduces a rounding error of up to omega
    # radians, so the post-rotation point is within ~r·omega of the start.
    assert distance(x, y, initial[0], initial[1]) < 20.0 * omega + 1e-6


def test_orbiting_planet_advances():
    cx, cy = CENTER
    initial_x, initial_y = cx + 15.0, cy
    x, y = predicted_position(initial_x, initial_y, 1.0, 0.05, steps=10)
    # After a few steps it should be off the starting position.
    assert distance(x, y, initial_x, initial_y) > 1.0
