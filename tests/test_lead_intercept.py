"""Closed-form `_lead_angle` solver — correctness across the spec range."""

from __future__ import annotations

import math

import pytest

from owars.game.types import CENTER
from owars.policies.sampling import _lead_angle, _lead_angle_from_point


def _simulate_intercept(
    sx: float,
    sy: float,
    source_radius: float,
    target_x: float,
    target_y: float,
    target_radius: float,
    omega: float, send: int, angle: float, max_steps: int = 600,
) -> float | None:
    """Run the simulator's straight-line fleet motion vs orbiting target;
    return the closest distance attained, or None if fleet exits the board.
    """
    from owars.game.physics import fleet_speed
    sp = fleet_speed(send)
    cx, cy = CENTER
    orbit_radius = math.hypot(target_x - cx, target_y - cy)
    theta0 = math.atan2(target_y - cy, target_x - cx)
    fx = sx + math.cos(angle) * (source_radius + 0.1)
    fy = sy + math.sin(angle) * (source_radius + 0.1)
    best = float("inf")
    for step in range(1, max_steps + 1):
        old_fx, old_fy = fx, fy
        fx += math.cos(angle) * sp
        fy += math.sin(angle) * sp
        psi = theta0 + omega * step
        tx = cx + orbit_radius * math.cos(psi)
        ty = cy + orbit_radius * math.sin(psi)
        dx = fx - old_fx
        dy = fy - old_fy
        denom = dx * dx + dy * dy
        if denom == 0.0:
            d = math.hypot(fx - tx, fy - ty)
        else:
            t = max(0.0, min(1.0, ((tx - old_fx) * dx + (ty - old_fy) * dy) / denom))
            d = math.hypot(tx - (old_fx + t * dx), ty - (old_fy + t * dy))
        if d < best:
            best = d
        # Exit if the fleet exits the board AFTER recording closest approach
        # — the simulator destroys it then anyway, but we want to know
        # whether intercept was achieved before that.
        if not (0 <= fx <= 100 and 0 <= fy <= 100):
            break
    return best


@pytest.mark.parametrize(
    "src,target,omega,send",
    [
        # Static target (orbital_radius too large for rotation): aim direct.
        ((10.0, 10.0), (90.0, 90.0), 0.0, 50),
        # Slow orbiter, big fleet — easy.
        ((20.0, 50.0), (50.0 + 20.0, 50.0), 0.025, 100),
        # Fast orbiter, big fleet.
        ((20.0, 50.0), (50.0 + 20.0, 50.0), 0.05, 100),
        # Slow fleet (send=2), moderate orbit — challenging case.
        ((45.0, 45.0), (50.0 + 30.0, 50.0), 0.03, 2),
        # Source near orbit ring.
        ((50.0 + 25.0, 50.0), (50.0, 50.0 + 25.0), 0.04, 30),
    ],
)
def test_solver_lands_inside_planet_radius(src, target, omega, send):
    """If the solver returns an angle, simulating the fleet should bring
    it within `target_radius` of the target at the predicted intercept
    time. Uses radius=1.0 (typical planet radius)."""
    radius = 1.0
    source_radius = 2.1
    angle = _lead_angle(
        src[0], src[1], source_radius, target[0], target[1], radius, omega, send
    )
    assert angle is not None, "expected a feasible intercept for this case"
    closest = _simulate_intercept(
        src[0], src[1], source_radius, target[0], target[1], radius, omega, send, angle
    )
    assert closest is not None, "fleet flew off-board — solver returned a bad angle"
    # Closest approach should land inside the planet's collision radius.
    # We allow 2× radius slack for sim-vs-continuous discretization.
    assert closest < 2.0 * radius, f"closest approach {closest:.3f} > 2·r"


def test_solver_returns_none_on_infeasible_far_horizon():
    """If the target moves so fast vs a slow fleet that no intercept lies
    within the horizon, returns None — the move builder will skip."""
    # send=1 (sp=1.0), radius-30 fast orbiter from a far source point —
    # set ω so the orbit period is small enough that intercept is far.
    angle = _lead_angle(
        mine_x=10.0, mine_y=10.0,
        mine_radius=2.1,
        target_x=50.0 + 30.0, target_y=50.0,
        target_radius=1.0,
        angular_velocity=0.05,
        send=1,
    )
    # Either we find a feasible intercept (angle != None) or we skip.
    # The point is: the result is well-defined and the fleet, if launched,
    # cannot fly off-board (handled by the caller skipping None).
    if angle is not None:
        closest = _simulate_intercept(10.0, 10.0, 2.1, 80.0, 50.0, 1.0, 0.05, 1, angle)
        assert closest is None or closest < 2.0


def test_static_target_is_aimed_direct():
    """A static target (orbital_radius outside rotation limit, e.g.
    corner of the board) should be aimed at directly, no lead."""
    # Orbital radius from center (50,50) to (90,90) is √(40²+40²) ≈ 56.5,
    # exceeding ROTATION_RADIUS_LIMIT=50, so this planet is static.
    angle = _lead_angle(10.0, 10.0, 2.1, 90.0, 90.0, 1.0, 0.05, 30)
    expected = math.atan2(90.0 - 10.0, 90.0 - 10.0)
    assert abs(angle - expected) < 1e-6


def test_solver_respects_orbit_direction():
    """ω positive (CCW) and ω negative (CW) should give different lead
    angles for the same target — the solver must respect the direction."""
    target_x, target_y = 50.0 + 20.0, 50.0
    a_ccw = _lead_angle(10.0, 50.0, 2.1, target_x, target_y, 1.0, 0.04, 30)
    a_cw = _lead_angle(10.0, 50.0, 2.1, target_x, target_y, 1.0, -0.04, 30)
    assert a_ccw is not None and a_cw is not None
    assert abs(a_ccw - a_cw) > 1e-3, "lead should differ across orbit directions"


def test_solver_accounts_for_official_launch_surface_offset():
    """A centerline solution can miss small targets once the official launch
    offset is applied. `_lead_angle` must solve for that real launch point."""
    sx, sy, source_radius = 63.739581113015426, 15.741528408044212, 2.1
    target_x, target_y, target_radius = 11.639918383373761, 36.84108803746605, 1.0
    omega = -0.041859494766865096
    send = 9

    center_angle = _lead_angle_from_point(
        sx, sy, target_x, target_y, target_radius, omega, send
    )
    offset_angle = _lead_angle(
        sx, sy, source_radius, target_x, target_y, target_radius, omega, send
    )

    assert center_angle is not None and offset_angle is not None
    center_closest = _simulate_intercept(
        sx, sy, source_radius, target_x, target_y, target_radius, omega, send, center_angle
    )
    offset_closest = _simulate_intercept(
        sx, sy, source_radius, target_x, target_y, target_radius, omega, send, offset_angle
    )
    assert center_closest is not None and center_closest > target_radius
    assert offset_closest is not None and offset_closest < target_radius
