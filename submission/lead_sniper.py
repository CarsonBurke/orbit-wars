"""Standalone lead-sniper Kaggle submission.

Exports `agent(obs)` with no repo or Torch dependency. This is the local
sniper baseline upgraded to lead orbiting planets instead of aiming at their
current coordinates.
"""

from __future__ import annotations

import math
from typing import Any


BOARD_SIZE = 100.0
CENTER = 50.0
ROTATION_RADIUS_LIMIT = 50.0
MAX_SHIP_SPEED = 6.0
SUN_RADIUS = 10.0
LOG_1000 = math.log(1000.0)
LEAD_T_HORIZON_STEPS = 600.0
LEAD_MAX_TURNS = int(LEAD_T_HORIZON_STEPS)
LEAD_MAX_SCAN_DISTANCE = math.hypot(BOARD_SIZE, BOARD_SIZE) + 8.0


def _get(obs: Any, key: str, default: Any = None) -> Any:
    if isinstance(obs, dict):
        return obs.get(key, default)
    return getattr(obs, key, default)


def _fleet_speed(ships: int) -> float:
    if ships <= 1:
        return 1.0
    frac = math.log(float(ships)) / LOG_1000
    return min(MAX_SHIP_SPEED, 1.0 + (MAX_SHIP_SPEED - 1.0) * (frac**1.5))


def _angle_to(source: list[float], target: list[float]) -> float:
    return math.atan2(float(target[3]) - float(source[3]), float(target[2]) - float(source[2]))


def _distance_sq(source: list[float], target: list[float]) -> float:
    dx = float(source[2]) - float(target[2])
    dy = float(source[3]) - float(target[3])
    return dx * dx + dy * dy


def _lead_solution_from_point(
    source_x: float,
    source_y: float,
    target_x: float,
    target_y: float,
    target_radius: float,
    angular_velocity: float,
    send: int,
) -> tuple[float, float, float, float] | None:
    speed = _fleet_speed(send)
    if speed <= 0.0:
        return None

    orbit_radius = math.hypot(target_x - CENTER, target_y - CENTER)
    is_orbiting = (
        orbit_radius + target_radius < ROTATION_RADIUS_LIMIT
        and abs(angular_velocity) > 1e-12
        and orbit_radius > 1e-9
    )
    if not is_orbiting:
        distance = math.hypot(target_x - source_x, target_y - source_y)
        if distance / speed > LEAD_T_HORIZON_STEPS:
            return None
        return (
            math.atan2(target_y - source_y, target_x - source_x),
            max(1.0, math.ceil(max(0.0, distance - target_radius) / speed)),
            target_x,
            target_y,
        )

    theta0 = math.atan2(target_y - CENTER, target_x - CENTER)
    previous_error: float | None = None
    max_turns = min(
        LEAD_MAX_TURNS,
        max(1, int(math.ceil((LEAD_MAX_SCAN_DISTANCE + target_radius) / speed)) + 1),
    )
    for turn in range(1, max_turns + 1):
        theta = theta0 + angular_velocity * (turn - 1)
        tx = CENTER + orbit_radius * math.cos(theta)
        ty = CENTER + orbit_radius * math.sin(theta)
        distance = math.hypot(tx - source_x, ty - source_y)
        error = distance - turn * speed
        if error <= target_radius:
            prev_dist = max(0.0, (turn - 1) * speed)
            if distance >= prev_dist - target_radius:
                return math.atan2(ty - source_y, tx - source_x), float(turn), tx, ty
        if (
            previous_error is not None
            and previous_error < -target_radius
            and error > target_radius
        ):
            break
        previous_error = error

    return None


def _point_to_segment_distance(
    px: float,
    py: float,
    ax: float,
    ay: float,
    bx: float,
    by: float,
) -> float:
    dx = bx - ax
    dy = by - ay
    denom = dx * dx + dy * dy
    if denom == 0.0:
        return math.hypot(px - ax, py - ay)
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / denom))
    return math.hypot(px - (ax + t * dx), py - (ay + t * dy))


def _orbit_state(planet: list[float], angular_velocity: float, comet_ids: set[int]) -> tuple[float, float]:
    if int(planet[0]) in comet_ids or abs(angular_velocity) <= 1e-12:
        return 0.0, 0.0
    x = float(planet[2])
    y = float(planet[3])
    radius = float(planet[4])
    orbit_radius = math.hypot(x - CENTER, y - CENTER)
    if orbit_radius + radius >= ROTATION_RADIUS_LIMIT or orbit_radius <= 1e-9:
        return 0.0, 0.0
    return orbit_radius, math.atan2(y - CENTER, x - CENTER)


def _orbit_position(
    x: float,
    y: float,
    orbit_radius: float,
    theta0: float,
    angular_velocity: float,
    steps: int,
) -> tuple[float, float]:
    if orbit_radius <= 0.0 or steps == 0:
        return x, y
    theta = theta0 + angular_velocity * steps
    return CENTER + orbit_radius * math.cos(theta), CENTER + orbit_radius * math.sin(theta)


def _route_clear(
    source: list[float],
    target: list[float],
    solution: tuple[float, float, float, float],
    send: int,
    blockers: list[tuple[int, float, float, float, float, float]],
    angular_velocity: float,
) -> bool:
    speed = _fleet_speed(send)
    if speed <= 0.0:
        return False
    source_id = int(source[0])
    target_id = int(target[0])
    source_x = float(source[2])
    source_y = float(source[3])
    source_radius = float(source[4])
    angle, intercept_time, _target_x, _target_y = solution
    dir_x = math.cos(angle)
    dir_y = math.sin(angle)
    start_x = source_x + dir_x * (source_radius + 0.1)
    start_y = source_y + dir_y * (source_radius + 0.1)
    final_turn = max(1, int(math.ceil(intercept_time)))
    final_x = start_x + dir_x * speed * final_turn
    final_y = start_y + dir_y * speed * final_turn
    if not (0.0 <= start_x <= BOARD_SIZE and 0.0 <= start_y <= BOARD_SIZE):
        return False
    if not (0.0 <= final_x <= BOARD_SIZE and 0.0 <= final_y <= BOARD_SIZE):
        return False
    if _point_to_segment_distance(CENTER, CENTER, start_x, start_y, final_x, final_y) < SUN_RADIUS:
        return False

    moving: list[tuple[int, float, float, float, float, float]] = []
    for blocker in blockers:
        blocker_id, x, y, radius, orbit_radius, theta0 = blocker
        if blocker_id == target_id:
            continue
        if blocker_id == source_id:
            if orbit_radius > 0.0:
                moving.append(blocker)
            continue
        if orbit_radius > 0.0:
            moving.append(blocker)
            continue
        if _point_to_segment_distance(x, y, start_x, start_y, final_x, final_y) < radius:
            return False

    for turn in range(1, final_turn + 1):
        old_x = start_x + dir_x * speed * (turn - 1)
        old_y = start_y + dir_y * speed * (turn - 1)
        new_x = start_x + dir_x * speed * turn
        new_y = start_y + dir_y * speed * turn
        for blocker_id, x, y, radius, orbit_radius, theta0 in moving:
            bx, by = _orbit_position(x, y, orbit_radius, theta0, angular_velocity, turn - 1)
            if blocker_id != source_id:
                if _point_to_segment_distance(bx, by, old_x, old_y, new_x, new_y) < radius:
                    return False
            if turn < final_turn:
                nbx, nby = _orbit_position(x, y, orbit_radius, theta0, angular_velocity, turn)
                if _point_to_segment_distance(new_x, new_y, bx, by, nbx, nby) < radius:
                    return False
    return True


def _lead_angle(source: list[float], target: list[float], angular_velocity: float, send: int) -> float | None:
    source_x = float(source[2])
    source_y = float(source[3])
    source_radius = float(source[4])
    target_x = float(target[2])
    target_y = float(target[3])
    target_radius = float(target[4])
    solution = _lead_solution_from_point(
        source_x,
        source_y,
        target_x,
        target_y,
        target_radius,
        angular_velocity,
        send,
    )
    if solution is None:
        return None
    angle = solution[0]
    offset = max(0.0, source_radius + 0.1)
    if offset <= 0.0:
        return angle

    for _ in range(4):
        start_x = source_x + math.cos(angle) * offset
        start_y = source_y + math.sin(angle) * offset
        refined = _lead_solution_from_point(
            start_x,
            start_y,
            target_x,
            target_y,
            target_radius,
            angular_velocity,
            send,
        )
        if refined is None:
            return None
        next_angle = refined[0]
        if abs(math.atan2(math.sin(next_angle - angle), math.cos(next_angle - angle))) < 1e-6:
            return next_angle
        angle = next_angle
    return angle


def agent(obs: Any, *_args: Any) -> list[list]:
    player = int(_get(obs, "player", 0) or 0)
    planets = list(_get(obs, "planets", []) or [])
    angular_velocity = float(_get(obs, "angular_velocity", 0.0) or 0.0)
    comet_ids = {int(pid) for pid in (_get(obs, "comet_planet_ids", []) or [])}
    targets = [planet for planet in planets if int(planet[1]) != player]
    if not targets:
        return []
    blockers = [
        (
            int(planet[0]),
            float(planet[2]),
            float(planet[3]),
            float(planet[4]),
            *_orbit_state(planet, angular_velocity, comet_ids),
        )
        for planet in planets
    ]

    moves: list[list] = []
    for source in planets:
        if int(source[1]) != player:
            continue
        for target in sorted(targets, key=lambda planet: _distance_sq(source, planet)):
            ships_needed = int(target[5]) + 1
            if int(source[5]) < ships_needed:
                continue
            solution = _lead_solution_from_point(
                float(source[2]),
                float(source[3]),
                float(target[2]),
                float(target[3]),
                float(target[4]),
                angular_velocity,
                ships_needed,
            )
            angle = None if solution is None else solution[0]
            if angle is not None:
                for _ in range(4):
                    start_x = float(source[2]) + math.cos(angle) * (float(source[4]) + 0.1)
                    start_y = float(source[3]) + math.sin(angle) * (float(source[4]) + 0.1)
                    refined = _lead_solution_from_point(
                        start_x,
                        start_y,
                        float(target[2]),
                        float(target[3]),
                        float(target[4]),
                        angular_velocity,
                        ships_needed,
                    )
                    if refined is None:
                        solution = None
                        break
                    if abs(math.atan2(math.sin(refined[0] - angle), math.cos(refined[0] - angle))) < 1e-6:
                        solution = refined
                        break
                    angle = refined[0]
                    solution = refined
            if solution is None:
                continue
            if not _route_clear(source, target, solution, ships_needed, blockers, angular_velocity):
                continue
            moves.append([int(source[0]), float(solution[0]), ships_needed])
            break
    return moves


if __name__ == "__main__":
    from kaggle_environments import make  # type: ignore[import-not-found]

    env = make("orbit_wars", debug=True)
    env.run([agent, "random"])
    print([float(state.reward or 0.0) for state in env.steps[-1]])
