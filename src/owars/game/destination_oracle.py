"""Exact fleet-destination oracle over parsed observations.

Python twin of `rust/owars_env/src/oracle.rs::infer_fleet_destinations_reference`:
a destination-only rollout that mirrors the simulator's per-turn order
(expire comets -> move fleets [board, sun, planets in vector order] ->
move planets/comets -> sweep by mover order) using the simulator's own
float expressions, so results are bit-identical to the Rust oracle on the
same platform. Rust is the rollout hot path; this module is the submission
and test safety path, so it favors exactness over speed (it is still
vectorized over fleets with NumPy).

Semantics pinned here (shared with the Rust oracle):

- A hit requires `point_to_segment_distance(...) < radius` with the
  simulator's sqrt-based predicate; tangency is not a hit.
- Within a turn the precedence is board exit, then sun, then pre-move
  planets in vector order, then moving-planet sweeps in mover order
  (non-comet movers in vector order, then comets in group order).
- Fleet positions accumulate per turn (`pos += delta`), never `start +
  n * delta`, matching simulator rounding.
- A comet spawned at a future step is unobservable: from the first turn
  it could touch a fleet, surviving fleets are `STATUS_UNKNOWN`. Board,
  sun, and pre-move hits on that boundary turn still resolve exactly.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from .observation import Observation
from .types import (
    BOARD_SIZE,
    CENTER,
    COMET_SPAWN_STEPS,
    MAX_SHIP_SPEED,
    ROTATION_RADIUS_LIMIT,
    SUN_RADIUS,
)

STATUS_NONE = 0
STATUS_PLANET = 1
STATUS_BOARD = 2
STATUS_SUN = 3
STATUS_HORIZON = 4
STATUS_UNKNOWN = 5

HORIZON_CAP = 600

# Same literal as `LOG_1000` in rust/owars_env/src/core.rs.
_LOG_1000 = 6.907755278982137
_CX, _CY = CENTER


def fleet_step_speed(ships: int, ship_speed: float = MAX_SHIP_SPEED) -> float:
    """Simulator fleet speed, including its clamps (twin of `fleet_step_speed`)."""
    s = float(max(int(ships), 1))
    normalized = max(math.log(s) / _LOG_1000, 0.0)
    return min(1.0 + (ship_speed - 1.0) * normalized**1.5, ship_speed)


def inference_horizon(step: int, episode_steps: int) -> int:
    """Lookahead in future turns; step `episode_steps - 1` is the last that moves fleets."""
    return min(HORIZON_CAP, max(episode_steps - 1 - step, 0))


def _unknown_comet_turn(step: int, horizon: int) -> int | None:
    """First future turn whose outcome can depend on a not-yet-spawned comet."""
    for spawn_step in COMET_SPAWN_STEPS:
        spawn_turn = spawn_step - step
        if spawn_turn >= 1:
            turn = spawn_turn + 1
            return turn if turn <= horizon else None
    return None


def _segment_point_distances(
    px: np.ndarray,
    py: np.ndarray,
    old: np.ndarray,
    new: np.ndarray,
) -> np.ndarray:
    """`point_to_segment_distance` from core.rs, broadcast over points x segments.

    `px`/`py` broadcast against per-segment `old`/`new` rows of shape [..., 2].
    Every elementwise op mirrors the Rust expression order, so each entry is
    the exact f64 the simulator would compute.
    """
    seg_x = new[..., 0] - old[..., 0]
    seg_y = new[..., 1] - old[..., 1]
    l2 = seg_x * seg_x + seg_y * seg_y
    rel_x = px - old[..., 0]
    rel_y = py - old[..., 1]
    with np.errstate(divide="ignore", invalid="ignore"):
        raw_t = (rel_x * seg_x + rel_y * seg_y) / l2
        t = np.clip(raw_t, 0.0, 1.0)
        dx = px - (old[..., 0] + t * seg_x)
        dy = py - (old[..., 1] + t * seg_y)
        moving_dist = np.sqrt(dx * dx + dy * dy)
    degenerate_dist = np.sqrt(rel_x * rel_x + rel_y * rel_y)
    return np.where(l2 == 0.0, degenerate_dist, moving_dist)


@dataclass
class _PlanetState:
    id: int
    x: float
    y: float
    radius: float
    row: int


@dataclass
class _CometGroup:
    planet_ids: list[int]
    paths: list[list[tuple[float, float]]]
    path_index: int


def _parse_comet_groups(comets: list[dict[str, Any]]) -> list[_CometGroup]:
    groups: list[_CometGroup] = []
    for group in comets:
        planet_ids = [int(pid) for pid in group.get("planet_ids") or []]
        paths = [
            [(float(point[0]), float(point[1])) for point in path if len(point) >= 2]
            for path in group.get("paths") or []
        ]
        raw_index = group.get("path_index")
        path_index = -1 if raw_index is None else int(raw_index)
        groups.append(_CometGroup(planet_ids=planet_ids, paths=paths, path_index=path_index))
    return groups


def _remove_comets(
    planets: list[_PlanetState],
    comets: list[_CometGroup],
    expired: list[int],
) -> list[_PlanetState]:
    """Twin of `remove_ref_comets`: drop expired ids, never reordering survivors."""
    if not expired:
        return planets
    gone = set(expired)
    planets = [p for p in planets if p.id not in gone]
    for group in comets:
        new_ids: list[int] = []
        new_paths: list[list[tuple[float, float]]] = []
        for idx, pid in enumerate(group.planet_ids):
            if pid in gone:
                continue
            new_ids.append(pid)
            if idx < len(group.paths):
                new_paths.append(group.paths[idx])
        group.planet_ids = new_ids
        group.paths = new_paths
    comets[:] = [group for group in comets if group.planet_ids]
    return planets


def infer_fleet_destinations(
    obs: Observation,
    *,
    episode_steps: int = 500,
    ship_speed: float = MAX_SHIP_SPEED,
    max_fleets: int | None = None,
    done: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Resolve every fleet's destination exactly.

    Returns `(dest_idx, eta, status)` arrays of length
    `len(obs.fleets)` unless `max_fleets` is provided; `dest_idx` is the row index into
    `obs.planets` (`-1` for non-planet outcomes), `eta` the number of future
    turns until the event, `status` one of the `STATUS_*` constants.
    """
    n_fleets = len(obs.fleets) if max_fleets is None else min(len(obs.fleets), max_fleets)
    dest = np.full(n_fleets, -1, dtype=np.int64)
    eta = np.zeros(n_fleets, dtype=np.float64)
    status = np.full(n_fleets, STATUS_NONE, dtype=np.int64)
    if n_fleets == 0 or done:
        return dest, eta, status
    horizon = inference_horizon(obs.step, episode_steps)
    if horizon == 0:
        status.fill(STATUS_HORIZON)
        return dest, eta, status
    unknown_turn = _unknown_comet_turn(obs.step, horizon)

    # The Rust binding falls back to current planets when the observation has
    # no initial_planets; mirror it so both paths rotate the same planets.
    initials = {p.id: (float(p.x), float(p.y)) for p in obs.initial_planets or obs.planets}
    planets = [
        _PlanetState(id=int(p.id), x=float(p.x), y=float(p.y), radius=float(p.radius), row=row)
        for row, p in enumerate(obs.planets)
    ]
    # Sweep hits report the simulator's row-by-id lookup (last duplicate id
    # wins), while pre-move hits report the hit instance's own row.
    row_by_id = {planet.id: planet.row for planet in planets}
    comets = _parse_comet_groups(obs.comets)

    pos = np.zeros((n_fleets, 2), dtype=np.float64)
    delta = np.zeros((n_fleets, 2), dtype=np.float64)
    alive = np.zeros(n_fleets, dtype=bool)
    for slot, fleet in enumerate(obs.fleets[:max_fleets]):
        # math.cos/sin raise on +/-inf where Rust yields NaN; both end up
        # skipping the fleet, so guard the angle first.
        if not math.isfinite(fleet.angle):
            continue
        speed = fleet_step_speed(fleet.ships, ship_speed)
        dx = math.cos(fleet.angle) * speed
        dy = math.sin(fleet.angle) * speed
        if not (speed > 0.0) or not all(map(math.isfinite, (dx, dy, fleet.x, fleet.y))):
            continue
        pos[slot] = (fleet.x, fleet.y)
        delta[slot] = (dx, dy)
        alive[slot] = True
        eta[slot] = float(horizon)
        status[slot] = STATUS_HORIZON

    for turn in range(1, horizon + 1):
        if not alive.any():
            break

        # Mirror remove_expired_comets_before_launch (pre-increment index).
        expired: list[int] = []
        for group in comets:
            path_idx = max(group.path_index, 0)
            for idx, pid in enumerate(group.planet_ids):
                if idx < len(group.paths) and path_idx >= len(group.paths[idx]):
                    expired.append(pid)
        planets = _remove_comets(planets, comets, expired)

        # Mirror move_fleets: board, then sun, then planets in vector order,
        # against positions snapshotted before this turn's planet motion.
        old = pos.copy()
        pos += delta
        moved = np.flatnonzero(alive)
        out_of_board = (
            (pos[moved, 0] < 0.0)
            | (pos[moved, 0] > BOARD_SIZE)
            | (pos[moved, 1] < 0.0)
            | (pos[moved, 1] > BOARD_SIZE)
        )
        hit_slots = moved[out_of_board]
        dest[hit_slots] = -1
        eta[hit_slots] = float(turn)
        status[hit_slots] = STATUS_BOARD
        alive[hit_slots] = False

        moved = np.flatnonzero(alive)
        if moved.size:
            sun_dist = _segment_point_distances(_CX, _CY, old[moved], pos[moved])
            hit_slots = moved[sun_dist < SUN_RADIUS]
            dest[hit_slots] = -1
            eta[hit_slots] = float(turn)
            status[hit_slots] = STATUS_SUN
            alive[hit_slots] = False

        moved = np.flatnonzero(alive)
        if moved.size and planets:
            px = np.array([[p.x] for p in planets])
            py = np.array([[p.y] for p in planets])
            radius = np.array([[p.radius] for p in planets])
            dist = _segment_point_distances(px, py, old[moved], pos[moved])
            hits = dist < radius
            any_hit = hits.any(axis=0)
            first_planet = hits.argmax(axis=0)
            rows = np.array([p.row for p in planets], dtype=np.int64)
            hit_slots = moved[any_hit]
            dest[hit_slots] = rows[first_planet[any_hit]]
            eta[hit_slots] = float(turn)
            status[hit_slots] = STATUS_PLANET
            alive[hit_slots] = False

        # From the first unobserved comet spawn boundary on, only board /
        # sun / lower-order pre-move outcomes (handled above) are exact.
        if unknown_turn == turn:
            dest[alive] = -1
            eta[alive] = float(turn)
            status[alive] = STATUS_UNKNOWN
            return dest, eta, status

        # Mirror move_planets_and_sweep + move_comets: non-comet movers in
        # vector order first, then comets in group order.
        movers: list[tuple[float, tuple[float, float], tuple[float, float], int]] = []
        comet_ids = {pid for group in comets for pid in group.planet_ids}
        for planet in planets:
            if planet.id in comet_ids:
                continue
            initial = initials.get(planet.id)
            if initial is None:
                continue
            dx = initial[0] - _CX
            dy = initial[1] - _CY
            orbital_radius = math.sqrt(dx * dx + dy * dy)
            old_pos = (planet.x, planet.y)
            if orbital_radius + planet.radius < ROTATION_RADIUS_LIMIT:
                angle = math.atan2(dy, dx) + obs.angular_velocity * float(obs.step + turn - 1)
                if math.isinf(angle):
                    # Rust cos(inf) is NaN; math.cos(inf) raises instead.
                    planet.x = math.nan
                    planet.y = math.nan
                else:
                    planet.x = _CX + orbital_radius * math.cos(angle)
                    planet.y = _CY + orbital_radius * math.sin(angle)
            if old_pos != (planet.x, planet.y):
                movers.append((planet.radius, old_pos, (planet.x, planet.y), row_by_id[planet.id]))

        expired = []
        planets_by_id: dict[int, _PlanetState] = {}
        for planet in planets:
            # First duplicate wins, matching the simulator's linear find.
            planets_by_id.setdefault(planet.id, planet)
        for group in comets:
            group.path_index += 1
            path_idx = max(group.path_index, 0)
            for idx, pid in enumerate(group.planet_ids):
                if idx >= len(group.paths):
                    expired.append(pid)
                    continue
                planet = planets_by_id.get(pid)
                if planet is None:
                    continue
                path = group.paths[idx]
                if path_idx >= len(path):
                    expired.append(pid)
                    continue
                old_pos = (planet.x, planet.y)
                planet.x, planet.y = path[path_idx]
                if old_pos[0] >= 0.0 and old_pos != (planet.x, planet.y):
                    movers.append(
                        (planet.radius, old_pos, (planet.x, planet.y), row_by_id[planet.id])
                    )
        planets = _remove_comets(planets, comets, expired)

        for radius, old_pos, new_pos, row in movers:
            moved = np.flatnonzero(alive)
            if not moved.size:
                break
            seg_old = np.array(old_pos)
            seg_new = np.array(new_pos)
            dist = _segment_point_distances(pos[moved, 0], pos[moved, 1], seg_old, seg_new)
            hit_slots = moved[dist < radius]
            dest[hit_slots] = row
            eta[hit_slots] = float(turn)
            status[hit_slots] = STATUS_PLANET
            alive[hit_slots] = False

    return dest, eta, status
