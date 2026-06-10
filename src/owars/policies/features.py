"""Feature encoding for the policy net.

Two parallel set-of-tokens streams:
  - planets: per-planet vector with positional, ownership, garrison /
    production signal, plus *orbit parameters* (current angle, radius,
    angular velocity) and a *direction-of-motion* unit vector. The model
    can project to any horizon it wants from these — we don't bake a
    fixed-horizon predicted position into the input.
  - fleets:  per-fleet vector with position, heading, ships, owner, the
    source planet's position (provenance), and optional intended-target
    metadata when the training env can preserve it.

Owner is encoded *seat-relative* with one slot per enemy ID `(owner -
player) mod 4`, so a 4-player FFA sees three stable enemy slots and a
2-player game sees only `enemy_0`. There is no "ally" slot — the
competition is FFA / 1v1, never team-based.

The encoder operates on padded tensors plus boolean masks. We pad to fixed
caps (`MAX_PLANETS`, `MAX_FLEETS`) for efficient batching during training;
inference handles arbitrary counts up to those caps.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from collections.abc import Sequence
from typing import Any

import numpy as np
import torch

from ..game import (
    BOARD_SIZE,
    CENTER,
    MAX_SHIP_SPEED,
    ROTATION_RADIUS_LIMIT,
    Fleet,
    Planet,
)
from ..game.observation import Observation
from ..game.physics import fleet_speed

MAX_PLANETS: int = 64
MAX_FLEETS: int = 384

PLANET_FEAT_DIM: int = 19
FLEET_FEAT_DIM: int = 20
GLOBAL_PLAYER_SLOTS: int = 4
GLOBAL_PLAYER_FEATS: int = 5
GLOBAL_NEUTRAL_FEATS: int = 3
GLOBAL_FEAT_DIM: int = (
    4 + GLOBAL_PLAYER_SLOTS * GLOBAL_PLAYER_FEATS + GLOBAL_NEUTRAL_FEATS
)

MAX_OMEGA: float = 0.05  # spec: ω ∈ [0.025, 0.05]
EPISODE_STEPS: int = 500
COMET_PERIOD_STEPS: int = 100
FIRST_COMET_STEP: int = 50
GLOBAL_PRODUCTION_SCALE: float = float(MAX_PLANETS * 5)
GLOBAL_SHIP_LOG_SCALE: float = 12.0


def _owner_onehot(
    owner: int, player: int, num_players: int
) -> tuple[float, float, float, float, float]:
    """Seat-relative owner one-hot: `[self, neutral, enemy_0, enemy_1, enemy_2]`.

    Enemy slots are indexed by `(owner - player) mod num_players - 1`. With
    the modulus matching seat count, both seats in a 2-player game see their
    opponent in `enemy_0` (symmetry preserved); in 4-player FFA the three
    enemy slots fill in turn-order.
    """
    if owner == -1:
        return (0.0, 1.0, 0.0, 0.0, 0.0)
    if owner == player:
        return (1.0, 0.0, 0.0, 0.0, 0.0)
    np_ = max(2, int(num_players))
    diff = (owner - player) % np_
    slot = diff - 1  # in {0, ..., np_-2}
    out = [0.0, 0.0, 0.0, 0.0, 0.0]
    if 0 <= slot <= 2:
        out[2 + slot] = 1.0
    return tuple(out)  # type: ignore[return-value]


def _planet_motion(
    p: Planet,
    angular_velocity: float,
    comet_motion_by_id: dict[int, tuple[float, float] | None],
) -> tuple[float, float, float, float, float, float, float]:
    """Return (cos_h, sin_h, speed_norm, orb_r_norm, omega_norm, is_orbiting, is_comet).

    For orbiters: heading is the unit tangent to the orbit, speed is r·|ω|.
    For comets: heading is the unit step direction along the precomputed
      path, speed is the cometSpeed (we infer it from the path step magnitude
      so per-game `cometSpeed` overrides cleanly).
    For static planets: heading=(0,0), speed=0, orbital params=0.
    """
    if p.id in comet_motion_by_id:
        step = comet_motion_by_id[p.id]
        if step is None:
            return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)
        dx, dy = step
        n = math.hypot(dx, dy)
        if n <= 0.0:
            return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)
        return (dx / n, dy / n, min(1.0, n / MAX_SHIP_SPEED), 0.0, 0.0, 0.0, 1.0)

    cx, cy = CENTER
    rx, ry = p.x - cx, p.y - cy
    orbital_radius = math.hypot(rx, ry)
    is_orbiting = (orbital_radius + p.radius) < ROTATION_RADIUS_LIMIT
    if is_orbiting and orbital_radius > 1e-9:
        # Velocity = ω · (-ry, rx). Sign of ω handles rotation direction.
        vx = -ry * angular_velocity
        vy = rx * angular_velocity
        speed = math.hypot(vx, vy)
        cos_h = vx / max(speed, 1e-9)
        sin_h = vy / max(speed, 1e-9)
        return (
            cos_h,
            sin_h,
            min(1.0, speed / MAX_SHIP_SPEED),
            orbital_radius / 50.0,
            abs(angular_velocity) / MAX_OMEGA,
            1.0,
            0.0,
        )
    return (0.0, 0.0, 0.0, orbital_radius / 50.0, 0.0, 0.0, 0.0)


def _planet_features(
    p: Planet,
    player: int,
    num_players: int,
    angular_velocity: float,
    comet_motion_by_id: dict[int, tuple[float, float] | None],
) -> list[float]:
    nx = (p.x - CENTER[0]) / BOARD_SIZE
    ny = (p.y - CENTER[1]) / BOARD_SIZE
    dist_to_sun = math.hypot(p.x - CENTER[0], p.y - CENTER[1]) / BOARD_SIZE
    cos_h, sin_h, sp, orb_r, om, is_orb, is_com = _planet_motion(
        p, angular_velocity, comet_motion_by_id
    )
    s, n_, e0, e1, e2 = _owner_onehot(p.owner, player, num_players)
    return [
        nx, ny, dist_to_sun, p.radius / 5.0,
        math.log1p(p.ships) / 8.0, p.production / 5.0,
        cos_h, sin_h, sp,
        orb_r, om,
        is_orb, is_com,
        s, n_, e0, e1, e2,
        1.0,  # planet-token marker
    ]


def _fleet_features(
    f: Fleet,
    player: int,
    num_players: int,
    planet_pos_by_id: dict[int, tuple[float, float]],
) -> list[float]:
    nx = (f.x - CENTER[0]) / BOARD_SIZE
    ny = (f.y - CENTER[1]) / BOARD_SIZE
    src = planet_pos_by_id.get(f.from_planet_id)
    if src is None:
        from_nx, from_ny, has_from = 0.0, 0.0, 0.0
    else:
        from_nx = (src[0] - CENTER[0]) / BOARD_SIZE
        from_ny = (src[1] - CENTER[1]) / BOARD_SIZE
        has_from = 1.0
    s, n_, e0, e1, e2 = _owner_onehot(f.owner, player, num_players)
    # Fleet's own speed is determined by its ship count via the official
    # log curve; encoding it explicitly is cheap and saves the model the
    # detour.
    sp = min(1.0, fleet_speed(f.ships) / MAX_SHIP_SPEED)
    return [
        nx, ny, math.cos(f.angle), math.sin(f.angle),
        math.log1p(f.ships) / 8.0,
        from_nx, from_ny, has_from,
        sp,
        0.0, 0.0, 0.0, 0.0, 0.0,
        s, n_, e0, e1, e2,
        0.0,  # planet-token marker (= fleet)
    ]


def _global_player_slot(owner: int, player: int, num_players: int) -> int | None:
    if owner < 0:
        return None
    if owner == player:
        return 0
    np_ = max(2, int(num_players))
    diff = (owner - player) % np_
    slot = diff
    return slot if 1 <= slot < GLOBAL_PLAYER_SLOTS else None


def _clip01(x: float) -> float:
    return min(1.0, max(0.0, float(x)))


def _global_features(
    step: int,
    player: int = 0,
    num_players: int = 2,
    planets: Sequence[Any] = (),
    fleets: Sequence[Any] = (),
) -> list[float]:
    step_norm = min(1.0, max(0.0, float(step) / float(EPISODE_STEPS)))
    remaining_norm = min(
        1.0,
        max(0.0, float(EPISODE_STEPS - int(step)) / float(EPISODE_STEPS)),
    )
    phase = ((int(step) - FIRST_COMET_STEP) % COMET_PERIOD_STEPS) / float(
        COMET_PERIOD_STEPS
    )
    angle = 2.0 * math.pi * phase
    player_stats = [[0.0, 0.0, 0.0, 0.0, 0.0] for _ in range(GLOBAL_PLAYER_SLOTS)]
    neutral_count = 0.0
    neutral_production = 0.0
    neutral_ships = 0.0

    for p in planets:
        owner = int(p.owner if hasattr(p, "owner") else p[1])
        ships = float(p.ships if hasattr(p, "ships") else p[5])
        production = float(p.production if hasattr(p, "production") else p[6])
        if owner < 0:
            neutral_count += 1.0
            neutral_production += production
            neutral_ships += ships
            continue
        slot = _global_player_slot(owner, player, num_players)
        if slot is None:
            continue
        player_stats[slot][0] += 1.0
        player_stats[slot][1] += production
        player_stats[slot][2] += ships

    for f in fleets:
        owner = int(f.owner if hasattr(f, "owner") else f[1])
        if owner < 0:
            continue
        slot = _global_player_slot(owner, player, num_players)
        if slot is None:
            continue
        ships = float(f.ships if hasattr(f, "ships") else f[6])
        player_stats[slot][3] += 1.0
        player_stats[slot][4] += ships

    out = [step_norm, remaining_norm, math.sin(angle), math.cos(angle)]
    for planet_count, production, planet_ships, fleet_count, fleet_ships in player_stats:
        out.extend(
            [
                _clip01(planet_count / float(MAX_PLANETS)),
                _clip01(production / GLOBAL_PRODUCTION_SCALE),
                _clip01(math.log1p(max(0.0, planet_ships)) / GLOBAL_SHIP_LOG_SCALE),
                _clip01(fleet_count / float(MAX_FLEETS)),
                _clip01(math.log1p(max(0.0, fleet_ships)) / GLOBAL_SHIP_LOG_SCALE),
            ]
        )
    out.extend(
        [
            _clip01(neutral_count / float(MAX_PLANETS)),
            _clip01(neutral_production / GLOBAL_PRODUCTION_SCALE),
            _clip01(math.log1p(max(0.0, neutral_ships)) / GLOBAL_SHIP_LOG_SCALE),
        ]
    )
    return out


def _comet_motion_by_id(o: Observation) -> dict[int, tuple[float, float] | None]:
    """For each comet planet ID, the next-step direction vector along its path.

    `o.comets` is a list of group dicts; we expect each to expose
    `planet_ids`, `paths` (list-of-paths, one per comet in the group), and
    `path_index` (current step). Robust to schema variation: missing keys
    yield `None` (which the feature builder treats as "unknown motion").
    """
    out: dict[int, tuple[float, float] | None] = {pid: None for pid in o.comet_planet_ids}
    for group in o.comets:
        ids = group.get("planet_ids") or []
        paths = group.get("paths") or []
        idx = group.get("path_index", 0)
        for k, pid in enumerate(ids):
            if k >= len(paths):
                continue
            path = paths[k]
            if not path or idx + 1 >= len(path):
                continue
            cur = path[idx]
            nxt = path[idx + 1]
            try:
                dx = float(nxt[0]) - float(cur[0])
                dy = float(nxt[1]) - float(cur[1])
            except (TypeError, IndexError, ValueError):
                continue
            out[int(pid)] = (dx, dy)
    return out


def _get_raw(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _comet_motion_by_id_raw(o: Any) -> dict[int, tuple[float, float] | None]:
    out: dict[int, tuple[float, float] | None] = {
        int(pid): None for pid in (_get_raw(o, "comet_planet_ids", []) or [])
    }
    for group in _get_raw(o, "comets", []) or []:
        ids = group.get("planet_ids") or []
        paths = group.get("paths") or []
        idx = group.get("path_index", 0)
        for k, pid in enumerate(ids):
            if k >= len(paths):
                continue
            path = paths[k]
            if len(path) == 0 or idx + 1 >= len(path):
                continue
            cur = path[idx]
            nxt = path[idx + 1]
            try:
                dx = float(nxt[0]) - float(cur[0])
                dy = float(nxt[1]) - float(cur[1])
            except (TypeError, IndexError, ValueError):
                continue
            out[int(pid)] = (dx, dy)
    return out


def _infer_num_players_raw(o: Any) -> int:
    explicit = _get_raw(o, "num_players", None)
    if explicit is not None:
        return max(2, int(explicit))
    candidates: list[int] = []
    initial_planets = _get_raw(o, "initial_planets", [])
    if initial_planets is None:
        initial_planets = []
    for p in initial_planets:
        owner = int(p[1])
        if owner >= 0:
            candidates.append(owner)
    if not candidates:
        planets = _get_raw(o, "planets", [])
        if planets is None:
            planets = []
        for p in planets:
            owner = int(p[1])
            if owner >= 0:
                candidates.append(owner)
    return max(2, max(candidates) + 1) if candidates else 2


def _planet_features_raw(
    p: Any,
    player: int,
    num_players: int,
    angular_velocity: float,
    comet_motion_by_id: dict[int, tuple[float, float] | None],
) -> list[float]:
    pid = int(p[0])
    owner = int(p[1])
    x = float(p[2])
    y = float(p[3])
    radius = float(p[4])
    ships = int(p[5])
    production = int(p[6])
    nx = (x - CENTER[0]) / BOARD_SIZE
    ny = (y - CENTER[1]) / BOARD_SIZE
    dist_to_sun = math.hypot(x - CENTER[0], y - CENTER[1]) / BOARD_SIZE
    if pid in comet_motion_by_id:
        step = comet_motion_by_id[pid]
        if step is None:
            cos_h, sin_h, sp, orb_r, om, is_orb, is_com = (
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0
            )
        else:
            dx, dy = step
            n = math.hypot(dx, dy)
            if n <= 0.0:
                cos_h, sin_h, sp, orb_r, om, is_orb, is_com = (
                    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0
                )
            else:
                cos_h, sin_h, sp, orb_r, om, is_orb, is_com = (
                    dx / n, dy / n, min(1.0, n / MAX_SHIP_SPEED), 0.0, 0.0, 0.0, 1.0
                )
    else:
        rx, ry = x - CENTER[0], y - CENTER[1]
        orbital_radius = math.hypot(rx, ry)
        is_orbiting = (orbital_radius + radius) < ROTATION_RADIUS_LIMIT
        if is_orbiting and orbital_radius > 1e-9:
            vx = -ry * angular_velocity
            vy = rx * angular_velocity
            speed = math.hypot(vx, vy)
            cos_h, sin_h, sp, orb_r, om, is_orb, is_com = (
                vx / max(speed, 1e-9),
                vy / max(speed, 1e-9),
                min(1.0, speed / MAX_SHIP_SPEED),
                orbital_radius / 50.0,
                abs(angular_velocity) / MAX_OMEGA,
                1.0,
                0.0,
            )
        else:
            cos_h, sin_h, sp, orb_r, om, is_orb, is_com = (
                0.0, 0.0, 0.0, orbital_radius / 50.0, 0.0, 0.0, 0.0
            )
    s, n_, e0, e1, e2 = _owner_onehot(owner, player, num_players)
    return [
        nx, ny, dist_to_sun, radius / 5.0,
        math.log1p(ships) / 8.0, production / 5.0,
        cos_h, sin_h, sp,
        orb_r, om,
        is_orb, is_com,
        s, n_, e0, e1, e2,
        1.0,
    ]


def _fleet_features_raw(
    f: Any,
    player: int,
    num_players: int,
    planet_pos_by_id: dict[int, tuple[float, float]],
    target_meta: tuple[int, float, float, float] | None = None,
) -> list[float]:
    owner = int(f[1])
    x = float(f[2])
    y = float(f[3])
    angle = float(f[4])
    from_planet_id = int(f[5])
    ships = int(f[6])
    if target_meta is not None:
        target_id, eta, target_x, target_y = target_meta
        has_target = 1.0 if target_id >= 0 else 0.0
        target_id_norm = target_id / 128.0 if target_id >= 0 else 0.0
        eta_norm = min(1.0, max(0.0, eta / 500.0)) if target_id >= 0 else 0.0
        target_nx = (target_x - CENTER[0]) / BOARD_SIZE if target_id >= 0 else 0.0
        target_ny = (target_y - CENTER[1]) / BOARD_SIZE if target_id >= 0 else 0.0
    else:
        target_id_norm, eta_norm, target_nx, target_ny, has_target = (
            0.0, 0.0, 0.0, 0.0, 0.0
        )
    src = planet_pos_by_id.get(from_planet_id)
    if src is None:
        from_nx, from_ny, has_from = 0.0, 0.0, 0.0
    else:
        from_nx = (src[0] - CENTER[0]) / BOARD_SIZE
        from_ny = (src[1] - CENTER[1]) / BOARD_SIZE
        has_from = 1.0
    s, n_, e0, e1, e2 = _owner_onehot(owner, player, num_players)
    sp = min(1.0, fleet_speed(ships) / MAX_SHIP_SPEED)
    return [
        (x - CENTER[0]) / BOARD_SIZE,
        (y - CENTER[1]) / BOARD_SIZE,
        math.cos(angle),
        math.sin(angle),
        math.log1p(ships) / 8.0,
        from_nx, from_ny, has_from,
        sp,
        target_id_norm, eta_norm, target_nx, target_ny, has_target,
        s, n_, e0, e1, e2,
        0.0,
    ]


def _fleet_target_metadata_raw(
    o: Any,
) -> dict[int, tuple[int, float, float, float]]:
    raw = _get_raw(o, "fleet_targets", None)
    if raw is None:
        raw = _get_raw(o, "fleet_target_metadata", None)
    if raw is None:
        return {}

    out: dict[int, tuple[int, float, float, float]] = {}
    if isinstance(raw, dict):
        items = raw.items()
    else:
        items = []
        for row in raw or []:
            try:
                items.append((row[0], row[1:]))
            except (TypeError, IndexError):
                continue

    for fleet_id, meta in items:
        try:
            fid = int(fleet_id)
            if isinstance(meta, dict):
                target_id = int(meta.get("target_id", -1))
                eta = float(meta.get("eta", 0.0))
                target_x = float(meta.get("target_x", meta.get("x", 0.0)))
                target_y = float(meta.get("target_y", meta.get("y", 0.0)))
            else:
                target_id = int(meta[0])
                eta = float(meta[1])
                target_x = float(meta[2])
                target_y = float(meta[3])
        except (TypeError, IndexError, ValueError):
            continue
        out[fid] = (target_id, eta, target_x, target_y)
    return out


@dataclass
class EncodedObs:
    planet_feats: torch.Tensor      # [P_max, planet_dim]
    planet_mask: torch.Tensor       # [P_max] bool — True = real planet
    planet_owned_mask: torch.Tensor # [P_max] bool — True = owned by `player`
    planet_ids: torch.Tensor        # [P_max] long — planet `id`, -1 if pad
    planet_garrison: torch.Tensor   # [P_max] float — current ships
    fleet_feats: torch.Tensor       # [F_max, fleet_dim]
    fleet_mask: torch.Tensor        # [F_max] bool
    global_feats: torch.Tensor | None = None  # [global_dim] or [B, global_dim]

    def to(self, device: str | torch.device) -> EncodedObs:
        return EncodedObs(
            planet_feats=self.planet_feats.to(device),
            planet_mask=self.planet_mask.to(device),
            planet_owned_mask=self.planet_owned_mask.to(device),
            planet_ids=self.planet_ids.to(device),
            planet_garrison=self.planet_garrison.to(device),
            fleet_feats=self.fleet_feats.to(device),
            fleet_mask=self.fleet_mask.to(device),
            global_feats=None
            if self.global_feats is None
            else self.global_feats.to(device),
        )


def _infer_num_players(o: Observation) -> int:
    """Best-effort seat count for owner-onehot indexing.

    The kaggle obs doesn't expose `num_players` directly, but at game start
    each seat owns one home planet, so `max(initial_planets owners) + 1` is
    the seat count. Falls back to current-state owners if `initial_planets`
    is empty (e.g., synthetic test obs).
    """
    candidates: list[int] = []
    for p in o.initial_planets:
        if p.owner >= 0:
            candidates.append(p.owner)
    if not candidates:
        for p in o.planets:
            if p.owner >= 0:
                candidates.append(p.owner)
    if not candidates:
        return 2
    return max(2, max(candidates) + 1)


def _fill_encoded_arrays(
    o: Observation,
    g_feats: np.ndarray,
    p_feats: np.ndarray,
    p_mask: np.ndarray,
    p_owned: np.ndarray,
    p_ids: np.ndarray,
    p_gar: np.ndarray,
    f_feats: np.ndarray,
    f_mask: np.ndarray,
    row: int | None = None,
) -> None:
    if row is None:
        g_feats_r = g_feats
        p_feats_r = p_feats
        p_mask_r = p_mask
        p_owned_r = p_owned
        p_ids_r = p_ids
        p_gar_r = p_gar
        f_feats_r = f_feats
        f_mask_r = f_mask
    else:
        g_feats_r = g_feats[row]
        p_feats_r = p_feats[row]
        p_mask_r = p_mask[row]
        p_owned_r = p_owned[row]
        p_ids_r = p_ids[row]
        p_gar_r = p_gar[row]
        f_feats_r = f_feats[row]
        f_mask_r = f_mask[row]

    comet_motion = _comet_motion_by_id(o)
    planet_pos = {p.id: (p.x, p.y) for p in o.planets}
    num_players = _infer_num_players(o)
    g_feats_r[:] = _global_features(
        o.step,
        o.player,
        num_players,
        o.planets,
        o.fleets,
    )

    for i, p in enumerate(o.planets[:MAX_PLANETS]):
        p_feats_r[i] = _planet_features(
            p, o.player, num_players, o.angular_velocity, comet_motion
        )
        p_mask_r[i] = True
        p_owned_r[i] = p.owner == o.player
        p_ids_r[i] = p.id
        p_gar_r[i] = p.ships

    for j, f in enumerate(o.fleets[:MAX_FLEETS]):
        f_feats_r[j] = _fleet_features(f, o.player, num_players, planet_pos)
        f_mask_r[j] = True


def _fill_encoded_arrays_raw(
    o: Any,
    g_feats: np.ndarray,
    p_feats: np.ndarray,
    p_mask: np.ndarray,
    p_owned: np.ndarray,
    p_ids: np.ndarray,
    p_gar: np.ndarray,
    f_feats: np.ndarray,
    f_mask: np.ndarray,
    row: int | None = None,
) -> None:
    if row is None:
        g_feats_r = g_feats
        p_feats_r = p_feats
        p_mask_r = p_mask
        p_owned_r = p_owned
        p_ids_r = p_ids
        p_gar_r = p_gar
        f_feats_r = f_feats
        f_mask_r = f_mask
    else:
        g_feats_r = g_feats[row]
        p_feats_r = p_feats[row]
        p_mask_r = p_mask[row]
        p_owned_r = p_owned[row]
        p_ids_r = p_ids[row]
        p_gar_r = p_gar[row]
        f_feats_r = f_feats[row]
        f_mask_r = f_mask[row]

    planets = _get_raw(o, "planets", [])
    fleets = _get_raw(o, "fleets", [])
    if planets is None:
        planets = []
    if fleets is None:
        fleets = []
    player = int(_get_raw(o, "player", 0) or 0)
    step = int(_get_raw(o, "step", 0) or 0)
    angular_velocity = float(_get_raw(o, "angular_velocity", 0.0) or 0.0)
    comet_motion = _comet_motion_by_id_raw(o)
    planet_pos = {int(p[0]): (float(p[2]), float(p[3])) for p in planets}
    fleet_targets = _fleet_target_metadata_raw(o)
    num_players = _infer_num_players_raw(o)
    g_feats_r[:] = _global_features(
        step,
        player,
        num_players,
        planets,
        fleets,
    )

    for i, p in enumerate(planets[:MAX_PLANETS]):
        p_feats_r[i] = _planet_features_raw(
            p, player, num_players, angular_velocity, comet_motion
        )
        p_mask_r[i] = True
        p_owned_r[i] = int(p[1]) == player
        p_ids_r[i] = int(p[0])
        p_gar_r[i] = int(p[5])

    for j, f in enumerate(fleets[:MAX_FLEETS]):
        target_meta = fleet_targets.get(int(f[0])) if int(f[1]) == player else None
        f_feats_r[j] = _fleet_features_raw(
            f,
            player,
            num_players,
            planet_pos,
            target_meta,
        )
        f_mask_r[j] = True


def _tensor_from_numpy(
    array: np.ndarray,
    device: str | torch.device,
    *,
    pin_memory: bool,
) -> torch.Tensor:
    t = torch.from_numpy(array)
    target = torch.device(device)
    if pin_memory and target.type == "cuda":
        t = t.pin_memory()
        return t.to(target, non_blocking=True)
    return t.to(target)


def encode_observation(
    o: Observation,
    device: str | torch.device = "cpu",
    *,
    pin_memory: bool = False,
) -> EncodedObs:
    g_feats = np.zeros(GLOBAL_FEAT_DIM, dtype=np.float32)
    p_feats = np.zeros((MAX_PLANETS, PLANET_FEAT_DIM), dtype=np.float32)
    p_mask = np.zeros(MAX_PLANETS, dtype=bool)
    p_owned = np.zeros(MAX_PLANETS, dtype=bool)
    p_ids = -np.ones(MAX_PLANETS, dtype=np.int64)
    p_gar = np.zeros(MAX_PLANETS, dtype=np.float32)
    f_feats = np.zeros((MAX_FLEETS, FLEET_FEAT_DIM), dtype=np.float32)
    f_mask = np.zeros(MAX_FLEETS, dtype=bool)

    _fill_encoded_arrays(
        o,
        g_feats,
        p_feats,
        p_mask,
        p_owned,
        p_ids,
        p_gar,
        f_feats,
        f_mask,
    )

    return EncodedObs(
        global_feats=_tensor_from_numpy(g_feats, device, pin_memory=pin_memory),
        planet_feats=_tensor_from_numpy(p_feats, device, pin_memory=pin_memory),
        planet_mask=_tensor_from_numpy(p_mask, device, pin_memory=pin_memory),
        planet_owned_mask=_tensor_from_numpy(p_owned, device, pin_memory=pin_memory),
        planet_ids=_tensor_from_numpy(p_ids, device, pin_memory=pin_memory),
        planet_garrison=_tensor_from_numpy(p_gar, device, pin_memory=pin_memory),
        fleet_feats=_tensor_from_numpy(f_feats, device, pin_memory=pin_memory),
        fleet_mask=_tensor_from_numpy(f_mask, device, pin_memory=pin_memory),
    )


def encode_observations(
    observations: list[Observation],
    device: str | torch.device = "cpu",
    *,
    pin_memory: bool = False,
) -> EncodedObs:
    """Encode observations directly into one batched tensor set.

    This keeps rollout from doing many tiny host-to-device copies before
    stacking. The fixed caps stay unchanged, so compiled training forwards
    still see stable shapes.
    """
    b = len(observations)
    g_feats = np.zeros((b, GLOBAL_FEAT_DIM), dtype=np.float32)
    p_feats = np.zeros((b, MAX_PLANETS, PLANET_FEAT_DIM), dtype=np.float32)
    p_mask = np.zeros((b, MAX_PLANETS), dtype=bool)
    p_owned = np.zeros((b, MAX_PLANETS), dtype=bool)
    p_ids = -np.ones((b, MAX_PLANETS), dtype=np.int64)
    p_gar = np.zeros((b, MAX_PLANETS), dtype=np.float32)
    f_feats = np.zeros((b, MAX_FLEETS, FLEET_FEAT_DIM), dtype=np.float32)
    f_mask = np.zeros((b, MAX_FLEETS), dtype=bool)

    for row, o in enumerate(observations):
        _fill_encoded_arrays(
            o,
            g_feats,
            p_feats,
            p_mask,
            p_owned,
            p_ids,
            p_gar,
            f_feats,
            f_mask,
            row=row,
        )

    return EncodedObs(
        global_feats=_tensor_from_numpy(g_feats, device, pin_memory=pin_memory),
        planet_feats=_tensor_from_numpy(p_feats, device, pin_memory=pin_memory),
        planet_mask=_tensor_from_numpy(p_mask, device, pin_memory=pin_memory),
        planet_owned_mask=_tensor_from_numpy(p_owned, device, pin_memory=pin_memory),
        planet_ids=_tensor_from_numpy(p_ids, device, pin_memory=pin_memory),
        planet_garrison=_tensor_from_numpy(p_gar, device, pin_memory=pin_memory),
        fleet_feats=_tensor_from_numpy(f_feats, device, pin_memory=pin_memory),
        fleet_mask=_tensor_from_numpy(f_mask, device, pin_memory=pin_memory),
    )


def encode_raw_observations(
    observations: list[Any],
    device: str | torch.device = "cpu",
    *,
    pin_memory: bool = False,
) -> EncodedObs:
    """Encode Kaggle-style observation dicts directly.

    This is the rollout hot path. It avoids constructing `Observation`,
    `Planet`, and `Fleet` Python objects for every alive seat at every env
    tick while preserving the exact tensor contract of `encode_observations`.
    """
    b = len(observations)
    g_feats = np.zeros((b, GLOBAL_FEAT_DIM), dtype=np.float32)
    p_feats = np.zeros((b, MAX_PLANETS, PLANET_FEAT_DIM), dtype=np.float32)
    p_mask = np.zeros((b, MAX_PLANETS), dtype=bool)
    p_owned = np.zeros((b, MAX_PLANETS), dtype=bool)
    p_ids = -np.ones((b, MAX_PLANETS), dtype=np.int64)
    p_gar = np.zeros((b, MAX_PLANETS), dtype=np.float32)
    f_feats = np.zeros((b, MAX_FLEETS, FLEET_FEAT_DIM), dtype=np.float32)
    f_mask = np.zeros((b, MAX_FLEETS), dtype=bool)

    for row, o in enumerate(observations):
        _fill_encoded_arrays_raw(
            o,
            g_feats,
            p_feats,
            p_mask,
            p_owned,
            p_ids,
            p_gar,
            f_feats,
            f_mask,
            row=row,
        )

    return EncodedObs(
        global_feats=_tensor_from_numpy(g_feats, device, pin_memory=pin_memory),
        planet_feats=_tensor_from_numpy(p_feats, device, pin_memory=pin_memory),
        planet_mask=_tensor_from_numpy(p_mask, device, pin_memory=pin_memory),
        planet_owned_mask=_tensor_from_numpy(p_owned, device, pin_memory=pin_memory),
        planet_ids=_tensor_from_numpy(p_ids, device, pin_memory=pin_memory),
        planet_garrison=_tensor_from_numpy(p_gar, device, pin_memory=pin_memory),
        fleet_feats=_tensor_from_numpy(f_feats, device, pin_memory=pin_memory),
        fleet_mask=_tensor_from_numpy(f_mask, device, pin_memory=pin_memory),
    )


def unbind_encoded(feats: EncodedObs) -> list[EncodedObs]:
    """Return batch-row views as unbatched `EncodedObs` objects."""
    if feats.planet_feats.dim() == 2:
        return [feats]
    return [
        EncodedObs(
            global_feats=None if feats.global_feats is None else feats.global_feats[i],
            planet_feats=feats.planet_feats[i],
            planet_mask=feats.planet_mask[i],
            planet_owned_mask=feats.planet_owned_mask[i],
            planet_ids=feats.planet_ids[i],
            planet_garrison=feats.planet_garrison[i],
            fleet_feats=feats.fleet_feats[i],
            fleet_mask=feats.fleet_mask[i],
        )
        for i in range(feats.planet_feats.shape[0])
    ]


def select_encoded(feats: EncodedObs, index: int, *, clone: bool = False) -> EncodedObs:
    """Return one batch row, optionally cloned to avoid retaining the full batch."""
    if feats.planet_feats.dim() == 2:
        return feats

    def row(t: torch.Tensor) -> torch.Tensor:
        out = t[index]
        return out.detach().clone() if clone else out

    return EncodedObs(
        global_feats=None if feats.global_feats is None else row(feats.global_feats),
        planet_feats=row(feats.planet_feats),
        planet_mask=row(feats.planet_mask),
        planet_owned_mask=row(feats.planet_owned_mask),
        planet_ids=row(feats.planet_ids),
        planet_garrison=row(feats.planet_garrison),
        fleet_feats=row(feats.fleet_feats),
        fleet_mask=row(feats.fleet_mask),
    )


def stack_encoded(feats_list: list[EncodedObs]) -> EncodedObs:
    """Stack a list of unbatched `EncodedObs` into a batched one.

    Each input has tensors shaped `[P, ...]` / `[P]` / `[F, ...]` / `[F]`;
    output tensors gain a leading batch dim. Used by the vectorized
    rollout to produce one big batch per env-step.
    """
    gf, pf, pm, pom, pid, pg, ff, fm = [], [], [], [], [], [], [], []
    for f in feats_list:
        gf.append(f.global_feats)
        pf.append(f.planet_feats)
        pm.append(f.planet_mask)
        pom.append(f.planet_owned_mask)
        pid.append(f.planet_ids)
        pg.append(f.planet_garrison)
        ff.append(f.fleet_feats)
        fm.append(f.fleet_mask)
    return EncodedObs(
        planet_feats=torch.stack(pf),
        planet_mask=torch.stack(pm),
        planet_owned_mask=torch.stack(pom),
        planet_ids=torch.stack(pid),
        planet_garrison=torch.stack(pg),
        fleet_feats=torch.stack(ff),
        fleet_mask=torch.stack(fm),
        global_feats=None if any(g is None for g in gf) else torch.stack(gf),
    )
