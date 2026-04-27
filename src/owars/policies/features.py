"""Feature encoding for the policy net.

Two parallel set-of-tokens streams:
  - planets: per-planet vector with positional, ownership, garrison /
    production signal, plus *orbit parameters* (current angle, radius,
    angular velocity) and a *direction-of-motion* unit vector. The model
    can project to any horizon it wants from these — we don't bake a
    fixed-horizon predicted position into the input.
  - fleets:  per-fleet vector with position, heading, ships, owner, and
    the source planet's position (provenance — "where did this come
    from").

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

import numpy as np
import torch

from ..game import (
    BOARD_SIZE,
    CENTER,
    Fleet,
    MAX_SHIP_SPEED,
    Planet,
    ROTATION_RADIUS_LIMIT,
)
from ..game.observation import Observation
from ..game.physics import fleet_speed

MAX_PLANETS: int = 64
MAX_FLEETS: int = 384

PLANET_FEAT_DIM: int = 19
FLEET_FEAT_DIM: int = 15

MAX_OMEGA: float = 0.05  # spec: ω ∈ [0.025, 0.05]


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
        s, n_, e0, e1, e2,
        0.0,  # planet-token marker (= fleet)
    ]


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


@dataclass
class EncodedObs:
    planet_feats: torch.Tensor      # [P_max, planet_dim]
    planet_mask: torch.Tensor       # [P_max] bool — True = real planet
    planet_owned_mask: torch.Tensor # [P_max] bool — True = owned by `player`
    planet_ids: torch.Tensor        # [P_max] long — planet `id`, -1 if pad
    planet_garrison: torch.Tensor   # [P_max] float — current ships
    fleet_feats: torch.Tensor       # [F_max, fleet_dim]
    fleet_mask: torch.Tensor        # [F_max] bool

    def to(self, device: str | torch.device) -> "EncodedObs":
        return EncodedObs(
            planet_feats=self.planet_feats.to(device),
            planet_mask=self.planet_mask.to(device),
            planet_owned_mask=self.planet_owned_mask.to(device),
            planet_ids=self.planet_ids.to(device),
            planet_garrison=self.planet_garrison.to(device),
            fleet_feats=self.fleet_feats.to(device),
            fleet_mask=self.fleet_mask.to(device),
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


def encode_observation(o: Observation, device: str | torch.device = "cpu") -> EncodedObs:
    p_feats = np.zeros((MAX_PLANETS, PLANET_FEAT_DIM), dtype=np.float32)
    p_mask = np.zeros(MAX_PLANETS, dtype=bool)
    p_owned = np.zeros(MAX_PLANETS, dtype=bool)
    p_ids = -np.ones(MAX_PLANETS, dtype=np.int64)
    p_gar = np.zeros(MAX_PLANETS, dtype=np.float32)

    comet_motion = _comet_motion_by_id(o)
    planet_pos = {p.id: (p.x, p.y) for p in o.planets}
    num_players = _infer_num_players(o)

    for i, p in enumerate(o.planets[:MAX_PLANETS]):
        p_feats[i] = _planet_features(
            p, o.player, num_players, o.angular_velocity, comet_motion
        )
        p_mask[i] = True
        p_owned[i] = p.owner == o.player
        p_ids[i] = p.id
        p_gar[i] = p.ships

    f_feats = np.zeros((MAX_FLEETS, FLEET_FEAT_DIM), dtype=np.float32)
    f_mask = np.zeros(MAX_FLEETS, dtype=bool)
    for j, f in enumerate(o.fleets[:MAX_FLEETS]):
        f_feats[j] = _fleet_features(f, o.player, num_players, planet_pos)
        f_mask[j] = True

    return EncodedObs(
        planet_feats=torch.from_numpy(p_feats).to(device),
        planet_mask=torch.from_numpy(p_mask).to(device),
        planet_owned_mask=torch.from_numpy(p_owned).to(device),
        planet_ids=torch.from_numpy(p_ids).to(device),
        planet_garrison=torch.from_numpy(p_gar).to(device),
        fleet_feats=torch.from_numpy(f_feats).to(device),
        fleet_mask=torch.from_numpy(f_mask).to(device),
    )


def stack_encoded(feats_list: list[EncodedObs]) -> EncodedObs:
    """Stack a list of unbatched `EncodedObs` into a batched one.

    Each input has tensors shaped `[P, ...]` / `[P]` / `[F, ...]` / `[F]`;
    output tensors gain a leading batch dim. Used by the vectorized
    rollout to produce one big batch per env-step.
    """
    pf, pm, pom, pid, pg, ff, fm = [], [], [], [], [], [], []
    for f in feats_list:
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
    )
