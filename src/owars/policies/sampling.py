"""Convert raw `PolicyOutput` into a list of legal `Move`s.

The policy emits, per owned planet:
  - a Bernoulli launch decision
  - a masked Categorical over target planets, conditional on launch
  - a Beta(α, β) on [0, 1] for the fraction-of-garrison to send, conditional
    on launch

The simulator's action format is `[from_planet_id, angle_radians, num_ships]`.
Fleets fly in *straight lines* at constant speed (`fleet_speed(num_ships)`),
so the launch angle is fully determined by the chosen target — there is no
mid-flight steering. We compute the angle deterministically by solving the
intercept equation in closed form (see `_lead_angle`) — no fixed-point
iteration that might oscillate.

For PPO we need, *per owned planet*, the Bernoulli + conditional
Categorical/Beta log-prob of the actually-sampled action. `sample_with_record`
returns those alongside the moves; `sample_actions` is the thin moves-only
wrapper used by inference paths that don't care about log-probs. The Beta
sample is the action — there is no separate latent (vs the previous
tanh-Gaussian, which had pre-squash `z` and post-squash fraction);
`Beta.log_prob(fraction)` is direct and exact.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn.functional as nn_functional
from torch.distributions import Beta

from ..game import angle_to
from ..game.observation import Observation
from ..game.physics import fleet_speed
from ..game.types import BOARD_SIZE, CENTER, ROTATION_RADIUS_LIMIT, SUN_RADIUS, Move
from .model import PolicyOutput

# Sample clamp for digamma/log stability in `Beta.log_prob`. With α,β ≥ 1
# (post-parameterization) `log_prob` is finite on the closed [0, 1] but the Beta-
# Jacobian `(α-1) log z + (β-1) log(1-z)` blows up if a sampled z hits
# exactly 0 or 1 with α=1 or β=1 (where the corresponding term is 0·log 0).
# Mirrors `cleanrl ppo_continuous_action_pmpo_d4_beta_relusq_v3.py:47`.
SAMPLE_EPS: float = 1e-7


def _deterministic_fraction(
    fraction_alpha: torch.Tensor,
    fraction_beta: torch.Tensor,
) -> torch.Tensor:
    """Return the deterministic fraction represented by the Beta head.

    The policy parameterizes α=1+c·μ and β=1+c·(1−μ), so μ is recoverable as
    the Beta mode `(α−1)/(α+β−2)`. This is the intended deterministic action;
    the ordinary Beta mean is deliberately pulled toward 0.5 by the +1 floor.
    """
    concentration = (fraction_alpha + fraction_beta - 2.0).clamp_min(SAMPLE_EPS)
    return ((fraction_alpha - 1.0) / concentration).clamp(
        SAMPLE_EPS, 1.0 - SAMPLE_EPS
    )


# Lead-intercept solver. The intercept condition for a fleet leaving source
# `(sx, sy)` at speed `sp` to meet a target on a circular orbit (radius R,
# angular velocity ω, initial angle θ₀) is the perfect-collision equation
# fleet(t) = target(t), which after squaring reduces to:
#
#     f(t) = sp²·t² + 2Rρ·cos(θ₀ + ωt − φ) − (R² + ρ²) = 0
#
# where ρ = ‖source − center‖, φ = atan2(sy − cy, sx − cx). At t=0 we have
# f(0) = −‖target_now − source‖² ≤ 0; for large t the parabolic term wins so
# f(t) → +∞. The cosine term oscillates with period T_orb = 2π/|ω|, but its
# influence is bounded by 2Rρ — the parabola eventually dominates. By IVT a
# root always exists, and by sampling f at intervals strictly less than half
# T_orb we cannot skip the *first* sign change (Nyquist for the cosine).
# Within that bracket bisection converges geometrically to machine precision.
#
# Returns `None` only when no intercept lies within `LEAD_T_HORIZON` board-
# steps — effectively "the fleet would still be in flight when the episode
# ends." That signals the move builder to silently skip the launch.
LEAD_BISECT_ITERS: int = 24
LEAD_T_HORIZON_STEPS: float = 600.0  # episode is 500 steps; a bit of slack
LEAD_MAX_TURNS: int = int(LEAD_T_HORIZON_STEPS)


@dataclass
class SampleRecord:
    """Per-planet record of the sampled action — used by PPO rollouts.

    The full distribution parameters (`launch_logits`, `target_logits`,
    `fraction_alpha`, `fraction_beta`) are recorded alongside the sample so
    PPO can compute the analytical KL divergence between the rollout-time
    policy and the current policy (PMPO penalty, dreamer4
    §`pmpo_kl_div_loss_weight`). Importance-ratio PPO uses only `log_prob`,
    but the KL term needs the full distributions — hence both.

    The Beta sample IS the action (no separate latent), so we only carry
    `fraction` ∈ (eps, 1-eps); recomputing `log_prob` at that value uses
    `Beta.log_prob` directly with no Jacobian gymnastics.
    """

    launch: torch.Tensor       # [P] float 0/1 Bernoulli sample
    target_idx: torch.Tensor   # [P] long, in [0, P)
    fraction: torch.Tensor     # [P] float in (eps, 1-eps) — Beta sample, used both for the move and for PPO's log_prob recompute
    log_prob: torch.Tensor     # [P] float — Bernoulli + launch*(Categorical + Beta)
    launch_logits: torch.Tensor       # [P] — old-policy Bernoulli logits (PMPO KL input)
    target_logits: torch.Tensor       # [P, P] — old-policy categorical logits, masked to the sampled legality context
    fraction_alpha: torch.Tensor      # [P] — old-policy Beta α
    fraction_beta: torch.Tensor       # [P] — old-policy Beta β


@dataclass
class SampleBatchRecord:
    """Batched PPO record for vector rollouts.

    Same fields as `SampleRecord`, with a leading row dimension. This avoids
    constructing one Python object and one small torch graph per bucket row in
    the rollout hot path.
    """

    launch: torch.Tensor
    target_idx: torch.Tensor
    fraction: torch.Tensor
    log_prob: torch.Tensor
    launch_logits: torch.Tensor
    target_logits: torch.Tensor
    fraction_alpha: torch.Tensor
    fraction_beta: torch.Tensor


@dataclass(slots=True)
class ActionContext:
    """Fast action builder context backed by simulator planet rows."""

    planets: Any
    angular_velocity: float
    comet_planet_ids: Any = ()


@dataclass(slots=True)
class LeadSolution:
    angle: float
    time: float
    x: float
    y: float


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
    qx = ax + t * dx
    qy = ay + t * dy
    return math.hypot(px - qx, py - qy)


def _launch_start(
    mine_x: float, mine_y: float, mine_radius: float, angle: float
) -> tuple[float, float]:
    offset = max(0.0, float(mine_radius) + 0.1)
    return mine_x + math.cos(angle) * offset, mine_y + math.sin(angle) * offset


def _segment_crosses_sun(
    ax: float, ay: float, bx: float, by: float
) -> bool:
    return _point_to_segment_distance(
        CENTER[0], CENTER[1], ax, ay, bx, by
    ) < SUN_RADIUS


def _is_inside_board(x: float, y: float) -> bool:
    return 0.0 <= x <= BOARD_SIZE and 0.0 <= y <= BOARD_SIZE


def _safe_flight_segment(
    mine_x: float,
    mine_y: float,
    mine_radius: float,
    angle: float,
    end_x: float,
    end_y: float,
) -> bool:
    start_x, start_y = _launch_start(mine_x, mine_y, mine_radius, angle)
    if not _is_inside_board(start_x, start_y):
        return False
    if not _is_inside_board(end_x, end_y):
        return False
    return not _segment_crosses_sun(start_x, start_y, end_x, end_y)


def _lead_solution_from_point(
    mine_x: float,
    mine_y: float,
    target_x: float,
    target_y: float,
    target_radius: float,
    angular_velocity: float,
    send: int,
) -> LeadSolution | None:
    """Closed-form first-intercept solution from an exact fleet start point.

    Solves `sp²·t² + 2Rρ·cos(θ₀ + ωt − φ) = R² + ρ²` for the smallest
    `t ≥ 0` via coarse Nyquist scan + bisection (see module docstring).
    The fleet is aimed at the target's *predicted* position at that t; by
    construction `sp·t = ‖target(t) − source‖`, so a fleet flying that
    angle for `t` board-steps lands exactly on the target.

    Static planets and ω=0 collapse to the zero-orbit case `t = d / sp`,
    handled inline.

    Returns `None` if no intercept exists within `LEAD_T_HORIZON_STEPS`.
    """
    sp = fleet_speed(send)
    if sp <= 0.0:
        return None
    cx, cy = CENTER
    orbit_radius = math.hypot(target_x - cx, target_y - cy)
    # Static target (matches `geometry.predicted_position`'s rule) or zero
    # angular velocity → no orbital motion → aim direct, t = d/sp.
    is_orbiting = (
        orbit_radius + target_radius < ROTATION_RADIUS_LIMIT
        and abs(angular_velocity) > 1e-12
        and orbit_radius > 1e-9
    )
    if not is_orbiting:
        d = math.hypot(target_x - mine_x, target_y - mine_y)
        if d / sp > LEAD_T_HORIZON_STEPS:
            return None
        return LeadSolution(
            angle=angle_to(mine_x, mine_y, target_x, target_y),
            time=float(max(1, math.ceil(max(0.0, d - target_radius) / sp))),
            x=target_x,
            y=target_y,
        )

    # Official collision checks are turn-discrete: on turn k, the fleet segment
    # is checked against the planet's pre-move position at phase k-1. Aim at
    # one of those exact checked positions, not a continuous-time interpolation.
    theta0 = math.atan2(target_y - cy, target_x - cx)
    previous_error: float | None = None
    for k in range(1, LEAD_MAX_TURNS + 1):
        theta = theta0 + angular_velocity * (k - 1)
        tx = cx + orbit_radius * math.cos(theta)
        ty = cy + orbit_radius * math.sin(theta)
        d = math.hypot(tx - mine_x, ty - mine_y)
        error = d - k * sp
        if error <= target_radius:
            prev_dist = max(0.0, (k - 1) * sp)
            if d >= prev_dist - target_radius:
                return LeadSolution(
                    angle=angle_to(mine_x, mine_y, tx, ty),
                    time=float(k),
                    x=tx,
                    y=ty,
                )
        if previous_error is not None and previous_error < -target_radius and error > target_radius:
            break
        previous_error = error

    return None


def _continuous_lead_solution_from_point(
    mine_x: float,
    mine_y: float,
    target_x: float,
    target_y: float,
    target_radius: float,
    angular_velocity: float,
    send: int,
) -> LeadSolution | None:
    """Continuous-time lead solution retained for geometry regression tests."""
    sp = fleet_speed(send)
    if sp <= 0.0:
        return None
    cx, cy = CENTER
    orbit_radius = math.hypot(target_x - cx, target_y - cy)
    is_orbiting = (
        orbit_radius + target_radius < ROTATION_RADIUS_LIMIT
        and abs(angular_velocity) > 1e-12
        and orbit_radius > 1e-9
    )
    if not is_orbiting:
        d = math.hypot(target_x - mine_x, target_y - mine_y)
        if d / sp > LEAD_T_HORIZON_STEPS:
            return None
        return LeadSolution(
            angle=angle_to(mine_x, mine_y, target_x, target_y),
            time=d / sp,
            x=target_x,
            y=target_y,
        )

    a = mine_x - cx
    b = mine_y - cy
    rho = math.hypot(a, b)
    phi = math.atan2(b, a)
    theta0 = math.atan2(target_y - cy, target_x - cx)
    cosine_scale = 2.0 * orbit_radius * rho
    distance_offset = orbit_radius * orbit_radius + rho * rho

    def f(t: float) -> float:
        return (
            sp * sp * t * t
            + cosine_scale * math.cos(theta0 + angular_velocity * t - phi)
            - distance_offset
        )

    # Source coincident with target (dist=0) ⇒ t* = 0; aim direct.
    if f(0.0) >= -1e-9:
        return LeadSolution(
            angle=angle_to(mine_x, mine_y, target_x, target_y),
            time=0.0,
            x=target_x,
            y=target_y,
        )

    # Nyquist for the cosine: dt < π/|ω| guarantees we see every sign change.
    # T/16 is a comfortable factor-of-8 safety margin; cost is trivial (a
    # handful of cos evaluations per move).
    orbit_period = 2.0 * math.pi / abs(angular_velocity)
    dt = orbit_period / 16.0
    # Latest possible intercept: when sp·t exceeds R + ρ the parabola is
    # always above the cosine ceiling, so f(t) > 0 from there on. Add one
    # full orbit period for safety on edge geometries.
    t_max = min((orbit_radius + rho) / sp + orbit_period, LEAD_T_HORIZON_STEPS)

    t_prev = 0.0
    f_prev = f(0.0)  # < 0 by the check above
    t = dt
    bracket: tuple[float, float] | None = None
    while t <= t_max:
        f_curr = f(t)
        if f_curr >= 0.0 and f_prev < 0.0:
            bracket = (t_prev, t)
            break
        t_prev = t
        f_prev = f_curr
        t += dt
    if bracket is None:
        return None  # no intercept within horizon

    lo, hi = bracket
    for _ in range(LEAD_BISECT_ITERS):
        mid = 0.5 * (lo + hi)
        if f(mid) < 0.0:
            lo = mid
        else:
            hi = mid
    t_star = 0.5 * (lo + hi)
    psi = theta0 + angular_velocity * t_star
    tx = cx + orbit_radius * math.cos(psi)
    ty = cy + orbit_radius * math.sin(psi)
    return LeadSolution(angle=angle_to(mine_x, mine_y, tx, ty), time=t_star, x=tx, y=ty)


def _lead_angle_from_point(
    mine_x: float,
    mine_y: float,
    target_x: float,
    target_y: float,
    target_radius: float,
    angular_velocity: float,
    send: int,
) -> float | None:
    solution = _lead_solution_from_point(
        mine_x, mine_y, target_x, target_y, target_radius, angular_velocity, send
    )
    return None if solution is None else solution.angle


def _lead_solution(
    mine_x: float,
    mine_y: float,
    mine_radius: float,
    target_x: float,
    target_y: float,
    target_radius: float,
    angular_velocity: float,
    send: int,
) -> LeadSolution | None:
    """First-intercept launch solution for the official action semantics.

    The simulator does not spawn a fleet at the source center. It starts the
    fleet just outside the planet along the submitted angle, so the start
    point itself depends on the angle. Iterating the point-solver a few times
    accounts for that offset and prevents small-radius targets from being
    missed by a centerline shot.
    """
    solution = _lead_solution_from_point(
        mine_x, mine_y, target_x, target_y, target_radius, angular_velocity, send
    )
    if solution is None:
        return None
    angle = solution.angle
    launch_offset = max(0.0, float(mine_radius) + 0.1)
    if launch_offset <= 0.0:
        return solution

    for _ in range(4):
        start_x = mine_x + math.cos(angle) * launch_offset
        start_y = mine_y + math.sin(angle) * launch_offset
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
        if abs(math.atan2(math.sin(refined.angle - angle), math.cos(refined.angle - angle))) < 1e-6:
            return refined
        angle = refined.angle
        solution = refined
    return solution


def _lead_angle(
    mine_x: float,
    mine_y: float,
    mine_radius: float,
    target_x: float,
    target_y: float,
    target_radius: float,
    angular_velocity: float,
    send: int,
) -> float | None:
    solution = _lead_solution(
        mine_x,
        mine_y,
        mine_radius,
        target_x,
        target_y,
        target_radius,
        angular_velocity,
        send,
    )
    return None if solution is None else solution.angle


def _build_moves_from_lists(
    launch_l: list[float],
    target_idx_l: list[int],
    frac_l: list[float],
    owned_l: list[bool],
    pmask_l: list[bool],
    ids_l: list[int],
    o: Observation,
) -> tuple[list[Move], list[bool]]:
    moves: list[Move] = []
    materialized = [False] * len(target_idx_l)
    by_id = {pl.id: pl for pl in o.planets}
    remaining_by_id = {pl.id: int(pl.ships) for pl in o.planets}
    omega = o.angular_velocity
    p = len(target_idx_l)
    for i in range(p):
        if not (owned_l[i] and pmask_l[i]):
            continue
        if launch_l[i] < 0.5:
            continue
        ti = target_idx_l[i]
        if ti == i:
            continue  # self-target is also masked at logits-time
        target_id = ids_l[ti]
        if target_id < 0 or target_id in o.comet_planet_ids:
            continue
        mine = by_id.get(ids_l[i])
        target = by_id.get(target_id)
        if mine is None or target is None:
            continue
        remaining = remaining_by_id.get(mine.id, int(mine.ships))
        if remaining < 2:
            continue

        f = max(0.0, min(1.0, frac_l[i]))
        send = max(1, min(remaining - 1, int(round(remaining * f))))
        if send <= 0:
            continue

        solution = _lead_solution(
            mine.x,
            mine.y,
            mine.radius,
            target.x,
            target.y,
            target.radius,
            omega,
            send,
        )
        if solution is None:
            continue  # solver couldn't find a feasible intercept — silently no-op
        if not _safe_flight_segment(
            mine.x, mine.y, mine.radius, solution.angle, solution.x, solution.y
        ):
            continue
        moves.append(Move(mine.id, solution.angle, send))
        materialized[i] = True
        remaining_by_id[mine.id] = remaining - send

    return moves, materialized


def _build_moves_from_packed_fields(
    fields_l: list[list[float]],
    o: Observation,
) -> list[Move]:
    return _build_moves_from_packed_fields_with_mask(fields_l, o)[0]


def _candidate_action_indices(fields_l: Any) -> Any:
    if hasattr(fields_l, "ndim"):
        mask = (
            (fields_l[:, 2] >= 0.5)
            & (fields_l[:, 3] >= 0.5)
            & (fields_l[:, 4] >= 0.5)
        )
        indices = mask.nonzero()
        if isinstance(indices, tuple):
            return indices[0]
        return indices.flatten()
    return (
        i
        for i, fields in enumerate(fields_l)
        if fields[2] >= 0.5 and fields[3] >= 0.5 and fields[4] >= 0.5
    )


def _planet_id(planet: Any) -> int:
    return int(planet[0])


def _planet_x(planet: Any) -> float:
    return float(planet[2])


def _planet_y(planet: Any) -> float:
    return float(planet[3])


def _planet_radius(planet: Any) -> float:
    return float(planet[4])


def _planet_ships(planet: Any) -> int:
    return int(planet[5])


def _planet_field_map(planets: Any) -> dict[int, tuple[float, float, float, int]]:
    return {
        _planet_id(pl): (
            _planet_x(pl),
            _planet_y(pl),
            _planet_radius(pl),
            _planet_ships(pl),
        )
        for pl in planets
    }


def _target_legal_mask_from_planets(
    frac_l: Sequence[float],
    active_source_l: Sequence[bool],
    pmask_l: Sequence[bool],
    ids_l: Sequence[int],
    planets: Any,
    angular_velocity: float,
    comet_planet_ids: Any,
) -> list[list[bool]]:
    p = len(ids_l)
    mask = [[False] * p for _ in range(p)]
    planet_fields = _planet_field_map(planets)
    comet_ids = {int(pid) for pid in comet_planet_ids}
    omega = float(angular_velocity)
    target_fields = [
        (j, planet_fields[target_id])
        for j, target_id in enumerate(int(v) for v in ids_l)
        if bool(pmask_l[j])
        and target_id >= 0
        and target_id not in comet_ids
        and target_id in planet_fields
    ]
    for i in range(p):
        if not (bool(active_source_l[i]) and bool(pmask_l[i])):
            continue
        source = planet_fields.get(int(ids_l[i]))
        if source is None:
            continue
        source_x, source_y, source_radius, remaining = source
        if remaining < 2:
            continue
        frac = max(0.0, min(1.0, float(frac_l[i])))
        send = max(1, min(remaining - 1, int(round(remaining * frac))))
        if send <= 0:
            continue
        for j, target in target_fields:
            if j == i:
                continue
            target_x, target_y, target_radius, _target_ships = target
            solution = _lead_solution(
                source_x,
                source_y,
                source_radius,
                target_x,
                target_y,
                target_radius,
                omega,
                send,
            )
            if solution is None:
                continue
            if not _safe_flight_segment(
                source_x,
                source_y,
                source_radius,
                solution.angle,
                solution.x,
                solution.y,
            ):
                continue
            mask[i][j] = True
    return mask


def _target_legal_mask_from_packed_legality_fields(
    fields_l: Any,
    planets: Any,
    angular_velocity: float,
    comet_planet_ids: Any,
) -> Any:
    p = len(fields_l)
    mask = np.zeros((p, p), dtype=bool)
    planet_fields = _planet_field_map(planets)
    comet_ids = {int(pid) for pid in comet_planet_ids}
    omega = float(angular_velocity)
    ids = [int(fields_l[j][4]) for j in range(p)]
    present = [float(fields_l[j][3]) >= 0.5 for j in range(p)]
    target_fields = [
        (j, planet_fields[target_id])
        for j, target_id in enumerate(ids)
        if present[j]
        and target_id >= 0
        and target_id not in comet_ids
        and target_id in planet_fields
    ]

    if target_fields:
        x = np.full(p, np.nan, dtype=np.float64)
        y = np.full(p, np.nan, dtype=np.float64)
        radius = np.zeros(p, dtype=np.float64)
        ships = np.zeros(p, dtype=np.int32)
        known = np.zeros(p, dtype=bool)
        for idx, planet_id in enumerate(ids):
            fields = planet_fields.get(planet_id)
            if fields is None:
                continue
            x[idx], y[idx], radius[idx], ships[idx] = fields
            known[idx] = True

        source = (
            (np.asarray(fields_l[:, 2], dtype=np.float64) >= 0.5)
            & np.asarray(present, dtype=bool)
            & known
            & (ships >= 2)
        )
        static_target = np.asarray(present, dtype=bool) & known
        if comet_ids:
            static_target &= np.asarray([planet_id not in comet_ids for planet_id in ids], dtype=bool)
        static_target &= np.asarray(ids, dtype=np.int64) >= 0
        orbit_radius = np.hypot(x - CENTER[0], y - CENTER[1])
        is_orbiting = (
            (orbit_radius + radius < ROTATION_RADIUS_LIMIT)
            & (abs(omega) > 1e-12)
            & (orbit_radius > 1e-9)
        )
        static_target &= ~is_orbiting
        source_idx = np.flatnonzero(source)
        target_idx = np.flatnonzero(static_target)
        if source_idx.size and target_idx.size:
            frac = np.clip(np.asarray(fields_l[source_idx, 1], dtype=np.float64), 0.0, 1.0)
            remaining = ships[source_idx].astype(np.float64)
            send = np.rint(remaining * frac).astype(np.int32)
            send = np.maximum(1, np.minimum(ships[source_idx] - 1, send))
            speed = np.ones(send.shape, dtype=np.float64)
            fast = send > 1
            speed[fast] = 1.0 + 5.0 * (
                np.log(send[fast].astype(np.float64)) / math.log(1000.0)
            ) ** 1.5

            dx = x[target_idx][None, :] - x[source_idx][:, None]
            dy = y[target_idx][None, :] - y[source_idx][:, None]
            distance = np.hypot(dx, dy)
            pair_ok = distance / speed[:, None] <= LEAD_T_HORIZON_STEPS
            pair_ok &= source_idx[:, None] != target_idx[None, :]

            angle = np.arctan2(dy, dx)
            offset = np.maximum(0.0, radius[source_idx] + 0.1)
            start_x = x[source_idx][:, None] + np.cos(angle) * offset[:, None]
            start_y = y[source_idx][:, None] + np.sin(angle) * offset[:, None]
            end_x = x[target_idx][None, :]
            end_y = y[target_idx][None, :]
            pair_ok &= (
                (start_x >= 0.0)
                & (start_x <= BOARD_SIZE)
                & (start_y >= 0.0)
                & (start_y <= BOARD_SIZE)
                & (end_x >= 0.0)
                & (end_x <= BOARD_SIZE)
                & (end_y >= 0.0)
                & (end_y <= BOARD_SIZE)
            )

            seg_x = end_x - start_x
            seg_y = end_y - start_y
            denom = seg_x * seg_x + seg_y * seg_y
            projection = ((CENTER[0] - start_x) * seg_x + (CENTER[1] - start_y) * seg_y)
            t = np.divide(projection, denom, out=np.zeros_like(projection), where=denom > 0.0)
            t = np.clip(t, 0.0, 1.0)
            qx = start_x + t * seg_x
            qy = start_y + t * seg_y
            sun_distance = np.hypot(CENTER[0] - qx, CENTER[1] - qy)
            pair_ok &= sun_distance >= SUN_RADIUS
            mask[np.ix_(source_idx, target_idx)] = pair_ok

    for i in range(p):
        fields = fields_l[i]
        if not (float(fields[2]) >= 0.5 and present[i]):
            continue
        source = planet_fields.get(ids[i])
        if source is None:
            continue
        source_x, source_y, source_radius, remaining = source
        if remaining < 2:
            continue
        frac = max(0.0, min(1.0, float(fields[1])))
        send = max(1, min(remaining - 1, int(round(remaining * frac))))
        if send <= 0:
            continue
        for j, target in target_fields:
            if j == i:
                continue
            target_orbit_radius = math.hypot(target[0] - CENTER[0], target[1] - CENTER[1])
            if not (
                target_orbit_radius + target[2] < ROTATION_RADIUS_LIMIT
                and abs(omega) > 1e-12
                and target_orbit_radius > 1e-9
            ):
                continue
            target_x, target_y, target_radius, _target_ships = target
            solution = _lead_solution(
                source_x,
                source_y,
                source_radius,
                target_x,
                target_y,
                target_radius,
                omega,
                send,
            )
            if solution is None:
                continue
            if not _safe_flight_segment(
                source_x,
                source_y,
                source_radius,
                solution.angle,
                solution.x,
                solution.y,
            ):
                continue
            mask[i, j] = True
    return mask


def _packed_legality_fields(
    launch: torch.Tensor,
    frac: torch.Tensor,
    owned: torch.Tensor,
    pmask: torch.Tensor,
    ids: torch.Tensor,
) -> Any:
    return torch.stack(
        (
            launch.float(),
            frac.float(),
            owned.float(),
            pmask.float(),
            ids.float(),
        ),
        dim=-1,
    ).detach().cpu().numpy()


def _target_legal_mask_tensor(mask_l: Any, device: torch.device | str) -> torch.Tensor:
    return torch.as_tensor(np.asarray(mask_l, dtype=bool), device=device, dtype=torch.bool)


def _packed_action_fields_from_legality_fields(
    target_idx: torch.Tensor,
    legality_fields_l: Any,
    target_legal_mask_l: Any,
) -> Any:
    target_idx_l = target_idx.detach().cpu().numpy()
    fields_l = np.empty(target_idx_l.shape + (6,), dtype=legality_fields_l.dtype)
    fields_l[..., 0] = target_idx_l
    fields_l[..., 1] = legality_fields_l[..., 1]
    fields_l[..., 2] = legality_fields_l[..., 0]
    fields_l[..., 3] = legality_fields_l[..., 2]
    fields_l[..., 4] = legality_fields_l[..., 3]
    fields_l[..., 5] = legality_fields_l[..., 4]
    fields_l[..., 2] *= np.asarray(target_legal_mask_l, dtype=bool).any(axis=-1)
    return fields_l


def _target_legal_mask_from_observation(
    frac: torch.Tensor,
    launch: torch.Tensor,
    owned: torch.Tensor,
    pmask: torch.Tensor,
    ids: torch.Tensor,
    obs: Observation,
) -> torch.Tensor:
    del launch
    source = owned.to(dtype=torch.bool) & pmask.to(dtype=torch.bool)
    return torch.as_tensor(
        _target_legal_mask_from_planets(
            frac.detach().cpu().tolist(),
            source.detach().cpu().tolist(),
            pmask.detach().cpu().tolist(),
            [int(v) for v in ids.detach().cpu().tolist()],
            obs.planets,
            obs.angular_velocity,
            obs.comet_planet_ids,
        ),
        device=frac.device,
        dtype=torch.bool,
    )


def _apply_target_legal_mask(
    target_logits: torch.Tensor,
    target_legal_mask: torch.Tensor,
    launch: torch.Tensor,
    owned: torch.Tensor,
    pmask: torch.Tensor,
) -> torch.Tensor:
    del launch
    source = owned.to(dtype=torch.bool) & pmask.to(dtype=torch.bool)
    source = source.to(device=target_logits.device).unsqueeze(-1)
    target_legal_mask = target_legal_mask.to(device=target_logits.device, dtype=torch.bool)
    masked = target_logits.float().masked_fill(~target_legal_mask, float("-inf"))
    return torch.where(source, masked, target_logits.float())


def _mask_impossible_launches(
    launch_logits: torch.Tensor,
    launch: torch.Tensor,
    target_legal_mask: torch.Tensor,
    owned: torch.Tensor,
    pmask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    source = owned.to(dtype=torch.bool) & pmask.to(dtype=torch.bool)
    has_legal_target = target_legal_mask.to(device=launch.device, dtype=torch.bool).any(dim=-1)
    impossible = source & ~has_legal_target
    launch_logits = launch_logits.float().masked_fill(impossible, -20.0)
    launch = launch.masked_fill(impossible, 0.0)
    return launch_logits, launch


def _sample_launch_fraction(
    launch_logits: torch.Tensor,
    fraction_alpha: torch.Tensor,
    fraction_beta: torch.Tensor,
    deterministic: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    launch_logits = launch_logits.float()
    fraction_alpha = fraction_alpha.float()
    fraction_beta = fraction_beta.float()
    if deterministic:
        launch = (launch_logits > 0.0).to(fraction_alpha.dtype)
        frac = _deterministic_fraction(fraction_alpha, fraction_beta)
        return launch, frac
    launch = torch.distributions.Bernoulli(logits=launch_logits).sample()
    frac = Beta(fraction_alpha, fraction_beta).sample().clamp(SAMPLE_EPS, 1.0 - SAMPLE_EPS)
    return launch, frac


def _sample_target(
    target_logits: torch.Tensor,
    deterministic: bool,
) -> torch.Tensor:
    safe_target_logits = _safe_target_logits(target_logits.float())
    if deterministic:
        return safe_target_logits.argmax(dim=-1)
    return torch.distributions.Categorical(logits=safe_target_logits).sample()


def _build_moves_from_packed_fields_with_mask(
    fields_l: list[list[float]],
    o: Observation,
) -> tuple[list[Move], list[bool]]:
    moves: list[Move] = []
    materialized = [False] * len(fields_l)
    by_id = {pl.id: pl for pl in o.planets}
    remaining_by_id = {pl.id: int(pl.ships) for pl in o.planets}
    omega = o.angular_velocity
    p = len(fields_l)
    for i in _candidate_action_indices(fields_l):
        i = int(i)
        fields = fields_l[i]
        ti = int(fields[0])
        if ti == i:
            continue
        target_id = int(fields_l[ti][5]) if 0 <= ti < p else -1
        if target_id < 0 or target_id in o.comet_planet_ids:
            continue
        mine = by_id.get(int(fields[5]))
        target = by_id.get(target_id)
        if mine is None or target is None:
            continue
        remaining = remaining_by_id.get(mine.id, int(mine.ships))
        if remaining < 2:
            continue

        f = max(0.0, min(1.0, float(fields[1])))
        send = max(1, min(remaining - 1, int(round(remaining * f))))
        if send <= 0:
            continue

        solution = _lead_solution(
            mine.x,
            mine.y,
            mine.radius,
            target.x,
            target.y,
            target.radius,
            omega,
            send,
        )
        if solution is None:
            continue
        if not _safe_flight_segment(
            mine.x, mine.y, mine.radius, solution.angle, solution.x, solution.y
        ):
            continue
        moves.append(Move(mine.id, solution.angle, send))
        materialized[i] = True
        remaining_by_id[mine.id] = remaining - send

    return moves, materialized


def _build_action_lists_from_packed_fields_raw(
    fields_l: list[list[float]],
    obs: Any,
) -> list[list]:
    return _build_action_lists_from_packed_fields_raw_with_mask(fields_l, obs)[0]


def _build_action_lists_from_packed_fields_raw_with_mask(
    fields_l: list[list[float]],
    obs: Any,
) -> tuple[list[list], list[bool]]:
    planets = obs.get("planets", []) if isinstance(obs, dict) else getattr(obs, "planets", [])
    by_id = {int(p[0]): p for p in planets}
    remaining_by_id = {int(p[0]): int(p[5]) for p in planets}
    comet_planet_ids = set(
        obs.get("comet_planet_ids", []) if isinstance(obs, dict) else getattr(obs, "comet_planet_ids", [])
    )
    omega = float(
        (obs.get("angular_velocity", 0.0) if isinstance(obs, dict) else getattr(obs, "angular_velocity", 0.0))
        or 0.0
    )
    actions: list[list] = []
    materialized = [False] * len(fields_l)
    p = len(fields_l)
    for i in _candidate_action_indices(fields_l):
        i = int(i)
        fields = fields_l[i]
        ti = int(fields[0])
        if ti == i:
            continue
        target_id = int(fields_l[ti][5]) if 0 <= ti < p else -1
        if target_id < 0 or target_id in comet_planet_ids:
            continue
        mine = by_id.get(int(fields[5]))
        target = by_id.get(target_id)
        if mine is None or target is None:
            continue
        mine_ships = remaining_by_id.get(int(mine[0]), int(mine[5]))
        if mine_ships < 2:
            continue

        frac = max(0.0, min(1.0, float(fields[1])))
        send = max(1, min(mine_ships - 1, int(round(mine_ships * frac))))
        if send <= 0:
            continue

        solution = _lead_solution(
            float(mine[2]),
            float(mine[3]),
            float(mine[4]),
            float(target[2]),
            float(target[3]),
            float(target[4]),
            omega,
            send,
        )
        if solution is None:
            continue
        if not _safe_flight_segment(
            float(mine[2]),
            float(mine[3]),
            float(mine[4]),
            solution.angle,
            solution.x,
            solution.y,
        ):
            continue
        actions.append(
            [
                int(mine[0]),
                float(solution.angle),
                int(send),
                int(target_id),
                float(solution.time),
                float(solution.x),
                float(solution.y),
            ]
        )
        materialized[i] = True
        remaining_by_id[int(mine[0])] = mine_ships - send

    return actions, materialized


def _build_action_lists_from_packed_fields_context(
    fields_l: list[list[float]],
    context: ActionContext,
) -> list[list]:
    return _build_action_lists_from_packed_fields_context_with_mask(fields_l, context)[0]


def _build_action_lists_from_packed_fields_context_with_mask(
    fields_l: list[list[float]],
    context: ActionContext,
) -> tuple[list[list], list[bool]]:
    by_id = {int(p[0]): p for p in context.planets}
    remaining_by_id = {int(p[0]): int(p[5]) for p in context.planets}
    comet_planet_ids = set(context.comet_planet_ids)
    omega = float(context.angular_velocity)
    actions: list[list] = []
    materialized = [False] * len(fields_l)
    p = len(fields_l)
    for i in _candidate_action_indices(fields_l):
        i = int(i)
        fields = fields_l[i]
        ti = int(fields[0])
        if ti == i:
            continue
        target_id = int(fields_l[ti][5]) if 0 <= ti < p else -1
        if target_id < 0 or target_id in comet_planet_ids:
            continue
        mine = by_id.get(int(fields[5]))
        target = by_id.get(target_id)
        if mine is None or target is None:
            continue
        mine_ships = remaining_by_id.get(int(mine[0]), int(mine[5]))
        if mine_ships < 2:
            continue

        frac = max(0.0, min(1.0, float(fields[1])))
        send = max(1, min(mine_ships - 1, int(round(mine_ships * frac))))
        if send <= 0:
            continue

        solution = _lead_solution(
            float(mine[2]),
            float(mine[3]),
            float(mine[4]),
            float(target[2]),
            float(target[3]),
            float(target[4]),
            omega,
            send,
        )
        if solution is None:
            continue
        if not _safe_flight_segment(
            float(mine[2]),
            float(mine[3]),
            float(mine[4]),
            solution.angle,
            solution.x,
            solution.y,
        ):
            continue
        actions.append(
            [
                int(mine[0]),
                float(solution.angle),
                int(send),
                int(target_id),
                float(solution.time),
                float(solution.x),
                float(solution.y),
            ]
        )
        materialized[i] = True
        remaining_by_id[int(mine[0])] = mine_ships - send

    return actions, materialized


def _build_moves(
    launch: torch.Tensor,
    target_idx: torch.Tensor,
    frac: torch.Tensor,
    owned: torch.Tensor,
    pmask: torch.Tensor,
    ids: torch.Tensor,
    o: Observation,
) -> list[Move]:
    return _build_moves_with_mask(launch, target_idx, frac, owned, pmask, ids, o)[0]


def _build_moves_with_mask(
    launch: torch.Tensor,
    target_idx: torch.Tensor,
    frac: torch.Tensor,
    owned: torch.Tensor,
    pmask: torch.Tensor,
    ids: torch.Tensor,
    o: Observation,
) -> tuple[list[Move], list[bool]]:
    """Translate a single env's sampled (launch, target_idx, frac) into legal Moves.

    All five tensors come from one batch element: one CPU pull to Python
    lists at the top of the function avoids `.item()` calls inside the
    per-planet loop (each `.item()` on a CUDA tensor forces a stream sync,
    so a P=64 planet loop was 64×4+ syncs per call).
    """
    return _build_moves_from_lists(
        launch.tolist(),
        target_idx.tolist(),
        frac.tolist(),
        owned.tolist(),
        pmask.tolist(),
        ids.tolist(),
        o,
    )


def _safe_target_logits(target_logits: torch.Tensor) -> torch.Tensor:
    """Make target categorical rows finite even when no legal target exists."""
    finite = torch.isfinite(target_logits).any(dim=-1, keepdim=True)
    return torch.where(finite, target_logits, torch.zeros_like(target_logits))


def _beta_log_prob(
    alpha: torch.Tensor,
    beta: torch.Tensor,
    value: torch.Tensor,
) -> torch.Tensor:
    log_norm = torch.lgamma(alpha) + torch.lgamma(beta) - torch.lgamma(alpha + beta)
    return (
        (alpha - 1.0) * value.log()
        + (beta - 1.0) * torch.log1p(-value)
        - log_norm
    )


def _record_from_materialized_launch(
    launch: torch.Tensor,
    target_idx: torch.Tensor,
    frac: torch.Tensor,
    launch_logits: torch.Tensor,
    target_logits: torch.Tensor,
    fraction_alpha: torch.Tensor,
    fraction_beta: torch.Tensor,
    materialized: list[bool],
) -> SampleRecord:
    del launch
    actual_launch = torch.as_tensor(
        materialized,
        device=target_idx.device,
        dtype=fraction_alpha.dtype,
    )
    safe_target_logits = _safe_target_logits(target_logits.float())
    launch_lp = -nn_functional.binary_cross_entropy_with_logits(
        launch_logits.float(),
        actual_launch.float(),
        reduction="none",
    )
    target_log_probs = torch.log_softmax(safe_target_logits, dim=-1)
    target_lp = target_log_probs.gather(
        -1,
        target_idx.clamp(0, safe_target_logits.shape[-1] - 1).unsqueeze(-1),
    ).squeeze(-1)
    frac_lp = _beta_log_prob(fraction_alpha.float(), fraction_beta.float(), frac.float())
    log_prob = launch_lp + actual_launch.float() * (target_lp + frac_lp)
    return SampleRecord(
        launch=actual_launch,
        target_idx=target_idx,
        fraction=frac,
        log_prob=log_prob,
        launch_logits=launch_logits,
        target_logits=target_logits,
        fraction_alpha=fraction_alpha,
        fraction_beta=fraction_beta,
    )


def _batch_record_from_materialized_launch(
    launch: torch.Tensor,
    target_idx: torch.Tensor,
    frac: torch.Tensor,
    launch_logits: torch.Tensor,
    target_logits: torch.Tensor,
    fraction_alpha: torch.Tensor,
    fraction_beta: torch.Tensor,
    materialized: list[list[bool]],
    rows: Sequence[int],
) -> SampleBatchRecord:
    row_idx = torch.as_tensor(rows, device=launch.device, dtype=torch.long)
    if len(rows) == 0:
        actual_launch = launch.new_zeros((0, launch.shape[1]))
    else:
        if len(materialized) != len(rows):
            raise RuntimeError("materialized action rows do not match record rows")
        actual_launch = torch.as_tensor(
            materialized,
            device=launch.device,
            dtype=launch.dtype,
        )
    target_idx_r = target_idx.index_select(0, row_idx)
    frac_r = frac.index_select(0, row_idx)
    launch_logits_r = launch_logits.index_select(0, row_idx)
    target_logits_r = target_logits.index_select(0, row_idx)
    fraction_alpha_r = fraction_alpha.index_select(0, row_idx)
    fraction_beta_r = fraction_beta.index_select(0, row_idx)

    safe_target_logits = _safe_target_logits(target_logits_r.float())
    launch_lp = -nn_functional.binary_cross_entropy_with_logits(
        launch_logits_r.float(),
        actual_launch.float(),
        reduction="none",
    )
    target_log_probs = torch.log_softmax(safe_target_logits, dim=-1)
    target_lp = target_log_probs.gather(
        -1,
        target_idx_r.clamp(0, safe_target_logits.shape[-1] - 1).unsqueeze(-1),
    ).squeeze(-1)
    frac_lp = _beta_log_prob(
        fraction_alpha_r.float(),
        fraction_beta_r.float(),
        frac_r.float(),
    )
    log_prob = launch_lp + actual_launch.float() * (target_lp + frac_lp)
    return SampleBatchRecord(
        launch=actual_launch,
        target_idx=target_idx_r,
        fraction=frac_r,
        log_prob=log_prob,
        launch_logits=launch_logits_r,
        target_logits=target_logits_r,
        fraction_alpha=fraction_alpha_r,
        fraction_beta=fraction_beta_r,
    )


def sample_with_record(
    out: PolicyOutput,
    o: Observation,
    deterministic: bool = False,
) -> tuple[list[Move], SampleRecord]:
    """Sample an action per planet, build the legal `Move` list, AND return
    the per-planet (launch, target_idx, fraction, log_prob) record so PPO can
    compute the importance ratio against the *actual* sampled action.
    """
    launch_logits = out.launch_logits[0]
    target_logits = out.target_logits[0]  # [P, P]
    fraction_alpha = out.fraction_alpha[0]
    fraction_beta = out.fraction_beta[0]
    owned = out.planet_owned_mask[0]
    pmask = out.planet_mask[0]
    ids = out.planet_ids[0]

    launch, frac = _sample_launch_fraction(
        launch_logits, fraction_alpha, fraction_beta, deterministic
    )
    target_legal_mask = _target_legal_mask_from_observation(
        frac, launch, owned, pmask, ids, o
    )
    target_logits = _apply_target_legal_mask(
        target_logits, target_legal_mask, launch, owned, pmask
    )
    launch_logits, launch = _mask_impossible_launches(
        launch_logits, launch, target_legal_mask, owned, pmask
    )
    target_idx = _sample_target(target_logits, deterministic)

    moves, materialized = _build_moves_with_mask(
        launch, target_idx, frac, owned, pmask, ids, o
    )
    record = _record_from_materialized_launch(
        launch,
        target_idx,
        frac,
        launch_logits,
        target_logits,
        fraction_alpha,
        fraction_beta,
        materialized,
    )
    return moves, record


def sample_actions(
    out: PolicyOutput,
    o: Observation,
    deterministic: bool = True,
) -> list[Move]:
    """Moves-only wrapper for inference paths (eval, agent submission)."""
    launch_logits = out.launch_logits[0]
    target_logits = out.target_logits[0]
    fraction_alpha = out.fraction_alpha[0]
    fraction_beta = out.fraction_beta[0]
    owned = out.planet_owned_mask[0]
    pmask = out.planet_mask[0]
    ids = out.planet_ids[0]
    launch, frac = _sample_launch_fraction(
        launch_logits, fraction_alpha, fraction_beta, deterministic
    )
    target_legal_mask = _target_legal_mask_from_observation(
        frac, launch, owned, pmask, ids, o
    )
    target_logits = _apply_target_legal_mask(
        target_logits, target_legal_mask, launch, owned, pmask
    )
    _launch_logits, launch = _mask_impossible_launches(
        launch_logits, launch, target_legal_mask, owned, pmask
    )
    target_idx = _sample_target(target_logits, deterministic)
    return _build_moves(
        launch,
        target_idx,
        frac,
        owned,
        pmask,
        ids,
        o,
    )


def sample_batch_with_records(
    out: PolicyOutput,
    parsed_list: list[Observation],
    deterministic: bool = False,
    record_rows: Sequence[int] | None = None,
) -> tuple[list[list[Move]], list[SampleRecord] | SampleBatchRecord]:
    """Batched counterpart to `sample_with_record`.

    `out` is a B>1 PolicyOutput (its tensors have a leading batch dim);
    `parsed_list` has length B with the parsed observation per element.
    Returns one move list per element. By default it also returns one
    `SampleRecord` per element for legacy callers. When `record_rows` is
    provided, it returns a single `SampleBatchRecord` for those rows only.

    The Bernoulli/Categorical/Beta samples are drawn once over the full [B, P]
    tensor — that's where the GPU win comes from. The per-element
    `_build_moves` walk is pure Python but cheap (one loop per env).
    """
    launch_logits = out.launch_logits      # [B, P]
    target_logits = out.target_logits      # [B, P, P]
    fraction_alpha = out.fraction_alpha    # [B, P]
    fraction_beta = out.fraction_beta      # [B, P]
    b_dim, p, _ = target_logits.shape
    assert len(parsed_list) == b_dim, (len(parsed_list), b_dim)

    launch, frac = _sample_launch_fraction(
        launch_logits, fraction_alpha, fraction_beta, deterministic
    )
    legality_fields_l = _packed_legality_fields(
        launch, frac, out.planet_owned_mask, out.planet_mask, out.planet_ids
    )
    target_legal_mask_l = [
        _target_legal_mask_from_packed_legality_fields(
            legality_fields_l[k],
            parsed_list[k].planets,
            parsed_list[k].angular_velocity,
            parsed_list[k].comet_planet_ids,
        )
        for k in range(b_dim)
    ]
    target_legal_mask = _target_legal_mask_tensor(
        target_legal_mask_l, target_logits.device
    )
    target_logits = _apply_target_legal_mask(
        target_logits, target_legal_mask, launch, out.planet_owned_mask, out.planet_mask
    )
    launch_logits, launch = _mask_impossible_launches(
        launch_logits,
        launch,
        target_legal_mask,
        out.planet_owned_mask,
        out.planet_mask,
    )
    target_idx = _sample_target(target_logits, deterministic)

    fields_l = _packed_action_fields_from_legality_fields(
        target_idx, legality_fields_l, target_legal_mask_l
    )

    moves_list: list[list[Move]] = []
    records: list[SampleRecord] = []
    record_pos = (
        {int(row): pos for pos, row in enumerate(record_rows)}
        if record_rows is not None
        else None
    )
    materialized_rows: list[list[bool] | None] = (
        [None] * len(record_rows) if record_rows is not None else []
    )
    for k in range(b_dim):
        moves, materialized = _build_moves_from_packed_fields_with_mask(
            fields_l[k], parsed_list[k]
        )
        moves_list.append(moves)
        if record_pos is None:
            records.append(
                _record_from_materialized_launch(
                    launch[k],
                    target_idx[k],
                    frac[k],
                    launch_logits[k],
                    target_logits[k],
                    fraction_alpha[k],
                    fraction_beta[k],
                    materialized,
                )
            )
        elif k in record_pos:
            materialized_rows[record_pos[k]] = materialized
    if record_rows is not None:
        return moves_list, _batch_record_from_materialized_launch(
            launch,
            target_idx,
            frac,
            launch_logits,
            target_logits,
            fraction_alpha,
            fraction_beta,
            [row for row in materialized_rows if row is not None],
            record_rows,
        )
    return moves_list, records


def sample_batch_with_records_raw(
    out: PolicyOutput,
    raw_observations: list[Any],
    deterministic: bool = False,
    record_rows: Sequence[int] | None = None,
) -> tuple[list[list[list]], list[SampleRecord] | SampleBatchRecord]:
    """Batched sampler that builds Kaggle action lists from raw obs dicts."""
    launch_logits = out.launch_logits
    target_logits = out.target_logits
    fraction_alpha = out.fraction_alpha
    fraction_beta = out.fraction_beta
    b_dim, p, _ = target_logits.shape
    assert len(raw_observations) == b_dim, (len(raw_observations), b_dim)

    launch, frac = _sample_launch_fraction(
        launch_logits, fraction_alpha, fraction_beta, deterministic
    )
    legality_fields_l = _packed_legality_fields(
        launch, frac, out.planet_owned_mask, out.planet_mask, out.planet_ids
    )
    target_legal_mask_l = [
        _target_legal_mask_from_packed_legality_fields(
            legality_fields_l[k],
            raw_observations[k].get("planets", [])
            if isinstance(raw_observations[k], dict)
            else getattr(raw_observations[k], "planets", []),
            (
                raw_observations[k].get("angular_velocity", 0.0)
                if isinstance(raw_observations[k], dict)
                else getattr(raw_observations[k], "angular_velocity", 0.0)
            )
            or 0.0,
            raw_observations[k].get("comet_planet_ids", [])
            if isinstance(raw_observations[k], dict)
            else getattr(raw_observations[k], "comet_planet_ids", []),
        )
        for k in range(b_dim)
    ]
    target_legal_mask = _target_legal_mask_tensor(
        target_legal_mask_l, target_logits.device
    )
    target_logits = _apply_target_legal_mask(
        target_logits, target_legal_mask, launch, out.planet_owned_mask, out.planet_mask
    )
    launch_logits, launch = _mask_impossible_launches(
        launch_logits,
        launch,
        target_legal_mask,
        out.planet_owned_mask,
        out.planet_mask,
    )
    target_idx = _sample_target(target_logits, deterministic)
    fields_l = _packed_action_fields_from_legality_fields(
        target_idx, legality_fields_l, target_legal_mask_l
    )

    actions_list: list[list[list]] = []
    records: list[SampleRecord] = []
    record_pos = (
        {int(row): pos for pos, row in enumerate(record_rows)}
        if record_rows is not None
        else None
    )
    materialized_rows: list[list[bool] | None] = (
        [None] * len(record_rows) if record_rows is not None else []
    )
    for k in range(b_dim):
        actions, materialized = _build_action_lists_from_packed_fields_raw_with_mask(
            fields_l[k], raw_observations[k]
        )
        actions_list.append(actions)
        if record_pos is None:
            records.append(
                _record_from_materialized_launch(
                    launch[k],
                    target_idx[k],
                    frac[k],
                    launch_logits[k],
                    target_logits[k],
                    fraction_alpha[k],
                    fraction_beta[k],
                    materialized,
                )
            )
        elif k in record_pos:
            materialized_rows[record_pos[k]] = materialized
    if record_rows is not None:
        return actions_list, _batch_record_from_materialized_launch(
            launch,
            target_idx,
            frac,
            launch_logits,
            target_logits,
            fraction_alpha,
            fraction_beta,
            [row for row in materialized_rows if row is not None],
            record_rows,
        )
    return actions_list, records


def sample_batch_with_records_context(
    out: PolicyOutput,
    contexts: list[ActionContext],
    deterministic: bool = False,
    record_rows: Sequence[int] | None = None,
) -> tuple[list[list[list]], list[SampleRecord] | SampleBatchRecord]:
    """Batched sampler that builds action lists from fast env contexts."""
    launch_logits = out.launch_logits
    target_logits = out.target_logits
    fraction_alpha = out.fraction_alpha
    fraction_beta = out.fraction_beta
    b_dim, p, _ = target_logits.shape
    assert len(contexts) == b_dim, (len(contexts), b_dim)

    launch, frac = _sample_launch_fraction(
        launch_logits, fraction_alpha, fraction_beta, deterministic
    )
    legality_fields_l = _packed_legality_fields(
        launch, frac, out.planet_owned_mask, out.planet_mask, out.planet_ids
    )
    target_legal_mask_l = [
        _target_legal_mask_from_packed_legality_fields(
            legality_fields_l[k],
            contexts[k].planets,
            contexts[k].angular_velocity,
            contexts[k].comet_planet_ids,
        )
        for k in range(b_dim)
    ]
    target_legal_mask = _target_legal_mask_tensor(
        target_legal_mask_l, target_logits.device
    )
    target_logits = _apply_target_legal_mask(
        target_logits, target_legal_mask, launch, out.planet_owned_mask, out.planet_mask
    )
    launch_logits, launch = _mask_impossible_launches(
        launch_logits,
        launch,
        target_legal_mask,
        out.planet_owned_mask,
        out.planet_mask,
    )
    target_idx = _sample_target(target_logits, deterministic)
    fields_l = _packed_action_fields_from_legality_fields(
        target_idx, legality_fields_l, target_legal_mask_l
    )

    actions_list: list[list[list]] = []
    records: list[SampleRecord] = []
    record_pos = (
        {int(row): pos for pos, row in enumerate(record_rows)}
        if record_rows is not None
        else None
    )
    materialized_rows: list[list[bool] | None] = (
        [None] * len(record_rows) if record_rows is not None else []
    )
    for k in range(b_dim):
        actions, materialized = _build_action_lists_from_packed_fields_context_with_mask(
            fields_l[k], contexts[k]
        )
        actions_list.append(actions)
        if record_pos is None:
            records.append(
                _record_from_materialized_launch(
                    launch[k],
                    target_idx[k],
                    frac[k],
                    launch_logits[k],
                    target_logits[k],
                    fraction_alpha[k],
                    fraction_beta[k],
                    materialized,
                )
            )
        elif k in record_pos:
            materialized_rows[record_pos[k]] = materialized
    if record_rows is not None:
        return actions_list, _batch_record_from_materialized_launch(
            launch,
            target_idx,
            frac,
            launch_logits,
            target_logits,
            fraction_alpha,
            fraction_beta,
            [row for row in materialized_rows if row is not None],
            record_rows,
        )
    return actions_list, records


def sample_batch_actions(
    out: PolicyOutput,
    parsed_list: list[Observation],
    deterministic: bool = True,
) -> list[list[Move]]:
    """Batched moves-only sampler for eval and opponent inference paths."""
    launch_logits = out.launch_logits
    target_logits = out.target_logits
    fraction_alpha = out.fraction_alpha
    fraction_beta = out.fraction_beta
    b_dim, p, _ = target_logits.shape
    assert len(parsed_list) == b_dim, (len(parsed_list), b_dim)

    launch, frac = _sample_launch_fraction(
        launch_logits, fraction_alpha, fraction_beta, deterministic
    )
    legality_fields_l = _packed_legality_fields(
        launch, frac, out.planet_owned_mask, out.planet_mask, out.planet_ids
    )
    target_legal_mask_l = [
        _target_legal_mask_from_packed_legality_fields(
            legality_fields_l[k],
            parsed_list[k].planets,
            parsed_list[k].angular_velocity,
            parsed_list[k].comet_planet_ids,
        )
        for k in range(b_dim)
    ]
    target_legal_mask = _target_legal_mask_tensor(
        target_legal_mask_l, target_logits.device
    )
    target_logits = _apply_target_legal_mask(
        target_logits, target_legal_mask, launch, out.planet_owned_mask, out.planet_mask
    )
    _launch_logits, launch = _mask_impossible_launches(
        launch_logits,
        launch,
        target_legal_mask,
        out.planet_owned_mask,
        out.planet_mask,
    )
    target_idx = _sample_target(target_logits, deterministic)

    fields_l = _packed_action_fields_from_legality_fields(
        target_idx, legality_fields_l, target_legal_mask_l
    )

    return [
        _build_moves_from_packed_fields(fields_l[k], parsed_list[k])
        for k in range(b_dim)
    ]


def sample_batch_actions_raw(
    out: PolicyOutput,
    raw_observations: list[Any],
    deterministic: bool = True,
) -> list[list[list]]:
    """Batched moves-only sampler for raw Kaggle-style observations."""
    launch_logits = out.launch_logits
    target_logits = out.target_logits
    fraction_alpha = out.fraction_alpha
    fraction_beta = out.fraction_beta
    b_dim, p, _ = target_logits.shape
    assert len(raw_observations) == b_dim, (len(raw_observations), b_dim)

    launch, frac = _sample_launch_fraction(
        launch_logits, fraction_alpha, fraction_beta, deterministic
    )
    legality_fields_l = _packed_legality_fields(
        launch, frac, out.planet_owned_mask, out.planet_mask, out.planet_ids
    )
    target_legal_mask_l = [
        _target_legal_mask_from_packed_legality_fields(
            legality_fields_l[k],
            raw_observations[k].get("planets", [])
            if isinstance(raw_observations[k], dict)
            else getattr(raw_observations[k], "planets", []),
            (
                raw_observations[k].get("angular_velocity", 0.0)
                if isinstance(raw_observations[k], dict)
                else getattr(raw_observations[k], "angular_velocity", 0.0)
            )
            or 0.0,
            raw_observations[k].get("comet_planet_ids", [])
            if isinstance(raw_observations[k], dict)
            else getattr(raw_observations[k], "comet_planet_ids", []),
        )
        for k in range(b_dim)
    ]
    target_legal_mask = _target_legal_mask_tensor(
        target_legal_mask_l, target_logits.device
    )
    target_logits = _apply_target_legal_mask(
        target_logits, target_legal_mask, launch, out.planet_owned_mask, out.planet_mask
    )
    _launch_logits, launch = _mask_impossible_launches(
        launch_logits,
        launch,
        target_legal_mask,
        out.planet_owned_mask,
        out.planet_mask,
    )
    target_idx = _sample_target(target_logits, deterministic)

    fields_l = _packed_action_fields_from_legality_fields(
        target_idx, legality_fields_l, target_legal_mask_l
    )
    return [
        _build_action_lists_from_packed_fields_raw(
            fields_l[k], raw_observations[k]
        )
        for k in range(b_dim)
    ]


def sample_batch_actions_context(
    out: PolicyOutput,
    contexts: list[ActionContext],
    deterministic: bool = True,
) -> list[list[list]]:
    """Batched moves-only sampler for fast env action contexts."""
    launch_logits = out.launch_logits
    target_logits = out.target_logits
    fraction_alpha = out.fraction_alpha
    fraction_beta = out.fraction_beta
    b_dim, p, _ = target_logits.shape
    assert len(contexts) == b_dim, (len(contexts), b_dim)

    launch, frac = _sample_launch_fraction(
        launch_logits, fraction_alpha, fraction_beta, deterministic
    )
    legality_fields_l = _packed_legality_fields(
        launch, frac, out.planet_owned_mask, out.planet_mask, out.planet_ids
    )
    target_legal_mask_l = [
        _target_legal_mask_from_packed_legality_fields(
            legality_fields_l[k],
            contexts[k].planets,
            contexts[k].angular_velocity,
            contexts[k].comet_planet_ids,
        )
        for k in range(b_dim)
    ]
    target_legal_mask = _target_legal_mask_tensor(
        target_legal_mask_l, target_logits.device
    )
    target_logits = _apply_target_legal_mask(
        target_logits, target_legal_mask, launch, out.planet_owned_mask, out.planet_mask
    )
    _launch_logits, launch = _mask_impossible_launches(
        launch_logits,
        launch,
        target_legal_mask,
        out.planet_owned_mask,
        out.planet_mask,
    )
    target_idx = _sample_target(target_logits, deterministic)

    fields_l = _packed_action_fields_from_legality_fields(
        target_idx, legality_fields_l, target_legal_mask_l
    )
    return [
        _build_action_lists_from_packed_fields_context(
            fields_l[k], contexts[k]
        )
        for k in range(b_dim)
    ]
