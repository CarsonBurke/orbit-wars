"""Convert raw `PolicyOutput` into a list of legal `Move`s.

The PPO policy emits, per owned planet:
  - a masked Categorical over noop + target planets
  - parameter-golf-style softcapped action logits
  - a Beta distribution on (0, 1) for the fraction-of-garrison to send,
    conditional on launch

The simulator's action format is `[from_planet_id, angle_radians, num_ships]`.
Fleets fly in *straight lines* at constant speed (`fleet_speed(num_ships)`),
so the launch angle is fully determined by the chosen target — there is no
mid-flight steering. We compute the angle deterministically by solving the
intercept equation in closed form (see `_lead_angle`) — no fixed-point
iteration that might oscillate.

For PPO we need, *per owned planet*, the noop/target Categorical log-prob
plus the conditional fraction log-prob of the actually-sampled action.
`sample_with_record` returns those alongside the moves; `sample_actions` is
the thin moves-only wrapper used by inference paths that don't care about
log-probs. The recorded PPO fraction is the native Beta sample executed by
the simulator; log-prob recomputation evaluates that same bounded-support
value directly. The SAC adapter still uses the legacy launch and squashed-
Normal path through the same helpers when `PolicyOutput.action_logit_softcap`
is unset.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn.functional as nn_functional

from ..game import angle_to
from ..game.observation import Observation
from ..game.physics import fleet_speed
from ..game.types import BOARD_SIZE, CENTER, ROTATION_RADIUS_LIMIT, SUN_RADIUS, Move
from .model import PolicyOutput

# Numerical floors for Gumbel sampling, Beta endpoints, and atanh inversion.
SAMPLE_EPS: float = 1e-7
SQUASH_EPS: float = 1e-6
BETA_SAMPLE_EPS: float = 1e-6


def _threshold_normal_launch_prob(
    mean: torch.Tensor,
    log_std: torch.Tensor | None,
    prob_floor: float = 0.0,
) -> torch.Tensor:
    if log_std is None:
        prob = mean.float().sigmoid()
    else:
        score = mean.float() * torch.exp(-log_std.float())
        prob = 0.5 * (1.0 + torch.erf(score / math.sqrt(2.0)))
    floor = float(prob_floor)
    if floor <= 0.0:
        return prob
    if floor >= 0.5:
        raise ValueError("prob_floor must be < 0.5")
    return floor + (1.0 - 2.0 * floor) * prob


def _threshold_normal_launch_log_prob(
    mean: torch.Tensor,
    log_std: torch.Tensor | None,
    launch: torch.Tensor,
    prob_floor: float = 0.0,
) -> torch.Tensor:
    if prob_floor > 0.0:
        prob = _threshold_normal_launch_prob(mean, log_std, prob_floor).clamp(
            1e-7,
            1.0 - 1e-7,
        )
        return torch.where(launch.float() > 0.5, prob.log(), torch.log1p(-prob))
    if log_std is None:
        return -nn_functional.binary_cross_entropy_with_logits(
            mean.float(),
            launch.float(),
            reduction="none",
        )
    score = mean.float() * torch.exp(-log_std.float())
    log_p1 = torch.special.log_ndtr(score)
    log_p0 = torch.special.log_ndtr(-score)
    return torch.where(launch.float() > 0.5, log_p1, log_p0)


def _threshold_normal_launch_entropy(
    mean: torch.Tensor,
    log_std: torch.Tensor | None,
    prob_floor: float = 0.0,
) -> torch.Tensor:
    p = _threshold_normal_launch_prob(mean, log_std, prob_floor).clamp(
        1e-7,
        1.0 - 1e-7,
    )
    return -(p * p.log() + (1.0 - p) * (1.0 - p).log())


def _categorical_action_logits(
    noop_logits: torch.Tensor,
    target_logits: torch.Tensor,
    action_logit_softcap: float | None,
) -> torch.Tensor:
    target_logits_f = target_logits.float()
    noop_logits_f = noop_logits.float()
    target_finite = torch.isfinite(target_logits_f)
    if action_logit_softcap is not None:
        softcap = float(action_logit_softcap)
        if softcap <= 0.0:
            raise ValueError("action_logit_softcap must be positive")
        target_logits_f = torch.where(
            target_finite,
            softcap * torch.tanh(target_logits_f / softcap),
            torch.full_like(target_logits_f, float("-inf")),
        )
        noop_finite = torch.isfinite(noop_logits_f)
        noop_logits_f = torch.where(
            noop_finite,
            softcap * torch.tanh(noop_logits_f / softcap),
            torch.full_like(noop_logits_f, -1.0e9),
        )
    else:
        target_logits_f = torch.where(
            target_finite,
            target_logits_f,
            torch.full_like(target_logits_f, float("-inf")),
        )
        noop_logits_f = torch.where(
            torch.isfinite(noop_logits_f),
            noop_logits_f,
            torch.full_like(noop_logits_f, -1.0e9),
        )
    # No `- log(target_count)` reweighting: the count-normalization used to be a
    # per-state offset baked into the action logits, which (a) corrupted the
    # deterministic mode toward no-op (every launch sat ~log(N) below noop) and
    # (b) is really a fixed prior on launch propensity. That prior now lives in
    # the model's learnable `noop_logit_bias` (initialized to ~log(N_typical)),
    # so the categorical mode is a clean argmax(noop, target_i) and the policy
    # learns the no-op/launch balance directly.
    action_logits = torch.cat(
        (noop_logits_f.unsqueeze(-1), target_logits_f),
        dim=-1,
    )
    return action_logits


def _categorical_action_log_probs(
    noop_logits: torch.Tensor,
    target_logits: torch.Tensor,
    action_logit_softcap: float | None,
    *,
    deterministic: bool = False,
) -> torch.Tensor:
    del deterministic
    action_logits = _categorical_action_logits(
        noop_logits,
        target_logits,
        action_logit_softcap,
    )
    return torch.log_softmax(action_logits, dim=-1)


def _deterministic_squashed_normal_fraction(
    fraction_mean: torch.Tensor,
    fraction_log_std: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return the deterministic fraction represented by the squashed Normal.

    `fraction_log_std` is accepted for API symmetry with the stochastic path.
    """
    del fraction_log_std
    return (0.5 * (torch.tanh(fraction_mean.float()) + 1.0)).clamp(
        SQUASH_EPS, 1.0 - SQUASH_EPS
    )


def _deterministic_beta_fraction(
    fraction_alpha: torch.Tensor,
    fraction_beta: torch.Tensor,
) -> torch.Tensor:
    alpha = fraction_alpha.float()
    beta = fraction_beta.float()
    interior_mode = (alpha - 1.0) / (alpha + beta - 2.0).clamp_min(BETA_SAMPLE_EPS)
    lower_mode = torch.full_like(alpha, BETA_SAMPLE_EPS)
    upper_mode = torch.full_like(alpha, 1.0 - BETA_SAMPLE_EPS)
    uniform_fallback = torch.full_like(alpha, 0.5)
    mode = torch.where(
        (alpha > 1.0) & (beta > 1.0),
        interior_mode,
        torch.where(
            (alpha <= 1.0) & (beta > 1.0),
            lower_mode,
            torch.where(
                (alpha > 1.0) & (beta <= 1.0),
                upper_mode,
                uniform_fallback,
            ),
        ),
    )
    return mode.clamp(BETA_SAMPLE_EPS, 1.0 - BETA_SAMPLE_EPS)


def _deterministic_fraction(
    fraction_param1: torch.Tensor,
    fraction_param2: torch.Tensor | None = None,
    *,
    fraction_dist: str = "beta",
) -> torch.Tensor:
    if fraction_dist == "beta":
        if fraction_param2 is None:
            raise ValueError("Beta fraction requires alpha and beta tensors")
        return _deterministic_beta_fraction(fraction_param1, fraction_param2)
    if fraction_dist == "squashed_normal":
        return _deterministic_squashed_normal_fraction(fraction_param1, fraction_param2)
    raise ValueError(f"unknown fraction_dist={fraction_dist!r}")


def _beta_log_prob(
    alpha: torch.Tensor,
    beta: torch.Tensor,
    fraction: torch.Tensor,
) -> torch.Tensor:
    x = fraction.float().clamp(BETA_SAMPLE_EPS, 1.0 - BETA_SAMPLE_EPS)
    alpha = alpha.float()
    beta = beta.float()
    log_norm = torch.lgamma(alpha) + torch.lgamma(beta) - torch.lgamma(alpha + beta)
    return (alpha - 1.0) * torch.log(x) + (beta - 1.0) * torch.log1p(-x) - log_norm


def _beta_entropy(alpha: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
    alpha = alpha.float()
    beta = beta.float()
    total = alpha + beta
    log_norm = torch.lgamma(alpha) + torch.lgamma(beta) - torch.lgamma(total)
    return (
        log_norm
        - (alpha - 1.0) * torch.digamma(alpha)
        - (beta - 1.0) * torch.digamma(beta)
        + (total - 2.0) * torch.digamma(total)
    )


def _squashed_normal_log_prob(
    mean: torch.Tensor,
    log_std: torch.Tensor,
    fraction: torch.Tensor,
) -> torch.Tensor:
    """CleanRL/SAC-style log-prob for fraction = 0.5 * (tanh(z) + 1).

    The affine half-range term is constant and ratio-invariant, so like the
    referenced CleanRL implementation we omit it from both sample-time and
    PPO recompute log-probs.
    """
    u = (2.0 * fraction.float() - 1.0).clamp(
        -1.0 + SQUASH_EPS, 1.0 - SQUASH_EPS
    )
    z = torch.atanh(u)
    log_std = log_std.float()
    inv_std = torch.exp(-log_std)
    log_prob_z = (
        -0.5 * ((z - mean.float()) * inv_std).square()
        - log_std
        - 0.5 * math.log(2.0 * math.pi)
    )
    squash_correction = torch.log(1.0 - u.square() + SQUASH_EPS)
    return log_prob_z - squash_correction


def _squashed_normal_entropy(log_std: torch.Tensor) -> torch.Tensor:
    """Approximate squashed entropy with the unsquashed Normal entropy."""
    return log_std.float() + 0.5 * (1.0 + math.log(2.0 * math.pi))


def _fraction_log_prob(
    fraction_param1: torch.Tensor,
    fraction_param2: torch.Tensor,
    fraction: torch.Tensor,
    fraction_dist: str,
) -> torch.Tensor:
    if fraction_dist == "beta":
        return _beta_log_prob(fraction_param1, fraction_param2, fraction)
    if fraction_dist == "squashed_normal":
        return _squashed_normal_log_prob(fraction_param1, fraction_param2, fraction)
    raise ValueError(f"unknown fraction_dist={fraction_dist!r}")


def _launched_fraction_log_prob(
    fraction_param1: torch.Tensor,
    fraction_param2: torch.Tensor,
    fraction: torch.Tensor,
    fraction_dist: str,
    launch: torch.Tensor,
) -> torch.Tensor:
    launch_mask = launch.float() > 0.5
    if fraction_dist == "beta":
        safe_param1 = torch.where(
            launch_mask,
            fraction_param1,
            torch.full_like(fraction_param1, 2.0),
        )
        safe_param2 = torch.where(
            launch_mask,
            fraction_param2,
            torch.full_like(fraction_param2, 2.0),
        )
    elif fraction_dist == "squashed_normal":
        safe_param1 = torch.where(
            launch_mask,
            fraction_param1,
            torch.zeros_like(fraction_param1),
        )
        safe_param2 = torch.where(
            launch_mask,
            fraction_param2,
            torch.zeros_like(fraction_param2),
        )
    else:
        raise ValueError(f"unknown fraction_dist={fraction_dist!r}")
    safe_fraction = torch.where(
        launch_mask,
        fraction,
        torch.full_like(fraction, 0.5),
    )
    frac_lp = _fraction_log_prob(safe_param1, safe_param2, safe_fraction, fraction_dist)
    return torch.where(launch_mask, frac_lp, torch.zeros_like(frac_lp))


def _action_log_prob_for_action(action_lp: torch.Tensor, launch: torch.Tensor) -> torch.Tensor:
    del launch
    return action_lp


def _fraction_entropy(
    fraction_param1: torch.Tensor,
    fraction_param2: torch.Tensor,
    fraction_dist: str,
) -> torch.Tensor:
    if fraction_dist == "beta":
        return _beta_entropy(fraction_param1, fraction_param2)
    if fraction_dist == "squashed_normal":
        return _squashed_normal_entropy(fraction_param2)
    raise ValueError(f"unknown fraction_dist={fraction_dist!r}")


def _policy_fraction_params(out: PolicyOutput) -> tuple[torch.Tensor, torch.Tensor, str]:
    if out.fraction_alpha is not None and out.fraction_beta is not None:
        return out.fraction_alpha, out.fraction_beta, "beta"
    if out.fraction_mean is not None and out.fraction_log_std is not None:
        return out.fraction_mean, out.fraction_log_std, "squashed_normal"
    raise ValueError("PolicyOutput requires either Beta or squashed-Normal fraction params")


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
LEAD_MAX_SCAN_DISTANCE: float = math.hypot(BOARD_SIZE, BOARD_SIZE) + 8.0


@dataclass
class SampleRecord:
    """Per-planet record of the sampled action — used by PPO rollouts.

    The target legality mask is recorded so PPO can recompute the current
    policy log-prob under the same action support used during rollout.

    The fraction is the post-squash action, so PPO carries only
    `fraction` ∈ (eps, 1-eps); recomputing `log_prob` recovers the
    pre-squash latent with atanh.
    """

    launch: torch.Tensor       # [P] float 0/1 — PPO action label: categorical path keeps the sampled target/no-op decision; legacy path uses materialized launch
    raw_launch: torch.Tensor   # [P] float 0/1 — the raw launch sample (masked only for unowned / no-legal-target sources); the MDP action SAC's actor optimizes and critic conditions on
    target_idx: torch.Tensor   # [P] long, in [0, P)
    fraction: torch.Tensor     # [P] float in (eps, 1-eps) — executed squashed fraction
    log_prob: torch.Tensor     # [P] float — launch event + launch*(Categorical + fraction)
    target_legal_mask: torch.Tensor   # [P, P] bool


@dataclass
class SampleBatchRecord:
    """Batched PPO record for vector rollouts.

    Same fields as `SampleRecord`, with a leading row dimension. This avoids
    constructing one Python object and one small torch graph per bucket row in
    the rollout hot path.
    """

    launch: torch.Tensor       # [B, P] PPO action label; see `SampleRecord.launch`
    raw_launch: torch.Tensor   # [B, P] raw masked launch sample (SAC)
    target_idx: torch.Tensor
    fraction: torch.Tensor
    log_prob: torch.Tensor
    target_legal_mask: torch.Tensor | None


@dataclass(slots=True)
class _PreparedBatchActions:
    launch: torch.Tensor
    target_idx: torch.Tensor
    frac: torch.Tensor
    launch_logits: torch.Tensor
    launch_log_std: torch.Tensor | None
    action_logit_softcap: float | None
    target_logits: torch.Tensor
    fraction_param1: torch.Tensor
    fraction_param2: torch.Tensor
    fraction_dist: str
    legality_fields_l: Any
    target_legal_mask_l: Any


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


def _route_blocker_orbit_state(
    x: float,
    y: float,
    radius: float,
    angular_velocity: float,
    *,
    is_comet: bool,
) -> tuple[float, float]:
    if is_comet or abs(angular_velocity) <= 1e-12:
        return 0.0, 0.0
    orbit_radius = math.hypot(x - CENTER[0], y - CENTER[1])
    if orbit_radius + radius >= ROTATION_RADIUS_LIMIT or orbit_radius <= 1e-9:
        return 0.0, 0.0
    return orbit_radius, math.atan2(y - CENTER[1], x - CENTER[0])


def _route_blocker_position(
    x: float,
    y: float,
    orbit_radius: float,
    orbit_theta0: float,
    angular_velocity: float,
    steps: int,
) -> tuple[float, float]:
    if orbit_radius <= 0.0 or steps == 0:
        return x, y
    theta = orbit_theta0 + angular_velocity * steps
    return (
        CENTER[0] + orbit_radius * math.cos(theta),
        CENTER[1] + orbit_radius * math.sin(theta),
    )


def _route_clear_to_solution(
    source_id: int,
    target_id: int,
    source_x: float,
    source_y: float,
    source_radius: float,
    solution: LeadSolution,
    send: int,
    blockers: Sequence[tuple[int, float, float, float, float, float]],
    angular_velocity: float,
) -> bool:
    speed = fleet_speed(send)
    if speed <= 0.0:
        return False
    start_x, start_y = _launch_start(source_x, source_y, source_radius, solution.angle)
    direction_x = math.cos(solution.angle)
    direction_y = math.sin(solution.angle)
    final_turn = max(1, int(math.ceil(solution.time)))
    final_x = start_x + direction_x * speed * final_turn
    final_y = start_y + direction_y * speed * final_turn
    if not _is_inside_board(start_x, start_y) or not _is_inside_board(final_x, final_y):
        return False
    if _segment_crosses_sun(start_x, start_y, final_x, final_y):
        return False

    moving: list[tuple[int, float, float, float, float, float]] = []
    for blocker in blockers:
        blocker_id, x, y, radius, orbit_radius, _orbit_theta0 = blocker
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
    if not moving:
        return True

    for turn in range(1, final_turn + 1):
        old_x = start_x + direction_x * speed * (turn - 1)
        old_y = start_y + direction_y * speed * (turn - 1)
        new_x = start_x + direction_x * speed * turn
        new_y = start_y + direction_y * speed * turn
        for blocker_id, x, y, radius, orbit_radius, orbit_theta0 in moving:
            bx, by = _route_blocker_position(
                x, y, orbit_radius, orbit_theta0, angular_velocity, turn - 1
            )
            if blocker_id != source_id and (
                _point_to_segment_distance(bx, by, old_x, old_y, new_x, new_y) < radius
            ):
                return False
            if turn < final_turn:
                nbx, nby = _route_blocker_position(
                    x, y, orbit_radius, orbit_theta0, angular_velocity, turn
                )
                if _point_to_segment_distance(new_x, new_y, bx, by, nbx, nby) < radius:
                    return False
    return True


def _route_blockers_from_rows(
    planets: Any,
    angular_velocity: float,
    comet_planet_ids: Any,
) -> list[tuple[int, float, float, float, float, float]]:
    comet_ids = {int(pid) for pid in comet_planet_ids}
    blockers = []
    for planet in planets:
        pid = _planet_id(planet)
        x = _planet_x(planet)
        y = _planet_y(planet)
        radius = _planet_radius(planet)
        blockers.append(
            (
                pid,
                x,
                y,
                radius,
                *_route_blocker_orbit_state(
                    x, y, radius, angular_velocity, is_comet=pid in comet_ids
                ),
            )
        )
    return blockers


def _lead_solution_from_point(
    mine_x: float,
    mine_y: float,
    target_x: float,
    target_y: float,
    target_radius: float,
    angular_velocity: float,
    send: int,
    *,
    target_is_comet: bool = False,
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
        not target_is_comet
        and orbit_radius + target_radius < ROTATION_RADIUS_LIMIT
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
    max_turns = min(
        LEAD_MAX_TURNS,
        max(1, int(math.ceil((LEAD_MAX_SCAN_DISTANCE + target_radius) / sp)) + 1),
    )
    for k in range(1, max_turns + 1):
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
    *,
    target_is_comet: bool = False,
) -> float | None:
    solution = _lead_solution_from_point(
        mine_x,
        mine_y,
        target_x,
        target_y,
        target_radius,
        angular_velocity,
        send,
        target_is_comet=target_is_comet,
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
    *,
    target_is_comet: bool = False,
) -> LeadSolution | None:
    """First-intercept launch solution for the official action semantics.

    The simulator does not spawn a fleet at the source center. It starts the
    fleet just outside the planet along the submitted angle, so the start
    point itself depends on the angle. Iterating the point-solver a few times
    accounts for that offset and prevents small-radius targets from being
    missed by a centerline shot.
    """
    solution = _lead_solution_from_point(
        mine_x,
        mine_y,
        target_x,
        target_y,
        target_radius,
        angular_velocity,
        send,
        target_is_comet=target_is_comet,
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
            target_is_comet=target_is_comet,
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
    *,
    target_is_comet: bool = False,
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
        target_is_comet=target_is_comet,
    )
    return None if solution is None else solution.angle


def _ships_to_send(remaining: int, frac: float) -> int:
    """Ships launched from a planet for a given launch fraction.

    Mirrors the Rust simulator's ``ships_to_send`` exactly, including its
    round-half-away-from-zero rule (Rust ``f64::round``). Python's built-in
    ``round`` is round-half-to-even, which diverges from Rust at exact
    half-ship boundaries (e.g. ``round(24.5)`` is 24 in Python but 25 in
    Rust). Training rolls out through the Rust env while the submission
    bundle launches through this Python path, so any mismatch here is a
    silent train/serve skew in the launched ship count.
    """
    if remaining < 2:
        return 0
    clamped = 0.0 if frac < 0.0 else 1.0 if frac > 1.0 else frac
    scaled = remaining * clamped
    raw = math.floor(scaled)
    # scaled - floor(scaled) is exact in IEEE-754, so the >= 0.5 test
    # reproduces round-half-away-from-zero for these non-negative values.
    if scaled - raw >= 0.5:
        raw += 1
    return max(1, min(remaining - 1, raw))


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
    blockers = [
        (
            int(pl.id),
            float(pl.x),
            float(pl.y),
            float(pl.radius),
            *_route_blocker_orbit_state(
                float(pl.x),
                float(pl.y),
                float(pl.radius),
                omega,
                is_comet=int(pl.id) in o.comet_planet_ids,
            ),
        )
        for pl in o.planets
    ]
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
        if target_id < 0:
            continue
        mine = by_id.get(ids_l[i])
        target = by_id.get(target_id)
        if mine is None or target is None:
            continue
        remaining = remaining_by_id.get(mine.id, int(mine.ships))
        if remaining < 2:
            continue

        f = max(0.0, min(1.0, frac_l[i]))
        send = _ships_to_send(remaining, f)
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
            target_is_comet=int(target.id) in o.comet_planet_ids,
        )
        if solution is None:
            continue  # solver couldn't find a feasible intercept — silently no-op
        if not _route_clear_to_solution(
            mine.id,
            target.id,
            mine.x,
            mine.y,
            mine.radius,
            solution,
            send,
            blockers,
            omega,
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
    blockers = _route_blockers_from_rows(planets, omega, comet_ids)
    target_fields = [
        (j, planet_fields[target_id])
        for j, target_id in enumerate(int(v) for v in ids_l)
        if bool(pmask_l[j])
        and target_id >= 0
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
        send = _ships_to_send(remaining, frac)
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
                target_is_comet=int(ids_l[j]) in comet_ids,
            )
            if solution is None:
                continue
            if not _route_clear_to_solution(
                int(ids_l[i]),
                int(ids_l[j]),
                source_x,
                source_y,
                source_radius,
                solution,
                send,
                blockers,
                omega,
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
    blockers = _route_blockers_from_rows(planets, omega, comet_ids)
    target_fields = [
        (j, planet_fields[target_id])
        for j, target_id in enumerate(ids)
        if present[j]
        and target_id >= 0
        and target_id in planet_fields
    ]

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
        send = _ships_to_send(remaining, frac)
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
                target_is_comet=int(ids[j]) in comet_ids,
            )
            if solution is None:
                continue
            if not _route_clear_to_solution(
                ids[i],
                ids[j],
                source_x,
                source_y,
                source_radius,
                solution,
                send,
                blockers,
                omega,
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
    # Gate the sampled ACTION to legal launch sources only: zero it on every slot
    # that is not an owned, present source WITH a legal target — i.e. owned-but-
    # no-legal-target (as before) AND non-owned / padded slots (whose Bernoulli
    # draw is otherwise left at ~0.5). Moves are built from a path that already
    # filters non-source slots, so this changes no materialized move; it only
    # makes `launch`/`raw_launch` (SAC's stored action and the materialize_gap
    # metric) reflect real owned-source launches. `launch_logits` keep their
    # original masking, so PPO's log-prob is unchanged (it never reads raw_launch).
    launch = launch * (source & has_legal_target).to(launch.dtype)
    return launch_logits, launch


def _sample_launch_fraction(
    launch_logits: torch.Tensor,
    fraction_param1: torch.Tensor,
    fraction_param2: torch.Tensor,
    deterministic: bool,
    fraction_dist: str = "beta",
    launch_log_std: torch.Tensor | None = None,
    launch_prob_floor: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    launch_logits = launch_logits.float()
    launch_log_std = None if launch_log_std is None else launch_log_std.float()
    fraction_param1 = fraction_param1.float()
    fraction_param2 = fraction_param2.float()
    if deterministic:
        launch = (launch_logits > 0.0).to(fraction_param1.dtype)
        frac = _deterministic_fraction(
            fraction_param1,
            fraction_param2,
            fraction_dist=fraction_dist,
        )
        return launch, frac
    launch_prob = _threshold_normal_launch_prob(
        launch_logits,
        launch_log_std,
        launch_prob_floor,
    )
    launch = (torch.rand_like(launch_logits) < launch_prob).to(fraction_param1.dtype)
    if fraction_dist == "beta":
        with torch.no_grad():
            dist = torch.distributions.Beta(fraction_param1, fraction_param2)
            frac = dist.sample().clamp(BETA_SAMPLE_EPS, 1.0 - BETA_SAMPLE_EPS)
        return launch, frac
    if fraction_dist == "squashed_normal":
        with torch.no_grad():
            z = fraction_param1 + torch.exp(fraction_param2) * torch.randn_like(
                fraction_param1
            )
        frac = (0.5 * (torch.tanh(z) + 1.0)).clamp(
            SQUASH_EPS, 1.0 - SQUASH_EPS
        )
        return launch, frac
    raise ValueError(f"unknown fraction_dist={fraction_dist!r}")


def _sample_fraction(
    fraction_param1: torch.Tensor,
    fraction_param2: torch.Tensor,
    deterministic: bool,
    fraction_dist: str = "beta",
) -> torch.Tensor:
    fraction_param1 = fraction_param1.float()
    fraction_param2 = fraction_param2.float()
    if deterministic:
        return _deterministic_fraction(
            fraction_param1,
            fraction_param2,
            fraction_dist=fraction_dist,
        )
    if fraction_dist == "beta":
        with torch.no_grad():
            dist = torch.distributions.Beta(fraction_param1, fraction_param2)
            return dist.sample().clamp(BETA_SAMPLE_EPS, 1.0 - BETA_SAMPLE_EPS)
    if fraction_dist == "squashed_normal":
        with torch.no_grad():
            z = fraction_param1 + torch.exp(fraction_param2) * torch.randn_like(
                fraction_param1
            )
        return (0.5 * (torch.tanh(z) + 1.0)).clamp(
            SQUASH_EPS, 1.0 - SQUASH_EPS
        )
    raise ValueError(f"unknown fraction_dist={fraction_dist!r}")


def _sample_target(
    target_logits: torch.Tensor,
    deterministic: bool,
) -> torch.Tensor:
    safe_target_logits = _safe_target_logits(target_logits.float())
    if deterministic:
        return safe_target_logits.argmax(dim=-1)
    uniform = torch.rand_like(safe_target_logits).clamp_(SAMPLE_EPS, 1.0 - SAMPLE_EPS)
    gumbel = -torch.log(-torch.log(uniform))
    return (safe_target_logits + gumbel).argmax(dim=-1)


def _sample_categorical_action(
    noop_logits: torch.Tensor,
    target_logits: torch.Tensor,
    action_logit_softcap: float,
    owned: torch.Tensor,
    pmask: torch.Tensor,
    deterministic: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    source = (owned.to(dtype=torch.bool) & pmask.to(dtype=torch.bool)).to(
        device=target_logits.device
    )
    target_logits = target_logits.float().masked_fill(~source.unsqueeze(-1), float("-inf"))
    action_logits = _categorical_action_logits(
        noop_logits,
        target_logits,
        action_logit_softcap,
    )
    if deterministic:
        action_idx = action_logits.argmax(dim=-1)
        target_idx = (action_idx - 1).clamp_min(0)
        launch = (action_idx > 0).to(noop_logits.dtype)
        launch = launch * source.to(dtype=launch.dtype)
        return launch, target_idx
    else:
        uniform = torch.rand_like(action_logits).clamp_(SAMPLE_EPS, 1.0 - SAMPLE_EPS)
        gumbel = -torch.log(-torch.log(uniform))
        action_idx = (action_logits + gumbel).argmax(dim=-1)
    launch = (action_idx > 0).to(noop_logits.dtype)
    target_idx = (action_idx - 1).clamp_min(0)
    return launch, target_idx


def _prepare_batch_action_fields(
    out: PolicyOutput,
    batch_len: int,
    deterministic: bool,
    target_mask_builder: Any,
) -> _PreparedBatchActions:
    """Sample factored actions and apply per-env target legality masks.

    The three batched front doors differ only in where planet rows come from
    (parsed observations, raw Kaggle dicts, or native-env contexts). This
    helper keeps their shared tensor path identical.
    """
    launch_logits = out.launch_logits
    launch_log_std = out.launch_log_std
    action_logit_softcap = out.action_logit_softcap
    target_logits = out.target_logits
    fraction_param1, fraction_param2, fraction_dist = _policy_fraction_params(out)
    b_dim, _p, _ = target_logits.shape
    assert batch_len == b_dim, (batch_len, b_dim)

    if action_logit_softcap is None:
        launch, frac = _sample_launch_fraction(
            launch_logits,
            fraction_param1,
            fraction_param2,
            deterministic,
            fraction_dist,
            launch_log_std=launch_log_std,
            launch_prob_floor=out.launch_prob_floor,
        )
        legality_launch = launch
    else:
        frac = _sample_fraction(
            fraction_param1,
            fraction_param2,
            deterministic,
            fraction_dist,
        )
        support_launch = (out.planet_owned_mask & out.planet_mask).to(
            dtype=frac.dtype,
            device=frac.device,
        )
        legality_launch = support_launch

    legality_fields_l = _packed_legality_fields(
        legality_launch,
        frac,
        out.planet_owned_mask,
        out.planet_mask,
        out.planet_ids,
    )
    target_legal_mask_l = [
        target_mask_builder(k, legality_fields_l[k]) for k in range(b_dim)
    ]
    target_legal_mask = _target_legal_mask_tensor(
        target_legal_mask_l, target_logits.device
    )
    target_logits = _apply_target_legal_mask(
        target_logits,
        target_legal_mask,
        legality_launch,
        out.planet_owned_mask,
        out.planet_mask,
    )

    if action_logit_softcap is None:
        launch_logits, launch = _mask_impossible_launches(
            launch_logits,
            launch,
            target_legal_mask,
            out.planet_owned_mask,
            out.planet_mask,
        )
        target_idx = _sample_target(target_logits, deterministic)
    else:
        launch, target_idx = _sample_categorical_action(
            launch_logits,
            target_logits,
            action_logit_softcap,
            out.planet_owned_mask,
            out.planet_mask,
            deterministic,
        )

    legality_fields_l = _packed_legality_fields(
        launch,
        frac,
        out.planet_owned_mask,
        out.planet_mask,
        out.planet_ids,
    )
    return _PreparedBatchActions(
        launch=launch,
        target_idx=target_idx,
        frac=frac,
        launch_logits=launch_logits,
        launch_log_std=launch_log_std,
        action_logit_softcap=action_logit_softcap,
        target_logits=target_logits,
        fraction_param1=fraction_param1,
        fraction_param2=fraction_param2,
        fraction_dist=fraction_dist,
        legality_fields_l=legality_fields_l,
        target_legal_mask_l=target_legal_mask_l,
    )


def _build_moves_from_packed_fields_with_mask(
    fields_l: list[list[float]],
    o: Observation,
) -> tuple[list[Move], list[bool]]:
    moves: list[Move] = []
    materialized = [False] * len(fields_l)
    by_id = {pl.id: pl for pl in o.planets}
    remaining_by_id = {pl.id: int(pl.ships) for pl in o.planets}
    omega = o.angular_velocity
    blockers = [
        (
            int(pl.id),
            float(pl.x),
            float(pl.y),
            float(pl.radius),
            *_route_blocker_orbit_state(
                float(pl.x),
                float(pl.y),
                float(pl.radius),
                omega,
                is_comet=int(pl.id) in o.comet_planet_ids,
            ),
        )
        for pl in o.planets
    ]
    p = len(fields_l)
    for i in _candidate_action_indices(fields_l):
        i = int(i)
        fields = fields_l[i]
        ti = int(fields[0])
        if ti == i:
            continue
        target_id = int(fields_l[ti][5]) if 0 <= ti < p else -1
        if target_id < 0:
            continue
        mine = by_id.get(int(fields[5]))
        target = by_id.get(target_id)
        if mine is None or target is None:
            continue
        remaining = remaining_by_id.get(mine.id, int(mine.ships))
        if remaining < 2:
            continue

        f = max(0.0, min(1.0, float(fields[1])))
        send = _ships_to_send(remaining, f)
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
            target_is_comet=int(target.id) in o.comet_planet_ids,
        )
        if solution is None:
            continue
        if not _route_clear_to_solution(
            mine.id,
            target.id,
            mine.x,
            mine.y,
            mine.radius,
            solution,
            send,
            blockers,
            omega,
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
    blockers = _route_blockers_from_rows(planets, omega, comet_planet_ids)
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
        if target_id < 0:
            continue
        mine = by_id.get(int(fields[5]))
        target = by_id.get(target_id)
        if mine is None or target is None:
            continue
        mine_ships = remaining_by_id.get(int(mine[0]), int(mine[5]))
        if mine_ships < 2:
            continue

        frac = max(0.0, min(1.0, float(fields[1])))
        send = _ships_to_send(mine_ships, frac)
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
            target_is_comet=int(target_id) in comet_planet_ids,
        )
        if solution is None:
            continue
        if not _route_clear_to_solution(
            int(mine[0]),
            int(target_id),
            float(mine[2]),
            float(mine[3]),
            float(mine[4]),
            solution,
            send,
            blockers,
            omega,
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


def _build_deterministic_action_lists_from_logits_raw(
    legality_fields_l: Any,
    target_logits_l: Any,
    obs: Any,
) -> list[list]:
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
    blockers = _route_blockers_from_rows(planets, omega, comet_planet_ids)
    ids = [int(fields[4]) for fields in legality_fields_l]
    present = [float(fields[3]) >= 0.5 for fields in legality_fields_l]
    actions: list[list] = []
    p = len(legality_fields_l)
    for i, fields in enumerate(legality_fields_l):
        if not (
            float(fields[0]) >= 0.5
            and float(fields[2]) >= 0.5
            and present[i]
            and ids[i] >= 0
        ):
            continue
        mine = by_id.get(ids[i])
        if mine is None:
            continue
        mine_ships = remaining_by_id.get(int(mine[0]), int(mine[5]))
        if mine_ships < 2:
            continue

        frac = max(0.0, min(1.0, float(fields[1])))
        send = _ships_to_send(mine_ships, frac)
        if send <= 0:
            continue

        for ti in np.argsort(-target_logits_l[i], kind="stable"):
            ti = int(ti)
            if ti == i or ti < 0 or ti >= p or not present[ti]:
                continue
            if not np.isfinite(target_logits_l[i][ti]):
                continue
            target_id = ids[ti]
            if target_id < 0:
                continue
            target = by_id.get(target_id)
            if target is None:
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
                target_is_comet=int(target_id) in comet_planet_ids,
            )
            if solution is None:
                continue
            if not _route_clear_to_solution(
                int(mine[0]),
                int(target_id),
                float(mine[2]),
                float(mine[3]),
                float(mine[4]),
                solution,
                send,
                blockers,
                omega,
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
            remaining_by_id[int(mine[0])] = mine_ships - send
            break

    return actions


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
    blockers = _route_blockers_from_rows(context.planets, omega, comet_planet_ids)
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
        if target_id < 0:
            continue
        mine = by_id.get(int(fields[5]))
        target = by_id.get(target_id)
        if mine is None or target is None:
            continue
        mine_ships = remaining_by_id.get(int(mine[0]), int(mine[5]))
        if mine_ships < 2:
            continue

        frac = max(0.0, min(1.0, float(fields[1])))
        send = _ships_to_send(mine_ships, frac)
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
            target_is_comet=int(target_id) in comet_planet_ids,
        )
        if solution is None:
            continue
        if not _route_clear_to_solution(
            int(mine[0]),
            int(target_id),
            float(mine[2]),
            float(mine[3]),
            float(mine[4]),
            solution,
            send,
            blockers,
            omega,
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


def _record_from_materialized_launch(
    launch: torch.Tensor,
    target_idx: torch.Tensor,
    frac: torch.Tensor,
    launch_logits: torch.Tensor,
    launch_log_std: torch.Tensor | None,
    action_logit_softcap: float | None,
    target_logits: torch.Tensor,
    fraction_param1: torch.Tensor,
    fraction_param2: torch.Tensor,
    fraction_dist: str,
    materialized: list[bool],
    launch_prob_floor: float = 0.0,
) -> SampleRecord:
    # `launch` is the raw launch sample. The legacy Bernoulli path records the
    # materialized launch for PPO compatibility. The categorical PPO path
    # records the sampled categorical action even if the wrapper cannot build a
    # move from the sampled fraction; otherwise a failed target launch would
    # incorrectly train the no-op column.
    raw_launch = launch.to(dtype=fraction_param1.dtype)
    actual_launch = torch.as_tensor(
        materialized,
        device=target_idx.device,
        dtype=fraction_param1.dtype,
    )
    record_launch = actual_launch if action_logit_softcap is None else raw_launch
    if action_logit_softcap is None:
        safe_target_logits = _safe_target_logits(target_logits.float())
        launch_lp = _threshold_normal_launch_log_prob(
            launch_logits.float(),
            None if launch_log_std is None else launch_log_std.float(),
            actual_launch.float(),
            launch_prob_floor,
        )
        target_log_probs = torch.log_softmax(safe_target_logits, dim=-1)
        target_lp = target_log_probs.gather(
            -1,
            target_idx.clamp(0, safe_target_logits.shape[-1] - 1).unsqueeze(-1),
        ).squeeze(-1)
        action_lp = launch_lp + actual_launch.float() * target_lp
    else:
        action_log_probs = _categorical_action_log_probs(
            launch_logits.float(),
            target_logits.float(),
            action_logit_softcap,
        )
        action_idx = torch.where(
            record_launch.float() > 0.5,
            target_idx.clamp(0, target_logits.shape[-1] - 1) + 1,
            torch.zeros_like(target_idx),
        )
        action_lp = action_log_probs.gather(-1, action_idx.unsqueeze(-1)).squeeze(-1)
    action_lp = _action_log_prob_for_action(action_lp, record_launch.float())
    fraction_action_lp = _launched_fraction_log_prob(
        fraction_param1.detach().float(),
        fraction_param2.detach().float(),
        frac.detach().float(),
        fraction_dist,
        record_launch.float(),
    )
    log_prob = action_lp + fraction_action_lp
    return SampleRecord(
        launch=record_launch,
        raw_launch=raw_launch,
        target_idx=target_idx,
        fraction=frac,
        log_prob=log_prob,
        target_legal_mask=torch.isfinite(target_logits),
    )


def _batch_record_from_materialized_launch(
    launch: torch.Tensor,
    target_idx: torch.Tensor,
    frac: torch.Tensor,
    launch_logits: torch.Tensor,
    launch_log_std: torch.Tensor | None,
    action_logit_softcap: float | None,
    target_logits: torch.Tensor,
    fraction_param1: torch.Tensor,
    fraction_param2: torch.Tensor,
    fraction_dist: str,
    materialized: Any,
    rows: Sequence[int],
    launch_prob_floor: float = 0.0,
) -> SampleBatchRecord:
    row_idx = torch.as_tensor(rows, device=launch.device, dtype=torch.long)
    raw_launch_r = launch.index_select(0, row_idx)
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
    target_logits_r = target_logits.index_select(0, row_idx)
    launch_log_std_r = (
        None if launch_log_std is None else launch_log_std.index_select(0, row_idx)
    )
    fraction_param1_r = fraction_param1.index_select(0, row_idx)
    fraction_param2_r = fraction_param2.index_select(0, row_idx)
    record_launch = actual_launch if action_logit_softcap is None else raw_launch_r

    if action_logit_softcap is None:
        safe_target_logits = _safe_target_logits(target_logits_r.float())
        launch_lp = _threshold_normal_launch_log_prob(
            launch_logits.index_select(0, row_idx).float(),
            None if launch_log_std_r is None else launch_log_std_r.float(),
            actual_launch.float(),
            launch_prob_floor,
        )
        target_log_probs = torch.log_softmax(safe_target_logits, dim=-1)
        target_lp = target_log_probs.gather(
            -1,
            target_idx_r.clamp(0, safe_target_logits.shape[-1] - 1).unsqueeze(-1),
        ).squeeze(-1)
        action_lp = launch_lp + actual_launch.float() * target_lp
    else:
        action_log_probs = _categorical_action_log_probs(
            launch_logits.index_select(0, row_idx).float(),
            target_logits_r.float(),
            action_logit_softcap,
        )
        action_idx = torch.where(
            record_launch.float() > 0.5,
            target_idx_r.clamp(0, target_logits_r.shape[-1] - 1) + 1,
            torch.zeros_like(target_idx_r),
        )
        action_lp = action_log_probs.gather(-1, action_idx.unsqueeze(-1)).squeeze(-1)
    action_lp = _action_log_prob_for_action(action_lp, record_launch.float())
    fraction_action_lp = _launched_fraction_log_prob(
        fraction_param1_r.detach().float(),
        fraction_param2_r.detach().float(),
        frac_r.detach().float(),
        fraction_dist,
        record_launch.float(),
    )
    log_prob = action_lp + fraction_action_lp
    return SampleBatchRecord(
        launch=record_launch,
        raw_launch=raw_launch_r,
        target_idx=target_idx_r,
        fraction=frac_r,
        log_prob=log_prob,
        target_legal_mask=torch.isfinite(target_logits_r),
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
    launch_log_std = None if out.launch_log_std is None else out.launch_log_std[0]
    action_logit_softcap = out.action_logit_softcap
    target_logits = out.target_logits[0]  # [P, P]
    fraction_param1_b, fraction_param2_b, fraction_dist = _policy_fraction_params(out)
    fraction_param1 = fraction_param1_b[0]
    fraction_param2 = fraction_param2_b[0]
    owned = out.planet_owned_mask[0]
    pmask = out.planet_mask[0]
    ids = out.planet_ids[0]

    if action_logit_softcap is None:
        launch, frac = _sample_launch_fraction(
            launch_logits,
            fraction_param1,
            fraction_param2,
            deterministic,
            fraction_dist,
            launch_log_std=launch_log_std,
            launch_prob_floor=out.launch_prob_floor,
        )
    else:
        frac = _sample_fraction(
            fraction_param1,
            fraction_param2,
            deterministic,
            fraction_dist,
        )
        support_launch = (owned & pmask).to(
            dtype=frac.dtype,
            device=frac.device,
        )
    target_legal_mask = _target_legal_mask_from_observation(
        frac,
        launch if action_logit_softcap is None else support_launch,
        owned,
        pmask,
        ids,
        o,
    )
    target_logits = _apply_target_legal_mask(
        target_logits,
        target_legal_mask,
        launch if action_logit_softcap is None else support_launch,
        owned,
        pmask,
    )
    if action_logit_softcap is None:
        launch_logits, launch = _mask_impossible_launches(
            launch_logits, launch, target_legal_mask, owned, pmask
        )
        target_idx = _sample_target(target_logits, deterministic)
    else:
        launch, target_idx = _sample_categorical_action(
            launch_logits,
            target_logits,
            action_logit_softcap,
            owned,
            pmask,
            deterministic,
        )

    moves, materialized = _build_moves_with_mask(
        launch, target_idx, frac, owned, pmask, ids, o
    )
    record = _record_from_materialized_launch(
        launch,
        target_idx,
        frac,
        launch_logits,
        launch_log_std,
        action_logit_softcap,
        target_logits,
        fraction_param1,
        fraction_param2,
        fraction_dist,
        materialized,
        launch_prob_floor=out.launch_prob_floor,
    )
    return moves, record


def sample_actions(
    out: PolicyOutput,
    o: Observation,
    deterministic: bool = True,
) -> list[Move]:
    """Moves-only wrapper for inference paths (eval, agent submission)."""
    launch_logits = out.launch_logits[0]
    launch_log_std = None if out.launch_log_std is None else out.launch_log_std[0]
    action_logit_softcap = out.action_logit_softcap
    target_logits = out.target_logits[0]
    fraction_param1_b, fraction_param2_b, fraction_dist = _policy_fraction_params(out)
    fraction_param1 = fraction_param1_b[0]
    fraction_param2 = fraction_param2_b[0]
    owned = out.planet_owned_mask[0]
    pmask = out.planet_mask[0]
    ids = out.planet_ids[0]
    if action_logit_softcap is None:
        launch, frac = _sample_launch_fraction(
            launch_logits,
            fraction_param1,
            fraction_param2,
            deterministic,
            fraction_dist,
            launch_log_std=launch_log_std,
            launch_prob_floor=out.launch_prob_floor,
        )
    else:
        frac = _sample_fraction(
            fraction_param1,
            fraction_param2,
            deterministic,
            fraction_dist,
        )
        support_launch = (owned & pmask).to(
            dtype=frac.dtype,
            device=frac.device,
        )
    target_legal_mask = _target_legal_mask_from_observation(
        frac,
        launch if action_logit_softcap is None else support_launch,
        owned,
        pmask,
        ids,
        o,
    )
    target_logits = _apply_target_legal_mask(
        target_logits,
        target_legal_mask,
        launch if action_logit_softcap is None else support_launch,
        owned,
        pmask,
    )
    if action_logit_softcap is None:
        _launch_logits, launch = _mask_impossible_launches(
            launch_logits, launch, target_legal_mask, owned, pmask
        )
        target_idx = _sample_target(target_logits, deterministic)
    else:
        launch, target_idx = _sample_categorical_action(
            launch_logits,
            target_logits,
            action_logit_softcap,
            owned,
            pmask,
            deterministic,
        )
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

    The launch/Categorical/fraction samples are drawn once over the full [B, P]
    tensor — that's where the GPU win comes from. The per-element
    `_build_moves` walk is pure Python but cheap (one loop per env).
    """
    prepared = _prepare_batch_action_fields(
        out,
        len(parsed_list),
        deterministic,
        lambda k, fields: _target_legal_mask_from_packed_legality_fields(
            fields,
            parsed_list[k].planets,
            parsed_list[k].angular_velocity,
            parsed_list[k].comet_planet_ids,
        ),
    )
    fields_l = _packed_action_fields_from_legality_fields(
        prepared.target_idx,
        prepared.legality_fields_l,
        prepared.target_legal_mask_l,
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
    for k in range(len(parsed_list)):
        moves, materialized = _build_moves_from_packed_fields_with_mask(
            fields_l[k], parsed_list[k]
        )
        moves_list.append(moves)
        if record_pos is None:
            records.append(
                _record_from_materialized_launch(
                    prepared.launch[k],
                    prepared.target_idx[k],
                    prepared.frac[k],
                    prepared.launch_logits[k],
                    None
                    if prepared.launch_log_std is None
                    else prepared.launch_log_std[k],
                    prepared.action_logit_softcap,
                    prepared.target_logits[k],
                    prepared.fraction_param1[k],
                    prepared.fraction_param2[k],
                    prepared.fraction_dist,
                    materialized,
                    launch_prob_floor=out.launch_prob_floor,
                )
            )
        elif k in record_pos:
            materialized_rows[record_pos[k]] = materialized
    if record_rows is not None:
        return moves_list, _batch_record_from_materialized_launch(
            prepared.launch,
            prepared.target_idx,
            prepared.frac,
            prepared.launch_logits,
            prepared.launch_log_std,
            prepared.action_logit_softcap,
            prepared.target_logits,
            prepared.fraction_param1,
            prepared.fraction_param2,
            prepared.fraction_dist,
            [row for row in materialized_rows if row is not None],
            record_rows,
            launch_prob_floor=out.launch_prob_floor,
        )
    return moves_list, records


def sample_batch_with_records_raw(
    out: PolicyOutput,
    raw_observations: list[Any],
    deterministic: bool = False,
    record_rows: Sequence[int] | None = None,
) -> tuple[list[list[list]], list[SampleRecord] | SampleBatchRecord]:
    """Batched sampler that builds Kaggle action lists from raw obs dicts."""
    def _raw_target_mask(k: int, fields: Any) -> Any:
        raw_obs = raw_observations[k]
        return _target_legal_mask_from_packed_legality_fields(
            fields,
            raw_obs.get("planets", [])
            if isinstance(raw_obs, dict)
            else getattr(raw_obs, "planets", []),
            (
                raw_obs.get("angular_velocity", 0.0)
                if isinstance(raw_obs, dict)
                else getattr(raw_obs, "angular_velocity", 0.0)
            )
            or 0.0,
            raw_obs.get("comet_planet_ids", [])
            if isinstance(raw_obs, dict)
            else getattr(raw_obs, "comet_planet_ids", []),
        )

    prepared = _prepare_batch_action_fields(
        out,
        len(raw_observations),
        deterministic,
        _raw_target_mask,
    )
    fields_l = _packed_action_fields_from_legality_fields(
        prepared.target_idx,
        prepared.legality_fields_l,
        prepared.target_legal_mask_l,
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
    for k in range(len(raw_observations)):
        actions, materialized = _build_action_lists_from_packed_fields_raw_with_mask(
            fields_l[k], raw_observations[k]
        )
        actions_list.append(actions)
        if record_pos is None:
            records.append(
                _record_from_materialized_launch(
                    prepared.launch[k],
                    prepared.target_idx[k],
                    prepared.frac[k],
                    prepared.launch_logits[k],
                    None
                    if prepared.launch_log_std is None
                    else prepared.launch_log_std[k],
                    prepared.action_logit_softcap,
                    prepared.target_logits[k],
                    prepared.fraction_param1[k],
                    prepared.fraction_param2[k],
                    prepared.fraction_dist,
                    materialized,
                    launch_prob_floor=out.launch_prob_floor,
                )
            )
        elif k in record_pos:
            materialized_rows[record_pos[k]] = materialized
    if record_rows is not None:
        return actions_list, _batch_record_from_materialized_launch(
            prepared.launch,
            prepared.target_idx,
            prepared.frac,
            prepared.launch_logits,
            prepared.launch_log_std,
            prepared.action_logit_softcap,
            prepared.target_logits,
            prepared.fraction_param1,
            prepared.fraction_param2,
            prepared.fraction_dist,
            [row for row in materialized_rows if row is not None],
            record_rows,
            launch_prob_floor=out.launch_prob_floor,
        )
    return actions_list, records


def sample_batch_with_records_context(
    out: PolicyOutput,
    contexts: list[ActionContext],
    deterministic: bool = False,
    record_rows: Sequence[int] | None = None,
) -> tuple[list[list[list]], list[SampleRecord] | SampleBatchRecord]:
    """Batched sampler that builds action lists from fast env contexts."""
    prepared = _prepare_batch_action_fields(
        out,
        len(contexts),
        deterministic,
        lambda k, fields: _target_legal_mask_from_packed_legality_fields(
            fields,
            contexts[k].planets,
            contexts[k].angular_velocity,
            contexts[k].comet_planet_ids,
        ),
    )
    fields_l = _packed_action_fields_from_legality_fields(
        prepared.target_idx,
        prepared.legality_fields_l,
        prepared.target_legal_mask_l,
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
    for k in range(len(contexts)):
        actions, materialized = _build_action_lists_from_packed_fields_context_with_mask(
            fields_l[k], contexts[k]
        )
        actions_list.append(actions)
        if record_pos is None:
            records.append(
                _record_from_materialized_launch(
                    prepared.launch[k],
                    prepared.target_idx[k],
                    prepared.frac[k],
                    prepared.launch_logits[k],
                    None
                    if prepared.launch_log_std is None
                    else prepared.launch_log_std[k],
                    prepared.action_logit_softcap,
                    prepared.target_logits[k],
                    prepared.fraction_param1[k],
                    prepared.fraction_param2[k],
                    prepared.fraction_dist,
                    materialized,
                    launch_prob_floor=out.launch_prob_floor,
                )
            )
        elif k in record_pos:
            materialized_rows[record_pos[k]] = materialized
    if record_rows is not None:
        return actions_list, _batch_record_from_materialized_launch(
            prepared.launch,
            prepared.target_idx,
            prepared.frac,
            prepared.launch_logits,
            prepared.launch_log_std,
            prepared.action_logit_softcap,
            prepared.target_logits,
            prepared.fraction_param1,
            prepared.fraction_param2,
            prepared.fraction_dist,
            [row for row in materialized_rows if row is not None],
            record_rows,
            launch_prob_floor=out.launch_prob_floor,
        )
    return actions_list, records


def sample_batch_actions(
    out: PolicyOutput,
    parsed_list: list[Observation],
    deterministic: bool = True,
) -> list[list[Move]]:
    """Batched moves-only sampler for eval and opponent inference paths."""
    prepared = _prepare_batch_action_fields(
        out,
        len(parsed_list),
        deterministic,
        lambda k, fields: _target_legal_mask_from_packed_legality_fields(
            fields,
            parsed_list[k].planets,
            parsed_list[k].angular_velocity,
            parsed_list[k].comet_planet_ids,
        ),
    )
    fields_l = _packed_action_fields_from_legality_fields(
        prepared.target_idx,
        prepared.legality_fields_l,
        prepared.target_legal_mask_l,
    )

    return [
        _build_moves_from_packed_fields(fields_l[k], parsed_list[k])
        for k in range(len(parsed_list))
    ]


def sample_batch_actions_raw(
    out: PolicyOutput,
    raw_observations: list[Any],
    deterministic: bool = True,
) -> list[list[list]]:
    """Batched moves-only sampler for raw Kaggle-style observations."""
    def _raw_target_mask(k: int, fields: Any) -> Any:
        raw_obs = raw_observations[k]
        return _target_legal_mask_from_packed_legality_fields(
            fields,
            raw_obs.get("planets", [])
            if isinstance(raw_obs, dict)
            else getattr(raw_obs, "planets", []),
            (
                raw_obs.get("angular_velocity", 0.0)
                if isinstance(raw_obs, dict)
                else getattr(raw_obs, "angular_velocity", 0.0)
            )
            or 0.0,
            raw_obs.get("comet_planet_ids", [])
            if isinstance(raw_obs, dict)
            else getattr(raw_obs, "comet_planet_ids", []),
        )

    prepared = _prepare_batch_action_fields(
        out,
        len(raw_observations),
        deterministic,
        _raw_target_mask,
    )
    fields_l = _packed_action_fields_from_legality_fields(
        prepared.target_idx,
        prepared.legality_fields_l,
        prepared.target_legal_mask_l,
    )
    return [
        _build_action_lists_from_packed_fields_raw(
            fields_l[k], raw_observations[k]
        )
        for k in range(len(raw_observations))
    ]


def sample_batch_actions_context(
    out: PolicyOutput,
    contexts: list[ActionContext],
    deterministic: bool = True,
) -> list[list[list]]:
    """Batched moves-only sampler for fast env action contexts."""
    prepared = _prepare_batch_action_fields(
        out,
        len(contexts),
        deterministic,
        lambda k, fields: _target_legal_mask_from_packed_legality_fields(
            fields,
            contexts[k].planets,
            contexts[k].angular_velocity,
            contexts[k].comet_planet_ids,
        ),
    )
    fields_l = _packed_action_fields_from_legality_fields(
        prepared.target_idx,
        prepared.legality_fields_l,
        prepared.target_legal_mask_l,
    )
    return [
        _build_action_lists_from_packed_fields_context(
            fields_l[k], contexts[k]
        )
        for k in range(len(contexts))
    ]
