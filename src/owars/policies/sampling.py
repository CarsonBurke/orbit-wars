"""Convert raw `PolicyOutput` into a list of legal `Move`s.

The policy emits, per owned planet:
  - a Categorical over `(target_planet, no-op)` with the no-op slot at index P
  - a Beta(α, β) on [0, 1] for the fraction-of-garrison to send

The simulator's action format is `[from_planet_id, angle_radians, num_ships]`.
Fleets fly in *straight lines* at constant speed (`fleet_speed(num_ships)`),
so the launch angle is fully determined by the chosen target — there is no
mid-flight steering. We compute the angle deterministically by solving the
intercept equation in closed form (see `_lead_angle`) — no fixed-point
iteration that might oscillate.

For PPO we need, *per owned planet*, the Categorical+Beta log-prob of the
actually-sampled action. `sample_with_record` returns those alongside the
moves; `sample_actions` is the thin moves-only wrapper used by inference paths
that don't care about log-probs. The Beta sample is the action — there is no
separate latent (vs the previous tanh-Gaussian, which had pre-squash `z` and
post-squash fraction); `Beta.log_prob(fraction)` is direct and exact.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
from torch.distributions import Beta

from ..game import angle_to
from ..game.observation import Observation
from ..game.physics import fleet_speed
from ..game.types import CENTER, ROTATION_RADIUS_LIMIT, Move
from .model import PolicyOutput

# Sample clamp for digamma/log stability in `Beta.log_prob`. With α,β ≥ 1
# (post-soft-cap) `log_prob` is finite on the closed [0, 1] but the Beta-
# Jacobian `(α-1) log z + (β-1) log(1-z)` blows up if a sampled z hits
# exactly 0 or 1 with α=1 or β=1 (where the corresponding term is 0·log 0).
# Mirrors `cleanrl ppo_continuous_action_pmpo_d4_beta_relusq_v3.py:47`.
SAMPLE_EPS: float = 1e-7

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


@dataclass
class SampleRecord:
    """Per-planet record of the sampled action — used by PPO rollouts.

    The full distribution parameters (`target_logits`, `fraction_alpha`,
    `fraction_beta`) are recorded alongside the sample so PPO can compute
    the analytical KL divergence between the rollout-time policy and the
    current policy (PMPO penalty, dreamer4 §`pmpo_kl_div_loss_weight`).
    Importance-ratio PPO uses only `log_prob`, but the KL term needs the
    full distributions — hence both.

    The Beta sample IS the action (no separate latent), so we only carry
    `fraction` ∈ (eps, 1-eps); recomputing `log_prob` at that value uses
    `Beta.log_prob` directly with no Jacobian gymnastics.
    """

    target_idx: torch.Tensor   # [P] long, in [0, P] (P = no-op slot)
    fraction: torch.Tensor     # [P] float in (eps, 1-eps) — Beta sample, used both for the move and for PPO's log_prob recompute
    log_prob: torch.Tensor     # [P] float — Categorical + Beta
    target_logits: torch.Tensor       # [P, P+1] — old-policy categorical logits (PMPO KL input)
    fraction_alpha: torch.Tensor      # [P] — old-policy Beta α (post soft-cap)
    fraction_beta: torch.Tensor       # [P] — old-policy Beta β (post soft-cap)


@dataclass(slots=True)
class ActionContext:
    """Fast action builder context backed by simulator planet rows."""

    planets: Any
    angular_velocity: float


def _lead_angle(
    mine_x: float,
    mine_y: float,
    target_x: float,
    target_y: float,
    target_radius: float,
    angular_velocity: float,
    send: int,
) -> float | None:
    """Closed-form first-intercept angle for fleet → orbiting target.

    Solves `sp²·t² + 2Rρ·cos(θ₀ + ωt − φ) = R² + ρ²` for the smallest
    `t ≥ 0` via coarse Nyquist scan + bisection (see module docstring).
    The fleet is aimed at the target's *predicted* position at that t; by
    construction `sp·t = ‖target(t) − source‖`, so a fleet flying that
    angle for `t` board-steps lands exactly on the target.

    Static planets and ω=0 collapse to the zero-orbit case `t = d / sp`,
    handled inline.

    Returns `None` if no intercept exists within `LEAD_T_HORIZON_STEPS`
    (i.e. the fleet would still be in flight after the episode ends).
    The caller silently skips such moves so they become no-ops rather
    than off-board flight paths.
    """
    sp = fleet_speed(send)
    if sp <= 0.0:
        return None
    cx, cy = CENTER
    R = math.hypot(target_x - cx, target_y - cy)
    # Static target (matches `geometry.predicted_position`'s rule) or zero
    # angular velocity → no orbital motion → aim direct, t = d/sp.
    is_orbiting = (
        R + target_radius < ROTATION_RADIUS_LIMIT
        and abs(angular_velocity) > 1e-12
        and R > 1e-9
    )
    if not is_orbiting:
        d = math.hypot(target_x - mine_x, target_y - mine_y)
        if d / sp > LEAD_T_HORIZON_STEPS:
            return None
        return angle_to(mine_x, mine_y, target_x, target_y)

    a = mine_x - cx
    b = mine_y - cy
    rho = math.hypot(a, b)
    phi = math.atan2(b, a)
    theta0 = math.atan2(target_y - cy, target_x - cx)
    A = 2.0 * R * rho
    C = R * R + rho * rho

    def f(t: float) -> float:
        return sp * sp * t * t + A * math.cos(theta0 + angular_velocity * t - phi) - C

    # Source coincident with target (dist=0) ⇒ t* = 0; aim direct.
    if f(0.0) >= -1e-9:
        return angle_to(mine_x, mine_y, target_x, target_y)

    # Nyquist for the cosine: dt < π/|ω| guarantees we see every sign change.
    # T/16 is a comfortable factor-of-8 safety margin; cost is trivial (a
    # handful of cos evaluations per move).
    T_orb = 2.0 * math.pi / abs(angular_velocity)
    dt = T_orb / 16.0
    # Latest possible intercept: when sp·t exceeds R + ρ the parabola is
    # always above the cosine ceiling, so f(t) > 0 from there on. Add one
    # full orbit period for safety on edge geometries.
    t_max = min((R + rho) / sp + T_orb, LEAD_T_HORIZON_STEPS)

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
    tx = cx + R * math.cos(psi)
    ty = cy + R * math.sin(psi)
    return angle_to(mine_x, mine_y, tx, ty)


def _build_moves_from_lists(
    target_idx_l: list[int],
    frac_l: list[float],
    owned_l: list[bool],
    pmask_l: list[bool],
    ids_l: list[int],
    o: Observation,
    max_moves: int,
) -> list[Move]:
    moves: list[Move] = []
    by_id = {pl.id: pl for pl in o.planets}
    omega = o.angular_velocity
    p = len(target_idx_l)
    for i in range(p):
        if not (owned_l[i] and pmask_l[i]):
            continue
        ti = target_idx_l[i]
        if ti == p or ti == i:
            continue  # no-op slot or self-target (the latter is also masked at logits-time)
        target_id = ids_l[ti]
        if target_id < 0:
            continue
        mine = by_id.get(ids_l[i])
        target = by_id.get(target_id)
        if mine is None or target is None or mine.ships < 2:
            continue

        f = max(0.0, min(1.0, frac_l[i]))
        send = max(1, min(mine.ships - 1, int(round(mine.ships * f))))
        if send <= 0:
            continue

        ang = _lead_angle(
            mine.x, mine.y, target.x, target.y, target.radius, omega, send
        )
        if ang is None:
            continue  # solver couldn't find a feasible intercept — silently no-op
        moves.append(Move(mine.id, ang, send))
        if len(moves) >= max_moves:
            break

    return moves


def _build_moves_from_packed_fields(
    fields_l: list[list[float]],
    o: Observation,
    max_moves: int,
) -> list[Move]:
    moves: list[Move] = []
    by_id = {pl.id: pl for pl in o.planets}
    omega = o.angular_velocity
    p = len(fields_l)
    for i, fields in enumerate(fields_l):
        ti = int(fields[0])
        if fields[2] < 0.5 or fields[3] < 0.5:
            continue
        if ti == p or ti == i:
            continue
        target_id = int(fields_l[ti][4]) if 0 <= ti < p else -1
        if target_id < 0:
            continue
        mine = by_id.get(int(fields[4]))
        target = by_id.get(target_id)
        if mine is None or target is None or mine.ships < 2:
            continue

        f = max(0.0, min(1.0, float(fields[1])))
        send = max(1, min(mine.ships - 1, int(round(mine.ships * f))))
        if send <= 0:
            continue

        ang = _lead_angle(
            mine.x, mine.y, target.x, target.y, target.radius, omega, send
        )
        if ang is None:
            continue
        moves.append(Move(mine.id, ang, send))
        if len(moves) >= max_moves:
            break

    return moves


def _build_action_lists_from_packed_fields_raw(
    fields_l: list[list[float]],
    obs: Any,
    max_moves: int,
) -> list[list]:
    planets = obs.get("planets", []) if isinstance(obs, dict) else getattr(obs, "planets", [])
    by_id = {int(p[0]): p for p in planets}
    omega = float(
        (obs.get("angular_velocity", 0.0) if isinstance(obs, dict) else getattr(obs, "angular_velocity", 0.0))
        or 0.0
    )
    actions: list[list] = []
    p = len(fields_l)
    for i, fields in enumerate(fields_l):
        ti = int(fields[0])
        if fields[2] < 0.5 or fields[3] < 0.5:
            continue
        if ti == p or ti == i:
            continue
        target_id = int(fields_l[ti][4]) if 0 <= ti < p else -1
        if target_id < 0:
            continue
        mine = by_id.get(int(fields[4]))
        target = by_id.get(target_id)
        if mine is None or target is None:
            continue
        mine_ships = int(mine[5])
        if mine_ships < 2:
            continue

        frac = max(0.0, min(1.0, float(fields[1])))
        send = max(1, min(mine_ships - 1, int(round(mine_ships * frac))))
        if send <= 0:
            continue

        ang = _lead_angle(
            float(mine[2]),
            float(mine[3]),
            float(target[2]),
            float(target[3]),
            float(target[4]),
            omega,
            send,
        )
        if ang is None:
            continue
        actions.append([int(mine[0]), float(ang), int(send)])
        if len(actions) >= max_moves:
            break

    return actions


def _build_action_lists_from_packed_fields_context(
    fields_l: list[list[float]],
    context: ActionContext,
    max_moves: int,
) -> list[list]:
    by_id = {int(p[0]): p for p in context.planets}
    omega = float(context.angular_velocity)
    actions: list[list] = []
    p = len(fields_l)
    for i, fields in enumerate(fields_l):
        ti = int(fields[0])
        if fields[2] < 0.5 or fields[3] < 0.5:
            continue
        if ti == p or ti == i:
            continue
        target_id = int(fields_l[ti][4]) if 0 <= ti < p else -1
        if target_id < 0:
            continue
        mine = by_id.get(int(fields[4]))
        target = by_id.get(target_id)
        if mine is None or target is None:
            continue
        mine_ships = int(mine[5])
        if mine_ships < 2:
            continue

        frac = max(0.0, min(1.0, float(fields[1])))
        send = max(1, min(mine_ships - 1, int(round(mine_ships * frac))))
        if send <= 0:
            continue

        ang = _lead_angle(
            float(mine[2]),
            float(mine[3]),
            float(target[2]),
            float(target[3]),
            float(target[4]),
            omega,
            send,
        )
        if ang is None:
            continue
        actions.append([int(mine[0]), float(ang), int(send)])
        if len(actions) >= max_moves:
            break

    return actions


def _packed_action_fields(
    target_idx: torch.Tensor,
    frac: torch.Tensor,
    owned: torch.Tensor,
    pmask: torch.Tensor,
    ids: torch.Tensor,
) -> list[list[list[float]]]:
    """Copy all Python action fields to host in one transfer.

    Separate `.cpu().tolist()` calls on CUDA each synchronize the stream.
    Packing the small [B, P] fields together keeps PPO records on-device
    while making env-action materialization pay one synchronization.
    """
    packed = torch.stack(
        (
            target_idx.float(),
            frac.float(),
            owned.float(),
            pmask.float(),
            ids.float(),
        ),
        dim=-1,
    )
    return packed.detach().cpu().tolist()


def _build_moves(
    target_idx: torch.Tensor,
    frac: torch.Tensor,
    owned: torch.Tensor,
    pmask: torch.Tensor,
    ids: torch.Tensor,
    o: Observation,
    max_moves: int,
) -> list[Move]:
    """Translate a single env's sampled (target_idx, frac) into legal Moves.

    All five tensors come from one batch element: one CPU pull to Python
    lists at the top of the function avoids `.item()` calls inside the
    per-planet loop (each `.item()` on a CUDA tensor forces a stream sync,
    so a P=64 planet loop was 64×4+ syncs per call).
    """
    return _build_moves_from_lists(
        target_idx.tolist(),
        frac.tolist(),
        owned.tolist(),
        pmask.tolist(),
        ids.tolist(),
        o,
        max_moves,
    )


def _sample_distributions(
    target_logits: torch.Tensor,
    fraction_alpha: torch.Tensor,
    fraction_beta: torch.Tensor,
    p: int,
    deterministic: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample (target_idx, fraction, log_prob) from the heads.

    Works for both unbatched [P, ...] and batched [B, P, ...] shapes.

    The Beta log-prob only counts when the discrete target is *not* no-op:
    on a no-op step the fraction sample is drawn but ignored downstream, so
    it shouldn't contribute to the importance ratio.

    All distribution math runs in fp32 (parameter-golf `sota_train_gpt.py:163`
    `F.cross_entropy(logits.float(), …)`). bf16 has only 7 mantissa bits, and
    `log_prob` differences across PPO updates are tiny (~0.01 nats). Computing
    them in bf16 puts a precision-noise floor on `approx_kl` ≈ 0.5·(ratio−1)²
    that swamps real signal. Casting here keeps the model forward in bf16
    (FA-2, autocast) while sampling and `old_log_prob` get fp32 precision.
    """
    target_logits = target_logits.float()
    fraction_alpha = fraction_alpha.float()
    fraction_beta = fraction_beta.float()
    frac_dist = Beta(fraction_alpha, fraction_beta)
    if deterministic:
        target_idx = target_logits.argmax(dim=-1)
        # Mean of Beta — well-defined for all (α,β) ≥ 1 and equals the mode
        # whenever α,β > 1 up to the (α-1)/(α+β-2) shift; using the mean
        # avoids the α=β=1 (uniform) edge case where the mode is undefined.
        frac = fraction_alpha / (fraction_alpha + fraction_beta)
        frac = frac.clamp(SAMPLE_EPS, 1.0 - SAMPLE_EPS)
        log_prob = torch.zeros_like(target_idx, dtype=fraction_alpha.dtype)
        return target_idx, frac, log_prob

    target_dist = torch.distributions.Categorical(logits=target_logits)
    target_idx = target_dist.sample()
    target_lp = target_dist.log_prob(target_idx)

    frac = frac_dist.sample().clamp(SAMPLE_EPS, 1.0 - SAMPLE_EPS)
    frac_lp = frac_dist.log_prob(frac)

    is_noop = target_idx == p
    move_mask = (~is_noop).to(frac_lp.dtype)
    log_prob = target_lp + move_mask * frac_lp
    return target_idx, frac, log_prob


def sample_with_record(
    out: PolicyOutput,
    o: Observation,
    deterministic: bool = False,
    max_moves: int = 16,
) -> tuple[list[Move], SampleRecord]:
    """Sample an action per planet, build the legal `Move` list, AND return
    the per-planet (target_idx, fraction, log_prob) record so PPO can
    compute the importance ratio against the *actual* sampled action.
    """
    target_logits = out.target_logits[0]  # [P, P+1]
    fraction_alpha = out.fraction_alpha[0]
    fraction_beta = out.fraction_beta[0]
    owned = out.planet_owned_mask[0]
    pmask = out.planet_mask[0]
    ids = out.planet_ids[0]
    p = target_logits.shape[0]

    target_idx, frac, log_prob = _sample_distributions(
        target_logits, fraction_alpha, fraction_beta, p, deterministic
    )

    moves = _build_moves(target_idx, frac, owned, pmask, ids, o, max_moves)
    record = SampleRecord(
        target_idx=target_idx,
        fraction=frac,
        log_prob=log_prob,
        target_logits=target_logits,
        fraction_alpha=fraction_alpha,
        fraction_beta=fraction_beta,
    )
    return moves, record


def sample_actions(
    out: PolicyOutput,
    o: Observation,
    deterministic: bool = True,
    max_moves: int = 16,
) -> list[Move]:
    """Moves-only wrapper for inference paths (eval, agent submission)."""
    target_logits = out.target_logits[0]
    fraction_alpha = out.fraction_alpha[0]
    fraction_beta = out.fraction_beta[0]
    if deterministic:
        target_idx = target_logits.argmax(dim=-1)
        frac = fraction_alpha / (fraction_alpha + fraction_beta)
    else:
        target_dist = torch.distributions.Categorical(logits=target_logits)
        target_idx = target_dist.sample()
        frac = Beta(fraction_alpha, fraction_beta).sample()
    frac = frac.clamp(SAMPLE_EPS, 1.0 - SAMPLE_EPS)
    return _build_moves(
        target_idx,
        frac,
        out.planet_owned_mask[0],
        out.planet_mask[0],
        out.planet_ids[0],
        o,
        max_moves,
    )


def sample_batch_with_records(
    out: PolicyOutput,
    parsed_list: list[Observation],
    deterministic: bool = False,
    max_moves: int = 16,
) -> tuple[list[list[Move]], list[SampleRecord]]:
    """Batched counterpart to `sample_with_record`.

    `out` is a B>1 PolicyOutput (its tensors have a leading batch dim);
    `parsed_list` has length B with the parsed observation per element.
    Returns one move list and one `SampleRecord` per element.

    The Categorical/Beta samples are drawn once over the full [B, P]
    tensor — that's where the GPU win comes from. The per-element
    `_build_moves` walk is pure Python but cheap (one loop per env).
    """
    target_logits = out.target_logits      # [B, P, P+1]
    fraction_alpha = out.fraction_alpha    # [B, P]
    fraction_beta = out.fraction_beta      # [B, P]
    b_dim, p, _ = target_logits.shape
    assert len(parsed_list) == b_dim, (len(parsed_list), b_dim)

    target_idx, frac, log_prob = _sample_distributions(
        target_logits, fraction_alpha, fraction_beta, p, deterministic
    )

    fields_l = _packed_action_fields(
        target_idx, frac, out.planet_owned_mask, out.planet_mask, out.planet_ids
    )

    moves_list: list[list[Move]] = []
    records: list[SampleRecord] = []
    for k in range(b_dim):
        moves = _build_moves_from_packed_fields(fields_l[k], parsed_list[k], max_moves)
        moves_list.append(moves)
        records.append(
            SampleRecord(
                target_idx=target_idx[k],
                fraction=frac[k],
                log_prob=log_prob[k],
                target_logits=target_logits[k],
                fraction_alpha=fraction_alpha[k],
                fraction_beta=fraction_beta[k],
            )
        )
    return moves_list, records


def sample_batch_with_records_raw(
    out: PolicyOutput,
    raw_observations: list[Any],
    deterministic: bool = False,
    max_moves: int = 16,
) -> tuple[list[list[list]], list[SampleRecord]]:
    """Batched sampler that builds Kaggle action lists from raw obs dicts."""
    target_logits = out.target_logits
    fraction_alpha = out.fraction_alpha
    fraction_beta = out.fraction_beta
    b_dim, p, _ = target_logits.shape
    assert len(raw_observations) == b_dim, (len(raw_observations), b_dim)

    target_idx, frac, log_prob = _sample_distributions(
        target_logits, fraction_alpha, fraction_beta, p, deterministic
    )
    fields_l = _packed_action_fields(
        target_idx, frac, out.planet_owned_mask, out.planet_mask, out.planet_ids
    )

    actions_list: list[list[list]] = []
    records: list[SampleRecord] = []
    for k in range(b_dim):
        actions_list.append(
            _build_action_lists_from_packed_fields_raw(
                fields_l[k], raw_observations[k], max_moves
            )
        )
        records.append(
            SampleRecord(
                target_idx=target_idx[k],
                fraction=frac[k],
                log_prob=log_prob[k],
                target_logits=target_logits[k],
                fraction_alpha=fraction_alpha[k],
                fraction_beta=fraction_beta[k],
            )
        )
    return actions_list, records


def sample_batch_with_records_context(
    out: PolicyOutput,
    contexts: list[ActionContext],
    deterministic: bool = False,
    max_moves: int = 16,
) -> tuple[list[list[list]], list[SampleRecord]]:
    """Batched sampler that builds action lists from fast env contexts."""
    target_logits = out.target_logits
    fraction_alpha = out.fraction_alpha
    fraction_beta = out.fraction_beta
    b_dim, p, _ = target_logits.shape
    assert len(contexts) == b_dim, (len(contexts), b_dim)

    target_idx, frac, log_prob = _sample_distributions(
        target_logits, fraction_alpha, fraction_beta, p, deterministic
    )
    fields_l = _packed_action_fields(
        target_idx, frac, out.planet_owned_mask, out.planet_mask, out.planet_ids
    )

    actions_list: list[list[list]] = []
    records: list[SampleRecord] = []
    for k in range(b_dim):
        actions_list.append(
            _build_action_lists_from_packed_fields_context(
                fields_l[k], contexts[k], max_moves
            )
        )
        records.append(
            SampleRecord(
                target_idx=target_idx[k],
                fraction=frac[k],
                log_prob=log_prob[k],
                target_logits=target_logits[k],
                fraction_alpha=fraction_alpha[k],
                fraction_beta=fraction_beta[k],
            )
        )
    return actions_list, records


def sample_batch_actions(
    out: PolicyOutput,
    parsed_list: list[Observation],
    deterministic: bool = True,
    max_moves: int = 16,
) -> list[list[Move]]:
    """Batched moves-only sampler for eval and opponent inference paths."""
    target_logits = out.target_logits
    fraction_alpha = out.fraction_alpha
    fraction_beta = out.fraction_beta
    b_dim, p, _ = target_logits.shape
    assert len(parsed_list) == b_dim, (len(parsed_list), b_dim)

    if deterministic:
        target_idx = target_logits.argmax(dim=-1)
        frac = fraction_alpha / (fraction_alpha + fraction_beta)
    else:
        target_dist = torch.distributions.Categorical(logits=target_logits)
        target_idx = target_dist.sample()
        frac = Beta(fraction_alpha, fraction_beta).sample()
    frac = frac.clamp(SAMPLE_EPS, 1.0 - SAMPLE_EPS)

    fields_l = _packed_action_fields(
        target_idx, frac, out.planet_owned_mask, out.planet_mask, out.planet_ids
    )

    return [
        _build_moves_from_packed_fields(fields_l[k], parsed_list[k], max_moves)
        for k in range(b_dim)
    ]


def sample_batch_actions_raw(
    out: PolicyOutput,
    raw_observations: list[Any],
    deterministic: bool = True,
    max_moves: int = 16,
) -> list[list[list]]:
    """Batched moves-only sampler for raw Kaggle-style observations."""
    target_logits = out.target_logits
    fraction_alpha = out.fraction_alpha
    fraction_beta = out.fraction_beta
    b_dim, p, _ = target_logits.shape
    assert len(raw_observations) == b_dim, (len(raw_observations), b_dim)

    if deterministic:
        target_idx = target_logits.argmax(dim=-1)
        frac = fraction_alpha / (fraction_alpha + fraction_beta)
    else:
        target_dist = torch.distributions.Categorical(logits=target_logits)
        target_idx = target_dist.sample()
        frac = Beta(fraction_alpha, fraction_beta).sample()
    frac = frac.clamp(SAMPLE_EPS, 1.0 - SAMPLE_EPS)

    fields_l = _packed_action_fields(
        target_idx, frac, out.planet_owned_mask, out.planet_mask, out.planet_ids
    )
    return [
        _build_action_lists_from_packed_fields_raw(
            fields_l[k], raw_observations[k], max_moves
        )
        for k in range(b_dim)
    ]


def sample_batch_actions_context(
    out: PolicyOutput,
    contexts: list[ActionContext],
    deterministic: bool = True,
    max_moves: int = 16,
) -> list[list[list]]:
    """Batched moves-only sampler for fast env action contexts."""
    target_logits = out.target_logits
    fraction_alpha = out.fraction_alpha
    fraction_beta = out.fraction_beta
    b_dim, p, _ = target_logits.shape
    assert len(contexts) == b_dim, (len(contexts), b_dim)

    if deterministic:
        target_idx = target_logits.argmax(dim=-1)
        frac = fraction_alpha / (fraction_alpha + fraction_beta)
    else:
        target_dist = torch.distributions.Categorical(logits=target_logits)
        target_idx = target_dist.sample()
        frac = Beta(fraction_alpha, fraction_beta).sample()
    frac = frac.clamp(SAMPLE_EPS, 1.0 - SAMPLE_EPS)

    fields_l = _packed_action_fields(
        target_idx, frac, out.planet_owned_mask, out.planet_mask, out.planet_ids
    )
    return [
        _build_action_lists_from_packed_fields_context(
            fields_l[k], contexts[k], max_moves
        )
        for k in range(b_dim)
    ]
