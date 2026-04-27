"""Convert raw `PolicyOutput` into a list of legal `Move`s.

The policy emits, per owned planet:
  - a Categorical over `(target_planet, no-op)` with the no-op slot at index P
  - a tanh-squashed Normal(μ, σ) on `z`, mapped to `fraction = (tanh z + 1)/2`

The simulator's action format is `[from_planet_id, angle_radians, num_ships]`.
Fleets fly in *straight lines* at constant speed (`fleet_speed(num_ships)`),
so the launch angle is fully determined by the chosen target — there is no
mid-flight steering. We compute the angle deterministically by solving the
intercept equation in closed form (see `_lead_angle`) — no fixed-point
iteration that might oscillate.

For PPO we need, *per owned planet*, the Categorical+tanh-Normal log-prob of
the actually-sampled action. `sample_with_record` returns those alongside the
moves; `sample_actions` is the thin moves-only wrapper used by inference paths
that don't care about log-probs. We store the **pre-squash** Normal sample `z`
in the record (not the squashed fraction): recomputing log_prob from `z` is
exact, while recovering `z = atanh(2·frac − 1)` from the squashed sample eats
numerical precision near the bounds.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from ..game import angle_to
from ..game.observation import Observation
from ..game.physics import fleet_speed
from ..game.types import CENTER, ROTATION_RADIUS_LIMIT, Move
from .model import PolicyOutput

# Numerical floor for the tanh Jacobian `1 - tanh²(z)` term in log_prob —
# matches the standard SAC implementation. Saturated tanh (|z|≥~6) drives
# `1 − tanh²` toward 0; without the floor the log term blows to −∞.
TANH_LOG_EPS: float = 1e-6

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
    """Per-planet record of the sampled action — used by PPO rollouts."""

    target_idx: torch.Tensor   # [P] long, in [0, P] (P = no-op slot)
    fraction: torch.Tensor     # [P] float in [0, 1] — squashed action used to build the move
    frac_z: torch.Tensor       # [P] float — pre-tanh Normal sample (used to recompute log_prob in PPO)
    log_prob: torch.Tensor     # [P] float — Categorical + tanh-Normal


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


def _tanh_normal_log_prob(
    mu: torch.Tensor,
    log_sigma: torch.Tensor,
    z: torch.Tensor,
    a: torch.Tensor,
) -> torch.Tensor:
    """log_prob of `frac = (tanh(z)+1)/2` under tanh-squashed Normal(μ, σ).

    Change-of-variables for `g(z) = (tanh(z)+1)/2`, `g'(z) = (1−tanh²(z))/2`:
        log p(frac) = log p(z) − log|g'(z)|
                    = log p(z) − log(1 − tanh²(z)) + log 2

    The `+ log 2` is constant per element so it cancels in PPO's importance
    ratio, but we keep it for honest absolute log-probs (any external
    diagnostic that compares them would see a `~1.4`-per-planet bias otherwise).
    """
    sigma = log_sigma.exp()
    normal_lp = -0.5 * ((z - mu) / sigma).pow(2) - log_sigma - 0.5 * math.log(2.0 * math.pi)
    # Numerically-stable `log(1 - tanh²(z))`: the naive form underflows for
    # |z| > ~7. The equivalent `2·(log 2 - z - softplus(-2z))` stays finite.
    # Used by SB3 / SAC.
    tanh_correction = 2.0 * (math.log(2.0) - z - torch.nn.functional.softplus(-2.0 * z))
    return normal_lp - tanh_correction + math.log(2.0)


def _sample_distributions(
    target_logits: torch.Tensor,
    fraction_mu: torch.Tensor,
    fraction_log_sigma: torch.Tensor,
    p: int,
    deterministic: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample (target_idx, fraction, frac_z, log_prob) from the heads.

    Works for both unbatched [P, ...] and batched [B, P, ...] shapes.

    The Normal log-prob only counts when the discrete target is *not* no-op:
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
    fraction_mu = fraction_mu.float()
    fraction_log_sigma = fraction_log_sigma.float()
    if deterministic:
        target_idx = target_logits.argmax(dim=-1)
        # μ is the mode of the Normal; tanh(μ) is the mode of the squashed
        # distribution iff σ is small (otherwise the squashed mode shifts
        # toward 0). For deterministic eval we accept that approximation —
        # exact mode would need a 1-D root solve per element.
        z = fraction_mu
        a = torch.tanh(z)
        frac = (a + 1.0) / 2.0
        log_prob = torch.zeros_like(target_idx, dtype=fraction_mu.dtype)
        return target_idx, frac, z, log_prob

    target_dist = torch.distributions.Categorical(logits=target_logits)
    target_idx = target_dist.sample()
    target_lp = target_dist.log_prob(target_idx)

    sigma = fraction_log_sigma.exp()
    # Reparameterization-style draw — matches the math used at PPO update time.
    eps = torch.randn_like(fraction_mu)
    z = fraction_mu + sigma * eps
    a = torch.tanh(z)
    frac = (a + 1.0) / 2.0
    frac_lp = _tanh_normal_log_prob(fraction_mu, fraction_log_sigma, z, a)

    is_noop = target_idx == p
    move_mask = (~is_noop).to(frac_lp.dtype)
    log_prob = target_lp + move_mask * frac_lp
    return target_idx, frac, z, log_prob


def sample_with_record(
    out: PolicyOutput,
    o: Observation,
    deterministic: bool = False,
    max_moves: int = 16,
) -> tuple[list[Move], SampleRecord]:
    """Sample an action per planet, build the legal `Move` list, AND return
    the per-planet (target_idx, fraction, frac_z, log_prob) record so PPO
    can compute the importance ratio against the *actual* sampled action.
    """
    target_logits = out.target_logits[0]  # [P, P+1]
    fraction_mu = out.fraction_mu[0]
    fraction_log_sigma = out.fraction_log_sigma[0]
    owned = out.planet_owned_mask[0]
    pmask = out.planet_mask[0]
    ids = out.planet_ids[0]
    p = target_logits.shape[0]

    target_idx, frac, frac_z, log_prob = _sample_distributions(
        target_logits, fraction_mu, fraction_log_sigma, p, deterministic
    )

    moves = _build_moves(target_idx, frac, owned, pmask, ids, o, max_moves)
    record = SampleRecord(
        target_idx=target_idx,
        fraction=frac,
        frac_z=frac_z,
        log_prob=log_prob,
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
    fraction_mu = out.fraction_mu[0]
    fraction_log_sigma = out.fraction_log_sigma[0]
    p = target_logits.shape[0]
    if deterministic:
        target_idx = target_logits.argmax(dim=-1)
        frac = (torch.tanh(fraction_mu) + 1.0) / 2.0
    else:
        target_dist = torch.distributions.Categorical(logits=target_logits)
        target_idx = target_dist.sample()
        z = fraction_mu + fraction_log_sigma.exp() * torch.randn_like(fraction_mu)
        frac = (torch.tanh(z) + 1.0) / 2.0
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

    The Categorical/Normal samples are drawn once over the full [B, P]
    tensor — that's where the GPU win comes from. The per-element
    `_build_moves` walk is pure Python but cheap (one loop per env).
    """
    target_logits = out.target_logits      # [B, P, P+1]
    fraction_mu = out.fraction_mu          # [B, P]
    fraction_log_sigma = out.fraction_log_sigma  # [B, P]
    b_dim, p, _ = target_logits.shape
    assert len(parsed_list) == b_dim, (len(parsed_list), b_dim)

    target_idx, frac, frac_z, log_prob = _sample_distributions(
        target_logits, fraction_mu, fraction_log_sigma, p, deterministic
    )

    target_idx_l: list[list[int]] = target_idx.detach().cpu().tolist()
    frac_l: list[list[float]] = frac.detach().cpu().tolist()
    owned_l: list[list[bool]] = out.planet_owned_mask.detach().cpu().tolist()
    pmask_l: list[list[bool]] = out.planet_mask.detach().cpu().tolist()
    ids_l: list[list[int]] = out.planet_ids.detach().cpu().tolist()

    moves_list: list[list[Move]] = []
    records: list[SampleRecord] = []
    for k in range(b_dim):
        moves = _build_moves_from_lists(
            target_idx_l[k],
            frac_l[k],
            owned_l[k],
            pmask_l[k],
            ids_l[k],
            parsed_list[k],
            max_moves,
        )
        moves_list.append(moves)
        records.append(
            SampleRecord(
                target_idx=target_idx[k],
                fraction=frac[k],
                frac_z=frac_z[k],
                log_prob=log_prob[k],
            )
        )
    return moves_list, records


def sample_batch_actions(
    out: PolicyOutput,
    parsed_list: list[Observation],
    deterministic: bool = True,
    max_moves: int = 16,
) -> list[list[Move]]:
    """Batched moves-only sampler for eval and opponent inference paths."""
    target_logits = out.target_logits
    fraction_mu = out.fraction_mu
    fraction_log_sigma = out.fraction_log_sigma
    b_dim, p, _ = target_logits.shape
    assert len(parsed_list) == b_dim, (len(parsed_list), b_dim)

    if deterministic:
        target_idx = target_logits.argmax(dim=-1)
        frac = (torch.tanh(fraction_mu) + 1.0) / 2.0
    else:
        target_dist = torch.distributions.Categorical(logits=target_logits)
        target_idx = target_dist.sample()
        z = fraction_mu + fraction_log_sigma.exp() * torch.randn_like(fraction_mu)
        frac = (torch.tanh(z) + 1.0) / 2.0

    target_idx_l: list[list[int]] = target_idx.detach().cpu().tolist()
    frac_l: list[list[float]] = frac.detach().cpu().tolist()
    owned_l: list[list[bool]] = out.planet_owned_mask.detach().cpu().tolist()
    pmask_l: list[list[bool]] = out.planet_mask.detach().cpu().tolist()
    ids_l: list[list[int]] = out.planet_ids.detach().cpu().tolist()

    return [
        _build_moves_from_lists(
            target_idx_l[k],
            frac_l[k],
            owned_l[k],
            pmask_l[k],
            ids_l[k],
            parsed_list[k],
            max_moves,
        )
        for k in range(b_dim)
    ]
