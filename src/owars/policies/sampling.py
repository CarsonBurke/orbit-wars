"""Convert raw `PolicyOutput` into a list of legal `Move`s.

The policy emits, per owned planet:
  - a categorical over `(target_planet, no-op)` with the no-op slot at index P
  - a Beta(α, β) over the fraction of garrison to send
  - a Beta(α, β) over a Δangle residual added to an analytic intercept

We translate that into the action format the simulator wants:
  `[from_planet_id, angle_radians, num_ships]`

`angle` is built as `analytic_lead + Δangle`, where:
  - `analytic_lead` is the bearing from our planet to the *predicted*
    position of the target at our fleet's arrival step (one re-prediction
    pass, like `agents/heuristic.py` does — the simulator computes arrival
    against the planet's actual position during travel, so a fixed-step
    prediction is approximate).
  - `Δangle` is `(u - 0.5) * (π/4)` for `u ~ Beta(α, β)`, so the residual
    spans `[-π/8, +π/8]` — wide enough to correct the analytic miss for
    fast orbiters, narrow enough that the prior keeps the policy aimed
    near the right region during cold-start.

For PPO we also need, *per owned planet*, the categorical+Beta(fraction)+
Beta(angle) log-prob of the actually-sampled action. `sample_with_record`
returns those alongside the moves; `sample_actions` is the thin moves-only
wrapper used by inference paths that don't care about log-probs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from ..game import angle_to, predicted_position
from ..game.observation import Observation
from ..game.physics import fleet_speed
from ..game.types import Move
from .model import PolicyOutput

# Δangle residual range. (-π/8, +π/8) is wide enough to swing past most
# analytic misses, narrow enough that a uniform-Beta prior doesn't ruin
# cold-start aim.
ANGLE_HALF_RANGE: float = math.pi / 8.0


@dataclass
class SampleRecord:
    """Per-planet record of the sampled action — used by PPO rollouts."""

    target_idx: torch.Tensor   # [P] long, in [0, P] (P = no-op slot)
    fraction: torch.Tensor     # [P] float in [0, 1]
    angle_offset: torch.Tensor # [P] float in [0, 1] — pre-rescale Beta sample
    log_prob: torch.Tensor     # [P] float — categorical + Beta(frac) + Beta(angle)


def _lead_angle(
    mine_x: float,
    mine_y: float,
    target_x: float,
    target_y: float,
    target_radius: float,
    angular_velocity: float,
    send: int,
) -> float:
    """Bearing to where `target` will be when our fleet of `send` ships arrives.

    One re-prediction pass: estimate steps from current distance, predict
    target position at that step, recompute steps from that distance, then
    aim. Mirrors `HeuristicAgent.act`'s lead logic — the simulator advances
    the planet during fleet travel, so a fixed-step prediction is approximate.
    """
    sp = fleet_speed(send)
    if sp <= 0.0:
        return angle_to(mine_x, mine_y, target_x, target_y)
    # First pass: conservative ceiling (matches `travel_steps` and
    # `agents/heuristic.py:70`'s first lead estimate). Second pass:
    # cheap refinement via plain int division (matches heuristic.py:85).
    # Mismatched first/second rounding is intentional in the heuristic and
    # the residual head can correct any remaining lead-error.
    d0 = math.hypot(target_x - mine_x, target_y - mine_y)
    steps = max(1, math.ceil(d0 / sp))
    tx, ty = predicted_position(target_x, target_y, target_radius, angular_velocity, steps)
    d1 = math.hypot(tx - mine_x, ty - mine_y)
    steps = max(1, int(d1 / sp))
    tx, ty = predicted_position(target_x, target_y, target_radius, angular_velocity, steps)
    return angle_to(mine_x, mine_y, tx, ty)


def _build_moves(
    target_idx: torch.Tensor,
    frac: torch.Tensor,
    angle_offset: torch.Tensor,
    owned: torch.Tensor,
    pmask: torch.Tensor,
    ids: torch.Tensor,
    o: Observation,
    max_moves: int,
) -> list[Move]:
    """Translate a single env's sampled (target_idx, frac, angle_offset) into legal Moves.

    All six tensors come from one batch element: one CPU pull to Python
    lists at the top of the function avoids `.item()` calls inside the
    per-planet loop (each `.item()` on a CUDA tensor forces a stream sync,
    so a P=64 planet loop was 64×5+ syncs per call).
    """
    p = target_idx.shape[0]
    target_idx_l: list[int] = target_idx.tolist()
    frac_l: list[float] = frac.tolist()
    angle_off_l: list[float] = angle_offset.tolist()
    ids_l: list[int] = ids.tolist()
    owned_l: list[bool] = owned.tolist()
    pmask_l: list[bool] = pmask.tolist()

    moves: list[Move] = []
    by_id = {pl.id: pl for pl in o.planets}
    omega = o.angular_velocity
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

        base = _lead_angle(
            mine.x, mine.y, target.x, target.y, target.radius, omega, send
        )
        delta = (angle_off_l[i] - 0.5) * (2.0 * ANGLE_HALF_RANGE)
        ang = ((base + delta + math.pi) % (2.0 * math.pi)) - math.pi
        moves.append(Move(mine.id, ang, send))
        if len(moves) >= max_moves:
            break

    return moves


def _sample_distributions(
    target_logits: torch.Tensor,
    f_alpha: torch.Tensor,
    f_beta: torch.Tensor,
    a_alpha: torch.Tensor,
    a_beta: torch.Tensor,
    p: int,
    deterministic: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample (target_idx, fraction, angle_offset, log_prob) from the heads.

    Works for both unbatched [P, ...] and batched [B, P, ...] shapes —
    pytorch broadcasts the Beta/Categorical primitives.

    The Beta log-probs only count when the discrete target is *not* no-op:
    on a no-op step the frac/angle samples are drawn but ignored downstream,
    so they shouldn't contribute to the importance ratio.
    """
    if deterministic:
        target_idx = target_logits.argmax(dim=-1)
        # Beta mode for α,β > 1 is (α-1)/(α+β-2); fall back to mean otherwise.
        f_mean = f_alpha / (f_alpha + f_beta).clamp_min(1e-6)
        f_mode_ok = (f_alpha > 1) & (f_beta > 1)
        f_mode = (f_alpha - 1) / (f_alpha + f_beta - 2).clamp_min(1e-6)
        frac = torch.where(f_mode_ok, f_mode, f_mean).clamp(0.0, 1.0)
        a_mean = a_alpha / (a_alpha + a_beta).clamp_min(1e-6)
        a_mode_ok = (a_alpha > 1) & (a_beta > 1)
        a_mode = (a_alpha - 1) / (a_alpha + a_beta - 2).clamp_min(1e-6)
        angle_off = torch.where(a_mode_ok, a_mode, a_mean).clamp(0.0, 1.0)
        log_prob = torch.zeros_like(target_idx, dtype=f_alpha.dtype)
        return target_idx, frac, angle_off, log_prob

    target_dist = torch.distributions.Categorical(logits=target_logits)
    target_idx = target_dist.sample()
    target_lp = target_dist.log_prob(target_idx)

    frac_dist = torch.distributions.Beta(f_alpha, f_beta)
    frac = frac_dist.sample()
    frac_for_lp = frac.clamp(1e-6, 1.0 - 1e-6)
    frac_lp = frac_dist.log_prob(frac_for_lp)

    angle_dist = torch.distributions.Beta(a_alpha, a_beta)
    angle_off = angle_dist.sample()
    angle_for_lp = angle_off.clamp(1e-6, 1.0 - 1e-6)
    angle_lp = angle_dist.log_prob(angle_for_lp)

    is_noop = target_idx == p
    move_mask = (~is_noop).to(frac_lp.dtype)
    log_prob = target_lp + move_mask * (frac_lp + angle_lp)
    return target_idx, frac, angle_off, log_prob


def sample_with_record(
    out: PolicyOutput,
    o: Observation,
    deterministic: bool = False,
    max_moves: int = 16,
) -> tuple[list[Move], SampleRecord]:
    """Sample an action per planet, build the legal `Move` list, AND return
    the per-planet (target_idx, fraction, angle_offset, log_prob) record so
    PPO can compute the importance ratio against the *actual* sampled action.
    """
    target_logits = out.target_logits[0]  # [P, P+1]
    f_alpha = out.fraction_alpha[0]
    f_beta = out.fraction_beta[0]
    a_alpha = out.angle_alpha[0]
    a_beta = out.angle_beta[0]
    owned = out.planet_owned_mask[0]
    pmask = out.planet_mask[0]
    ids = out.planet_ids[0]
    p = target_logits.shape[0]

    target_idx, frac, angle_off, log_prob = _sample_distributions(
        target_logits, f_alpha, f_beta, a_alpha, a_beta, p, deterministic
    )

    moves = _build_moves(target_idx, frac, angle_off, owned, pmask, ids, o, max_moves)
    record = SampleRecord(
        target_idx=target_idx,
        fraction=frac,
        angle_offset=angle_off,
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
    moves, _ = sample_with_record(
        out, o, deterministic=deterministic, max_moves=max_moves
    )
    return moves


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
    target_logits = out.target_logits  # [B, P, P+1]
    f_alpha = out.fraction_alpha       # [B, P]
    f_beta = out.fraction_beta         # [B, P]
    a_alpha = out.angle_alpha          # [B, P]
    a_beta = out.angle_beta            # [B, P]
    b_dim, p, _ = target_logits.shape
    assert len(parsed_list) == b_dim, (len(parsed_list), b_dim)

    target_idx, frac, angle_off, log_prob = _sample_distributions(
        target_logits, f_alpha, f_beta, a_alpha, a_beta, p, deterministic
    )

    moves_list: list[list[Move]] = []
    records: list[SampleRecord] = []
    for k in range(b_dim):
        moves = _build_moves(
            target_idx[k],
            frac[k],
            angle_off[k],
            out.planet_owned_mask[k],
            out.planet_mask[k],
            out.planet_ids[k],
            parsed_list[k],
            max_moves,
        )
        moves_list.append(moves)
        records.append(
            SampleRecord(
                target_idx=target_idx[k],
                fraction=frac[k],
                angle_offset=angle_off[k],
                log_prob=log_prob[k],
            )
        )
    return moves_list, records
