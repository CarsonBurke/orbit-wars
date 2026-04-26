"""Convert raw `PolicyOutput` into a list of legal `Move`s.

The policy emits, per owned planet:
  - a categorical over `(target_planet, no-op)` with the no-op slot at index P
  - a Beta(α, β) over the fraction of garrison to send

We translate that into the action format the simulator wants:
  `[from_planet_id, angle_radians, num_ships]`

`angle` is computed as the bearing from our planet to the chosen target's
*current* position. (One-step orbit prediction is baked into the planet
features; the model can learn to over/under-shoot if it wants to lead
moving targets, but the geometric default is "aim at the target now".)

For PPO we also need, *per owned planet*, the categorical+Beta log-prob of
the actually-sampled action. `sample_with_record` returns those alongside
the moves; `sample_actions` is the thin moves-only wrapper used by
inference paths that don't care about log-probs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from ..game import angle_to
from ..game.observation import Observation
from ..game.types import Move
from .model import PolicyOutput


@dataclass
class SampleRecord:
    """Per-planet record of the sampled action — used by PPO rollouts."""

    target_idx: torch.Tensor  # [P] long, in [0, P] (P = no-op slot)
    fraction: torch.Tensor    # [P] float in [0, 1]
    log_prob: torch.Tensor    # [P] float — categorical + Beta log-prob


def _build_moves(
    target_idx: torch.Tensor,
    frac: torch.Tensor,
    out: PolicyOutput,
    o: Observation,
    max_moves: int,
) -> list[Move]:
    owned = out.planet_owned_mask[0]
    pmask = out.planet_mask[0]
    ids = out.planet_ids[0]
    p = out.target_logits.shape[1]

    moves: list[Move] = []
    by_id = {pl.id: pl for pl in o.planets}
    for i in range(p):
        if not (bool(owned[i]) and bool(pmask[i])):
            continue
        ti = int(target_idx[i].item())
        if ti == p:
            continue  # no-op slot
        if ti == i:
            continue  # self (also masked at logits-time, defensive double-check)
        target_id = int(ids[ti].item())
        if target_id < 0:
            continue
        mine = by_id.get(int(ids[i].item()))
        target = by_id.get(target_id)
        if mine is None or target is None or mine.ships < 2:
            continue

        f = max(0.0, min(1.0, float(frac[i].item())))
        send = max(1, min(mine.ships - 1, int(round(mine.ships * f))))
        if send <= 0:
            continue

        ang = angle_to(mine.x, mine.y, target.x, target.y)
        ang = ((ang + math.pi) % (2.0 * math.pi)) - math.pi
        moves.append(Move(mine.id, ang, send))
        if len(moves) >= max_moves:
            break

    return moves


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
    alpha = out.fraction_alpha[0]
    beta = out.fraction_beta[0]
    p = target_logits.shape[0]

    if deterministic:
        target_idx = target_logits.argmax(dim=-1)
        # Beta mode for α,β > 1 is (α-1)/(α+β-2); fall back to mean otherwise.
        mean = alpha / (alpha + beta).clamp_min(1e-6)
        mode_ok = (alpha > 1) & (beta > 1)
        mode = (alpha - 1) / (alpha + beta - 2).clamp_min(1e-6)
        frac = torch.where(mode_ok, mode, mean).clamp(0.0, 1.0)
        # Eval-time log_prob isn't used by PPO, but keep it well-defined.
        log_prob = torch.zeros(p, device=target_logits.device)
    else:
        target_dist = torch.distributions.Categorical(logits=target_logits)
        target_idx = target_dist.sample()
        target_lp = target_dist.log_prob(target_idx)

        beta_dist = torch.distributions.Beta(alpha, beta)
        frac = beta_dist.sample()
        # Beta log_prob blows up at 0 or 1; clamp the *probability argument*
        # only — the move uses the unclamped frac anyway.
        frac_for_lp = frac.clamp(1e-6, 1.0 - 1e-6)
        frac_lp = beta_dist.log_prob(frac_for_lp)

        # Beta log-prob only contributes when we actually move (not no-op).
        is_noop = target_idx == p
        log_prob = target_lp + (~is_noop).float() * frac_lp

    moves = _build_moves(target_idx, frac, out, o, max_moves)
    record = SampleRecord(target_idx=target_idx, fraction=frac, log_prob=log_prob)
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
