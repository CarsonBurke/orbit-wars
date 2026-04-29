"""PPO update — dreamer4-aligned PMPO surrogate + distributional critic.

This file is **not** vanilla clipped PPO. It mirrors dreamer4's
`learn_from_experience` pipeline (`dreamer4.py:4258–4548`) with the small
adaptations the Orbit Wars action structure imposes:

  1. **PMPO policy loss** (no PPO clip). On a per-action log-prob `lp`:
        scaled_lp = lp · |tanh(adv)|
     split by sign of `adv`:
        policy_loss = −α · mean(scaled_lp[adv ≥ 0])
                      + (1−α) · mean(scaled_lp[adv < 0])
     with `α = pmpo_pos_to_neg_weight = 0.5` (dreamer4 default).
     This is the principled replacement for the clipped surrogate when
     PMPO is on — running both at once double-counts the trust region.

  2. **Reverse PMPO KL penalty** `λ · KL(old ‖ new)` (dreamer4
     `pmpo_reverse_kl=True`, `dreamer4.py:4323-4324`):
        Categorical: Σ p_old · (log p_old − log p_new)
        Beta:        kl_divergence(old, new)
     The reverse direction punishes the *new* policy putting low mass
     where the *old* policy put high mass — mass-covering w.r.t. old.

  3. **Decoupled GAE** stays — critic targets are λ_critic-weighted
     returns (typically λ=1 → MC outcome) while the actor advantage uses
     `λ_policy < 1`. Advantages are *not* z-score normalized — PMPO's
     `tanh(adv).abs()` already bounds advantage magnitude (dreamer4
     deliberately disables advantage normalization when `use_pmpo`).

  4. **Distributional value loss** (HL-Gauss CE, `dreamer4.py:4509-4515`).
     The critic emits `value_logits` over a fixed bin support; the loss
     is cross-entropy against `target_probs(returns)`. No value clipping —
     distributional CE has bounded per-element gradients
     (`softmax_i − target_i ∈ [-1, 1]`), and dreamer4's
     `max(ce, ce_of_clipped_v)` clip degenerates with our narrow HL-Gauss
     σ (the re-encoded clipped scalar doesn't overlap with the return target,
     so clipped CE saturates at `−log(eps) ≈ 46` and dominates the loss).

The cold-start fix — value pretraining with a frozen behavior policy —
is in `train.py::pretrain_value`. This file is just the per-update math.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Beta, kl_divergence

from ..policies.features import EncodedObs
from ..policies.model import OrbitPolicy


def _slice_feats(batch: dict[str, torch.Tensor], mb) -> EncodedObs:
    """Build an `EncodedObs` view over a minibatch slice. Avoids the
    ad-hoc per-loop class that used to live inline."""
    return EncodedObs(
        planet_feats=batch["planet_feats"][mb],
        planet_mask=batch["planet_mask"][mb],
        planet_owned_mask=batch["planet_owned_mask"][mb],
        planet_ids=batch["planet_ids"][mb],
        planet_garrison=batch["planet_garrison"][mb],
        fleet_feats=batch["fleet_feats"][mb],
        fleet_mask=batch["fleet_mask"][mb],
    )


def _fixed_minibatches(
    n: int,
    minibatch_size: int,
    device: torch.device,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Return shuffled fixed-size minibatches plus per-row loss weights.

    `torch.compile(dynamic=False, fullgraph=True)` specializes on batch
    dimension. A short tail minibatch would force another graph, and enough
    distinct rollout lengths eventually hit Dynamo's recompile limit. Pad the
    final chunk by reusing shuffled rows so every PPO model call sees the same
    leading dimension, but assign padding rows zero weight so every rollout row
    contributes once per epoch.
    """
    if n <= 0:
        return []
    size = max(1, int(minibatch_size))
    idx = torch.randperm(n, device=device)
    if n < size:
        extra = idx[torch.randint(n, (size - n,), device=device)]
        mb = torch.cat((idx, extra), dim=0)
        weight = torch.cat(
            (
                torch.ones(n, device=device, dtype=torch.float32),
                torch.zeros(size - n, device=device, dtype=torch.float32),
            ),
            dim=0,
        )
        return [(mb, weight)]

    batches: list[tuple[torch.Tensor, torch.Tensor]] = []
    for start in range(0, n, size):
        mb = idx[start : start + size]
        real = mb.shape[0]
        weight = torch.ones(real, device=device, dtype=torch.float32)
        if mb.shape[0] < size:
            mb = torch.cat((mb, idx[: size - mb.shape[0]]), dim=0)
            weight = torch.cat(
                (
                    weight,
                    torch.zeros(size - real, device=device, dtype=torch.float32),
                ),
                dim=0,
            )
        batches.append((mb, weight))
    return batches


def _weighted_mean(values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    weights = weights.to(device=values.device, dtype=values.dtype)
    return (values * weights).sum() / weights.sum().clamp_min(1.0)


def _move_probability(target_probs: torch.Tensor, noop_idx: int) -> torch.Tensor:
    return 1.0 - target_probs[..., noop_idx]


def _conditional_action_entropy(
    target_log_probs: torch.Tensor,
    beta_entropy: torch.Tensor,
    noop_idx: int,
) -> torch.Tensor:
    """Entropy of target + fraction, where fraction exists only on moves."""
    min_real = torch.finfo(target_log_probs.dtype).min
    target_probs = target_log_probs.exp()
    target_entropy = -(target_probs * target_log_probs.clamp_min(min_real)).sum(
        dim=-1
    )
    return target_entropy + _move_probability(target_probs, noop_idx) * beta_entropy


def _conditional_fraction_kl(
    frac_kl: torch.Tensor,
    source_target_probs: torch.Tensor,
    noop_idx: int,
) -> torch.Tensor:
    """Beta KL contribution for a fraction head conditional on making a move."""
    return _move_probability(source_target_probs, noop_idx) * frac_kl


@dataclass
class PPOLog:
    policy_loss: float
    value_loss: float
    entropy: float
    approx_kl: float       # importance-ratio diagnostic E[old_lp − new_lp]; PMPO does not use this for the loss
    pmpo_kl: float         # analytical KL(old ‖ new) over the full distributions (the regularizer)
    pos_frac: float        # fraction of owned-planet samples whose advantage was ≥ 0


def compute_gae(
    rewards: np.ndarray,
    values: np.ndarray,
    gamma: float,
    lam: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Generalized advantage estimation, single trajectory.

    With `lam=1.0` and a value head that's not yet trustworthy this still
    bootstraps through `values[t+1]`, which is what we want once the
    critic is warm. For the *cold start*, set `lam=1.0` AND zero out
    `values` (or use `compute_mc_return` directly) — the critic has
    nothing to bootstrap from yet.
    """
    T = len(rewards)
    advs = np.zeros(T, dtype=np.float32)
    gae = 0.0
    for t in reversed(range(T)):
        next_v = values[t + 1] if t + 1 < T else 0.0
        delta = rewards[t] + gamma * next_v - values[t]
        gae = delta + gamma * lam * gae
        advs[t] = gae
    returns = advs + values
    return advs, returns


def compute_mc_return(rewards: np.ndarray, gamma: float = 1.0) -> np.ndarray:
    """Plain Monte-Carlo discounted return Σ γ^k r_{t+k}.

    With γ=1 and terminal-only ±1 reward this collapses to "every state
    in this trajectory has target = the eventual game outcome", which is
    exactly the value-pretraining target.
    """
    T = len(rewards)
    out = np.zeros(T, dtype=np.float32)
    running = 0.0
    for t in reversed(range(T)):
        running = rewards[t] + gamma * running
        out[t] = running
    return out


def length_adaptive_lambda(episode_length: int, alpha: float) -> float:
    """λ = 1 − 1/(α l) (VAPO §4.2, eq. 5).

    For α=0.05 (their value): l=100 → 0.80, l=200 → 0.90, l=500 → 0.96.
    For Orbit Wars (l ~ 200–500) this lands close to a fixed 0.95 — the
    knob matters more in regimes with high episode-length variance.
    """
    if alpha <= 0.0:
        return 0.95
    denom = max(1.0, alpha * float(episode_length))
    return max(0.0, min(0.999, 1.0 - 1.0 / denom))


def ppo_update(
    model: OrbitPolicy,
    optimizer: torch.optim.Optimizer,
    batch: dict[str, torch.Tensor],
    *,
    value_coef: float,
    entropy_coef: float,
    pmpo_kl_coef: float,
    pmpo_pos_to_neg_weight: float,
    pmpo_reverse_kl: bool,
    epochs: int,
    minibatch_size: int,
    grad_clip: float,
) -> PPOLog:
    """Run `epochs × ⌈N/B⌉` minibatch updates on `batch`.

    Expected keys:
      `planet_feats`, `planet_mask`, `planet_owned_mask`, `planet_ids`,
      `planet_garrison`, `fleet_feats`, `fleet_mask`,
      `target_idx` [B,P], `fraction` [B,P],
      `old_log_prob` [B,P], `advantage` [B], `return` [B],
      `owned_mask` [B,P],
      `old_target_logits` [B,P,P+1],
      `old_fraction_alpha` [B,P], `old_fraction_beta` [B,P].

    PMPO surrogate: `policy_loss = −α·mean(scaled[pos]) + (1−α)·mean(scaled[neg])`
    where `scaled = chosen_log_prob · |tanh(advantage)|`. No PPO clip.
    Trust region is purely the analytical reverse KL term
    `pmpo_kl_coef · KL(old ‖ new)` (`pmpo_reverse_kl=True` matches
    dreamer4 default; setting `False` flips to forward KL).

    Distributional value loss: cross-entropy against HL-Gauss-encoded
    returns (`value_encoder.target_probs(returns)`). No value clipping.
    """
    n = batch["planet_feats"].shape[0]
    device = batch["planet_feats"].device

    metric_sum: torch.Tensor | None = None
    n_steps = 0

    # `model` may be the torch.compile wrapper around the live OrbitPolicy.
    # `value_encoder` lives on the underlying module; reach through `_orig_mod`
    # if compiled, otherwise use the model directly.
    orig_model = getattr(model, "_orig_mod", model)
    value_encoder = orig_model.value_encoder
    num_bins = value_encoder.num_bins

    # bf16 autocast unlocks the SDPA Flash-Attention 2 kernel (head_dim must
    # also be FA-eligible — see model config). bf16 has fp32-equivalent range
    # so no GradScaler is needed; AdamW keeps fp32 master weights via
    # PyTorch's autocast handling. Outside cuda we stay in fp32.
    #
    # Autocast wraps ONLY the model forward — log_softmax / log_prob / KL /
    # value loss all run in fp32 after the cast, mirroring pg's
    # `F.cross_entropy(logits.float(), …)` pattern (sota_train_gpt.py:163).
    # bf16's 7 mantissa bits put a noise floor on log-prob differences
    # (~0.01 nats per update is below bf16 precision); doing distribution
    # math in bf16 amplifies that noise into the regularizers.
    autocast_enabled = (
        next(model.parameters()).is_cuda
        if any(True for _ in model.parameters())
        else False
    )
    for _ in range(epochs):
        for mb, row_weight in _fixed_minibatches(n, minibatch_size, device):

            with torch.autocast(
                device_type="cuda", dtype=torch.bfloat16, enabled=autocast_enabled
            ):
                out = model(_slice_feats(batch, mb))

            target_logits = out.target_logits.float()
            fraction_alpha = out.fraction_alpha.float()
            fraction_beta = out.fraction_beta.float()
            value_logits = out.value_logits.float()

            owned_f = batch["owned_mask"][mb].float()
            p = target_logits.shape[1]
            target = batch["target_idx"][mb].clamp(0, p)
            target_log_probs = F.log_softmax(target_logits, dim=-1)
            target_dist_probs = target_log_probs.exp()
            target_lp = target_log_probs.gather(-1, target.unsqueeze(-1)).squeeze(-1)

            # Beta log-prob at the recorded sample. The fraction component
            # is only counted when the action wasn't no-op (target slot == p),
            # mirroring what `sample_with_record` stored in `old_log_prob`.
            # `fraction` is already clamped into (eps, 1-eps) at sample time,
            # so `Beta.log_prob` is finite for all (α, β) ≥ 1.
            fraction = batch["fraction"][mb].float()
            new_beta = Beta(fraction_alpha, fraction_beta)
            frac_lp = new_beta.log_prob(fraction)
            is_noop = (target == p).float()
            move_mask = 1.0 - is_noop
            chosen = target_lp + move_mask * frac_lp

            old_log_prob = batch["old_log_prob"][mb].float()  # [B, P]
            advantage = batch["advantage"][mb].float()         # [B]
            adv_b = advantage.unsqueeze(-1).expand_as(chosen)  # [B, P]

            # PMPO policy loss (dreamer4.py:4265-4296). Replaces the clipped
            # PPO surrogate. Magnitude shaping `tanh(adv).abs()` ∈ [0, 1)
            # bounds the per-step contribution regardless of advantage scale,
            # which is why dreamer4 deliberately *does not* z-score advantages
            # under PMPO.
            scaled_lp = chosen * adv_b.tanh().abs()
            row_w = row_weight.to(device=owned_f.device, dtype=owned_f.dtype)
            row_w_b = row_w.unsqueeze(-1)
            owned_w = owned_f * row_w_b
            pos_w = owned_w * (adv_b >= 0.0).to(owned_f.dtype)
            neg_w = owned_w * (adv_b < 0.0).to(owned_f.dtype)

            pos_loss = _weighted_mean(scaled_lp, pos_w)
            neg_loss = _weighted_mean(scaled_lp, neg_w)
            α = pmpo_pos_to_neg_weight
            policy_loss = -α * pos_loss + (1.0 - α) * neg_loss

            # ---------------- distributional value loss ----------------
            ret = batch["return"][mb].float()
            target_probs = value_encoder.target_probs(ret)            # [B, num_bins]
            log_v_probs = F.log_softmax(value_logits, dim=-1)         # [B, num_bins]
            value_ce = -(target_probs * log_v_probs).sum(dim=-1)
            value_loss = _weighted_mean(value_ce, row_w)

            # ---------------- entropy bonus (categorical + Beta) -------
            beta_entropy = new_beta.entropy()
            # Fraction is conditional on the categorical selecting a move, so
            # its analytical entropy is weighted by current P(target != no-op),
            # not by the old sampled action.
            planet_entropy = _conditional_action_entropy(
                target_log_probs, beta_entropy, p
            )
            denom = owned_w.sum().clamp_min(1.0)
            entropy = (planet_entropy * owned_w).sum() / denom

            # ---------------- PMPO analytical KL ----------------------
            pmpo_kl = torch.zeros((), dtype=policy_loss.dtype, device=policy_loss.device)
            if pmpo_kl_coef != 0.0:
                old_target_logits = batch["old_target_logits"][mb].float()
                old_target_log_probs = F.log_softmax(old_target_logits, dim=-1)
                # Self-target slots have `target_logits = -inf` (`model.py`
                # `_self_target_mask`), so `log_softmax` yields `-inf` there.
                # `(-inf) − (-inf) = NaN`; clamp log-probs to dtype-min before
                # the subtraction. At masked slots the corresponding `p_old`
                # (or `p_new`) is 0 so the contribution is 0 by construction.
                kl_min = torch.finfo(target_log_probs.dtype).min
                log_p_new_safe = target_log_probs.clamp_min(kl_min)
                log_p_old_safe = old_target_log_probs.clamp_min(kl_min)

                old_alpha = batch["old_fraction_alpha"][mb].float()
                old_beta = batch["old_fraction_beta"][mb].float()
                old_beta_dist = Beta(old_alpha, old_beta)

                if pmpo_reverse_kl:
                    # KL(old ‖ new) — dreamer4 default; mass-covering w.r.t.
                    # the rollout policy.
                    target_probs_old = old_target_log_probs.exp()
                    target_kl = (
                        target_probs_old * (log_p_old_safe - log_p_new_safe)
                    ).sum(dim=-1)  # [B, P]
                    frac_kl = kl_divergence(old_beta_dist, new_beta)
                    frac_kl = _conditional_fraction_kl(frac_kl, target_probs_old, p)
                else:
                    # KL(new ‖ old) — forward direction; mode-seeking.
                    target_kl = (
                        target_dist_probs * (log_p_new_safe - log_p_old_safe)
                    ).sum(dim=-1)
                    frac_kl = kl_divergence(new_beta, old_beta_dist)
                    frac_kl = _conditional_fraction_kl(frac_kl, target_dist_probs, p)
                # The fraction distribution is conditional on selecting a real
                # target. The joint action KL is therefore categorical KL plus
                # source-policy P(move) times the Beta KL.
                planet_kl = target_kl + frac_kl
                # KL is non-negative analytically; bf16-forward → fp32-cast
                # leaves last-bit noise that can dip slightly below zero
                # when new ≈ old (first PPO minibatch). Clamp to keep the
                # logged scalar honest and avoid surprising consumers.
                pmpo_kl = ((planet_kl * owned_w).sum() / denom).clamp_min(0.0)

            loss = (
                policy_loss
                + value_coef * value_loss
                - entropy_coef * entropy
                + pmpo_kl_coef * pmpo_kl
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            with torch.no_grad():
                # Importance-ratio diagnostic. Even though PMPO doesn't use it
                # for the loss, watching `approx_kl` is the cleanest proxy for
                # per-update policy drift — an order-of-magnitude jump here is
                # exactly the cold-start signal we want to catch.
                kl = ((old_log_prob - chosen) * owned_w).sum() / denom
                pos_count = pos_w.sum()
                total_owned = owned_w.sum().clamp_min(1.0)
                pos_frac = pos_count / total_owned

            metrics = torch.stack(
                [
                    policy_loss.detach(),
                    value_loss.detach(),
                    entropy.detach(),
                    kl.detach(),
                    pmpo_kl.detach(),
                    pos_frac.detach(),
                ]
            ).float()
            if metric_sum is None:
                metric_sum = torch.zeros_like(metrics)
            metric_sum += metrics
            n_steps += 1

    n_steps = max(1, n_steps)
    if metric_sum is None:
        logs = [0.0] * 6
    else:
        logs = (metric_sum / n_steps).detach().cpu().tolist()
    return PPOLog(
        policy_loss=float(logs[0]),
        value_loss=float(logs[1]),
        entropy=float(logs[2]),
        approx_kl=float(logs[3]),
        pmpo_kl=float(logs[4]),
        pos_frac=float(logs[5]),
    )


def value_only_update(
    model: OrbitPolicy,
    optimizer: torch.optim.Optimizer,
    batch: dict[str, torch.Tensor],
    *,
    epochs: int,
    minibatch_size: int,
    grad_clip: float,
) -> float:
    """Critic-only distributional CE update for the value-pretraining phase.

    `batch["return"]` should be Monte-Carlo returns (γ=1 for terminal-only
    reward → just the trajectory outcome). Run this for a few hundred
    steps against a frozen behavior policy before turning on PPO. Mirrors
    the same HL-Gauss CE loss `ppo_update` uses, so the cold-start critic
    sees the same target distribution it'll be trained against later.

    Reports mean value loss over the pass.
    """
    n = batch["planet_feats"].shape[0]
    device = batch["planet_feats"].device
    total: torch.Tensor | None = None
    n_steps = 0

    orig_model = getattr(model, "_orig_mod", model)
    value_encoder = orig_model.value_encoder

    autocast_enabled = (
        next(model.parameters()).is_cuda
        if any(True for _ in model.parameters())
        else False
    )
    for _ in range(epochs):
        for mb, row_weight in _fixed_minibatches(n, minibatch_size, device):
            with torch.autocast(
                device_type="cuda", dtype=torch.bfloat16, enabled=autocast_enabled
            ):
                out = model(_slice_feats(batch, mb))
            value_logits = out.value_logits.float()
            ret = batch["return"][mb].float()
            target_probs = value_encoder.target_probs(ret)
            log_probs = F.log_softmax(value_logits, dim=-1)
            value_ce = -(target_probs * log_probs).sum(dim=-1)
            value_loss = _weighted_mean(value_ce, row_weight.to(value_ce.device))

            optimizer.zero_grad(set_to_none=True)
            value_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            if total is None:
                total = value_loss.detach().new_zeros(())
            total += value_loss.detach()
            n_steps += 1

    if total is None:
        return 0.0
    return float((total / max(1, n_steps)).detach().cpu())
