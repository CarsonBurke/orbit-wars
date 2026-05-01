"""PPO update — dreamer4-aligned PMPO surrogate + distributional critic.

This file is **not** vanilla clipped PPO. It mirrors dreamer4's
`learn_from_experience` pipeline (`dreamer4.py:4258–4548`) with the small
adaptations the Orbit Wars action structure imposes:

  1. **PMPO policy loss** (no PPO clip). On a per-action-factor log-prob `lp`:
        scaled_lp = lp · |tanh(adv)|
     split by sign of `adv`:
        policy_loss = −α · mean(scaled_lp[adv ≥ 0])
                      + (1−α) · mean(scaled_lp[adv < 0])
     with `α = pmpo_pos_to_neg_weight = 0.5` (dreamer4 default).
     This is the principled replacement for the clipped surrogate when
     PMPO is on — running both at once double-counts the trust region.

  2. **PMPO KL penalty**. `pmpo_reverse_kl=True` uses dreamer4's reverse-KL
     direction for each modeled factor (`dreamer4.py:4323-4324`):
        Bernoulli launch: Σ p_old · (log p_old − log p_new)
        Categorical target | launch: same, weighted by source P(launch)
        Beta fraction | launch: kl_divergence(old, new), weighted by source P(launch)
     This is the analytical KL for the policy's latent factored action
     distribution. The actor log-prob still uses the materialized action after
     the legality layer, but the trust-region penalty follows dreamer4 and
     regularizes the full distribution, not only sampled/executed branches.

  3. **Conventional GAE** supplies both critic returns and actor advantages.
     Advantages are *not* z-score normalized — PMPO's `tanh(adv).abs()` already
     bounds advantage magnitude (dreamer4 deliberately disables advantage
     normalization when `use_pmpo`).

  4. **Distributional value loss** (HL-Gauss CE, `dreamer4.py:4509-4515`).
     The critic emits `value_logits` over a fixed bin support; the loss
     is cross-entropy against `target_probs(returns)`. No value clipping —
     distributional CE has bounded per-element gradients
     (`softmax_i − target_i ∈ [-1, 1]`), and dreamer4's
     `max(ce, ce_of_clipped_v)` clip degenerates with our narrow HL-Gauss
     σ (the re-encoded clipped scalar doesn't overlap with the return target,
     so clipped CE saturates at `−log(eps) ≈ 46` and dominates the loss).

Unlike dreamer4's world-model setup, this is a pure policy/value agent: policy
gradients are allowed through the shared encoder.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812

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


def _mark_cuda_graph_step(device: torch.device) -> None:
    """Tell Inductor's CUDA graph runtime that a new replay step begins.

    This mirrors parameter-golf's training loop: use `torch.compile` with a
    CUDA-graph-friendly mode, then mark each static-shape minibatch call. It
    avoids hand-written `torch.cuda.CUDAGraph` buffers around mutable optimizer
    state while still letting Inductor replay eligible compiled regions.
    """
    if device.type != "cuda":
        return
    mark = getattr(torch.compiler, "cudagraph_mark_step_begin", None)
    if callable(mark):
        mark()


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


def _weighted_max(values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    weights = weights.to(device=values.device, dtype=values.dtype)
    masked = values.masked_fill(weights <= 0, float("-inf"))
    out = masked.max()
    return torch.where(torch.isfinite(out), out, values.sum() * 0.0)


def _minibatch_loss_scale(row_weight: torch.Tensor) -> torch.Tensor:
    """Scale padded-tail minibatch gradients by the fraction of real rows."""
    return row_weight.sum().clamp_min(1.0) / max(1, row_weight.numel())


def _safe_target_logits(target_logits: torch.Tensor) -> torch.Tensor:
    finite = torch.isfinite(target_logits).any(dim=-1, keepdim=True)
    return torch.where(finite, target_logits, torch.zeros_like(target_logits))


def _bernoulli_log_probs(logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    log_p1 = -F.softplus(-logits)
    log_p0 = -F.softplus(logits)
    return log_p0, log_p1


def _bernoulli_kl(
    source_logits: torch.Tensor,
    target_logits: torch.Tensor,
) -> torch.Tensor:
    source_p = source_logits.sigmoid()
    source_log_p0, source_log_p1 = _bernoulli_log_probs(source_logits)
    target_log_p0, target_log_p1 = _bernoulli_log_probs(target_logits)
    return source_p * (source_log_p1 - target_log_p1) + (1.0 - source_p) * (
        source_log_p0 - target_log_p0
    )


def _bernoulli_entropy(logits: torch.Tensor) -> torch.Tensor:
    p = logits.sigmoid()
    log_p0, log_p1 = _bernoulli_log_probs(logits)
    return -(p * log_p1 + (1.0 - p) * log_p0)


def _conditional_action_entropy(
    launch_logits: torch.Tensor,
    target_log_probs: torch.Tensor,
    beta_entropy: torch.Tensor,
) -> torch.Tensor:
    """Entropy of launch + P(launch) * (target + fraction)."""
    min_real = torch.finfo(target_log_probs.dtype).min
    target_probs = target_log_probs.exp()
    target_entropy = -(target_probs * target_log_probs.clamp_min(min_real)).sum(
        dim=-1
    )
    launch_prob = launch_logits.sigmoid()
    return _bernoulli_entropy(launch_logits) + launch_prob * (
        target_entropy + beta_entropy
    )


def _beta_log_normalizer(alpha: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
    return torch.lgamma(alpha) + torch.lgamma(beta) - torch.lgamma(alpha + beta)


def _beta_log_prob(
    alpha: torch.Tensor,
    beta: torch.Tensor,
    value: torch.Tensor,
) -> torch.Tensor:
    return (
        (alpha - 1.0) * value.log()
        + (beta - 1.0) * torch.log1p(-value)
        - _beta_log_normalizer(alpha, beta)
    )


def _beta_entropy(alpha: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
    log_norm = _beta_log_normalizer(alpha, beta)
    total = alpha + beta
    return (
        log_norm
        - (alpha - 1.0) * torch.digamma(alpha)
        - (beta - 1.0) * torch.digamma(beta)
        + (total - 2.0) * torch.digamma(total)
    )


def _beta_kl(
    source_alpha: torch.Tensor,
    source_beta: torch.Tensor,
    target_alpha: torch.Tensor,
    target_beta: torch.Tensor,
) -> torch.Tensor:
    source_total = source_alpha + source_beta
    return (
        _beta_log_normalizer(target_alpha, target_beta)
        - _beta_log_normalizer(source_alpha, source_beta)
        + (source_alpha - target_alpha) * torch.digamma(source_alpha)
        + (source_beta - target_beta) * torch.digamma(source_beta)
        + (target_alpha - source_alpha + target_beta - source_beta)
        * torch.digamma(source_total)
    )


def _module_device(model: torch.nn.Module) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


class _PPOMinibatchKernel(torch.nn.Module):
    """Fixed-shape PPO minibatch loss/metric kernel.

    Slicing and optimizer state mutation remain in Python; all hot tensor
    work between "minibatch tensors in" and "loss/metrics out" lives here so
    `torch.compile(..., mode="reduce-overhead")` can specialize and CUDA-graph
    replay the repeated minibatch body.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        *,
        value_coef: float,
        target_entropy_coef: float,
        fraction_entropy_coef: float,
        pmpo_kl_coef: float,
        pmpo_pos_to_neg_weight: float,
        pmpo_reverse_kl: bool,
        autocast_enabled: bool,
    ) -> None:
        super().__init__()
        self.model = model
        self.value_coef = float(value_coef)
        self.target_entropy_coef = float(target_entropy_coef)
        self.fraction_entropy_coef = float(fraction_entropy_coef)
        self.pmpo_kl_coef = float(pmpo_kl_coef)
        self.pmpo_pos_to_neg_weight = float(pmpo_pos_to_neg_weight)
        self.pmpo_reverse_kl = bool(pmpo_reverse_kl)
        self.autocast_enabled = bool(autocast_enabled)

    def forward(
        self,
        planet_feats: torch.Tensor,
        planet_mask: torch.Tensor,
        planet_owned_mask: torch.Tensor,
        planet_ids: torch.Tensor,
        planet_garrison: torch.Tensor,
        fleet_feats: torch.Tensor,
        fleet_mask: torch.Tensor,
        row_weight: torch.Tensor,
        launch: torch.Tensor,
        target_idx: torch.Tensor,
        fraction: torch.Tensor,
        old_log_prob: torch.Tensor,
        advantage: torch.Tensor,
        ret: torch.Tensor,
        owned_mask: torch.Tensor,
        old_launch_logits: torch.Tensor,
        old_target_logits: torch.Tensor,
        old_fraction_alpha: torch.Tensor,
        old_fraction_beta: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        feats = EncodedObs(
            planet_feats=planet_feats,
            planet_mask=planet_mask,
            planet_owned_mask=planet_owned_mask,
            planet_ids=planet_ids,
            planet_garrison=planet_garrison,
            fleet_feats=fleet_feats,
            fleet_mask=fleet_mask,
        )
        with torch.autocast(
            device_type="cuda", dtype=torch.bfloat16, enabled=self.autocast_enabled
        ):
            out = self.model(feats)

        old_target_mask = torch.isfinite(old_target_logits)
        has_legal_target = old_target_mask.any(dim=-1)
        launch_logits = out.launch_logits.float().masked_fill(~has_legal_target, -20.0)
        target_logits = out.target_logits.float().masked_fill(~old_target_mask, float("-inf"))
        target_logits = _safe_target_logits(target_logits)
        fraction_alpha = out.fraction_alpha.float()
        fraction_beta = out.fraction_beta.float()
        value_logits = out.value_logits.float()

        owned_f = owned_mask.float()
        p = target_logits.shape[1]
        launch_f = launch.float().clamp(0.0, 1.0)
        target = target_idx.clamp(0, p - 1)
        launch_lp = -F.binary_cross_entropy_with_logits(
            launch_logits, launch_f, reduction="none"
        )
        target_log_probs = F.log_softmax(target_logits, dim=-1)
        target_dist_probs = target_log_probs.exp()
        target_lp = target_log_probs.gather(-1, target.unsqueeze(-1)).squeeze(-1)
        frac_lp = _beta_log_prob(fraction_alpha, fraction_beta, fraction.float())
        chosen = launch_lp + launch_f * (target_lp + frac_lp)

        factor_log_probs = torch.stack((launch_lp, target_lp, frac_lp), dim=-1)
        factor_mask = torch.stack(
            (torch.ones_like(launch_f), launch_f, launch_f),
            dim=-1,
        )
        adv_b = advantage.float().unsqueeze(-1).expand_as(chosen)
        adv_f = advantage.float().unsqueeze(-1).unsqueeze(-1).expand_as(factor_log_probs)
        scaled_lp = factor_log_probs * adv_f.tanh().abs()
        row_w = row_weight.to(device=owned_f.device, dtype=owned_f.dtype)
        row_w_b = row_w.unsqueeze(-1)
        owned_w = owned_f * row_w_b
        factor_w = owned_w.unsqueeze(-1) * factor_mask
        pos_w = factor_w * (adv_f >= 0.0).to(owned_f.dtype)
        neg_w = factor_w * (adv_f < 0.0).to(owned_f.dtype)

        pos_loss = _weighted_mean(scaled_lp, pos_w)
        neg_loss = _weighted_mean(scaled_lp, neg_w)
        alpha = self.pmpo_pos_to_neg_weight
        policy_loss = -alpha * pos_loss + (1.0 - alpha) * neg_loss

        value_encoder = self.model.value_encoder
        target_probs = value_encoder.target_probs(ret.float())
        log_v_probs = F.log_softmax(value_logits, dim=-1)
        value_ce = -(target_probs * log_v_probs).sum(dim=-1)
        value_loss = _weighted_mean(value_ce, row_w)

        beta_entropy = _beta_entropy(fraction_alpha, fraction_beta)
        p_move_current = launch_logits.sigmoid()
        planet_entropy = _conditional_action_entropy(
            launch_logits, target_log_probs, beta_entropy
        )
        denom = owned_w.sum().clamp_min(1.0)
        entropy = (planet_entropy * owned_w).sum() / denom
        launch_entropy = (_bernoulli_entropy(launch_logits) * owned_w).sum() / denom
        target_entropy_per_planet = -(
            target_dist_probs
            * target_log_probs.clamp_min(torch.finfo(target_log_probs.dtype).min)
        ).sum(dim=-1)
        target_entropy = ((p_move_current * target_entropy_per_planet) * owned_w).sum()
        target_entropy = target_entropy / denom
        fraction_entropy = ((p_move_current * beta_entropy) * owned_w).sum() / denom
        move_prob = (p_move_current * owned_w).sum() / denom
        target_confidence = (target_dist_probs.amax(dim=-1) * owned_w).sum() / denom
        fraction_alpha_mean = (fraction_alpha * owned_w).sum() / denom
        fraction_alpha_max = _weighted_max(fraction_alpha, owned_w)
        fraction_beta_mean = (fraction_beta * owned_w).sum() / denom
        fraction_beta_max = _weighted_max(fraction_beta, owned_w)
        fraction_concentration = fraction_alpha + fraction_beta - 2.0
        fraction_mode = (
            (fraction_alpha - 1.0)
            / fraction_concentration.clamp_min(torch.finfo(fraction_alpha.dtype).eps)
        )
        fraction_mode_mean = (fraction_mode * owned_w).sum() / denom
        fraction_concentration_mean = (fraction_concentration * owned_w).sum() / denom
        fraction_concentration_max = _weighted_max(fraction_concentration, owned_w)
        entropy_bonus = (
            self.target_entropy_coef * (launch_entropy + target_entropy)
            + self.fraction_entropy_coef * fraction_entropy
        )

        pmpo_kl = torch.zeros((), dtype=policy_loss.dtype, device=policy_loss.device)
        pmpo_target_kl = torch.zeros_like(pmpo_kl)
        pmpo_fraction_kl = torch.zeros_like(pmpo_kl)
        if self.pmpo_kl_coef != 0.0:
            old_target_logits_safe = _safe_target_logits(old_target_logits.float())
            old_target_log_probs = F.log_softmax(old_target_logits_safe, dim=-1)
            kl_min = torch.finfo(target_log_probs.dtype).min
            log_p_new_safe = target_log_probs.clamp_min(kl_min)
            log_p_old_safe = old_target_log_probs.clamp_min(kl_min)
            old_launch_logits_f = old_launch_logits.float()
            old_alpha = old_fraction_alpha.float()
            old_beta = old_fraction_beta.float()
            if self.pmpo_reverse_kl:
                launch_kl = _bernoulli_kl(old_launch_logits_f, launch_logits)
                conditional_kl_weight = old_launch_logits_f.sigmoid()
                target_probs_old = old_target_log_probs.exp()
                target_kl = (
                    target_probs_old * (log_p_old_safe - log_p_new_safe)
                ).sum(dim=-1)
                frac_kl = _beta_kl(old_alpha, old_beta, fraction_alpha, fraction_beta)
            else:
                launch_kl = _bernoulli_kl(launch_logits, old_launch_logits_f)
                conditional_kl_weight = launch_logits.sigmoid()
                target_kl = (
                    target_dist_probs * (log_p_new_safe - log_p_old_safe)
                ).sum(dim=-1)
                frac_kl = _beta_kl(fraction_alpha, fraction_beta, old_alpha, old_beta)
            planet_kl = launch_kl + conditional_kl_weight * (target_kl + frac_kl)
            pmpo_kl = ((planet_kl * owned_w).sum() / denom).clamp_min(0.0)
            pmpo_target_kl = (
                ((conditional_kl_weight * target_kl) * owned_w).sum() / denom
            ).clamp_min(0.0)
            pmpo_fraction_kl = (
                ((conditional_kl_weight * frac_kl) * owned_w).sum() / denom
            ).clamp_min(0.0)

        loss = (
            policy_loss
            + self.value_coef * value_loss
            - entropy_bonus
            + self.pmpo_kl_coef * pmpo_kl
        )

        kl = ((old_log_prob.float() - chosen) * owned_w).sum() / denom
        pos_count = (owned_w * (adv_b >= 0.0).to(owned_f.dtype)).sum()
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
                target_entropy.detach(),
                fraction_entropy.detach(),
                move_prob.detach(),
                target_confidence.detach(),
                fraction_alpha_mean.detach(),
                fraction_alpha_max.detach(),
                fraction_beta_mean.detach(),
                fraction_beta_max.detach(),
                fraction_mode_mean.detach(),
                fraction_concentration_mean.detach(),
                fraction_concentration_max.detach(),
                pmpo_target_kl.detach(),
                pmpo_fraction_kl.detach(),
            ]
        ).float()
        return loss, metrics


class _ValueOnlyMinibatchKernel(torch.nn.Module):
    def __init__(self, model: torch.nn.Module, *, autocast_enabled: bool) -> None:
        super().__init__()
        self.model = model
        self.autocast_enabled = bool(autocast_enabled)

    def forward(
        self,
        planet_feats: torch.Tensor,
        planet_mask: torch.Tensor,
        planet_owned_mask: torch.Tensor,
        planet_ids: torch.Tensor,
        planet_garrison: torch.Tensor,
        fleet_feats: torch.Tensor,
        fleet_mask: torch.Tensor,
        row_weight: torch.Tensor,
        ret: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        feats = EncodedObs(
            planet_feats=planet_feats,
            planet_mask=planet_mask,
            planet_owned_mask=planet_owned_mask,
            planet_ids=planet_ids,
            planet_garrison=planet_garrison,
            fleet_feats=fleet_feats,
            fleet_mask=fleet_mask,
        )
        with torch.autocast(
            device_type="cuda", dtype=torch.bfloat16, enabled=self.autocast_enabled
        ):
            out = self.model(feats)
        value_logits = out.value_logits.float()
        target_probs = self.model.value_encoder.target_probs(ret.float())
        log_probs = F.log_softmax(value_logits, dim=-1)
        value_ce = -(target_probs * log_probs).sum(dim=-1)
        value_loss = _weighted_mean(value_ce, row_weight.to(value_ce.device))
        return value_loss, value_loss.detach().float()


def _kernel_cache(model: torch.nn.Module) -> dict:
    cache = model.__dict__.get("_owars_minibatch_kernel_cache")
    if cache is None:
        cache = {}
        model.__dict__["_owars_minibatch_kernel_cache"] = cache
    return cache


def _compile_kernel(
    kernel: torch.nn.Module,
    *,
    device: torch.device,
    compile_mode: str | None,
) -> torch.nn.Module:
    if device.type != "cuda" or compile_mode is None:
        return kernel
    return torch.compile(
        kernel,
        dynamic=False,
        fullgraph=True,
        mode=compile_mode,
    )


def _get_ppo_kernel(
    model: torch.nn.Module,
    *,
    value_coef: float,
    target_entropy_coef: float,
    fraction_entropy_coef: float,
    pmpo_kl_coef: float,
    pmpo_pos_to_neg_weight: float,
    pmpo_reverse_kl: bool,
    compile_mode: str | None,
) -> torch.nn.Module:
    device = _module_device(model)
    mode = compile_mode if device.type == "cuda" else None
    key = (
        "ppo",
        mode,
        float(value_coef),
        float(target_entropy_coef),
        float(fraction_entropy_coef),
        float(pmpo_kl_coef),
        float(pmpo_pos_to_neg_weight),
        bool(pmpo_reverse_kl),
    )
    cache = _kernel_cache(model)
    cached = cache.get(key)
    if cached is not None:
        return cached
    kernel = _PPOMinibatchKernel(
        model,
        value_coef=value_coef,
        target_entropy_coef=target_entropy_coef,
        fraction_entropy_coef=fraction_entropy_coef,
        pmpo_kl_coef=pmpo_kl_coef,
        pmpo_pos_to_neg_weight=pmpo_pos_to_neg_weight,
        pmpo_reverse_kl=pmpo_reverse_kl,
        autocast_enabled=device.type == "cuda",
    )
    kernel = _compile_kernel(kernel, device=device, compile_mode=mode)
    cache[key] = kernel
    return kernel


def _get_value_only_kernel(
    model: torch.nn.Module,
    *,
    compile_mode: str | None,
) -> torch.nn.Module:
    device = _module_device(model)
    mode = compile_mode if device.type == "cuda" else None
    key = ("value_only", mode)
    cache = _kernel_cache(model)
    cached = cache.get(key)
    if cached is not None:
        return cached
    kernel = _ValueOnlyMinibatchKernel(
        model,
        autocast_enabled=device.type == "cuda",
    )
    kernel = _compile_kernel(kernel, device=device, compile_mode=mode)
    cache[key] = kernel
    return kernel


@dataclass
class PPOLog:
    policy_loss: float
    value_loss: float
    entropy: float
    approx_kl: float       # importance-ratio diagnostic E[old_lp − new_lp]; PMPO does not use this for the loss
    pmpo_kl: float         # launch KL plus probability-weighted conditional target/fraction KL
    pos_frac: float        # fraction of owned-planet samples whose advantage was ≥ 0
    target_entropy: float = 0.0
    fraction_entropy: float = 0.0
    move_prob: float = 0.0
    target_confidence: float = 0.0
    fraction_alpha_mean: float = 0.0
    fraction_alpha_max: float = 0.0
    fraction_beta_mean: float = 0.0
    fraction_beta_max: float = 0.0
    fraction_mode_mean: float = 0.0
    fraction_concentration_mean: float = 0.0
    fraction_concentration_max: float = 0.0
    pmpo_target_kl: float = 0.0
    pmpo_fraction_kl: float = 0.0


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
    horizon = len(rewards)
    advs = np.zeros(horizon, dtype=np.float32)
    gae = 0.0
    for t in reversed(range(horizon)):
        next_v = values[t + 1] if t + 1 < horizon else 0.0
        delta = rewards[t] + gamma * next_v - values[t]
        gae = delta + gamma * lam * gae
        advs[t] = gae
    returns = advs + values
    return advs, returns


def compute_mc_return(rewards: np.ndarray, gamma: float = 1.0) -> np.ndarray:
    """Plain Monte-Carlo discounted return Σ γ^k r_{t+k}.
    """
    horizon = len(rewards)
    out = np.zeros(horizon, dtype=np.float32)
    running = 0.0
    for t in reversed(range(horizon)):
        running = rewards[t] + gamma * running
        out[t] = running
    return out


def ppo_update(
    model: OrbitPolicy,
    optimizer: torch.optim.Optimizer,
    batch: dict[str, torch.Tensor],
    *,
    value_coef: float,
    target_entropy_coef: float,
    fraction_entropy_coef: float,
    pmpo_kl_coef: float,
    pmpo_pos_to_neg_weight: float,
    pmpo_reverse_kl: bool,
    epochs: int,
    minibatch_size: int,
    grad_clip: float,
    compile_mode: str | None = None,
) -> PPOLog:
    """Run `epochs × ⌈N/B⌉` minibatch updates on `batch`.

    Expected keys:
      `planet_feats`, `planet_mask`, `planet_owned_mask`, `planet_ids`,
      `planet_garrison`, `fleet_feats`, `fleet_mask`,
      `launch` [B,P], `target_idx` [B,P], `fraction` [B,P],
      `old_log_prob` [B,P], `advantage` [B], `return` [B],
      `owned_mask` [B,P],
      `old_launch_logits` [B,P],
      `old_target_logits` [B,P,P],
      `old_fraction_alpha` [B,P], `old_fraction_beta` [B,P].

    PMPO surrogate: `policy_loss = −α·mean(scaled[pos]) + (1−α)·mean(scaled[neg])`
    where `scaled = action_factor_log_prob · |tanh(advantage)|`. Launch is
    always a factor; target/fraction are factors only for materialized launches.
    No PPO clip.
    Trust region is the PMPO KL term: full Bernoulli launch KL plus
    probability-weighted conditional target/fraction KL. `pmpo_reverse_kl=True`
    matches dreamer4's direction for each factor; setting `False` flips the
    factor direction.

    Distributional value loss: cross-entropy against HL-Gauss-encoded
    returns (`value_encoder.target_probs(returns)`). No value clipping.
    """
    n = batch["planet_feats"].shape[0]
    device = batch["planet_feats"].device

    metric_sum: torch.Tensor | None = None
    n_steps = 0

    kernel = _get_ppo_kernel(
        model,
        value_coef=value_coef,
        target_entropy_coef=target_entropy_coef,
        fraction_entropy_coef=fraction_entropy_coef,
        pmpo_kl_coef=pmpo_kl_coef,
        pmpo_pos_to_neg_weight=pmpo_pos_to_neg_weight,
        pmpo_reverse_kl=pmpo_reverse_kl,
        compile_mode=compile_mode,
    )
    for _ in range(epochs):
        for mb, row_weight in _fixed_minibatches(n, minibatch_size, device):
            _mark_cuda_graph_step(device)
            loss, metrics = kernel(
                batch["planet_feats"][mb],
                batch["planet_mask"][mb],
                batch["planet_owned_mask"][mb],
                batch["planet_ids"][mb],
                batch["planet_garrison"][mb],
                batch["fleet_feats"][mb],
                batch["fleet_mask"][mb],
                row_weight,
                batch["launch"][mb],
                batch["target_idx"][mb],
                batch["fraction"][mb],
                batch["old_log_prob"][mb],
                batch["advantage"][mb],
                batch["return"][mb],
                batch["owned_mask"][mb],
                batch["old_launch_logits"][mb],
                batch["old_target_logits"][mb],
                batch["old_fraction_alpha"][mb],
                batch["old_fraction_beta"][mb],
            )

            optimizer.zero_grad(set_to_none=True)
            (loss * _minibatch_loss_scale(row_weight)).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            if metric_sum is None:
                metric_sum = torch.zeros_like(metrics)
            metric_sum += metrics
            n_steps += 1

    n_steps = max(1, n_steps)
    logs = (
        [0.0] * 19
        if metric_sum is None
        else (metric_sum / n_steps).detach().cpu().tolist()
    )
    return PPOLog(
        policy_loss=float(logs[0]),
        value_loss=float(logs[1]),
        entropy=float(logs[2]),
        approx_kl=float(logs[3]),
        pmpo_kl=float(logs[4]),
        pos_frac=float(logs[5]),
        target_entropy=float(logs[6]),
        fraction_entropy=float(logs[7]),
        move_prob=float(logs[8]),
        target_confidence=float(logs[9]),
        fraction_alpha_mean=float(logs[10]),
        fraction_alpha_max=float(logs[11]),
        fraction_beta_mean=float(logs[12]),
        fraction_beta_max=float(logs[13]),
        fraction_mode_mean=float(logs[14]),
        fraction_concentration_mean=float(logs[15]),
        fraction_concentration_max=float(logs[16]),
        pmpo_target_kl=float(logs[17]),
        pmpo_fraction_kl=float(logs[18]),
    )


def value_only_update(
    model: OrbitPolicy,
    optimizer: torch.optim.Optimizer,
    batch: dict[str, torch.Tensor],
    *,
    epochs: int,
    minibatch_size: int,
    grad_clip: float,
    compile_mode: str | None = None,
) -> float:
    """Critic-only distributional CE update for the value-pretraining phase.

    `batch["return"]` should be Monte-Carlo returns from the configured reward.
    Run this for a few hundred steps against a frozen behavior policy before
    turning on PPO. Mirrors the same HL-Gauss CE loss `ppo_update` uses, so the
    cold-start critic sees the same target distribution it'll train against
    later.

    Reports mean value loss over the pass.
    """
    n = batch["planet_feats"].shape[0]
    device = batch["planet_feats"].device
    total: torch.Tensor | None = None
    n_steps = 0
    kernel = _get_value_only_kernel(model, compile_mode=compile_mode)
    for _ in range(epochs):
        for mb, row_weight in _fixed_minibatches(n, minibatch_size, device):
            _mark_cuda_graph_step(device)
            value_loss, metric = kernel(
                batch["planet_feats"][mb],
                batch["planet_mask"][mb],
                batch["planet_owned_mask"][mb],
                batch["planet_ids"][mb],
                batch["planet_garrison"][mb],
                batch["fleet_feats"][mb],
                batch["fleet_mask"][mb],
                row_weight,
                batch["return"][mb],
            )

            optimizer.zero_grad(set_to_none=True)
            (value_loss * _minibatch_loss_scale(row_weight)).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            if total is None:
                total = metric.new_zeros(())
            total += metric
            n_steps += 1

    if total is None:
        return 0.0
    return float((total / max(1, n_steps)).detach().cpu())
