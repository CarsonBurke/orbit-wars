"""Asymmetric clip-higher PPO update with a distributional critic.

The policy loss is CleanRL iterthink_v24_beta's asymmetric "clip-higher"
(DAPO) surrogate:

    J = E[ min(r*A, clamp(r, 1-clip_coef, 1+clip_coef_high)*A) ]

with a looser upper bound (`clip_coef_high`) than lower (`clip_coef`), so an
already-favored action's probability is capped per update while an
under-weighted one can still recover. The critic emits `value_logits` over a
fixed bin support and trains with cross-entropy against HL-Gauss-encoded
returns. No scalar value clipping is used because the distributional CE
gradients are already bounded per bin.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812

from ..policies.features import EncodedObs, fleet_target_planet_idx_or_empty
from ..policies.model import OrbitPolicy, normalize_matrices
from ..policies.sampling import (
    _categorical_action_log_probs,
    _threshold_normal_launch_entropy,
    _threshold_normal_launch_log_prob,
    _threshold_normal_launch_prob,
)


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
        global_feats=None
        if batch.get("global_feats") is None
        else batch["global_feats"][mb],
        fleet_target_planet_idx=None
        if batch.get("fleet_target_planet_idx") is None
        else batch["fleet_target_planet_idx"][mb],
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
    *,
    weight_device: torch.device | None = None,
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
    weight_device = device if weight_device is None else weight_device
    idx = torch.randperm(n, device=device)
    if n < size:
        extra = idx[torch.randint(n, (size - n,), device=device)]
        mb = torch.cat((idx, extra), dim=0)
        weight = torch.cat(
            (
                torch.ones(n, device=weight_device, dtype=torch.float32),
                torch.zeros(size - n, device=weight_device, dtype=torch.float32),
            ),
            dim=0,
        )
        return [(mb, weight)]

    batches: list[tuple[torch.Tensor, torch.Tensor]] = []
    for start in range(0, n, size):
        mb = idx[start : start + size]
        real = mb.shape[0]
        weight = torch.ones(real, device=weight_device, dtype=torch.float32)
        if mb.shape[0] < size:
            mb = torch.cat((mb, idx[: size - mb.shape[0]]), dim=0)
            weight = torch.cat(
                (
                    weight,
                    torch.zeros(size - real, device=weight_device, dtype=torch.float32),
                ),
                dim=0,
            )
        batches.append((mb, weight))
    return batches


def _fixed_minibatches_by_count(
    n: int,
    minibatch_count: int,
    device: torch.device,
    *,
    weight_device: torch.device | None = None,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Return exactly `minibatch_count` shuffled equal-shape minibatches.

    The final logical minibatch is padded with repeated rows and zero weights
    when `n` is not divisible by `minibatch_count`, so every real rollout row
    contributes once per epoch and every optimizer step sees the same leading
    dimension within that rollout update.
    """
    if n <= 0:
        return []
    count = max(1, int(minibatch_count))
    size = math.ceil(n / count)
    weight_device = device if weight_device is None else weight_device
    idx = torch.randperm(n, device=device)
    batches: list[tuple[torch.Tensor, torch.Tensor]] = []
    cursor = 0
    for _ in range(count):
        real = max(0, min(size, n - cursor))
        mb = idx[cursor : cursor + real]
        cursor += real
        weight = torch.ones(real, device=weight_device, dtype=torch.float32)
        if real < size:
            extra = idx[torch.randint(n, (size - real,), device=device)]
            mb = torch.cat((mb, extra), dim=0)
            weight = torch.cat(
                (
                    weight,
                    torch.zeros(size - real, device=weight_device, dtype=torch.float32),
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


def _minibatch_loss_scale(
    row_weight: torch.Tensor,
    denominator: int | None = None,
) -> torch.Tensor:
    """Scale padded-tail minibatch gradients by the fraction of real rows."""
    return row_weight.sum().clamp_min(1.0) / max(
        1,
        row_weight.numel() if denominator is None else int(denominator),
    )


def _distributional_value_loss(
    value_encoder: torch.nn.Module,
    value_logits: torch.Tensor,
    returns: torch.Tensor,
    row_weight: torch.Tensor,
    return_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """HL-Gauss CE, summed over valid critic MTP horizons per row.

    `value_logits` may be legacy `[B, bins]` or MTP `[B, H, bins]`.
    For MTP, `returns` and `return_mask` are `[B, H]`; invalid episode-tail
    horizons contribute zero. The horizon losses are summed per row, then
    reduced with the usual minibatch row weights.
    """
    if value_logits.dim() == 2:
        if returns.dim() == 2:
            returns = returns[:, 0]
            if return_mask is not None:
                row_weight = row_weight * return_mask[:, 0].to(
                    device=row_weight.device,
                    dtype=row_weight.dtype,
                )
        target_probs = value_encoder.target_probs(returns.float())
        log_probs = F.log_softmax(value_logits, dim=-1)
        value_ce = -(target_probs * log_probs).sum(dim=-1)
        return _weighted_mean(value_ce, row_weight.to(value_ce.device))

    if returns.dim() == 1:
        returns = returns.unsqueeze(-1)
    mtp_h = returns.shape[-1]
    value_logits = value_logits[:, :mtp_h]
    if return_mask is None:
        return_mask = torch.ones_like(returns, dtype=torch.bool)
    else:
        return_mask = return_mask[:, :mtp_h]
    return_mask_f = return_mask.to(device=value_logits.device, dtype=value_logits.dtype)
    target_probs = value_encoder.target_probs(returns.float())
    log_probs = F.log_softmax(value_logits, dim=-1)
    value_ce = -(target_probs * log_probs).sum(dim=-1)
    value_ce = (value_ce * return_mask_f).sum(dim=-1)
    return _weighted_mean(value_ce, row_weight.to(value_ce.device))


def _safe_target_logits(target_logits: torch.Tensor) -> torch.Tensor:
    finite = torch.isfinite(target_logits).any(dim=-1, keepdim=True)
    return torch.where(finite, target_logits, torch.zeros_like(target_logits))


def _bernoulli_log_probs(logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    log_p1 = -F.softplus(-logits)
    log_p0 = -F.softplus(logits)
    return log_p0, log_p1


def _bernoulli_entropy(logits: torch.Tensor) -> torch.Tensor:
    p = logits.sigmoid()
    log_p0, log_p1 = _bernoulli_log_probs(logits)
    return -(p * log_p1 + (1.0 - p) * log_p0)


def _conditional_action_entropy(
    launch_logits: torch.Tensor,
    target_log_probs: torch.Tensor,
    fraction_entropy: torch.Tensor,
) -> torch.Tensor:
    """Entropy of launch + P(launch) * (target + fraction)."""
    min_real = torch.finfo(target_log_probs.dtype).min
    target_probs = target_log_probs.exp()
    target_entropy = -(target_probs * target_log_probs.clamp_min(min_real)).sum(
        dim=-1
    )
    launch_prob = launch_logits.sigmoid()
    return _bernoulli_entropy(launch_logits) + launch_prob * (
        target_entropy + fraction_entropy
    )


BETA_SAMPLE_EPS: float = 1e-6
RANK_GAUSS_CLAMP: float = 0.999


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


def _deterministic_beta_fraction(alpha: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
    return (alpha.float() / (alpha.float() + beta.float())).clamp(
        BETA_SAMPLE_EPS, 1.0 - BETA_SAMPLE_EPS
    )


def _rank_gaussian_advantage(
    advantage: torch.Tensor,
    *,
    clamp: float = RANK_GAUSS_CLAMP,
) -> torch.Tensor:
    """Map raw advantages to empirical Gaussian quantiles over the full batch."""
    flat = advantage.float().flatten()
    if flat.numel() == 0:
        return advantage.float()
    ranks = flat.argsort().argsort().to(torch.float32)
    quantile = (ranks + 0.5) / float(flat.numel())
    centered = (2.0 * quantile - 1.0).clamp(-float(clamp), float(clamp))
    shaped = math.sqrt(2.0) * torch.erfinv(centered)
    return shaped.reshape_as(advantage.float()).to(device=advantage.device)


def _shape_policy_advantage(
    advantage: torch.Tensor,
    *,
    transform: str,
) -> torch.Tensor:
    if transform == "rankgauss":
        return _rank_gaussian_advantage(advantage)
    if transform == "none":
        return advantage.float()
    raise ValueError(f"unknown PPO advantage_transform={transform!r}")


def _module_device(model: torch.nn.Module) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def _slice_to_device(
    tensor: torch.Tensor,
    mb: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    out = tensor[mb]
    if out.device == device:
        return out
    return out.to(device, non_blocking=True)


def _global_feats_or_empty(batch: dict[str, torch.Tensor]) -> torch.Tensor:
    global_feats = batch.get("global_feats")
    if global_feats is not None:
        return global_feats
    return batch["planet_feats"].new_zeros(batch["planet_feats"].shape[0], 0)


def _stage_ppo_minibatch(
    batch: dict[str, torch.Tensor],
    policy_advantage: torch.Tensor,
    return_mtp: torch.Tensor,
    return_mtp_mask: torch.Tensor,
    mb: torch.Tensor,
    row_weight: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, ...]:
    """Slice one logical minibatch and move it to the model device.

    The full PPO rollout batch may live on CPU to keep VRAM bounded. The compiled
    CUDA kernel still receives static-shape CUDA tensors; host-to-device staging
    stays outside the compiled fullgraph body.
    """
    return (
        _slice_to_device(_global_feats_or_empty(batch), mb, device),
        _slice_to_device(batch["planet_feats"], mb, device),
        _slice_to_device(batch["planet_mask"], mb, device),
        _slice_to_device(batch["planet_owned_mask"], mb, device),
        _slice_to_device(batch["planet_ids"], mb, device),
        _slice_to_device(batch["planet_garrison"], mb, device),
        _slice_to_device(batch["fleet_feats"], mb, device),
        _slice_to_device(batch["fleet_mask"], mb, device),
        _slice_to_device(
            fleet_target_planet_idx_or_empty(
                EncodedObs(
                    planet_feats=batch["planet_feats"],
                    planet_mask=batch["planet_mask"],
                    planet_owned_mask=batch["planet_owned_mask"],
                    planet_ids=batch["planet_ids"],
                    planet_garrison=batch["planet_garrison"],
                    fleet_feats=batch["fleet_feats"],
                    fleet_mask=batch["fleet_mask"],
                    fleet_target_planet_idx=batch.get("fleet_target_planet_idx"),
                )
            ),
            mb,
            device,
        ),
        (
            row_weight
            if row_weight.device == device
            else row_weight.to(device, non_blocking=True)
        ),
        _slice_to_device(batch["launch"], mb, device),
        _slice_to_device(batch["target_idx"], mb, device),
        _slice_to_device(batch["fraction"], mb, device),
        _slice_to_device(batch["old_log_prob"], mb, device),
        _slice_to_device(policy_advantage, mb, device),
        _slice_to_device(return_mtp, mb, device),
        _slice_to_device(return_mtp_mask, mb, device),
        _slice_to_device(batch["owned_mask"], mb, device),
        _slice_to_device(batch["target_legal_mask"], mb, device),
    )


def _stage_value_minibatch(
    batch: dict[str, torch.Tensor],
    return_mtp: torch.Tensor,
    return_mtp_mask: torch.Tensor,
    mb: torch.Tensor,
    row_weight: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, ...]:
    """Slice one value-pretrain minibatch and move it to the model device."""
    return (
        _slice_to_device(_global_feats_or_empty(batch), mb, device),
        _slice_to_device(batch["planet_feats"], mb, device),
        _slice_to_device(batch["planet_mask"], mb, device),
        _slice_to_device(batch["planet_owned_mask"], mb, device),
        _slice_to_device(batch["planet_ids"], mb, device),
        _slice_to_device(batch["planet_garrison"], mb, device),
        _slice_to_device(batch["fleet_feats"], mb, device),
        _slice_to_device(batch["fleet_mask"], mb, device),
        _slice_to_device(
            fleet_target_planet_idx_or_empty(
                EncodedObs(
                    planet_feats=batch["planet_feats"],
                    planet_mask=batch["planet_mask"],
                    planet_owned_mask=batch["planet_owned_mask"],
                    planet_ids=batch["planet_ids"],
                    planet_garrison=batch["planet_garrison"],
                    fleet_feats=batch["fleet_feats"],
                    fleet_mask=batch["fleet_mask"],
                    fleet_target_planet_idx=batch.get("fleet_target_planet_idx"),
                )
            ),
            mb,
            device,
        ),
        (
            row_weight
            if row_weight.device == device
            else row_weight.to(device, non_blocking=True)
        ),
        _slice_to_device(return_mtp, mb, device),
        _slice_to_device(return_mtp_mask, mb, device),
    )


_ACTOR_CLIP_PATTERNS: tuple[str, ...] = (
    "target_query",
    "target_key",
    "target_noop_key",
    "target_q_gain",
    "fraction_alpha_head",
    "fraction_beta_head",
)

_CRITIC_CLIP_PATTERNS: tuple[str, ...] = ("value_head",)


def _grad_clip_groups(
    model: torch.nn.Module,
) -> tuple[list[torch.nn.Parameter], list[torch.nn.Parameter], list[torch.nn.Parameter]]:
    cache = model.__dict__.get("_owars_grad_clip_groups")
    if cache is not None:
        return cache
    actor: list[torch.nn.Parameter] = []
    critic: list[torch.nn.Parameter] = []
    shared: list[torch.nn.Parameter] = []
    for name, param in model.named_parameters():
        if any(pattern in name for pattern in _ACTOR_CLIP_PATTERNS):
            actor.append(param)
        elif any(pattern in name for pattern in _CRITIC_CLIP_PATTERNS):
            critic.append(param)
        else:
            shared.append(param)
    cache = (actor, critic, shared)
    model.__dict__["_owars_grad_clip_groups"] = cache
    return cache


def _clip_grad_norm(
    params: list[torch.nn.Parameter],
    max_norm: float,
    device: torch.device,
) -> torch.Tensor:
    if not params:
        return torch.zeros((), device=device)
    return torch.nn.utils.clip_grad_norm_(params, max_norm)


def _grad_clip_scale(raw_norm: torch.Tensor, max_norm: float) -> torch.Tensor:
    if not math.isfinite(max_norm):
        return torch.ones_like(raw_norm)
    max_norm_t = torch.as_tensor(max_norm, device=raw_norm.device, dtype=raw_norm.dtype)
    return (max_norm_t / (raw_norm + 1e-6)).clamp(max=1.0)


def _clear_param_grads(params: list[torch.nn.Parameter]) -> None:
    for param in params:
        param.grad = None


def _backward_actor_critic_with_group_clips(
    model: torch.nn.Module,
    actor_loss: torch.Tensor,
    critic_loss: torch.Tensor,
    max_norm: float,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Backprop actor and critic losses from one graph, then merge grads.

    This mirrors the CleanRL dual-backward path: the critic loss is
    backpropagated first, critic-head plus shared-trunk gradients are clipped
    and stashed, then the actor loss is backpropagated and actor-head plus
    shared-trunk gradients are clipped. The stashed critic gradients are finally
    added back, so shared parameters receive a sum of separately clipped actor
    and critic signals.

    The retained graph is consumed immediately by the actor backward in the
    same minibatch; no trajectory, PPO batch, or logger should keep references
    to tensors from that graph after this helper returns.
    """
    actor, critic, shared = _grad_clip_groups(model)
    device = _module_device(model)
    clip_norm = max_norm if max_norm > 0.0 else float("inf")
    actor_params = [*actor, *shared]
    critic_params = [*critic, *shared]
    all_params = [*actor, *critic, *shared]

    _clear_param_grads(all_params)
    critic_loss.backward(retain_graph=True)
    critic_shared_raw_norm = _clip_grad_norm(shared, float("inf"), device)
    critic_norm = _clip_grad_norm(critic_params, clip_norm, device)
    critic_clip_scale = _grad_clip_scale(critic_norm, clip_norm)
    critic_shared_norm = critic_shared_raw_norm * critic_clip_scale
    critic_clip_frac = (critic_clip_scale < 1.0).to(critic_norm.dtype)
    critic_grads = [
        (param, param.grad.detach().clone())
        for param in critic_params
        if param.grad is not None
    ]

    _clear_param_grads(all_params)
    actor_loss.backward()
    actor_shared_raw_norm = _clip_grad_norm(shared, float("inf"), device)
    actor_norm = _clip_grad_norm(actor_params, clip_norm, device)
    actor_clip_scale = _grad_clip_scale(actor_norm, clip_norm)
    actor_shared_norm = actor_shared_raw_norm * actor_clip_scale
    actor_clip_frac = (actor_clip_scale < 1.0).to(actor_norm.dtype)
    for param, grad in critic_grads:
        param.grad = grad if param.grad is None else param.grad + grad

    shared_norm = _clip_grad_norm(shared, float("inf"), device)
    return (
        actor_norm,
        critic_norm,
        actor_shared_norm,
        critic_shared_norm,
        shared_norm,
        actor_shared_raw_norm,
        critic_shared_raw_norm,
        actor_clip_scale,
        critic_clip_scale,
        actor_clip_frac,
        critic_clip_frac,
    )


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
        norm_advantage: bool,
        clip_coef: float,
        clip_coef_high: float,
        autocast_enabled: bool,
        include_value: bool,
    ) -> None:
        super().__init__()
        self.model = model
        self.value_coef = float(value_coef)
        self.target_entropy_coef = float(target_entropy_coef)
        self.fraction_entropy_coef = float(fraction_entropy_coef)
        self.norm_advantage = bool(norm_advantage)
        self.clip_coef = float(clip_coef)
        self.clip_coef_high = float(clip_coef_high)
        self.autocast_enabled = bool(autocast_enabled)
        self.include_value = bool(include_value)
        self._model_accepts_head_flags = isinstance(model, OrbitPolicy)

    def forward(
        self,
        global_feats: torch.Tensor,
        planet_feats: torch.Tensor,
        planet_mask: torch.Tensor,
        planet_owned_mask: torch.Tensor,
        planet_ids: torch.Tensor,
        planet_garrison: torch.Tensor,
        fleet_feats: torch.Tensor,
        fleet_mask: torch.Tensor,
        fleet_target_planet_idx: torch.Tensor,
        row_weight: torch.Tensor,
        launch: torch.Tensor,
        target_idx: torch.Tensor,
        fraction: torch.Tensor,
        old_log_prob: torch.Tensor,
        advantage: torch.Tensor,
        ret_mtp: torch.Tensor,
        ret_mtp_mask: torch.Tensor,
        owned_mask: torch.Tensor,
        target_legal_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        feats = EncodedObs(
            planet_feats=planet_feats,
            planet_mask=planet_mask,
            planet_owned_mask=planet_owned_mask,
            planet_ids=planet_ids,
            planet_garrison=planet_garrison,
            fleet_feats=fleet_feats,
            fleet_mask=fleet_mask,
            global_feats=global_feats,
            fleet_target_planet_idx=fleet_target_planet_idx,
        )
        with torch.autocast(
            device_type="cuda", dtype=torch.bfloat16, enabled=self.autocast_enabled
        ):
            if self._model_accepts_head_flags:
                out = self.model(feats, include_value=self.include_value)
            else:
                out = self.model(feats)

        target_legal_mask = target_legal_mask.bool()
        has_legal_target = target_legal_mask.any(dim=-1)
        launch_logits = out.launch_logits.float()
        out_action_logit_softcap = getattr(out, "action_logit_softcap", None)
        action_logit_softcap = (
            None if out_action_logit_softcap is None else float(out_action_logit_softcap)
        )
        if action_logit_softcap is None:
            launch_logits = launch_logits.masked_fill(~has_legal_target, -20.0)
        out_launch_log_std = getattr(out, "launch_log_std", None)
        launch_log_std = (
            None if out_launch_log_std is None else out_launch_log_std.float()
        )
        launch_prob_floor = float(getattr(out, "launch_prob_floor", 0.0))
        target_logits = out.target_logits.float().masked_fill(~target_legal_mask, float("-inf"))
        if out.fraction_alpha is None or out.fraction_beta is None:
            raise ValueError("PPO OrbitPolicy output must include Beta fraction params")
        fraction_alpha = out.fraction_alpha.float()
        fraction_beta = out.fraction_beta.float()
        value_logits = out.value_logits.float()

        owned_f = owned_mask.float()
        p = target_logits.shape[1]
        launch_f = launch.float().clamp(0.0, 1.0)
        target = target_idx.clamp(0, p - 1)
        if action_logit_softcap is None:
            target_logits = _safe_target_logits(target_logits)
            launch_lp = _threshold_normal_launch_log_prob(
                launch_logits,
                launch_log_std,
                launch_f,
                launch_prob_floor,
            )
            target_log_probs = F.log_softmax(target_logits, dim=-1)
            target_dist_probs = target_log_probs.exp()
            target_lp = target_log_probs.gather(-1, target.unsqueeze(-1)).squeeze(-1)
            action_lp = launch_lp + launch_f * target_lp
        else:
            action_log_probs = _categorical_action_log_probs(
                launch_logits,
                target_logits,
                action_logit_softcap,
            )
            action_probs = action_log_probs.exp()
            action_idx = torch.where(
                launch_f > 0.5,
                target + 1,
                torch.zeros_like(target),
            )
            action_lp = action_log_probs.gather(-1, action_idx.unsqueeze(-1)).squeeze(-1)
            target_log_probs = action_log_probs[..., 1:]
            target_dist_probs = action_probs[..., 1:]
        frac_lp = _beta_log_prob(fraction_alpha, fraction_beta, fraction.float())
        chosen = action_lp + launch_f * frac_lp

        adv_b = advantage.float().unsqueeze(-1).expand_as(chosen)
        row_w = row_weight.to(device=owned_f.device, dtype=owned_f.dtype)
        row_w_b = row_w.unsqueeze(-1)
        owned_w = owned_f * row_w_b
        denom = owned_w.sum().clamp_min(1.0)

        log_ratio = torch.where(
            owned_f > 0.0,
            chosen - old_log_prob.float(),
            torch.zeros_like(chosen),
        )
        ratio = log_ratio.exp()
        adv_actor = adv_b
        if self.norm_advantage:
            adv_mean = (adv_actor * owned_w).sum() / denom
            adv_var = (((adv_actor - adv_mean).square()) * owned_w).sum() / denom
            adv_actor = (adv_actor - adv_mean) * torch.rsqrt(adv_var + 1e-8)
        # Asymmetric PPO "clip-higher" (DAPO / cleanRL iterthink_v24_beta): the
        # pessimistic max of the unclipped and ratio-clamped surrogates, with a
        # looser upper bound (`clip_coef_high`) than lower (`clip_coef`). The
        # clamp caps how far one update can push an already-favored action while
        # still letting the policy recover an under-weighted one.
        ratio_clamped = ratio.clamp(1.0 - self.clip_coef, 1.0 + self.clip_coef_high)
        pg_unclipped = -adv_actor * ratio
        pg_clipped = -adv_actor * ratio_clamped
        policy_loss = _weighted_mean(torch.maximum(pg_unclipped, pg_clipped), owned_w)

        if self.include_value:
            value_loss = _distributional_value_loss(
                self.model.value_encoder,
                value_logits,
                ret_mtp,
                row_w,
                ret_mtp_mask,
            )
        else:
            value_loss = policy_loss * 0.0

        frac_entropy_per_planet = _beta_entropy(fraction_alpha, fraction_beta)
        if action_logit_softcap is None:
            p_move_current = _threshold_normal_launch_prob(
                launch_logits,
                launch_log_std,
                launch_prob_floor,
            )
            planet_entropy = _conditional_action_entropy(
                torch.logit(p_move_current.clamp(1e-7, 1.0 - 1e-7)),
                target_log_probs,
                frac_entropy_per_planet,
            )
            launch_entropy = (
                _threshold_normal_launch_entropy(
                    launch_logits,
                    launch_log_std,
                    launch_prob_floor,
                )
                * owned_w
            ).sum() / denom
            min_real = torch.finfo(target_log_probs.dtype).min
            target_entropy_per_planet = -(
                target_dist_probs * target_log_probs.clamp_min(min_real)
            ).sum(dim=-1)
            target_entropy = (
                (p_move_current * target_entropy_per_planet) * owned_w
            ).sum() / denom
        else:
            safe_action_log_probs = torch.where(
                torch.isfinite(action_log_probs),
                action_log_probs,
                torch.zeros_like(action_log_probs),
            )
            action_entropy = -(
                action_probs * safe_action_log_probs
            ).sum(dim=-1)
            p_move_current = target_dist_probs.sum(dim=-1)
            planet_entropy = action_entropy + p_move_current * frac_entropy_per_planet
            target_entropy = (action_entropy * owned_w).sum() / denom
            launch_entropy = target_entropy * 0.0
        entropy = (planet_entropy * owned_w).sum() / denom
        fraction_entropy = ((p_move_current * frac_entropy_per_planet) * owned_w).sum() / denom
        move_prob = (p_move_current * owned_w).sum() / denom
        legal_target_count = target_legal_mask.to(dtype=owned_f.dtype).sum(dim=-1)
        legal_target_count_mean = (legal_target_count * owned_w).sum() / denom
        uniform_move_prior = (
            (legal_target_count > 0.0).to(dtype=owned_f.dtype) * 0.5
            * owned_w
        ).sum() / denom
        launch_mean = (launch_logits * owned_w).sum() / denom
        if action_logit_softcap is not None:
            launch_log_std_mean = launch_logits.sum() * 0.0
            action_logit_softcap_metric = launch_logits.new_tensor(
                float(action_logit_softcap)
            )
            target_best = target_logits.amax(dim=-1)
            target_best = torch.where(has_legal_target, target_best, launch_logits)
            launch_score_mean = (
                ((launch_logits - target_best) * owned_w).sum()
                / denom
            )
        elif launch_log_std is None:
            launch_log_std_mean = launch_logits.sum() * 0.0
            action_logit_softcap_metric = launch_logits.sum() * 0.0
            launch_score_mean = launch_logits.sum() * 0.0
        else:
            launch_log_std_f = launch_log_std.float()
            launch_log_std_mean = (launch_log_std_f * owned_w).sum() / denom
            action_logit_softcap_metric = launch_logits.sum() * 0.0
            launch_score = launch_logits * torch.exp(-launch_log_std_f)
            launch_score_mean = (launch_score * owned_w).sum() / denom
        target_confidence = (target_dist_probs.amax(dim=-1) * owned_w).sum() / denom
        fraction_alpha_mean = (fraction_alpha * owned_w).sum() / denom
        fraction_beta_mean = (fraction_beta * owned_w).sum() / denom
        concentration = fraction_alpha + fraction_beta
        fraction_concentration_mean = (concentration * owned_w).sum() / denom
        fraction_concentration_max = _weighted_max(concentration, owned_w)
        fraction_skew_abs_mean = (
            ((fraction_alpha - fraction_beta).abs() * owned_w).sum() / denom
        )
        deterministic_fraction = _deterministic_beta_fraction(
            fraction_alpha, fraction_beta
        )
        deterministic_fraction_mean = (deterministic_fraction * owned_w).sum() / denom
        entropy_bonus = (
            self.target_entropy_coef * (launch_entropy + target_entropy)
            + self.fraction_entropy_coef * fraction_entropy
        )

        actor_loss = policy_loss - entropy_bonus
        critic_loss = self.value_coef * value_loss

        # CleanRL-style PPO KL diagnostic: collapse the factorized per-planet
        # log-probs into one joint action log-prob per rollout row, then apply
        # Joschu's k3 approximation `E[(r - 1) - log r]`. The policy surrogate
        # above remains per-source-planet; this is only the reported KL scale.
        per_planet_kl = (((ratio - 1.0) - log_ratio) * owned_w).sum() / denom
        log_ratio_abs_mean = (log_ratio.abs() * owned_w).sum() / denom
        log_ratio_abs_max = _weighted_max(log_ratio.abs(), owned_w)
        # Trust-region diagnostics: fraction of owned-planet ratios outside the
        # asymmetric clip band, and the subset hitting the looser upper bound
        # (the side `clip_coef_high` deliberately relaxes).
        clipped_low = ratio < (1.0 - self.clip_coef)
        clipped_high = ratio > (1.0 + self.clip_coef_high)
        ratio_clip_frac = (
            (clipped_low | clipped_high).to(owned_f.dtype) * owned_w
        ).sum() / denom
        ratio_clip_frac_high = (
            clipped_high.to(owned_f.dtype) * owned_w
        ).sum() / denom
        executed_launch_frac = (launch_f * owned_w).sum() / denom

        row_log_ratio = (log_ratio * owned_f).sum(dim=-1)
        row_log_ratio = torch.where(row_w > 0.0, row_log_ratio, row_log_ratio * 0.0)
        row_ratio = row_log_ratio.exp()
        row_denom = row_w.sum().clamp_min(1.0)
        row_launch_count = (launch_f * owned_w).sum(dim=-1)
        turn_no_action_frac = (
            ((row_launch_count <= 0.0).to(row_w.dtype) * row_w).sum() / row_denom
        )
        kl = (((row_ratio - 1.0) - row_log_ratio) * row_w).sum() / row_denom
        row_log_ratio_abs_mean = (row_log_ratio.abs() * row_w).sum() / row_denom
        owned_planets_mean = owned_w.sum() / row_denom
        pos_count = (owned_w * (adv_b >= 0.0).to(owned_f.dtype)).sum()
        total_owned = owned_w.sum().clamp_min(1.0)
        pos_frac = pos_count / total_owned
        metrics = torch.stack(
            [
                policy_loss.detach(),
                value_loss.detach(),
                entropy.detach(),
                kl.detach(),
                ratio_clip_frac_high.detach(),
                pos_frac.detach(),
                target_entropy.detach(),
                fraction_entropy.detach(),
                move_prob.detach(),
                target_confidence.detach(),
                fraction_alpha_mean.detach(),
                fraction_beta_mean.detach(),
                fraction_concentration_mean.detach(),
                fraction_concentration_max.detach(),
                fraction_skew_abs_mean.detach(),
                deterministic_fraction_mean.detach(),
                per_planet_kl.detach(),
                log_ratio_abs_mean.detach(),
                log_ratio_abs_max.detach(),
                ratio_clip_frac.detach(),
                owned_planets_mean.detach(),
                executed_launch_frac.detach(),
                row_log_ratio_abs_mean.detach(),
                launch_mean.detach(),
                launch_log_std_mean.detach(),
                launch_score_mean.detach(),
                action_logit_softcap_metric.detach(),
                turn_no_action_frac.detach(),
                legal_target_count_mean.detach(),
                uniform_move_prior.detach(),
            ]
        ).float()
        return actor_loss, critic_loss, metrics


class _ValueOnlyMinibatchKernel(torch.nn.Module):
    def __init__(self, model: torch.nn.Module, *, autocast_enabled: bool) -> None:
        super().__init__()
        self.model = model
        self.autocast_enabled = bool(autocast_enabled)
        self._model_accepts_head_flags = isinstance(model, OrbitPolicy)

    def forward(
        self,
        global_feats: torch.Tensor,
        planet_feats: torch.Tensor,
        planet_mask: torch.Tensor,
        planet_owned_mask: torch.Tensor,
        planet_ids: torch.Tensor,
        planet_garrison: torch.Tensor,
        fleet_feats: torch.Tensor,
        fleet_mask: torch.Tensor,
        fleet_target_planet_idx: torch.Tensor,
        row_weight: torch.Tensor,
        ret_mtp: torch.Tensor,
        ret_mtp_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        feats = EncodedObs(
            planet_feats=planet_feats,
            planet_mask=planet_mask,
            planet_owned_mask=planet_owned_mask,
            planet_ids=planet_ids,
            planet_garrison=planet_garrison,
            fleet_feats=fleet_feats,
            fleet_mask=fleet_mask,
            global_feats=global_feats,
            fleet_target_planet_idx=fleet_target_planet_idx,
        )
        with torch.autocast(
            device_type="cuda", dtype=torch.bfloat16, enabled=self.autocast_enabled
        ):
            if self._model_accepts_head_flags:
                out = self.model(feats, include_actor=False)
            else:
                out = self.model(feats)
        value_logits = out.value_logits.float()
        value_loss = _distributional_value_loss(
            self.model.value_encoder,
            value_logits,
            ret_mtp,
            row_weight.to(value_logits.device),
            ret_mtp_mask,
        )
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
    norm_advantage: bool,
    clip_coef: float,
    clip_coef_high: float,
    compile_mode: str | None,
    include_value: bool,
) -> torch.nn.Module:
    device = _module_device(model)
    mode = compile_mode if device.type == "cuda" else None
    key = (
        "ppo",
        mode,
        float(value_coef),
        float(target_entropy_coef),
        float(fraction_entropy_coef),
        bool(norm_advantage),
        float(clip_coef),
        float(clip_coef_high),
        bool(include_value),
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
        norm_advantage=norm_advantage,
        clip_coef=clip_coef,
        clip_coef_high=clip_coef_high,
        autocast_enabled=device.type == "cuda",
        include_value=include_value,
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
    approx_kl: float
    ratio_clip_frac_high: float
    pos_frac: float
    actor_grad_norm: float = 0.0
    critic_grad_norm: float = 0.0
    actor_shared_grad_norm: float = 0.0
    critic_shared_grad_norm: float = 0.0
    shared_grad_norm: float = 0.0
    actor_shared_raw_grad_norm: float = 0.0
    critic_shared_raw_grad_norm: float = 0.0
    actor_clip_scale: float = 1.0
    critic_clip_scale: float = 1.0
    actor_clip_frac: float = 0.0
    critic_clip_frac: float = 0.0
    target_entropy: float = 0.0
    fraction_entropy: float = 0.0
    move_prob: float = 0.0
    target_confidence: float = 0.0
    fraction_alpha_mean: float = 0.0
    fraction_beta_mean: float = 0.0
    fraction_concentration_mean: float = 0.0
    fraction_concentration_max: float = 0.0
    fraction_skew_abs_mean: float = 0.0
    deterministic_fraction_mean: float = 0.0
    per_planet_approx_kl: float = 0.0
    log_ratio_abs_mean: float = 0.0
    log_ratio_abs_max: float = 0.0
    ratio_clip_frac: float = 0.0
    owned_planets_mean: float = 0.0
    executed_launch_frac: float = 0.0
    source_non_action_frac: float = 0.0
    turn_no_action_frac: float = 0.0
    row_log_ratio_abs_mean: float = 0.0
    launch_mean: float = 0.0
    launch_log_std_mean: float = 0.0
    launch_score_mean: float = 0.0
    action_logit_softcap: float = 0.0
    legal_target_count_mean: float = 0.0
    uniform_move_prior: float = 0.0
    # Number of PPO epochs actually run.
    epochs_run: float = 0.0


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
    norm_advantage: bool,
    advantage_transform: str,
    clip_coef: float,
    clip_coef_high: float,
    epochs: int,
    minibatch_size: int,
    grad_clip: float,
    minibatch_count: int | None = None,
    compile_mode: str | None = None,
) -> PPOLog:
    """Run PPO minibatch updates on `batch`.

    If `minibatch_count` is set, each epoch runs exactly that many equal-shape
    minibatches. Otherwise each epoch runs `ceil(N / minibatch_size)` batches.

    Expected keys:
      `planet_feats`, `planet_mask`, `planet_owned_mask`, `planet_ids`,
      `planet_garrison`, `fleet_feats`, `fleet_mask`,
      `launch` [B,P], `target_idx` [B,P], `fraction` [B,P],
      `old_log_prob` [B,P], `advantage` [B], `return` [B],
      optional `return_mtp` [B,H], `return_mtp_mask` [B,H],
      `owned_mask` [B,P], `target_legal_mask` [B,P,P].

    The actor objective is asymmetric clip-higher:
      `mean(max(-A*ratio, -A*clamp(ratio, 1-clip_coef, 1+clip_coef_high)))`.

    Distributional value loss: cross-entropy against HL-Gauss-encoded
    returns (`value_encoder.target_probs(returns)`). No value clipping.
    """
    n = batch["planet_feats"].shape[0]
    batch_device = batch["planet_feats"].device
    device = _module_device(model)
    policy_advantage = _shape_policy_advantage(
        batch["advantage"],
        transform=advantage_transform,
    )
    return_mtp = batch.get("return_mtp", batch["return"].unsqueeze(-1))
    return_mtp_mask = batch.get(
        "return_mtp_mask",
        torch.ones_like(return_mtp, dtype=torch.bool),
    )

    metric_sum: torch.Tensor | None = None
    last_metrics: torch.Tensor | None = None
    actor_grad_norm_sum = torch.zeros((), device=device)
    critic_grad_norm_sum = torch.zeros((), device=device)
    actor_shared_grad_norm_sum = torch.zeros((), device=device)
    critic_shared_grad_norm_sum = torch.zeros((), device=device)
    shared_grad_norm_sum = torch.zeros((), device=device)
    actor_shared_raw_grad_norm_sum = torch.zeros((), device=device)
    critic_shared_raw_grad_norm_sum = torch.zeros((), device=device)
    actor_clip_scale_sum = torch.zeros((), device=device)
    critic_clip_scale_sum = torch.zeros((), device=device)
    actor_clip_frac_sum = torch.zeros((), device=device)
    critic_clip_frac_sum = torch.zeros((), device=device)
    n_steps = 0

    kernel = _get_ppo_kernel(
        model,
        value_coef=value_coef,
        target_entropy_coef=target_entropy_coef,
        fraction_entropy_coef=fraction_entropy_coef,
        norm_advantage=norm_advantage,
        clip_coef=clip_coef,
        clip_coef_high=clip_coef_high,
        compile_mode=compile_mode,
        include_value=True,
    )
    epochs_run = 0
    for _ in range(epochs):
        if minibatch_count is not None:
            logical_minibatch_size = math.ceil(n / max(1, int(minibatch_count)))
            minibatches = _fixed_minibatches_by_count(
                n,
                minibatch_count,
                batch_device,
                weight_device=device,
            )
        else:
            logical_minibatch_size = None
            minibatches = _fixed_minibatches(
                n, minibatch_size, batch_device, weight_device=device
            )
        for mb, row_weight in minibatches:
            if compile_mode is not None:
                _mark_cuda_graph_step(device)
            staged = _stage_ppo_minibatch(
                batch,
                policy_advantage,
                return_mtp,
                return_mtp_mask,
                mb,
                row_weight,
                device,
            )
            actor_loss, critic_loss, metrics = kernel(*staged)
            metrics_for_step = metrics.detach().clone()

            optimizer.zero_grad(set_to_none=True)
            loss_scale = _minibatch_loss_scale(row_weight, logical_minibatch_size)
            (
                actor_grad_norm,
                critic_grad_norm,
                actor_shared_grad_norm,
                critic_shared_grad_norm,
                shared_grad_norm,
                actor_shared_raw_grad_norm,
                critic_shared_raw_grad_norm,
                actor_clip_scale,
                critic_clip_scale,
                actor_clip_frac,
                critic_clip_frac,
            ) = (
                _backward_actor_critic_with_group_clips(
                    model,
                    actor_loss * loss_scale,
                    critic_loss * loss_scale,
                    grad_clip,
                )
            )
            optimizer.step()
            # nGPT: re-project the encoder-trunk matrices onto the unit
            # hypersphere after EVERY optimizer step (the actual nGPT invariant;
            # `ngpt/train.py:499-500`, and `normalize_matrices`' own contract).
            # Muon's update is ambient-additive, so each step nudges the trunk
            # rows off the sphere; deferring re-projection to once-per-update
            # lets that drift compound across the epochs×minibatches steps and
            # pulls the policy away from the frozen `old_log_prob`, exploding
            # within-update KL that ratio clipping cannot constrain. Eager —
            # outside the compiled kernel.
            normalize_matrices(model)
            del actor_loss, critic_loss, metrics
            actor_grad_norm_sum += actor_grad_norm.detach()
            critic_grad_norm_sum += critic_grad_norm.detach()
            actor_shared_grad_norm_sum += actor_shared_grad_norm.detach()
            critic_shared_grad_norm_sum += critic_shared_grad_norm.detach()
            shared_grad_norm_sum += shared_grad_norm.detach()
            actor_shared_raw_grad_norm_sum += actor_shared_raw_grad_norm.detach()
            critic_shared_raw_grad_norm_sum += critic_shared_raw_grad_norm.detach()
            actor_clip_scale_sum += actor_clip_scale.detach()
            critic_clip_scale_sum += critic_clip_scale.detach()
            actor_clip_frac_sum += actor_clip_frac.detach()
            critic_clip_frac_sum += critic_clip_frac.detach()

            if metric_sum is None:
                metric_sum = torch.zeros_like(metrics_for_step)
            metric_sum += metrics_for_step
            last_metrics = metrics_for_step
            n_steps += 1
        epochs_run += 1

    n_steps = max(1, n_steps)
    mean_logs_t = (
        torch.zeros(30, device=device)
        if metric_sum is None
        else metric_sum / n_steps
    )
    # Match CleanRL's PPO KL logging: `approx_kl` is the latest minibatch's
    # estimate after the PPO epoch loop, not an epoch mean. The surrounding
    # diagnostics stay averaged to preserve their lower-noise TensorBoard
    # behavior.
    kl_logs_t = (
        torch.zeros(30, device=device)
        if last_metrics is None
        else last_metrics
    )
    grad_logs_t = (
        torch.stack(
            [
                actor_grad_norm_sum,
                critic_grad_norm_sum,
                actor_shared_grad_norm_sum,
                critic_shared_grad_norm_sum,
                shared_grad_norm_sum,
                actor_shared_raw_grad_norm_sum,
                critic_shared_raw_grad_norm_sum,
                actor_clip_scale_sum,
                critic_clip_scale_sum,
                actor_clip_frac_sum,
                critic_clip_frac_sum,
            ]
        )
        / n_steps
    )
    logs = torch.cat((mean_logs_t, kl_logs_t, grad_logs_t)).detach().cpu().tolist()
    mean_logs = logs[:30]
    kl_logs = logs[30:60]
    grad_logs = logs[60:]
    return PPOLog(
        policy_loss=float(mean_logs[0]),
        value_loss=float(mean_logs[1]),
        entropy=float(mean_logs[2]),
        approx_kl=float(kl_logs[3]),
        ratio_clip_frac_high=float(mean_logs[4]),
        pos_frac=float(mean_logs[5]),
        actor_grad_norm=float(grad_logs[0]),
        critic_grad_norm=float(grad_logs[1]),
        actor_shared_grad_norm=float(grad_logs[2]),
        critic_shared_grad_norm=float(grad_logs[3]),
        shared_grad_norm=float(grad_logs[4]),
        actor_shared_raw_grad_norm=float(grad_logs[5]),
        critic_shared_raw_grad_norm=float(grad_logs[6]),
        actor_clip_scale=float(grad_logs[7]),
        critic_clip_scale=float(grad_logs[8]),
        actor_clip_frac=float(grad_logs[9]),
        critic_clip_frac=float(grad_logs[10]),
        target_entropy=float(mean_logs[6]),
        fraction_entropy=float(mean_logs[7]),
        move_prob=float(mean_logs[8]),
        target_confidence=float(mean_logs[9]),
        fraction_alpha_mean=float(mean_logs[10]),
        fraction_beta_mean=float(mean_logs[11]),
        fraction_concentration_mean=float(mean_logs[12]),
        fraction_concentration_max=float(mean_logs[13]),
        fraction_skew_abs_mean=float(mean_logs[14]),
        deterministic_fraction_mean=float(mean_logs[15]),
        per_planet_approx_kl=float(kl_logs[16]),
        log_ratio_abs_mean=float(mean_logs[17]),
        log_ratio_abs_max=float(mean_logs[18]),
        ratio_clip_frac=float(mean_logs[19]),
        owned_planets_mean=float(mean_logs[20]),
        executed_launch_frac=float(mean_logs[21]),
        source_non_action_frac=float(1.0 - mean_logs[21]),
        turn_no_action_frac=float(mean_logs[27]),
        row_log_ratio_abs_mean=float(mean_logs[22]),
        launch_mean=float(mean_logs[23]),
        launch_log_std_mean=float(mean_logs[24]),
        launch_score_mean=float(mean_logs[25]),
        action_logit_softcap=float(mean_logs[26]),
        legal_target_count_mean=float(mean_logs[28]),
        uniform_move_prior=float(mean_logs[29]),
        epochs_run=float(epochs_run),
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

    `batch["return"]` should use the same configured lambda-return convention
    as PPO. Mirrors the same HL-Gauss CE loss `ppo_update` uses, so the
    cold-start critic sees the same target distribution it will train against
    later.

    Reports mean value loss over the pass.
    """
    n = batch["planet_feats"].shape[0]
    batch_device = batch["planet_feats"].device
    device = _module_device(model)
    total: torch.Tensor | None = None
    n_steps = 0
    kernel = _get_value_only_kernel(model, compile_mode=compile_mode)
    return_mtp = batch.get("return_mtp", batch["return"].unsqueeze(-1))
    return_mtp_mask = batch.get(
        "return_mtp_mask",
        torch.ones_like(return_mtp, dtype=torch.bool),
    )
    for _ in range(epochs):
        for mb, row_weight in _fixed_minibatches(
            n,
            minibatch_size,
            batch_device,
            weight_device=device,
        ):
            if compile_mode is not None:
                _mark_cuda_graph_step(device)
            staged = _stage_value_minibatch(
                batch,
                return_mtp,
                return_mtp_mask,
                mb,
                row_weight,
                device,
            )
            value_loss, metric = kernel(*staged)

            optimizer.zero_grad(set_to_none=True)
            (value_loss * _minibatch_loss_scale(row_weight)).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            if total is None:
                total = metric.new_zeros(())
            total += metric
            n_steps += 1

    # nGPT hypersphere re-projection once per value-pretrain update (see the
    # rationale in `ppo_update`).
    normalize_matrices(model)
    if total is None:
        return 0.0
    return float((total / max(1, n_steps)).detach().cpu())
