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
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812

from ..policies.features import (
    EncodedObs,
    fleet_target_planet_idx_or_empty,
    planet_inbound_feats_or_empty,
)
from ..policies.model import OrbitPolicy, normalize_matrices
from ..policies.sampling import (
    _categorical_action_log_probs,
    _threshold_normal_launch_entropy,
    _threshold_normal_launch_log_prob,
    _threshold_normal_launch_prob,
)

PinnedSliceCache = dict[
    tuple[int, str, torch.dtype, tuple[int, ...]],
    tuple[torch.Tensor, torch.cuda.Event | None],
]


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
        global_feats=None if batch.get("global_feats") is None else batch["global_feats"][mb],
        fleet_target_planet_idx=None
        if batch.get("fleet_target_planet_idx") is None
        else batch["fleet_target_planet_idx"][mb],
        planet_inbound_feats=None
        if batch.get("planet_inbound_feats") is None
        else batch["planet_inbound_feats"][mb],
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
    """Return shuffled equal-shape minibatches, capped to non-empty batches.

    The final logical minibatch is padded with repeated rows and zero weights
    when `n` is not divisible by `minibatch_count`, so every real rollout row
    contributes once per epoch and every optimizer step sees the same leading
    dimension within that rollout update. If `minibatch_count > n`, cap the
    count at `n` to avoid zero-real-row optimizer steps.
    """
    if n <= 0:
        return []
    count = min(max(1, int(minibatch_count)), n)
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


def _fixed_order_minibatches(
    n: int,
    minibatch_size: int,
    device: torch.device,
    *,
    minibatch_count: int | None = None,
    weight_device: torch.device | None = None,
) -> list[tuple[torch.Tensor, torch.Tensor, int]]:
    """Return deterministic fixed-size minibatches plus real row counts."""
    if n <= 0:
        return []
    if minibatch_count is not None:
        count = min(max(1, int(minibatch_count)), n)
        size = math.ceil(n / count)
    else:
        size = max(1, int(minibatch_size))
    weight_device = device if weight_device is None else weight_device
    batches: list[tuple[torch.Tensor, torch.Tensor, int]] = []
    for start in range(0, n, size):
        stop = min(n, start + size)
        real = stop - start
        mb = torch.arange(start, stop, device=device)
        weight = torch.ones(real, device=weight_device, dtype=torch.float32)
        if real < size:
            pad = torch.arange(0, size - real, device=device).remainder(n)
            mb = torch.cat((mb, pad), dim=0)
            weight = torch.cat(
                (
                    weight,
                    torch.zeros(size - real, device=weight_device, dtype=torch.float32),
                ),
                dim=0,
            )
        batches.append((mb, weight, real))
    return batches


def _weighted_mean(values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    weights = weights.to(device=values.device, dtype=values.dtype)
    active = weights > 0
    safe_values = torch.where(active, values, torch.zeros_like(values))
    return (safe_values * weights).sum() / weights.sum().clamp_min(1.0)


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


def _raise_if_nonfinite_tensor(name: str, tensor: torch.Tensor) -> None:
    if bool(torch.isfinite(tensor).all()):
        return
    finite = torch.isfinite(tensor)
    bad = int((~finite).sum().item())
    if bool(finite.any()):
        finite_vals = tensor.detach()[finite]
        detail = (
            f" finite_min={float(finite_vals.min().item()):.6g}"
            f" finite_max={float(finite_vals.max().item()):.6g}"
        )
    else:
        detail = " no_finite_values"
    if tensor.numel() <= 128:
        bad_idx = (~finite).nonzero(as_tuple=False).detach().cpu().tolist()
        detail = f"{detail} bad_indices={bad_idx}"
    raise FloatingPointError(
        f"{name} contains {bad}/{tensor.numel()} non-finite values; "
        f"shape={tuple(tensor.shape)}{detail}"
    )


def _raise_if_nonfinite_model_tensors(
    model: torch.nn.Module,
    *,
    what: str,
    gradients: bool,
) -> None:
    for name, param in model.named_parameters():
        tensor = param.grad if gradients else param
        if tensor is None:
            continue
        if bool(torch.isfinite(tensor).all()):
            continue
        _raise_if_nonfinite_tensor(f"{what}:{name}", tensor)


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
    target_entropy = -(target_probs * target_log_probs.clamp_min(min_real)).sum(dim=-1)
    launch_prob = launch_logits.sigmoid()
    return _bernoulli_entropy(launch_logits) + launch_prob * (target_entropy + fraction_entropy)


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


def _launched_beta_log_prob(
    alpha: torch.Tensor,
    beta: torch.Tensor,
    fraction: torch.Tensor,
    launch: torch.Tensor,
) -> torch.Tensor:
    launch_mask = launch.float() > 0.5
    safe_alpha = torch.where(launch_mask, alpha, torch.full_like(alpha, 2.0))
    safe_beta = torch.where(launch_mask, beta, torch.full_like(beta, 2.0))
    safe_fraction = torch.where(launch_mask, fraction, torch.full_like(fraction, 0.5))
    frac_lp = _beta_log_prob(safe_alpha, safe_beta, safe_fraction)
    return torch.where(launch_mask, frac_lp, torch.zeros_like(frac_lp))


def _action_log_prob_for_action(action_lp: torch.Tensor, launch: torch.Tensor) -> torch.Tensor:
    launch_mask = launch.float() > 0.5
    safe_noop_lp = torch.nan_to_num(
        action_lp.float(),
        nan=0.0,
        neginf=-1.0e9,
        posinf=0.0,
    )
    return torch.where(launch_mask, action_lp, safe_noop_lp)


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
    *,
    pinned_cache: PinnedSliceCache | None = None,
    slot: int = 0,
    name: str = "",
) -> torch.Tensor:
    if pinned_cache is not None and tensor.device.type == "cpu" and device.type == "cuda":
        mb_cpu = mb if mb.device.type == "cpu" else mb.cpu()
        shape = (int(mb_cpu.numel()), *tuple(tensor.shape[1:]))
        key = (int(slot), name, tensor.dtype, shape)
        entry = pinned_cache.get(key)
        if entry is None:
            out = torch.empty(shape, dtype=tensor.dtype, device="cpu", pin_memory=True)
            event = None
            pinned_cache[key] = (out, event)
        else:
            out, event = entry
            if event is not None:
                event.synchronize()
        torch.index_select(tensor, 0, mb_cpu, out=out)
        moved = out.to(device, non_blocking=True)
        if event is None:
            event = torch.cuda.Event()
        event.record(torch.cuda.current_stream(device))
        pinned_cache[key] = (out, event)
        return moved

    out = tensor[mb]
    if out.device == device:
        return out
    if out.device.type == "cpu" and device.type == "cuda":
        out = out.contiguous()
        if not out.is_pinned():
            out = out.pin_memory()
    return out.to(device, non_blocking=True)


def _record_streams(value: Any, stream: torch.cuda.Stream) -> None:
    if isinstance(value, torch.Tensor) and value.device.type == "cuda":
        value.record_stream(stream)
    elif isinstance(value, tuple | list):
        for item in value:
            _record_streams(item, stream)


def _prefetch_staged_minibatches(
    stage_fn: Any,
    minibatches: list[tuple[torch.Tensor, torch.Tensor]],
    device: torch.device,
) -> Any:
    """Yield staged minibatches, preloading the next CUDA transfer.

    The CPU rollout batch is sliced outside the compiled PPO kernel. On CUDA,
    each sliced CPU minibatch is pinned in `_slice_to_device`; this helper moves
    those pinned copies to the model device on side streams while the current
    minibatch runs backward/optimizer work on the default stream.
    """
    if device.type != "cuda" or len(minibatches) <= 1:
        for mb, row_weight in minibatches:
            yield stage_fn(mb, row_weight, 0)
        return

    with torch.cuda.device(device):
        streams = [torch.cuda.Stream(), torch.cuda.Stream()]

    def submit(
        item: tuple[torch.Tensor, torch.Tensor],
        stream: torch.cuda.Stream,
        slot: int,
    ):
        mb, row_weight = item
        with torch.cuda.stream(stream):
            return stage_fn(mb, row_weight, slot)

    current_stream = torch.cuda.current_stream(device)
    staged = submit(minibatches[0], streams[0], 0)
    ready_stream = streams[0]
    next_stream_idx = 1
    for item in minibatches[1:]:
        current_stream.wait_stream(ready_stream)
        _record_streams(staged, current_stream)
        next_stream = streams[next_stream_idx]
        next_staged = submit(item, next_stream, next_stream_idx)
        next_stream_idx = 1 - next_stream_idx
        yield staged
        staged = next_staged
        ready_stream = next_stream
        current_stream = torch.cuda.current_stream(device)

    current_stream.wait_stream(ready_stream)
    _record_streams(staged, current_stream)
    yield staged


def _global_feats_or_empty(batch: dict[str, torch.Tensor]) -> torch.Tensor:
    global_feats = batch.get("global_feats")
    if global_feats is not None:
        return global_feats
    return batch["planet_feats"].new_zeros(batch["planet_feats"].shape[0], 0)


def _batch_encoded_view(batch: dict[str, torch.Tensor]) -> EncodedObs:
    return EncodedObs(
        planet_feats=batch["planet_feats"],
        planet_mask=batch["planet_mask"],
        planet_owned_mask=batch["planet_owned_mask"],
        planet_ids=batch["planet_ids"],
        planet_garrison=batch["planet_garrison"],
        fleet_feats=batch["fleet_feats"],
        fleet_mask=batch["fleet_mask"],
        fleet_target_planet_idx=batch.get("fleet_target_planet_idx"),
        planet_inbound_feats=batch.get("planet_inbound_feats"),
        global_feats=batch.get("global_feats"),
    )


def _stage_target_legal_mask(
    batch: dict[str, Any],
    mb: torch.Tensor,
    device: torch.device,
    *,
    pinned_cache: PinnedSliceCache | None = None,
    slot: int = 0,
) -> torch.Tensor:
    dense = batch.get("target_legal_mask")
    if dense is not None:
        return _slice_to_device(
            dense,
            mb,
            device,
            pinned_cache=pinned_cache,
            slot=slot,
            name="target_legal_mask",
        )

    mb_cpu = mb.detach().to(device="cpu", dtype=torch.long)
    owned = batch["target_legal_source_owned_mask"].index_select(0, mb_cpu).bool()
    rows = int(owned.shape[0])
    planets = int(owned.shape[1])
    cache_key = (int(slot), "target_legal_mask_compact", torch.bool, (rows, planets, planets))
    event = None
    if pinned_cache is not None and device.type == "cuda":
        cached = pinned_cache.get(cache_key)
        if cached is None:
            out = torch.empty(
                rows,
                planets,
                planets,
                dtype=torch.bool,
                device="cpu",
                pin_memory=True,
            )
        else:
            out, event = cached
            if event is not None:
                event.synchronize()
        out.fill_(True)
    else:
        out = torch.ones(rows, planets, planets, dtype=torch.bool)
    owned_rows, owned_cols = torch.nonzero(owned, as_tuple=True)
    if owned_rows.numel():
        out[owned_rows, owned_cols] = False

    row_idx = batch["target_legal_row_idx"].long()
    row_offsets = batch.get("target_legal_row_offsets")
    if row_offsets is not None:
        row_offsets = row_offsets.long()
        starts = row_offsets.index_select(0, mb_cpu)
        stops = row_offsets.index_select(0, mb_cpu + 1)
        counts = stops - starts
        total = int(counts.sum().item())
        if total:
            out_rows = torch.repeat_interleave(
                torch.arange(rows, dtype=torch.long),
                counts,
            )
            segment_starts = counts.cumsum(0) - counts
            gather_idx = torch.arange(total, dtype=torch.long) + torch.repeat_interleave(
                starts - segment_starts,
                counts,
            )
            out[
                out_rows,
                batch["target_legal_source_idx"].long().index_select(0, gather_idx),
            ] = batch["target_legal_source_mask"].bool().index_select(0, gather_idx)
    elif row_idx.numel():
        row_to_mb = torch.full(
            (int(batch["target_legal_source_owned_mask"].shape[0]),),
            -1,
            dtype=torch.long,
        )
        row_to_mb[mb_cpu] = torch.arange(rows, dtype=torch.long)
        compact_rows = row_to_mb.index_select(0, row_idx)
        keep = compact_rows >= 0
        if bool(keep.any()):
            out[
                compact_rows[keep],
                batch["target_legal_source_idx"].long()[keep],
            ] = batch["target_legal_source_mask"].bool()[keep]

    if device.type == "cuda":
        if not out.is_pinned():
            out = out.pin_memory()
        moved = out.to(device, non_blocking=True)
        if pinned_cache is not None:
            if event is None:
                event = torch.cuda.Event()
            event.record(torch.cuda.current_stream(device))
            pinned_cache[cache_key] = (out, event)
        out = moved
    elif out.device != device:
        out = out.to(device)
    return out


def _stage_ppo_minibatch(
    batch: dict[str, torch.Tensor],
    global_feats: torch.Tensor,
    fleet_target_planet_idx: torch.Tensor,
    planet_inbound_feats: torch.Tensor,
    policy_advantage: torch.Tensor,
    return_mtp: torch.Tensor,
    return_mtp_mask: torch.Tensor,
    mb: torch.Tensor,
    row_weight: torch.Tensor,
    device: torch.device,
    pinned_cache: PinnedSliceCache | None = None,
    slot: int = 0,
) -> tuple[torch.Tensor, ...]:
    """Slice one logical minibatch and move it to the model device.

    The full PPO rollout batch may live on CPU to keep VRAM bounded. The compiled
    CUDA kernel still receives static-shape CUDA tensors; host-to-device staging
    stays outside the compiled fullgraph body.
    """
    return (
        _slice_to_device(
            global_feats, mb, device, pinned_cache=pinned_cache, slot=slot, name="global_feats"
        ),
        _slice_to_device(
            batch["planet_feats"],
            mb,
            device,
            pinned_cache=pinned_cache,
            slot=slot,
            name="planet_feats",
        ),
        _slice_to_device(
            batch["planet_mask"],
            mb,
            device,
            pinned_cache=pinned_cache,
            slot=slot,
            name="planet_mask",
        ),
        _slice_to_device(
            batch["planet_owned_mask"],
            mb,
            device,
            pinned_cache=pinned_cache,
            slot=slot,
            name="planet_owned_mask",
        ),
        _slice_to_device(
            batch["planet_ids"], mb, device, pinned_cache=pinned_cache, slot=slot, name="planet_ids"
        ),
        _slice_to_device(
            batch["planet_garrison"],
            mb,
            device,
            pinned_cache=pinned_cache,
            slot=slot,
            name="planet_garrison",
        ),
        _slice_to_device(
            batch["fleet_feats"],
            mb,
            device,
            pinned_cache=pinned_cache,
            slot=slot,
            name="fleet_feats",
        ),
        _slice_to_device(
            batch["fleet_mask"], mb, device, pinned_cache=pinned_cache, slot=slot, name="fleet_mask"
        ),
        _slice_to_device(
            fleet_target_planet_idx,
            mb,
            device,
            pinned_cache=pinned_cache,
            slot=slot,
            name="fleet_target_planet_idx",
        ),
        _slice_to_device(
            planet_inbound_feats,
            mb,
            device,
            pinned_cache=pinned_cache,
            slot=slot,
            name="planet_inbound_feats",
        ),
        (row_weight if row_weight.device == device else row_weight.to(device, non_blocking=True)),
        _slice_to_device(
            batch["launch"], mb, device, pinned_cache=pinned_cache, slot=slot, name="launch"
        ),
        _slice_to_device(
            batch["target_idx"], mb, device, pinned_cache=pinned_cache, slot=slot, name="target_idx"
        ),
        _slice_to_device(
            batch["fraction"], mb, device, pinned_cache=pinned_cache, slot=slot, name="fraction"
        ),
        _slice_to_device(
            batch["old_log_prob"],
            mb,
            device,
            pinned_cache=pinned_cache,
            slot=slot,
            name="old_log_prob",
        ),
        _slice_to_device(
            policy_advantage, mb, device, pinned_cache=pinned_cache, slot=slot, name="advantage"
        ),
        _slice_to_device(
            return_mtp, mb, device, pinned_cache=pinned_cache, slot=slot, name="return_mtp"
        ),
        _slice_to_device(
            return_mtp_mask,
            mb,
            device,
            pinned_cache=pinned_cache,
            slot=slot,
            name="return_mtp_mask",
        ),
        _slice_to_device(
            batch["owned_mask"], mb, device, pinned_cache=pinned_cache, slot=slot, name="owned_mask"
        ),
        _stage_target_legal_mask(
            batch,
            mb,
            device,
            pinned_cache=pinned_cache,
            slot=slot,
        ),
    )


def _source_indices_for_minibatch(
    batch: dict[str, torch.Tensor],
    mb_cpu: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    offsets = batch["actor_source_row_offsets"].long()
    starts = offsets.index_select(0, mb_cpu)
    stops = offsets.index_select(0, mb_cpu + 1)
    counts = stops - starts
    total = int(counts.sum().item())
    if total == 0:
        return torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long)
    local_rows = torch.repeat_interleave(
        torch.arange(int(mb_cpu.numel()), dtype=torch.long),
        counts,
    )
    segment_starts = counts.cumsum(0) - counts
    source_idx = torch.arange(total, dtype=torch.long) + torch.repeat_interleave(
        starts - segment_starts,
        counts,
    )
    return source_idx, local_rows


def _source_planet_bucket(observed_sources_per_row: int, planet_width: int) -> int:
    """Log-spaced static source rows for source-major PPO graphs."""
    width = max(1, int(planet_width))
    observed = max(1, int(observed_sources_per_row))
    for bucket in (8, 12, 16, 24, 32, 48, 64):
        capped = min(bucket, width)
        if observed <= capped:
            return capped
    return width


def _source_capacity_for_minibatch(
    static_minibatch_rows: int,
    planet_width: int,
    observed_sources_per_row: int,
) -> int:
    """Bucketed source-major PPO capacity for one compiled minibatch.

    Owned source planets vary over training. Using the exact observed width in
    the compiled kernel shape causes Dynamo to specialize on many adjacent
    counts and eventually hit the fullgraph recompile limit; padding all the way
    to planet width avoids recompiles but makes the graph much larger than the
    actual source-major workload. A small log-spaced bucket schedule keeps graph
    variants bounded while staying close to the current source count.
    """
    return max(1, int(static_minibatch_rows)) * _source_planet_bucket(
        observed_sources_per_row,
        planet_width,
    )


def _pad_first_dim_cpu(
    tensor: torch.Tensor,
    rows: int,
    *,
    fill: int | float | bool = 0,
) -> torch.Tensor:
    current = int(tensor.shape[0])
    if current == rows:
        return tensor.contiguous()
    if current > rows:
        raise RuntimeError("source-major PPO minibatch exceeded static source capacity")
    out = tensor.new_full((rows, *tensor.shape[1:]), fill)
    if current:
        out[:current] = tensor
    return out


def _move_staged_source_field(
    tensor: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    if device.type == "cuda" and tensor.device.type == "cpu" and not tensor.is_pinned():
        tensor = tensor.pin_memory()
    return tensor.to(device, non_blocking=device.type == "cuda")


def _stage_source_ppo_minibatch(
    batch: dict[str, torch.Tensor],
    global_feats: torch.Tensor,
    fleet_target_planet_idx: torch.Tensor,
    planet_inbound_feats: torch.Tensor,
    policy_advantage: torch.Tensor,
    return_mtp: torch.Tensor,
    return_mtp_mask: torch.Tensor,
    mb: torch.Tensor,
    row_weight: torch.Tensor,
    device: torch.device,
    *,
    source_capacity: int,
    pinned_cache: PinnedSliceCache | None = None,
    slot: int = 0,
) -> tuple[torch.Tensor, ...]:
    mb_cpu = mb.detach().to(device="cpu", dtype=torch.long)
    source_idx, source_row_local = _source_indices_for_minibatch(batch, mb_cpu)
    source_count = int(source_idx.numel())
    source_exists = torch.zeros(source_capacity, dtype=torch.bool)
    if source_count:
        source_exists[:source_count] = True
    row_weight_cpu = row_weight.detach().to(device="cpu", dtype=torch.float32)
    source_weight = torch.zeros(source_capacity, dtype=torch.float32)
    if source_count:
        source_weight[:source_count] = row_weight_cpu.index_select(0, source_row_local)
    source_row_local = _pad_first_dim_cpu(source_row_local, source_capacity)
    source_col_idx = _pad_first_dim_cpu(
        batch["actor_source_col_idx"].long().index_select(0, source_idx),
        source_capacity,
    )
    launch = _pad_first_dim_cpu(
        batch["actor_launch"].float().index_select(0, source_idx),
        source_capacity,
    )
    target_idx = _pad_first_dim_cpu(
        batch["actor_target_idx"].long().index_select(0, source_idx),
        source_capacity,
    )
    fraction = _pad_first_dim_cpu(
        batch["actor_fraction"].float().index_select(0, source_idx),
        source_capacity,
        fill=0.5,
    )
    source_global_rows = batch["actor_source_row_idx"].long().index_select(0, source_idx)
    source_global_cols = batch["actor_source_col_idx"].long().index_select(0, source_idx)
    old_log_prob_src = batch["old_log_prob"].float()[source_global_rows, source_global_cols]
    old_log_prob = _pad_first_dim_cpu(old_log_prob_src, source_capacity)
    target_legal_mask = _pad_first_dim_cpu(
        batch["actor_target_legal_mask"].bool().index_select(0, source_idx),
        source_capacity,
        fill=False,
    )
    source_valid = source_exists & (source_weight > 0.0)
    return (
        _slice_to_device(
            global_feats, mb, device, pinned_cache=pinned_cache, slot=slot, name="global_feats"
        ),
        _slice_to_device(
            batch["planet_feats"],
            mb,
            device,
            pinned_cache=pinned_cache,
            slot=slot,
            name="planet_feats",
        ),
        _slice_to_device(
            batch["planet_mask"],
            mb,
            device,
            pinned_cache=pinned_cache,
            slot=slot,
            name="planet_mask",
        ),
        _slice_to_device(
            batch["planet_owned_mask"],
            mb,
            device,
            pinned_cache=pinned_cache,
            slot=slot,
            name="planet_owned_mask",
        ),
        _slice_to_device(
            batch["planet_ids"], mb, device, pinned_cache=pinned_cache, slot=slot, name="planet_ids"
        ),
        _slice_to_device(
            batch["planet_garrison"],
            mb,
            device,
            pinned_cache=pinned_cache,
            slot=slot,
            name="planet_garrison",
        ),
        _slice_to_device(
            batch["fleet_feats"],
            mb,
            device,
            pinned_cache=pinned_cache,
            slot=slot,
            name="fleet_feats",
        ),
        _slice_to_device(
            batch["fleet_mask"], mb, device, pinned_cache=pinned_cache, slot=slot, name="fleet_mask"
        ),
        _slice_to_device(
            fleet_target_planet_idx,
            mb,
            device,
            pinned_cache=pinned_cache,
            slot=slot,
            name="fleet_target_planet_idx",
        ),
        _slice_to_device(
            planet_inbound_feats,
            mb,
            device,
            pinned_cache=pinned_cache,
            slot=slot,
            name="planet_inbound_feats",
        ),
        (row_weight if row_weight.device == device else row_weight.to(device, non_blocking=True)),
        _move_staged_source_field(source_row_local, device),
        _move_staged_source_field(source_col_idx, device),
        _move_staged_source_field(source_valid, device),
        _move_staged_source_field(source_weight, device),
        _move_staged_source_field(launch, device),
        _move_staged_source_field(target_idx, device),
        _move_staged_source_field(fraction, device),
        _move_staged_source_field(old_log_prob, device),
        _slice_to_device(
            policy_advantage, mb, device, pinned_cache=pinned_cache, slot=slot, name="advantage"
        ),
        _slice_to_device(
            return_mtp, mb, device, pinned_cache=pinned_cache, slot=slot, name="return_mtp"
        ),
        _slice_to_device(
            return_mtp_mask,
            mb,
            device,
            pinned_cache=pinned_cache,
            slot=slot,
            name="return_mtp_mask",
        ),
        _move_staged_source_field(target_legal_mask, device),
    )


def _stage_value_minibatch(
    batch: dict[str, torch.Tensor],
    global_feats: torch.Tensor,
    fleet_target_planet_idx: torch.Tensor,
    planet_inbound_feats: torch.Tensor,
    return_mtp: torch.Tensor,
    return_mtp_mask: torch.Tensor,
    mb: torch.Tensor,
    row_weight: torch.Tensor,
    device: torch.device,
    pinned_cache: PinnedSliceCache | None = None,
    slot: int = 0,
) -> tuple[torch.Tensor, ...]:
    """Slice one value-pretrain minibatch and move it to the model device."""
    return (
        _slice_to_device(
            global_feats, mb, device, pinned_cache=pinned_cache, slot=slot, name="global_feats"
        ),
        _slice_to_device(
            batch["planet_feats"],
            mb,
            device,
            pinned_cache=pinned_cache,
            slot=slot,
            name="planet_feats",
        ),
        _slice_to_device(
            batch["planet_mask"],
            mb,
            device,
            pinned_cache=pinned_cache,
            slot=slot,
            name="planet_mask",
        ),
        _slice_to_device(
            batch["planet_owned_mask"],
            mb,
            device,
            pinned_cache=pinned_cache,
            slot=slot,
            name="planet_owned_mask",
        ),
        _slice_to_device(
            batch["planet_ids"], mb, device, pinned_cache=pinned_cache, slot=slot, name="planet_ids"
        ),
        _slice_to_device(
            batch["planet_garrison"],
            mb,
            device,
            pinned_cache=pinned_cache,
            slot=slot,
            name="planet_garrison",
        ),
        _slice_to_device(
            batch["fleet_feats"],
            mb,
            device,
            pinned_cache=pinned_cache,
            slot=slot,
            name="fleet_feats",
        ),
        _slice_to_device(
            batch["fleet_mask"], mb, device, pinned_cache=pinned_cache, slot=slot, name="fleet_mask"
        ),
        _slice_to_device(
            fleet_target_planet_idx,
            mb,
            device,
            pinned_cache=pinned_cache,
            slot=slot,
            name="fleet_target_planet_idx",
        ),
        _slice_to_device(
            planet_inbound_feats,
            mb,
            device,
            pinned_cache=pinned_cache,
            slot=slot,
            name="planet_inbound_feats",
        ),
        (row_weight if row_weight.device == device else row_weight.to(device, non_blocking=True)),
        _slice_to_device(
            return_mtp, mb, device, pinned_cache=pinned_cache, slot=slot, name="return_mtp"
        ),
        _slice_to_device(
            return_mtp_mask,
            mb,
            device,
            pinned_cache=pinned_cache,
            slot=slot,
            name="return_mtp_mask",
        ),
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


def _grad_norm(
    params: list[torch.nn.Parameter],
    device: torch.device,
) -> torch.Tensor:
    norms = [param.grad.detach().norm(2).to(device) for param in params if param.grad is not None]
    if not norms:
        return torch.zeros((), device=device)
    return torch.linalg.vector_norm(torch.stack(norms), ord=2)


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
    critic_shared_raw_norm = _grad_norm(shared, device)
    critic_norm = _clip_grad_norm(critic_params, clip_norm, device)
    critic_shared_norm = _grad_norm(shared, device)
    critic_clip_scale = _grad_clip_scale(critic_norm, clip_norm)
    critic_clip_frac = (critic_clip_scale < 1.0).to(critic_norm.dtype)
    critic_grads = [
        (param, param.grad.detach().clone()) for param in critic_params if param.grad is not None
    ]

    _clear_param_grads(all_params)
    actor_loss.backward()
    actor_shared_raw_norm = _grad_norm(shared, device)
    actor_norm = _clip_grad_norm(actor_params, clip_norm, device)
    actor_shared_norm = _grad_norm(shared, device)
    actor_clip_scale = _grad_clip_scale(actor_norm, clip_norm)
    actor_clip_frac = (actor_clip_scale < 1.0).to(actor_norm.dtype)
    for param, grad in critic_grads:
        param.grad = grad if param.grad is None else param.grad + grad

    shared_norm = _grad_norm(shared, device)
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
        planet_inbound_feats: torch.Tensor,
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
            planet_inbound_feats=planet_inbound_feats,
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
        launch_log_std = None if out_launch_log_std is None else out_launch_log_std.float()
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
        action_lp = _action_log_prob_for_action(action_lp, launch_f)
        fraction_action_lp = _launched_beta_log_prob(
            fraction_alpha,
            fraction_beta,
            fraction.float(),
            launch_f,
        )
        chosen = action_lp + fraction_action_lp

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
        log_ratio = torch.nan_to_num(
            log_ratio,
            nan=0.0,
            neginf=-60.0,
            posinf=60.0,
        )
        log_ratio = log_ratio.clamp(-60.0, 60.0)
        ratio = log_ratio.exp()
        adv_actor = adv_b
        if self.norm_advantage:
            adv_mean = (adv_actor * owned_w).sum() / denom
            adv_var = (((adv_actor - adv_mean).square()) * owned_w).sum() / denom
            adv_actor = (adv_actor - adv_mean) * torch.rsqrt(adv_var + 1e-8)
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

        use_entropy_bonus = (
            self.target_entropy_coef != 0.0 or self.fraction_entropy_coef != 0.0
        )
        if action_logit_softcap is None:
            p_move_current = _threshold_normal_launch_prob(
                launch_logits,
                launch_log_std,
                launch_prob_floor,
            )
        else:
            p_move_current = target_dist_probs.sum(dim=-1)
        entropy_fraction_alpha = fraction_alpha if use_entropy_bonus else fraction_alpha.detach()
        entropy_fraction_beta = fraction_beta if use_entropy_bonus else fraction_beta.detach()
        entropy_owned_w = owned_w if use_entropy_bonus else owned_w.detach()
        entropy_p_move_current = p_move_current if use_entropy_bonus else p_move_current.detach()
        raw_frac_entropy_per_planet = _beta_entropy(
            entropy_fraction_alpha,
            entropy_fraction_beta,
        )
        frac_entropy_per_planet = torch.where(
            torch.isfinite(raw_frac_entropy_per_planet),
            raw_frac_entropy_per_planet,
            torch.zeros_like(raw_frac_entropy_per_planet),
        )
        if action_logit_softcap is None:
            entropy_target_log_probs = (
                target_log_probs if use_entropy_bonus else target_log_probs.detach()
            )
            entropy_target_dist_probs = (
                target_dist_probs if use_entropy_bonus else target_dist_probs.detach()
            )
            entropy_launch_logits = launch_logits if use_entropy_bonus else launch_logits.detach()
            entropy_launch_log_std = (
                None
                if launch_log_std is None
                else launch_log_std if use_entropy_bonus else launch_log_std.detach()
            )
            planet_entropy = _conditional_action_entropy(
                torch.logit(entropy_p_move_current.clamp(1e-7, 1.0 - 1e-7)),
                entropy_target_log_probs,
                frac_entropy_per_planet,
            )
            launch_entropy = (
                _threshold_normal_launch_entropy(
                    entropy_launch_logits,
                    entropy_launch_log_std,
                    launch_prob_floor,
                )
                * entropy_owned_w
            ).sum() / denom
            min_real = torch.finfo(entropy_target_log_probs.dtype).min
            target_entropy_per_planet = -(
                entropy_target_dist_probs * entropy_target_log_probs.clamp_min(min_real)
            ).sum(dim=-1)
            target_entropy = (
                (entropy_p_move_current * target_entropy_per_planet) * entropy_owned_w
            ).sum() / denom
        else:
            entropy_action_log_probs = (
                action_log_probs if use_entropy_bonus else action_log_probs.detach()
            )
            entropy_action_probs = action_probs if use_entropy_bonus else action_probs.detach()
            safe_action_log_probs = torch.where(
                torch.isfinite(entropy_action_log_probs),
                entropy_action_log_probs,
                torch.zeros_like(entropy_action_log_probs),
            )
            action_entropy = -(entropy_action_probs * safe_action_log_probs).sum(dim=-1)
            planet_entropy = action_entropy + entropy_p_move_current * frac_entropy_per_planet
            target_entropy = (action_entropy * entropy_owned_w).sum() / denom
            launch_entropy = target_entropy * 0.0
        entropy = (planet_entropy * entropy_owned_w).sum() / denom
        fraction_entropy = (
            (entropy_p_move_current * frac_entropy_per_planet) * entropy_owned_w
        ).sum() / denom
        move_prob = (p_move_current * owned_w).sum() / denom
        legal_target_count = target_legal_mask.to(dtype=owned_f.dtype).sum(dim=-1)
        legal_target_count_mean = (legal_target_count * owned_w).sum() / denom
        uniform_move_prior = (
            (legal_target_count > 0.0).to(dtype=owned_f.dtype) * 0.5 * owned_w
        ).sum() / denom
        launch_mean = (launch_logits * owned_w).sum() / denom
        if action_logit_softcap is not None:
            launch_log_std_mean = launch_logits.sum() * 0.0
            action_logit_softcap_metric = launch_logits.new_tensor(float(action_logit_softcap))
            has_finite_target = torch.isfinite(target_logits).any(dim=-1)
            target_best = target_logits.amax(dim=-1)
            target_best = torch.where(has_finite_target, target_best, launch_logits)
            launch_score = torch.where(
                has_finite_target,
                launch_logits - target_best,
                torch.zeros_like(launch_logits),
            )
            launch_score_mean = (launch_score * owned_w).sum() / denom
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
        fraction_skew_abs_mean = ((fraction_alpha - fraction_beta).abs() * owned_w).sum() / denom
        deterministic_fraction = _deterministic_beta_fraction(fraction_alpha, fraction_beta)
        deterministic_fraction_mean = (deterministic_fraction * owned_w).sum() / denom
        entropy_bonus = launch_entropy.sum() * 0.0
        if self.target_entropy_coef != 0.0:
            entropy_bonus = entropy_bonus + self.target_entropy_coef * (
                launch_entropy + target_entropy
            )
        if self.fraction_entropy_coef != 0.0:
            entropy_bonus = entropy_bonus + self.fraction_entropy_coef * fraction_entropy

        actor_loss = policy_loss - entropy_bonus
        critic_loss = self.value_coef * value_loss

        # CleanRL-style PPO KL diagnostic: collapse the factorized per-planet
        # log-probs into one joint action log-prob per rollout row, then apply
        # Joschu's k3 approximation `E[(r - 1) - log r]`. The policy surrogate
        # above remains per-source-planet; this is only the reported KL scale.
        per_planet_kl = (((ratio - 1.0) - log_ratio) * owned_w).sum() / denom
        log_ratio_abs_mean = (log_ratio.abs() * owned_w).sum() / denom
        log_ratio_abs_max = _weighted_max(log_ratio.abs(), owned_w)
        clipped_low = ratio < (1.0 - self.clip_coef)
        clipped_high = ratio > (1.0 + self.clip_coef_high)
        ratio_clip_frac = ((clipped_low | clipped_high).to(owned_f.dtype) * owned_w).sum() / denom
        ratio_clip_frac_high = (clipped_high.to(owned_f.dtype) * owned_w).sum() / denom
        executed_launch_frac = (launch_f * owned_w).sum() / denom

        row_log_ratio = (log_ratio * owned_f).sum(dim=-1)
        row_log_ratio = torch.where(row_w > 0.0, row_log_ratio, row_log_ratio * 0.0)
        row_ratio = row_log_ratio.clamp(-60.0, 60.0).exp()
        row_denom = row_w.sum().clamp_min(1.0)
        row_launch_count = (launch_f * owned_w).sum(dim=-1)
        turn_no_action_frac = ((row_launch_count <= 0.0).to(row_w.dtype) * row_w).sum() / row_denom
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


class _PPOSourceMinibatchKernel(torch.nn.Module):
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
        planet_inbound_feats: torch.Tensor,
        row_weight: torch.Tensor,
        source_row_local: torch.Tensor,
        source_col_idx: torch.Tensor,
        source_valid: torch.Tensor,
        source_weight: torch.Tensor,
        launch: torch.Tensor,
        target_idx: torch.Tensor,
        fraction: torch.Tensor,
        old_log_prob: torch.Tensor,
        advantage: torch.Tensor,
        ret_mtp: torch.Tensor,
        ret_mtp_mask: torch.Tensor,
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
            planet_inbound_feats=planet_inbound_feats,
        )
        with torch.autocast(
            device_type="cuda", dtype=torch.bfloat16, enabled=self.autocast_enabled
        ):
            out = self.model(
                feats,
                include_value=self.include_value,
                actor_source_rows=source_row_local,
                actor_source_cols=source_col_idx,
                actor_source_valid=source_valid,
                target_planets=int(target_legal_mask.shape[1]),
            )

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
        launch_log_std = None if out_launch_log_std is None else out_launch_log_std.float()
        launch_prob_floor = float(getattr(out, "launch_prob_floor", 0.0))
        target_logits = out.target_logits.float().masked_fill(~target_legal_mask, float("-inf"))
        if out.fraction_alpha is None or out.fraction_beta is None:
            raise ValueError("PPO OrbitPolicy output must include Beta fraction params")
        fraction_alpha = out.fraction_alpha.float()
        fraction_beta = out.fraction_beta.float()
        value_logits = out.value_logits.float()

        source_w = source_weight.float()
        source_valid_f = source_valid.to(dtype=source_w.dtype)
        denom = source_w.sum().clamp_min(1.0)
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
        action_lp = _action_log_prob_for_action(action_lp, launch_f)
        fraction_action_lp = _launched_beta_log_prob(
            fraction_alpha,
            fraction_beta,
            fraction.float(),
            launch_f,
        )
        chosen = action_lp + fraction_action_lp

        adv_actor = advantage.float().index_select(0, source_row_local)
        log_ratio = torch.where(
            source_valid,
            chosen - old_log_prob.float(),
            torch.zeros_like(chosen),
        )
        log_ratio = torch.nan_to_num(
            log_ratio,
            nan=0.0,
            neginf=-60.0,
            posinf=60.0,
        ).clamp(-60.0, 60.0)
        ratio = log_ratio.exp()
        if self.norm_advantage:
            adv_mean = (adv_actor * source_w).sum() / denom
            adv_var = (((adv_actor - adv_mean).square()) * source_w).sum() / denom
            adv_actor = (adv_actor - adv_mean) * torch.rsqrt(adv_var + 1e-8)
        ratio_clamped = ratio.clamp(1.0 - self.clip_coef, 1.0 + self.clip_coef_high)
        pg_unclipped = -adv_actor * ratio
        pg_clipped = -adv_actor * ratio_clamped
        policy_loss = _weighted_mean(torch.maximum(pg_unclipped, pg_clipped), source_w)

        row_w = row_weight.to(device=value_logits.device, dtype=value_logits.dtype)
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

        use_entropy_bonus = (
            self.target_entropy_coef != 0.0 or self.fraction_entropy_coef != 0.0
        )
        if action_logit_softcap is None:
            p_move_current = _threshold_normal_launch_prob(
                launch_logits,
                launch_log_std,
                launch_prob_floor,
            )
        else:
            p_move_current = target_dist_probs.sum(dim=-1)
        entropy_fraction_alpha = fraction_alpha if use_entropy_bonus else fraction_alpha.detach()
        entropy_fraction_beta = fraction_beta if use_entropy_bonus else fraction_beta.detach()
        entropy_source_w = source_w if use_entropy_bonus else source_w.detach()
        entropy_p_move_current = p_move_current if use_entropy_bonus else p_move_current.detach()
        raw_frac_entropy_per_planet = _beta_entropy(
            entropy_fraction_alpha,
            entropy_fraction_beta,
        )
        frac_entropy_per_planet = torch.where(
            torch.isfinite(raw_frac_entropy_per_planet),
            raw_frac_entropy_per_planet,
            torch.zeros_like(raw_frac_entropy_per_planet),
        )
        if action_logit_softcap is None:
            entropy_target_log_probs = (
                target_log_probs if use_entropy_bonus else target_log_probs.detach()
            )
            entropy_target_dist_probs = (
                target_dist_probs if use_entropy_bonus else target_dist_probs.detach()
            )
            entropy_launch_logits = launch_logits if use_entropy_bonus else launch_logits.detach()
            entropy_launch_log_std = (
                None
                if launch_log_std is None
                else launch_log_std if use_entropy_bonus else launch_log_std.detach()
            )
            planet_entropy = _conditional_action_entropy(
                torch.logit(entropy_p_move_current.clamp(1e-7, 1.0 - 1e-7)),
                entropy_target_log_probs,
                frac_entropy_per_planet,
            )
            launch_entropy = (
                _threshold_normal_launch_entropy(
                    entropy_launch_logits,
                    entropy_launch_log_std,
                    launch_prob_floor,
                )
                * entropy_source_w
            ).sum() / denom
            min_real = torch.finfo(entropy_target_log_probs.dtype).min
            target_entropy_per_planet = -(
                entropy_target_dist_probs * entropy_target_log_probs.clamp_min(min_real)
            ).sum(dim=-1)
            target_entropy = (
                (entropy_p_move_current * target_entropy_per_planet) * entropy_source_w
            ).sum() / denom
        else:
            entropy_action_log_probs = (
                action_log_probs if use_entropy_bonus else action_log_probs.detach()
            )
            entropy_action_probs = action_probs if use_entropy_bonus else action_probs.detach()
            safe_action_log_probs = torch.where(
                torch.isfinite(entropy_action_log_probs),
                entropy_action_log_probs,
                torch.zeros_like(entropy_action_log_probs),
            )
            action_entropy = -(entropy_action_probs * safe_action_log_probs).sum(dim=-1)
            planet_entropy = action_entropy + entropy_p_move_current * frac_entropy_per_planet
            target_entropy = (action_entropy * entropy_source_w).sum() / denom
            launch_entropy = target_entropy * 0.0
        entropy = (planet_entropy * entropy_source_w).sum() / denom
        fraction_entropy = (
            (entropy_p_move_current * frac_entropy_per_planet) * entropy_source_w
        ).sum() / denom
        move_prob = (p_move_current * source_w).sum() / denom
        legal_target_count = target_legal_mask.to(dtype=source_w.dtype).sum(dim=-1)
        legal_target_count_mean = (legal_target_count * source_w).sum() / denom
        uniform_move_prior = (
            (legal_target_count > 0.0).to(dtype=source_w.dtype) * 0.5 * source_w
        ).sum() / denom
        launch_mean = (launch_logits * source_w).sum() / denom
        if action_logit_softcap is not None:
            launch_log_std_mean = launch_logits.sum() * 0.0
            action_logit_softcap_metric = launch_logits.new_tensor(float(action_logit_softcap))
            has_finite_target = torch.isfinite(target_logits).any(dim=-1)
            target_best = target_logits.amax(dim=-1)
            target_best = torch.where(has_finite_target, target_best, launch_logits)
            launch_score = torch.where(
                has_finite_target,
                launch_logits - target_best,
                torch.zeros_like(launch_logits),
            )
            launch_score_mean = (launch_score * source_w).sum() / denom
        elif launch_log_std is None:
            launch_log_std_mean = launch_logits.sum() * 0.0
            action_logit_softcap_metric = launch_logits.sum() * 0.0
            launch_score_mean = launch_logits.sum() * 0.0
        else:
            launch_log_std_f = launch_log_std.float()
            launch_log_std_mean = (launch_log_std_f * source_w).sum() / denom
            action_logit_softcap_metric = launch_logits.sum() * 0.0
            launch_score = launch_logits * torch.exp(-launch_log_std_f)
            launch_score_mean = (launch_score * source_w).sum() / denom
        target_confidence = (target_dist_probs.amax(dim=-1) * source_w).sum() / denom
        fraction_alpha_mean = (fraction_alpha * source_w).sum() / denom
        fraction_beta_mean = (fraction_beta * source_w).sum() / denom
        concentration = fraction_alpha + fraction_beta
        fraction_concentration_mean = (concentration * source_w).sum() / denom
        fraction_concentration_max = _weighted_max(concentration, source_w)
        fraction_skew_abs_mean = ((fraction_alpha - fraction_beta).abs() * source_w).sum() / denom
        deterministic_fraction = _deterministic_beta_fraction(fraction_alpha, fraction_beta)
        deterministic_fraction_mean = (deterministic_fraction * source_w).sum() / denom
        entropy_bonus = launch_entropy.sum() * 0.0
        if self.target_entropy_coef != 0.0:
            entropy_bonus = entropy_bonus + self.target_entropy_coef * (
                launch_entropy + target_entropy
            )
        if self.fraction_entropy_coef != 0.0:
            entropy_bonus = entropy_bonus + self.fraction_entropy_coef * fraction_entropy

        actor_loss = policy_loss - entropy_bonus
        critic_loss = self.value_coef * value_loss

        per_planet_kl = (((ratio - 1.0) - log_ratio) * source_w).sum() / denom
        log_ratio_abs_mean = (log_ratio.abs() * source_w).sum() / denom
        log_ratio_abs_max = _weighted_max(log_ratio.abs(), source_w)
        clipped_low = ratio < (1.0 - self.clip_coef)
        clipped_high = ratio > (1.0 + self.clip_coef_high)
        ratio_clip_frac = ((clipped_low | clipped_high).to(source_w.dtype) * source_w).sum() / denom
        ratio_clip_frac_high = (clipped_high.to(source_w.dtype) * source_w).sum() / denom
        executed_launch_frac = (launch_f * source_w).sum() / denom

        rows = int(row_weight.shape[0])
        row_log_ratio = torch.zeros(rows, dtype=log_ratio.dtype, device=log_ratio.device)
        row_log_ratio.scatter_add_(0, source_row_local, log_ratio * source_valid_f)
        row_log_ratio = torch.where(row_w > 0.0, row_log_ratio, row_log_ratio * 0.0)
        row_ratio = row_log_ratio.clamp(-60.0, 60.0).exp()
        row_denom = row_w.sum().clamp_min(1.0)
        row_launch_count = torch.zeros(rows, dtype=source_w.dtype, device=source_w.device)
        row_launch_count.scatter_add_(0, source_row_local, launch_f * source_w)
        turn_no_action_frac = ((row_launch_count <= 0.0).to(row_w.dtype) * row_w).sum() / row_denom
        kl = (((row_ratio - 1.0) - row_log_ratio) * row_w).sum() / row_denom
        row_log_ratio_abs_mean = (row_log_ratio.abs() * row_w).sum() / row_denom
        owned_planets_mean = source_w.sum() / row_denom
        pos_count = (source_w * (adv_actor >= 0.0).to(source_w.dtype)).sum()
        total_owned = source_w.sum().clamp_min(1.0)
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
        planet_inbound_feats: torch.Tensor,
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
            planet_inbound_feats=planet_inbound_feats,
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


def _old_log_prob_from_output(
    out: Any,
    launch: torch.Tensor,
    target_idx: torch.Tensor,
    fraction: torch.Tensor,
    target_legal_mask: torch.Tensor,
) -> torch.Tensor:
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
    launch_log_std = None if out_launch_log_std is None else out_launch_log_std.float()
    launch_prob_floor = float(getattr(out, "launch_prob_floor", 0.0))
    target_logits = out.target_logits.float().masked_fill(
        ~target_legal_mask,
        float("-inf"),
    )
    if out.fraction_alpha is None or out.fraction_beta is None:
        raise ValueError("PPO OrbitPolicy output must include Beta fraction params")
    fraction_alpha = out.fraction_alpha.float()
    fraction_beta = out.fraction_beta.float()
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
        target_lp = target_log_probs.gather(
            -1,
            target.unsqueeze(-1),
        ).squeeze(-1)
        action_lp = launch_lp + launch_f * target_lp
    else:
        action_log_probs = _categorical_action_log_probs(
            launch_logits,
            target_logits,
            action_logit_softcap,
        )
        action_idx = torch.where(
            launch_f > 0.5,
            target + 1,
            torch.zeros_like(target),
        )
        action_lp = action_log_probs.gather(-1, action_idx.unsqueeze(-1)).squeeze(-1)
    action_lp = _action_log_prob_for_action(action_lp, launch_f)
    fraction_action_lp = _launched_beta_log_prob(
        fraction_alpha,
        fraction_beta,
        fraction.float(),
        launch_f,
    )
    return action_lp + fraction_action_lp


class _OldLogProbKernel(torch.nn.Module):
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
        planet_inbound_feats: torch.Tensor,
        launch: torch.Tensor,
        target_idx: torch.Tensor,
        fraction: torch.Tensor,
        target_legal_mask: torch.Tensor,
    ) -> torch.Tensor:
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
            planet_inbound_feats=planet_inbound_feats,
        )
        with torch.autocast(
            device_type="cuda", dtype=torch.bfloat16, enabled=self.autocast_enabled
        ):
            if self._model_accepts_head_flags:
                out = self.model(feats, include_value=False)
            else:
                out = self.model(feats)

        return _old_log_prob_from_output(
            out,
            launch,
            target_idx,
            fraction,
            target_legal_mask,
        )


class _OldLogProbValueKernel(torch.nn.Module):
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
        planet_inbound_feats: torch.Tensor,
        launch: torch.Tensor,
        target_idx: torch.Tensor,
        fraction: torch.Tensor,
        target_legal_mask: torch.Tensor,
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
            planet_inbound_feats=planet_inbound_feats,
        )
        with torch.autocast(
            device_type="cuda", dtype=torch.bfloat16, enabled=self.autocast_enabled
        ):
            out = self.model(feats)

        old_log_prob = _old_log_prob_from_output(
            out,
            launch,
            target_idx,
            fraction,
            target_legal_mask,
        )
        return old_log_prob, out.value.float()


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
    shape_key: tuple[int, int, int, int, int],
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
        shape_key,
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


def _get_source_ppo_kernel(
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
    shape_key: tuple[int, ...],
) -> torch.nn.Module:
    device = _module_device(model)
    mode = compile_mode if device.type == "cuda" else None
    key = (
        "ppo_source",
        mode,
        float(value_coef),
        float(target_entropy_coef),
        float(fraction_entropy_coef),
        bool(norm_advantage),
        float(clip_coef),
        float(clip_coef_high),
        bool(include_value),
        shape_key,
    )
    cache = _kernel_cache(model)
    cached = cache.get(key)
    if cached is not None:
        return cached
    kernel = _PPOSourceMinibatchKernel(
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
    shape_key: tuple[int, int, int, int],
) -> torch.nn.Module:
    device = _module_device(model)
    mode = compile_mode if device.type == "cuda" else None
    key = ("value_only", mode, shape_key)
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


def _get_old_log_prob_kernel(
    model: torch.nn.Module,
    *,
    compile_mode: str | None,
    shape_key: tuple[int, int, int, int, int, int],
) -> torch.nn.Module:
    device = _module_device(model)
    mode = compile_mode if device.type == "cuda" else None
    key = ("old_log_prob", mode, shape_key)
    cache = _kernel_cache(model)
    cached = cache.get(key)
    if cached is not None:
        return cached
    kernel = _OldLogProbKernel(
        model,
        autocast_enabled=device.type == "cuda",
    )
    kernel = _compile_kernel(kernel, device=device, compile_mode=mode)
    cache[key] = kernel
    return kernel


def _get_old_log_prob_value_kernel(
    model: torch.nn.Module,
    *,
    compile_mode: str | None,
    shape_key: tuple[int, int, int, int, int, int],
) -> torch.nn.Module:
    device = _module_device(model)
    mode = compile_mode if device.type == "cuda" else None
    key = ("old_log_prob_value", mode, shape_key)
    cache = _kernel_cache(model)
    cached = cache.get(key)
    if cached is not None:
        return cached
    kernel = _OldLogProbValueKernel(
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
    """Plain Monte-Carlo discounted return Σ γ^k r_{t+k}."""
    horizon = len(rewards)
    out = np.zeros(horizon, dtype=np.float32)
    running = 0.0
    for t in reversed(range(horizon)):
        running = rewards[t] + gamma * running
        out[t] = running
    return out


def _stage_old_log_prob_minibatch(
    batch: dict[str, torch.Tensor],
    global_feats: torch.Tensor,
    fleet_target_planet_idx: torch.Tensor,
    planet_inbound_feats: torch.Tensor,
    mb: torch.Tensor,
    device: torch.device,
    pinned_cache: PinnedSliceCache | None = None,
    slot: int = 0,
) -> tuple[torch.Tensor, ...]:
    return (
        _slice_to_device(
            global_feats, mb, device, pinned_cache=pinned_cache, slot=slot, name="global_feats"
        ),
        _slice_to_device(
            batch["planet_feats"],
            mb,
            device,
            pinned_cache=pinned_cache,
            slot=slot,
            name="planet_feats",
        ),
        _slice_to_device(
            batch["planet_mask"],
            mb,
            device,
            pinned_cache=pinned_cache,
            slot=slot,
            name="planet_mask",
        ),
        _slice_to_device(
            batch["planet_owned_mask"],
            mb,
            device,
            pinned_cache=pinned_cache,
            slot=slot,
            name="planet_owned_mask",
        ),
        _slice_to_device(
            batch["planet_ids"], mb, device, pinned_cache=pinned_cache, slot=slot, name="planet_ids"
        ),
        _slice_to_device(
            batch["planet_garrison"],
            mb,
            device,
            pinned_cache=pinned_cache,
            slot=slot,
            name="planet_garrison",
        ),
        _slice_to_device(
            batch["fleet_feats"],
            mb,
            device,
            pinned_cache=pinned_cache,
            slot=slot,
            name="fleet_feats",
        ),
        _slice_to_device(
            batch["fleet_mask"], mb, device, pinned_cache=pinned_cache, slot=slot, name="fleet_mask"
        ),
        _slice_to_device(
            fleet_target_planet_idx,
            mb,
            device,
            pinned_cache=pinned_cache,
            slot=slot,
            name="fleet_target_planet_idx",
        ),
        _slice_to_device(
            planet_inbound_feats,
            mb,
            device,
            pinned_cache=pinned_cache,
            slot=slot,
            name="planet_inbound_feats",
        ),
        _slice_to_device(
            batch["launch"], mb, device, pinned_cache=pinned_cache, slot=slot, name="launch"
        ),
        _slice_to_device(
            batch["target_idx"], mb, device, pinned_cache=pinned_cache, slot=slot, name="target_idx"
        ),
        _slice_to_device(
            batch["fraction"], mb, device, pinned_cache=pinned_cache, slot=slot, name="fraction"
        ),
        _stage_target_legal_mask(
            batch,
            mb,
            device,
            pinned_cache=pinned_cache,
            slot=slot,
        ),
    )


def compute_old_log_probs(
    model: OrbitPolicy,
    batch: dict[str, torch.Tensor],
    *,
    minibatch_size: int,
    minibatch_count: int | None = None,
    compile_mode: str | None = None,
    output_device: torch.device | str | None = None,
) -> torch.Tensor:
    """Compute frozen behavior log-probs for a stacked PPO rollout batch."""
    n = int(batch["planet_feats"].shape[0])
    if n <= 0:
        return batch["launch"].new_empty(batch["launch"].shape).float()
    batch_device = batch["planet_feats"].device
    batch_view = _batch_encoded_view(batch)
    global_feats = _global_feats_or_empty(batch)
    fleet_target_planet_idx = fleet_target_planet_idx_or_empty(batch_view)
    planet_inbound_feats = planet_inbound_feats_or_empty(batch_view)
    device = _module_device(model)
    out_device = torch.device(output_device) if output_device is not None else batch_device
    if minibatch_count is not None:
        static_minibatch_rows = math.ceil(n / max(1, min(int(minibatch_count), n)))
    else:
        static_minibatch_rows = max(1, int(minibatch_size))
    was_training = model.training
    model.eval()
    try:
        kernel = _get_old_log_prob_kernel(
            model,
            compile_mode=compile_mode,
            shape_key=(
                static_minibatch_rows,
                int(batch["planet_feats"].shape[1]),
                int(batch["fleet_feats"].shape[1]),
                int(global_feats.shape[1]),
                int(planet_inbound_feats.shape[-2]),
                int(planet_inbound_feats.shape[-1]),
            ),
        )
        chunks: list[torch.Tensor] = []
        pinned_cache: PinnedSliceCache | None = {} if device.type == "cuda" else None
        with torch.no_grad():
            for mb, _row_weight, real in _fixed_order_minibatches(
                n,
                minibatch_size,
                batch_device,
                minibatch_count=minibatch_count,
                weight_device=device,
            ):
                staged = _stage_old_log_prob_minibatch(
                    batch,
                    global_feats,
                    fleet_target_planet_idx,
                    planet_inbound_feats,
                    mb,
                    device,
                    pinned_cache=pinned_cache,
                )
                if compile_mode is not None:
                    _mark_cuda_graph_step(device)
                old_lp = kernel(*staged)
                # `reduce-overhead` may return views into CUDA-graph replay
                # buffers. Clone on the producing device before any transfer so
                # the next replay cannot overwrite an in-flight CPU copy.
                retained = old_lp[:real].detach().clone()
                if retained.device != out_device:
                    retained = retained.to(
                        out_device,
                        non_blocking=out_device.type != "cpu",
                    )
                chunks.append(retained)
        return torch.cat(chunks, dim=0).float()
    finally:
        model.train(was_training)


def compute_old_log_probs_and_values(
    model: OrbitPolicy,
    batch: dict[str, torch.Tensor],
    *,
    minibatch_size: int,
    minibatch_count: int | None = None,
    compile_mode: str | None = None,
    output_device: torch.device | str | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute behavior log-probs and critic values for stacked learner rows."""
    n = int(batch["planet_feats"].shape[0])
    if n <= 0:
        return (
            batch["launch"].new_empty(batch["launch"].shape).float(),
            batch["launch"].new_empty(0).float(),
        )
    batch_device = batch["planet_feats"].device
    batch_view = _batch_encoded_view(batch)
    global_feats = _global_feats_or_empty(batch)
    fleet_target_planet_idx = fleet_target_planet_idx_or_empty(batch_view)
    planet_inbound_feats = planet_inbound_feats_or_empty(batch_view)
    device = _module_device(model)
    out_device = torch.device(output_device) if output_device is not None else batch_device
    if minibatch_count is not None:
        static_minibatch_rows = math.ceil(n / max(1, min(int(minibatch_count), n)))
    else:
        static_minibatch_rows = max(1, int(minibatch_size))
    was_training = model.training
    model.eval()
    try:
        kernel = _get_old_log_prob_value_kernel(
            model,
            compile_mode=compile_mode,
            shape_key=(
                static_minibatch_rows,
                int(batch["planet_feats"].shape[1]),
                int(batch["fleet_feats"].shape[1]),
                int(global_feats.shape[1]),
                int(planet_inbound_feats.shape[-2]),
                int(planet_inbound_feats.shape[-1]),
            ),
        )
        logprob_chunks: list[torch.Tensor] = []
        value_chunks: list[torch.Tensor] = []
        pinned_cache: PinnedSliceCache | None = {} if device.type == "cuda" else None
        with torch.no_grad():
            for mb, _row_weight, real in _fixed_order_minibatches(
                n,
                minibatch_size,
                batch_device,
                minibatch_count=minibatch_count,
                weight_device=device,
            ):
                staged = _stage_old_log_prob_minibatch(
                    batch,
                    global_feats,
                    fleet_target_planet_idx,
                    planet_inbound_feats,
                    mb,
                    device,
                    pinned_cache=pinned_cache,
                )
                if compile_mode is not None:
                    _mark_cuda_graph_step(device)
                old_lp, value = kernel(*staged)
                retained_lp = old_lp[:real].detach().clone()
                retained_value = value[:real].detach().clone()
                if retained_lp.device != out_device:
                    retained_lp = retained_lp.to(
                        out_device,
                        non_blocking=out_device.type != "cpu",
                    )
                if retained_value.device != out_device:
                    retained_value = retained_value.to(
                        out_device,
                        non_blocking=out_device.type != "cpu",
                    )
                logprob_chunks.append(retained_lp)
                value_chunks.append(retained_value)
        return (
            torch.cat(logprob_chunks, dim=0).float(),
            torch.cat(value_chunks, dim=0).float(),
        )
    finally:
        model.train(was_training)


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

    If `minibatch_count` is set, each epoch runs up to that many equal-shape
    minibatches, capped by `N` to avoid zero-sample optimizer steps. Otherwise
    each epoch runs `ceil(N / minibatch_size)` batches.

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
    if "old_log_prob_computed" not in batch or not bool(batch["old_log_prob_computed"]):
        raise ValueError(
            "ppo_update requires real old_log_prob values; recompute deferred "
            "old log-probs with compute_old_log_probs before calling ppo_update"
        )
    n = batch["planet_feats"].shape[0]
    batch_device = batch["planet_feats"].device
    batch_view = _batch_encoded_view(batch)
    global_feats = _global_feats_or_empty(batch)
    fleet_target_planet_idx = fleet_target_planet_idx_or_empty(batch_view)
    planet_inbound_feats = planet_inbound_feats_or_empty(batch_view)
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

    if minibatch_count is not None:
        static_minibatch_rows = math.ceil(n / max(1, min(int(minibatch_count), n)))
    else:
        static_minibatch_rows = max(1, int(minibatch_size))
    use_source_actor = (
        isinstance(model, OrbitPolicy)
        and "actor_source_row_idx" in batch
        and "actor_target_legal_mask" in batch
    )
    source_capacity = 0
    if use_source_actor:
        actor_offsets = batch["actor_source_row_offsets"].long()
        max_sources_per_row = int((actor_offsets[1:] - actor_offsets[:-1]).max().item())
        source_capacity = _source_capacity_for_minibatch(
            static_minibatch_rows,
            int(batch["planet_feats"].shape[1]),
            max_sources_per_row,
        )
        kernel = _get_source_ppo_kernel(
            model,
            value_coef=value_coef,
            target_entropy_coef=target_entropy_coef,
            fraction_entropy_coef=fraction_entropy_coef,
            norm_advantage=norm_advantage,
            clip_coef=clip_coef,
            clip_coef_high=clip_coef_high,
            compile_mode=compile_mode,
            include_value=True,
            shape_key=(
                static_minibatch_rows,
                source_capacity,
                int(batch["planet_feats"].shape[1]),
                int(batch["fleet_feats"].shape[1]),
                int(global_feats.shape[1]),
                int(return_mtp.shape[1]),
                int(planet_inbound_feats.shape[-2]),
                int(planet_inbound_feats.shape[-1]),
                int(batch["actor_target_legal_mask"].shape[1]),
            ),
        )
    else:
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
            shape_key=(
                static_minibatch_rows,
                int(batch["planet_feats"].shape[1]),
                int(batch["fleet_feats"].shape[1]),
                int(global_feats.shape[1]),
                int(return_mtp.shape[1]),
                int(planet_inbound_feats.shape[-2]),
                int(planet_inbound_feats.shape[-1]),
            ),
        )
    epochs_run = 0
    pinned_cache: PinnedSliceCache | None = {} if device.type == "cuda" else None
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
            minibatches = _fixed_minibatches(n, minibatch_size, batch_device, weight_device=device)

        def stage(
            mb: torch.Tensor,
            row_weight: torch.Tensor,
            slot: int,
        ) -> tuple[torch.Tensor, ...]:
            if use_source_actor:
                return _stage_source_ppo_minibatch(
                    batch,
                    global_feats,
                    fleet_target_planet_idx,
                    planet_inbound_feats,
                    policy_advantage,
                    return_mtp,
                    return_mtp_mask,
                    mb,
                    row_weight,
                    device,
                    source_capacity=source_capacity,
                    pinned_cache=pinned_cache,
                    slot=slot,
                )
            return _stage_ppo_minibatch(
                batch,
                global_feats,
                fleet_target_planet_idx,
                planet_inbound_feats,
                policy_advantage,
                return_mtp,
                return_mtp_mask,
                mb,
                row_weight,
                device,
                pinned_cache=pinned_cache,
                slot=slot,
            )

        for staged in _prefetch_staged_minibatches(stage, minibatches, device):
            row_weight = staged[10]
            if compile_mode is not None:
                _mark_cuda_graph_step(device)
            actor_loss, critic_loss, metrics = kernel(*staged)
            _raise_if_nonfinite_tensor(f"ppo_step_{n_steps}:metrics", metrics)
            _raise_if_nonfinite_tensor(f"ppo_step_{n_steps}:actor_loss", actor_loss)
            _raise_if_nonfinite_tensor(f"ppo_step_{n_steps}:critic_loss", critic_loss)
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
            ) = _backward_actor_critic_with_group_clips(
                model,
                actor_loss * loss_scale,
                critic_loss * loss_scale,
                grad_clip,
            )
            _raise_if_nonfinite_model_tensors(
                model,
                what=f"ppo_step_{n_steps}:grad",
                gradients=True,
            )
            optimizer.step()
            _raise_if_nonfinite_model_tensors(
                model,
                what=f"ppo_step_{n_steps}:param_after_step",
                gradients=False,
            )
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
            _raise_if_nonfinite_model_tensors(
                model,
                what=f"ppo_step_{n_steps}:param_after_normalize",
                gradients=False,
            )
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
    mean_logs_t = torch.zeros(30, device=device) if metric_sum is None else metric_sum / n_steps
    # Match CleanRL's PPO KL logging: `approx_kl` is the latest minibatch's
    # estimate after the PPO epoch loop, not an epoch mean. The surrounding
    # diagnostics stay averaged to preserve their lower-noise TensorBoard
    # behavior.
    kl_logs_t = torch.zeros(30, device=device) if last_metrics is None else last_metrics
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
    batch_view = _batch_encoded_view(batch)
    global_feats = _global_feats_or_empty(batch)
    fleet_target_planet_idx = fleet_target_planet_idx_or_empty(batch_view)
    planet_inbound_feats = planet_inbound_feats_or_empty(batch_view)
    device = _module_device(model)
    total: torch.Tensor | None = None
    n_steps = 0
    return_mtp = batch.get("return_mtp", batch["return"].unsqueeze(-1))
    return_mtp_mask = batch.get(
        "return_mtp_mask",
        torch.ones_like(return_mtp, dtype=torch.bool),
    )
    static_minibatch_rows = max(1, int(minibatch_size))
    kernel = _get_value_only_kernel(
        model,
        compile_mode=compile_mode,
        shape_key=(
            static_minibatch_rows,
            int(batch["planet_feats"].shape[1]),
            int(batch["fleet_feats"].shape[1]),
            int(global_feats.shape[1]),
            int(planet_inbound_feats.shape[-2]),
            int(planet_inbound_feats.shape[-1]),
        ),
    )
    pinned_cache: PinnedSliceCache | None = {} if device.type == "cuda" else None
    for _ in range(epochs):
        minibatches = _fixed_minibatches(
            n,
            minibatch_size,
            batch_device,
            weight_device=device,
        )

        def stage(
            mb: torch.Tensor,
            row_weight: torch.Tensor,
            slot: int,
        ) -> tuple[torch.Tensor, ...]:
            return _stage_value_minibatch(
                batch,
                global_feats,
                fleet_target_planet_idx,
                planet_inbound_feats,
                return_mtp,
                return_mtp_mask,
                mb,
                row_weight,
                device,
                pinned_cache=pinned_cache,
                slot=slot,
            )

        for staged in _prefetch_staged_minibatches(stage, minibatches, device):
            row_weight = staged[10]
            if compile_mode is not None:
                _mark_cuda_graph_step(device)
            value_loss, metric = kernel(*staged)

            optimizer.zero_grad(set_to_none=True)
            (value_loss * _minibatch_loss_scale(row_weight)).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            normalize_matrices(model)

            if total is None:
                total = metric.new_zeros(())
            total += metric
            n_steps += 1

    if total is None:
        return 0.0
    return float((total / max(1, n_steps)).detach().cpu())
