"""SPO-asym policy update with a distributional critic.

The policy loss follows CleanRL's SPO asymmetric variant:

    J = E[r*A - |A|*(r-1)^2/(2*eps)]

where `eps` is larger when ratio drift agrees with the advantage sign and
smaller otherwise. The critic emits `value_logits` over a fixed bin support
and trains with cross-entropy against HL-Gauss-encoded returns. No scalar value
clipping is used because the distributional CE gradients are already bounded
per bin.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
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


SQUASH_EPS: float = 1e-6


def _squashed_normal_log_prob(
    mean: torch.Tensor,
    log_std: torch.Tensor,
    fraction: torch.Tensor,
) -> torch.Tensor:
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
    return log_std.float() + 0.5 * (1.0 + math.log(2.0 * math.pi))


def _module_device(model: torch.nn.Module) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


_ACTOR_CLIP_PATTERNS: tuple[str, ...] = (
    "target_query",
    "target_key",
    "target_q_gain",
    "launch_head",
    "fraction_head",
    "fraction_log_std",
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


def _clone_grads(
    params: list[torch.nn.Parameter],
) -> list[tuple[torch.nn.Parameter, torch.Tensor]]:
    return [
        (param, param.grad.detach().clone())
        for param in params
        if param.grad is not None
    ]


def _clear_grads(params: Iterable[torch.nn.Parameter]) -> None:
    for param in params:
        param.grad = None


def _add_grads(saved: list[tuple[torch.nn.Parameter, torch.Tensor]]) -> None:
    for param, grad in saved:
        if param.grad is None:
            param.grad = grad
        else:
            param.grad.add_(grad)


def _backward_actor_critic_with_separate_clips(
    model: torch.nn.Module,
    actor_loss: torch.Tensor,
    critic_loss: torch.Tensor,
    max_norm: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Backprop actor and critic losses with independent clipping.

    The shared encoder receives both policy and value gradients, but the two
    norms are clipped before they are added together. This prevents value CE
    from setting the policy step size through a single combined norm.
    """
    actor, critic, shared = _grad_clip_groups(model)
    device = _module_device(model)
    if max_norm <= 0.0:
        (actor_loss + critic_loss).backward()
        zero = torch.zeros((), device=device)
        return zero, zero

    actor_params = actor + shared
    critic_params = critic + shared

    actor_loss.backward(retain_graph=True)
    actor_norm = _clip_grad_norm(actor_params, max_norm, device)
    actor_grads = _clone_grads(actor_params)
    _clear_grads(model.parameters())

    critic_loss.backward()
    critic_norm = _clip_grad_norm(critic_params, max_norm, device)
    _add_grads(actor_grads)
    return actor_norm, critic_norm


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
        spo_eps_low: float,
        spo_eps_high: float,
        autocast_enabled: bool,
    ) -> None:
        super().__init__()
        self.model = model
        self.value_coef = float(value_coef)
        self.target_entropy_coef = float(target_entropy_coef)
        self.fraction_entropy_coef = float(fraction_entropy_coef)
        self.norm_advantage = bool(norm_advantage)
        self.spo_eps_low = float(spo_eps_low)
        self.spo_eps_high = float(spo_eps_high)
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
        )
        with torch.autocast(
            device_type="cuda", dtype=torch.bfloat16, enabled=self.autocast_enabled
        ):
            out = self.model(feats)

        target_legal_mask = target_legal_mask.bool()
        has_legal_target = target_legal_mask.any(dim=-1)
        launch_logits = out.launch_logits.float().masked_fill(~has_legal_target, -20.0)
        target_logits = out.target_logits.float().masked_fill(~target_legal_mask, float("-inf"))
        target_logits = _safe_target_logits(target_logits)
        fraction_mean = out.fraction_mean.float()
        fraction_log_std = out.fraction_log_std.float()
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
        frac_lp = _squashed_normal_log_prob(
            fraction_mean, fraction_log_std, fraction.float()
        )
        chosen = launch_lp + launch_f * (target_lp + frac_lp)

        adv_b = advantage.float().unsqueeze(-1).expand_as(chosen)
        row_w = row_weight.to(device=owned_f.device, dtype=owned_f.dtype)
        row_w_b = row_w.unsqueeze(-1)
        owned_w = owned_f * row_w_b
        denom = owned_w.sum().clamp_min(1.0)

        log_ratio = chosen - old_log_prob.float()
        ratio = log_ratio.exp()
        adv_actor = adv_b
        if self.norm_advantage:
            adv_mean = (adv_actor * owned_w).sum() / denom
            adv_var = (((adv_actor - adv_mean).square()) * owned_w).sum() / denom
            adv_actor = (adv_actor - adv_mean) * torch.rsqrt(adv_var + 1e-8)
        ratio_diff = ratio - 1.0
        eps = torch.where(
            adv_actor * ratio_diff > 0.0,
            torch.full_like(ratio, self.spo_eps_high),
            torch.full_like(ratio, self.spo_eps_low),
        )
        spo_penalty = adv_actor.abs() * ratio_diff.square() / (2.0 * eps)
        policy_loss = -_weighted_mean(adv_actor * ratio - spo_penalty, owned_w)
        spo_penalty_mean = _weighted_mean(spo_penalty, owned_w)

        value_encoder = self.model.value_encoder
        target_probs = value_encoder.target_probs(ret.float())
        log_v_probs = F.log_softmax(value_logits, dim=-1)
        value_ce = -(target_probs * log_v_probs).sum(dim=-1)
        value_loss = _weighted_mean(value_ce, row_w)

        frac_entropy_per_planet = _squashed_normal_entropy(fraction_log_std)
        p_move_current = launch_logits.sigmoid()
        planet_entropy = _conditional_action_entropy(
            launch_logits, target_log_probs, frac_entropy_per_planet
        )
        entropy = (planet_entropy * owned_w).sum() / denom
        launch_entropy = (_bernoulli_entropy(launch_logits) * owned_w).sum() / denom
        target_entropy_per_planet = -(
            target_dist_probs
            * target_log_probs.clamp_min(torch.finfo(target_log_probs.dtype).min)
        ).sum(dim=-1)
        target_entropy = ((p_move_current * target_entropy_per_planet) * owned_w).sum()
        target_entropy = target_entropy / denom
        fraction_entropy = ((p_move_current * frac_entropy_per_planet) * owned_w).sum() / denom
        move_prob = (p_move_current * owned_w).sum() / denom
        target_confidence = (target_dist_probs.amax(dim=-1) * owned_w).sum() / denom
        fraction_mean_mean = (fraction_mean * owned_w).sum() / denom
        fraction_mean_abs_max = _weighted_max(fraction_mean.abs(), owned_w)
        fraction_log_std_mean = (fraction_log_std * owned_w).sum() / denom
        fraction_log_std_min = -_weighted_max(-fraction_log_std, owned_w)
        fraction_log_std_max = _weighted_max(fraction_log_std, owned_w)
        deterministic_fraction = 0.5 * (torch.tanh(fraction_mean) + 1.0)
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
        row_log_ratio = (log_ratio * owned_f).sum(dim=-1)
        row_log_ratio = torch.where(row_w > 0.0, row_log_ratio, row_log_ratio * 0.0)
        row_ratio = row_log_ratio.exp()
        row_denom = row_w.sum().clamp_min(1.0)
        kl = (((row_ratio - 1.0) - row_log_ratio) * row_w).sum() / row_denom
        pos_count = (owned_w * (adv_b >= 0.0).to(owned_f.dtype)).sum()
        total_owned = owned_w.sum().clamp_min(1.0)
        pos_frac = pos_count / total_owned
        metrics = torch.stack(
            [
                policy_loss.detach(),
                value_loss.detach(),
                entropy.detach(),
                kl.detach(),
                spo_penalty_mean.detach(),
                pos_frac.detach(),
                target_entropy.detach(),
                fraction_entropy.detach(),
                move_prob.detach(),
                target_confidence.detach(),
                fraction_mean_mean.detach(),
                fraction_mean_abs_max.detach(),
                fraction_log_std_mean.detach(),
                fraction_log_std_min.detach(),
                fraction_log_std_max.detach(),
                deterministic_fraction_mean.detach(),
            ]
        ).float()
        return actor_loss, critic_loss, metrics


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
    norm_advantage: bool,
    spo_eps_low: float,
    spo_eps_high: float,
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
        bool(norm_advantage),
        float(spo_eps_low),
        float(spo_eps_high),
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
        spo_eps_low=spo_eps_low,
        spo_eps_high=spo_eps_high,
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
    approx_kl: float
    spo_penalty: float
    pos_frac: float
    target_entropy: float = 0.0
    fraction_entropy: float = 0.0
    move_prob: float = 0.0
    target_confidence: float = 0.0
    fraction_mean_mean: float = 0.0
    fraction_mean_abs_max: float = 0.0
    fraction_log_std_mean: float = 0.0
    fraction_log_std_min: float = 0.0
    fraction_log_std_max: float = 0.0
    deterministic_fraction_mean: float = 0.0


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
    spo_eps_low: float,
    spo_eps_high: float,
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
      `owned_mask` [B,P], `target_legal_mask` [B,P,P].

    The actor objective is SPO asym:
      `-mean(ratio * advantage - |advantage| * (ratio - 1)^2 / (2 * eps))`.

    Distributional value loss: cross-entropy against HL-Gauss-encoded
    returns (`value_encoder.target_probs(returns)`). No value clipping.
    """
    n = batch["planet_feats"].shape[0]
    device = batch["planet_feats"].device

    metric_sum: torch.Tensor | None = None
    last_metrics: torch.Tensor | None = None
    n_steps = 0

    kernel = _get_ppo_kernel(
        model,
        value_coef=value_coef,
        target_entropy_coef=target_entropy_coef,
        fraction_entropy_coef=fraction_entropy_coef,
        norm_advantage=norm_advantage,
        spo_eps_low=spo_eps_low,
        spo_eps_high=spo_eps_high,
        compile_mode=compile_mode,
    )
    for _ in range(epochs):
        for mb, row_weight in _fixed_minibatches(n, minibatch_size, device):
            if compile_mode is not None:
                _mark_cuda_graph_step(device)
            actor_loss, critic_loss, metrics = kernel(
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
                batch["target_legal_mask"][mb],
            )

            optimizer.zero_grad(set_to_none=True)
            loss_scale = _minibatch_loss_scale(row_weight)
            _backward_actor_critic_with_separate_clips(
                model,
                actor_loss * loss_scale,
                critic_loss * loss_scale,
                grad_clip,
            )
            optimizer.step()

            if metric_sum is None:
                metric_sum = torch.zeros_like(metrics)
            metric_sum += metrics
            last_metrics = metrics
            n_steps += 1

    n_steps = max(1, n_steps)
    mean_logs = (
        [0.0] * 16
        if metric_sum is None
        else (metric_sum / n_steps).detach().cpu().tolist()
    )
    # Match CleanRL's PPO KL logging: `approx_kl` is the latest minibatch's
    # estimate after the PPO epoch loop, not an epoch mean. The surrounding
    # diagnostics stay averaged to preserve their lower-noise TensorBoard
    # behavior.
    kl_logs = (
        [0.0] * 16
        if last_metrics is None
        else last_metrics.detach().cpu().tolist()
    )
    return PPOLog(
        policy_loss=float(mean_logs[0]),
        value_loss=float(mean_logs[1]),
        entropy=float(mean_logs[2]),
        approx_kl=float(kl_logs[3]),
        spo_penalty=float(mean_logs[4]),
        pos_frac=float(mean_logs[5]),
        target_entropy=float(mean_logs[6]),
        fraction_entropy=float(mean_logs[7]),
        move_prob=float(mean_logs[8]),
        target_confidence=float(mean_logs[9]),
        fraction_mean_mean=float(mean_logs[10]),
        fraction_mean_abs_max=float(mean_logs[11]),
        fraction_log_std_mean=float(mean_logs[12]),
        fraction_log_std_min=float(mean_logs[13]),
        fraction_log_std_max=float(mean_logs[14]),
        deterministic_fraction_mean=float(mean_logs[15]),
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
            if compile_mode is not None:
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
