"""PPO update with VAPO-style critic — decoupled-GAE + token-level loss.

VAPO (arxiv 2504.05118) and its prerequisite VC-PPO (arxiv 2503.01491) make
two changes that matter most for sparse-terminal-reward, finite-horizon
self-play:

  1. **Decoupled GAE**: the *critic* regresses on Monte-Carlo returns (λ=1,
     unbiased), while the *actor* uses a variance-reduced advantage with a
     smaller λ. Mixing-λ in the value target biases the critic toward 0
     during the cold start; using λ=1 for V_target is provably non-biasing
     for the policy gradient (VC-PPO §3.3, eqs. 7–8).
  2. **Token-level (step-level) policy loss**: instead of mean-of-means
     across episodes, sum across all (episode, step) pairs and divide by
     the total number of active steps. Stops long episodes from being
     down-weighted (VAPO §4.2, eq. 7). For Orbit Wars the gain is small
     (episodes are bounded ≤500) but it's free.

Continuous fraction action: tanh-squashed Normal(μ, σ) per owned planet.
We re-evaluate log_prob from the recorded *pre-tanh* sample `z` (no
`atanh` round-trip — exact at all squashed values).

The cold-start fix — value pretraining with a frozen behavior policy — is
in `train.py::pretrain_value`, not here. This file is just the per-update
math.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

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


@dataclass
class PPOLog:
    policy_loss: float
    value_loss: float
    entropy: float
    approx_kl: float
    clip_frac: float


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
    clip_eps_low: float,
    clip_eps_high: float,
    value_coef: float,
    entropy_coef: float,
    epochs: int,
    minibatch_size: int,
    grad_clip: float,
) -> PPOLog:
    """Run `epochs × ⌈N/B⌉` minibatch updates on `batch`.

    Expected keys:
      `planet_feats`, `planet_mask`, `planet_owned_mask`, `planet_ids`,
      `planet_garrison`, `fleet_feats`, `fleet_mask`,
      `target_idx` [B,P], `frac_z` [B,P],
      `old_log_prob` [B,P], `advantage` [B], `return` [B],
      `owned_mask` [B,P].

    `frac_z` is the *pre-tanh* Normal sample recorded at rollout time. The
    new policy's log_prob is computed by re-evaluating Normal(μ, σ).log_prob
    at that exact `z` plus the tanh Jacobian — exact, no `atanh` round-trip.

    Policy loss is *token-level* (VAPO §4.2): summed over all
    (sample, owned-planet) pairs and divided by the count of active
    pairs in the minibatch. Standard PPO would average per-sample first.

    This is *clip-PPO*, not KL-PPO: the trust region is enforced exclusively
    by `clip(ratio, 1 ± ε)`. We don't run a KL early-stop — `approx_kl` is
    logged for diagnostics only. Per-update drift is bounded *by
    construction* via (a) orthogonal init + Muon's NS5-bounded spectral-
    norm step, (b) `logit_softcap` capping output logit magnitudes, and
    (c) `muon_weight_decay` preventing weight-norm drift over time.
    """
    n = batch["planet_feats"].shape[0]
    device = batch["planet_feats"].device

    metric_sum: torch.Tensor | None = None
    n_steps = 0

    # bf16 autocast unlocks the SDPA Flash-Attention 2 kernel (head_dim must
    # also be FA-eligible — see model config). bf16 has fp32-equivalent range
    # so no GradScaler is needed; AdamW keeps fp32 master weights via
    # PyTorch's autocast handling. Outside cuda we stay in fp32.
    #
    # Autocast wraps ONLY the model forward — log_softmax / log_prob / ratio
    # / value_loss all run in fp32 after the cast, mirroring pg's
    # `F.cross_entropy(logits.float(), …)` pattern (sota_train_gpt.py:163).
    # bf16's 7 mantissa bits put a noise floor on log-prob differences
    # (~0.01 nats per update is below bf16 precision); computing the
    # importance ratio in bf16 amplifies that noise into approx_kl.
    autocast_enabled = (
        next(model.parameters()).is_cuda
        if any(True for _ in model.parameters())
        else False
    )
    for _ in range(epochs):
        idx = torch.randperm(n, device=device)
        for start in range(0, n, minibatch_size):
            mb = idx[start : start + minibatch_size]

            with torch.autocast(
                device_type="cuda", dtype=torch.bfloat16, enabled=autocast_enabled
            ):
                out = model(_slice_feats(batch, mb))

            target_logits = out.target_logits.float()
            fraction_mu = out.fraction_mu.float()
            fraction_log_sigma = out.fraction_log_sigma.float()
            value = out.value.float()

            owned_f = batch["owned_mask"][mb].float()
            p = target_logits.shape[1]
            target = batch["target_idx"][mb].clamp(0, p)
            target_log_probs = F.log_softmax(target_logits, dim=-1)
            target_lp = target_log_probs.gather(-1, target.unsqueeze(-1)).squeeze(-1)

            # tanh-squashed Normal log-prob at the recorded pre-tanh sample
            # `z` — exact (no atanh round-trip). The fraction component is
            # only counted when the action wasn't no-op (target slot == p),
            # mirroring what `sample_with_record` stored in `old_log_prob`.
            z = batch["frac_z"][mb].float()
            sigma = fraction_log_sigma.exp()
            normal_lp = (
                -0.5 * ((z - fraction_mu) / sigma).pow(2)
                - fraction_log_sigma
                - 0.5 * math.log(2.0 * math.pi)
            )
            # Stable log(1 - tanh²(z)) form — see `sampling._tanh_normal_log_prob`.
            # Sign on `log 2`: see the change-of-variables derivation in
            # `sampling._tanh_normal_log_prob`'s docstring.
            tanh_correction = 2.0 * (
                math.log(2.0) - z - F.softplus(-2.0 * z)
            )
            frac_lp = normal_lp - tanh_correction + math.log(2.0)
            is_noop = (target == p).float()
            move_mask = 1.0 - is_noop
            chosen = target_lp + move_mask * frac_lp

            old_log_prob = batch["old_log_prob"][mb].float()  # [B, P]
            # Advantages were normalized once over the whole batch in
            # `_stack_trajectories`; standard PPO does this rather than
            # per-minibatch (per-mb adds noise from each minibatch's own
            # mean/std).
            advantage = batch["advantage"][mb].float()

            # Token-level (per-owned-planet) PPO ratio. We broadcast the
            # per-trajectory advantage over the planet axis.
            ratio = (chosen - old_log_prob).exp()  # [B, P]
            adv_b = advantage.unsqueeze(-1).expand_as(ratio)
            unclipped = ratio * adv_b
            # Asymmetric clip — `1 - eps_low` on the lower bound, `1 +
            # eps_high` on the upper. Equal eps recovers symmetric PPO.
            clipped = (
                torch.clamp(ratio, 1.0 - clip_eps_low, 1.0 + clip_eps_high) * adv_b
            )
            per_token = -torch.min(unclipped, clipped)
            denom = owned_f.sum().clamp_min(1.0)
            policy_loss = (per_token * owned_f).sum() / denom

            value_loss = (value - batch["return"][mb].float()).pow(2).mean()

            # Entropy bonus is the sum of per-axis entropies: Categorical
            # over targets + Normal over the pre-tanh fraction sample. We
            # use the *unsquashed* Normal entropy `0.5·log(2πe·σ²)`; the
            # tanh-squash correction is small and adds estimator noise
            # without changing the qualitative gradient (SAC convention).
            min_real = torch.finfo(target_log_probs.dtype).min
            log_probs_safe = target_log_probs.clamp_min(min_real)
            target_entropy = -(target_log_probs.exp() * log_probs_safe).sum(dim=-1)
            normal_entropy = 0.5 * math.log(2.0 * math.pi * math.e) + fraction_log_sigma
            # Only count the Normal entropy where the action would actually
            # use it — i.e. on owned planets that *aren't* no-op. For owned
            # no-op planets the fraction sample is drawn but ignored, so
            # rewarding its entropy would pay the policy to be uncertain
            # about an action it doesn't take.
            planet_entropy = target_entropy + move_mask * normal_entropy
            entropy = (planet_entropy * owned_f).sum() / denom

            loss = policy_loss + value_coef * value_loss - entropy_coef * entropy

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            with torch.no_grad():
                kl = ((old_log_prob - chosen) * owned_f).sum() / denom
                # Count tokens whose ratio drifted past *either* asymmetric
                # bound — preserves the diagnostic of "fraction clipped" even
                # when eps_low ≠ eps_high.
                clip_low = ratio < (1.0 - clip_eps_low)
                clip_high = ratio > (1.0 + clip_eps_high)
                clipped_mask = (clip_low | clip_high).float()
                clip_frac = (clipped_mask * owned_f).sum() / denom

            metrics = torch.stack(
                [
                    policy_loss.detach(),
                    value_loss.detach(),
                    entropy.detach(),
                    kl.detach(),
                    clip_frac.detach(),
                ]
            ).float()
            if metric_sum is None:
                metric_sum = torch.zeros_like(metrics)
            metric_sum += metrics
            n_steps += 1

    n_steps = max(1, n_steps)
    if metric_sum is None:
        logs = [0.0] * 5
    else:
        logs = (metric_sum / n_steps).detach().cpu().tolist()
    return PPOLog(
        policy_loss=float(logs[0]),
        value_loss=float(logs[1]),
        entropy=float(logs[2]),
        approx_kl=float(logs[3]),
        clip_frac=float(logs[4]),
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
    """Critic-only MSE update for the value-pretraining phase.

    `batch["return"]` here should be Monte-Carlo returns (γ=1 for terminal-
    only reward → just the trajectory outcome). Run this for a few hundred
    steps against a frozen behavior policy before turning on PPO.

    Reports mean value loss over the pass.
    """
    n = batch["planet_feats"].shape[0]
    device = batch["planet_feats"].device
    total: torch.Tensor | None = None
    n_steps = 0

    autocast_enabled = (
        next(model.parameters()).is_cuda
        if any(True for _ in model.parameters())
        else False
    )
    for _ in range(epochs):
        idx = torch.randperm(n, device=device)
        for start in range(0, n, minibatch_size):
            mb = idx[start : start + minibatch_size]
            with torch.autocast(
                device_type="cuda", dtype=torch.bfloat16, enabled=autocast_enabled
            ):
                out = model(_slice_feats(batch, mb))
            # fp32 loss math (same rationale as `ppo_update`).
            value_loss = (
                out.value.float() - batch["return"][mb].float()
            ).pow(2).mean()

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
