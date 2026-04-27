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

The cold-start fix — value pretraining with a frozen behavior policy — is
in `train.py::pretrain_value`, not here. This file is just the per-update
math.
"""

from __future__ import annotations

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
      `target_idx` [B,P], `fraction` [B,P], `angle_offset` [B,P],
      `old_log_prob` [B,P], `advantage` [B], `return` [B],
      `owned_mask` [B,P].

    Policy loss is *token-level* (VAPO §4.2): summed over all
    (sample, owned-planet) pairs and divided by the count of active
    pairs in the minibatch. Standard PPO would average per-sample first.
    """
    n = batch["planet_feats"].shape[0]
    idx = np.arange(n)

    pl_loss = vl_loss = ent_log = kl_log = clip_log = 0.0
    n_steps = 0

    # bf16 autocast unlocks the SDPA Flash-Attention 2 kernel (head_dim must
    # also be FA-eligible — see model config). bf16 has fp32-equivalent range
    # so no GradScaler is needed; AdamW keeps fp32 master weights via
    # PyTorch's autocast handling. Outside cuda we stay in fp32.
    autocast_enabled = (
        next(model.parameters()).is_cuda
        if any(True for _ in model.parameters())
        else False
    )
    for _ in range(epochs):
        np.random.shuffle(idx)
        for start in range(0, n, minibatch_size):
            mb = idx[start : start + minibatch_size]

            with torch.autocast(
                device_type="cuda", dtype=torch.bfloat16, enabled=autocast_enabled
            ):
                out = model(_slice_feats(batch, mb))

                owned_f = batch["owned_mask"][mb].float()
                p = out.target_logits.shape[1]
                target = batch["target_idx"][mb].clamp(0, p)
                target_log_probs = F.log_softmax(out.target_logits, dim=-1)
                target_lp = target_log_probs.gather(-1, target.unsqueeze(-1)).squeeze(-1)

                # Beta log-probs for the recorded fraction *and* angle residual;
                # both are only counted when the action wasn't no-op (target slot
                # == p). Mirrors what `sample_with_record` stored in `old_log_prob`.
                frac = batch["fraction"][mb].clamp(1e-6, 1.0 - 1e-6)
                frac_dist = torch.distributions.Beta(out.fraction_alpha, out.fraction_beta)
                frac_lp = frac_dist.log_prob(frac)
                ang = batch["angle_offset"][mb].clamp(1e-6, 1.0 - 1e-6)
                angle_dist = torch.distributions.Beta(out.angle_alpha, out.angle_beta)
                angle_lp = angle_dist.log_prob(ang)
                is_noop = (target == p).float()
                move_mask = 1.0 - is_noop
                chosen = target_lp + move_mask * (frac_lp + angle_lp)

                old_log_prob = batch["old_log_prob"][mb]  # [B, P]
                # Advantages were normalized once over the whole batch in
                # `_stack_trajectories`; standard PPO does this rather than
                # per-minibatch (per-mb adds noise from each minibatch's own
                # mean/std).
                advantage = batch["advantage"][mb]

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

                value_loss = (out.value - batch["return"][mb]).pow(2).mean()

                # Reuse `target_log_probs` for the entropy term — cheaper than
                # constructing Categorical(logits=...).entropy() which redoes
                # log_softmax internally. We clamp -inf log-probs to the finite
                # min (Categorical.entropy's exact trick): exp(-inf)=0 and 0*
                # -3.4e38=0, but 0*-inf=nan and a torch.where mask wouldn't
                # save us — both branches are evaluated and the nan poisons
                # the backward pass.
                min_real = torch.finfo(target_log_probs.dtype).min
                log_probs_safe = target_log_probs.clamp_min(min_real)
                entropy = -(target_log_probs.exp() * log_probs_safe).sum(dim=-1)
                entropy = (entropy * owned_f).sum() / denom

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

            pl_loss += float(policy_loss.item())
            vl_loss += float(value_loss.item())
            ent_log += float(entropy.item())
            kl_log += float(kl.item())
            clip_log += float(clip_frac.item())
            n_steps += 1

    n_steps = max(1, n_steps)
    return PPOLog(
        policy_loss=pl_loss / n_steps,
        value_loss=vl_loss / n_steps,
        entropy=ent_log / n_steps,
        approx_kl=kl_log / n_steps,
        clip_frac=clip_log / n_steps,
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
    idx = np.arange(n)
    total = 0.0
    n_steps = 0

    for _ in range(epochs):
        np.random.shuffle(idx)
        for start in range(0, n, minibatch_size):
            mb = idx[start : start + minibatch_size]
            out = model(_slice_feats(batch, mb))
            value_loss = (out.value - batch["return"][mb]).pow(2).mean()

            optimizer.zero_grad(set_to_none=True)
            value_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            total += float(value_loss.item())
            n_steps += 1

    return total / max(1, n_steps)
