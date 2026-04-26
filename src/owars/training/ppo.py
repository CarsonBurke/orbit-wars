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

from ..policies.model import OrbitPolicy


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
    clip_eps: float,
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
      `target_idx` [B,P], `fraction` [B,P], `old_log_prob` [B,P],
      `advantage` [B], `return` [B], `owned_mask` [B,P].

    Policy loss is *token-level* (VAPO §4.2): summed over all
    (sample, owned-planet) pairs and divided by the count of active
    pairs in the minibatch. Standard PPO would average per-sample first.
    """
    n = batch["planet_feats"].shape[0]
    idx = np.arange(n)

    pl_loss = vl_loss = ent_log = kl_log = clip_log = 0.0
    n_steps = 0

    for _ in range(epochs):
        np.random.shuffle(idx)
        for start in range(0, n, minibatch_size):
            mb = idx[start : start + minibatch_size]

            class _Feats:
                pass

            feats = _Feats()
            feats.planet_feats = batch["planet_feats"][mb]
            feats.planet_mask = batch["planet_mask"][mb]
            feats.planet_owned_mask = batch["planet_owned_mask"][mb]
            feats.planet_ids = batch["planet_ids"][mb]
            feats.planet_garrison = batch["planet_garrison"][mb]
            feats.fleet_feats = batch["fleet_feats"][mb]
            feats.fleet_mask = batch["fleet_mask"][mb]

            out = model(feats)  # type: ignore[arg-type]

            owned = batch["owned_mask"][mb]
            owned_f = owned.float()
            p = out.target_logits.shape[1]
            target = batch["target_idx"][mb].clamp(0, p)
            target_log_probs = F.log_softmax(out.target_logits, dim=-1)
            target_lp = target_log_probs.gather(-1, target.unsqueeze(-1)).squeeze(-1)

            # Beta log-prob for the recorded fraction; only counted when the
            # action wasn't no-op (target slot == p). Mirrors what the
            # rollout's `sample_with_record` stored in `old_log_prob`.
            frac = batch["fraction"][mb].clamp(1e-6, 1.0 - 1e-6)
            beta_dist = torch.distributions.Beta(out.fraction_alpha, out.fraction_beta)
            frac_lp = beta_dist.log_prob(frac)
            is_noop = (target == p).float()
            chosen = target_lp + (1.0 - is_noop) * frac_lp

            old_log_prob = batch["old_log_prob"][mb]  # [B, P]
            advantage = batch["advantage"][mb]
            advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)

            # Token-level (per-owned-planet) PPO ratio. We broadcast the
            # per-trajectory advantage over the planet axis.
            ratio = (chosen - old_log_prob).exp()  # [B, P]
            adv_b = advantage.unsqueeze(-1).expand_as(ratio)
            unclipped = ratio * adv_b
            clipped = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv_b
            per_token = -torch.min(unclipped, clipped)
            denom = owned_f.sum().clamp_min(1.0)
            policy_loss = (per_token * owned_f).sum() / denom

            value_loss = (out.value - batch["return"][mb]).pow(2).mean()

            # Categorical(logits=...).entropy() handles -inf logits cleanly
            # (0 * log 0 → 0 internally); the naive p*log(p) form returns
            # nan whenever any target slot is masked.
            entropy = torch.distributions.Categorical(
                logits=out.target_logits
            ).entropy()
            entropy = (entropy * owned_f).sum() / denom

            loss = policy_loss + value_coef * value_loss - entropy_coef * entropy

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            with torch.no_grad():
                kl = ((old_log_prob - chosen) * owned_f).sum() / denom
                clip_frac = (((ratio - 1.0).abs() > clip_eps).float() * owned_f).sum() / denom

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

            class _Feats:
                pass

            feats = _Feats()
            feats.planet_feats = batch["planet_feats"][mb]
            feats.planet_mask = batch["planet_mask"][mb]
            feats.planet_owned_mask = batch["planet_owned_mask"][mb]
            feats.planet_ids = batch["planet_ids"][mb]
            feats.planet_garrison = batch["planet_garrison"][mb]
            feats.fleet_feats = batch["fleet_feats"][mb]
            feats.fleet_mask = batch["fleet_mask"][mb]

            out = model(feats)  # type: ignore[arg-type]
            value_loss = (out.value - batch["return"][mb]).pow(2).mean()

            optimizer.zero_grad(set_to_none=True)
            value_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            total += float(value_loss.item())
            n_steps += 1

    return total / max(1, n_steps)
