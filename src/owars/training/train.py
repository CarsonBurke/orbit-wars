"""Single-run training driver.

Loop:
  0. (Optional) **Value pretraining** — roll N episodes against a frozen
     behavior policy and fit the critic to the configured bootstrapped
     lambda-return target. Gives PPO a warm value function so the actor's
     advantage is less noisy at step 0.
  1. For each PPO update:
     a. Sample N episodes against opponents. The normal league mode fills each
        opponent seat from live self-play or a top-K frozen snapshot. Fixed mode
        instead samples static builtins such as `sniper`.
     b. Compute GAE advantages and lambda-return critic targets. Dense rewards
        use `ppo.gae_lambda` for both by default; sparse terminal reward
        ablations can set `ppo.value_gae_lambda=1.0` for MC critic targets.
     c. PPO update with token-level loss.
     d. Update Elo from each game's per-seat scores. The learner
        rating drives snapshot retention — when a snapshot's Elo
        drops below the top-K cut, it's evicted.
     e. Every `snapshot_every`, save a frozen snapshot at the learner's
        current Elo.

The loop is small on purpose. Tuning lives in YAML configs and the
opponent pool — see `STRATEGY.md`.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from collections.abc import Sequence
from contextlib import ExitStack, suppress
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from ..policies.config import OrbitPolicyConfig
from ..policies.features import EncodedObs, active_fleet_width, bucket_fleet_width
from ..policies.model import (
    OrbitPolicy,
    ngpt_control_stats,
    normalize_matrices,
    restore_fp32_params,
)
from ..utils import TBLogger, set_seed
from .config import OptimCfg, RunConfig, load_config
from .elo import EloTracker
from .league import (
    BUILTIN,
    LEARNER_NAME,
    FixedOpponentPool,
    NoBuiltinTrainingPool,
    OpponentPool,
    OpponentSlot,
)
from .muon import MultiOptimizer, Muon
from .numpy_env import NumpyVecEnv
from .ppo import (
    compute_gae,
    compute_old_log_probs_and_values,
    compute_old_policy_dist,
    ppo_update,
    value_only_update,
)
from .rollout import Trajectory, TrajectoryRecordRef
from .sharded_numpy_env import ShardedNumpyVecEnv
from .vec_env import VecEnv
from .vec_rollout import alternating_learner_seats, rollout_episodes_batched

# Subset of control tensors that route to the dedicated `control_lr` AdamW
# group — the nGPT hypersphere controls (per-channel eigen LRs
# `attn_alpha`/`mlp_alpha`/`cross_alpha`, QK scale `sqk`, MLP scale `suv`) and
# the target-readout attention temperature (`q_gain`/`target_q_gain`). They get
# their own group (zero weight decay, reference-faithful base lr) rather than a
# faster lr. The summary tokens (`actor_token`, `critic_token`) intentionally
# stay in the slow default group: they are learnable biases on the residual
# stream and moving them at scalar speed destabilizes early training.
_CONTROL_LR_PATTERNS: tuple[str, ...] = (
    "attn_alpha",
    "mlp_alpha",
    "cross_alpha",
    "skip_alpha",  # optional per-channel U-net skip (block_skip)
    "sqk",
    "suv",
    "q_gain",  # also matches `target_q_gain` via substring
)

# Parameter-name prefixes for transformer block stacks. This mirrors
# parameter-golf's optimizer ownership: Muon sees block matrices only, while
# embeddings/readouts are handled by AdamW.
_MUON_BLOCK_PREFIXES: tuple[str, ...] = (
    "layers.",
    "fleet_tokenizer.layers.",
    "destination_fleet_conditioner.cross_attn.",
)

# Task-specific readouts stay out of Muon. They are small enough that fused
# AdamW is faster than Newton-Schulz, and they map directly to policy/value
# logits where a full spectral step is too aggressive at cold start.
_HEAD_LR_PATTERNS: tuple[str, ...] = (
    "target_query",
    "target_key",
    "target_noop_key",
    "fraction_alpha_head",
    "fraction_beta_head",
    "value_head",
)


@dataclass
class DiscountedReturnNormalizer:
    gamma: float
    clip: float | None = 10.0
    epsilon: float = 1.0e-8
    mean: float = 0.0
    var: float = 1.0
    count: float = 1.0e-4
    batch_clip_frac: float = 0.0
    batch_reward_absmax: float = 0.0
    _batch_clip_count: float = 0.0
    _batch_reward_count: float = 0.0

    def normalize_episode(self, rewards: np.ndarray) -> np.ndarray:
        return self.normalize_episodes([rewards])[0]

    def normalize_episodes(self, episodes: list[np.ndarray]) -> list[np.ndarray]:
        self.begin_batch()
        out = [np.empty_like(rewards, dtype=np.float32) for rewards in episodes]
        running = np.zeros(len(episodes), dtype=np.float32)
        max_len = max((len(rewards) for rewards in episodes), default=0)
        for step in range(max_len):
            active = [
                episode_idx for episode_idx, rewards in enumerate(episodes) if step < len(rewards)
            ]
            if not active:
                continue
            rewards = np.asarray(
                [episodes[episode_idx][step] for episode_idx in active],
                dtype=np.float32,
            )
            scaled, next_running = self._normalize_step_vector(
                rewards,
                running[active],
            )
            running[active] = next_running
            for local_idx, episode_idx in enumerate(active):
                out[episode_idx][step] = scaled[local_idx]
        return out

    def _normalize_step_vector(
        self,
        rewards: np.ndarray,
        running: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        running = running * self.gamma + rewards
        self._update(running)
        scaled = rewards / math.sqrt(self.var + self.epsilon)
        abs_scaled = np.abs(scaled)
        self.batch_reward_absmax = max(
            self.batch_reward_absmax,
            float(abs_scaled.max()) if abs_scaled.size else 0.0,
        )
        if self.clip is not None:
            self._batch_clip_count += float(np.count_nonzero(abs_scaled > self.clip))
            self._batch_reward_count += float(abs_scaled.size)
            self.batch_clip_frac = self._batch_clip_count / max(
                self._batch_reward_count,
                1.0,
            )
            scaled = np.clip(scaled, -self.clip, self.clip)
        else:
            self.batch_clip_frac = 0.0
        return scaled.astype(np.float32, copy=False), running

    def begin_batch(self) -> None:
        self.batch_clip_frac = 0.0
        self.batch_reward_absmax = 0.0
        self._batch_clip_count = 0.0
        self._batch_reward_count = 0.0

    def _update(self, values: np.ndarray) -> None:
        if values.size == 0:
            return
        batch_mean = float(values.mean())
        batch_var = float(values.var())
        batch_count = float(values.size)
        delta = batch_mean - self.mean
        total = self.count + batch_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta * delta * self.count * batch_count / total
        self.mean += delta * batch_count / total
        self.var = max(m2 / total, self.epsilon)
        self.count = total

    def state_dict(self) -> dict[str, float | None]:
        return {
            "gamma": self.gamma,
            "clip": self.clip,
            "epsilon": self.epsilon,
            "mean": self.mean,
            "var": self.var,
            "count": self.count,
        }

    def load_state_dict(self, state: dict[str, float | None]) -> None:
        gamma = float(state.get("gamma", self.gamma) or self.gamma)
        if not math.isclose(gamma, self.gamma, rel_tol=0.0, abs_tol=1.0e-12):
            raise ValueError("checkpoint critic_return_normalizer gamma does not match config")
        clip_raw = state.get("clip", self.clip)
        clip = None if clip_raw is None else float(clip_raw)
        if clip != self.clip:
            raise ValueError("checkpoint critic_return_normalizer clip does not match config")
        epsilon = float(state.get("epsilon", self.epsilon) or self.epsilon)
        if not math.isclose(epsilon, self.epsilon, rel_tol=0.0, abs_tol=1.0e-18):
            raise ValueError("checkpoint critic_return_normalizer epsilon does not match config")
        self.mean = float(state.get("mean", self.mean) or 0.0)
        self.var = float(state.get("var", self.var) or 1.0)
        self.count = float(state.get("count", self.count) or 1.0e-4)


@dataclass
class PercentileReturnNormalizer:
    """DreamerV3 percentile return normalizer for actor advantages.

    Mirrors dreamerv3 `Normalize(impl='perc')` (embodied/jax/utils.py) under the
    `retnorm` config (rate=0.01, limit=1.0, perclo=5, perchi=95, debias=False):
    track an EMA of the `perclo`/`perchi` percentiles of returns, then scale the
    policy advantage by `1 / max(limit, hi - lo)` with no mean subtraction. The
    `max(limit, .)` floor stops near-degenerate return spreads from amplifying
    advantages — the failure mode that a per-minibatch z-score has at cold start.
    """

    rate: float = 0.01
    perclo: float = 5.0
    perchi: float = 95.0
    limit: float = 1.0
    lo: float = 0.0
    hi: float = 0.0

    def update(self, returns: np.ndarray) -> None:
        if returns.size == 0:
            return
        lo = float(np.percentile(returns, self.perclo))
        hi = float(np.percentile(returns, self.perchi))
        self.lo = (1.0 - self.rate) * self.lo + self.rate * lo
        self.hi = (1.0 - self.rate) * self.hi + self.rate * hi

    @property
    def scale(self) -> float:
        return max(self.limit, self.hi - self.lo)

    def state_dict(self) -> dict[str, float]:
        return {
            "rate": self.rate,
            "perclo": self.perclo,
            "perchi": self.perchi,
            "limit": self.limit,
            "lo": self.lo,
            "hi": self.hi,
        }

    def load_state_dict(self, state: dict[str, float]) -> None:
        for key in ("rate", "perclo", "perchi", "limit"):
            stored = state.get(key)
            if stored is not None and not math.isclose(
                float(stored), getattr(self, key), rel_tol=0.0, abs_tol=1.0e-12
            ):
                raise ValueError(
                    f"checkpoint advantage_return_normalizer {key} does not match config"
                )
        self.lo = float(state.get("lo", self.lo) or 0.0)
        self.hi = float(state.get("hi", self.hi) or 0.0)


def kl_lr_ema_alpha(half_life: float) -> float:
    """Return EMA alpha for a half-life measured in PPO updates."""
    if half_life <= 0.0:
        raise ValueError("half_life must be positive")
    return 1.0 - 0.5 ** (1.0 / half_life)


def update_kl_lr_controller(
    *,
    kl_ema: float,
    lr_scale: float,
    observed_kl: float,
    cfg: OptimCfg,
) -> tuple[float, float]:
    """Update the KL EMA and persistent LR scale for the next PPO update."""
    if not math.isfinite(observed_kl):
        return kl_ema, lr_scale
    alpha = kl_lr_ema_alpha(cfg.kl_lr_ema_half_life)
    signal = max(0.0, float(observed_kl))
    next_ema = alpha * signal + (1.0 - alpha) * float(kl_ema)
    next_scale = float(lr_scale) * math.sqrt(cfg.kl_lr_target / max(next_ema, 1e-12))
    next_scale = min(cfg.kl_lr_max_scale, max(cfg.kl_lr_min_scale, next_scale))
    return next_ema, next_scale


def kl_lr_signal_from_log(log) -> float:
    """Return the KL signal used for adaptive LR.

    Orbit Wars has a factorized action per owned source planet. The adaptive
    LR controller should track the latest-minibatch per-planet KL, not the
    joint row KL reported by `approx_kl`.
    """
    return float(log.per_planet_approx_kl)


def _split_params(
    model: OrbitPolicy,
) -> tuple[
    list[torch.nn.Parameter],
    list[torch.nn.Parameter],
    list[torch.nn.Parameter],
    list[torch.nn.Parameter],
]:
    """Partition `model.parameters()` into
    (muon_blocks, adamw_default, adamw_control, adamw_head).

    Muon (blocks): 2D matrices inside the transformer block stacks, matching
    parameter-golf's optimizee split.

    AdamW (head-lr): task readout matrices — `target_query`, `target_key`,
    Beta fraction heads, categorical action heads, and `value_head`.

    AdamW (control-lr): per-channel residual scales and `q_gain`s — the nGPT
    hypersphere controls, kept in their own zero-decay group at the
    reference-faithful base lr, see `OptimCfg.control_lr`.

    AdamW (default-lr): everything else — input projections, biases,
    summary tokens, and latent tokens.
    """
    muon_blocks: list[torch.nn.Parameter] = []
    adamw_default: list[torch.nn.Parameter] = []
    adamw_control: list[torch.nn.Parameter] = []
    adamw_head: list[torch.nn.Parameter] = []
    for name, p in model.named_parameters():
        is_muon_block = (
            p.ndim == 2
            and any(name.startswith(prefix) for prefix in _MUON_BLOCK_PREFIXES)
            and not any(pat in name for pat in _CONTROL_LR_PATTERNS)
        )
        is_control_lr = any(pat in name for pat in _CONTROL_LR_PATTERNS)
        is_head_lr = any(pat in name for pat in _HEAD_LR_PATTERNS)
        if is_muon_block:
            muon_blocks.append(p)
        elif is_head_lr:
            adamw_head.append(p)
        elif is_control_lr:
            adamw_control.append(p)
        else:
            adamw_default.append(p)
    return muon_blocks, adamw_default, adamw_control, adamw_head


def _build_optimizer(model: OrbitPolicy, cfg: OptimCfg) -> MultiOptimizer:
    """Construct the dual Muon + AdamW optimizer.

    See `OptimCfg` and `muon.py` for rationale. The combined object exposes
    `step` / `zero_grad` / `param_groups` so the PPO loop's clip-grad and
    step calls work transparently across both children.
    """
    muon_blocks, adamw_default, adamw_control, adamw_head = _split_params(model)
    # One Muon param-group for transformer block matrices, matching
    # parameter-golf's optimizee split. Task heads and input projections are
    # AdamW below; running Newton-Schulz on those tiny readouts is slower and
    # was the source of oversized cold-start policy-head moves.
    muon_opt = Muon(
        [{"params": muon_blocks, "lr": cfg.muon_lr}],
        lr=cfg.muon_lr,
        momentum=cfg.muon_momentum,
        backend_steps=cfg.muon_backend_steps,
        normuon=cfg.muon_normuon,
        beta2=cfg.muon_beta2,
        fused=cfg.muon_fused,
        weight_decay=cfg.muon_weight_decay,
        momentum_warmup_steps=cfg.muon_momentum_warmup_steps,
        momentum_warmup_start=cfg.muon_momentum_warmup_start,
    )
    # parameter-golf-style AdamW: betas=(0.9, 0.95) instead of the default
    # (0.9, 0.999). β2=0.95 makes the second-moment estimate reach steady
    # state in ~20 steps instead of ~1000 — important for cold-start, where
    # the first few hundred updates have rapidly-changing gradient
    # statistics and a stale running variance underestimates the current
    # step size, which translates into oversized parameter updates.
    #
    # Three AdamW param-groups: default tensors at `lr`, control tensors at
    # `control_lr` (reference-faithful: == `lr`, the nGPT scalars train at the
    # base AdamW lr, not at muon_lr), and task readouts at `head_lr`.
    adamw_opt = torch.optim.AdamW(
        [
            {"params": adamw_default, "lr": cfg.lr},
            # The control group is the nGPT hypersphere scalars (eigen LRs,
            # `sqk`, `suv`) plus the readout temperature — all of which directly
            # set magnitudes that are NOT re-projected by `normalize_matrices`.
            # Decaying them would slowly collapse the eigen-LR toward identity
            # and flatten the QK/MLP scales, so they get weight_decay=0 (nGPT
            # keeps decay off every ndim<2 param, `ngpt/model.py:305-306`).
            {"params": adamw_control, "lr": cfg.control_lr, "weight_decay": 0.0},
            {"params": adamw_head, "lr": cfg.head_lr},
        ],
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
        betas=(0.9, 0.95),
        fused=True,
    )
    return MultiOptimizer([muon_opt, adamw_opt], lr_warmup_steps=cfg.lr_warmup_steps)


def _build_model(cfg: RunConfig) -> OrbitPolicy:
    pcfg = OrbitPolicyConfig(
        dim=cfg.model.dim,
        ff_dim=cfg.model.ff_dim,
        depth=cfg.model.depth,
        n_heads=cfg.model.n_heads,
        n_kv_heads=cfg.model.n_kv_heads,
        dropout=cfg.model.dropout,
        eigen_alpha_init=cfg.model.eigen_alpha_init,
        qk_gain_init=cfg.model.qk_gain_init,
        block_skip=cfg.model.block_skip,
        planet_rope_fraction=cfg.model.planet_rope_fraction,
        planet_rope_base=cfg.model.planet_rope_base,
        encoder_backend=cfg.model.encoder_backend,
        num_fleet_latents=cfg.model.num_fleet_latents,
        fleet_tokenizer_depth=cfg.model.fleet_tokenizer_depth,
        value_hidden=cfg.model.value_hidden,
        value_num_bins=cfg.model.value_num_bins,
        value_sigma_to_bin_ratio=cfg.model.value_sigma_to_bin_ratio,
        critic_mtp_horizon=cfg.model.critic_mtp_horizon,
        value_min=cfg.model.value_min,
        value_max=cfg.model.value_max,
        value_symlog=cfg.model.value_symlog,
        value_bucket=cfg.model.value_bucket,
        action_logit_softcap=cfg.model.action_logit_softcap,
        global_features=cfg.model.global_features,
    )
    return OrbitPolicy(pcfg)


def _build_reward_normalizer(cfg: RunConfig) -> DiscountedReturnNormalizer | None:
    if cfg.ppo.critic_return_norm != "discounted_return_rms":
        return None
    return DiscountedReturnNormalizer(
        gamma=(
            cfg.ppo.gamma
            if cfg.ppo.critic_return_norm_gamma is None
            else cfg.ppo.critic_return_norm_gamma
        ),
        clip=cfg.ppo.critic_return_norm_clip,
        epsilon=cfg.ppo.critic_return_norm_epsilon,
    )


def _restore_reward_normalizer_from_checkpoint(
    reward_normalizer: DiscountedReturnNormalizer | None,
    ckpt: object,
    path: str | Path,
) -> None:
    if reward_normalizer is None:
        return
    if isinstance(ckpt, dict) and isinstance(
        ckpt.get("critic_return_normalizer"),
        dict,
    ):
        reward_normalizer.load_state_dict(ckpt["critic_return_normalizer"])
        return
    raise ValueError(
        f"{path} does not contain critic_return_normalizer; checkpoints trained "
        "with raw/symlog critic targets are not compatible with "
        "ppo.critic_return_norm=discounted_return_rms. Start a fresh run or set "
        "ppo.critic_return_norm: none for actor-only/manual migration."
    )


def _build_percentile_return_normalizer(
    cfg: RunConfig,
) -> PercentileReturnNormalizer | None:
    # The stateful EMA normalizer exists ONLY for the "ema" scope (DreamerV3
    # retnorm: a slow global percentile EMA). The "batch" scope keeps no state and
    # uses no EMA — it recomputes a fresh whole-rollout percentile spread each
    # update via _fresh_percentile_return_scale, so it builds no normalizer here.
    if (
        cfg.ppo.advantage_return_norm != "perc"
        or cfg.ppo.advantage_return_norm_scope != "ema"
    ):
        return None
    return PercentileReturnNormalizer(
        rate=cfg.ppo.advantage_return_norm_rate,
        perclo=cfg.ppo.advantage_return_norm_perclo,
        perchi=cfg.ppo.advantage_return_norm_perchi,
        limit=cfg.ppo.advantage_return_norm_limit,
    )


def _fresh_percentile_return_scale(
    returns: torch.Tensor,
    *,
    perclo: float,
    perchi: float,
    limit: float,
) -> float:
    """Fresh whole-rollout percentile-range scale for the "batch" retnorm scope.

    Stateless and EMA-free: each update divides the actor advantage by
    `max(limit, p95 - p5)` of the current rollout's returns. Same percentile and
    floor formula as the EMA normalizer's `.scale`, minus the smoothing.
    """
    arr = returns.detach().to(torch.float32).cpu().numpy()
    if arr.size == 0:
        return float(limit)
    lo = float(np.percentile(arr, perclo))
    hi = float(np.percentile(arr, perchi))
    return max(float(limit), hi - lo)


def _restore_percentile_return_normalizer_from_checkpoint(
    return_pct_normalizer: PercentileReturnNormalizer | None,
    ckpt: object,
) -> None:
    # Best-effort: the EMA only warms up actor-advantage scaling, so a checkpoint
    # without it (older runs, or one saved before this normalizer existed) just
    # re-warms from zero rather than erroring.
    if return_pct_normalizer is None:
        return
    if isinstance(ckpt, dict) and isinstance(
        ckpt.get("advantage_return_normalizer"), dict
    ):
        return_pct_normalizer.load_state_dict(ckpt["advantage_return_normalizer"])


def _normalizer_checkpoint_extra(
    reward_normalizer: DiscountedReturnNormalizer | None,
    return_pct_normalizer: PercentileReturnNormalizer | None = None,
) -> dict[str, object] | None:
    extra: dict[str, object] = {}
    if reward_normalizer is not None:
        extra["critic_return_normalizer"] = reward_normalizer.state_dict()
    if return_pct_normalizer is not None:
        extra["advantage_return_normalizer"] = return_pct_normalizer.state_dict()
    return extra or None


def _compile_mode_for_model(model: OrbitPolicy, cfg: RunConfig) -> str | None:
    return cfg.run.compile_mode or None


def _rollout_compile_mode_for_model(_model: OrbitPolicy, cfg: RunConfig) -> str | None:
    if not cfg.rollout.compile_policy:
        return None
    return cfg.run.compile_mode or None


def _trajectory_record_refs(trajs: list[Trajectory]) -> list[TrajectoryRecordRef]:
    refs: list[TrajectoryRecordRef] = []
    for traj in trajs:
        refs.extend(getattr(traj, "record_refs", ()))
    return refs


def _record_ref_chunks_and_positions(
    refs: list[TrajectoryRecordRef],
) -> tuple[list[dict[str, object]], torch.Tensor]:
    chunks: list[dict[str, object]] = []
    chunk_offsets: dict[int, int] = {}
    positions: list[int] = []
    offset = 0
    for ref in refs:
        key = id(ref.chunk)
        if key not in chunk_offsets:
            chunk_offsets[key] = offset
            chunks.append(ref.chunk)
            offset += int(ref.chunk["planet_feats"].shape[0])
        positions.append(chunk_offsets[key] + int(ref.row))
    return chunks, torch.as_tensor(positions, dtype=torch.long)


def _cat_chunk_field(
    chunks: list[dict[str, object]],
    key: str,
    *,
    pad_width: int | None = None,
    fill: int | float | bool = 0,
) -> torch.Tensor | None:
    values = [chunk[key] for chunk in chunks]
    if any(value is None for value in values):
        return None
    tensors = [value for value in values if isinstance(value, torch.Tensor)]
    if len(tensors) != len(values):
        raise TypeError(f"record chunk field {key!r} is not a tensor")
    if pad_width is None:
        return torch.cat(tensors, dim=0)
    padded = []
    for tensor in tensors:
        current = int(tensor.shape[1])
        if current == pad_width:
            padded.append(tensor)
            continue
        out = tensor.new_full((tensor.shape[0], pad_width, *tensor.shape[2:]), fill)
        out[:, :current] = tensor
        padded.append(out)
    return torch.cat(padded, dim=0)


def _stack_chunk_field(
    chunks: list[dict[str, object]],
    positions: torch.Tensor,
    key: str,
    *,
    pad_width: int | None = None,
    fill: int | float | bool = 0,
) -> torch.Tensor | None:
    field = _cat_chunk_field(chunks, key, pad_width=pad_width, fill=fill)
    if field is None:
        return None
    return field.index_select(0, positions)


def _stack_encoded_record_refs(
    chunks: list[dict[str, object]],
    positions: torch.Tensor,
) -> dict[str, torch.Tensor | None]:
    inbound = [chunk["planet_inbound_feats"] for chunk in chunks]
    fleet_width = max(int(chunk["fleet_feats"].shape[1]) for chunk in chunks)
    if any(value is None for value in inbound):
        fleet_width = max(1, fleet_width)

    return {
        "global_feats": _stack_chunk_field(chunks, positions, "global_feats"),
        "planet_feats": _stack_chunk_field(chunks, positions, "planet_feats"),
        "planet_mask": _stack_chunk_field(chunks, positions, "planet_mask"),
        "planet_owned_mask": _stack_chunk_field(
            chunks,
            positions,
            "planet_owned_mask",
        ),
        "planet_ids": _stack_chunk_field(chunks, positions, "planet_ids"),
        "planet_garrison": _stack_chunk_field(chunks, positions, "planet_garrison"),
        "fleet_feats": _stack_chunk_field(
            chunks,
            positions,
            "fleet_feats",
            pad_width=fleet_width,
        ),
        "fleet_mask": _stack_chunk_field(
            chunks,
            positions,
            "fleet_mask",
            pad_width=fleet_width,
            fill=False,
        ),
        "fleet_target_planet_idx": _stack_chunk_field(
            chunks,
            positions,
            "fleet_target_planet_idx",
            pad_width=fleet_width,
            fill=-1,
        ),
        "planet_inbound_feats": _stack_chunk_field(
            chunks,
            positions,
            "planet_inbound_feats",
        ),
    }


def _stack_target_legal_record_refs(
    chunks: list[dict[str, object]],
    positions: torch.Tensor,
) -> torch.Tensor | None:
    if not any("target_legal_source_mask" in chunk for chunk in chunks):
        dense = _stack_chunk_field(chunks, positions, "target_legal_mask")
        return dense.bool() if dense is not None else None

    planets = None
    for chunk in chunks:
        if "target_legal_source_mask" in chunk:
            planets = int(chunk["planet_owned_mask"].shape[1])
            break
        dense = chunk["target_legal_mask"]
        if dense is not None and dense.numel() > 0:
            planets = int(dense.shape[1])
            break
    if planets is None:
        return None

    final = torch.empty(int(positions.numel()), planets, planets, dtype=torch.bool)
    offset = 0
    for chunk in chunks:
        chunk_rows = int(chunk["planet_feats"].shape[0])
        keep = (positions >= offset) & (positions < offset + chunk_rows)
        if not bool(keep.any()):
            offset += chunk_rows
            continue
        out_rows = torch.nonzero(keep, as_tuple=False).flatten()
        local_rows = positions.index_select(0, out_rows) - offset
        if "target_legal_source_mask" not in chunk:
            dense = chunk["target_legal_mask"].bool()
            if dense.numel() == 0:
                raise RuntimeError("chunked PPO trajectory records are missing target legality")
            final[out_rows] = dense.index_select(0, local_rows)
            offset += chunk_rows
            continue
        owned = chunk["planet_owned_mask"].bool() & chunk["planet_mask"].bool()
        expanded = torch.ones(int(local_rows.numel()), planets, planets, dtype=torch.bool)
        selected_owned = owned.index_select(0, local_rows)
        owned_rows, owned_cols = torch.nonzero(selected_owned, as_tuple=True)
        if owned_rows.numel():
            expanded[owned_rows, owned_cols] = False
        row_idx = chunk["target_legal_row_idx"].long()
        if row_idx.numel():
            source_idx = chunk["target_legal_source_idx"].long()
            row_to_output = torch.full((chunk_rows,), -1, dtype=torch.long)
            row_to_output[local_rows] = torch.arange(local_rows.numel(), dtype=torch.long)
            compact_rows = row_to_output.index_select(0, row_idx)
            compact_keep = compact_rows >= 0
            if bool(compact_keep.any()):
                expanded[compact_rows[compact_keep], source_idx[compact_keep]] = chunk[
                    "target_legal_source_mask"
                ].bool()[compact_keep]
        final[out_rows] = expanded
        offset += chunk_rows
    return final


def _stack_target_legal_record_ref_fields(
    chunks: list[dict[str, object]],
    positions: torch.Tensor,
) -> dict[str, torch.Tensor | None]:
    if not chunks:
        return {"target_legal_mask": None}
    if not all("target_legal_source_mask" in chunk for chunk in chunks):
        return {"target_legal_mask": _stack_target_legal_record_refs(chunks, positions)}

    planet_owned = _stack_chunk_field(chunks, positions, "planet_owned_mask")
    planet_mask = _stack_chunk_field(chunks, positions, "planet_mask")
    if planet_owned is None or planet_mask is None:
        return {"target_legal_mask": _stack_target_legal_record_refs(chunks, positions)}
    owned = planet_owned.bool() & planet_mask.bool()

    row_parts: list[torch.Tensor] = []
    source_parts: list[torch.Tensor] = []
    mask_parts: list[torch.Tensor] = []
    offset = 0
    for chunk in chunks:
        chunk_rows = int(chunk["planet_feats"].shape[0])
        keep = (positions >= offset) & (positions < offset + chunk_rows)
        if not bool(keep.any()):
            offset += chunk_rows
            continue
        out_rows = torch.nonzero(keep, as_tuple=False).flatten()
        local_rows = positions.index_select(0, out_rows) - offset
        row_idx = chunk["target_legal_row_idx"].long()
        if row_idx.numel():
            row_to_output = torch.full((chunk_rows,), -1, dtype=torch.long)
            row_to_output[local_rows] = out_rows
            compact_rows = row_to_output.index_select(0, row_idx)
            compact_keep = compact_rows >= 0
            if bool(compact_keep.any()):
                row_parts.append(compact_rows[compact_keep])
                source_parts.append(
                    chunk["target_legal_source_idx"].long()[compact_keep]
                )
                mask_parts.append(
                    chunk["target_legal_source_mask"].bool()[compact_keep]
                )
        offset += chunk_rows

    if row_parts:
        row_idx = torch.cat(row_parts, dim=0)
        source_idx = torch.cat(source_parts, dim=0)
        source_mask = torch.cat(mask_parts, dim=0)
        order = torch.argsort(row_idx, stable=True)
        row_idx = row_idx.index_select(0, order)
        source_idx = source_idx.index_select(0, order)
        source_mask = source_mask.index_select(0, order)
    else:
        planets = int(owned.shape[1])
        row_idx = torch.empty(0, dtype=torch.long)
        source_idx = torch.empty(0, dtype=torch.long)
        source_mask = torch.empty(0, planets, dtype=torch.bool)
    counts = torch.bincount(row_idx, minlength=int(positions.numel()))
    row_offsets = torch.empty(int(positions.numel()) + 1, dtype=torch.long)
    row_offsets[0] = 0
    row_offsets[1:] = counts.cumsum(0)
    return {
        "target_legal_mask": None,
        "target_legal_source_owned_mask": owned,
        "target_legal_row_offsets": row_offsets,
        "target_legal_row_idx": row_idx,
        "target_legal_source_idx": source_idx,
        "target_legal_source_mask": source_mask,
    }


def _stack_source_actor_record_ref_fields(
    chunks: list[dict[str, object]],
    positions: torch.Tensor,
) -> dict[str, torch.Tensor]:
    if not chunks or not all("source_row_idx" in chunk for chunk in chunks):
        return {}

    row_parts: list[torch.Tensor] = []
    col_parts: list[torch.Tensor] = []
    launch_parts: list[torch.Tensor] = []
    raw_launch_parts: list[torch.Tensor] = []
    target_parts: list[torch.Tensor] = []
    fraction_parts: list[torch.Tensor] = []
    log_prob_parts: list[torch.Tensor] = []
    legal_parts: list[torch.Tensor] = []
    have_log_prob = all("source_log_prob" in chunk for chunk in chunks)
    have_legal = all("source_target_legal_mask" in chunk for chunk in chunks)
    offset = 0
    for chunk in chunks:
        chunk_rows = int(chunk["planet_feats"].shape[0])
        keep = (positions >= offset) & (positions < offset + chunk_rows)
        if not bool(keep.any()):
            offset += chunk_rows
            continue
        out_rows = torch.nonzero(keep, as_tuple=False).flatten()
        local_rows = positions.index_select(0, out_rows) - offset
        source_row_idx = chunk["source_row_idx"].long()
        if source_row_idx.numel():
            row_to_output = torch.full((chunk_rows,), -1, dtype=torch.long)
            row_to_output[local_rows] = out_rows
            actor_rows = row_to_output.index_select(0, source_row_idx)
            actor_keep = actor_rows >= 0
            if bool(actor_keep.any()):
                row_parts.append(actor_rows[actor_keep])
                col_parts.append(chunk["source_col_idx"].long()[actor_keep])
                launch_parts.append(chunk["source_launch"].float()[actor_keep])
                raw_launch_parts.append(chunk["source_raw_launch"].float()[actor_keep])
                target_parts.append(chunk["source_target_idx"].long()[actor_keep])
                fraction_parts.append(chunk["source_fraction"].float()[actor_keep])
                if have_log_prob:
                    log_prob_parts.append(chunk["source_log_prob"].float()[actor_keep])
                if have_legal:
                    legal_parts.append(
                        chunk["source_target_legal_mask"].bool()[actor_keep]
                    )
        offset += chunk_rows

    if row_parts:
        row_idx = torch.cat(row_parts, dim=0)
        order = torch.argsort(row_idx, stable=True)
        row_idx = row_idx.index_select(0, order)
        out = {
            "actor_source_row_idx": row_idx,
            "actor_source_col_idx": torch.cat(col_parts, dim=0).index_select(0, order),
            "actor_launch": torch.cat(launch_parts, dim=0).index_select(0, order),
            "actor_raw_launch": torch.cat(raw_launch_parts, dim=0).index_select(0, order),
            "actor_target_idx": torch.cat(target_parts, dim=0).index_select(0, order),
            "actor_fraction": torch.cat(fraction_parts, dim=0).index_select(0, order),
        }
        if have_log_prob:
            out["actor_old_log_prob"] = torch.cat(log_prob_parts, dim=0).index_select(
                0,
                order,
            )
        if have_legal:
            out["actor_target_legal_mask"] = torch.cat(legal_parts, dim=0).index_select(
                0,
                order,
            )
    else:
        out = {
            "actor_source_row_idx": torch.empty(0, dtype=torch.long),
            "actor_source_col_idx": torch.empty(0, dtype=torch.long),
            "actor_launch": torch.empty(0, dtype=torch.float32),
            "actor_raw_launch": torch.empty(0, dtype=torch.float32),
            "actor_target_idx": torch.empty(0, dtype=torch.long),
            "actor_fraction": torch.empty(0, dtype=torch.float32),
        }
        if have_log_prob:
            out["actor_old_log_prob"] = torch.empty(0, dtype=torch.float32)
        if have_legal:
            planets = int(chunks[0]["planet_feats"].shape[1])
            out["actor_target_legal_mask"] = torch.empty(0, planets, dtype=torch.bool)

    counts = torch.bincount(
        out["actor_source_row_idx"],
        minlength=int(positions.numel()),
    )
    out["actor_source_row_offsets"] = torch.cat(
        (
            torch.zeros(1, dtype=torch.long),
            counts.cumsum(0).to(torch.long),
        ),
        dim=0,
    )
    return out


def _stack_encoded(trajs: list[Trajectory]) -> dict[str, torch.Tensor | None]:
    """Walk every (traj, step) once and emit stacked EncodedObs tensors.

    Encoder-only: actor-side records (launch / target_idx / fraction /
    old_log_prob / owned_mask) are added by `_stack_trajectories`, which lets
    `_pretrain_value_batch` skip them entirely.
    """
    refs = _trajectory_record_refs(trajs)
    if refs:
        if any(t.encoded for t in trajs):
            raise RuntimeError("cannot mix chunked and row-wise trajectory records")
        chunks, positions = _record_ref_chunks_and_positions(refs)
        return _stack_encoded_record_refs(chunks, positions)

    gf, pf, pm, pom, pid, pg, ff, fm, ft, pi = [], [], [], [], [], [], [], [], [], []
    fleet_width = 0
    saw_without_inbound = False
    for t in trajs:
        for e in t.encoded:
            saw_without_inbound = saw_without_inbound or e.planet_inbound_feats is None
            fleet_width = max(fleet_width, int(e.fleet_feats.shape[0]))
            gf.append(e.global_feats)
            pf.append(e.planet_feats)
            pm.append(e.planet_mask)
            pom.append(e.planet_owned_mask)
            pid.append(e.planet_ids)
            pg.append(e.planet_garrison)
            ff.append(e.fleet_feats)
            fm.append(e.fleet_mask)
            ft.append(e.fleet_target_planet_idx)
            pi.append(e.planet_inbound_feats)
    if saw_without_inbound:
        fleet_width = max(1, fleet_width)

    def pad_fleet(t: torch.Tensor, fill: float | bool | int = 0) -> torch.Tensor:
        current = int(t.shape[0])
        if current == fleet_width:
            return t
        out = t.new_full((fleet_width, *t.shape[1:]), fill)
        out[:current] = t
        return out

    return {
        "global_feats": None if any(g is None for g in gf) else torch.stack(gf),
        "planet_feats": torch.stack(pf),
        "planet_mask": torch.stack(pm),
        "planet_owned_mask": torch.stack(pom),
        "planet_ids": torch.stack(pid),
        "planet_garrison": torch.stack(pg),
        "fleet_feats": torch.stack([pad_fleet(t) for t in ff]),
        "fleet_mask": torch.stack([pad_fleet(t, fill=False) for t in fm]),
        "fleet_target_planet_idx": None
        if any(t is None for t in ft)
        else torch.stack([pad_fleet(t, fill=-1) for t in ft]),
        "planet_inbound_feats": None if any(t is None for t in pi) else torch.stack(pi),
    }


def _normalized_reward_arrays(
    trajs: list[Trajectory],
    reward_normalizer: DiscountedReturnNormalizer | None,
) -> list[np.ndarray]:
    rewards_by_traj = [np.asarray(t.reward, dtype=np.float32) for t in trajs]
    if reward_normalizer is not None:
        rewards_by_traj = reward_normalizer.normalize_episodes(rewards_by_traj)
    return rewards_by_traj


def _gae_batch_fields(
    rewards_by_traj: list[np.ndarray],
    values: np.ndarray,
    *,
    gamma: float,
    gae_lambda: float,
    value_gae_lambda: float | None,
    critic_mtp_horizon: int,
) -> dict[str, torch.Tensor]:
    target_lam = gae_lambda if value_gae_lambda is None else value_gae_lambda
    mtp_h = max(1, int(critic_mtp_horizon))
    total_steps = sum(len(rewards) for rewards in rewards_by_traj)
    if int(values.shape[0]) != total_steps:
        values = np.zeros(total_steps, dtype=np.float32)
    advs_all, rets_all, mtp_all, mtp_mask_all = [], [], [], []
    offset = 0
    for rewards in rewards_by_traj:
        horizon = len(rewards)
        traj_values = values[offset : offset + horizon]
        offset += horizon
        adv, ret = compute_gae(rewards, traj_values, gamma, gae_lambda)
        if target_lam != gae_lambda:
            _target_adv, ret = compute_gae(rewards, traj_values, gamma, target_lam)
        advs_all.append(adv)
        rets_all.append(ret)
        mtp = np.zeros((horizon, mtp_h), dtype=np.float32)
        mtp_mask = np.zeros((horizon, mtp_h), dtype=np.bool_)
        for h in range(mtp_h):
            valid = max(0, horizon - h)
            if valid:
                mtp[:valid, h] = ret[h:]
                mtp_mask[:valid, h] = True
        mtp_all.append(mtp)
        mtp_mask_all.append(mtp_mask)

    advs = (
        torch.from_numpy(np.concatenate(advs_all)).float()
        if advs_all
        else torch.empty(0, dtype=torch.float32)
    )
    rets = (
        torch.from_numpy(np.concatenate(rets_all)).float()
        if rets_all
        else torch.empty(0, dtype=torch.float32)
    )
    return_mtp = (
        torch.from_numpy(np.concatenate(mtp_all)).float()
        if mtp_all
        else torch.empty(0, mtp_h, dtype=torch.float32)
    )
    return_mtp_mask = (
        torch.from_numpy(np.concatenate(mtp_mask_all)).bool()
        if mtp_mask_all
        else torch.empty(0, mtp_h, dtype=torch.bool)
    )
    values_t = torch.from_numpy(values).float()
    if values_t.numel() != rets.numel():
        values_t = torch.zeros_like(rets)
    return {
        "advantage": advs,
        "return": rets,
        "return_mtp": return_mtp,
        "return_mtp_mask": return_mtp_mask,
        "value": values_t,
        "raw_advantage_abs_mean": torch.tensor(
            float(advs.abs().mean()) if advs.numel() else 0.0,
            dtype=torch.float32,
        ),
    }


def _refresh_batch_advantages_from_values(
    batch: dict[str, torch.Tensor],
    values: torch.Tensor,
    *,
    gamma: float,
    gae_lambda: float,
    value_gae_lambda: float | None,
    critic_mtp_horizon: int,
) -> None:
    lengths = batch["_trajectory_lengths"].to(dtype=torch.long).tolist()
    rewards_flat = batch["_normalized_rewards"].detach().to(torch.float32).cpu().numpy()
    rewards_by_traj = []
    offset = 0
    for length in lengths:
        end = offset + int(length)
        rewards_by_traj.append(rewards_flat[offset:end])
        offset = end
    fields = _gae_batch_fields(
        rewards_by_traj,
        values.detach().to(torch.float32).cpu().numpy(),
        gamma=gamma,
        gae_lambda=gae_lambda,
        value_gae_lambda=value_gae_lambda,
        critic_mtp_horizon=critic_mtp_horizon,
    )
    batch.update(fields)
    batch["values_computed"] = torch.tensor(True, dtype=torch.bool)


def _stack_trajectories(
    trajs: list[Trajectory],
    gamma: float,
    gae_lambda: float,
    value_gae_lambda: float | None = None,
    critic_mtp_horizon: int = 1,
    reward_normalizer: DiscountedReturnNormalizer | None = None,
    include_old_log_prob: bool = True,
) -> dict[str, torch.Tensor]:
    """Flatten per-step records into one PPO batch with optional decoupled GAE."""
    batch = _stack_encoded(trajs)
    refs = _trajectory_record_refs(trajs)

    if refs:
        if any(t.launch or t.target_idx or t.fraction or t.value for t in trajs):
            raise RuntimeError("cannot mix chunked and row-wise trajectory records")
        chunks, positions = _record_ref_chunks_and_positions(refs)
        launch = _stack_chunk_field(chunks, positions, "launch")
        target_idx = _stack_chunk_field(chunks, positions, "target_idx")
        fraction = _stack_chunk_field(chunks, positions, "fraction")
        value = _stack_chunk_field(chunks, positions, "value")
        target_legal_fields = _stack_target_legal_record_ref_fields(chunks, positions)
        target_legal_mask = target_legal_fields.get("target_legal_mask")
        values_computed = all(bool(chunk.get("values_computed", True)) for chunk in chunks)
        if (
            launch is None
            or target_idx is None
            or fraction is None
            or (values_computed and value is None)
            or (
                target_legal_mask is None
                and target_legal_fields.get("target_legal_source_mask") is None
            )
        ):
            raise RuntimeError("chunked PPO trajectory records are missing required fields")
        batch["launch"] = launch.float()
        batch["target_idx"] = target_idx.long()
        batch["fraction"] = fraction.float()
        if include_old_log_prob:
            old_log_prob = _stack_chunk_field(chunks, positions, "log_prob")
            old_log_prob_computed = old_log_prob is not None and all(
                bool(chunk.get("old_log_prob_computed", False)) for chunk in chunks
            )
            batch["old_log_prob"] = (
                old_log_prob.float() if old_log_prob is not None else torch.zeros_like(launch)
            )
            batch["old_log_prob_computed"] = torch.tensor(
                old_log_prob_computed,
                dtype=torch.bool,
            )
        else:
            batch["old_log_prob"] = torch.zeros_like(batch["launch"])
            batch["old_log_prob_computed"] = torch.tensor(False)
        batch["values_computed"] = torch.tensor(values_computed, dtype=torch.bool)
        batch["owned_mask"] = batch["planet_owned_mask"].bool() & batch["planet_mask"].bool()
        batch.update(_stack_source_actor_record_ref_fields(chunks, positions))
        if target_legal_mask is not None:
            batch["target_legal_mask"] = target_legal_mask.bool()
        else:
            batch["target_legal_mask"] = None
            batch["target_legal_source_owned_mask"] = target_legal_fields[
                "target_legal_source_owned_mask"
            ].bool()
            batch["target_legal_row_offsets"] = target_legal_fields[
                "target_legal_row_offsets"
            ].long()
            batch["target_legal_row_idx"] = target_legal_fields[
                "target_legal_row_idx"
            ].long()
            batch["target_legal_source_idx"] = target_legal_fields[
                "target_legal_source_idx"
            ].long()
            batch["target_legal_source_mask"] = target_legal_fields[
                "target_legal_source_mask"
            ].bool()
        if value is None:
            all_values = np.zeros(int(positions.numel()), dtype=np.float32)
        else:
            all_values = value.detach().to(torch.float32).cpu().numpy()
    else:
        launch, tidx, frac, lp, owned, target_legal, values_flat = [], [], [], [], [], [], []
        for t in trajs:
            launch.extend(t.launch)
            tidx.extend(t.target_idx)
            frac.extend(t.fraction)
            if include_old_log_prob:
                lp.extend(t.log_prob)
            owned.extend(t.owned_mask)
            target_legal.extend(t.target_legal_mask)
            values_flat.extend(t.value)
        batch["launch"] = torch.stack(launch).float()
        batch["target_idx"] = torch.stack(tidx).long()
        batch["fraction"] = torch.stack(frac).float()
        old_log_prob_computed = include_old_log_prob and len(lp) == len(launch)
        batch["old_log_prob"] = (
            torch.stack(lp).float()
            if old_log_prob_computed
            else torch.zeros_like(batch["launch"])
        )
        batch["old_log_prob_computed"] = torch.tensor(
            old_log_prob_computed,
            dtype=torch.bool,
        )
        batch["values_computed"] = torch.tensor(bool(values_flat), dtype=torch.bool)
        batch["owned_mask"] = torch.stack(owned).bool()
        batch["target_legal_mask"] = torch.stack(target_legal).bool()

        # Single global CPU pull of every per-step value across the batch.
        if values_flat:
            all_values = torch.stack(values_flat).detach().to(torch.float32).cpu().numpy()
        else:
            all_values = np.zeros(0, dtype=np.float32)

    rewards_by_traj = _normalized_reward_arrays(trajs, reward_normalizer)
    batch["_trajectory_lengths"] = torch.tensor(
        [len(rewards) for rewards in rewards_by_traj],
        dtype=torch.long,
    )
    batch["_normalized_rewards"] = (
        torch.from_numpy(np.concatenate(rewards_by_traj)).float()
        if rewards_by_traj
        else torch.empty(0, dtype=torch.float32)
    )
    batch.update(
        _gae_batch_fields(
            rewards_by_traj,
            all_values,
            gamma=gamma,
            gae_lambda=gae_lambda,
            value_gae_lambda=value_gae_lambda,
            critic_mtp_horizon=critic_mtp_horizon,
        )
    )
    return batch


def _trim_ppo_batch_fleet_width(
    batch: dict[str, torch.Tensor],
    *,
    pad_to_bucket: bool = False,
) -> dict[str, torch.Tensor]:
    """Trim fleet tensors to a stable compile bucket before PPO staging.

    Fleet tensors can dominate PPO batch storage. Keeping the full rollout batch
    on CPU avoids persistent VRAM allocation; trimming to a small set
    of widths also bounds the number of `torch.compile(dynamic=False)` graph
    specializations while reducing both staged minibatch memory and compute.
    """
    fleet_mask = batch.get("fleet_mask")
    fleet_feats = batch.get("fleet_feats")
    if fleet_mask is None or fleet_feats is None or fleet_mask.dim() != 2:
        return batch
    current = int(fleet_mask.shape[1])
    if batch.get("planet_inbound_feats") is not None:
        width = 0
    else:
        if current <= 1:
            return batch
        width = (
            bucket_fleet_width(active_fleet_width(fleet_mask))
            if pad_to_bucket
            else bucket_fleet_width(active_fleet_width(fleet_mask), current)
        )
    if width == current:
        return batch

    def resize_fleet(t: torch.Tensor, *, fill: int | float | bool = 0) -> torch.Tensor:
        if width < current:
            return t[:, :width].contiguous()
        out = t.new_full((t.shape[0], width, *t.shape[2:]), fill)
        out[:, :current] = t
        return out.contiguous()

    batch = dict(batch)
    batch["fleet_feats"] = resize_fleet(fleet_feats)
    batch["fleet_mask"] = resize_fleet(fleet_mask, fill=False)
    if batch.get("fleet_target_planet_idx") is not None:
        batch["fleet_target_planet_idx"] = resize_fleet(
            batch["fleet_target_planet_idx"],
            fill=-1,
        )
    return batch


def _slice_encoded_obs_to_device(
    batch: dict[str, torch.Tensor],
    mb,
    device: torch.device,
) -> EncodedObs:
    """Build one model-device minibatch view for non-training diagnostics."""
    return EncodedObs(
        planet_feats=batch["planet_feats"][mb].to(device, non_blocking=True),
        planet_mask=batch["planet_mask"][mb].to(device, non_blocking=True),
        planet_owned_mask=batch["planet_owned_mask"][mb].to(device, non_blocking=True),
        planet_ids=batch["planet_ids"][mb].to(device, non_blocking=True),
        planet_garrison=batch["planet_garrison"][mb].to(device, non_blocking=True),
        fleet_feats=batch["fleet_feats"][mb].to(device, non_blocking=True),
        fleet_mask=batch["fleet_mask"][mb].to(device, non_blocking=True),
        global_feats=None
        if batch.get("global_feats") is None
        else batch["global_feats"][mb].to(device, non_blocking=True),
        fleet_target_planet_idx=None
        if batch.get("fleet_target_planet_idx") is None
        else batch["fleet_target_planet_idx"][mb].to(device, non_blocking=True),
        planet_inbound_feats=None
        if batch.get("planet_inbound_feats") is None
        else batch["planet_inbound_feats"][mb].to(device, non_blocking=True),
    )


def _explained_variance(pred: torch.Tensor, target: torch.Tensor) -> float:
    """CleanRL-style EV: 1 - Var[target - pred] / Var[target]."""
    with torch.no_grad():
        pred_f = pred.detach().float().flatten()
        target_f = target.detach().float().flatten()
        if target_f.numel() == 0:
            return float("nan")
        var_y = torch.var(target_f, unbiased=False)
        if var_y <= 0:
            return float("nan")
        ev = 1.0 - torch.var(target_f - pred_f, unbiased=False) / var_y
        return float(ev.cpu())


def _pretrain_value_batch(
    trajs: list[Trajectory],
    gamma: float,
    gae_lambda: float,
    critic_mtp_horizon: int = 1,
    reward_normalizer: DiscountedReturnNormalizer | None = None,
) -> dict[str, torch.Tensor]:
    """Critic-target = configured bootstrapped GAE/lambda return.

    Encoder fields only — `value_only_update` doesn't read the actor-side
    records, so we skip stacking and host→device-copying them.
    """
    batch = _stack_encoded(trajs)
    mtp_h = max(1, int(critic_mtp_horizon))
    rets_all, mtp_all, mtp_mask_all = [], [], []
    rewards_by_traj = [np.asarray(t.reward, dtype=np.float32) for t in trajs]
    if reward_normalizer is not None:
        rewards_by_traj = reward_normalizer.normalize_episodes(rewards_by_traj)
    for t, rewards in zip(trajs, rewards_by_traj, strict=True):
        values = (
            torch.stack(t.value).detach().to(torch.float32).cpu().numpy()
            if t.value
            else np.zeros_like(rewards, dtype=np.float32)
        )
        _adv, ret = compute_gae(rewards, values, gamma, gae_lambda)
        rets_all.append(ret)
        horizon = len(rewards)
        mtp = np.zeros((horizon, mtp_h), dtype=np.float32)
        mtp_mask = np.zeros((horizon, mtp_h), dtype=np.bool_)
        for h in range(mtp_h):
            valid = max(0, horizon - h)
            if valid:
                mtp[:valid, h] = ret[h:]
                mtp_mask[:valid, h] = True
        mtp_all.append(mtp)
        mtp_mask_all.append(mtp_mask)
    batch["return"] = torch.from_numpy(np.concatenate(rets_all)).float()
    batch["return_mtp"] = torch.from_numpy(np.concatenate(mtp_all)).float()
    batch["return_mtp_mask"] = torch.from_numpy(np.concatenate(mtp_mask_all)).bool()
    return batch


def pretrain_value(
    cfg: RunConfig,
    model: OrbitPolicy,
    optimizer: torch.optim.Optimizer,
    logger: TBLogger,
    device: torch.device,
    vec: VecEnv,
    reward_normalizer: DiscountedReturnNormalizer | None = None,
) -> None:
    """VAPO-style cold-start: regress V(s) onto trajectory outcome.

    Behavior policy is `cfg.ppo.pretrain_behavior` (defaults to
    `heuristic`). The actor parameters update too — that's fine, the
    critic shares the encoder and we only run a few hundred steps before
    PPO takes over. Uses the same batched rollout as PPO so cold-start
    isn't an order of magnitude slower than the hot loop.
    """
    if cfg.ppo.pretrain_updates <= 0:
        return
    behavior = BUILTIN.get(cfg.ppo.pretrain_behavior)
    if behavior is None:
        raise ValueError(f"unknown pretrain_behavior={cfg.ppo.pretrain_behavior!r}")
    # Wrap the heuristic baseline as a frozen-style OpponentSlot so
    # rollout_episodes_batched can use it. Behavior policies are pure
    # Python (no model forward), so they fall into the per-snapshot
    # branch — exactly what we want.
    behavior_slot = OpponentSlot(name=cfg.ppo.pretrain_behavior, agent=behavior)
    opponents_per_env = [
        [behavior_slot] * (cfg.game.num_players - 1) for _ in range(cfg.rollout.num_envs)
    ]

    for step in range(cfg.ppo.pretrain_updates):
        # Round episode count up to the nearest num_envs batch.
        batches = max(1, math.ceil(cfg.ppo.pretrain_episodes / cfg.rollout.num_envs))
        trajs: list[Trajectory] = []
        for batch_idx in range(batches):
            learner_seats = alternating_learner_seats(
                cfg.rollout.num_envs,
                cfg.game.num_players,
                offset=step * batches + batch_idx,
            )
            trajs.extend(
                rollout_episodes_batched(
                    model,
                    vec,
                    opponents_per_env,
                    num_players=cfg.game.num_players,
                    learner_seat=learner_seats,
                    device=str(device),
                    deterministic=False,
                    reward_cfg=cfg.reward,
                    compile_mode=_rollout_compile_mode_for_model(model, cfg),
                    compile_fleet_width=cfg.rollout.compile_fleet_width,
                    policy_graph_rows=cfg.rollout.num_envs,
                    snapshot_compile_rows=cfg.rollout.snapshot_compile_rows,
                    learner_action_agent=behavior,
                )
            )
        compile_mode = _compile_mode_for_model(model, cfg)
        batch = _trim_ppo_batch_fleet_width(
            _pretrain_value_batch(
                trajs,
                gamma=cfg.ppo.gamma,
                gae_lambda=(
                    cfg.ppo.gae_lambda
                    if cfg.ppo.value_gae_lambda is None
                    else cfg.ppo.value_gae_lambda
                ),
                critic_mtp_horizon=cfg.model.critic_mtp_horizon,
                reward_normalizer=reward_normalizer,
            ),
            pad_to_bucket=compile_mode is not None and device.type == "cuda",
        )
        loss = value_only_update(
            model,
            optimizer,
            batch,
            epochs=1,
            minibatch_size=cfg.optim.minibatch_size,
            grad_clip=cfg.optim.grad_clip,
            compile_mode=compile_mode,
        )
        rets = batch["return"].cpu().numpy()
        # Explained variance: 1 - Var(target - pred)/Var(target). Tracked
        # because raw value loss can be misleading when the target
        # distribution shifts (e.g., as the behavior policy gets crushed).
        # Minibatched: a full-batch forward over `pretrain_episodes ×
        # episode_steps` samples blows up the [B, heads, tokens, tokens]
        # attention tensor (tokens ≈ planets + active fleets + summary tokens).
        n = batch["planet_feats"].shape[0]
        mb = cfg.optim.minibatch_size
        autocast_enabled = device.type == "cuda"
        with (
            torch.no_grad(),
            torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=autocast_enabled),
        ):
            chunks = [
                model(_slice_encoded_obs_to_device(batch, slice(s, s + mb), device)).value
                for s in range(0, n, mb)
            ]
            preds = torch.cat(chunks).float().cpu().numpy()
        var_y = float(np.var(rets)) + 1e-9
        ev = 1.0 - float(np.var(rets - preds)) / var_y
        logger.scalars("pretrain", {"value_loss": loss, "explained_variance": ev}, step)


def _seat_names(learner_seat: int, slots: list[OpponentSlot]) -> list[str]:
    """Build the seat-ordered identity list for one game.

    The learner sits at `learner_seat`; the remaining seats are filled by
    `slots` in order. For Elo we want the *identity* per seat — a self-
    play seat shares LEARNER_NAME with the actual learner.
    """
    names: list[str] = []
    op_ix = 0
    num_seats = len(slots) + 1
    for seat in range(num_seats):
        if seat == learner_seat:
            names.append(LEARNER_NAME)
        else:
            names.append(slots[op_ix].name)
            op_ix += 1
    return names


def _save_ppo_checkpoint(
    model: OrbitPolicy,
    path: Path,
    reward_normalizer: DiscountedReturnNormalizer | None = None,
    return_pct_normalizer: PercentileReturnNormalizer | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"model": model.state_dict(), "config": model.cfg.to_dict()}
    if reward_normalizer is not None:
        payload["critic_return_normalizer"] = reward_normalizer.state_dict()
    if return_pct_normalizer is not None:
        payload["advantage_return_normalizer"] = return_pct_normalizer.state_dict()
    torch.save(payload, path)


def _value_pretrain_params(model: OrbitPolicy) -> list[torch.nn.Parameter]:
    """Params that *actually* get gradient from value-only loss.

    Includes the encoder (shared backbone), prefix tokens (actor_token
    feeds the encoder self-attention so h_critic depends on it; critic_token
    feeds the value head directly), and the value head. Excludes the actor
    heads (target_query/key, categorical action heads, fraction alpha/beta heads) — they receive zero
    gradient from the value loss, and including them would let AdamW's
    weight-decay pull them toward zero with no learning signal, leaving PPO
    to start from a worse-than-init policy.
    """
    encoder = [model.global_embed, model.planet_embed, model.fleet_embed, *model.layers]
    if model.fleet_tokenizer is not None:
        encoder.append(model.fleet_tokenizer)
    if model.destination_fleet_conditioner is not None:
        encoder.append(model.destination_fleet_conditioner)
    value = [model.value_head]
    params: list[torch.nn.Parameter] = [
        model.actor_token,
        model.critic_token,
        model.global_token,
    ]
    for m in encoder + value:
        params.extend(m.parameters())
    return params


def _build_training_vec(cfg: RunConfig, num_players: int, num_envs: int) -> VecEnv:
    if cfg.rollout.env_backend not in {"kaggle", "numpy", "numpy_mp", "rust"}:
        raise ValueError(f"unknown rollout.env_backend: {cfg.rollout.env_backend!r}")
    if num_envs <= 0:
        raise ValueError("num_envs must be positive")
    vec_kwargs = dict(
        num_envs=num_envs,
        num_players=num_players,
        episode_steps=cfg.game.episode_steps,
        ship_speed=cfg.game.ship_speed,
    )
    if cfg.rollout.env_backend == "numpy":
        return NumpyVecEnv(
            **vec_kwargs,
            replay_env_idx=None,
            random_seed=cfg.run.seed,
        )
    if cfg.rollout.env_backend == "numpy_mp":
        return ShardedNumpyVecEnv(
            **vec_kwargs,
            replay_env_idx=None,
            num_workers=cfg.rollout.num_workers,
            random_seed=cfg.run.seed,
        )
    if cfg.rollout.env_backend == "rust":
        from .rust_env import RustVecEnv

        return RustVecEnv(
            **vec_kwargs,
            replay_env_idx=None,
            random_seed=cfg.run.seed,
            strict_target_legality=cfg.rollout.strict_target_legality,
        )
    return VecEnv(**vec_kwargs, replay_env_idx=0)


def _train_num_players(cfg: RunConfig) -> tuple[int, ...]:
    return tuple(cfg.game.train_num_players or [cfg.game.num_players])


def _format_episode_counts(
    total_games: int,
    train_num_players: tuple[int, ...],
    rng: random.Random,
) -> dict[int, int]:
    """Split one PPO update's games uniformly across training formats."""
    if total_games <= 0:
        return {players: 0 for players in train_num_players}
    if not train_num_players:
        raise ValueError("train_num_players must be non-empty")
    formats = list(train_num_players)
    rng.shuffle(formats)
    base, extra = divmod(int(total_games), len(formats))
    return {players: base + (1 if idx < extra else 0) for idx, players in enumerate(formats)}


def _training_vec_counts(cfg: RunConfig) -> dict[int, int]:
    counts = _format_episode_counts(
        cfg.rollout.num_envs,
        _train_num_players(cfg),
        random.Random(cfg.run.seed + 0x5E1F),
    )
    if cfg.ppo.pretrain_updates > 0:
        counts[cfg.game.num_players] = max(
            cfg.rollout.num_envs,
            counts.get(cfg.game.num_players, 0),
        )
    return counts


def _policy_compile_rows_for_rollout(
    cfg: RunConfig,
    *,
    num_envs: int,
    num_players: int,
    bucket_multiple: int = 64,
) -> int:
    full_rows = int(num_envs) * int(num_players)
    if cfg.opponents.mode == "no_builtins":
        available_weight = cfg.opponents.current_learner_prob + cfg.opponents.active_pool_prob
        current_prob = (
            cfg.opponents.current_learner_prob / available_weight
            if available_weight > 0.0
            else 1.0
        )
    elif cfg.opponents.mode == "league":
        current_prob = cfg.opponents.self_play_prob
    else:
        current_prob = 0.0
    expected_rows = int(num_envs) * (1.0 + current_prob * max(0, int(num_players) - 1))
    bucket = max(1, int(bucket_multiple))
    capped = int(math.ceil(expected_rows / bucket) * bucket)
    return min(full_rows, max(1, capped))


def _policy_compile_rows_for_sampled_rollout(
    opponents_per_env: Sequence[Sequence[OpponentSlot]],
    *,
    learner_seats: Sequence[int],
    num_players: int,
    bucket_multiple: int = 64,
) -> int:
    """Static graph row cap from the actual first-step learner-model rows.

    The learner forward covers each env's learner seat plus any opponent seat
    assigned to the current learner identity. Rounding keeps compile-shape
    variety bounded while avoiding the larger conservative expectation used
    before opponents are sampled.
    """
    rows = 0
    for env_idx, slots in enumerate(opponents_per_env):
        learner_seat = int(learner_seats[env_idx])
        rows += 1
        slot_idx = 0
        for seat in range(int(num_players)):
            if seat == learner_seat:
                continue
            if slot_idx < len(slots) and slots[slot_idx].name == LEARNER_NAME:
                rows += 1
            slot_idx += 1
    bucket = max(1, int(bucket_multiple))
    return max(1, int(math.ceil(rows / bucket) * bucket))


def _ppo_minibatch_size_for_fleet_width(cfg: RunConfig, fleet_width: int) -> int:
    """Keep high-fleet compiled PPO batches under the VRAM cliff."""
    size = int(cfg.optim.minibatch_size)
    if fleet_width > 1024:
        size = min(size, 2048)
    return max(1, size)


def train_one_run(cfg: RunConfig, load_weights: str | None = None) -> dict:
    set_seed(cfg.run.seed)
    if cfg.run.torch_num_threads > 0:
        torch.set_num_threads(cfg.run.torch_num_threads)
        with suppress(RuntimeError):
            torch.set_num_interop_threads(max(1, cfg.run.torch_num_threads))
    if cfg.run.device != "cuda":
        raise ValueError("training is CUDA-only; set run.device: cuda")
    if not torch.cuda.is_available():
        raise RuntimeError("training requires CUDA")
    device = torch.device("cuda")
    # TF32 tensor cores for residual float32 matmuls only; the model runs in
    # bf16 (explicit master-cast below + autocast), so attention/Linear stay
    # bf16 and are unaffected — this just upgrades the leftover fp32 GEMMs.
    torch.set_float32_matmul_precision("high")
    model = _build_model(cfg).to(device)
    # parameter-golf fp32-master pattern: cast everything to bf16, then
    # restore fp32 for the params that actually need precision (Linear
    # weights, biases, control tensors, summary tokens). This is the
    # explicit equivalent of relying on autocast's implicit weight
    # casting — but with a deterministic dtype boundary that doesn't fight
    # `torch.compile` tracing.
    if device.type == "cuda":
        model.bfloat16()
        restore_fp32_params(model)
    reward_normalizer = _build_reward_normalizer(cfg)
    return_pct_normalizer = _build_percentile_return_normalizer(cfg)
    if load_weights is not None:
        # Resume: load model weights only — fresh optimizer state and a fresh
        # opponent pool. Restoring the optimizer is rarely worth it across
        # league/self-play composition changes since AdamW's running stats
        # decay quickly anyway, and a fresh pool means no stale snapshots
        # whose weights no longer match the current architecture.
        ckpt = torch.load(load_weights, map_location=device, weights_only=False)
        state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
        model.load_state_dict(state, strict=True)
        _restore_reward_normalizer_from_checkpoint(
            reward_normalizer,
            ckpt,
            load_weights,
        )
        _restore_percentile_return_normalizer_from_checkpoint(
            return_pct_normalizer,
            ckpt,
        )
        print(f"loaded weights from {load_weights}")
    # Re-project the trunk matrices onto the hypersphere: the bf16 round-trip
    # in `restore_fp32_params` (and any loaded checkpoint) can leave row norms
    # a hair off 1.0. The training loop keeps them there after each step.
    normalize_matrices(model)
    # Value-pretrain AdamW. Split on ndim (nGPT `model.py:305-306`): the 1D
    # hypersphere scalars (eigen LRs, `sqk`, `suv`) must NOT be decayed — they
    # set magnitudes that aren't re-projected, so decay would collapse them.
    # The 2D trunk matrices that DO get decayed here are re-projected every
    # step by `normalize_matrices`, and decoupled decay is scale-only, so the
    # renorm makes it an exact no-op for them — only the scalars needed
    # protecting.
    pretrain_params = _value_pretrain_params(model)
    pretrain_opt = torch.optim.AdamW(
        [
            {
                "params": [p for p in pretrain_params if p.ndim >= 2],
                "weight_decay": cfg.optim.weight_decay,
            },
            {
                "params": [p for p in pretrain_params if p.ndim < 2],
                "weight_decay": 0.0,
            },
        ],
        lr=cfg.ppo.pretrain_lr,
    )
    optimizer = _build_optimizer(model, cfg.optim)

    elo = EloTracker(
        initial_rating=cfg.opponents.initial_rating,
        k_factor=cfg.opponents.k_factor,
    )
    elo.ensure(LEARNER_NAME)
    snapshot_device = (
        str(device) if cfg.opponents.snapshot_device == "train" else cfg.opponents.snapshot_device
    )
    snapshot_compile_mode = (
        _rollout_compile_mode_for_model(model, cfg)
        if torch.device(snapshot_device).type == "cuda"
        else None
    )
    if cfg.opponents.mode == "fixed":
        for name in cfg.opponents.fixed_opponents:
            elo.set(name, cfg.opponents.initial_rating)
        pool = FixedOpponentPool(
            cfg.opponents.fixed_opponents,
            rng=random.Random(cfg.run.seed),
        )
    elif cfg.opponents.mode == "no_builtins":
        pool = NoBuiltinTrainingPool(
            active_pool_size=cfg.opponents.active_pool_size,
            active_sample_panel_size=cfg.opponents.active_sample_panel_size,
            historical_training_archive_size=(cfg.opponents.historical_training_archive_size),
            current_learner_prob=cfg.opponents.current_learner_prob,
            active_pool_prob=cfg.opponents.active_pool_prob,
            historical_archive_prob=cfg.opponents.historical_archive_prob,
            difficulty_weight=cfg.opponents.active_difficulty_weight,
            uncertainty_weight=cfg.opponents.active_uncertainty_weight,
            recency_weight=cfg.opponents.active_recency_weight,
            hardness_weight=cfg.opponents.active_hardness_weight,
            recency_half_life_updates=(cfg.opponents.active_recency_half_life_updates),
            min_games_before_eviction=(cfg.opponents.min_games_before_active_eviction),
            stats_ema_decay=cfg.opponents.active_stats_ema_decay,
            historical_sample_panel_size=cfg.opponents.historical_sample_panel_size,
            historical_agent_cache_size=cfg.opponents.historical_agent_cache_size,
            recent_eviction_archive_size=cfg.opponents.recent_eviction_archive_size,
            notable_archive_size=cfg.opponents.notable_archive_size,
            device=snapshot_device,
            compile_mode=snapshot_compile_mode,
            rng=random.Random(cfg.run.seed),
        )
    else:
        pool = OpponentPool(
            elo=elo,
            top_k=cfg.opponents.top_k,
            self_play_prob=cfg.opponents.self_play_prob,
            device=snapshot_device,
            compile_mode=snapshot_compile_mode,
            rng=random.Random(cfg.run.seed),
        )
    logger = TBLogger(cfg.run.name, root=cfg.run.log_root)

    vec_counts = _training_vec_counts(cfg)
    with ExitStack() as stack:
        vecs = {
            num_players: stack.enter_context(_build_training_vec(cfg, num_players, num_envs))
            for num_players, num_envs in vec_counts.items()
            if num_envs > 0
        }
        # Pretrain doesn't write replays — skip the per-episode render +
        # pipe-transfer cost. _ppo_loop re-enables before the first update.
        for vec in vecs.values():
            vec.set_recording(False)
        pretrain_value(
            cfg,
            model,
            pretrain_opt,
            logger,
            device,
            vecs[cfg.game.num_players],
            reward_normalizer,
        )
        return _ppo_loop(
            cfg,
            model,
            optimizer,
            elo,
            pool,
            logger,
            device,
            vecs,
            reward_normalizer,
            return_pct_normalizer,
        )


def _ppo_loop(
    cfg: RunConfig,
    model: OrbitPolicy,
    optimizer: torch.optim.Optimizer,
    elo: EloTracker,
    pool: OpponentPool | FixedOpponentPool | NoBuiltinTrainingPool,
    logger: TBLogger,
    device: torch.device,
    vecs: dict[int, VecEnv],
    reward_normalizer: DiscountedReturnNormalizer | None = None,
    return_pct_normalizer: PercentileReturnNormalizer | None = None,
) -> dict:
    summary: dict = {"updates": []}
    cumulative_margin = 0.0
    cumulative_win_margin = 0.0
    cumulative_loss_margin = 0.0
    cumulative_games = 0
    best_win_rate = float("-inf")
    best_margin = float("-inf")
    best_update = -1

    if cfg.opponents.mode == "league":
        # Seed the pool with a snapshot of the random-init model. Without this,
        # `_sample_one` returns LEARNER_NAME for every slot until the first
        # snapshot lands at `snapshot_every`, every game is learner-vs-learner,
        # `update_from_game` early-returns on a single identity, and Elo stays
        # frozen. The init snapshot is *not* pinned — if it's bad it'll lose
        # rating and UCB-eviction will cull it like any other weak snapshot.
        init_ckpt = Path(cfg.run.ckpt_root) / cfg.run.name / "snapshot_init.pt"
        pool.add_snapshot(
            "init",
            model,
            init_ckpt,
            checkpoint_extra=_normalizer_checkpoint_extra(
                reward_normalizer, return_pct_normalizer
            ),
        )
    elif cfg.opponents.mode == "no_builtins":
        init_ckpt = Path(cfg.run.ckpt_root) / cfg.run.name / "snapshot_init.pt"
        pool.set_current_update(0)
        pool.add_snapshot(
            "init",
            model,
            init_ckpt,
            created_update=0,
            checkpoint_extra=_normalizer_checkpoint_extra(
                reward_normalizer, return_pct_normalizer
            ),
        )

    # `ppo_update` owns minibatch-level compile/capture. Keeping compilation
    # there lets Inductor see the policy forward, PPO loss, metrics, and
    # compiled backward as one fixed-shape training kernel.

    # One rendered game per update lands here for replay-capable backends.
    # Pretrain disabled recording; turn it back on for the PPO loop.
    for vec in vecs.values():
        if getattr(vec, "supports_replay", True):
            vec.set_recording(True)
        else:
            vec.set_recording(False)
    if not any(getattr(vec, "supports_replay", True) for vec in vecs.values()):
        print("rust env backend does not render HTML replays; skipping per-update replay dumps")
    replays_dir = logger.path / "replays"
    replays_dir.mkdir(parents=True, exist_ok=True)

    # KL-feedback LR controller state. PPO still runs every configured epoch;
    # the smoothed per-planet KL only adapts the next update's LR.
    kl_lr_ema = cfg.optim.kl_lr_target
    kl_lr_scale = 1.0
    train_num_players = _train_num_players(cfg)
    format_rng = random.Random(cfg.run.seed + 0x5E1F)

    for update in range(cfg.run.total_updates):
        if cfg.opponents.mode == "no_builtins":
            pool.set_current_update(update)
        update_t0 = perf_counter()
        phase_t0 = update_t0
        trajs = []
        rollout_s = 0.0
        rollout_timings: dict[str, float] | None = (
            {} if cfg.rollout.detail_timing or cfg.rollout.sample_detail_timing else None
        )
        sample_timings = rollout_timings if cfg.rollout.sample_detail_timing else None
        bookkeeping_s = 0.0
        format_counts = {num_players: 0 for num_players in train_num_players}
        total_games = cfg.rollout.num_envs * cfg.rollout.games_per_env_per_update
        episode_counts = _format_episode_counts(
            total_games,
            train_num_players,
            format_rng,
        )
        episode_cursor = 0
        format_order = list(train_num_players)
        format_rng.shuffle(format_order)
        for num_players in format_order:
            remaining_format_games = episode_counts.get(num_players, 0)
            if remaining_format_games <= 0:
                continue
            vec = vecs[num_players]
            format_chunk_idx = 0
            while remaining_format_games > 0:
                rollout_envs = min(vec.num_envs, remaining_format_games)
                remaining_format_games -= rollout_envs
                format_counts[num_players] += rollout_envs
                opponent_panel = (
                    pool.sample_panel(current_update=update)
                    if cfg.opponents.mode == "no_builtins"
                    else None
                )
                # Sample opponents once per env for this wave, then play all
                # envs in parallel. Each env's seat assignment is fixed for the
                # episode; the rollout batches policy forwards across all
                # alive envs.
                opponents_per_env = [
                    (
                        pool.sample(
                            num_players - 1,
                            current_update=update,
                            panel=opponent_panel,
                        )
                        if cfg.opponents.mode == "no_builtins"
                        else pool.sample(num_players - 1)
                    )
                    for _ in range(rollout_envs)
                ]
                seat_offset = update * total_games + episode_cursor
                learner_seats = alternating_learner_seats(
                    rollout_envs, num_players, offset=seat_offset
                )
                policy_graph_rows = _policy_compile_rows_for_sampled_rollout(
                    opponents_per_env,
                    learner_seats=learner_seats,
                    num_players=num_players,
                )
                phase_t0 = perf_counter()
                batch_trajs = rollout_episodes_batched(
                    model,
                    vec,
                    opponents_per_env,
                    num_players=num_players,
                    learner_seat=learner_seats,
                    device=str(device),
                    reward_cfg=cfg.reward,
                    compile_mode=_rollout_compile_mode_for_model(model, cfg),
                    compile_fleet_width=cfg.rollout.compile_fleet_width,
                    policy_graph_rows=policy_graph_rows,
                    snapshot_compile_rows=cfg.rollout.snapshot_compile_rows,
                    defer_log_prob=True,
                    chunk_records=True,
                    timings=rollout_timings,
                    sample_timings=sample_timings,
                )
                rollout_s += perf_counter() - phase_t0

                phase_t0 = perf_counter()
                if vec.last_replay_html is not None:
                    suffix = (
                        ""
                        if len(train_num_players) == 1
                        else f"_{num_players}p_{format_chunk_idx:02d}"
                    )
                    (replays_dir / f"update_{update:04d}{suffix}.html").write_text(
                        vec.last_replay_html
                    )

                for env_idx, traj in enumerate(batch_trajs):
                    slots = opponents_per_env[env_idx]
                    seat_names = _seat_names(traj.learner_seat, slots)
                    seats = list(zip(seat_names, traj.seat_rewards, strict=True))
                    if cfg.opponents.mode == "no_builtins":
                        pool.record_game(seats, current_update=update)
                    else:
                        elo.update_from_game(seats)
                trajs.extend(batch_trajs)
                bookkeeping_s += perf_counter() - phase_t0
                episode_cursor += rollout_envs
                format_chunk_idx += 1

        phase_t0 = perf_counter()
        batch = _stack_trajectories(
            trajs,
            gamma=cfg.ppo.gamma,
            gae_lambda=cfg.ppo.gae_lambda,
            value_gae_lambda=cfg.ppo.value_gae_lambda,
            critic_mtp_horizon=cfg.model.critic_mtp_horizon,
            reward_normalizer=reward_normalizer,
            include_old_log_prob=True,
        )
        stack_s = perf_counter() - phase_t0

        phase_t0 = perf_counter()
        compile_mode = _compile_mode_for_model(model, cfg)
        batch = _trim_ppo_batch_fleet_width(
            batch,
            pad_to_bucket=compile_mode is not None and device.type == "cuda",
        )
        batch_prepare_s = perf_counter() - phase_t0

        phase_t0 = perf_counter()
        # Apply the KL-adapted scale chosen by previous updates. The current
        # update's KL is folded into the controller after the PPO pass so every
        # minibatch in this PPO update uses one fixed optimizer scale.
        lr_scale = kl_lr_scale
        if hasattr(optimizer, "set_lr_scale"):
            optimizer.set_lr_scale(lr_scale)
        ppo_minibatch_size = _ppo_minibatch_size_for_fleet_width(
            cfg,
            int(batch["fleet_feats"].shape[1]),
        )
        old_log_prob_native = bool(batch.get("old_log_prob_computed", False))
        values_native = bool(batch.get("values_computed", False))
        old_log_prob_t0 = perf_counter()
        if cfg.ppo.policy_objective == "pmpo":
            # PMPO needs the FULL frozen old per-planet distribution for the
            # analytical reverse KL, not just the taken action's log-prob — always
            # recompute it here (the model is unchanged since rollout, so this
            # equals the rollout policy exactly). Reuses the recomputed value for
            # the advantage refresh, same as the PPO path below.
            old_dist = compute_old_policy_dist(
                model,
                batch,
                minibatch_size=ppo_minibatch_size,
                minibatch_count=cfg.optim.minibatch_count,
                compile_mode=compile_mode,
            )
            batch["old_log_prob"] = old_dist["old_log_prob"]
            batch["old_log_prob_computed"] = torch.tensor(True)
            batch["old_launch_logits"] = old_dist["old_launch_logits"]
            batch["old_target_logits"] = old_dist["old_target_logits"]
            batch["old_fraction_alpha"] = old_dist["old_fraction_alpha"]
            batch["old_fraction_beta"] = old_dist["old_fraction_beta"]
            _refresh_batch_advantages_from_values(
                batch,
                old_dist["value"],
                gamma=cfg.ppo.gamma,
                gae_lambda=cfg.ppo.gae_lambda,
                value_gae_lambda=cfg.ppo.value_gae_lambda,
                critic_mtp_horizon=cfg.model.critic_mtp_horizon,
            )
        elif not old_log_prob_native or not values_native:
            old_log_prob, behavior_values = compute_old_log_probs_and_values(
                model,
                batch,
                minibatch_size=ppo_minibatch_size,
                minibatch_count=cfg.optim.minibatch_count,
                compile_mode=compile_mode,
            )
            batch["old_log_prob"] = old_log_prob
            batch["old_log_prob_computed"] = torch.tensor(True)
            _refresh_batch_advantages_from_values(
                batch,
                behavior_values,
                gamma=cfg.ppo.gamma,
                gae_lambda=cfg.ppo.gae_lambda,
                value_gae_lambda=cfg.ppo.value_gae_lambda,
                critic_mtp_horizon=cfg.model.critic_mtp_horizon,
            )
        old_log_prob_s = perf_counter() - old_log_prob_t0
        # retnorm: scale the actor advantage by max(limit, p95-p5) of this
        # rollout's returns, computed once per update (never per minibatch).
        #   - scope "ema":   slow global percentile EMA (DreamerV3 retnorm).
        #   - scope "batch": fresh whole-rollout spread each update, no EMA.
        return_norm_scale = 1.0
        if return_pct_normalizer is not None:
            return_pct_normalizer.update(
                batch["return"].detach().to(torch.float32).cpu().numpy()
            )
            return_norm_scale = return_pct_normalizer.scale
        elif (
            cfg.ppo.advantage_return_norm == "perc"
            and cfg.ppo.advantage_return_norm_scope == "batch"
        ):
            return_norm_scale = _fresh_percentile_return_scale(
                batch["return"],
                perclo=cfg.ppo.advantage_return_norm_perclo,
                perchi=cfg.ppo.advantage_return_norm_perchi,
                limit=cfg.ppo.advantage_return_norm_limit,
            )
        log = ppo_update(
            model,
            optimizer,
            batch,
            value_coef=cfg.ppo.value_coef,
            target_entropy_coef=cfg.ppo.target_entropy_coef,
            fraction_entropy_coef=cfg.ppo.fraction_entropy_coef,
            norm_advantage=cfg.ppo.norm_advantage,
            norm_advantage_scope=cfg.ppo.norm_advantage_scope,
            advantage_transform=cfg.ppo.advantage_transform,
            return_norm_scale=return_norm_scale,
            clip_coef=cfg.ppo.clip_coef,
            clip_coef_high=cfg.ppo.clip_coef_high,
            policy_objective=cfg.ppo.policy_objective,
            pmpo_pos_to_neg_weight=cfg.ppo.pmpo_pos_to_neg_weight,
            pmpo_kl_coef=cfg.ppo.pmpo_kl_coef,
            pmpo_reverse_kl=cfg.ppo.pmpo_reverse_kl,
            epochs=cfg.optim.epochs_per_update,
            minibatch_size=ppo_minibatch_size,
            grad_clip=cfg.optim.grad_clip,
            minibatch_count=cfg.optim.minibatch_count,
            compile_mode=compile_mode,
        )
        kl_lr_signal = kl_lr_signal_from_log(log)
        if cfg.ppo.policy_objective != "pmpo":
            kl_lr_ema, kl_lr_scale = update_kl_lr_controller(
                kl_ema=kl_lr_ema,
                lr_scale=kl_lr_scale,
                observed_kl=kl_lr_signal,
                cfg=cfg.optim,
            )
        # else PMPO: hold the LR scale constant — the analytical reverse-KL
        # penalty inside the loss is the sole trust region ("no other kl
        # measures"), so the PPO-style approx KL must NOT drive the optimizer.
        ppo_s = perf_counter() - phase_t0
        value_ev = _explained_variance(batch["value"], batch["return"])

        phase_t0 = perf_counter()
        margins = [float(t.final_score) for t in trajs]
        episode_returns = [float(sum(t.reward)) for t in trajs]
        episode_lengths = [float(len(t.reward)) for t in trajs]
        win_rate = float(np.mean([t.won for t in trajs]))
        margin = float(np.mean(margins))
        episodic_return = float(np.mean(episode_returns)) if episode_returns else 0.0
        episodic_length = float(np.mean(episode_lengths)) if episode_lengths else 0.0
        update_win_margin = sum(m for t, m in zip(trajs, margins, strict=True) if t.won)
        update_loss_margin = sum(
            m for t, m in zip(trajs, margins, strict=True) if not t.won and not t.drawn
        )
        cumulative_margin += sum(margins)
        cumulative_win_margin += update_win_margin
        cumulative_loss_margin += update_loss_margin
        cumulative_games += len(margins)
        cumulative_mean_margin = cumulative_margin / max(1, cumulative_games)
        snapshot_elos = (
            []
            if cfg.opponents.mode == "no_builtins"
            else [elo.get(n) for n in pool.snapshot_names()]
        )
        metrics_s = perf_counter() - phase_t0

        phase_t0 = perf_counter()
        logger.scalars(
            "losses",
            {
                "policy_loss": log.policy_loss,
                "value_loss": log.value_loss,
                "entropy": log.entropy,
                "approx_kl": log.approx_kl,
                "per_planet_approx_kl": log.per_planet_approx_kl,
                "ratio_clip_frac_high": log.ratio_clip_frac_high,
                "ratio_clip_frac": log.ratio_clip_frac,
                "explained_variance": value_ev,
                "actor_grad_norm": log.actor_grad_norm,
                "critic_grad_norm": log.critic_grad_norm,
                "actor_shared_grad_norm": log.actor_shared_grad_norm,
                "critic_shared_grad_norm": log.critic_shared_grad_norm,
                "shared_merged_grad_norm": log.shared_grad_norm,
                "actor_shared_raw_grad_norm": log.actor_shared_raw_grad_norm,
                "critic_shared_raw_grad_norm": log.critic_shared_raw_grad_norm,
                "actor_clip_scale": log.actor_clip_scale,
                "critic_clip_scale": log.critic_clip_scale,
                "actor_clip_frac": log.actor_clip_frac,
                "critic_clip_frac": log.critic_clip_frac,
                "log_ratio_abs_mean": log.log_ratio_abs_mean,
                "log_ratio_abs_max": log.log_ratio_abs_max,
                "row_log_ratio_abs_mean": log.row_log_ratio_abs_mean,
                "raw_advantage_abs_mean": float(batch["raw_advantage_abs_mean"]),
                "advantage_return_scale": return_norm_scale,
                "epochs_run": log.epochs_run,
                "pmpo_reverse_kl": log.reverse_kl,
                "pmpo_pos_loss": log.pmpo_pos_loss,
                "pmpo_neg_loss": log.pmpo_neg_loss,
            },
            update,
        )
        if reward_normalizer is not None:
            target_edge_mass = 0.0
            if bool(batch["return_mtp_mask"].any()):
                with torch.no_grad():
                    return_mtp = batch["return_mtp"].to(device, non_blocking=True)
                    return_mtp_mask = batch["return_mtp_mask"].to(device, non_blocking=True)
                    target_probs = model.value_encoder.target_probs(return_mtp)
                    edge_mass = target_probs[..., 0] + target_probs[..., -1]
                    target_edge_mass = float(edge_mass.masked_select(return_mtp_mask).mean().item())
            logger.scalars(
                "reward_norm",
                {
                    "mean": reward_normalizer.mean,
                    "std": math.sqrt(reward_normalizer.var),
                    "count": reward_normalizer.count,
                    "reward_clip_frac": reward_normalizer.batch_clip_frac,
                    "reward_absmax_preclip": reward_normalizer.batch_reward_absmax,
                    "return_mean": float(batch["return"].mean()),
                    "return_std": float(batch["return"].std())
                    if batch["return"].numel() > 1
                    else 0.0,
                    "return_min": float(batch["return"].min()),
                    "return_max": float(batch["return"].max()),
                    "return_absmax": float(batch["return"].abs().max()),
                    "target_edge_mass": target_edge_mass,
                },
                update,
            )
        logger.scalars(
            "optim",
            {
                "lr_scale": lr_scale,
                "kl_lr_scale_next": kl_lr_scale,
                "kl_lr_ema": kl_lr_ema,
                "kl_lr_signal": kl_lr_signal,
            },
            update,
        )
        control_stats = ngpt_control_stats(model)
        if control_stats:
            logger.scalars("ngpt", control_stats, update)
        batch_rows = int(batch["planet_feats"].shape[0])
        fleet_width = int(batch["fleet_feats"].shape[1])
        logical_minibatch_size = (
            math.ceil(batch_rows / cfg.optim.minibatch_count)
            if cfg.optim.minibatch_count is not None
            else ppo_minibatch_size
        )
        logger.scalars(
            "batch",
            {
                "rows": batch_rows,
                "fleet_width": fleet_width,
                "logical_minibatch_size": logical_minibatch_size,
            },
            update,
        )
        logger.scalars(
            "policy",
            {
                "target_entropy": log.target_entropy,
                "fraction_entropy": log.fraction_entropy,
                "target_confidence": log.target_confidence,
                "move_prob": log.move_prob,
                "executed_launch_frac": log.executed_launch_frac,
                "source_non_action_frac": log.source_non_action_frac,
                "turn_no_action_frac": log.turn_no_action_frac,
                "legal_target_count_mean": log.legal_target_count_mean,
                "uniform_move_prior": log.uniform_move_prior,
                "owned_planets_mean": log.owned_planets_mean,
                "pos_advantage_frac": log.pos_frac,
                "launch_mean": log.launch_mean,
                "launch_log_std_mean": log.launch_log_std_mean,
                "launch_score_mean": log.launch_score_mean,
                "action_logit_softcap": log.action_logit_softcap,
            },
            update,
        )
        logger.scalars(
            "fraction",
            {
                "alpha_mean": log.fraction_alpha_mean,
                "beta_mean": log.fraction_beta_mean,
                "concentration_mean": log.fraction_concentration_mean,
                "concentration_max": log.fraction_concentration_max,
                "skew_abs_mean": log.fraction_skew_abs_mean,
                "deterministic_mean": log.deterministic_fraction_mean,
            },
            update,
        )
        logger.scalars(
            "rollout",
            {
                "win_rate": win_rate,
                "margin": margin,
                "games_2p": float(format_counts.get(2, 0)),
                "games_4p": float(format_counts.get(4, 0)),
                "games_4p_frac": (format_counts.get(4, 0) / max(1, sum(format_counts.values()))),
                "cumulative_margin": cumulative_margin,
                "cumulative_win_margin": cumulative_win_margin,
                "cumulative_loss_margin": cumulative_loss_margin,
                "cumulative_mean_margin": cumulative_mean_margin,
                "episodic_return_mean": episodic_return,
                "episodic_return_std": (float(np.std(episode_returns)) if episode_returns else 0.0),
                "episodic_return_min": min(episode_returns) if episode_returns else 0.0,
                "episodic_return_max": max(episode_returns) if episode_returns else 0.0,
                "episodic_length_mean": episodic_length,
            },
            update,
        )
        league_metrics = {
            "elo_learner": elo.get(LEARNER_NAME),
            "pool_size": float(len(snapshot_elos)),
            "pool_elo_max": max(snapshot_elos) if snapshot_elos else float("nan"),
            "pool_elo_min": min(snapshot_elos) if snapshot_elos else float("nan"),
        }
        if cfg.opponents.mode == "no_builtins":
            active_names = pool.active_snapshot_names()
            historical_names = pool.historical_snapshot_names()
            historical_log_names = pool.historical_snapshot_names("log")
            historical_recent_names = pool.historical_snapshot_names("recent_eviction")
            historical_notable_names = pool.historical_snapshot_names("notable")
            active_stats = [pool.snapshot_stats(name) for name in active_names]
            learner_win_rates = [stats.learner_win_rate() for stats in active_stats]
            games_vs_current = [stats.games_vs_current for stats in active_stats]
            league_metrics.update(
                {
                    "pool_size": float(len(active_names)),
                    "historical_archive_size": float(len(historical_names)),
                    "historical_log_archive_size": float(len(historical_log_names)),
                    "historical_recent_eviction_size": float(len(historical_recent_names)),
                    "historical_notable_size": float(len(historical_notable_names)),
                    "all_snapshot_count": float(len(pool.all_snapshot_names())),
                    "active_games_vs_current_mean": (
                        float(np.mean(games_vs_current)) if games_vs_current else 0.0
                    ),
                    "active_learner_win_rate_mean": (
                        float(np.mean(learner_win_rates)) if learner_win_rates else float("nan")
                    ),
                    "active_learner_win_rate_abs_dist_from_half": (
                        float(np.mean([abs(wr - 0.5) for wr in learner_win_rates]))
                        if learner_win_rates
                        else float("nan")
                    ),
                }
            )
        logger.scalars("league", league_metrics, update)
        logging_s = perf_counter() - phase_t0

        phase_t0 = perf_counter()
        summary["updates"].append(
            {
                "update": update,
                "win_rate": win_rate,
                "margin": margin,
                "episodic_return": episodic_return,
                "episodic_length": episodic_length,
                "cumulative_margin": cumulative_margin,
                "cumulative_mean_margin": cumulative_mean_margin,
                "elo_learner": elo.get(LEARNER_NAME),
            }
        )

        is_best = (win_rate > best_win_rate) or (win_rate == best_win_rate and margin > best_margin)
        if cfg.opponents.mode == "fixed" and is_best:
            best_win_rate = win_rate
            best_margin = margin
            best_update = update
            best_path = Path(cfg.run.ckpt_root) / cfg.run.name / "best.pt"
            _save_ppo_checkpoint(model, best_path, reward_normalizer, return_pct_normalizer)
            best_meta = {
                "update": best_update,
                "win_rate": best_win_rate,
                "margin": best_margin,
                "episodic_return": episodic_return,
                "episodic_length": episodic_length,
                "run_dir": str(logger.path),
            }
            best_path.with_name("best.json").write_text(
                json.dumps(best_meta, indent=2, sort_keys=True)
            )

        if cfg.opponents.mode == "league" and (update + 1) % cfg.opponents.snapshot_every == 0:
            ckpt = Path(cfg.run.ckpt_root) / cfg.run.name / f"snapshot_{update:04d}.pt"
            pool.add_snapshot(
                f"{update:04d}",
                model,
                ckpt,
                checkpoint_extra=_normalizer_checkpoint_extra(
                    reward_normalizer, return_pct_normalizer
                ),
            )
        elif (
            cfg.opponents.mode == "no_builtins" and (update + 1) % cfg.opponents.snapshot_every == 0
        ):
            pool.set_current_update(update + 1)
            ckpt = Path(cfg.run.ckpt_root) / cfg.run.name / f"snapshot_{update:04d}.pt"
            pool.add_snapshot(
                f"{update:04d}",
                model,
                ckpt,
                created_update=update + 1,
                checkpoint_extra=_normalizer_checkpoint_extra(
                    reward_normalizer, return_pct_normalizer
                ),
            )
        elif cfg.opponents.mode == "fixed" and (update + 1) % cfg.opponents.snapshot_every == 0:
            _save_ppo_checkpoint(
                model,
                Path(cfg.run.ckpt_root) / cfg.run.name / "latest.pt",
                reward_normalizer,
                return_pct_normalizer,
            )
        snapshot_s = perf_counter() - phase_t0
        update_s = perf_counter() - update_t0
        learner_steps = sum(len(t.reward) for t in trajs)
        learner_steps_per_s = learner_steps / max(rollout_s, 1e-9)
        end_to_end_steps_per_s = learner_steps / max(update_s, 1e-9)
        games_per_minute = len(trajs) * 60.0 / max(update_s, 1e-9)

        logger.scalars(
            "timing",
            {
                "update_s": update_s,
                "rollout_s": rollout_s,
                "bookkeeping_s": bookkeeping_s,
                "stack_s": stack_s,
                "batch_prepare_s": batch_prepare_s,
                "old_log_prob_s": old_log_prob_s,
                "ppo_s": ppo_s,
                "metrics_s": metrics_s,
                "logging_s": logging_s,
                "snapshot_s": snapshot_s,
                "learner_steps_per_s": learner_steps_per_s,
                "games_per_minute": games_per_minute,
                "rollout_games_per_minute": len(trajs) * 60.0 / max(rollout_s, 1e-9),
            },
            update,
        )
        if rollout_timings is not None:
            logger.scalars("rollout_detail", rollout_timings, update)
        logger.scalars("charts", {"SPS": end_to_end_steps_per_s}, update)

    final_path = Path(cfg.run.ckpt_root) / cfg.run.name / "final.pt"
    _save_ppo_checkpoint(model, final_path, reward_normalizer, return_pct_normalizer)

    elo_path = final_path.with_name("elo.json")
    elo_path.write_text(json.dumps(elo.snapshot_dict(), indent=2, sort_keys=True))

    logger.close()
    summary["final_ckpt"] = str(final_path)
    summary["elo_path"] = str(elo_path)
    summary["elo_learner_final"] = elo.get(LEARNER_NAME)
    summary["cumulative_margin"] = cumulative_margin
    summary["cumulative_mean_margin"] = cumulative_margin / max(1, cumulative_games)
    return summary


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--name", default=None, help="Override run.name.")
    p.add_argument("--total-updates", type=int, default=None)
    p.add_argument("--num-envs", type=int, default=None)
    p.add_argument("--num-workers", type=int, default=None)
    p.add_argument("--episode-steps", type=int, default=None)
    p.add_argument(
        "--env-backend",
        choices=("numpy", "numpy_mp", "rust", "kaggle"),
        default=None,
    )
    p.add_argument(
        "--compile-mode",
        default=None,
        help="Override run.compile_mode; use 'none' to disable torch.compile.",
    )
    p.add_argument(
        "--rollout-detail-timing",
        action="store_true",
        help="Log rollout_detail/* scalars for rollout bucketing, policy, reward, and env stepping.",
    )
    p.add_argument(
        "--sample-detail-timing",
        action="store_true",
        help="Log rollout_detail/* scalars for Rust sampler legality, transfer, and materialization phases.",
    )
    p.add_argument(
        "--load",
        default=None,
        help="Path to a .pt checkpoint whose `model` state_dict should be "
        "loaded into the policy before training starts. Optimizer state and "
        "the opponent pool are NOT restored.",
    )
    args = p.parse_args()
    cfg = load_config(args.config)
    if args.name is not None:
        cfg.run.name = args.name
    if args.total_updates is not None:
        cfg.run.total_updates = args.total_updates
    if args.num_envs is not None:
        cfg.rollout.num_envs = args.num_envs
    if args.num_workers is not None:
        cfg.rollout.num_workers = args.num_workers
    if args.episode_steps is not None:
        cfg.game.episode_steps = args.episode_steps
    if args.env_backend is not None:
        cfg.rollout.env_backend = args.env_backend
    if args.compile_mode is not None:
        cfg.run.compile_mode = "" if args.compile_mode.lower() == "none" else args.compile_mode
    if args.rollout_detail_timing:
        cfg.rollout.detail_timing = True
    if args.sample_detail_timing:
        cfg.rollout.detail_timing = True
        cfg.rollout.sample_detail_timing = True
    summary = train_one_run(cfg, load_weights=args.load)
    print({k: v for k, v in summary.items() if k != "updates"})


if __name__ == "__main__":
    main()
