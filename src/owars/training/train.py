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
from contextlib import suppress
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from ..policies.config import OrbitPolicyConfig
from ..policies.model import OrbitPolicy, normalize_matrices, restore_fp32_params
from ..utils import TBLogger, set_seed
from .config import OptimCfg, RunConfig, load_config
from .elo import EloTracker
from .league import (
    BUILTIN,
    LEARNER_NAME,
    FixedOpponentPool,
    OpponentPool,
    OpponentSlot,
)
from .muon import MultiOptimizer, Muon
from .numpy_env import NumpyVecEnv
from .ppo import (
    _slice_feats,
    compute_gae,
    ppo_update,
    value_only_update,
)
from .rollout import Trajectory
from .sharded_numpy_env import ShardedNumpyVecEnv
from .vec_env import VecEnv
from .vec_rollout import alternating_learner_seats, rollout_episodes_batched

# Subset of control tensors that route to the *fast* AdamW group at
# `control_lr` (≈ `muon_lr`) — the nGPT hypersphere controls (per-channel
# eigen LRs `attn_alpha`/`mlp_alpha`/`cross_alpha`, QK scale `sqk`, MLP scale
# `suv`) and the target-readout attention temperature (`q_gain`/
# `target_q_gain`). These need update magnitudes comparable to Muon's matrix
# updates. The summary tokens (`actor_token`, `critic_token`) intentionally
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

    AdamW (control-lr): per-channel residual scales and `q_gain`s — need
    update magnitudes comparable to Muon's matrix updates, see
    `OptimCfg.control_lr`.

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
        row_normalize=cfg.muon_row_normalize,
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
    # `control_lr` (≈ muon_lr, parity with matrix updates), and task readouts
    # at `head_lr`.
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
    return MultiOptimizer(
        [muon_opt, adamw_opt], lr_warmup_steps=cfg.lr_warmup_steps
    )


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
        critic_mtp_horizon=cfg.model.critic_mtp_horizon,
        value_min=cfg.model.value_min,
        value_max=cfg.model.value_max,
        value_symlog=cfg.model.value_symlog,
        action_logit_softcap=cfg.model.action_logit_softcap,
    )
    return OrbitPolicy(pcfg)


def _compile_mode_for_model(model: OrbitPolicy, cfg: RunConfig) -> str | None:
    return cfg.run.compile_mode or None


def _stack_encoded(trajs: list[Trajectory]) -> dict[str, torch.Tensor]:
    """Walk every (traj, step) once and emit stacked EncodedObs tensors.

    Per-step records on Trajectory are already device tensors (see the
    Trajectory docstring) — we just gather and stack here. Encoder-only:
    the actor-side records (launch / target_idx / fraction / old_log_prob /
    owned_mask) are added by `_stack_trajectories`, which lets
    `_pretrain_value_batch` skip them entirely.
    """
    pf, pm, pom, pid, pg, ff, fm = [], [], [], [], [], [], []
    for t in trajs:
        for e in t.encoded:
            pf.append(e.planet_feats)
            pm.append(e.planet_mask)
            pom.append(e.planet_owned_mask)
            pid.append(e.planet_ids)
            pg.append(e.planet_garrison)
            ff.append(e.fleet_feats)
            fm.append(e.fleet_mask)
    return {
        "planet_feats": torch.stack(pf),
        "planet_mask": torch.stack(pm),
        "planet_owned_mask": torch.stack(pom),
        "planet_ids": torch.stack(pid),
        "planet_garrison": torch.stack(pg),
        "fleet_feats": torch.stack(ff),
        "fleet_mask": torch.stack(fm),
    }


def _stack_trajectories(
    trajs: list[Trajectory],
    gamma: float,
    gae_lambda: float,
    value_gae_lambda: float | None = None,
    critic_mtp_horizon: int = 1,
) -> dict[str, torch.Tensor]:
    """Flatten per-step records into one PPO batch with optional decoupled GAE."""
    batch = _stack_encoded(trajs)

    launch, tidx, frac, lp, owned, target_legal = [], [], [], [], [], []
    for t in trajs:
        launch.extend(t.launch)
        tidx.extend(t.target_idx)
        frac.extend(t.fraction)
        lp.extend(t.log_prob)
        owned.extend(t.owned_mask)
        target_legal.extend(t.target_legal_mask)
    batch["launch"] = torch.stack(launch).float()
    batch["target_idx"] = torch.stack(tidx).long()
    batch["fraction"] = torch.stack(frac).float()
    batch["old_log_prob"] = torch.stack(lp).float()
    batch["owned_mask"] = torch.stack(owned).bool()
    batch["target_legal_mask"] = torch.stack(target_legal).bool()

    # Single global CPU pull of every per-step value across the batch —
    # one sync instead of one per trajectory.
    value_tensors = [torch.stack(t.value) for t in trajs if t.value]
    if value_tensors:
        all_values = torch.cat(value_tensors).detach().to(torch.float32).cpu().numpy()
    else:
        all_values = np.zeros(0, dtype=np.float32)

    target_lam = gae_lambda if value_gae_lambda is None else value_gae_lambda
    mtp_h = max(1, int(critic_mtp_horizon))
    advs_all, rets_all, mtp_all, mtp_mask_all = [], [], [], []
    offset = 0
    for t in trajs:
        rewards = np.asarray(t.reward, dtype=np.float32)
        horizon = len(rewards)
        values = all_values[offset : offset + horizon]
        offset += horizon
        adv, ret = compute_gae(rewards, values, gamma, gae_lambda)
        if target_lam != gae_lambda:
            _target_adv, ret = compute_gae(rewards, values, gamma, target_lam)
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

    advs = torch.from_numpy(np.concatenate(advs_all)).float()
    rets = torch.from_numpy(np.concatenate(rets_all)).float()
    values = torch.from_numpy(all_values).float()
    if values.numel() != rets.numel():
        values = torch.zeros_like(rets)
    batch["advantage"] = advs
    batch["return"] = rets
    batch["return_mtp"] = torch.from_numpy(np.concatenate(mtp_all)).float()
    batch["return_mtp_mask"] = torch.from_numpy(np.concatenate(mtp_mask_all)).bool()
    batch["value"] = values
    batch["raw_advantage_abs_mean"] = torch.tensor(
        float(advs.abs().mean()) if advs.numel() else 0.0,
        dtype=torch.float32,
    )
    return batch


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
) -> dict[str, torch.Tensor]:
    """Critic-target = configured bootstrapped GAE/lambda return.

    Encoder fields only — `value_only_update` doesn't read the actor-side
    records, so we skip stacking and host→device-copying them.
    """
    batch = _stack_encoded(trajs)
    mtp_h = max(1, int(critic_mtp_horizon))
    rets_all, mtp_all, mtp_mask_all = [], [], []
    for t in trajs:
        rewards = np.asarray(t.reward, dtype=np.float32)
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


def pretrain_value(cfg: RunConfig, model: OrbitPolicy, optimizer: torch.optim.Optimizer,
                   logger: TBLogger, device: torch.device, vec: VecEnv) -> None:
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
        [behavior_slot] * (cfg.game.num_players - 1)
        for _ in range(cfg.rollout.num_envs)
    ]

    for step in range(cfg.ppo.pretrain_updates):
        # Round episode count up to the nearest num_envs batch.
        batches = max(1, cfg.ppo.pretrain_episodes // cfg.rollout.num_envs)
        trajs: list[Trajectory] = []
        for batch_idx in range(batches):
            learner_seats = alternating_learner_seats(
                cfg.rollout.num_envs,
                cfg.game.num_players,
                offset=step * batches + batch_idx,
            )
            trajs.extend(rollout_episodes_batched(
                model,
                vec,
                opponents_per_env,
                num_players=cfg.game.num_players,
                learner_seat=learner_seats,
                device=str(device),
                deterministic=False,
                reward_cfg=cfg.reward,
                compile_mode=_compile_mode_for_model(model, cfg),
                policy_graph_rows=cfg.rollout.num_envs,
            ))
        batch = {
            k: v.to(device)
            for k, v in _pretrain_value_batch(
                trajs,
                gamma=cfg.ppo.gamma,
                gae_lambda=(
                    cfg.ppo.gae_lambda
                    if cfg.ppo.value_gae_lambda is None
                    else cfg.ppo.value_gae_lambda
                ),
                critic_mtp_horizon=cfg.model.critic_mtp_horizon,
            ).items()
        }
        loss = value_only_update(
            model, optimizer, batch,
            epochs=1,
            minibatch_size=cfg.optim.minibatch_size,
            grad_clip=cfg.optim.grad_clip,
            compile_mode=_compile_mode_for_model(model, cfg),
        )
        rets = batch["return"].cpu().numpy()
        # Explained variance: 1 - Var(target - pred)/Var(target). Tracked
        # because raw value loss can be misleading when the target
        # distribution shifts (e.g., as the behavior policy gets crushed).
        # Minibatched: a full-batch forward over `pretrain_episodes ×
        # episode_steps` samples blows up the [B, heads, tokens, tokens]
        # attention tensor (tokens ≈ 64 planets + 384 fleets + 2 summary).
        n = batch["planet_feats"].shape[0]
        mb = cfg.optim.minibatch_size
        autocast_enabled = device.type == "cuda"
        with (
            torch.no_grad(),
            torch.autocast(
                device_type="cuda", dtype=torch.bfloat16, enabled=autocast_enabled
            ),
        ):
            chunks = [
                model(_slice_feats(batch, slice(s, s + mb))).value
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


def _save_ppo_checkpoint(model: OrbitPolicy, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "config": model.cfg.to_dict()}, path)


def _value_pretrain_params(model: OrbitPolicy) -> list[torch.nn.Parameter]:
    """Params that *actually* get gradient from value-only loss.

    Includes the encoder (shared backbone), both summary tokens (actor_token
    feeds the encoder self-attention so h_critic depends on it; critic_token
    feeds the value head directly), and the value head. Excludes the actor
    heads (target_query/key, categorical action heads, fraction alpha/beta heads) — they receive zero
    gradient from the value loss, and including them would let AdamW's
    weight-decay pull them toward zero with no learning signal, leaving PPO
    to start from a worse-than-init policy.
    """
    encoder = [model.planet_embed, model.fleet_embed, *model.layers]
    if model.fleet_tokenizer is not None:
        encoder.append(model.fleet_tokenizer)
    value = [model.value_head]
    params: list[torch.nn.Parameter] = [model.actor_token, model.critic_token]
    for m in encoder + value:
        params.extend(m.parameters())
    return params


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
    if load_weights is not None:
        # Resume: load model weights only — fresh optimizer state and a fresh
        # opponent pool. Restoring the optimizer is rarely worth it across
        # league/self-play composition changes since AdamW's running stats
        # decay quickly anyway, and a fresh pool means no stale snapshots
        # whose weights no longer match the current architecture.
        ckpt = torch.load(load_weights, map_location=device, weights_only=False)
        state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
        model.load_state_dict(state, strict=True)
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
        str(device) if cfg.opponents.snapshot_device == "train"
        else cfg.opponents.snapshot_device
    )
    if cfg.opponents.mode == "fixed":
        for name in cfg.opponents.fixed_opponents:
            elo.set(name, cfg.opponents.initial_rating)
        pool = FixedOpponentPool(
            cfg.opponents.fixed_opponents,
            rng=random.Random(cfg.run.seed),
        )
    else:
        pool = OpponentPool(
            elo=elo,
            top_k=cfg.opponents.top_k,
            self_play_prob=cfg.opponents.self_play_prob,
            device=snapshot_device,
            rng=random.Random(cfg.run.seed),
        )
    logger = TBLogger(cfg.run.name, root=cfg.run.log_root)

    # One subprocess pool reused across pretraining + every PPO update.
    # Spawning per-call cost ~16-32 s of pure interpreter startup × every
    # rollout (cumulative ~1 h on a full run); `vec.reset()` is cheap.
    # `replay_env_idx=0` keeps env 0's full step history so the PPO loop
    # can dump one rendered game per update; the other workers trim
    # `env.steps` to save memory.
    if cfg.rollout.env_backend not in {"kaggle", "numpy", "numpy_mp", "rust"}:
        raise ValueError(f"unknown rollout.env_backend: {cfg.rollout.env_backend!r}")
    vec_kwargs = dict(
        num_envs=cfg.rollout.num_envs,
        num_players=cfg.game.num_players,
        episode_steps=cfg.game.episode_steps,
        ship_speed=cfg.game.ship_speed,
    )
    if cfg.rollout.env_backend == "numpy":
        vec = NumpyVecEnv(
            **vec_kwargs,
            replay_env_idx=None,
            random_seed=cfg.run.seed,
        )
    elif cfg.rollout.env_backend == "numpy_mp":
        vec = ShardedNumpyVecEnv(
            **vec_kwargs,
            replay_env_idx=None,
            num_workers=cfg.rollout.num_workers,
            random_seed=cfg.run.seed,
        )
    elif cfg.rollout.env_backend == "rust":
        from .rust_env import RustVecEnv

        vec = RustVecEnv(
            **vec_kwargs,
            replay_env_idx=None,
            random_seed=cfg.run.seed,
        )
    else:
        vec = VecEnv(**vec_kwargs, replay_env_idx=0)
    with vec:
        # Pretrain doesn't write replays — skip the per-episode render +
        # pipe-transfer cost. _ppo_loop re-enables before the first update.
        vec.set_recording(False)
        pretrain_value(cfg, model, pretrain_opt, logger, device, vec)
        return _ppo_loop(cfg, model, optimizer, elo, pool, logger, device, vec)


def _ppo_loop(
    cfg: RunConfig,
    model: OrbitPolicy,
    optimizer: torch.optim.Optimizer,
    elo: EloTracker,
    pool: OpponentPool | FixedOpponentPool,
    logger: TBLogger,
    device: torch.device,
    vec: VecEnv,
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
        pool.add_snapshot("init", model, init_ckpt)

    # `ppo_update` owns minibatch-level compile/capture. Keeping compilation
    # there lets Inductor see the policy forward, PPO loss, metrics, and
    # compiled backward as one fixed-shape training kernel.

    # One rendered game per update lands here (env 0 is the recording
    # worker; see VecEnv(replay_env_idx=0) above). Pretrain disabled
    # recording; turn it back on for the PPO loop.
    if getattr(vec, "supports_replay", True):
        vec.set_recording(True)
    else:
        vec.set_recording(False)
        print("rust env backend does not render HTML replays; skipping per-update replay dumps")
    replays_dir = logger.path / "replays"
    replays_dir.mkdir(parents=True, exist_ok=True)

    for update in range(cfg.run.total_updates):
        update_t0 = perf_counter()
        phase_t0 = update_t0
        trajs = []
        rollout_s = 0.0
        bookkeeping_s = 0.0
        for game_batch in range(cfg.rollout.games_per_env_per_update):
            # Sample opponents once per env for this wave, then play all envs
            # in parallel. Each env's seat assignment is fixed for the episode;
            # the rollout batches the policy forward across all alive envs.
            opponents_per_env = [
                pool.sample(cfg.game.num_players - 1)
                for _ in range(cfg.rollout.num_envs)
            ]
            seat_offset = update * cfg.rollout.games_per_env_per_update + game_batch
            learner_seats = alternating_learner_seats(
                cfg.rollout.num_envs, cfg.game.num_players, offset=seat_offset
            )
            phase_t0 = perf_counter()
            batch_trajs = rollout_episodes_batched(
                model,
                vec,
                opponents_per_env,
                num_players=cfg.game.num_players,
                learner_seat=learner_seats,
                device=str(device),
                reward_cfg=cfg.reward,
                compile_mode=_compile_mode_for_model(model, cfg),
            )
            rollout_s += perf_counter() - phase_t0

            phase_t0 = perf_counter()
            if vec.last_replay_html is not None:
                suffix = (
                    ""
                    if cfg.rollout.games_per_env_per_update == 1
                    else f"_gamebatch_{game_batch:02d}"
                )
                (replays_dir / f"update_{update:04d}{suffix}.html").write_text(
                    vec.last_replay_html
                )

            for env_idx, traj in enumerate(batch_trajs):
                slots = opponents_per_env[env_idx]
                seat_names = _seat_names(traj.learner_seat, slots)
                elo.update_from_game(
                    list(zip(seat_names, traj.seat_rewards, strict=True))
                )
            trajs.extend(batch_trajs)
            bookkeeping_s += perf_counter() - phase_t0

        phase_t0 = perf_counter()
        batch = _stack_trajectories(
            trajs,
            gamma=cfg.ppo.gamma,
            gae_lambda=cfg.ppo.gae_lambda,
            value_gae_lambda=cfg.ppo.value_gae_lambda,
            critic_mtp_horizon=cfg.model.critic_mtp_horizon,
        )
        stack_s = perf_counter() - phase_t0

        phase_t0 = perf_counter()
        batch = {k: v.to(device) for k, v in batch.items()}
        batch_to_device_s = perf_counter() - phase_t0

        phase_t0 = perf_counter()
        compile_mode = _compile_mode_for_model(model, cfg)
        log = ppo_update(
            model,
            optimizer,
            batch,
            value_coef=cfg.ppo.value_coef,
            target_entropy_coef=cfg.ppo.target_entropy_coef,
            fraction_entropy_coef=cfg.ppo.fraction_entropy_coef,
            norm_advantage=cfg.ppo.norm_advantage,
            advantage_transform=cfg.ppo.advantage_transform,
            spo_eps_low=cfg.ppo.spo_eps_low,
            spo_eps_high=cfg.ppo.spo_eps_high,
            epochs=cfg.optim.epochs_per_update,
            minibatch_size=cfg.optim.minibatch_size,
            grad_clip=cfg.optim.grad_clip,
            minibatch_count=cfg.optim.minibatch_count,
            compile_mode=compile_mode,
        )
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
        snapshot_elos = [elo.get(n) for n in pool.snapshot_names()]
        metrics_s = perf_counter() - phase_t0

        phase_t0 = perf_counter()
        logger.scalars(
            "loss",
            {
                "policy": log.policy_loss,
                "value": log.value_loss,
            },
            update,
        )
        logger.scalars(
            "losses",
            {
                "policy_loss": log.policy_loss,
                "value_loss": log.value_loss,
                "entropy": log.entropy,
                "approx_kl": log.approx_kl,
                "per_planet_approx_kl": log.per_planet_approx_kl,
                "spo_penalty": log.spo_penalty,
                "spo_clip_frac": log.spo_clip_frac,
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
            },
            update,
        )
        batch_rows = int(batch["planet_feats"].shape[0])
        logical_minibatch_size = (
            math.ceil(batch_rows / cfg.optim.minibatch_count)
            if cfg.optim.minibatch_count is not None
            else cfg.optim.minibatch_size
        )
        logger.scalars(
            "batch",
            {
                "rows": batch_rows,
                "logical_minibatch_size": logical_minibatch_size,
                "minibatch_size": logical_minibatch_size,
            },
            update,
        )
        logger.scalars(
            "kl",
            {
                "approx": log.approx_kl,
                "per_planet_approx": log.per_planet_approx_kl,
                "spo_penalty": log.spo_penalty,
                "spo_clip_frac": log.spo_clip_frac,
                "log_ratio_abs_mean": log.log_ratio_abs_mean,
                "log_ratio_abs_max": log.log_ratio_abs_max,
                "row_log_ratio_abs_mean": log.row_log_ratio_abs_mean,
            },
            update,
        )
        logger.scalars(
            "policy",
            {
                "entropy": log.entropy,
                "action_entropy": log.target_entropy,
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
                "raw_advantage_abs_mean": float(batch["raw_advantage_abs_mean"]),
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
                "cumulative_margin": cumulative_margin,
                "cumulative_win_margin": cumulative_win_margin,
                "cumulative_loss_margin": cumulative_loss_margin,
                "cumulative_mean_margin": cumulative_mean_margin,
                "episodic_return_mean": episodic_return,
                "episodic_return_std": (
                    float(np.std(episode_returns)) if episode_returns else 0.0
                ),
                "episodic_return_min": min(episode_returns) if episode_returns else 0.0,
                "episodic_return_max": max(episode_returns) if episode_returns else 0.0,
                "episodic_length_mean": episodic_length,
            },
            update,
        )
        logger.scalars(
            "league",
            {
                "elo_learner": elo.get(LEARNER_NAME),
                "pool_size": float(len(snapshot_elos)),
                "pool_elo_max": max(snapshot_elos) if snapshot_elos else float("nan"),
                "pool_elo_min": min(snapshot_elos) if snapshot_elos else float("nan"),
            },
            update,
        )
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

        is_best = (win_rate > best_win_rate) or (
            win_rate == best_win_rate and margin > best_margin
        )
        if cfg.opponents.mode == "fixed" and is_best:
            best_win_rate = win_rate
            best_margin = margin
            best_update = update
            best_path = Path(cfg.run.ckpt_root) / cfg.run.name / "best.pt"
            _save_ppo_checkpoint(model, best_path)
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

        if (
            cfg.opponents.mode == "league"
            and (update + 1) % cfg.opponents.snapshot_every == 0
        ):
            ckpt = Path(cfg.run.ckpt_root) / cfg.run.name / f"snapshot_{update:04d}.pt"
            pool.add_snapshot(f"{update:04d}", model, ckpt)
        elif (
            cfg.opponents.mode == "fixed"
            and (update + 1) % cfg.opponents.snapshot_every == 0
        ):
            _save_ppo_checkpoint(
                model,
                Path(cfg.run.ckpt_root) / cfg.run.name / "latest.pt",
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
                "batch_to_device_s": batch_to_device_s,
                "ppo_s": ppo_s,
                "metrics_s": metrics_s,
                "logging_s": logging_s,
                "snapshot_s": snapshot_s,
                "learner_steps_per_s": learner_steps_per_s,
                "end_to_end_steps_per_s": end_to_end_steps_per_s,
            },
            update,
        )
        logger.scalars(
            "charts",
            {
                "SPS": end_to_end_steps_per_s,
                "episodic_return": episodic_return,
                "episodic_length": episodic_length,
                "rollout_SPS": learner_steps_per_s,
                "games_per_minute": games_per_minute,
                "rollout_games_per_minute": len(trajs) * 60.0 / max(rollout_s, 1e-9),
            },
            update,
        )

    final_path = Path(cfg.run.ckpt_root) / cfg.run.name / "final.pt"
    _save_ppo_checkpoint(model, final_path)

    elo_path = final_path.with_name("elo.json")
    elo_path.write_text(json.dumps(elo.snapshot_dict(), indent=2, sort_keys=True))

    logger.close()
    summary["final_ckpt"] = str(final_path)
    summary["elo_path"] = str(elo_path)
    summary["elo_learner_final"] = elo.get(LEARNER_NAME)
    summary["cumulative_margin"] = cumulative_margin
    summary["cumulative_mean_margin"] = (
        cumulative_margin / max(1, cumulative_games)
    )
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
    summary = train_one_run(cfg, load_weights=args.load)
    print({k: v for k, v in summary.items() if k != "updates"})


if __name__ == "__main__":
    main()
