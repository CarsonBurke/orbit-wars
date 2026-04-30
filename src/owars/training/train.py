"""Single-run training driver.

Loop:
  0. (Optional) **Value pretraining** — roll N episodes against a frozen
     behavior policy, compute MC returns (γ=1, λ=1 → trajectory outcome
     for every state), fit the critic with MSE only. Gives PPO a warm
     value function so the actor's advantage isn't garbage at step 0.
     This is the single highest-leverage knob from VAPO/VC-PPO.
  1. For each PPO update:
     a. Sample N episodes against opponents from the pool. Per opponent
        slot, with probability `self_play_prob` (default 0.8) the seat is
        filled by the live learner; otherwise by a uniformly-chosen
        snapshot from the top-K by Elo.
     b. **Decoupled GAE**: critic target uses λ_critic=1 (Monte-Carlo,
        unbiased); actor advantage uses λ_policy < 1 (variance-reduced,
        optionally length-adaptive).
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
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from ..policies.config import OrbitPolicyConfig
from ..policies.model import OrbitPolicy, restore_fp32_params
from ..utils import TBLogger, set_seed
from .config import OptimCfg, RunConfig, load_config
from .elo import EloTracker
from .league import BUILTIN, LEARNER_NAME, OpponentPool, OpponentSlot
from .muon import MultiOptimizer, Muon
from .numpy_env import NumpyVecEnv
from .ppo import (
    _slice_feats,
    compute_gae,
    compute_mc_return,
    length_adaptive_lambda,
    ppo_update,
    value_only_update,
)
from .rollout import Trajectory
from .sharded_numpy_env import ShardedNumpyVecEnv
from .vec_env import VecEnv
from .vec_rollout import alternating_learner_seats, rollout_episodes_batched

# Parameter-name patterns that should *never* go to Muon even when shape
# is 2D. These are control tensors (per-channel scales, residual mixes)
# that conceptually act as scalars-per-channel; orthogonalizing their
# 2D shape would destroy the per-channel meaning. Mirrors parameter-golf
# `CONTROL_TENSOR_NAME_PATTERNS`.
_CONTROL_TENSOR_PATTERNS: tuple[str, ...] = (
    "attn_scale",
    "ff_scale",
    "resid_mix",
    "actor_token",
    "critic_token",
)

# Subset of control tensors that route to the *fast* AdamW group at
# `control_lr` (≈ `muon_lr`) — per-channel residual scales and the
# attention-temperature gains (trunk `q_gain` + `target_q_gain`). The
# summary tokens (`actor_token`, `critic_token`) intentionally stay in
# the slow default group: they are learnable biases on the residual
# stream and moving them at scalar speed destabilizes early training.
_CONTROL_LR_PATTERNS: tuple[str, ...] = (
    "attn_scale",
    "ff_scale",
    "resid_mix",
    "q_gain",  # also matches `target_q_gain` via substring
)

# Parameter-name patterns for the *slow* Muon group: action-head readouts
# that produce policy logits / fraction params. These get a lower Muon LR
# than the trunk (parameter-golf `head_lr=0.008` vs `matrix_lr=0.022`),
# because every spectral-norm step here translates ~directly into Δlogit
# / Δμ / Δlog σ → ratio drift → approx_kl. `value_head` is excluded — its
# updates don't reach the policy.
_HEAD_LR_PATTERNS: tuple[str, ...] = (
    "target_query",
    "target_key",
    "fraction_head",
    "launch_head",
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
    (muon_trunk, muon_head, adamw_default, adamw_control).

    Muon (trunk): 2D weight matrices that aren't control tensors and
    aren't action-head readouts — encoder block weights, projection
    embeds, value-head matrices.

    Muon (head): action-head readout matrices — `target_query`,
    `target_key`, `fraction_head`, `launch_head`. Same Muon optimizer
    state, slower LR.

    AdamW (control-lr): per-channel residual scales and `q_gain`s — need
    update magnitudes comparable to Muon's matrix updates, see
    `OptimCfg.control_lr`.

    AdamW (default-lr): everything else — biases, summary tokens.
    """
    muon_trunk: list[torch.nn.Parameter] = []
    muon_head: list[torch.nn.Parameter] = []
    adamw_default: list[torch.nn.Parameter] = []
    adamw_control: list[torch.nn.Parameter] = []
    for name, p in model.named_parameters():
        is_control = any(pat in name for pat in _CONTROL_TENSOR_PATTERNS)
        is_control_lr = any(pat in name for pat in _CONTROL_LR_PATTERNS)
        is_head_lr = any(pat in name for pat in _HEAD_LR_PATTERNS)
        if p.ndim == 2 and not is_control:
            if is_head_lr:
                muon_head.append(p)
            else:
                muon_trunk.append(p)
        elif is_control_lr:
            adamw_control.append(p)
        else:
            adamw_default.append(p)
    return muon_trunk, muon_head, adamw_default, adamw_control


def _build_optimizer(model: OrbitPolicy, cfg: OptimCfg) -> MultiOptimizer:
    """Construct the dual Muon + AdamW optimizer.

    See `OptimCfg` and `muon.py` for rationale. The combined object exposes
    `step` / `zero_grad` / `param_groups` so the PPO loop's clip-grad and
    step calls work transparently across both children.
    """
    muon_trunk, muon_head, adamw_default, adamw_control = _split_params(model)
    # Two Muon param-groups: trunk at `muon_lr`, action-head readouts at
    # the slower `muon_head_lr`. One Muon optimizer instance keeps the
    # NS5 step counter (and momentum-warmup schedule) shared across both
    # groups — same opt-step pacing, different per-group LR.
    muon_opt = Muon(
        [
            {"params": muon_trunk, "lr": cfg.muon_lr},
            {"params": muon_head, "lr": cfg.muon_head_lr},
        ],
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
    # Two AdamW param-groups: control tensors at `control_lr` (≈ muon_lr,
    # parity with matrix updates) and everything else at `lr`.
    adamw_opt = torch.optim.AdamW(
        [
            {"params": adamw_default, "lr": cfg.lr},
            {"params": adamw_control, "lr": cfg.control_lr},
        ],
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
        betas=(0.9, 0.95),
        fused=True,
    )
    return MultiOptimizer([muon_opt, adamw_opt])


def _build_model(cfg: RunConfig) -> OrbitPolicy:
    pcfg = OrbitPolicyConfig(
        dim=cfg.model.dim,
        ff_dim=cfg.model.ff_dim,
        depth=cfg.model.depth,
        n_heads=cfg.model.n_heads,
        dropout=cfg.model.dropout,
        encoder_backend=cfg.model.encoder_backend,
        value_hidden=cfg.model.value_hidden,
        value_num_bins=cfg.model.value_num_bins,
        value_min=cfg.model.value_min,
        value_max=cfg.model.value_max,
    )
    return OrbitPolicy(pcfg)


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
    lambda_critic: float,
    lambda_policy: float,
    lambda_policy_alpha: float,
) -> dict[str, torch.Tensor]:
    """Flatten per-step records into one batch with **decoupled GAE**.

    Critic target = GAE-λ_critic returns (typically λ=1 → MC return).
    Actor advantage = GAE-λ_policy advantages (optionally length-adaptive).
    """
    batch = _stack_encoded(trajs)

    launch, tidx, frac, lp, owned = [], [], [], [], []
    olaunch, otl, oalpha, obeta = [], [], [], []
    for t in trajs:
        launch.extend(t.launch)
        tidx.extend(t.target_idx)
        frac.extend(t.fraction)
        lp.extend(t.log_prob)
        owned.extend(t.owned_mask)
        olaunch.extend(t.old_launch_logits)
        otl.extend(t.old_target_logits)
        oalpha.extend(t.old_fraction_alpha)
        obeta.extend(t.old_fraction_beta)
    batch["launch"] = torch.stack(launch).float()
    batch["target_idx"] = torch.stack(tidx).long()
    batch["fraction"] = torch.stack(frac).float()
    batch["old_log_prob"] = torch.stack(lp).float()
    batch["owned_mask"] = torch.stack(owned).bool()
    batch["old_launch_logits"] = torch.stack(olaunch).float()
    batch["old_target_logits"] = torch.stack(otl).float()
    batch["old_fraction_alpha"] = torch.stack(oalpha).float()
    batch["old_fraction_beta"] = torch.stack(obeta).float()

    # Single global CPU pull of every per-step value across the batch —
    # one sync instead of one per trajectory.
    value_tensors = [torch.stack(t.value) for t in trajs if t.value]
    if value_tensors:
        all_values = torch.cat(value_tensors).detach().to(torch.float32).cpu().numpy()
    else:
        all_values = np.zeros(0, dtype=np.float32)

    advs_all, rets_all = [], []
    offset = 0
    for t in trajs:
        rewards = np.asarray(t.reward, dtype=np.float32)
        T = len(rewards)
        values = all_values[offset : offset + T]
        offset += T
        if lambda_policy_alpha > 0.0:
            lam_p = length_adaptive_lambda(T, lambda_policy_alpha)
        else:
            lam_p = lambda_policy
        adv_p, _ = compute_gae(rewards, values, gamma, lam_p)
        _, ret_c = compute_gae(rewards, values, gamma, lambda_critic)
        advs_all.append(adv_p)
        rets_all.append(ret_c)

    advs = torch.from_numpy(np.concatenate(advs_all)).float()
    rets = torch.from_numpy(np.concatenate(rets_all)).float()
    # PMPO uses `tanh(adv).abs()` magnitude shaping, which is bounded in
    # [0, 1) regardless of advantage scale — so we deliberately do *not*
    # z-score advantages here (dreamer4 `dreamer4.py:4130` makes the same
    # choice: `normalize_advantages = default(None, not use_pmpo)`). The
    # raw advantage signal carries through to PMPO's pos/neg split.
    batch["advantage"] = advs
    batch["return"] = rets
    return batch


def _pretrain_value_batch(trajs: list[Trajectory]) -> dict[str, torch.Tensor]:
    """Critic-target = pure Monte-Carlo trajectory return (γ=1, λ=1).

    With terminal-only ±1 reward this is a constant per trajectory =
    the eventual game outcome — i.e. supervised regression of V(s) onto
    the win indicator. Exactly the cold-start signal we want.

    Encoder fields only — `value_only_update` doesn't read the actor-side
    records, so we skip stacking and host→device-copying them.
    """
    batch = _stack_encoded(trajs)
    rets_all = [compute_mc_return(np.asarray(t.reward, dtype=np.float32), gamma=1.0) for t in trajs]
    batch["return"] = torch.from_numpy(np.concatenate(rets_all)).float()
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
            ))
        batch = {k: v.to(device) for k, v in _pretrain_value_batch(trajs).items()}
        loss = value_only_update(
            model, optimizer, batch,
            epochs=1,
            minibatch_size=cfg.optim.minibatch_size,
            grad_clip=cfg.optim.grad_clip,
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


def _value_pretrain_params(model: OrbitPolicy) -> list[torch.nn.Parameter]:
    """Params that *actually* get gradient from value-only loss.

    Includes the encoder (shared backbone), both summary tokens (actor_token
    feeds the encoder self-attention so h_critic depends on it; critic_token
    feeds the value head directly), and the value head. Excludes the actor
    heads (target_query/key, launch_head, fraction_head) — they receive zero
    gradient from the value loss, and including them would let AdamW's
    weight-decay pull them toward zero with no learning signal, leaving PPO
    to start from a worse-than-init policy.
    """
    encoder = [model.planet_embed, model.fleet_embed, *model.layers]
    value = [model.value_head]
    params: list[torch.nn.Parameter] = [model.actor_token, model.critic_token]
    for m in encoder + value:
        params.extend(m.parameters())
    return params


def train_one_run(cfg: RunConfig, load_weights: str | None = None) -> dict:
    set_seed(cfg.run.seed)
    if cfg.run.torch_num_threads > 0:
        torch.set_num_threads(cfg.run.torch_num_threads)
        try:
            torch.set_num_interop_threads(max(1, cfg.run.torch_num_threads))
        except RuntimeError:
            pass
    if cfg.run.device != "cuda":
        raise ValueError("training is CUDA-only; set run.device: cuda")
    if not torch.cuda.is_available():
        raise RuntimeError("training requires CUDA")
    device = torch.device("cuda")
    # parameter-golf `sota_train_gpt.py:465` pins SDPA to flash-only. We
    # *do not* — the nested-jagged SDPA dispatcher (`torch/nested/_internal/
    # sdpa.py`) only has flash and math jagged kernels; mem_efficient and
    # cudnn aren't reachable through the jagged path at all, and flash's
    # eligibility heuristic rejects small-batch rollouts → math is the only
    # fallback. Disabling math caused "No viable backend" during rollout.
    # Leaving the default backends in place: jagged forwards consistently
    # pick flash when it's eligible and math otherwise.
    model = _build_model(cfg).to(device)
    # parameter-golf fp32-master pattern: cast everything to bf16, then
    # restore fp32 for the params that actually need precision (Linear
    # weights, biases, control tensors, summary tokens). This is the
    # explicit equivalent of relying on autocast's implicit weight
    # casting — but with a deterministic dtype boundary that doesn't
    # fight `torch.compile` or nested-jagged subclass tracking.
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
        model.load_state_dict(state)
        print(f"loaded weights from {load_weights}")
    pretrain_opt = torch.optim.AdamW(
        _value_pretrain_params(model),
        lr=cfg.ppo.pretrain_lr,
        weight_decay=cfg.optim.weight_decay,
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
    if cfg.rollout.env_backend not in {"kaggle", "numpy", "numpy_mp"}:
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
    pool: OpponentPool,
    logger: TBLogger,
    device: torch.device,
    vec: VecEnv,
) -> dict:
    summary: dict = {"updates": []}
    cumulative_margin = 0.0
    cumulative_win_margin = 0.0
    cumulative_loss_margin = 0.0
    cumulative_games = 0

    # Seed the pool with a snapshot of the random-init model. Without this,
    # `_sample_one` returns LEARNER_NAME for every slot until the first
    # snapshot lands at `snapshot_every`, every game is learner-vs-learner,
    # `update_from_game` early-returns on a single identity, and Elo stays
    # frozen. The init snapshot is *not* pinned — if it's bad it'll lose
    # rating and UCB-eviction will cull it like any other weak snapshot.
    init_ckpt = Path(cfg.run.ckpt_root) / cfg.run.name / "snapshot_init.pt"
    pool.add_snapshot("init", model, init_ckpt)

    # Compile only the PPO-update model forward/backward path. The summary
    # tokens are stored flat so AOTAutograd's broadcast reductions match the
    # parameter shapes at larger PPO batch sizes.
    train_model: OrbitPolicy = model
    if device.type == "cuda":
        train_model = torch.compile(model, dynamic=False, fullgraph=True)  # type: ignore[assignment]

    # One rendered game per update lands here (env 0 is the recording
    # worker; see VecEnv(replay_env_idx=0) above). Pretrain disabled
    # recording; turn it back on for the PPO loop.
    vec.set_recording(True)
    replays_dir = logger.path / "replays"
    replays_dir.mkdir(parents=True, exist_ok=True)

    for update in range(cfg.run.total_updates):
        play_count: dict[str, int] = defaultdict(int)
        win_count: dict[str, int] = defaultdict(int)

        # Sample opponents once per env, then play all envs in parallel.
        # Each env's seat assignment is fixed for the episode; the rollout
        # batches the *policy forward* across envs each step.
        opponents_per_env = [
            pool.sample(cfg.game.num_players - 1)
            for _ in range(cfg.rollout.num_envs)
        ]
        learner_seats = alternating_learner_seats(
            cfg.rollout.num_envs, cfg.game.num_players, offset=update
        )
        trajs = rollout_episodes_batched(
            model,
            vec,
            opponents_per_env,
            num_players=cfg.game.num_players,
            learner_seat=learner_seats,
            device=str(device),
            reward_cfg=cfg.reward,
        )

        if vec.last_replay_html is not None:
            (replays_dir / f"update_{update:04d}.html").write_text(
                vec.last_replay_html
            )

        for env_idx, traj in enumerate(trajs):
            slots = opponents_per_env[env_idx]
            seat_names = _seat_names(traj.learner_seat, slots)
            elo.update_from_game(list(zip(seat_names, traj.seat_rewards, strict=True)))
            for s in slots:
                play_count[s.name] += 1
                if traj.won:
                    win_count[s.name] += 1

        batch = _stack_trajectories(
            trajs,
            gamma=cfg.ppo.gamma,
            lambda_critic=cfg.ppo.lambda_critic,
            lambda_policy=cfg.ppo.lambda_policy,
            lambda_policy_alpha=cfg.ppo.lambda_policy_alpha,
        )
        batch = {k: v.to(device) for k, v in batch.items()}

        log = ppo_update(
            train_model,
            optimizer,
            batch,
            value_coef=cfg.ppo.value_coef,
            target_entropy_coef=cfg.ppo.target_entropy_coef,
            fraction_entropy_coef=cfg.ppo.fraction_entropy_coef,
            pmpo_kl_coef=cfg.ppo.pmpo_kl_coef,
            pmpo_pos_to_neg_weight=cfg.ppo.pmpo_pos_to_neg_weight,
            pmpo_reverse_kl=cfg.ppo.pmpo_reverse_kl,
            epochs=cfg.optim.epochs_per_update,
            minibatch_size=cfg.optim.minibatch_size,
            grad_clip=cfg.optim.grad_clip,
        )

        margins = [float(t.final_score) for t in trajs]
        win_rate = float(np.mean([t.won for t in trajs]))
        margin = float(np.mean(margins))
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
        logger.scalars(
            "loss",
            {
                "policy": log.policy_loss,
                "value": log.value_loss,
            },
            update,
        )
        logger.scalars(
            "kl",
            {
                "approx": log.approx_kl,
                "pmpo": log.pmpo_kl,
                "pmpo_target": log.pmpo_target_kl,
                "pmpo_fraction": log.pmpo_fraction_kl,
            },
            update,
        )
        logger.scalars(
            "policy",
            {
                "entropy": log.entropy,
                "target_entropy": log.target_entropy,
                "fraction_entropy": log.fraction_entropy,
                "target_confidence": log.target_confidence,
                "move_prob": log.move_prob,
                "pos_advantage_frac": log.pos_frac,
            },
            update,
        )
        logger.scalars(
            "fraction",
            {
                "alpha_mean": log.fraction_alpha_mean,
                "alpha_max": log.fraction_alpha_max,
                "beta_mean": log.fraction_beta_mean,
                "beta_max": log.fraction_beta_max,
                "mode_mean": log.fraction_mode_mean,
                "concentration_mean": log.fraction_concentration_mean,
                "concentration_max": log.fraction_concentration_max,
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
        summary["updates"].append(
            {
                "update": update,
                "win_rate": win_rate,
                "margin": margin,
                "cumulative_margin": cumulative_margin,
                "cumulative_mean_margin": cumulative_mean_margin,
                "elo_learner": elo.get(LEARNER_NAME),
            }
        )

        if (update + 1) % cfg.opponents.snapshot_every == 0:
            ckpt = Path(cfg.run.ckpt_root) / cfg.run.name / f"snapshot_{update:04d}.pt"
            pool.add_snapshot(f"{update:04d}", model, ckpt)

    final_path = Path(cfg.run.ckpt_root) / cfg.run.name / "final.pt"
    final_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "config": model.cfg.to_dict()}, final_path)

    import json

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
    p.add_argument(
        "--load",
        default=None,
        help="Path to a .pt checkpoint whose `model` state_dict should be "
        "loaded into the policy before training starts. Optimizer state and "
        "the opponent pool are NOT restored.",
    )
    args = p.parse_args()
    cfg = load_config(args.config)
    summary = train_one_run(cfg, load_weights=args.load)
    print({k: v for k, v in summary.items() if k != "updates"})


if __name__ == "__main__":
    main()
