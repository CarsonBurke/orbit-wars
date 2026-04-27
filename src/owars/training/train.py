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
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

import random

from ..policies.config import OrbitPolicyConfig
from ..policies.model import OrbitPolicy
from ..utils import TBLogger, set_seed
from .config import RunConfig, load_config
from .elo import EloTracker
from .league import BUILTIN, LEARNER_NAME, OpponentPool, OpponentSlot
from .ppo import (
    _slice_feats,
    compute_gae,
    compute_mc_return,
    length_adaptive_lambda,
    ppo_update,
    value_only_update,
)
from .rollout import Trajectory
from .vec_env import VecEnv
from .vec_rollout import rollout_episodes_batched


def _build_model(cfg: RunConfig) -> OrbitPolicy:
    pcfg = OrbitPolicyConfig(
        dim=cfg.model.dim,
        ff_dim=cfg.model.ff_dim,
        depth=cfg.model.depth,
        n_heads=cfg.model.n_heads,
        dropout=cfg.model.dropout,
        fraction_concentration=cfg.model.fraction_concentration,
    )
    return OrbitPolicy(pcfg)


def _stack_encoded(trajs: list[Trajectory]) -> dict[str, torch.Tensor]:
    """Walk every (traj, step) once and emit stacked EncodedObs tensors.

    Per-step records on Trajectory are already device tensors (see the
    Trajectory docstring) — we just gather and stack here. Encoder-only:
    the actor-side records (target_idx / fraction / old_log_prob /
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

    tidx, frac, ang, lp, owned = [], [], [], [], []
    for t in trajs:
        tidx.extend(t.target_idx)
        frac.extend(t.fraction)
        ang.extend(t.angle_offset)
        lp.extend(t.log_prob)
        owned.extend(t.owned_mask)
    batch["target_idx"] = torch.stack(tidx).long()
    batch["fraction"] = torch.stack(frac).float()
    batch["angle_offset"] = torch.stack(ang).float()
    batch["old_log_prob"] = torch.stack(lp).float()
    batch["owned_mask"] = torch.stack(owned).bool()

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
    # Normalize once over the full batch — see ppo_update for why.
    batch["advantage"] = (advs - advs.mean()) / (advs.std() + 1e-8)
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
        for _ in range(batches):
            trajs.extend(rollout_episodes_batched(
                model,
                vec,
                opponents_per_env,
                num_players=cfg.game.num_players,
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
        with torch.no_grad():
            chunks = [
                model(_slice_feats(batch, slice(s, s + mb))).value
                for s in range(0, n, mb)
            ]
            preds = torch.cat(chunks).cpu().numpy()
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
    heads (target_query/key, noop_logit, fraction_head, angle_head) — they
    receive zero gradient from the value loss, and including them would let
    AdamW's weight-decay pull them toward zero with no learning signal,
    leaving PPO to start from a worse-than-init policy.
    """
    encoder = [model.planet_embed, model.fleet_embed, *model.layers]
    value = [model.value_head]
    params: list[torch.nn.Parameter] = [model.actor_token, model.critic_token]
    for m in encoder + value:
        params.extend(m.parameters())
    return params


def train_one_run(cfg: RunConfig, load_weights: str | None = None) -> dict:
    set_seed(cfg.run.seed)
    device = torch.device(cfg.run.device if torch.cuda.is_available() else "cpu")
    model = _build_model(cfg).to(device)
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
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.optim.lr,
        weight_decay=cfg.optim.weight_decay,
    )

    elo = EloTracker(
        initial_rating=cfg.opponents.initial_rating,
        k_factor=cfg.opponents.k_factor,
    )
    elo.ensure(LEARNER_NAME)
    pool = OpponentPool(
        elo=elo,
        top_k=cfg.opponents.top_k,
        self_play_prob=cfg.opponents.self_play_prob,
        rng=random.Random(cfg.run.seed),
    )
    logger = TBLogger(cfg.run.name, root=cfg.run.log_root)

    # One subprocess pool reused across pretraining + every PPO update.
    # Spawning per-call cost ~16-32 s of pure interpreter startup × every
    # rollout (cumulative ~1 h on a full run); `vec.reset()` is cheap.
    # `replay_env_idx=0` keeps env 0's full step history so the PPO loop
    # can dump one rendered game per update; the other workers trim
    # `env.steps` to save memory.
    with VecEnv(
        num_envs=cfg.rollout.num_envs,
        num_players=cfg.game.num_players,
        episode_steps=cfg.game.episode_steps,
        ship_speed=cfg.game.ship_speed,
        replay_env_idx=0,
    ) as vec:
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

    # Compile *only* the PPO-update forward+backward path. The minibatch
    # shape there is fixed at `[minibatch_size, MAX_PLANETS, ...]` and gets
    # called epochs × ⌈n/mb⌉ times per update (~32×200 invocations on the
    # default config) — one-time inductor cost amortizes cleanly. We keep
    # the *rollout* using the raw `model` because its bucket size varies
    # per env-step (envs finish on different steps), which would force
    # recompiles. Same module, same parameters — only the forward dispatch
    # differs. Snapshotting (deepcopy) and `_value_pretrain_params` also
    # operate on the raw module.
    train_model: OrbitPolicy = model
    if device.type == "cuda":
        train_model = torch.compile(model)  # type: ignore[assignment]

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
        trajs = rollout_episodes_batched(
            model,
            vec,
            opponents_per_env,
            num_players=cfg.game.num_players,
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
            elo.update_from_game(list(zip(seat_names, traj.seat_rewards)))
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
            clip_eps_low=cfg.ppo.clip_eps_low,
            clip_eps_high=cfg.ppo.clip_eps_high,
            value_coef=cfg.ppo.value_coef,
            entropy_coef=cfg.ppo.entropy_coef,
            epochs=cfg.optim.epochs_per_update,
            minibatch_size=cfg.optim.minibatch_size,
            grad_clip=cfg.optim.grad_clip,
        )

        win_rate = float(np.mean([t.won for t in trajs]))
        margin = float(np.mean([t.final_score for t in trajs]))
        snapshot_elos = [elo.get(n) for n in pool.snapshot_names()]
        logger.scalars(
            "train",
            {
                "policy_loss": log.policy_loss,
                "value_loss": log.value_loss,
                "entropy": log.entropy,
                "approx_kl": log.approx_kl,
                "clip_frac": log.clip_frac,
                "win_rate": win_rate,
                "margin": margin,
                "elo_learner": elo.get(LEARNER_NAME),
                "elo_pool_size": float(len(snapshot_elos)),
                "elo_pool_max": max(snapshot_elos) if snapshot_elos else float("nan"),
                "elo_pool_min": min(snapshot_elos) if snapshot_elos else float("nan"),
            },
            update,
        )
        summary["updates"].append(
            {
                "update": update,
                "win_rate": win_rate,
                "margin": margin,
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
