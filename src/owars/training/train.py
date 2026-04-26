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
    compute_gae,
    compute_mc_return,
    length_adaptive_lambda,
    ppo_update,
    value_only_update,
)
from .rollout import Trajectory, rollout_episode


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


def _stack_step_features(trajs: list[Trajectory]) -> dict[str, list]:
    pf, pm, pom, pid, pg, ff, fm = [], [], [], [], [], [], []
    tidx, frac, lp, owned = [], [], [], []
    for t in trajs:
        for step_i, e in enumerate(t.encoded):
            pf.append(e.planet_feats)
            pm.append(e.planet_mask)
            pom.append(e.planet_owned_mask)
            pid.append(e.planet_ids)
            pg.append(e.planet_garrison)
            ff.append(e.fleet_feats)
            fm.append(e.fleet_mask)
            tidx.append(torch.from_numpy(t.target_idx[step_i]))
            frac.append(torch.from_numpy(t.fraction[step_i]))
            lp.append(torch.from_numpy(t.log_prob[step_i]))
            owned.append(torch.from_numpy(t.owned_mask[step_i]))
    return dict(pf=pf, pm=pm, pom=pom, pid=pid, pg=pg, ff=ff, fm=fm,
                tidx=tidx, frac=frac, lp=lp, owned=owned)


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
    s = _stack_step_features(trajs)
    advs_all, rets_all = [], []

    for t in trajs:
        rewards = np.asarray(t.reward, dtype=np.float32)
        values = np.asarray(t.value, dtype=np.float32)
        T = len(rewards)
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
    return {
        "planet_feats": torch.stack(s["pf"]),
        "planet_mask": torch.stack(s["pm"]),
        "planet_owned_mask": torch.stack(s["pom"]),
        "planet_ids": torch.stack(s["pid"]),
        "planet_garrison": torch.stack(s["pg"]),
        "fleet_feats": torch.stack(s["ff"]),
        "fleet_mask": torch.stack(s["fm"]),
        "target_idx": torch.stack(s["tidx"]).long(),
        "fraction": torch.stack(s["frac"]).float(),
        "old_log_prob": torch.stack(s["lp"]).float(),
        "advantage": advs,
        "return": rets,
        "owned_mask": torch.stack(s["owned"]).bool(),
    }


def _pretrain_value_batch(trajs: list[Trajectory]) -> dict[str, torch.Tensor]:
    """Critic-target = pure Monte-Carlo trajectory return (γ=1, λ=1).

    With terminal-only ±1 reward this is a constant per trajectory =
    the eventual game outcome — i.e. supervised regression of V(s) onto
    the win indicator. Exactly the cold-start signal we want.
    """
    s = _stack_step_features(trajs)
    rets_all = [compute_mc_return(np.asarray(t.reward, dtype=np.float32), gamma=1.0) for t in trajs]
    rets = torch.from_numpy(np.concatenate(rets_all)).float()
    # Pretraining batch only needs the value-relevant fields.
    n = sum(len(t.reward) for t in trajs)
    zeros_b = torch.zeros(n, dtype=torch.float32)
    zeros_bp = torch.zeros((n, s["pf"][0].shape[0]), dtype=torch.float32)
    zeros_bp_long = torch.zeros((n, s["pf"][0].shape[0]), dtype=torch.long)
    zeros_bp_bool = torch.zeros((n, s["pf"][0].shape[0]), dtype=torch.bool)
    return {
        "planet_feats": torch.stack(s["pf"]),
        "planet_mask": torch.stack(s["pm"]),
        "planet_owned_mask": torch.stack(s["pom"]),
        "planet_ids": torch.stack(s["pid"]),
        "planet_garrison": torch.stack(s["pg"]),
        "fleet_feats": torch.stack(s["ff"]),
        "fleet_mask": torch.stack(s["fm"]),
        "target_idx": zeros_bp_long,
        "fraction": zeros_bp,
        "old_log_prob": zeros_bp,
        "advantage": zeros_b,
        "return": rets,
        "owned_mask": zeros_bp_bool,
    }


def pretrain_value(cfg: RunConfig, model: OrbitPolicy, optimizer: torch.optim.Optimizer,
                   logger: TBLogger, device: torch.device) -> None:
    """VAPO-style cold-start: regress V(s) onto trajectory outcome.

    Behavior policy is `cfg.ppo.pretrain_behavior` (defaults to
    `heuristic`). The actor parameters update too — that's fine, the
    critic shares the encoder and we only run a few hundred steps before
    PPO takes over.
    """
    if cfg.ppo.pretrain_updates <= 0:
        return
    behavior = BUILTIN.get(cfg.ppo.pretrain_behavior)
    if behavior is None:
        raise ValueError(f"unknown pretrain_behavior={cfg.ppo.pretrain_behavior!r}")
    others = [behavior] * (cfg.game.num_players - 1)

    for step in range(cfg.ppo.pretrain_updates):
        trajs: list[Trajectory] = []
        for _ in range(cfg.ppo.pretrain_episodes):
            traj = rollout_episode(
                model, others,
                num_players=cfg.game.num_players,
                episode_steps=cfg.game.episode_steps,
                ship_speed=cfg.game.ship_speed,
                device=str(device),
                deterministic=False,
                reward_cfg=cfg.reward,
            )
            trajs.append(traj)
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
        with torch.no_grad():
            from ..policies.features import EncodedObs

            class _F:
                pass

            f = _F()
            f.planet_feats = batch["planet_feats"]
            f.planet_mask = batch["planet_mask"]
            f.planet_owned_mask = batch["planet_owned_mask"]
            f.planet_ids = batch["planet_ids"]
            f.planet_garrison = batch["planet_garrison"]
            f.fleet_feats = batch["fleet_feats"]
            f.fleet_mask = batch["fleet_mask"]
            preds = model(f).value.cpu().numpy()
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

    Includes the encoder (shared backbone) and the value head; excludes
    actor heads (target_query/key, noop_logit, fraction_head). Including
    actor params here would be a silent bug: AdamW with weight_decay would
    pull them toward zero each step despite zero gradient signal, leaving
    PPO to start from a worse-than-init policy.
    """
    encoder = [model.planet_embed, model.fleet_embed, *model.layers]
    value = [model.value_head]
    params: list[torch.nn.Parameter] = []
    for m in encoder + value:
        params.extend(m.parameters())
    return params


def train_one_run(cfg: RunConfig) -> dict:
    set_seed(cfg.run.seed)
    device = torch.device(cfg.run.device if torch.cuda.is_available() else "cpu")
    model = _build_model(cfg).to(device)
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

    pretrain_value(cfg, model, pretrain_opt, logger, device)

    summary: dict = {"updates": []}

    for update in range(cfg.run.total_updates):
        trajs: list[Trajectory] = []
        play_count: dict[str, int] = defaultdict(int)
        win_count: dict[str, int] = defaultdict(int)

        for _ in range(cfg.rollout.episodes_per_update):
            slots = pool.sample(cfg.game.num_players - 1, current_model=model)
            traj = rollout_episode(
                model,
                [s.agent for s in slots],
                num_players=cfg.game.num_players,
                episode_steps=cfg.game.episode_steps,
                ship_speed=cfg.game.ship_speed,
                device=str(device),
                reward_cfg=cfg.reward,
            )
            trajs.append(traj)

            # Build per-seat (identity, score) and feed Elo. The learner's
            # seat is LEARNER_NAME; opponent seats carry their slot name
            # (LEARNER_NAME for self-play, "frozen:..." for snapshots).
            seat_names = _seat_names(traj.learner_seat, slots)
            elo.update_from_game(list(zip(seat_names, traj.seat_rewards)))

            # Per-opponent-identity win-rate tracking.
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
            model,
            optimizer,
            batch,
            clip_eps=cfg.ppo.clip_eps,
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
    args = p.parse_args()
    cfg = load_config(args.config)
    summary = train_one_run(cfg)
    print({k: v for k, v in summary.items() if k != "updates"})


if __name__ == "__main__":
    main()
