"""Vectorized rollouts: N parallel envs + one batched policy forward per step.

Per env-step the orchestrator does:

  1. Walk every alive (env, seat) and bucket the obs by *agent identity*
     — `LEARNER_NAME` for the live model (covers the learner's seat *and*
     any self-play opponent seats), or `frozen:<label>` for snapshots.
  2. Encode + stack the learner bucket into one big batch and run a
     single `model(...)` forward (this is where the wall-clock win comes
     from — at 16 envs × 2 seats with 80% self-play, the learner bucket
     averages ~25 obs per step instead of doing 25 separate B=1 forwards).
  3. For each snapshot bucket, run the snapshot's `LearnedAgent` per
     element (small batches; we don't bother batching across snapshots
     because a different snapshot = a different model).
  4. Distribute the resulting moves back into per-env action lists, then
     step the alive envs in parallel via `VecEnv.step_subset`.
  5. Trajectories are recorded *only for the learner_seat* of each env —
     self-play opponent seats are batched into the forward for compute,
     but their per-step records aren't collected (PPO trains on the
     learner's transitions only).
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import torch

from ..game import parse_observation
from ..policies.features import encode_observations, select_encoded
from ..policies.model import OrbitPolicy
from ..policies.sampling import sample_batch_actions, sample_batch_with_records
from .config import RewardCfg
from .league import LEARNER_NAME, OpponentSlot
from .rollout import Trajectory, record_step
from .vec_env import VecEnv


def _empty_traj() -> Trajectory:
    return Trajectory(
        encoded=[], target_idx=[], frac_z=[],
        log_prob=[], value=[], reward=[], owned_mask=[],
    )


def _resolve_seat_agents(
    opponents_per_env: list[list[OpponentSlot]],
    num_players: int,
    learner_seat: int,
) -> list[list[OpponentSlot | None]]:
    """For each env, return a list indexed by seat. `None` marks the
    learner's own seat; self-play opponent seats keep an OpponentSlot
    whose name is LEARNER_NAME (so they batch into the learner forward).
    """
    out: list[list[OpponentSlot | None]] = []
    for slots in opponents_per_env:
        per_seat: list[OpponentSlot | None] = [None] * num_players
        op_ix = 0
        for seat in range(num_players):
            if seat == learner_seat:
                continue  # already None
            per_seat[seat] = slots[op_ix]
            op_ix += 1
        out.append(per_seat)
    return out


def _finalize_trajectory(
    traj: Trajectory,
    final: Any,
    learner_seat: int,
    reward_cfg: RewardCfg,
) -> None:
    """Write seat_rewards / outcome onto a finished trajectory."""
    if final is None:
        return
    seat_rewards = [float(s.reward or 0.0) for s in final]
    learner_reward = seat_rewards[learner_seat]
    others = [r for i, r in enumerate(seat_rewards) if i != learner_seat]
    margin = learner_reward - max(others) if others else learner_reward
    traj.final_score = margin
    traj.won = learner_reward > max(others) if others else True
    traj.drawn = bool(others) and learner_reward == max(others)
    traj.seat_rewards = seat_rewards
    traj.learner_seat = learner_seat

    if traj.won:
        outcome = reward_cfg.win_value
    elif traj.drawn:
        outcome = reward_cfg.draw_value
    else:
        outcome = reward_cfg.loss_value
    if traj.reward:
        traj.reward[-1] += outcome + reward_cfg.margin_scale * margin


def rollout_episodes_batched(
    model: OrbitPolicy,
    vec: VecEnv,
    opponents_per_env: list[list[OpponentSlot]],
    *,
    num_players: int,
    device: str = "cpu",
    learner_seat: int = 0,
    deterministic: bool = False,
    reward_cfg: RewardCfg | None = None,
    record_trajectories: bool = True,
    max_moves_per_turn: int = 16,
) -> list[Trajectory]:
    """Play `len(opponents_per_env)` episodes in parallel; one Trajectory per env.

    `vec` is a long-lived `VecEnv` owned by the caller — we just call
    `vec.reset()` here. Spawning subprocess workers per call would burn
    seconds of pure startup time on each PPO update; the caller creates
    the pool once and reuses it.

    `opponents_per_env[i]` is a list of `num_players-1` `OpponentSlot`s for
    env `i` (the seat order skips `learner_seat`). The opponent assignment
    is fixed for the whole episode — sampled once by the caller before the
    rollout.
    """
    num_envs = len(opponents_per_env)
    assert num_envs == vec.num_envs, (
        f"opponents_per_env has {num_envs} entries but vec has {vec.num_envs} workers"
    )
    if reward_cfg is None:
        reward_cfg = RewardCfg()
    seat_agents = _resolve_seat_agents(opponents_per_env, num_players, learner_seat)

    trajectories = [_empty_traj() for _ in range(num_envs)]
    finals: list[Any] = [None] * num_envs

    states = vec.reset()
    dones = [False] * num_envs

    while not all(dones):
        # 1. Bucket (env, seat, obs) tuples by agent identity.
        learner_bucket: list[tuple[int, int, Any]] = []
        opp_buckets: dict[str, list[tuple[int, int, Any, OpponentSlot]]] = (
            defaultdict(list)
        )
        for env_idx in range(num_envs):
            if dones[env_idx]:
                continue
            state = states[env_idx]
            for seat in range(num_players):
                seat_state = state[seat]
                obs = seat_state["observation"]
                slot = seat_agents[env_idx][seat]
                if slot is None or slot.name == LEARNER_NAME:
                    learner_bucket.append((env_idx, seat, obs))
                else:
                    opp_buckets[slot.name].append((env_idx, seat, obs, slot))

        actions_per_env: dict[int, list[Any]] = {
            i: [None] * num_players for i in range(num_envs) if not dones[i]
        }

        # 2. One batched forward for the learner identity.
        if learner_bucket:
            _step_learner_bucket(
                model,
                learner_bucket,
                actions_per_env,
                trajectories,
                learner_seat,
                device,
                deterministic,
                record_trajectories,
                max_moves_per_turn,
            )

        # 3. Per-snapshot inference. Learned snapshots expose `act_batch`;
        # builtin Python baselines stay on the scalar callable path.
        for _name, bucket in opp_buckets.items():
            agent = bucket[0][3].agent
            act_batch = getattr(agent, "act_batch", None)
            if callable(act_batch):
                obs_list = [obs for _env_idx, _seat, obs, _slot in bucket]
                batched_actions = act_batch(obs_list)
                for (env_idx, seat, _obs, _slot), acts in zip(bucket, batched_actions):
                    actions_per_env[env_idx][seat] = acts
            else:
                for env_idx, seat, obs, slot in bucket:
                    actions_per_env[env_idx][seat] = slot.agent(obs)

        # 4. Step alive envs in parallel.
        active = [i for i in range(num_envs) if not dones[i]]
        actions_list = [actions_per_env[i] for i in active]
        results = vec.step_subset(active, actions_list)
        for i, (state, done, final) in results.items():
            states[i] = state
            if done:
                dones[i] = True
                finals[i] = final

    # 5. Apply terminal reward + record seat_rewards on each trajectory.
    for env_idx in range(num_envs):
        _finalize_trajectory(
            trajectories[env_idx], finals[env_idx], learner_seat, reward_cfg
        )

    return trajectories


def _step_learner_bucket(
    model: OrbitPolicy,
    bucket: list[tuple[int, int, Any]],
    actions_per_env: dict[int, list[Any]],
    trajectories: list[Trajectory],
    learner_seat: int,
    device: str,
    deterministic: bool,
    record_trajectories: bool,
    max_moves_per_turn: int,
) -> None:
    """Encode + batch-forward the learner identity across (env, seat) pairs.

    Self-play opponent seats batch in here too for compute efficiency, but
    only the seat that == `learner_seat` gets recorded into its trajectory
    — PPO trains on the learner's transitions, not the self-play side's.
    """
    parsed_list = [parse_observation(obs) for _, _, obs in bucket]
    stacked = encode_observations(
        parsed_list,
        device=device,
        pin_memory=torch.device(device).type == "cuda",
    )
    # bf16 autocast on CUDA is what unlocks FA-2 dispatch in
    # `SelfAttention.forward` — fp32 inputs make SDPA fall back to the
    # mem-efficient kernel. Same regime as `ppo_update`.
    autocast_enabled = torch.device(device).type == "cuda"
    with (
        torch.no_grad(),
        torch.autocast(
            device_type="cuda", dtype=torch.bfloat16, enabled=autocast_enabled
        ),
    ):
        out = model(stacked)
    if record_trajectories:
        moves_list, records = sample_batch_with_records(
            out,
            parsed_list,
            deterministic=deterministic,
            max_moves=max_moves_per_turn,
        )
    else:
        moves_list = sample_batch_actions(
            out,
            parsed_list,
            deterministic=deterministic,
            max_moves=max_moves_per_turn,
        )
        records = []

    for k, (env_idx, seat, _obs) in enumerate(bucket):
        actions_per_env[env_idx][seat] = [m.as_list() for m in moves_list[k]]
        if record_trajectories and seat == learner_seat:
            record_step(
                trajectories[env_idx],
                feats=select_encoded(stacked, k, clone=True),
                value_t=out.value[k],
                owned_mask_t=out.planet_owned_mask[k],
                record=records[k],
            )
