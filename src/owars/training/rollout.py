"""Roll out a single Orbit Wars episode and collect a trajectory.

Uses `kaggle_environments.make("orbit_wars")` so we train against the
*exact* simulator the leaderboard runs. The cost: each step pays the env's
deserialization overhead, so PPO iteration speed is bounded by env stepping.
For first-pass training that's fine; if it becomes a bottleneck the natural
next move is to vendor a numpy reimplementation (see `STRATEGY.md`).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import torch

from ..game import parse_observation
from ..policies.features import EncodedObs, encode_observation
from ..policies.model import OrbitPolicy
from ..policies.sampling import sample_with_record
from .config import RewardCfg

AgentFn = Callable[[Any], list[list]]


@dataclass
class Trajectory:
    """Per-step records for the *learning* agent only."""

    encoded: list[EncodedObs]
    target_idx: list[np.ndarray]   # [P] long, only owned slots are "active"
    fraction: list[np.ndarray]     # [P] float in [0, 1]
    log_prob: list[np.ndarray]     # [P] float
    value: list[float]
    reward: list[float]
    owned_mask: list[np.ndarray]   # [P] bool — which slots had a real action
    final_score: float = 0.0
    won: bool = False
    drawn: bool = False
    seat_rewards: list[float] = field(default_factory=list)  # all seats, in seat order
    learner_seat: int = 0


def make_env(num_players: int, episode_steps: int, ship_speed: float, debug: bool = False):
    """Lazy import so test environments without kaggle-environments still work."""
    from kaggle_environments import make  # type: ignore[import-not-found]

    return make(
        "orbit_wars",
        configuration={
            "episodeSteps": episode_steps,
            "shipSpeed": ship_speed,
        },
        debug=debug,
    )


def _policy_step(
    model: OrbitPolicy, obs: Any, device: str, deterministic: bool
) -> tuple[list[list], dict]:
    parsed = parse_observation(obs)
    feats = encode_observation(parsed, device=device)
    out = model(feats)
    moves, record = sample_with_record(out, parsed, deterministic=deterministic)
    return [m.as_list() for m in moves], {
        "feats": feats,
        "policy_out": out,
        "moves": moves,
        "parsed": parsed,
        "record": record,
    }


def rollout_episode(
    model: OrbitPolicy,
    opponents: list[AgentFn],
    *,
    num_players: int,
    episode_steps: int,
    ship_speed: float,
    learner_seat: int = 0,
    device: str = "cpu",
    deterministic: bool = False,
    reward_cfg: RewardCfg | None = None,
) -> Trajectory:
    """Play one episode; return a `Trajectory` for the learner's seat.

    `opponents` is a list of length `num_players - 1`; they fill the other
    seats in order.
    """
    if len(opponents) != num_players - 1:
        raise ValueError(
            f"need {num_players - 1} opponents for {num_players}-player game"
        )

    if reward_cfg is None:
        reward_cfg = RewardCfg()

    env = make_env(num_players, episode_steps, ship_speed)
    agents: list[Any] = []
    learner_ix = learner_seat
    op_ix = 0
    for seat in range(num_players):
        if seat == learner_ix:
            agents.append("__learner__")
        else:
            agents.append(opponents[op_ix])
            op_ix += 1

    state = env.reset(num_agents=num_players)

    traj = Trajectory(
        encoded=[], target_idx=[], fraction=[], log_prob=[], value=[],
        reward=[], owned_mask=[],
    )

    while not env.done:
        actions: list[list] = []
        for seat, slot in enumerate(state):
            if agents[seat] == "__learner__":
                obs = slot["observation"]
                acts, info = _policy_step(model, obs, device, deterministic)
                actions.append(acts)
                _record_step(traj, info, env)
            else:
                obs = slot["observation"]
                actions.append(agents[seat](obs))
        state = env.step(actions)

    final = env.steps[-1]
    seat_rewards = [float(s.reward or 0.0) for s in final]
    learner_reward = seat_rewards[learner_ix]
    others = [r for i, r in enumerate(seat_rewards) if i != learner_ix]
    margin = learner_reward - max(others) if others else learner_reward
    traj.final_score = margin
    traj.won = learner_reward > max(others) if others else True
    traj.drawn = bool(others) and learner_reward == max(others)
    traj.seat_rewards = seat_rewards
    traj.learner_seat = learner_ix

    if traj.won:
        outcome = reward_cfg.win_value
    elif traj.drawn:
        outcome = reward_cfg.draw_value
    else:
        outcome = reward_cfg.loss_value
    if traj.reward:
        traj.reward[-1] += outcome + reward_cfg.margin_scale * margin
    return traj


def _record_step(traj: Trajectory, info: dict, env) -> None:
    out = info["policy_out"]
    record = info["record"]

    # The sampler already produced the per-planet action and its log-prob;
    # PPO's importance ratio depends on these being the *actual* sampled
    # action, not a re-derivation from the move list.
    target_idx = record.target_idx.detach().cpu().numpy().astype(np.int64)
    fraction = record.fraction.detach().cpu().numpy().astype(np.float32)
    log_prob = record.log_prob.detach().cpu().numpy().astype(np.float32)
    owned = out.planet_owned_mask[0].cpu().numpy()

    traj.encoded.append(info["feats"])
    traj.target_idx.append(target_idx)
    traj.fraction.append(fraction)
    traj.log_prob.append(log_prob)
    traj.value.append(float(out.value[0].item()))
    traj.owned_mask.append(owned.copy())
    traj.reward.append(0.0)
