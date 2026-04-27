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

import torch

from ..game import parse_observation
from ..policies.features import EncodedObs, encode_observation
from ..policies.model import OrbitPolicy
from ..policies.sampling import SampleRecord, sample_with_record
from .config import RewardCfg

AgentFn = Callable[[Any], list[list]]


@dataclass
class Trajectory:
    """Per-step records for the *learning* agent only.

    Per-step tensors stay on the *rollout device* (i.e. wherever the
    policy ran). Pulling them to CPU per-step would force a stream
    sync per env-step — at 16 envs × 500 steps that's 8000 syncs per
    PPO update on GPU. Instead we accumulate device tensors and
    `.cpu()` the whole list once when building the PPO batch.
    """

    encoded: list[EncodedObs]
    target_idx: list[torch.Tensor]    # [P] long
    fraction: list[torch.Tensor]      # [P] float in [0, 1]
    angle_offset: list[torch.Tensor]  # [P] float in [0, 1] — pre-rescale Beta sample
    log_prob: list[torch.Tensor]      # [P] float
    value: list[torch.Tensor]         # scalar tensors (no .item() in the hot loop)
    reward: list[float]
    owned_mask: list[torch.Tensor]    # [P] bool
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
        encoded=[], target_idx=[], fraction=[], angle_offset=[],
        log_prob=[], value=[], reward=[], owned_mask=[],
    )

    while not env.done:
        actions: list[list] = []
        for seat, slot in enumerate(state):
            if agents[seat] == "__learner__":
                obs = slot["observation"]
                acts, info = _policy_step(model, obs, device, deterministic)
                actions.append(acts)
                _record_step(traj, info)
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


def record_step(
    traj: Trajectory,
    feats: EncodedObs,
    value_t: torch.Tensor,
    owned_mask_t: torch.Tensor,
    record: Any,
) -> None:
    """Append one step's per-planet records into a `Trajectory`.

    All five tensor inputs are stored as-is on the rollout device — see
    `Trajectory`'s docstring for why we don't pull to CPU here.
    """
    traj.encoded.append(feats)
    traj.target_idx.append(record.target_idx.detach())
    traj.fraction.append(record.fraction.detach())
    traj.angle_offset.append(record.angle_offset.detach())
    traj.log_prob.append(record.log_prob.detach())
    traj.value.append(value_t.detach())
    traj.owned_mask.append(owned_mask_t.detach())
    traj.reward.append(0.0)


def _record_step(traj: Trajectory, info: dict) -> None:
    """Serial-rollout adapter — pulls slot 0 out of the model output."""
    out = info["policy_out"]
    record_step(
        traj,
        feats=info["feats"],
        value_t=out.value[0],
        owned_mask_t=out.planet_owned_mask[0],
        record=info["record"],
    )
