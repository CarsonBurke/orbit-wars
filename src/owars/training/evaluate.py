"""Evaluate a trained checkpoint by playing N games against each baseline.

Reports win-rate, mean margin, and a Glicko-style rating per matchup. Run
this after training to confirm the policy actually improved against
*non-mirror* opponents (self-play win-rate alone is not enough).
"""

from __future__ import annotations

import argparse

import numpy as np
import torch

from ..policies.config import OrbitPolicyConfig
from ..policies.model import OrbitPolicy
from .league import BUILTIN, OpponentSlot
from .numpy_env import NumpyVecEnv
from .rollout import rollout_episode
from .sharded_numpy_env import ShardedNumpyVecEnv
from .vec_env import VecEnv
from .vec_rollout import alternating_learner_seats, rollout_episodes_batched


def evaluate_ckpt(
    ckpt_path: str,
    *,
    n_games: int = 50,
    num_players: int = 2,
    episode_steps: int = 500,
    ship_speed: float = 6.0,
    device: str = "cpu",
    baselines: tuple[str, ...] = ("random", "sniper", "heuristic"),
    num_envs: int = 16,
    env_backend: str = "kaggle",
    num_workers: int = 0,
) -> dict:
    state = torch.load(ckpt_path, map_location=device)
    cfg = OrbitPolicyConfig(**state["config"])
    model = OrbitPolicy(cfg).to(device)
    model.load_state_dict(state["model"])
    model.eval()

    if env_backend not in {"kaggle", "numpy", "numpy_mp"}:
        raise ValueError(f"unknown env_backend: {env_backend!r}")
    out: dict[str, dict[str, float]] = {}
    for opp_name in baselines:
        opp = BUILTIN[opp_name]
        opp_slot = OpponentSlot(name=opp_name, agent=opp)
        envs = max(1, min(num_envs, max(1, n_games)))
        wins = draws = 0
        margins: list[float] = []
        if envs == 1 and env_backend == "kaggle":
            opps = [opp] * (num_players - 1)
            for game_idx in range(n_games):
                traj = rollout_episode(
                    model, opps,
                    num_players=num_players, episode_steps=episode_steps,
                    ship_speed=ship_speed, device=device, deterministic=False,
                    learner_seat=game_idx % num_players,
                )
                wins += int(traj.won)
                draws += int(traj.drawn)
                margins.append(traj.final_score)
            out[opp_name] = {
                "win_rate": wins / max(1, n_games),
                "draw_rate": draws / max(1, n_games),
                "mean_margin": float(np.mean(margins)),
                "std_margin": float(np.std(margins)),
                "n_games": n_games,
            }
            continue

        opponents_per_env = [
            [opp_slot] * (num_players - 1)
            for _ in range(envs)
        ]
        vec_kwargs = dict(
            num_envs=envs,
            num_players=num_players,
            episode_steps=episode_steps,
            ship_speed=ship_speed,
            replay_env_idx=None,
        )
        if env_backend == "numpy":
            vec = NumpyVecEnv(**vec_kwargs)
        elif env_backend == "numpy_mp":
            vec = ShardedNumpyVecEnv(**vec_kwargs, num_workers=num_workers)
        else:
            vec = VecEnv(**vec_kwargs)
        with vec:
            batch_idx = 0
            while len(margins) < n_games:
                learner_seats = alternating_learner_seats(
                    envs, num_players, offset=batch_idx
                )
                trajs = rollout_episodes_batched(
                    model,
                    vec,
                    opponents_per_env,
                    num_players=num_players,
                    learner_seat=learner_seats,
                    device=device,
                    deterministic=False,
                    record_trajectories=False,
                )
                for traj in trajs[: n_games - len(margins)]:
                    wins += int(traj.won)
                    draws += int(traj.drawn)
                    margins.append(traj.final_score)
                batch_idx += 1
        out[opp_name] = {
            "win_rate": wins / max(1, n_games),
            "draw_rate": draws / max(1, n_games),
            "mean_margin": float(np.mean(margins)),
            "std_margin": float(np.std(margins)),
            "n_games": n_games,
        }
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--games", type=int, default=50)
    p.add_argument("--num-players", type=int, default=2, choices=(2, 4))
    p.add_argument(
        "--baselines", nargs="+",
        default=["random", "sniper", "heuristic"],
        choices=list(BUILTIN.keys()),
    )
    p.add_argument("--device", default="cpu")
    p.add_argument("--num-envs", type=int, default=16)
    p.add_argument(
        "--env-backend", choices=("kaggle", "numpy", "numpy_mp"), default="kaggle"
    )
    p.add_argument("--num-workers", type=int, default=0)
    args = p.parse_args()

    results = evaluate_ckpt(
        args.ckpt, n_games=args.games, num_players=args.num_players,
        baselines=tuple(args.baselines),
        device=args.device,
        num_envs=args.num_envs,
        env_backend=args.env_backend,
        num_workers=args.num_workers,
    )
    for k, v in results.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
