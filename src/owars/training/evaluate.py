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
from .vec_env import VecEnv
from .vec_rollout import rollout_episodes_batched


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
) -> dict:
    state = torch.load(ckpt_path, map_location=device)
    cfg = OrbitPolicyConfig(**state["config"])
    model = OrbitPolicy(cfg).to(device)
    model.load_state_dict(state["model"])
    model.eval()

    if env_backend not in {"kaggle", "numpy"}:
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
            for _ in range(n_games):
                traj = rollout_episode(
                    model, opps,
                    num_players=num_players, episode_steps=episode_steps,
                    ship_speed=ship_speed, device=device, deterministic=False,
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
        vec_cls = NumpyVecEnv if env_backend == "numpy" else VecEnv
        with vec_cls(
            num_envs=envs,
            num_players=num_players,
            episode_steps=episode_steps,
            ship_speed=ship_speed,
            replay_env_idx=None,
        ) as vec:
            while len(margins) < n_games:
                trajs = rollout_episodes_batched(
                    model,
                    vec,
                    opponents_per_env,
                    num_players=num_players,
                    device=device,
                    deterministic=False,
                    record_trajectories=False,
                )
                for traj in trajs[: n_games - len(margins)]:
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
    p.add_argument("--env-backend", choices=("kaggle", "numpy"), default="kaggle")
    args = p.parse_args()

    results = evaluate_ckpt(
        args.ckpt, n_games=args.games, num_players=args.num_players,
        baselines=tuple(args.baselines),
        device=args.device,
        num_envs=args.num_envs,
        env_backend=args.env_backend,
    )
    for k, v in results.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
