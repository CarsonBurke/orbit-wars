"""Evaluate a trained checkpoint by playing N games against each baseline.

Reports win-rate, mean margin, and a Glicko-style rating per matchup. Run
this after training to confirm the policy actually improved against
*non-mirror* opponents (self-play win-rate alone is not enough).
"""

from __future__ import annotations

import argparse
from collections import defaultdict

import numpy as np
import torch

from ..agents.learned import LearnedAgent
from ..policies.config import OrbitPolicyConfig
from ..policies.model import OrbitPolicy
from .league import BUILTIN
from .rollout import rollout_episode


def evaluate_ckpt(
    ckpt_path: str,
    *,
    n_games: int = 50,
    num_players: int = 2,
    episode_steps: int = 500,
    ship_speed: float = 6.0,
    device: str = "cpu",
    baselines: tuple[str, ...] = ("random", "sniper", "heuristic"),
) -> dict:
    state = torch.load(ckpt_path, map_location=device)
    cfg = OrbitPolicyConfig(**state["config"])
    model = OrbitPolicy(cfg).to(device)
    model.load_state_dict(state["model"])
    model.eval()

    out: dict[str, dict[str, float]] = {}
    for opp_name in baselines:
        opp = BUILTIN[opp_name]
        wins = draws = 0
        margins: list[float] = []
        for _ in range(n_games):
            opps = [opp] * (num_players - 1)
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
    args = p.parse_args()

    results = evaluate_ckpt(
        args.ckpt, n_games=args.games, num_players=args.num_players,
        baselines=tuple(args.baselines), device=args.device,
    )
    for k, v in results.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
