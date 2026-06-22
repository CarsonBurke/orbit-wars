"""Evaluate a trained checkpoint by playing N games against each baseline.

Reports win-rate, mean margin, and a Glicko-style rating per matchup. Run
this after training to confirm the policy actually improved against
*non-mirror* opponents (self-play win-rate alone is not enough).
"""

from __future__ import annotations

import argparse
import warnings

import numpy as np
import torch

from ..policies.config import OrbitPolicyConfig
from ..policies.model import OrbitPolicy, restore_fp32_params
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
    baselines: tuple[str, ...] = ("random", "sniper_v18", "heuristic"),
    num_envs: int = 16,
    env_backend: str = "rust",
    num_workers: int = 0,
    compile_mode: str | None = "reduce-overhead",
    deterministic: bool = True,
) -> dict:
    state = torch.load(ckpt_path, map_location=device)
    cfg = OrbitPolicyConfig(**state["config"])
    model = OrbitPolicy(cfg).to(device)
    if torch.device(device).type == "cuda":
        model.bfloat16()
        restore_fp32_params(model)
    model.load_state_dict(state["model"])
    model.eval()
    compile_mode = compile_mode if torch.device(device).type == "cuda" else None

    if env_backend not in {"kaggle", "numpy", "numpy_mp", "rust"}:
        raise ValueError(f"unknown env_backend: {env_backend!r}")
    out: dict[str, dict[str, float]] = {}
    for opp_name in baselines:
        opp = BUILTIN[opp_name]
        opp_slot = OpponentSlot(name=opp_name, agent=opp)
        envs = max(1, min(num_envs, max(1, n_games)))
        wins = draws = 0
        margins: list[float] = []
        if envs == 1 and env_backend == "kaggle":
            if compile_mode is not None:
                warnings.warn(
                    "compile_mode is only used by vectorized evaluation; "
                    "use --num-envs > 1 or a fast env backend to benchmark compiled CUDA inference.",
                    RuntimeWarning,
                    stacklevel=2,
                )
            opps = [opp] * (num_players - 1)
            for game_idx in range(n_games):
                traj = rollout_episode(
                    model, opps,
                    num_players=num_players, episode_steps=episode_steps,
                    ship_speed=ship_speed, device=device, deterministic=deterministic,
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
        elif env_backend == "rust":
            from .rust_env import RustVecEnv

            vec = RustVecEnv(**vec_kwargs)
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
                    deterministic=deterministic,
                    record_trajectories=False,
                    compile_mode=compile_mode,
                    policy_graph_rows=envs,
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
        default=["random", "sniper_v18", "heuristic"],
        choices=list(BUILTIN.keys()),
    )
    p.add_argument("--device", default="cpu")
    p.add_argument("--num-envs", type=int, default=16)
    p.add_argument(
        "--env-backend", choices=("kaggle", "numpy", "numpy_mp", "rust"), default="rust"
    )
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument(
        "--stochastic",
        action="store_true",
        help="Evaluate with rollout-style stochastic sampling instead of deterministic deployment actions.",
    )
    p.add_argument(
        "--compile-mode",
        default="reduce-overhead",
        help="CUDA torch.compile mode for batched policy forwards; use 'none' to disable.",
    )
    args = p.parse_args()
    compile_mode = None if args.compile_mode.lower() == "none" else args.compile_mode

    if args.device == "cuda":
        # TF32 for the residual fp32 GEMMs around the bf16 model (mirrors the
        # training entrypoint); no effect on CPU eval or the bf16 path itself.
        torch.set_float32_matmul_precision("high")

    results = evaluate_ckpt(
        args.ckpt, n_games=args.games, num_players=args.num_players,
        baselines=tuple(args.baselines),
        device=args.device,
        num_envs=args.num_envs,
        env_backend=args.env_backend,
        num_workers=args.num_workers,
        compile_mode=compile_mode,
        deterministic=not args.stochastic,
    )
    for k, v in results.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
