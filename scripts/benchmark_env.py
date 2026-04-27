#!/usr/bin/env python
"""Benchmark Orbit Wars environment backends.

Examples:
    PYTHONPATH=src python scripts/benchmark_env.py --backend all --num-envs 16
"""

from __future__ import annotations

import argparse
import math
from time import perf_counter
from typing import Any

from owars.training.numpy_env import NumpyOrbitWarsEnv, NumpyVecEnv
from owars.training.vec_env import VecEnv


def _empty_actions(num_players: int) -> list[list[Any]]:
    return [[] for _ in range(num_players)]


def _obs(state: Any) -> Any:
    first = state[0]
    if isinstance(first, dict):
        return first["observation"]
    return first.observation


def _simple_actions(state: Any, num_players: int) -> list[list[Any]]:
    obs = _obs(state)
    planets = [list(p) for p in obs["planets"]]
    actions: list[list[Any]] = [[] for _ in range(num_players)]
    for player in range(num_players):
        mine = [p for p in planets if int(p[1]) == player and int(p[5]) >= 8]
        targets = [p for p in planets if int(p[1]) != player]
        for src in mine[:2]:
            if not targets:
                continue
            target = min(
                targets,
                key=lambda p: (float(p[2]) - float(src[2])) ** 2
                + (float(p[3]) - float(src[3])) ** 2,
            )
            angle = math.atan2(
                float(target[3]) - float(src[3]),
                float(target[2]) - float(src[2]),
            )
            actions[player].append([int(src[0]), angle, max(1, int(src[5]) // 3)])
    return actions


def _actions(state: Any, num_players: int, workload: str) -> list[list[Any]]:
    if workload == "noop":
        return _empty_actions(num_players)
    if workload == "simple":
        return _simple_actions(state, num_players)
    raise ValueError(f"unknown workload: {workload!r}")


def bench_kaggle_single(
    num_envs: int,
    num_players: int,
    episode_steps: int,
    ship_speed: float,
    workload: str,
) -> dict[str, float]:
    from kaggle_environments import make

    reset_s = 0.0
    step_s = 0.0
    steps = 0
    start = perf_counter()
    for _ in range(num_envs):
        env = make(
            "orbit_wars",
            configuration={"episodeSteps": episode_steps, "shipSpeed": ship_speed},
            debug=False,
        )
        t0 = perf_counter()
        state = env.reset(num_agents=num_players)
        reset_s += perf_counter() - t0
        while not env.done:
            t0 = perf_counter()
            state = env.step(_actions(state, num_players, workload))
            step_s += perf_counter() - t0
            steps += 1
    wall_s = perf_counter() - start
    return _summary("kaggle_single", steps, reset_s, step_s, wall_s)


def bench_numpy_single(
    num_envs: int,
    num_players: int,
    episode_steps: int,
    ship_speed: float,
    workload: str,
) -> dict[str, float]:
    reset_s = 0.0
    step_s = 0.0
    steps = 0
    start = perf_counter()
    for i in range(num_envs):
        env = NumpyOrbitWarsEnv(
            num_players=num_players,
            episode_steps=episode_steps,
            ship_speed=ship_speed,
            random_seed=i,
        )
        t0 = perf_counter()
        state = env.reset()
        reset_s += perf_counter() - t0
        while not env.done:
            t0 = perf_counter()
            state = env.step(_actions(state, num_players, workload))
            step_s += perf_counter() - t0
            steps += 1
    wall_s = perf_counter() - start
    return _summary("numpy_single", steps, reset_s, step_s, wall_s)


def bench_numpy_vec(
    num_envs: int,
    num_players: int,
    episode_steps: int,
    ship_speed: float,
    workload: str,
) -> dict[str, float]:
    reset_s = 0.0
    step_s = 0.0
    steps = 0
    start = perf_counter()
    with NumpyVecEnv(
        num_envs=num_envs,
        num_players=num_players,
        episode_steps=episode_steps,
        ship_speed=ship_speed,
        random_seed=0,
    ) as vec:
        t0 = perf_counter()
        states = vec.reset()
        reset_s += perf_counter() - t0
        done = [False] * num_envs
        while not all(done):
            active = [i for i, is_done in enumerate(done) if not is_done]
            actions = [_actions(states[i], num_players, workload) for i in active]
            t0 = perf_counter()
            results = vec.step_subset(active, actions)
            step_s += perf_counter() - t0
            steps += len(active)
            for idx, (state, is_done, _final) in results.items():
                states[idx] = state
                done[idx] = is_done
    wall_s = perf_counter() - start
    return _summary("numpy_vec", steps, reset_s, step_s, wall_s)


def bench_kaggle_vec(
    num_envs: int,
    num_players: int,
    episode_steps: int,
    ship_speed: float,
    workload: str,
) -> dict[str, float]:
    reset_s = 0.0
    step_s = 0.0
    steps = 0
    start = perf_counter()
    with VecEnv(
        num_envs=num_envs,
        num_players=num_players,
        episode_steps=episode_steps,
        ship_speed=ship_speed,
        replay_env_idx=None,
    ) as vec:
        t0 = perf_counter()
        states = vec.reset()
        reset_s += perf_counter() - t0
        done = [False] * num_envs
        while not all(done):
            active = [i for i, is_done in enumerate(done) if not is_done]
            actions = [_actions(states[i], num_players, workload) for i in active]
            t0 = perf_counter()
            results = vec.step_subset(active, actions)
            step_s += perf_counter() - t0
            steps += len(active)
            for idx, (state, is_done, _final) in results.items():
                states[idx] = state
                done[idx] = is_done
    wall_s = perf_counter() - start
    return _summary("kaggle_vec", steps, reset_s, step_s, wall_s)


def _summary(name: str, steps: int, reset_s: float, step_s: float, wall_s: float) -> dict[str, float]:
    return {
        "backend": name,
        "steps": float(steps),
        "reset_s": reset_s,
        "step_s": step_s,
        "wall_s": wall_s,
        "step_sps": steps / step_s if step_s > 0 else 0.0,
        "wall_sps": steps / wall_s if wall_s > 0 else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--backend",
        choices=("all", "kaggle_single", "kaggle_vec", "numpy_single", "numpy_vec"),
        default="all",
    )
    parser.add_argument("--num-envs", type=int, default=16)
    parser.add_argument("--num-players", type=int, default=2, choices=(2, 4))
    parser.add_argument("--episode-steps", type=int, default=500)
    parser.add_argument("--ship-speed", type=float, default=6.0)
    parser.add_argument("--workload", choices=("noop", "simple"), default="noop")
    args = parser.parse_args()

    benches = {
        "kaggle_single": bench_kaggle_single,
        "kaggle_vec": bench_kaggle_vec,
        "numpy_single": bench_numpy_single,
        "numpy_vec": bench_numpy_vec,
    }
    names = list(benches) if args.backend == "all" else [args.backend]
    for name in names:
        result = benches[name](
            args.num_envs,
            args.num_players,
            args.episode_steps,
            args.ship_speed,
            args.workload,
        )
        print(
            f"{result['backend']:>14}: "
            f"steps={int(result['steps'])} "
            f"reset={result['reset_s']:.3f}s "
            f"step={result['step_s']:.3f}s "
            f"wall={result['wall_s']:.3f}s "
            f"step_sps={result['step_sps']:.1f} "
            f"wall_sps={result['wall_sps']:.1f}"
        )


if __name__ == "__main__":
    main()
