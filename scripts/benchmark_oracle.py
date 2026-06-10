#!/usr/bin/env python
"""Benchmark the fleet-destination oracle through the Python binding.

Measures, per spec acceptance scenario (static 40p/384f, orbiting 40p/384f,
mixed with active comets):

- cold call: per-env cache invalidated before every call (worst case)
- warm call: repeated query of an unchanged env (cache hit)
- duplicate rows: both players of one env in a single cold call, which must
  cost <= --dup-ratio-limit x the single-row cold call
- the pure-Python exact oracle, for fallback-path visibility

Exits non-zero when a target is missed so it can gate training runs.

Examples:
    PYTHONPATH=src python scripts/benchmark_oracle.py
    PYTHONPATH=src python scripts/benchmark_oracle.py --iters 500 --no-check
"""

from __future__ import annotations

import argparse
import math
import sys
from time import perf_counter
from typing import Any

import numpy as np

from owars.game.destination_oracle import infer_fleet_destinations
from owars.game.observation import parse_observation

EPISODE_STEPS = 2000  # long synthetic episode: no horizon or comet-spawn cutoff
STEP = 460  # past the last comet spawn, so lookahead is fully known
NUM_FLEETS = 384


def _fleets(rng: np.random.Generator, count: int) -> list[list[Any]]:
    return [
        [
            i,
            i % 2,
            float(rng.uniform(1.0, 99.0)),
            float(rng.uniform(1.0, 99.0)),
            float(rng.uniform(-math.pi, math.pi)),
            -1,
            1 + int(rng.uniform(0.0, 1.0) ** 2 * 800.0),
        ]
        for i in range(count)
    ]


def _static_planets(count: int) -> list[list[Any]]:
    # Corner clusters spreading toward the board edge: orbital_radius +
    # radius >= 50 for every member, so none ever rotate.
    corners = [(12.0, 12.0), (88.0, 12.0), (12.0, 88.0), (88.0, 88.0)]
    planets = []
    for i in range(count):
        cx, cy = corners[i % 4]
        k = i // 4
        dx = -3.5 if cx < 50.0 else 3.5
        dy = -3.5 if cy < 50.0 else 3.5
        planets.append(
            [
                i,
                0 if i == 0 else 1 if i == 1 else -1,
                cx + dx * (k % 3),
                cy + dy * (k // 3),
                1.0 + math.log(1.0 + i % 5) * 0.55,
                30,
                1 + i % 5,
            ]
        )
    return planets


def _orbiting_planets(count: int) -> list[list[Any]]:
    planets = []
    for i in range(count):
        radius = 1.0 + math.log(1.0 + i % 5) * 0.55
        orb_r = 13.0 + i * (46.0 - 13.0 - math.ceil(radius)) / count
        angle = i * 2.399_963  # golden angle spread
        planets.append(
            [
                i,
                0 if i == 0 else 1 if i == 1 else -1,
                50.0 + orb_r * math.cos(angle),
                50.0 + orb_r * math.sin(angle),
                radius,
                30,
                1 + i % 5,
            ]
        )
    return planets


def _comet_group(next_id: int, path_index: int) -> tuple[list[list[Any]], dict[str, Any]]:
    base = [[2.0 + i * 3.4, 8.0 + i * 2.9] for i in range(30)]
    paths = [
        base,
        [[100.0 - x, y] for x, y in base],
        [[x, 100.0 - y] for x, y in base],
        [[100.0 - x, 100.0 - y] for x, y in base],
    ]
    planet_ids = [next_id + i for i in range(4)]
    planets = []
    for pid, path in zip(planet_ids, paths):
        x, y = path[path_index] if path_index >= 0 else (-99.0, -99.0)
        planets.append([pid, -1, x, y, 1.0, 5, 1])
    return planets, {"planet_ids": planet_ids, "paths": paths, "path_index": path_index}


def _obs_dict(
    planets: list[list[Any]],
    comets: list[dict[str, Any]],
    fleet_seed: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(fleet_seed)
    return {
        "player": 0,
        "step": STEP,
        "planets": [row.copy() for row in planets],
        "fleets": _fleets(rng, NUM_FLEETS),
        "angular_velocity": 0.04,
        "initial_planets": [row.copy() for row in planets],
        "next_fleet_id": NUM_FLEETS,
        "comets": comets,
        "comet_planet_ids": [pid for group in comets for pid in group["planet_ids"]],
    }


def _scenarios() -> list[tuple[str, dict[str, Any], float]]:
    mixed_planets = _static_planets(20)
    for row in _orbiting_planets(16):
        row[0] += 100
        mixed_planets.append(row)
    comet_planets, group = _comet_group(500, 6)
    mixed_planets.extend(comet_planets)
    old_to_new = {row[0]: i for i, row in enumerate(mixed_planets)}
    for i, row in enumerate(mixed_planets):
        row[0] = i
    group["planet_ids"] = [old_to_new[pid] for pid in group["planet_ids"]]
    return [
        ("static 40p / 384f", _obs_dict(_static_planets(40), [], 1), 250.0),
        ("orbiting 40p / 384f", _obs_dict(_orbiting_planets(40), [], 2), 500.0),
        ("mixed + active comets", _obs_dict(mixed_planets, [group], 3), 500.0),
    ]


def _time_cold(core: Any, obs: dict[str, Any], rows: list[tuple[int, int]], iters: int) -> float:
    # Median of per-call samples: cold calls sit in the tens of microseconds,
    # where scheduler jitter would make a mean-based gate flaky.
    samples = []
    for _ in range(iters):
        core.load_observation(0, obs)  # invalidates the env's oracle cache
        t0 = perf_counter()
        core.fleet_destination_oracle(rows)
        samples.append(perf_counter() - t0)
    return float(np.median(samples)) * 1e6


def _time_warm(core: Any, rows: list[tuple[int, int]], iters: int) -> float:
    core.fleet_destination_oracle(rows)
    samples = []
    for _ in range(iters):
        t0 = perf_counter()
        core.fleet_destination_oracle(rows)
        samples.append(perf_counter() - t0)
    return float(np.median(samples)) * 1e6


def _time_python(obs: dict[str, Any], iters: int) -> float:
    parsed = parse_observation(obs)
    t0 = perf_counter()
    for _ in range(iters):
        infer_fleet_destinations(parsed, episode_steps=EPISODE_STEPS)
    return (perf_counter() - t0) * 1e6 / iters


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--python-iters", type=int, default=5)
    parser.add_argument("--dup-ratio-limit", type=float, default=1.1)
    parser.add_argument("--no-check", action="store_true", help="report only, never fail")
    args = parser.parse_args()

    from owars.training.rust_env import RustVecEnv

    rust = RustVecEnv(
        num_envs=1,
        num_players=2,
        episode_steps=EPISODE_STEPS,
        ship_speed=6.0,
        random_seed=0,
    )
    core = rust._core

    ok = True
    for name, obs, limit_us in _scenarios():
        core.load_observation(0, obs)
        for _ in range(20):  # warmup: JIT-free, but settles allocator/threads
            core.load_observation(0, obs)
            core.fleet_destination_oracle([(0, 0)])

        cold_us = _time_cold(core, obs, [(0, 0)], args.iters)
        dup_us = _time_cold(core, obs, [(0, 0), (0, 1)], args.iters)
        warm_us = _time_warm(core, [(0, 0)], args.iters)
        python_us = _time_python(obs, args.python_iters)
        dup_ratio = dup_us / cold_us if cold_us > 0 else float("inf")

        scenario_ok = cold_us <= limit_us and dup_ratio <= args.dup_ratio_limit
        ok &= scenario_ok
        print(
            f"{name:<24} cold {cold_us:8.1f} us (limit {limit_us:6.1f})   "
            f"dup2 {dup_us:8.1f} us (x{dup_ratio:4.2f}, limit x{args.dup_ratio_limit:.2f})   "
            f"warm {warm_us:7.1f} us   python {python_us / 1e3:8.2f} ms   "
            f"{'OK' if scenario_ok else 'FAIL'}"
        )

    if args.no_check:
        return 0
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
