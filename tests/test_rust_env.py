from __future__ import annotations

import json
import math
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest

from owars.training.numpy_env import NumpyOrbitWarsEnv


ROOT = Path(__file__).resolve().parents[1]
RUST_CRATE = ROOT / "rust" / "owars_env"


def _fixture_obs(*, comet: bool = False) -> dict[str, Any]:
    planets: list[list[Any]] = [
        [0, 0, 20.0, 20.0, 1.0, 50, 5],
        [1, 1, 80.0, 80.0, 1.0, 50, 5],
        [2, -1, 20.0, 80.0, 1.0, 20, 3],
        [3, -1, 80.0, 20.0, 1.0, 20, 3],
    ]
    comets = []
    comet_planet_ids = []
    if comet:
        planets = [
            [0, 0, 10.0, 80.0, 1.0, 20, 1],
            [1, 1, 80.0, 10.0, 1.0, 20, 1],
            [10, -1, -99.0, -99.0, 1.0, 5, 1],
        ]
        comet_planet_ids = [10]
        comets = [
            {
                "planet_ids": [10],
                "paths": [[[10.0, 10.0], [14.0, 10.0]]],
                "path_index": -1,
            }
        ]
    return {
        "player": 0,
        "step": 1,
        "planets": planets,
        "fleets": [],
        "angular_velocity": 0.03,
        "initial_planets": [row.copy() for row in planets],
        "next_fleet_id": 0,
        "comets": comets,
        "comet_planet_ids": comet_planet_ids,
    }


def _simple_actions(obs: dict[str, Any], num_players: int) -> list[list[list[float | int]]]:
    planets = [list(p) for p in obs["planets"]]
    actions: list[list[list[float | int]]] = [[] for _ in range(num_players)]
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
            angle = math.atan2(float(target[3]) - float(src[3]), float(target[2]) - float(src[2]))
            actions[player].append([int(src[0]), angle, max(1, int(src[5]) // 3)])
    return actions


def _round_rows(rows: list[list[Any]]) -> list[list[Any]]:
    rounded = []
    for row in rows:
        rounded.append(
            [
                int(row[0]),
                int(row[1]),
                round(float(row[2]), 10),
                round(float(row[3]), 10),
                round(float(row[4]), 10),
                int(row[5]),
                int(row[6]),
            ]
        )
    return rounded


@pytest.mark.parametrize(("workload", "steps"), [("noop", 40), ("simple", 40)])
@pytest.mark.skipif(shutil.which("cargo") is None, reason="cargo is not installed")
def test_rust_fixture_trace_matches_numpy_loaded_state(workload: str, steps: int):
    env = NumpyOrbitWarsEnv(num_players=2, episode_steps=120, ship_speed=6.0)
    env.load_observation(_fixture_obs())
    state = [env._state([[], []])[0], env._state([[], []])[1]]
    for _ in range(steps):
        actions = [[], []] if workload == "noop" else _simple_actions(state[0]["observation"], 2)
        state = env.step(actions)

    proc = subprocess.run(
        [
            "cargo",
            "run",
            "--quiet",
            "--bin",
            "trace_fixture",
            "--",
            "--steps",
            str(steps),
            "--workload",
            workload,
        ],
        cwd=RUST_CRATE,
        check=True,
        capture_output=True,
        text=True,
    )
    rust = json.loads(proc.stdout)

    assert rust["step"] == state[0]["observation"]["step"]
    assert rust["done"] == env.done
    assert rust["next_fleet_id"] == state[0]["observation"]["next_fleet_id"]
    assert _round_rows(rust["planets"]) == _round_rows(state[0]["observation"]["planets"])
    assert _round_rows(rust["fleets"]) == _round_rows(state[0]["observation"]["fleets"])


@pytest.mark.skipif(shutil.which("cargo") is None, reason="cargo is not installed")
def test_rust_active_comet_trace_matches_numpy_loaded_state():
    env = NumpyOrbitWarsEnv(num_players=2, episode_steps=120, ship_speed=6.0)
    env.load_observation(_fixture_obs(comet=True))
    state = [env._state([[], []])[0], env._state([[], []])[1]]
    for _ in range(3):
        state = env.step([[], []])

    proc = subprocess.run(
        [
            "cargo",
            "run",
            "--quiet",
            "--bin",
            "trace_fixture",
            "--",
            "--scenario",
            "comet",
            "--steps",
            "3",
            "--workload",
            "noop",
        ],
        cwd=RUST_CRATE,
        check=True,
        capture_output=True,
        text=True,
    )
    rust = json.loads(proc.stdout)

    assert rust["step"] == state[0]["observation"]["step"]
    assert rust["done"] == env.done
    assert _round_rows(rust["planets"]) == _round_rows(state[0]["observation"]["planets"])


@pytest.mark.parametrize(("seed", "workload"), [(0, "noop"), (0, "simple"), (1, "simple")])
@pytest.mark.skipif(shutil.which("cargo") is None, reason="cargo is not installed")
def test_rust_generated_episode_matches_numpy_through_comet_spawns(seed: int, workload: str):
    env = NumpyOrbitWarsEnv(
        num_players=2,
        episode_steps=500,
        ship_speed=6.0,
        random_seed=seed,
    )
    state = env.reset()
    for _ in range(500):
        actions = [[], []] if workload == "noop" else _simple_actions(state[0]["observation"], 2)
        state = env.step(actions)
        if env.done:
            break

    proc = subprocess.run(
        [
            "cargo",
            "run",
            "--quiet",
            "--bin",
            "trace_fixture",
            "--",
            "--scenario",
            "generated",
            "--seed",
            str(seed),
            "--steps",
            "500",
            "--workload",
            workload,
        ],
        cwd=RUST_CRATE,
        check=True,
        capture_output=True,
        text=True,
    )
    rust = json.loads(proc.stdout)
    obs = state[0]["observation"]

    assert rust["step"] == obs["step"]
    assert rust["done"] == env.done
    assert rust["next_fleet_id"] == obs["next_fleet_id"]
    assert _round_rows(rust["planets"]) == _round_rows(obs["planets"])
    assert _round_rows(rust["fleets"]) == _round_rows(obs["fleets"])


@pytest.mark.skipif(shutil.which("cargo") is None, reason="cargo is not installed")
def test_rust_generated_comet_spawn_step_matches_numpy():
    env = NumpyOrbitWarsEnv(
        num_players=2,
        episode_steps=500,
        ship_speed=6.0,
        random_seed=0,
    )
    state = env.reset()
    for _ in range(50):
        state = env.step([[], []])

    proc = subprocess.run(
        [
            "cargo",
            "run",
            "--quiet",
            "--bin",
            "trace_fixture",
            "--",
            "--scenario",
            "generated",
            "--seed",
            "0",
            "--steps",
            "50",
            "--workload",
            "noop",
        ],
        cwd=RUST_CRATE,
        check=True,
        capture_output=True,
        text=True,
    )
    rust = json.loads(proc.stdout)
    obs = state[0]["observation"]

    assert obs["step"] == 50
    assert rust["step"] == 50
    assert _round_rows(rust["planets"]) == _round_rows(obs["planets"])
