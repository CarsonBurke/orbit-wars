from __future__ import annotations

import json
import math
import shutil
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from owars.policies.features import encode_raw_observations
from owars.training.numpy_env import NumpyOrbitWarsEnv

ROOT = Path(__file__).resolve().parents[1]
RUST_CRATE = ROOT / "rust" / "owars_env"
RUST_PY_CRATE = ROOT / "rust" / "owars_env_py"


def _build_rust_extension() -> None:
    subprocess.run(
        ["cargo", "build", "--release"],
        cwd=RUST_PY_CRATE,
        check=True,
        capture_output=True,
        text=True,
    )


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


def _oracle_obs(
    *,
    planets: list[list[Any]],
    fleets: list[list[Any]],
    step: int = 1,
    angular_velocity: float = 0.0,
) -> dict[str, Any]:
    return {
        "player": 0,
        "step": step,
        "planets": [row.copy() for row in planets],
        "fleets": [row.copy() for row in fleets],
        "angular_velocity": angular_velocity,
        "initial_planets": [row.copy() for row in planets],
        "next_fleet_id": len(fleets),
        "comets": [],
        "comet_planet_ids": [],
    }


def _fleet_destination_oracle(
    obs: dict[str, Any],
    *,
    ship_speed: float = 6.0,
) -> dict[str, np.ndarray]:
    from owars.training.rust_env import RustVecEnv

    rust = RustVecEnv(
        num_envs=1,
        num_players=2,
        episode_steps=500,
        ship_speed=ship_speed,
        random_seed=0,
    )
    rust._core.load_observation(0, obs)
    return rust._core.fleet_destination_oracle([(0, 0)])


@pytest.mark.skipif(shutil.which("cargo") is None, reason="cargo is not installed")
def test_rust_fleet_destination_oracle_static_direct_hit():
    _build_rust_extension()
    obs = _oracle_obs(
        planets=[
            [0, 0, 5.0, 10.0, 1.0, 10, 1],
            [1, 1, 20.0, 10.0, 1.0, 10, 1],
        ],
        fleets=[[0, 0, 15.0, 10.0, 0.0, 0, 1]],
    )

    oracle = _fleet_destination_oracle(obs)

    assert int(oracle["status"][0, 0]) == 1
    assert int(oracle["dest_idx"][0, 0]) == 1
    assert float(oracle["eta"][0, 0]) == 5.0


@pytest.mark.skipif(shutil.which("cargo") is None, reason="cargo is not installed")
def test_rust_fleet_destination_oracle_board_before_sun():
    _build_rust_extension()
    obs = _oracle_obs(
        planets=[
            [0, 0, 95.0, 80.0, 1.0, 10, 1],
            [1, 1, 70.0, 50.0, 3.0, 10, 1],
        ],
        fleets=[[0, 0, 95.0, 50.0, math.pi, 0, 1000]],
    )

    oracle = _fleet_destination_oracle(obs, ship_speed=100.0)

    assert int(oracle["status"][0, 0]) == 2
    assert int(oracle["dest_idx"][0, 0]) == -1
    assert float(oracle["eta"][0, 0]) == 1.0


@pytest.mark.skipif(shutil.which("cargo") is None, reason="cargo is not installed")
def test_rust_fleet_destination_oracle_sun_before_planet():
    _build_rust_extension()
    obs = _oracle_obs(
        planets=[
            [0, 0, 20.0, 80.0, 1.0, 10, 1],
            [1, 1, 70.0, 50.0, 3.0, 10, 1],
        ],
        fleets=[[0, 0, 20.0, 50.0, 0.0, 0, 100]],
    )

    oracle = _fleet_destination_oracle(obs, ship_speed=100.0)

    assert int(oracle["status"][0, 0]) == 3
    assert int(oracle["dest_idx"][0, 0]) == -1
    assert float(oracle["eta"][0, 0]) == 1.0


@pytest.mark.skipif(shutil.which("cargo") is None, reason="cargo is not installed")
def test_rust_fleet_destination_oracle_uses_planet_vector_order_for_ties():
    _build_rust_extension()
    obs = _oracle_obs(
        planets=[
            [5, 1, 30.0, 10.0, 2.0, 10, 1],
            [3, 1, 30.0, 10.0, 2.0, 10, 1],
            [0, 0, 10.0, 10.0, 1.0, 10, 1],
        ],
        fleets=[[0, 0, 25.0, 10.0, 0.0, 0, 1]],
    )

    oracle = _fleet_destination_oracle(obs)

    assert int(oracle["status"][0, 0]) == 1
    assert int(oracle["dest_idx"][0, 0]) == 0
    assert float(oracle["eta"][0, 0]) == 4.0


@pytest.mark.skipif(shutil.which("cargo") is None, reason="cargo is not installed")
def test_rust_fleet_destination_oracle_strict_tangent_is_not_hit():
    _build_rust_extension()
    obs = _oracle_obs(
        planets=[
            [0, 0, 90.0, 10.0, 1.0, 10, 1],
            [1, 1, 96.0, 11.0, 1.0, 10, 1],
        ],
        fleets=[[0, 0, 95.0, 10.0, 0.0, 0, 1]],
        step=450,
    )

    oracle = _fleet_destination_oracle(obs)

    assert int(oracle["status"][0, 0]) == 2
    assert int(oracle["dest_idx"][0, 0]) == -1
    assert float(oracle["eta"][0, 0]) == 6.0


@pytest.mark.skipif(shutil.which("cargo") is None, reason="cargo is not installed")
def test_rust_fleet_destination_oracle_moving_sweep_hit():
    _build_rust_extension()
    obs = _oracle_obs(
        planets=[
            [1, 1, 70.0, 50.0, 1.0, 10, 1],
            [0, 0, 20.0, 80.0, 1.0, 10, 1],
        ],
        fleets=[[0, 0, 70.0, 52.0, -math.pi / 2.0, 0, 1]],
        angular_velocity=0.1,
    )

    oracle = _fleet_destination_oracle(obs)

    assert int(oracle["status"][0, 0]) == 1
    assert int(oracle["dest_idx"][0, 0]) == 0
    assert float(oracle["eta"][0, 0]) == 1.0


@pytest.mark.skipif(shutil.which("cargo") is None, reason="cargo is not installed")
def test_rust_fleet_destination_oracle_ignores_bogus_fleet_metadata():
    _build_rust_extension()
    obs = _oracle_obs(
        planets=[
            [0, 0, 5.0, 10.0, 1.0, 10, 1],
            [1, 1, 20.0, 10.0, 1.0, 10, 1],
            [2, 1, 80.0, 80.0, 5.0, 10, 1],
        ],
        fleets=[[0, 0, 15.0, 10.0, 0.0, 0, 1, 2, 999.0, 80.0, 80.0]],
    )

    oracle = _fleet_destination_oracle(obs)

    assert int(oracle["status"][0, 0]) == 1
    assert int(oracle["dest_idx"][0, 0]) == 1
    assert float(oracle["eta"][0, 0]) == 5.0


@pytest.mark.skipif(shutil.which("cargo") is None, reason="cargo is not installed")
def test_rust_fleet_destination_oracle_marks_future_comet_spawn_unknown():
    _build_rust_extension()
    obs = _oracle_obs(
        planets=[
            [0, 0, 5.0, 20.0, 1.0, 10, 1],
            [1, 1, 80.0, 80.0, 1.0, 10, 1],
        ],
        fleets=[[0, 0, 20.0, 20.0, 0.0, 0, 1]],
        step=48,
    )

    oracle = _fleet_destination_oracle(obs)

    assert int(oracle["status"][0, 0]) == 5
    assert int(oracle["dest_idx"][0, 0]) == -1
    assert float(oracle["eta"][0, 0]) == 3.0


@pytest.mark.skipif(shutil.which("cargo") is None, reason="cargo is not installed")
def test_rust_fleet_destination_oracle_returns_all_fleets_with_owner_features():
    _build_rust_extension()
    from owars.training.rust_env import RustVecEnv

    obs = _oracle_obs(
        planets=[
            [0, 0, 20.0, 10.0, 1.0, 10, 1],
            [1, 1, 80.0, 10.0, 1.0, 10, 1],
        ],
        fleets=[
            [0, 0, 15.0, 10.0, 0.0, 0, 1],
            [1, 1, 85.0, 10.0, math.pi, 1, 1],
        ],
    )
    rust = RustVecEnv(
        num_envs=1,
        num_players=2,
        episode_steps=500,
        ship_speed=6.0,
        random_seed=0,
    )
    rust._core.load_observation(0, obs)

    rows = [(0, 0), (0, 1)]
    oracle = rust._core.fleet_destination_oracle(rows)
    encoded, _ = rust.policy_batch_no_context(rows, device="cpu")

    assert oracle["status"][:, :2].tolist() == [[1, 1], [1, 1]]
    assert oracle["dest_idx"][:, :2].tolist() == [[0, 1], [0, 1]]
    assert encoded.fleet_mask[:, :2].tolist() == [[True, True], [True, True]]
    assert float(encoded.fleet_feats[0, 0, 14]) == 1.0  # player 0 self
    assert float(encoded.fleet_feats[0, 1, 16]) == 1.0  # player 0 enemy
    assert float(encoded.fleet_feats[1, 0, 16]) == 1.0  # player 1 enemy
    assert float(encoded.fleet_feats[1, 1, 14]) == 1.0  # player 1 self


@pytest.mark.skipif(shutil.which("cargo") is None, reason="cargo is not installed")
def test_rust_load_observation_round_trips_rust_observation_rows():
    _build_rust_extension()
    from owars.training.rust_env import RustVecEnv

    rust = RustVecEnv(
        num_envs=1,
        num_players=2,
        episode_steps=500,
        ship_speed=6.0,
        random_seed=0,
    )
    rust.reset()
    rust.step_subset_fast([0], [[[], []]])
    obs = rust.observation(0, 0)

    rust._core.load_observation(0, obs)
    encoded, _ = rust.policy_batch_no_context([(0, 0)], device="cpu")

    assert bool(encoded.planet_mask[0].any())


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


@pytest.mark.skipif(shutil.which("cargo") is None, reason="cargo is not installed")
def test_rust_vec_env_policy_batch_matches_numpy_vec_env():
    _build_rust_extension()
    from owars.training.numpy_env import NumpyVecEnv
    from owars.training.rust_env import RustVecEnv

    rust = RustVecEnv(
        num_envs=1,
        num_players=2,
        episode_steps=500,
        ship_speed=6.0,
        random_seed=0,
    )
    numpy = NumpyVecEnv(
        num_envs=1,
        num_players=2,
        episode_steps=500,
        ship_speed=6.0,
        random_seed=0,
    )
    numpy_states = numpy.reset()
    rust.reset()
    numpy_states[0] = numpy.step_subset([0], [[[], []]])[0][0]
    rust.step_subset_fast([0], [[[], []]])

    assert _round_rows(rust.observation(0, 0)["planets"]) == _round_rows(
        numpy_states[0][0]["observation"]["planets"]
    )

    fast, contexts = rust.policy_batch([(0, 0), (0, 1)], device="cpu")
    expected = encode_raw_observations(
        [numpy_states[0][0]["observation"], numpy_states[0][1]["observation"]],
        device="cpu",
    )
    assert torch.allclose(fast.planet_feats, expected.planet_feats)
    assert torch.equal(fast.planet_mask, expected.planet_mask)
    assert torch.equal(fast.planet_owned_mask, expected.planet_owned_mask)
    assert torch.equal(fast.planet_ids, expected.planet_ids)
    assert torch.allclose(fast.fleet_feats, expected.fleet_feats)
    assert torch.equal(fast.fleet_mask, expected.fleet_mask)
    assert len(contexts) == 2

    obs = numpy_states[0][0]["observation"]
    src = next(p for p in obs["planets"] if int(p[1]) == 0 and int(p[5]) >= 8)
    target = next(p for p in obs["planets"] if int(p[1]) != 0)
    angle = math.atan2(float(target[3]) - float(src[3]), float(target[2]) - float(src[2]))
    action = [[[int(src[0]), angle, 5, int(target[0]), 3.0, float(target[2]), float(target[3])]], []]
    numpy_states[0] = numpy.step_subset([0], [action])[0][0]
    rust.step_subset_fast([0], [action])

    fast, _ = rust.policy_batch([(0, 0), (0, 1)], device="cpu")
    expected = encode_raw_observations(
        [numpy_states[0][0]["observation"], numpy_states[0][1]["observation"]],
        device="cpu",
    )
    assert torch.allclose(fast.planet_feats, expected.planet_feats)
    assert torch.equal(fast.planet_mask, expected.planet_mask)
    assert torch.equal(fast.planet_owned_mask, expected.planet_owned_mask)
    assert torch.equal(fast.planet_ids, expected.planet_ids)
    assert torch.allclose(fast.fleet_feats, expected.fleet_feats)
    assert torch.equal(fast.fleet_mask, expected.fleet_mask)


@pytest.mark.skipif(shutil.which("cargo") is None, reason="cargo is not installed")
def test_rust_vec_env_step_subset_fast_rejects_duplicate_indices():
    _build_rust_extension()
    from owars.training.rust_env import RustVecEnv

    rust = RustVecEnv(
        num_envs=2,
        num_players=2,
        episode_steps=80,
        ship_speed=6.0,
        random_seed=0,
    )
    rust.reset()
    with pytest.raises(ValueError, match="indices must be unique"):
        rust.step_subset_fast([0, 0], [[[], []], [[], []]])


@pytest.mark.skipif(shutil.which("cargo") is None, reason="cargo is not installed")
@pytest.mark.parametrize(
    ("name", "agent_name"),
    [
        ("sniper", "sniper_agent"),
        ("sniper_v2", "sniper_v2_agent"),
        ("sniper_v3", "sniper_v3_agent"),
        ("sniper_v4", "sniper_v4_agent"),
        ("sniper_v5", "sniper_v5_agent"),
        ("sniper_v6", "sniper_v6_agent"),
        ("sniper_v7", "sniper_v7_agent"),
        ("sniper_v8", "sniper_v8_agent"),
        ("sniper_v9", "sniper_v9_agent"),
        ("sniper_v10", "sniper_v10_agent"),
        ("sniper_v11", "sniper_v11_agent"),
        ("sniper_v12", "sniper_v12_agent"),
        ("sniper_v13", "sniper_v13_agent"),
        ("sniper_v14", "sniper_v14_agent"),
        ("sniper_v15", "sniper_v15_agent"),
        ("sniper_v16", "sniper_v16_agent"),
        ("sniper_v17", "sniper_v17_agent"),
    ],
)
def test_rust_vec_env_native_sniper_matches_python_sniper(name: str, agent_name: str):
    _build_rust_extension()
    import owars.agents.sniper as sniper_agents
    from owars.training.rust_env import RustVecEnv

    agent = getattr(sniper_agents, agent_name)
    rust = RustVecEnv(
        num_envs=2,
        num_players=2,
        episode_steps=80,
        ship_speed=6.0,
        random_seed=0,
    )
    rust.reset()
    rust.step_subset_fast([0, 1], [[[], []], [[], []]])
    rows = [(0, 0), (0, 1), (1, 0), (1, 1)]
    native = rust.builtin_actions(name, rows, native_actions=False)

    for row_actions, (env_idx, player) in zip(native, rows, strict=True):
        expected = agent(rust.observation(env_idx, player))
        assert len(row_actions) == len(expected)
        for got, want in zip(row_actions, expected, strict=True):
            assert int(got[0]) == int(want[0])
            assert math.isclose(float(got[1]), float(want[1]), rel_tol=0.0, abs_tol=1e-12)
            assert int(got[2]) == int(want[2])


@pytest.mark.skipif(shutil.which("cargo") is None, reason="cargo is not installed")
def test_rust_vec_env_sniper_profile_actions_match_builtin_v14():
    _build_rust_extension()
    from owars.training.rust_env import RustVecEnv

    profile = {
        "reserve_base": 1,
        "reserve_production": 0.35,
        "send_buffer": 1,
        "enemy_growth": True,
        "enemy_value": 3.55,
        "neutral_value": 1.0,
        "production_weight": 6.5,
        "ship_cost_weight": 0.66,
        "time_cost_weight": 0.34,
        "duplicate_penalty": 0.20,
        "net_defense_reserve": True,
        "defense_horizon": 42.0,
        "reinforce_owned": True,
        "defense_arrival_slack": 1.0,
        "defense_score_weight": 9.0,
        "chronological_forecast": True,
        "comet_max_eta": 8.0,
        "counter_recapture": True,
        "recapture_min_gap": 0.5,
        "recapture_max_gap": 8.0,
        "recapture_score_weight": 6.0,
        "recapture_gap_cost": 0.25,
    }
    rust = RustVecEnv(
        num_envs=2,
        num_players=2,
        episode_steps=80,
        ship_speed=6.0,
        random_seed=0,
    )
    rust.reset()
    rust.step_subset_fast([0, 1], [[[], []], [[], []]])
    rows = [(0, 0), (0, 1), (1, 0), (1, 1)]

    profiled = rust.sniper_profile_actions(profile, rows, native_actions=False)
    builtin = rust.builtin_actions("sniper_v14", rows, native_actions=False)

    assert profiled == builtin


@pytest.mark.skipif(shutil.which("cargo") is None, reason="cargo is not installed")
def test_rust_vec_env_reset_subset_advances_seed():
    _build_rust_extension()
    from owars.training.rust_env import RustVecEnv

    rust = RustVecEnv(
        num_envs=1,
        num_players=2,
        episode_steps=80,
        ship_speed=6.0,
        random_seed=0,
    )
    rust.reset()
    rust.step_subset_fast([0], [[[], []]])
    first = rust.observation(0, 0)

    rust.reset_subset([0])
    rust.step_subset_fast([0], [[[], []]])
    second = rust.observation(0, 0)

    assert first["planets"] != second["planets"]


@pytest.mark.skipif(shutil.which("cargo") is None, reason="cargo is not installed")
def test_rust_vec_env_policy_batch_no_context_matches_features():
    _build_rust_extension()
    from owars.training.rust_env import RustVecEnv

    rust = RustVecEnv(
        num_envs=1,
        num_players=2,
        episode_steps=500,
        ship_speed=6.0,
        random_seed=0,
    )
    rust.reset()
    rows = [(0, 0), (0, 1)]
    with_context, contexts = rust.policy_batch(rows, device="cpu")
    without_context, no_contexts = rust.policy_batch_no_context(rows, device="cpu")

    assert len(contexts) == 2
    assert no_contexts == []
    assert torch.equal(without_context.planet_feats, with_context.planet_feats)
    assert torch.equal(without_context.planet_mask, with_context.planet_mask)
    assert torch.equal(without_context.planet_owned_mask, with_context.planet_owned_mask)
    assert torch.equal(without_context.planet_ids, with_context.planet_ids)
    assert torch.equal(without_context.planet_garrison, with_context.planet_garrison)
    assert torch.equal(without_context.fleet_feats, with_context.fleet_feats)
    assert torch.equal(without_context.fleet_mask, with_context.fleet_mask)


@pytest.mark.skipif(shutil.which("cargo") is None, reason="cargo is not installed")
def test_rust_state_legal_mask_matches_feature_legal_mask():
    _build_rust_extension()
    from owars.training.rust_env import RustVecEnv

    rust = RustVecEnv(
        num_envs=2,
        num_players=2,
        episode_steps=500,
        ship_speed=6.0,
        random_seed=0,
    )
    rust.reset()
    rust.step_subset_fast([0, 1], [[[], []], [[], []]])
    rows = [(0, 0), (1, 0), (0, 1), (1, 1)]
    fast, _contexts = rust.policy_batch(rows, device="cpu")
    frac = torch.full(fast.planet_mask.shape, 0.75, dtype=torch.float32).numpy()

    state_mask = rust._core.legal_target_mask_from_state(rows, frac)
    active_state_mask = rust._core.legal_target_mask_from_state_active(
        rows,
        frac,
        np.ones_like(frac, dtype=bool),
    )
    inactive_state_mask = rust._core.legal_target_mask_from_state_active(
        rows,
        frac,
        np.zeros_like(frac, dtype=bool),
    )
    mixed_active = np.zeros_like(frac, dtype=bool)
    owned_sources = (
        fast.planet_owned_mask.numpy().astype(bool)
        & fast.planet_mask.numpy().astype(bool)
    )
    for row, cols in enumerate(owned_sources):
        source_cols = np.flatnonzero(cols)
        if len(source_cols) > 0 and row % 2 == 0:
            mixed_active[row, source_cols[0]] = True
    mixed_state_mask = rust._core.legal_target_mask_from_state_active(
        rows,
        frac,
        mixed_active,
    )
    feature_mask = rust._core.legal_target_mask(
        rows,
        frac,
        fast.planet_owned_mask.numpy(),
        fast.planet_mask.numpy(),
        fast.planet_ids.numpy(),
    )

    assert state_mask.any()
    assert np.array_equal(state_mask, feature_mask)
    assert np.array_equal(active_state_mask, state_mask)
    assert not inactive_state_mask.any()
    expected_mixed = np.zeros_like(state_mask)
    expected_mixed[mixed_active] = state_mask[mixed_active]
    assert np.array_equal(mixed_state_mask, expected_mixed)


@pytest.mark.skipif(shutil.which("cargo") is None, reason="cargo is not installed")
def test_rust_state_sampler_matches_feature_sampler_after_comets_4p():
    _build_rust_extension()
    from owars.training.rust_env import RustVecEnv

    rust = RustVecEnv(
        num_envs=2,
        num_players=4,
        episode_steps=500,
        ship_speed=6.0,
        random_seed=1,
    )
    rust.reset()
    noop = [[[], [], [], []], [[], [], [], []]]
    for _ in range(65):
        rust.step_subset_fast([0, 1], noop)

    rows = [(1, 2), (0, 0), (1, 0), (0, 3), (1, 1), (0, 2)]
    fast, _contexts = rust.policy_batch(rows, device="cpu")
    frac = torch.linspace(
        0.25,
        0.85,
        steps=fast.planet_mask.numel(),
        dtype=torch.float32,
    ).reshape(fast.planet_mask.shape)
    frac_np = frac.numpy()

    state_mask = rust._core.legal_target_mask_from_state(rows, frac_np)
    feature_mask = rust._core.legal_target_mask(
        rows,
        frac_np,
        fast.planet_owned_mask.numpy(),
        fast.planet_mask.numpy(),
        fast.planet_ids.numpy(),
    )
    assert state_mask.any()
    assert np.array_equal(state_mask, feature_mask)

    has_target = state_mask.any(axis=-1)
    launch = (fast.planet_owned_mask.numpy() & has_target).astype(np.float32)
    target_idx = np.zeros(fast.planet_mask.shape, dtype=np.int64)
    for row in range(state_mask.shape[0]):
        for source in range(state_mask.shape[1]):
            legal_targets = np.flatnonzero(state_mask[row, source])
            if len(legal_targets):
                target_idx[row, source] = int(legal_targets[0])

    state_materialized = rust._core.materialize_actions_from_state(
        rows,
        launch,
        target_idx,
        frac_np,
        False,
    )
    masked_materialized = rust._core.materialize_masked_actions_from_state(
        rows,
        launch,
        target_idx,
        frac_np,
        False,
    )
    masked_fields_materialized = rust._core.materialize_masked_action_fields_from_state(
        rows,
        np.stack((launch, target_idx.astype(np.float32), frac_np), axis=-1),
        False,
    )
    feature_materialized = rust._core.materialize_actions(
        rows,
        launch,
        target_idx,
        frac_np,
        fast.planet_owned_mask.numpy(),
        fast.planet_mask.numpy(),
        fast.planet_ids.numpy(),
        False,
    )

    assert state_materialized["actions"] == feature_materialized["actions"]
    assert masked_materialized["actions"] == state_materialized["actions"]
    assert masked_fields_materialized["actions"] == state_materialized["actions"]
    assert np.array_equal(
        state_materialized["materialized"],
        feature_materialized["materialized"],
    )
    assert np.array_equal(
        masked_materialized["materialized"],
        state_materialized["materialized"],
    )
    assert np.array_equal(
        masked_fields_materialized["materialized"],
        state_materialized["materialized"],
    )


@pytest.mark.skipif(shutil.which("cargo") is None, reason="cargo is not installed")
def test_rust_fast_rollout_uses_policy_batch_no_context(monkeypatch):
    _build_rust_extension()
    from owars.policies.config import OrbitPolicyConfig
    from owars.policies.model import OrbitPolicy
    from owars.training.league import LEARNER_NAME, OpponentSlot
    from owars.training.rust_env import RustVecEnv
    from owars.training.vec_rollout import rollout_episodes_batched

    rust = RustVecEnv(
        num_envs=2,
        num_players=2,
        episode_steps=8,
        ship_speed=6.0,
        random_seed=0,
    )

    def fail_policy_batch(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("fast rollout should use policy_batch_no_context")

    monkeypatch.setattr(rust, "policy_batch", fail_policy_batch)
    model = OrbitPolicy(OrbitPolicyConfig(dim=16, ff_dim=32, depth=1, n_heads=2))
    opponent = OpponentSlot(LEARNER_NAME, agent=None)
    trajs = rollout_episodes_batched(
        model,
        rust,
        [[opponent], [opponent]],
        num_players=2,
        device="cpu",
    )

    assert len(trajs) == 2
    assert all(traj.encoded for traj in trajs)
    assert all(traj.reward for traj in trajs)


@pytest.mark.skipif(shutil.which("cargo") is None, reason="cargo is not installed")
def test_rust_fast_rollout_uses_native_sniper_without_observations(monkeypatch):
    _build_rust_extension()
    from owars.agents.sniper import sniper_agent
    from owars.policies.config import OrbitPolicyConfig
    from owars.policies.model import OrbitPolicy
    from owars.training.league import OpponentSlot
    from owars.training.rust_env import RustVecEnv
    from owars.training.vec_rollout import rollout_episodes_batched

    rust = RustVecEnv(
        num_envs=2,
        num_players=2,
        episode_steps=8,
        ship_speed=6.0,
        random_seed=0,
    )

    def fail_observation(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("native sniper rollout should not materialize observations")

    calls = {"builtin_actions": 0}
    original_builtin_actions = rust.builtin_actions

    def count_builtin_actions(*args: Any, **kwargs: Any) -> Any:
        calls["builtin_actions"] += 1
        return original_builtin_actions(*args, **kwargs)

    monkeypatch.setattr(rust, "observation", fail_observation)
    monkeypatch.setattr(rust, "observations", fail_observation)
    monkeypatch.setattr(rust, "builtin_actions", count_builtin_actions)

    model = OrbitPolicy(OrbitPolicyConfig(dim=16, ff_dim=32, depth=1, n_heads=2))
    opponent = OpponentSlot("sniper", agent=sniper_agent)
    trajs = rollout_episodes_batched(
        model,
        rust,
        [[opponent], [opponent]],
        num_players=2,
        device="cpu",
    )

    assert len(trajs) == 2
    assert calls["builtin_actions"] > 0


@pytest.mark.skipif(shutil.which("cargo") is None, reason="cargo is not installed")
def test_rust_vec_env_native_sampler_matches_context_sampler():
    _build_rust_extension()
    from owars.policies.model import PolicyOutput
    from owars.policies.sampling import (
        sample_batch_actions_context,
        sample_batch_with_records_context,
    )
    from owars.training.rust_env import RustVecEnv

    rust = RustVecEnv(
        num_envs=2,
        num_players=2,
        episode_steps=500,
        ship_speed=6.0,
        random_seed=0,
    )
    rust.reset()
    rust.step_subset_fast([0, 1], [[[], []], [[], []]])
    rows = [(0, 0), (0, 1), (1, 0), (1, 1)]
    fast, contexts = rust.policy_batch(rows, device="cpu")
    b, p = fast.planet_ids.shape
    launch_logits = torch.where(
        fast.planet_owned_mask,
        torch.full((b, p), 100.0),
        torch.full((b, p), -100.0),
    )
    target_logits = torch.zeros((b, p, p))
    out = PolicyOutput(
        launch_logits=launch_logits,
        target_logits=target_logits,
        value=torch.zeros(b),
        value_logits=torch.zeros(b, 51),
        planet_owned_mask=fast.planet_owned_mask,
        planet_mask=fast.planet_mask,
        planet_ids=fast.planet_ids,
        action_logit_softcap=8.0,
        fraction_alpha=torch.full((b, p), 20.0),
        fraction_beta=torch.full((b, p), 2.0),
    )

    expected_actions, expected_records = sample_batch_with_records_context(
        out, contexts, deterministic=True, record_rows=list(range(b))
    )
    got_actions, got_records = rust.sample_batch_with_records(
        out, rows, deterministic=True, record_rows=list(range(b))
    )

    assert got_actions == expected_actions
    assert rust.sample_batch_actions(
        out, rows, deterministic=True
    ) == sample_batch_actions_context(out, contexts, deterministic=True)
    assert torch.equal(got_records.launch, expected_records.launch)
    assert torch.equal(got_records.target_idx, expected_records.target_idx)
    assert torch.allclose(got_records.log_prob, expected_records.log_prob)
    assert torch.equal(got_records.target_legal_mask, expected_records.target_legal_mask)


@pytest.mark.skipif(shutil.which("cargo") is None, reason="cargo is not installed")
def test_rust_vec_env_native_sampler_matches_raw_python_sampler():
    _build_rust_extension()
    from owars.policies.model import PolicyOutput
    from owars.policies.sampling import (
        sample_batch_actions_raw,
        sample_batch_with_records_raw,
    )
    from owars.training.rust_env import RustVecEnv

    rust = RustVecEnv(
        num_envs=2,
        num_players=4,
        episode_steps=500,
        ship_speed=6.0,
        random_seed=3,
    )
    rust.reset()
    noop = [[[], [], [], []], [[], [], [], []]]
    for _ in range(55):
        rust.step_subset_fast([0, 1], noop)

    rows = [(0, 0), (1, 2), (0, 3), (1, 1)]
    fast, _contexts = rust.policy_batch(rows, device="cpu")
    raw_observations = rust.observations(rows)
    b, p = fast.planet_ids.shape

    source_rank = torch.arange(p, dtype=torch.float32).view(1, p, 1)
    target_rank = torch.arange(p, dtype=torch.float32).view(1, 1, p)
    target_logits = target_rank - 0.01 * source_rank
    out = PolicyOutput(
        launch_logits=torch.where(
            fast.planet_owned_mask,
            torch.full((b, p), 100.0),
            torch.full((b, p), -100.0),
        ),
        target_logits=target_logits.expand(b, -1, -1).clone(),
        value=torch.zeros(b),
        value_logits=torch.zeros(b, 51),
        planet_owned_mask=fast.planet_owned_mask,
        planet_mask=fast.planet_mask,
        planet_ids=fast.planet_ids,
        action_logit_softcap=8.0,
        fraction_alpha=torch.full((b, p), 12.0),
        fraction_beta=torch.full((b, p), 3.0),
    )

    expected_actions, expected_records = sample_batch_with_records_raw(
        out, raw_observations, deterministic=True, record_rows=list(range(b))
    )
    got_actions, got_records = rust.sample_batch_with_records(
        out, rows, deterministic=True, record_rows=list(range(b))
    )

    assert got_actions == expected_actions
    assert rust.sample_batch_actions(
        out, rows, deterministic=True
    ) == sample_batch_actions_raw(out, raw_observations, deterministic=True)
    assert torch.equal(got_records.launch, expected_records.launch)
    assert torch.equal(got_records.target_idx, expected_records.target_idx)
    assert torch.allclose(got_records.fraction, expected_records.fraction)
    assert torch.allclose(got_records.log_prob, expected_records.log_prob)
    assert torch.equal(got_records.target_legal_mask, expected_records.target_legal_mask)


@pytest.mark.skipif(shutil.which("cargo") is None, reason="cargo is not installed")
def test_rust_vec_env_policy_inference_matches_raw_python_path():
    _build_rust_extension()
    from owars.policies.config import OrbitPolicyConfig
    from owars.policies.model import OrbitPolicy
    from owars.policies.sampling import sample_batch_actions_raw
    from owars.training.rust_env import RustVecEnv

    torch.manual_seed(123)
    model = OrbitPolicy(
        OrbitPolicyConfig(
            dim=32,
            ff_dim=64,
            depth=1,
            n_heads=2,
            num_fleet_latents=8,
            fleet_tokenizer_depth=1,
            value_num_bins=51,
            critic_mtp_horizon=1,
        )
    ).eval()

    rust = RustVecEnv(
        num_envs=2,
        num_players=4,
        episode_steps=500,
        ship_speed=6.0,
        random_seed=5,
    )
    rust.reset()
    noop = [[[], [], [], []], [[], [], [], []]]
    for _ in range(52):
        rust.step_subset_fast([0, 1], noop)

    rows = [(0, 0), (1, 2), (0, 3), (1, 1)]
    fast, _contexts = rust.policy_batch(rows, device="cpu")
    raw_observations = rust.observations(rows)
    raw = encode_raw_observations(raw_observations, device="cpu")

    with torch.inference_mode():
        fast_out = model(fast, include_value=False)
        raw_out = model(raw, include_value=False)

    assert torch.allclose(fast_out.launch_logits, raw_out.launch_logits)
    assert torch.allclose(fast_out.target_logits, raw_out.target_logits, equal_nan=True)
    assert torch.allclose(fast_out.fraction_alpha, raw_out.fraction_alpha)
    assert torch.allclose(fast_out.fraction_beta, raw_out.fraction_beta)
    assert torch.equal(fast_out.planet_owned_mask, raw_out.planet_owned_mask)
    assert torch.equal(fast_out.planet_mask, raw_out.planet_mask)
    assert torch.equal(fast_out.planet_ids, raw_out.planet_ids)

    assert rust.sample_batch_actions(
        fast_out, rows, deterministic=True
    ) == sample_batch_actions_raw(raw_out, raw_observations, deterministic=True)


@pytest.mark.skipif(shutil.which("cargo") is None, reason="cargo is not installed")
def test_rust_vec_env_policy_batch_matches_comet_features():
    _build_rust_extension()
    from owars.training.numpy_env import NumpyVecEnv
    from owars.training.rust_env import RustVecEnv

    rust = RustVecEnv(
        num_envs=1,
        num_players=2,
        episode_steps=500,
        ship_speed=6.0,
        random_seed=0,
    )
    numpy = NumpyVecEnv(
        num_envs=1,
        num_players=2,
        episode_steps=500,
        ship_speed=6.0,
        random_seed=0,
    )
    numpy_states = numpy.reset()
    rust.reset()
    for _ in range(50):
        numpy_states[0] = numpy.step_subset([0], [[[], []]])[0][0]
        rust.step_subset_fast([0], [[[], []]])

    assert numpy_states[0][0]["observation"]["comet_planet_ids"]
    fast, _ = rust.policy_batch([(0, 0), (0, 1)], device="cpu")
    expected = encode_raw_observations(
        [numpy_states[0][0]["observation"], numpy_states[0][1]["observation"]],
        device="cpu",
    )
    assert torch.allclose(fast.planet_feats, expected.planet_feats)
    assert torch.equal(fast.planet_mask, expected.planet_mask)
    assert torch.equal(fast.planet_ids, expected.planet_ids)
    assert torch.allclose(fast.fleet_feats, expected.fleet_feats)
    assert torch.equal(fast.fleet_mask, expected.fleet_mask)


@pytest.mark.skipif(shutil.which("cargo") is None, reason="cargo is not installed")
def test_rust_vec_env_policy_batch_matches_4p_owner_slots():
    _build_rust_extension()
    from owars.training.numpy_env import NumpyVecEnv
    from owars.training.rust_env import RustVecEnv

    rust = RustVecEnv(
        num_envs=1,
        num_players=4,
        episode_steps=80,
        ship_speed=6.0,
        random_seed=0,
    )
    numpy = NumpyVecEnv(
        num_envs=1,
        num_players=4,
        episode_steps=80,
        ship_speed=6.0,
        random_seed=0,
    )
    numpy_states = numpy.reset()
    rust.reset()
    numpy_states[0] = numpy.step_subset([0], [[[], [], [], []]])[0][0]
    rust.step_subset_fast([0], [[[], [], [], []]])

    rows = [(0, seat) for seat in range(4)]
    fast, _ = rust.policy_batch(rows, device="cpu")
    expected = encode_raw_observations(
        [numpy_states[0][seat]["observation"] for seat in range(4)],
        device="cpu",
    )
    assert torch.allclose(fast.planet_feats, expected.planet_feats)
    assert torch.equal(fast.planet_mask, expected.planet_mask)
    assert torch.equal(fast.planet_owned_mask, expected.planet_owned_mask)
    assert torch.equal(fast.planet_ids, expected.planet_ids)
    assert torch.allclose(fast.fleet_feats, expected.fleet_feats)
    assert torch.equal(fast.fleet_mask, expected.fleet_mask)


@pytest.mark.skipif(shutil.which("cargo") is None, reason="cargo is not installed")
def test_rust_vec_env_reward_potentials_match_numpy():
    _build_rust_extension()
    from owars.training.numpy_env import NumpyVecEnv
    from owars.training.rust_env import RustVecEnv

    rust = RustVecEnv(
        num_envs=2,
        num_players=2,
        episode_steps=80,
        ship_speed=6.0,
        random_seed=3,
    )
    numpy = NumpyVecEnv(
        num_envs=2,
        num_players=2,
        episode_steps=80,
        ship_speed=6.0,
        random_seed=3,
    )
    numpy.reset()
    rust.reset()
    for _ in range(5):
        numpy.step_subset([0, 1], [[[], []], [[], []]])
        rust.step_subset_fast([0, 1], [[[], []], [[], []]])

    rows = [(0, 0), (0, 1), (1, 0), (1, 1)]
    rust_potentials = rust.reward_potentials(rows, production_weight=1.0)
    numpy_potentials = numpy.reward_potentials(rows, production_weight=1.0)
    assert rust_potentials.tolist() == pytest.approx(numpy_potentials.tolist())

    rust_production = rust.production_margins(rows)
    numpy_production = numpy.production_margins(rows)
    assert rust_production.tolist() == pytest.approx(numpy_production.tolist())
