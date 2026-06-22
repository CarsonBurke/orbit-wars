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

from owars.policies.features import MAX_PLANETS, encode_raw_observations
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


def test_native_required_api_rejects_stale_fast_rollout_extension() -> None:
    from types import SimpleNamespace

    from owars.training.rust_env import _native_has_required_api

    class StaleCore:
        def builtin_actions(self) -> None:
            pass

    class CurrentCore:
        def builtin_actions(self) -> None:
            pass

        def enqueue_builtin_actions(self) -> None:
            pass

        def step_subset_flat_actions(self) -> None:
            pass

        def step_subset_pending_actions(self) -> None:
            pass

        def categorical_beta_actions_from_state_compact_sources(self) -> None:
            pass

        def enqueue_categorical_beta_actions_from_state_compact_sources(self) -> None:
            pass

    assert not _native_has_required_api(SimpleNamespace(RustCoreVecEnv=StaleCore))
    assert _native_has_required_api(SimpleNamespace(RustCoreVecEnv=CurrentCore))


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
                key=lambda p: (
                    (float(p[2]) - float(src[2])) ** 2 + (float(p[3]) - float(src[3])) ** 2
                ),
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
    encoded, _ = rust.policy_batch_no_context(rows, device="cpu", include_fleet_targets=True)
    encoded_with_context, contexts = rust.policy_batch(
        rows,
        device="cpu",
        include_fleet_targets=True,
    )

    assert oracle["status"][:, :2].tolist() == [[1, 1], [1, 1]]
    assert oracle["dest_idx"][:, :2].tolist() == [[0, 1], [0, 1]]
    assert encoded.fleet_feats.shape == (2, 0, 20)
    assert encoded.fleet_mask.shape == (2, 0)
    assert encoded.fleet_target_planet_idx is not None
    assert encoded.fleet_target_planet_idx.shape == (2, 0)
    assert encoded.planet_inbound_feats is not None
    obs_p0 = dict(obs, player=0)
    obs_p1 = dict(obs, player=1)
    expected = encode_raw_observations(
        [obs_p0, obs_p1],
        include_fleet_targets=True,
    )
    assert expected.planet_inbound_feats is not None
    torch.testing.assert_close(encoded.planet_inbound_feats, expected.planet_inbound_feats)
    assert len(contexts) == 2
    assert encoded_with_context.fleet_feats.shape == (2, 0, 20)
    assert encoded_with_context.fleet_mask.shape == (2, 0)
    assert encoded_with_context.fleet_target_planet_idx is not None
    torch.testing.assert_close(
        encoded_with_context.planet_inbound_feats,
        expected.planet_inbound_feats,
    )


@pytest.mark.skipif(shutil.which("cargo") is None, reason="cargo is not installed")
def test_rust_policy_batch_supports_large_fleet_count():
    _build_rust_extension()
    from owars.training.rust_env import RustVecEnv

    obs = _oracle_obs(
        planets=[
            [0, 0, 20.0, 10.0, 1.0, 10, 1],
            [1, 1, 80.0, 10.0, 1.0, 10, 1],
        ],
        fleets=[[1000 + i, i % 2, 15.0, 10.0, 0.0, 0, 20] for i in range(400)],
    )
    rust = RustVecEnv(
        num_envs=1,
        num_players=2,
        episode_steps=500,
        ship_speed=6.0,
        random_seed=0,
    )
    rust._core.load_observation(0, obs)

    encoded, _ = rust.policy_batch_no_context(
        [(0, 0), (0, 1)], device="cpu", include_fleet_targets=True
    )

    assert encoded.fleet_feats.shape == (2, 0, 20)
    assert encoded.planet_inbound_feats is not None
    assert encoded.planet_inbound_feats.shape == (2, MAX_PLANETS, 13)
    assert encoded.fleet_mask.shape == (2, 0)
    assert encoded.fleet_target_planet_idx is not None
    assert encoded.fleet_target_planet_idx.shape == (2, 0)


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
    assert fast.fleet_target_planet_idx is None
    assert expected.fleet_target_planet_idx is None
    assert len(contexts) == 2

    obs = numpy_states[0][0]["observation"]
    src = next(p for p in obs["planets"] if int(p[1]) == 0 and int(p[5]) >= 8)
    target = next(p for p in obs["planets"] if int(p[1]) != 0)
    angle = math.atan2(float(target[3]) - float(src[3]), float(target[2]) - float(src[2]))
    action = [
        [[int(src[0]), angle, 5, int(target[0]), 3.0, float(target[2]), float(target[3])]],
        [],
    ]
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
    assert fast.fleet_target_planet_idx is None
    assert expected.fleet_target_planet_idx is None


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
def test_rust_core_flat_actions_match_nested_step_parser():
    _build_rust_extension()
    from owars.training.rust_env import RustVecEnv

    nested = RustVecEnv(
        num_envs=1,
        num_players=2,
        episode_steps=80,
        ship_speed=6.0,
        random_seed=0,
    )
    flat = RustVecEnv(
        num_envs=1,
        num_players=2,
        episode_steps=80,
        ship_speed=6.0,
        random_seed=0,
    )
    nested.reset()
    flat.reset()
    actions = [[], []]
    nested._core.step_subset_fast([0], [actions])
    flat._core.step_subset_flat_actions([0], [0, 0], [0, 1], actions)

    nested_features, _ = nested.policy_batch([(0, 0), (0, 1)], device="cpu")
    flat_features, _ = flat.policy_batch([(0, 0), (0, 1)], device="cpu")
    assert torch.allclose(nested_features.planet_feats, flat_features.planet_feats)
    assert torch.equal(nested_features.planet_mask, flat_features.planet_mask)
    assert torch.equal(nested_features.planet_owned_mask, flat_features.planet_owned_mask)
    assert torch.equal(nested_features.planet_ids, flat_features.planet_ids)
    assert torch.allclose(nested_features.fleet_feats, flat_features.fleet_feats)
    assert torch.equal(nested_features.fleet_mask, flat_features.fleet_mask)


@pytest.mark.skipif(shutil.which("cargo") is None, reason="cargo is not installed")
def test_rust_core_flat_actions_rejects_duplicate_player_rows():
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
    with pytest.raises(ValueError, match="flat action rows must be unique"):
        rust._core.step_subset_flat_actions([0], [0, 0], [0, 0], [[], []])


@pytest.mark.skipif(shutil.which("cargo") is None, reason="cargo is not installed")
def test_rust_pending_builtin_actions_match_flat_native_actions():
    _build_rust_extension()
    from owars.training.rust_env import RustVecEnv

    flat = RustVecEnv(
        num_envs=1,
        num_players=2,
        episode_steps=80,
        ship_speed=6.0,
        random_seed=0,
    )
    pending = RustVecEnv(
        num_envs=1,
        num_players=2,
        episode_steps=80,
        ship_speed=6.0,
        random_seed=0,
    )
    flat.reset()
    pending.reset()
    rows = [(0, 0), (0, 1)]
    actions = flat.builtin_actions("sniper_v17", rows, native_actions=True)

    flat.step_subset_flat_actions([0], [0, 0], [0, 1], actions)
    pending.enqueue_builtin_actions("sniper_v17", rows)
    pending.step_subset_pending_actions([0], [], [], [])

    flat_features, _ = flat.policy_batch(rows, device="cpu")
    pending_features, _ = pending.policy_batch(rows, device="cpu")
    assert torch.allclose(flat_features.planet_feats, pending_features.planet_feats)
    assert torch.equal(flat_features.planet_mask, pending_features.planet_mask)
    assert torch.equal(flat_features.planet_owned_mask, pending_features.planet_owned_mask)
    assert torch.equal(flat_features.planet_ids, pending_features.planet_ids)
    assert torch.allclose(flat_features.fleet_feats, pending_features.fleet_feats)
    assert torch.equal(flat_features.fleet_mask, pending_features.fleet_mask)


@pytest.mark.skipif(shutil.which("cargo") is None, reason="cargo is not installed")
def test_rust_step_subset_fast_accepts_grouped_native_actions():
    _build_rust_extension()
    from owars.training.rust_env import RustVecEnv

    flat = RustVecEnv(
        num_envs=1,
        num_players=2,
        episode_steps=80,
        ship_speed=6.0,
        random_seed=0,
    )
    grouped = RustVecEnv(
        num_envs=1,
        num_players=2,
        episode_steps=80,
        ship_speed=6.0,
        random_seed=0,
    )
    flat.reset()
    grouped.reset()
    rows = [(0, 0), (0, 1)]
    actions = flat.builtin_actions("sniper_v17", rows, native_actions=True)

    flat.step_subset_flat_actions([0], [0, 0], [0, 1], actions)
    grouped.step_subset_fast([0], [[actions[0], actions[1]]])

    flat_features, _ = flat.policy_batch(rows, device="cpu")
    grouped_features, _ = grouped.policy_batch(rows, device="cpu")
    assert torch.allclose(flat_features.planet_feats, grouped_features.planet_feats)
    assert torch.equal(flat_features.planet_mask, grouped_features.planet_mask)
    assert torch.equal(flat_features.planet_owned_mask, grouped_features.planet_owned_mask)
    assert torch.equal(flat_features.planet_ids, grouped_features.planet_ids)
    assert torch.allclose(flat_features.fleet_feats, grouped_features.fleet_feats)
    assert torch.equal(flat_features.fleet_mask, grouped_features.fleet_mask)


@pytest.mark.skipif(shutil.which("cargo") is None, reason="cargo is not installed")
def test_rust_pending_actions_allow_python_overrides():
    _build_rust_extension()
    from owars.training.rust_env import RustVecEnv

    flat = RustVecEnv(
        num_envs=1,
        num_players=2,
        episode_steps=80,
        ship_speed=6.0,
        random_seed=0,
    )
    pending = RustVecEnv(
        num_envs=1,
        num_players=2,
        episode_steps=80,
        ship_speed=6.0,
        random_seed=0,
    )
    flat.reset()
    pending.reset()
    rows = [(0, 0), (0, 1)]
    actions = flat.builtin_actions("sniper_v17", rows, native_actions=True)

    flat.step_subset_flat_actions([0], [0, 0], [0, 1], [[], actions[1]])
    pending.enqueue_builtin_actions("sniper_v17", rows)
    pending.step_subset_pending_actions([0], [0], [0], [[]])

    flat_features, _ = flat.policy_batch(rows, device="cpu")
    pending_features, _ = pending.policy_batch(rows, device="cpu")
    assert torch.allclose(flat_features.planet_feats, pending_features.planet_feats)
    assert torch.equal(flat_features.planet_mask, pending_features.planet_mask)
    assert torch.equal(flat_features.planet_owned_mask, pending_features.planet_owned_mask)
    assert torch.equal(flat_features.planet_ids, pending_features.planet_ids)
    assert torch.allclose(flat_features.fleet_feats, pending_features.fleet_feats)
    assert torch.equal(flat_features.fleet_mask, pending_features.fleet_mask)


@pytest.mark.skipif(shutil.which("cargo") is None, reason="cargo is not installed")
def test_rust_pending_actions_survive_failed_step_validation():
    _build_rust_extension()
    from owars.training.rust_env import RustVecEnv

    expected = RustVecEnv(
        num_envs=1,
        num_players=2,
        episode_steps=80,
        ship_speed=6.0,
        random_seed=0,
    )
    pending = RustVecEnv(
        num_envs=1,
        num_players=2,
        episode_steps=80,
        ship_speed=6.0,
        random_seed=0,
    )
    expected.reset()
    pending.reset()
    rows = [(0, 0), (0, 1)]
    actions = expected.builtin_actions("sniper_v17", rows, native_actions=True)

    expected.step_subset_flat_actions([0], [0, 0], [0, 1], actions)
    pending.enqueue_builtin_actions("sniper_v17", rows)
    with pytest.raises(ValueError, match="flat action rows must be unique"):
        pending.step_subset_pending_actions([0], [0, 0], [0, 0], [[], []])
    pending.step_subset_pending_actions([0], [], [], [])

    expected_features, _ = expected.policy_batch(rows, device="cpu")
    pending_features, _ = pending.policy_batch(rows, device="cpu")
    assert torch.allclose(expected_features.planet_feats, pending_features.planet_feats)
    assert torch.equal(expected_features.planet_mask, pending_features.planet_mask)
    assert torch.equal(expected_features.planet_owned_mask, pending_features.planet_owned_mask)
    assert torch.equal(expected_features.planet_ids, pending_features.planet_ids)
    assert torch.allclose(expected_features.fleet_feats, pending_features.fleet_feats)
    assert torch.equal(expected_features.fleet_mask, pending_features.fleet_mask)


@pytest.mark.skipif(shutil.which("cargo") is None, reason="cargo is not installed")
def test_rust_pending_enqueue_rejects_duplicate_rows():
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
    with pytest.raises(ValueError, match="pending action rows must be unique"):
        rust.enqueue_builtin_actions("sniper_v17", [(0, 0), (0, 0)])


@pytest.mark.skipif(shutil.which("cargo") is None, reason="cargo is not installed")
@pytest.mark.parametrize(
    ("name", "agent_name"),
    [
        # "sniper" is the default and resolves to the oracle-forecast v18.
        ("sniper", "sniper_v18_agent"),
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
        ("sniper_v18", "sniper_v18_agent"),
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

    # Compared at a single early step: the Python lead-solver and the Rust
    # lead-solver are not bit-identical for complex orbiting-target geometry deep
    # in an episode (sub-0.01 rad angle drift in the bisection), so deeper
    # full-action parity is not guaranteed. The destination-oracle horizon
    # consistency (the part v18 depends on) is covered bit-exactly by
    # test_destination_oracle.py::test_observation_episode_steps_threads_into_oracle.
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
    rust._core.load_observation(0, _fixture_obs())
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
    assert without_context.fleet_target_planet_idx is None
    assert with_context.fleet_target_planet_idx is None
    source_rows, source_cols = torch.nonzero(
        without_context.planet_owned_mask
        & without_context.planet_mask
        & (without_context.planet_garrison >= 2.0),
        as_tuple=True,
    )
    assert np.array_equal(
        without_context.compact_source_rows,
        source_rows.numpy().astype(np.int64, copy=False),
    )
    assert np.array_equal(
        without_context.compact_source_cols,
        source_cols.numpy().astype(np.int64, copy=False),
    )


@pytest.mark.skipif(shutil.which("cargo") is None, reason="cargo is not installed")
def test_rust_vec_env_policy_batch_no_context_direct_destination_matches_default():
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
    rust._core.load_observation(0, _fixture_obs())
    rows = [(0, 0), (0, 1)]
    expected, _ = rust.policy_batch_no_context(
        rows,
        device="cpu",
        include_fleet_targets=True,
        pin_memory=False,
        reuse_pinned_buffers=False,
    )
    got, contexts = rust.policy_batch_no_context(
        rows,
        device="cpu",
        include_fleet_targets=True,
        pin_memory=True,
        reuse_pinned_buffers=True,
    )

    assert contexts == []
    assert torch.equal(got.global_feats, expected.global_feats)
    assert torch.equal(got.planet_feats, expected.planet_feats)
    assert torch.equal(got.planet_mask, expected.planet_mask)
    assert torch.equal(got.planet_owned_mask, expected.planet_owned_mask)
    assert torch.equal(got.planet_ids, expected.planet_ids)
    assert torch.equal(got.planet_garrison, expected.planet_garrison)
    assert torch.equal(got.fleet_feats, expected.fleet_feats)
    assert torch.equal(got.fleet_mask, expected.fleet_mask)
    assert torch.equal(got.fleet_target_planet_idx, expected.fleet_target_planet_idx)
    assert torch.equal(got.planet_inbound_feats, expected.planet_inbound_feats)
    assert np.array_equal(got.compact_source_rows, expected.compact_source_rows)
    assert np.array_equal(got.compact_source_cols, expected.compact_source_cols)


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
    owned_sources = fast.planet_owned_mask.numpy().astype(bool) & fast.planet_mask.numpy().astype(
        bool
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
    active_fields = np.stack((frac, mixed_active.astype(np.float32)), axis=-1)
    compact = rust._core.compact_legal_target_mask_from_state_active_fields(
        rows,
        active_fields,
    )
    compact_state_mask = np.zeros_like(mixed_state_mask)
    if compact["row_idx"].size:
        compact_state_mask[compact["row_idx"], compact["source_idx"]] = compact["mask"]
    assert np.array_equal(compact_state_mask, mixed_state_mask)
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
def test_relaxed_target_legality_keeps_sun_mask_but_allows_planet_blocked_routes():
    _build_rust_extension()
    from owars.training.rust_env import RustVecEnv

    blocked_obs = {
        "player": 0,
        "step": 0,
        "planets": [
            [0, 0, 10.0, 10.0, 1.0, 50, 3],
            [1, 1, 90.0, 10.0, 1.0, 30, 2],
            [2, -1, 50.0, 5.0, 6.0, 10, 1],
        ],
        "fleets": [],
        "angular_velocity": 0.04,
        "initial_planets": [
            [0, 0, 10.0, 10.0, 1.0, 50, 3],
            [1, 1, 90.0, 10.0, 1.0, 30, 2],
            [2, -1, 50.0, 5.0, 6.0, 10, 1],
        ],
        "comet_planet_ids": [],
        "comets": [],
    }
    sun_obs = {
        **blocked_obs,
        "planets": [
            [0, 0, 10.0, 10.0, 1.0, 50, 3],
            [1, 1, 90.0, 90.0, 1.0, 30, 2],
        ],
        "initial_planets": [
            [0, 0, 10.0, 10.0, 1.0, 50, 3],
            [1, 1, 90.0, 90.0, 1.0, 30, 2],
        ],
    }

    strict = RustVecEnv(
        num_envs=1,
        num_players=2,
        episode_steps=500,
        ship_speed=6.0,
        random_seed=0,
        strict_target_legality=True,
    )
    relaxed = RustVecEnv(
        num_envs=1,
        num_players=2,
        episode_steps=500,
        ship_speed=6.0,
        random_seed=0,
        strict_target_legality=False,
    )
    frac = np.full((1, MAX_PLANETS), 0.75, dtype=np.float32)

    strict._core.load_observation(0, blocked_obs)
    relaxed._core.load_observation(0, blocked_obs)
    strict_blocked = strict._core.legal_target_mask_from_state([(0, 0)], frac)
    relaxed_blocked = relaxed._core.legal_target_mask_from_state([(0, 0)], frac)
    assert not strict_blocked[0, 0, 1]
    assert relaxed_blocked[0, 0, 1]

    strict._core.load_observation(0, sun_obs)
    relaxed._core.load_observation(0, sun_obs)
    strict_sun = strict._core.legal_target_mask_from_state([(0, 0)], frac)
    relaxed_sun = relaxed._core.legal_target_mask_from_state([(0, 0)], frac)
    assert not strict_sun[0, 0, 1]
    assert not relaxed_sun[0, 0, 1]


@pytest.mark.skipif(shutil.which("cargo") is None, reason="cargo is not installed")
def test_relaxed_compact_sampler_executes_recorded_planet_blocked_target():
    _build_rust_extension()
    from owars.training.rust_env import RustVecEnv

    obs = {
        "player": 0,
        "step": 0,
        "planets": [
            [0, 0, 10.0, 10.0, 1.0, 50, 3],
            [1, 1, 90.0, 10.0, 1.0, 30, 2],
            [2, -1, 50.0, 5.0, 6.0, 10, 1],
        ],
        "fleets": [],
        "angular_velocity": 0.04,
        "initial_planets": [
            [0, 0, 10.0, 10.0, 1.0, 50, 3],
            [1, 1, 90.0, 10.0, 1.0, 30, 2],
            [2, -1, 50.0, 5.0, 6.0, 10, 1],
        ],
        "comet_planet_ids": [],
        "comets": [],
    }
    rust = RustVecEnv(
        num_envs=1,
        num_players=2,
        episode_steps=500,
        ship_speed=6.0,
        random_seed=0,
        strict_target_legality=False,
    )
    rust._core.load_observation(0, obs)
    planets = MAX_PLANETS
    target_logits = np.full((1, planets), -8.0, dtype=np.float32)
    target_logits[0, 1] = 8.0
    result = rust._core.categorical_beta_actions_from_state_compact_sources(
        [(0, 0)],
        np.array([0], dtype=np.int64),
        np.array([0], dtype=np.int64),
        np.array([-8.0], dtype=np.float32),
        target_logits,
        np.array([8.0], dtype=np.float32),
        np.array([2.0], dtype=np.float32),
        planets,
        8.0,
        True,
        [0],
        False,
    )

    assert result["target_idx"][0, 0] == 1
    assert result["launch"][0, 0] == 1.0
    assert len(result["actions"][0]) == 1
    assert result["actions"][0][0][0] == 0
    assert result["actions"][0][0][3] == 1


def test_compact_record_legal_mask_keeps_non_sources_unconstrained():
    from owars.training.rust_env import _record_legal_mask_from_compact

    compact = {
        "row_idx": np.array([0, 2, 2], dtype=np.int64),
        "source_idx": np.array([1, 0, 2], dtype=np.int64),
        "mask": np.array(
            [
                [False, True, True],
                [True, False, True],
                [False, True, False],
            ],
            dtype=bool,
        ),
    }
    record_rows = [2, 0]
    source_mask = np.array(
        [
            [True, False, True],
            [False, True, False],
        ],
        dtype=bool,
    )

    record_target_legal = _record_legal_mask_from_compact(
        compact,
        record_rows=record_rows,
        planets=3,
    )
    got = np.where(source_mask[:, :, None], record_target_legal, True)

    expected = np.ones((2, 3, 3), dtype=bool)
    expected[0, 0] = [True, False, True]
    expected[0, 2] = [False, True, False]
    expected[1, 1] = [False, True, True]
    assert np.array_equal(got, expected)


@pytest.mark.skipif(shutil.which("cargo") is None, reason="cargo is not installed")
def test_rust_cpu_categorical_deferred_records_match_logprob_path():
    _build_rust_extension()
    from owars.policies.model import PolicyOutput
    from owars.training.ppo import compute_old_log_probs
    from owars.training.rust_env import RustVecEnv

    rust = RustVecEnv(
        num_envs=2,
        num_players=4,
        episode_steps=500,
        ship_speed=6.0,
        random_seed=11,
    )
    rust.reset()
    noop = [[[], [], [], []], [[], [], [], []]]
    for _ in range(45):
        rust.step_subset_fast([0, 1], noop)

    rows = [(0, 0), (1, 2), (0, 3), (1, 1)]
    fast, _contexts = rust.policy_batch(rows, device="cpu")
    b, p = fast.planet_ids.shape
    source_rank = torch.arange(p, dtype=torch.float32).view(1, p, 1)
    target_rank = torch.arange(p, dtype=torch.float32).view(1, 1, p)
    target_logits = target_rank - 0.03 * source_rank
    out = PolicyOutput(
        launch_logits=torch.where(
            fast.planet_owned_mask,
            torch.full((b, p), 7.0),
            torch.full((b, p), -7.0),
        ),
        target_logits=target_logits.expand(b, -1, -1).clone(),
        value=torch.zeros(b),
        value_logits=torch.zeros(b, 51),
        planet_owned_mask=fast.planet_owned_mask,
        planet_mask=fast.planet_mask,
        planet_ids=fast.planet_ids,
        action_logit_softcap=8.0,
        fraction_alpha=torch.full((b, p), 9.0),
        fraction_beta=torch.full((b, p), 2.0),
    )
    record_rows = [2, 0]

    got_actions, got_records = rust.sample_batch_with_records(
        out,
        rows,
        deterministic=False,
        record_rows=record_rows,
        native_actions=True,
        enqueue_actions=True,
        compute_log_prob=False,
        compact_legal_records=True,
    )

    assert got_actions is None
    assert not getattr(got_records, "old_log_prob_computed", False)
    assert torch.count_nonzero(got_records.log_prob) == 0

    class FixedPolicy(torch.nn.Module):
        def forward(self, _feats):
            return PolicyOutput(
                launch_logits=out.launch_logits[record_rows],
                target_logits=out.target_logits[record_rows],
                value=torch.zeros(len(record_rows)),
                value_logits=torch.zeros(len(record_rows), 51),
                planet_owned_mask=fast.planet_owned_mask[record_rows],
                planet_mask=fast.planet_mask[record_rows],
                planet_ids=fast.planet_ids[record_rows],
                action_logit_softcap=out.action_logit_softcap,
                fraction_alpha=out.fraction_alpha[record_rows],
                fraction_beta=out.fraction_beta[record_rows],
            )

    source_mask = (fast.planet_owned_mask & fast.planet_mask)[record_rows]
    from owars.training.train import _stack_target_legal_record_refs

    target_legal_mask = (
        _stack_target_legal_record_refs(
            [
                {
                    "planet_feats": fast.planet_feats[record_rows],
                    "planet_mask": fast.planet_mask[record_rows],
                    "planet_owned_mask": fast.planet_owned_mask[record_rows],
                    "target_legal_mask": got_records.target_legal_mask,
                    "target_legal_row_idx": got_records.target_legal_row_idx,
                    "target_legal_source_idx": got_records.target_legal_source_idx,
                    "target_legal_source_mask": got_records.target_legal_source_mask,
                }
            ],
            torch.arange(len(record_rows)),
        )
        if got_records.target_legal_mask is None
        else got_records.target_legal_mask
    )
    old_log_prob = compute_old_log_probs(
        FixedPolicy(),
        {
            "global_feats": fast.global_feats[record_rows],
            "planet_feats": fast.planet_feats[record_rows],
            "planet_mask": fast.planet_mask[record_rows],
            "planet_owned_mask": fast.planet_owned_mask[record_rows],
            "planet_ids": fast.planet_ids[record_rows],
            "planet_garrison": fast.planet_garrison[record_rows],
            "fleet_feats": fast.fleet_feats[record_rows],
            "fleet_mask": fast.fleet_mask[record_rows],
            "launch": got_records.launch,
            "target_idx": got_records.target_idx,
            "fraction": got_records.fraction,
            "target_legal_mask": target_legal_mask,
        },
        minibatch_size=len(record_rows),
    )
    assert torch.isfinite(old_log_prob[source_mask]).all()
    assert not torch.allclose(
        old_log_prob[source_mask],
        torch.zeros_like(old_log_prob[source_mask]),
    )


@pytest.mark.skipif(shutil.which("cargo") is None, reason="cargo is not installed")
def test_rust_pending_categorical_actions_match_flat_native_actions():
    _build_rust_extension()
    from owars.policies.model import PolicyOutput
    from owars.training.rust_env import RustVecEnv

    flat = RustVecEnv(
        num_envs=2,
        num_players=4,
        episode_steps=500,
        ship_speed=6.0,
        random_seed=14,
    )
    pending = RustVecEnv(
        num_envs=2,
        num_players=4,
        episode_steps=500,
        ship_speed=6.0,
        random_seed=14,
    )
    flat.reset()
    pending.reset()
    noop = [[[], [], [], []], [[], [], [], []]]
    for _ in range(45):
        flat.step_subset_fast([0, 1], noop)
        pending.step_subset_fast([0, 1], noop)

    rows = [(0, 0), (1, 2), (0, 3), (1, 1)]
    fast, _contexts = flat.policy_batch(rows, device="cpu")
    b, p = fast.planet_ids.shape
    source_rank = torch.arange(p, dtype=torch.float32).view(1, p, 1)
    target_rank = torch.arange(p, dtype=torch.float32).view(1, 1, p)
    out = PolicyOutput(
        launch_logits=torch.where(
            fast.planet_owned_mask,
            torch.full((b, p), 7.0),
            torch.full((b, p), -7.0),
        ),
        target_logits=(target_rank - 0.03 * source_rank).expand(b, -1, -1).clone(),
        value=torch.zeros(b),
        value_logits=torch.zeros(b, 51),
        planet_owned_mask=fast.planet_owned_mask,
        planet_mask=fast.planet_mask,
        planet_ids=fast.planet_ids,
        action_logit_softcap=8.0,
        fraction_alpha=torch.full((b, p), 9.0),
        fraction_beta=torch.full((b, p), 2.0),
    )

    actions = flat.sample_batch_actions(
        out,
        rows,
        deterministic=True,
        native_actions=True,
    )
    assert (
        pending.sample_batch_actions(
            out,
            rows,
            deterministic=True,
            native_actions=True,
            enqueue_actions=True,
        )
        is None
    )
    flat.step_subset_flat_actions(
        [0, 1],
        [env_idx for env_idx, _seat in rows],
        [seat for _env_idx, seat in rows],
        actions,
    )
    pending.step_subset_pending_actions([0, 1], [], [], [])

    flat_features, _ = flat.policy_batch(rows, device="cpu")
    pending_features, _ = pending.policy_batch(rows, device="cpu")
    assert torch.allclose(flat_features.planet_feats, pending_features.planet_feats)
    assert torch.equal(flat_features.planet_mask, pending_features.planet_mask)
    assert torch.equal(flat_features.planet_owned_mask, pending_features.planet_owned_mask)
    assert torch.equal(flat_features.planet_ids, pending_features.planet_ids)
    assert torch.allclose(flat_features.fleet_feats, pending_features.fleet_feats)
    assert torch.equal(flat_features.fleet_mask, pending_features.fleet_mask)


@pytest.mark.skipif(shutil.which("cargo") is None, reason="cargo is not installed")
def test_rust_pending_categorical_records_match_flat_native_actions():
    _build_rust_extension()
    from owars.policies.model import PolicyOutput
    from owars.training.rust_env import RustVecEnv

    flat = RustVecEnv(
        num_envs=2,
        num_players=4,
        episode_steps=500,
        ship_speed=6.0,
        random_seed=15,
    )
    pending = RustVecEnv(
        num_envs=2,
        num_players=4,
        episode_steps=500,
        ship_speed=6.0,
        random_seed=15,
    )
    flat.reset()
    pending.reset()
    noop = [[[], [], [], []], [[], [], [], []]]
    for _ in range(45):
        flat.step_subset_fast([0, 1], noop)
        pending.step_subset_fast([0, 1], noop)

    rows = [(0, 0), (1, 2), (0, 3), (1, 1)]
    fast, _contexts = flat.policy_batch(rows, device="cpu")
    b, p = fast.planet_ids.shape
    source_rank = torch.arange(p, dtype=torch.float32).view(1, p, 1)
    target_rank = torch.arange(p, dtype=torch.float32).view(1, 1, p)
    out = PolicyOutput(
        launch_logits=torch.where(
            fast.planet_owned_mask,
            torch.full((b, p), 7.0),
            torch.full((b, p), -7.0),
        ),
        target_logits=(target_rank - 0.03 * source_rank).expand(b, -1, -1).clone(),
        value=torch.zeros(b),
        value_logits=torch.zeros(b, 51),
        planet_owned_mask=fast.planet_owned_mask,
        planet_mask=fast.planet_mask,
        planet_ids=fast.planet_ids,
        action_logit_softcap=8.0,
        fraction_alpha=torch.full((b, p), 9.0),
        fraction_beta=torch.full((b, p), 2.0),
    )
    record_rows = [2, 0]

    actions, expected_records = flat.sample_batch_with_records(
        out,
        rows,
        deterministic=False,
        record_rows=record_rows,
        native_actions=True,
        compute_log_prob=False,
        compact_legal_records=True,
    )
    pending_actions, got_records = pending.sample_batch_with_records(
        out,
        rows,
        deterministic=False,
        record_rows=record_rows,
        native_actions=True,
        enqueue_actions=True,
        compute_log_prob=False,
        compact_legal_records=True,
    )

    assert pending_actions is None
    assert torch.equal(got_records.launch, expected_records.launch)
    assert torch.equal(got_records.target_idx, expected_records.target_idx)
    assert torch.allclose(got_records.fraction, expected_records.fraction)
    assert not getattr(got_records, "old_log_prob_computed", False)
    assert torch.count_nonzero(got_records.log_prob) == 0

    flat.step_subset_flat_actions(
        [0, 1],
        [env_idx for env_idx, _seat in rows],
        [seat for _env_idx, seat in rows],
        actions,
    )
    pending.step_subset_pending_actions([0, 1], [], [], [])

    flat_features, _ = flat.policy_batch(rows, device="cpu")
    pending_features, _ = pending.policy_batch(rows, device="cpu")
    assert torch.allclose(flat_features.planet_feats, pending_features.planet_feats)
    assert torch.equal(flat_features.planet_mask, pending_features.planet_mask)
    assert torch.equal(flat_features.planet_owned_mask, pending_features.planet_owned_mask)
    assert torch.equal(flat_features.planet_ids, pending_features.planet_ids)
    assert torch.allclose(flat_features.fleet_feats, pending_features.fleet_feats)
    assert torch.equal(flat_features.fleet_mask, pending_features.fleet_mask)


@pytest.mark.skipif(shutil.which("cargo") is None, reason="cargo is not installed")
def test_rust_pending_categorical_rejects_invalid_rows_without_panic():
    _build_rust_extension()
    from owars.policies.model import PolicyOutput
    from owars.training.rust_env import RustVecEnv

    rust = RustVecEnv(
        num_envs=1,
        num_players=2,
        episode_steps=80,
        ship_speed=6.0,
        random_seed=0,
    )
    rust.reset()
    fast, _contexts = rust.policy_batch([(0, 0)], device="cpu")
    b, p = fast.planet_ids.shape
    out = PolicyOutput(
        launch_logits=torch.full((b, p), 7.0),
        target_logits=torch.zeros((b, p, p)),
        value=torch.zeros(b),
        value_logits=torch.zeros(b, 51),
        planet_owned_mask=fast.planet_owned_mask,
        planet_mask=fast.planet_mask,
        planet_ids=fast.planet_ids,
        action_logit_softcap=8.0,
        fraction_alpha=torch.full((b, p), 9.0),
        fraction_beta=torch.full((b, p), 2.0),
    )

    with pytest.raises(IndexError, match="env index out of range"):
        rust.sample_batch_actions(
            out,
            [(99, 0)],
            deterministic=True,
            native_actions=True,
            enqueue_actions=True,
        )


@pytest.mark.skipif(shutil.which("cargo") is None, reason="cargo is not installed")
def test_rust_compact_categorical_target_planet_crop_matches_full_width():
    _build_rust_extension()
    from owars.policies.model import PolicyOutput
    from owars.training.rust_env import RustVecEnv

    rust = RustVecEnv(
        num_envs=1,
        num_players=2,
        episode_steps=80,
        ship_speed=6.0,
        random_seed=0,
    )
    rust.reset()
    rust._core.load_observation(0, _fixture_obs())
    rows = [(0, 0), (0, 1)]
    fast, _contexts = rust.policy_batch(rows, device="cpu")
    b, p = fast.planet_ids.shape
    fast_no_context, _contexts = rust.policy_batch_no_context(rows, device="cpu")
    assert fast_no_context.compact_target_planets == int(
        torch.nonzero(fast_no_context.planet_mask.any(dim=0))[-1].item()
    ) + 1
    source_indices = torch.nonzero(
        fast.planet_owned_mask & fast.planet_mask,
        as_tuple=False,
    )
    source_rows = np.ascontiguousarray(source_indices[:, 0].numpy(), dtype=np.int64)
    source_cols = np.ascontiguousarray(source_indices[:, 1].numpy(), dtype=np.int64)
    target_rank = torch.arange(p, dtype=torch.float32).view(1, 1, p)
    out = PolicyOutput(
        launch_logits=torch.where(
            fast.planet_owned_mask,
            torch.full((b, p), 7.0),
            torch.full((b, p), -7.0),
        ),
        target_logits=target_rank.expand(b, p, -1).clone(),
        value=torch.zeros(b),
        value_logits=torch.zeros(b, 51),
        planet_owned_mask=fast.planet_owned_mask,
        planet_mask=fast.planet_mask,
        planet_ids=fast.planet_ids,
        action_logit_softcap=8.0,
        fraction_alpha=torch.full((b, p), 9.0),
        fraction_beta=torch.full((b, p), 2.0),
    )
    live_planets = int(torch.nonzero(fast.planet_mask.any(dim=0))[-1].item()) + 1

    full = rust.sample_batch_actions(
        out,
        rows,
        deterministic=True,
        native_actions=True,
        compact_source_rows=source_rows,
        compact_source_cols=source_cols,
    )
    cropped_rust = RustVecEnv(
        num_envs=1,
        num_players=2,
        episode_steps=80,
        ship_speed=6.0,
        random_seed=0,
    )
    cropped_rust.reset()
    cropped_rust._core.load_observation(0, _fixture_obs())
    cropped = cropped_rust.sample_batch_actions(
        out,
        rows,
        deterministic=True,
        native_actions=True,
        compact_source_rows=source_rows,
        compact_source_cols=source_cols,
        compact_target_planets=live_planets,
    )

    rust.step_subset_flat_actions([0], [0, 0], [0, 1], full)
    cropped_rust.step_subset_flat_actions([0], [0, 0], [0, 1], cropped)
    full_features, _ = rust.policy_batch(rows, device="cpu")
    cropped_features, _ = cropped_rust.policy_batch(rows, device="cpu")
    assert torch.allclose(full_features.planet_feats, cropped_features.planet_feats)
    assert torch.equal(full_features.planet_mask, cropped_features.planet_mask)
    assert torch.equal(full_features.planet_owned_mask, cropped_features.planet_owned_mask)
    assert torch.equal(full_features.planet_ids, cropped_features.planet_ids)


@pytest.mark.skipif(shutil.which("cargo") is None, reason="cargo is not installed")
def test_rust_compact_categorical_record_crop_pads_to_full_width():
    _build_rust_extension()
    from owars.policies.model import PolicyOutput
    from owars.training.rust_env import RustVecEnv

    rust = RustVecEnv(
        num_envs=1,
        num_players=2,
        episode_steps=80,
        ship_speed=6.0,
        random_seed=0,
    )
    rust.reset()
    rust._core.load_observation(0, _fixture_obs())
    rows = [(0, 0), (0, 1)]
    fast, _contexts = rust.policy_batch(rows, device="cpu")
    b, p = fast.planet_ids.shape
    source_indices = torch.nonzero(
        fast.planet_owned_mask & fast.planet_mask,
        as_tuple=False,
    )
    source_rows = np.ascontiguousarray(source_indices[:, 0].numpy(), dtype=np.int64)
    source_cols = np.ascontiguousarray(source_indices[:, 1].numpy(), dtype=np.int64)
    target_rank = torch.arange(p, dtype=torch.float32).view(1, 1, p)
    out = PolicyOutput(
        launch_logits=torch.where(
            fast.planet_owned_mask,
            torch.full((b, p), 7.0),
            torch.full((b, p), -7.0),
        ),
        target_logits=target_rank.expand(b, p, -1).clone(),
        value=torch.zeros(b),
        value_logits=torch.zeros(b, 51),
        planet_owned_mask=fast.planet_owned_mask,
        planet_mask=fast.planet_mask,
        planet_ids=fast.planet_ids,
        action_logit_softcap=8.0,
        fraction_alpha=torch.full((b, p), 9.0),
        fraction_beta=torch.full((b, p), 2.0),
    )
    live_planets = int(torch.nonzero(fast.planet_mask.any(dim=0))[-1].item()) + 1

    _actions, records = rust.sample_batch_with_records(
        out,
        rows,
        deterministic=False,
        record_rows=[0, 1],
        native_actions=True,
        enqueue_actions=True,
        compute_log_prob=False,
        compact_legal_records=True,
        compact_source_rows=source_rows,
        compact_source_cols=source_cols,
        compact_target_planets=live_planets,
    )

    assert records.launch.shape == (len(rows), p)
    assert records.raw_launch.shape == (len(rows), p)
    assert records.target_idx.shape == (len(rows), p)
    assert records.fraction.shape == (len(rows), p)
    assert records.target_legal_source_mask.shape[1] == p
    assert torch.all(records.fraction[:, live_planets:] == 0.5)
    assert not records.target_legal_source_mask[:, live_planets:].any()
    assert torch.equal(records.source_row_idx, torch.from_numpy(source_rows))
    assert torch.equal(records.source_col_idx, torch.from_numpy(source_cols))
    assert records.source_launch.shape == (len(source_rows),)
    assert records.source_raw_launch.shape == (len(source_rows),)
    assert records.source_target_idx.shape == (len(source_rows),)
    assert records.source_fraction.shape == (len(source_rows),)
    assert records.source_target_legal_mask.shape == (len(source_rows), p)
    assert not records.source_target_legal_mask[:, live_planets:].any()
    assert records.source_row_offsets[0].item() == 0
    assert records.source_row_offsets[-1].item() == len(source_rows)


@pytest.mark.skipif(shutil.which("cargo") is None, reason="cargo is not installed")
def test_rust_source_major_categorical_uses_compact_sampler_on_cpu_deterministic():
    _build_rust_extension()
    from owars.policies.model import PolicyOutput
    from owars.training.rust_env import RustVecEnv

    rust = RustVecEnv(
        num_envs=1,
        num_players=2,
        episode_steps=80,
        ship_speed=6.0,
        random_seed=0,
    )
    rust.reset()
    rust._core.load_observation(0, _fixture_obs())
    rows = [(0, 0), (0, 1)]
    fast, _contexts = rust.policy_batch(rows, device="cpu")
    b, p = fast.planet_ids.shape
    source_indices = torch.nonzero(
        fast.planet_owned_mask & fast.planet_mask,
        as_tuple=False,
    )
    source_rows = source_indices[:, 0].long()
    source_cols = source_indices[:, 1].long()
    source_count = int(source_rows.numel())
    target_rank = torch.arange(p, dtype=torch.float32).view(1, p).expand(source_count, p)
    out = PolicyOutput(
        launch_logits=torch.full((source_count,), 7.0),
        target_logits=target_rank.clone(),
        value=torch.zeros(b),
        value_logits=torch.zeros(b, 51),
        planet_owned_mask=fast.planet_owned_mask,
        planet_mask=fast.planet_mask,
        planet_ids=fast.planet_ids,
        actor_source_rows=source_rows,
        actor_source_cols=source_cols,
        actor_source_valid=torch.ones(source_count, dtype=torch.bool),
        target_planets=p,
        action_logit_softcap=8.0,
        fraction_alpha=torch.full((source_count,), 9.0),
        fraction_beta=torch.full((source_count,), 2.0),
    )

    actions, records = rust.sample_batch_with_records(
        out,
        rows,
        deterministic=True,
        record_rows=[0, 1],
        native_actions=True,
        enqueue_actions=False,
        compute_log_prob=False,
        compact_legal_records=True,
    )

    assert len(actions) == len(rows)
    assert torch.equal(records.source_row_idx, source_rows)
    assert torch.equal(records.source_col_idx, source_cols)
    assert records.source_launch.shape == (source_count,)
    assert records.source_target_legal_mask.shape[1] == p


@pytest.mark.skipif(shutil.which("cargo") is None, reason="cargo is not installed")
def test_rust_enqueue_compact_categorical_rejects_duplicate_rows():
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
    empty_idx = np.empty(0, dtype=np.int64)
    empty_scalar = np.empty(0, dtype=np.float32)
    empty_target = np.empty((0, MAX_PLANETS), dtype=np.float32)

    with pytest.raises(ValueError, match="unique per env/player"):
        rust._core.enqueue_categorical_beta_actions_from_state_compact_sources(
            [(0, 0), (0, 0)],
            empty_idx,
            empty_idx,
            empty_scalar,
            empty_target,
            empty_scalar,
            empty_scalar,
            MAX_PLANETS,
            8.0,
            True,
            [],
        )


@pytest.mark.skipif(shutil.which("cargo") is None, reason="cargo is not installed")
def test_rust_compact_categorical_rejects_noncontiguous_arrays_without_panic():
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
    rust._core.load_observation(0, _fixture_obs())
    rows = [(0, 0), (0, 1)]
    fast, _contexts = rust.policy_batch(rows, device="cpu")
    _b, p = fast.planet_ids.shape
    source_indices = torch.nonzero(
        fast.planet_owned_mask & fast.planet_mask,
        as_tuple=False,
    )
    source_count = int(source_indices.shape[0])
    assert source_count > 0
    source_rows = np.ascontiguousarray(source_indices[:, 0].numpy(), dtype=np.int64)
    source_cols = np.ascontiguousarray(source_indices[:, 1].numpy(), dtype=np.int64)
    launch = np.zeros(source_count, dtype=np.float32)
    alpha = np.full(source_count, 9.0, dtype=np.float32)
    beta = np.full(source_count, 2.0, dtype=np.float32)
    target_storage = np.zeros((source_count, p * 2), dtype=np.float32)
    target_noncontiguous = target_storage[:, ::2]

    with pytest.raises(ValueError, match="target_logits must be contiguous"):
        rust._core.categorical_beta_actions_from_state_compact_sources(
            rows,
            source_rows,
            source_cols,
            launch,
            target_noncontiguous,
            alpha,
            beta,
            p,
            8.0,
            True,
            [],
            True,
        )


@pytest.mark.skipif(shutil.which("cargo") is None, reason="cargo is not installed")
def test_rust_compact_categorical_rejects_duplicate_sources():
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
    rust._core.load_observation(0, _fixture_obs())
    rows = [(0, 0), (0, 1)]
    fast, _contexts = rust.policy_batch(rows, device="cpu")
    _b, p = fast.planet_ids.shape
    source_indices = torch.nonzero(
        fast.planet_owned_mask & fast.planet_mask,
        as_tuple=False,
    )
    source = source_indices[0]
    source_rows = np.ascontiguousarray([int(source[0]), int(source[0])], dtype=np.int64)
    source_cols = np.ascontiguousarray([int(source[1]), int(source[1])], dtype=np.int64)
    launch = np.zeros(2, dtype=np.float32)
    target = np.zeros((2, p), dtype=np.float32)
    alpha = np.full(2, 9.0, dtype=np.float32)
    beta = np.full(2, 2.0, dtype=np.float32)

    with pytest.raises(ValueError, match="unique per row/source"):
        rust._core.categorical_beta_actions_from_state_compact_sources(
            rows,
            source_rows,
            source_cols,
            launch,
            target,
            alpha,
            beta,
            p,
            8.0,
            True,
            [],
            True,
        )


@pytest.mark.skipif(shutil.which("cargo") is None, reason="cargo is not installed")
def test_rust_compact_categorical_rejects_unsorted_sources():
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
    rust._core.load_observation(0, _fixture_obs())
    rows = [(0, 0), (0, 1)]
    fast, _contexts = rust.policy_batch(rows, device="cpu")
    _b, p = fast.planet_ids.shape
    source_indices = torch.nonzero(
        fast.planet_owned_mask & fast.planet_mask,
        as_tuple=False,
    )
    assert int(source_indices.shape[0]) >= 2
    swapped = source_indices[[1, 0]]
    source_rows = np.ascontiguousarray(swapped[:, 0].numpy(), dtype=np.int64)
    source_cols = np.ascontiguousarray(swapped[:, 1].numpy(), dtype=np.int64)
    launch = np.zeros(2, dtype=np.float32)
    target = np.zeros((2, p), dtype=np.float32)
    alpha = np.full(2, 9.0, dtype=np.float32)
    beta = np.full(2, 2.0, dtype=np.float32)

    with pytest.raises(ValueError, match="sorted by row/source"):
        rust._core.categorical_beta_actions_from_state_compact_sources(
            rows,
            source_rows,
            source_cols,
            launch,
            target,
            alpha,
            beta,
            p,
            8.0,
            True,
            [],
            True,
        )


@pytest.mark.skipif(shutil.which("cargo") is None, reason="cargo is not installed")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_rust_cuda_categorical_deferred_records_match_cpu_logprob_path():
    _build_rust_extension()
    from owars.policies.model import PolicyOutput
    from owars.training.rust_env import RustVecEnv

    rust = RustVecEnv(
        num_envs=2,
        num_players=4,
        episode_steps=500,
        ship_speed=6.0,
        random_seed=12,
    )
    rust.reset()
    noop = [[[], [], [], []], [[], [], [], []]]
    for _ in range(45):
        rust.step_subset_fast([0, 1], noop)

    rows = [(0, 0), (1, 2), (0, 3), (1, 1)]
    fast, _contexts = rust.policy_batch(rows, device="cpu")
    b, p = fast.planet_ids.shape
    source_rank = torch.arange(p, dtype=torch.float32).view(1, p, 1)
    target_rank = torch.arange(p, dtype=torch.float32).view(1, 1, p)
    target_logits = target_rank - 0.03 * source_rank
    launch_logits = torch.where(
        fast.planet_owned_mask,
        torch.full((b, p), 7.0),
        torch.full((b, p), -7.0),
    )
    expected_out = PolicyOutput(
        launch_logits=launch_logits,
        target_logits=target_logits.expand(b, -1, -1).clone(),
        value=torch.zeros(b),
        value_logits=torch.zeros(b, 51),
        planet_owned_mask=fast.planet_owned_mask,
        planet_mask=fast.planet_mask,
        planet_ids=fast.planet_ids,
        action_logit_softcap=8.0,
        fraction_alpha=torch.full((b, p), 9.0),
        fraction_beta=torch.full((b, p), 2.0),
    )
    cuda_out = PolicyOutput(
        launch_logits=expected_out.launch_logits.cuda(),
        target_logits=expected_out.target_logits.cuda(),
        value=torch.zeros(b, device="cuda"),
        value_logits=torch.zeros(b, 51, device="cuda"),
        planet_owned_mask=fast.planet_owned_mask.cuda(),
        planet_mask=fast.planet_mask.cuda(),
        planet_ids=fast.planet_ids.cuda(),
        action_logit_softcap=8.0,
        fraction_alpha=expected_out.fraction_alpha.cuda(),
        fraction_beta=expected_out.fraction_beta.cuda(),
    )
    record_rows = [2, 0]
    timings: dict[str, float] = {}

    expected_actions, expected_records = rust.sample_batch_with_records(
        expected_out,
        rows,
        deterministic=True,
        record_rows=record_rows,
        compute_log_prob=True,
    )
    got_actions, got_records = rust.sample_batch_with_records(
        cuda_out,
        rows,
        deterministic=True,
        record_rows=record_rows,
        compute_log_prob=False,
        timings=timings,
    )

    assert got_actions == expected_actions
    assert torch.equal(got_records.launch, expected_records.launch)
    assert torch.equal(got_records.target_idx, expected_records.target_idx)
    launched = got_records.launch > 0.5
    assert torch.allclose(got_records.fraction[launched], expected_records.fraction[launched])
    assert torch.equal(got_records.target_legal_mask, expected_records.target_legal_mask)
    assert torch.count_nonzero(got_records.log_prob) == 0
    assert not getattr(got_records, "old_log_prob_computed", False)
    assert timings["native_action_s"] > 0.0
    assert "action_select_s" not in timings
    assert rust.sample_batch_actions(
        cuda_out,
        rows,
        deterministic=True,
    ) == rust.sample_batch_actions(expected_out, rows, deterministic=True)


@pytest.mark.skipif(shutil.which("cargo") is None, reason="cargo is not installed")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_rust_cuda_categorical_no_record_stochastic_uses_native_compact_path():
    _build_rust_extension()
    from owars.policies.model import PolicyOutput
    from owars.training.rust_env import RustVecEnv

    rust = RustVecEnv(
        num_envs=2,
        num_players=4,
        episode_steps=500,
        ship_speed=6.0,
        random_seed=41,
    )
    rust.reset()
    noop = [[[], [], [], []], [[], [], [], []]]
    for _ in range(45):
        rust.step_subset_fast([0, 1], noop)

    rows = [(0, 0), (1, 2), (0, 3), (1, 1)]
    fast, _contexts = rust.policy_batch(rows, device="cpu")
    b, p = fast.planet_ids.shape
    source_rank = torch.arange(p, dtype=torch.float32).view(1, p, 1)
    target_rank = torch.arange(p, dtype=torch.float32).view(1, 1, p)
    target_logits = (target_rank - 0.025 * source_rank).expand(b, -1, -1).clone()
    launch_logits = torch.where(
        fast.planet_owned_mask,
        torch.full((b, p), 4.0),
        torch.full((b, p), -4.0),
    )
    cpu_out = PolicyOutput(
        launch_logits=launch_logits,
        target_logits=target_logits,
        value=torch.zeros(b),
        value_logits=torch.zeros(b, 51),
        planet_owned_mask=fast.planet_owned_mask,
        planet_mask=fast.planet_mask,
        planet_ids=fast.planet_ids,
        action_logit_softcap=8.0,
        fraction_alpha=torch.full((b, p), 6.0),
        fraction_beta=torch.full((b, p), 3.0),
    )
    cuda_out = PolicyOutput(
        launch_logits=cpu_out.launch_logits.cuda(),
        target_logits=cpu_out.target_logits.cuda(),
        value=torch.zeros(b, device="cuda"),
        value_logits=torch.zeros(b, 51, device="cuda"),
        planet_owned_mask=fast.planet_owned_mask.cuda(),
        planet_mask=fast.planet_mask.cuda(),
        planet_ids=fast.planet_ids.cuda(),
        action_logit_softcap=8.0,
        fraction_alpha=cpu_out.fraction_alpha.cuda(),
        fraction_beta=cpu_out.fraction_beta.cuda(),
    )
    cpu_timings: dict[str, float] = {}
    cuda_timings: dict[str, float] = {}

    expected_actions, expected_records = rust.sample_batch_with_records(
        cpu_out,
        rows,
        deterministic=False,
        record_rows=[],
        compute_log_prob=False,
        timings=cpu_timings,
    )
    got_actions, got_records = rust.sample_batch_with_records(
        cuda_out,
        rows,
        deterministic=False,
        record_rows=[],
        compute_log_prob=False,
        timings=cuda_timings,
    )

    assert got_actions == expected_actions
    assert got_records.launch.numel() == 0
    assert expected_records.launch.numel() == 0
    assert cpu_timings["native_action_s"] > 0.0
    assert cuda_timings["native_action_s"] > 0.0
    assert "action_select_s" not in cuda_timings
    assert "legal_rust_s" not in cuda_timings


@pytest.mark.skipif(shutil.which("cargo") is None, reason="cargo is not installed")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_rust_cuda_categorical_deferred_stochastic_records_recompute_logprob():
    _build_rust_extension()
    from owars.policies.model import PolicyOutput
    from owars.policies.sampling import (
        _categorical_action_log_probs,
        _fraction_log_prob,
    )
    from owars.training.rust_env import RustVecEnv

    rust = RustVecEnv(
        num_envs=2,
        num_players=4,
        episode_steps=500,
        ship_speed=6.0,
        random_seed=13,
    )
    rust.reset()
    noop = [[[], [], [], []], [[], [], [], []]]
    for _ in range(45):
        rust.step_subset_fast([0, 1], noop)

    rows = [(0, 0), (1, 2), (0, 3), (1, 1)]
    fast, _contexts = rust.policy_batch(rows, device="cpu")
    b, p = fast.planet_ids.shape
    source_rank = torch.arange(p, dtype=torch.float32).view(1, p, 1)
    target_rank = torch.arange(p, dtype=torch.float32).view(1, 1, p)
    target_logits = target_rank - 0.02 * source_rank
    launch_logits = torch.where(
        fast.planet_owned_mask,
        torch.full((b, p), 4.0),
        torch.full((b, p), -4.0),
    )
    cpu_out = PolicyOutput(
        launch_logits=launch_logits,
        target_logits=target_logits.expand(b, -1, -1).clone(),
        value=torch.zeros(b),
        value_logits=torch.zeros(b, 51),
        planet_owned_mask=fast.planet_owned_mask,
        planet_mask=fast.planet_mask,
        planet_ids=fast.planet_ids,
        action_logit_softcap=8.0,
        fraction_alpha=torch.full((b, p), 6.0),
        fraction_beta=torch.full((b, p), 3.0),
    )
    cuda_out = PolicyOutput(
        launch_logits=cpu_out.launch_logits.cuda(),
        target_logits=cpu_out.target_logits.cuda(),
        value=torch.zeros(b, device="cuda"),
        value_logits=torch.zeros(b, 51, device="cuda"),
        planet_owned_mask=fast.planet_owned_mask.cuda(),
        planet_mask=fast.planet_mask.cuda(),
        planet_ids=fast.planet_ids.cuda(),
        action_logit_softcap=8.0,
        fraction_alpha=cpu_out.fraction_alpha.cuda(),
        fraction_beta=cpu_out.fraction_beta.cuda(),
    )
    record_rows = [0, 2]
    timings: dict[str, float] = {}

    torch.manual_seed(1234)
    actions, records = rust.sample_batch_with_records(
        cuda_out,
        rows,
        deterministic=False,
        record_rows=record_rows,
        compute_log_prob=False,
        timings=timings,
    )
    torch.manual_seed(1234)
    repeated_actions, repeated_records = rust.sample_batch_with_records(
        cuda_out,
        rows,
        deterministic=False,
        record_rows=record_rows,
        compute_log_prob=False,
    )

    assert actions == repeated_actions
    assert timings["native_action_s"] > 0.0
    assert "action_select_s" not in timings
    assert torch.equal(records.launch, repeated_records.launch)
    assert torch.equal(records.target_idx, repeated_records.target_idx)
    assert torch.allclose(records.fraction, repeated_records.fraction)
    assert torch.equal(records.target_legal_mask, repeated_records.target_legal_mask)
    assert torch.count_nonzero(records.log_prob) == 0
    assert not getattr(records, "old_log_prob_computed", False)

    target_logits_r = cpu_out.target_logits[record_rows].masked_fill(
        ~records.target_legal_mask,
        float("-inf"),
    )
    action_log_probs = _categorical_action_log_probs(
        cpu_out.launch_logits[record_rows],
        target_logits_r,
        cpu_out.action_logit_softcap,
    )
    action_idx = torch.where(
        records.launch > 0.5,
        records.target_idx.clamp(0, p - 1) + 1,
        torch.zeros_like(records.target_idx),
    )
    action_lp = action_log_probs.gather(-1, action_idx.unsqueeze(-1)).squeeze(-1)
    frac_lp = _fraction_log_prob(
        cpu_out.fraction_alpha[record_rows],
        cpu_out.fraction_beta[record_rows],
        records.fraction,
        "beta",
    )
    recomputed = action_lp + records.launch.float() * frac_lp

    assert torch.isfinite(recomputed).all()
    assert not torch.allclose(recomputed, torch.zeros_like(recomputed))


@pytest.mark.skipif(shutil.which("cargo") is None, reason="cargo is not installed")
def test_native_categorical_records_depleted_owned_planet_as_constrained():
    _build_rust_extension()
    from owars.training.rust_env import RustVecEnv

    obs = {
        "player": 0,
        "step": 10,
        "planets": [
            [0, 0, 10.0, 10.0, 1.0, 1, 3],
            [1, 0, 20.0, 10.0, 1.0, 30, 3],
            [2, 1, 90.0, 90.0, 1.0, 30, 3],
        ],
        "fleets": [],
        "angular_velocity": 0.0,
        "initial_planets": [],
        "next_fleet_id": 0,
        "comets": [],
        "comet_planet_ids": [],
    }
    rust = RustVecEnv(
        num_envs=1,
        num_players=2,
        episode_steps=500,
        ship_speed=6.0,
        random_seed=0,
    )
    rust._core.load_observation(0, obs)
    planets = MAX_PLANETS
    launch_logits = np.full((1, planets), 8.0, dtype=np.float32)
    target_logits = np.zeros((1, planets, planets), dtype=np.float32)
    alpha = np.full((1, planets), 8.0, dtype=np.float32)
    beta = np.full((1, planets), 2.0, dtype=np.float32)
    alpha[0, 0] = 30.0
    beta[0, 0] = 1.5

    result = rust._core.categorical_beta_actions_from_state(
        [(0, 0)],
        launch_logits,
        target_logits,
        alpha,
        beta,
        8.0,
        True,
        [0],
        False,
    )

    assert not result["target_legal_mask"][0, 0].any()

    stochastic = rust._core.categorical_beta_actions_from_state(
        [(0, 0)],
        launch_logits,
        target_logits,
        alpha,
        beta,
        8.0,
        False,
        [0],
        False,
    )
    source_cols = np.array([0, 1], dtype=np.int64)
    compact = rust._core.categorical_beta_actions_from_state_compact_sources(
        [(0, 0)],
        np.zeros_like(source_cols),
        source_cols,
        np.ascontiguousarray(launch_logits[0, source_cols]),
        np.ascontiguousarray(target_logits[0, source_cols]),
        np.ascontiguousarray(alpha[0, source_cols]),
        np.ascontiguousarray(beta[0, source_cols]),
        planets,
        8.0,
        False,
        [0],
        False,
    )

    assert np.array_equal(compact["launch"], stochastic["launch"])
    assert np.array_equal(compact["target_idx"], stochastic["target_idx"])
    assert np.allclose(compact["fraction"], stochastic["fraction"])


@pytest.mark.skipif(shutil.which("cargo") is None, reason="cargo is not installed")
def test_native_categorical_ignores_nonfinite_legal_target_logits():
    _build_rust_extension()
    from owars.training.rust_env import RustVecEnv

    obs = {
        "player": 0,
        "step": 10,
        "planets": [
            [0, 0, 10.0, 10.0, 1.0, 30, 3],
            [1, 1, 90.0, 90.0, 1.0, 30, 3],
        ],
        "fleets": [],
        "angular_velocity": 0.0,
        "initial_planets": [],
        "next_fleet_id": 0,
        "comets": [],
        "comet_planet_ids": [],
    }
    rust = RustVecEnv(
        num_envs=1,
        num_players=2,
        episode_steps=500,
        ship_speed=6.0,
        random_seed=0,
    )
    rust._core.load_observation(0, obs)
    planets = MAX_PLANETS
    launch_logits = np.full((1, planets), -8.0, dtype=np.float32)
    target_logits = np.full((1, planets, planets), -8.0, dtype=np.float32)
    target_logits[0, 0, 1] = np.inf
    alpha = np.full((1, planets), 8.0, dtype=np.float32)
    beta = np.full((1, planets), 2.0, dtype=np.float32)

    result = rust._core.categorical_beta_actions_from_state(
        [(0, 0)],
        launch_logits,
        target_logits,
        alpha,
        beta,
        8.0,
        True,
        [0],
        False,
    )

    assert result["launch"][0, 0] == 0.0
    assert result["actions"][0] == []


@pytest.mark.skipif(shutil.which("cargo") is None, reason="cargo is not installed")
def test_native_categorical_softcaps_positive_infinite_noop_logit():
    _build_rust_extension()
    from owars.training.rust_env import RustVecEnv

    obs = {
        "player": 0,
        "step": 10,
        "planets": [
            [0, 0, 10.0, 10.0, 1.0, 30, 3],
            [1, 1, 90.0, 90.0, 1.0, 30, 3],
        ],
        "fleets": [],
        "angular_velocity": 0.0,
        "initial_planets": [],
        "next_fleet_id": 0,
        "comets": [],
        "comet_planet_ids": [],
    }
    rust = RustVecEnv(
        num_envs=1,
        num_players=2,
        episode_steps=500,
        ship_speed=6.0,
        random_seed=0,
    )
    rust._core.load_observation(0, obs)
    planets = MAX_PLANETS
    launch_logits = np.full((1, planets), -8.0, dtype=np.float32)
    launch_logits[0, 0] = np.inf
    target_logits = np.full((1, planets, planets), 8.0, dtype=np.float32)
    alpha = np.full((1, planets), 8.0, dtype=np.float32)
    beta = np.full((1, planets), 2.0, dtype=np.float32)

    result = rust._core.categorical_beta_actions_from_state(
        [(0, 0)],
        launch_logits,
        target_logits,
        alpha,
        beta,
        8.0,
        True,
        [0],
        False,
    )

    assert result["launch"][0, 0] == 0.0
    assert result["actions"][0] == []


@pytest.mark.skipif(shutil.which("cargo") is None, reason="cargo is not installed")
def test_native_categorical_compact_sources_match_dense_stochastic():
    _build_rust_extension()
    from owars.training.rust_env import RustVecEnv

    rust = RustVecEnv(
        num_envs=2,
        num_players=4,
        episode_steps=500,
        ship_speed=6.0,
        random_seed=29,
    )
    rust.reset()
    noop = [[[], [], [], []], [[], [], [], []]]
    for _ in range(35):
        rust.step_subset_fast([0, 1], noop)

    rows = [(0, 0), (1, 2), (0, 3), (1, 1)]
    fast, _contexts = rust.policy_batch(rows, device="cpu")
    batch, planets = fast.planet_ids.shape
    source_rank = torch.arange(planets, dtype=torch.float32).view(1, planets, 1)
    target_rank = torch.arange(planets, dtype=torch.float32).view(1, 1, planets)
    target_logits_t = (target_rank - 0.03 * source_rank).expand(batch, -1, -1).clone()
    launch_logits_t = torch.where(
        fast.planet_owned_mask,
        torch.full((batch, planets), 3.0),
        torch.full((batch, planets), -3.0),
    )
    alpha_t = torch.full((batch, planets), 5.0)
    beta_t = torch.full((batch, planets), 2.0)
    target_logits = target_logits_t.numpy().astype(np.float32, copy=True)
    launch_logits = launch_logits_t.numpy().astype(np.float32, copy=True)
    alpha = alpha_t.numpy().astype(np.float32, copy=True)
    beta = beta_t.numpy().astype(np.float32, copy=True)
    record_rows = [0, 2]

    dense = rust._core.categorical_beta_actions_from_state(
        rows,
        launch_logits,
        target_logits,
        alpha,
        beta,
        8.0,
        False,
        record_rows,
        False,
    )
    source_indices = torch.nonzero(
        fast.planet_owned_mask & fast.planet_mask,
        as_tuple=False,
    ).numpy()
    compact = rust._core.categorical_beta_actions_from_state_compact_sources(
        rows,
        np.ascontiguousarray(source_indices[:, 0], dtype=np.int64),
        np.ascontiguousarray(source_indices[:, 1], dtype=np.int64),
        np.ascontiguousarray(launch_logits[source_indices[:, 0], source_indices[:, 1]]),
        np.ascontiguousarray(target_logits[source_indices[:, 0], source_indices[:, 1]]),
        np.ascontiguousarray(alpha[source_indices[:, 0], source_indices[:, 1]]),
        np.ascontiguousarray(beta[source_indices[:, 0], source_indices[:, 1]]),
        planets,
        8.0,
        False,
        record_rows,
        False,
    )

    assert compact["actions"] == dense["actions"]
    assert np.array_equal(compact["launch"], dense["launch"])
    assert np.array_equal(compact["target_idx"], dense["target_idx"])
    assert np.allclose(compact["fraction"], dense["fraction"])
    assert np.array_equal(compact["target_legal_mask"], dense["target_legal_mask"])


@pytest.mark.skipif(shutil.which("cargo") is None, reason="cargo is not installed")
def test_native_categorical_empty_record_rows_return_action_only_payloads():
    _build_rust_extension()
    from owars.training.rust_env import RustVecEnv

    rust = RustVecEnv(
        num_envs=2,
        num_players=4,
        episode_steps=500,
        ship_speed=6.0,
        random_seed=29,
    )
    rust.reset()
    rows = [(0, 0), (1, 2), (0, 3), (1, 1)]
    fast, _contexts = rust.policy_batch(rows, device="cpu")
    batch, planets = fast.planet_ids.shape
    target_logits = np.zeros((batch, planets, planets), dtype=np.float32)
    launch_logits = np.where(
        fast.planet_owned_mask.numpy(),
        np.float32(3.0),
        np.float32(-3.0),
    )
    alpha = np.full((batch, planets), 5.0, dtype=np.float32)
    beta = np.full((batch, planets), 2.0, dtype=np.float32)
    source_indices = torch.nonzero(
        fast.planet_owned_mask & fast.planet_mask,
        as_tuple=False,
    ).numpy()

    dense = rust._core.categorical_beta_actions_from_state(
        rows,
        launch_logits,
        target_logits,
        alpha,
        beta,
        8.0,
        True,
        [],
        False,
    )
    compact = rust._core.categorical_beta_actions_from_state_compact_sources(
        rows,
        np.ascontiguousarray(source_indices[:, 0], dtype=np.int64),
        np.ascontiguousarray(source_indices[:, 1], dtype=np.int64),
        np.ascontiguousarray(launch_logits[source_indices[:, 0], source_indices[:, 1]]),
        np.ascontiguousarray(target_logits[source_indices[:, 0], source_indices[:, 1]]),
        np.ascontiguousarray(alpha[source_indices[:, 0], source_indices[:, 1]]),
        np.ascontiguousarray(beta[source_indices[:, 0], source_indices[:, 1]]),
        planets,
        8.0,
        True,
        [],
        False,
    )
    enqueued = rust._core.enqueue_categorical_beta_actions_from_state_compact_sources(
        rows,
        np.ascontiguousarray(source_indices[:, 0], dtype=np.int64),
        np.ascontiguousarray(source_indices[:, 1], dtype=np.int64),
        np.ascontiguousarray(launch_logits[source_indices[:, 0], source_indices[:, 1]]),
        np.ascontiguousarray(target_logits[source_indices[:, 0], source_indices[:, 1]]),
        np.ascontiguousarray(alpha[source_indices[:, 0], source_indices[:, 1]]),
        np.ascontiguousarray(beta[source_indices[:, 0], source_indices[:, 1]]),
        planets,
        8.0,
        True,
        [],
    )

    assert set(dense.keys()) == {"actions"}
    assert set(compact.keys()) == {"actions"}
    assert dense["actions"] == compact["actions"]
    assert dict(enqueued) == {}


@pytest.mark.skipif(shutil.which("cargo") is None, reason="cargo is not installed")
def test_native_categorical_compact_sources_match_dense_deterministic_actions():
    _build_rust_extension()
    from owars.training.rust_env import RustVecEnv

    rust = RustVecEnv(
        num_envs=2,
        num_players=4,
        episode_steps=500,
        ship_speed=6.0,
        random_seed=31,
    )
    rust.reset()
    noop = [[[], [], [], []], [[], [], [], []]]
    for _ in range(35):
        rust.step_subset_fast([0, 1], noop)

    rows = [(0, 0), (1, 2), (0, 3), (1, 1)]
    fast, _contexts = rust.policy_batch(rows, device="cpu")
    batch, planets = fast.planet_ids.shape
    source_rank = torch.arange(planets, dtype=torch.float32).view(1, planets, 1)
    target_rank = torch.arange(planets, dtype=torch.float32).view(1, 1, planets)
    target_logits_t = (target_rank - 0.03 * source_rank).expand(batch, -1, -1).clone()
    launch_logits_t = torch.where(
        fast.planet_owned_mask,
        torch.full((batch, planets), 3.0),
        torch.full((batch, planets), -3.0),
    )
    alpha_t = torch.full((batch, planets), 5.0)
    beta_t = torch.full((batch, planets), 2.0)
    target_logits = target_logits_t.numpy().astype(np.float32, copy=True)
    launch_logits = launch_logits_t.numpy().astype(np.float32, copy=True)
    alpha = alpha_t.numpy().astype(np.float32, copy=True)
    beta = beta_t.numpy().astype(np.float32, copy=True)

    record_rows = [0, 2]
    dense = rust._core.categorical_beta_actions_from_state(
        rows,
        launch_logits,
        target_logits,
        alpha,
        beta,
        8.0,
        True,
        record_rows,
        False,
    )
    source_indices = torch.nonzero(
        fast.planet_owned_mask & fast.planet_mask,
        as_tuple=False,
    ).numpy()
    compact = rust._core.categorical_beta_actions_from_state_compact_sources(
        rows,
        np.ascontiguousarray(source_indices[:, 0], dtype=np.int64),
        np.ascontiguousarray(source_indices[:, 1], dtype=np.int64),
        np.ascontiguousarray(launch_logits[source_indices[:, 0], source_indices[:, 1]]),
        np.ascontiguousarray(target_logits[source_indices[:, 0], source_indices[:, 1]]),
        np.ascontiguousarray(alpha[source_indices[:, 0], source_indices[:, 1]]),
        np.ascontiguousarray(beta[source_indices[:, 0], source_indices[:, 1]]),
        planets,
        8.0,
        True,
        record_rows,
        False,
    )

    assert compact["actions"] == dense["actions"]
    record_source_mask = (
        fast.planet_owned_mask[record_rows] & fast.planet_mask[record_rows]
    ).numpy()
    assert np.array_equal(
        compact["launch"][record_source_mask],
        dense["launch"][record_source_mask],
    )
    assert np.array_equal(
        compact["target_idx"][record_source_mask],
        dense["target_idx"][record_source_mask],
    )
    assert np.allclose(
        compact["fraction"][record_source_mask],
        dense["fraction"][record_source_mask],
    )
    assert np.allclose(compact["fraction"][~record_source_mask], 0.5)
    assert np.array_equal(compact["target_legal_mask"], dense["target_legal_mask"])


@pytest.mark.skipif(shutil.which("cargo") is None, reason="cargo is not installed")
def test_native_categorical_compact_sources_match_dense_randomized_cases():
    _build_rust_extension()
    from owars.training.rust_env import RustVecEnv

    noop = [[[], [], [], []], [[], [], [], []]]
    cases = ((37, 0), (38, 20), (39, 80))
    for seed, advance_steps in cases:
        rust = RustVecEnv(
            num_envs=2,
            num_players=4,
            episode_steps=500,
            ship_speed=6.0,
            random_seed=seed,
        )
        rust.reset()
        for _ in range(advance_steps):
            rust.step_subset_fast([0, 1], noop)

        rows = [(0, 0), (1, 2), (0, 3), (1, 1)]
        fast, _contexts = rust.policy_batch(rows, device="cpu")
        batch, planets = fast.planet_ids.shape
        rng = np.random.default_rng(seed)
        launch_logits = rng.normal(size=(batch, planets)).astype(np.float32)
        target_logits = rng.normal(size=(batch, planets, planets)).astype(np.float32)
        alpha = rng.uniform(1.1, 6.0, size=(batch, planets)).astype(np.float32)
        beta = rng.uniform(1.1, 6.0, size=(batch, planets)).astype(np.float32)
        record_rows = [0, min(2, batch - 1)]

        dense = rust._core.categorical_beta_actions_from_state(
            rows,
            launch_logits,
            target_logits,
            alpha,
            beta,
            8.0,
            False,
            record_rows,
            False,
        )
        source_indices = torch.nonzero(
            fast.planet_owned_mask & fast.planet_mask,
            as_tuple=False,
        ).numpy()
        compact = rust._core.categorical_beta_actions_from_state_compact_sources(
            rows,
            np.ascontiguousarray(source_indices[:, 0], dtype=np.int64),
            np.ascontiguousarray(source_indices[:, 1], dtype=np.int64),
            np.ascontiguousarray(launch_logits[source_indices[:, 0], source_indices[:, 1]]),
            np.ascontiguousarray(target_logits[source_indices[:, 0], source_indices[:, 1]]),
            np.ascontiguousarray(alpha[source_indices[:, 0], source_indices[:, 1]]),
            np.ascontiguousarray(beta[source_indices[:, 0], source_indices[:, 1]]),
            planets,
            8.0,
            False,
            record_rows,
            False,
        )

        assert compact["actions"] == dense["actions"]
        assert np.array_equal(compact["launch"], dense["launch"])
        assert np.array_equal(compact["target_idx"], dense["target_idx"])
        assert np.allclose(compact["fraction"], dense["fraction"])
        assert np.array_equal(compact["target_legal_mask"], dense["target_legal_mask"])


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

    # Self-play: both seats in each env are live learners (designated seat +
    # LEARNER_NAME opponent), so every seat is recorded → 2 envs × 2 seats.
    assert len(trajs) == 4
    assert all(traj.encoded for traj in trajs)
    assert all(traj.reward for traj in trajs)
    assert all(traj.log_prob for traj in trajs)
    assert any(
        any(
            torch.isfinite(logp).all() and not torch.allclose(logp, torch.zeros_like(logp))
            for logp in traj.log_prob
        )
        for traj in trajs
    )


@pytest.mark.skipif(shutil.which("cargo") is None, reason="cargo is not installed")
def test_rust_fast_rollout_prefetches_learner_and_snapshot_features(monkeypatch):
    _build_rust_extension()
    from types import SimpleNamespace

    from owars.policies.config import OrbitPolicyConfig
    from owars.policies.model import OrbitPolicy
    from owars.training.league import OpponentSlot
    from owars.training.rust_env import RustVecEnv
    from owars.training.vec_rollout import rollout_episodes_batched

    rust = RustVecEnv(
        num_envs=1,
        num_players=2,
        episode_steps=4,
        ship_speed=6.0,
        random_seed=0,
    )
    model = OrbitPolicy(
        OrbitPolicyConfig(
            dim=16,
            ff_dim=32,
            depth=1,
            n_heads=2,
            encoder_backend="destination_conditioned",
        )
    )
    snapshot = SimpleNamespace(
        model=model,
        device="cpu",
        deterministic=True,
        compile_mode=None,
    )
    opponent = OpponentSlot("snapshot:test", agent=snapshot)
    row_counts: list[int] = []
    original_policy_batch_no_context = rust.policy_batch_no_context

    def count_policy_batch_no_context(rows: Any, *args: Any, **kwargs: Any) -> Any:
        row_counts.append(len(rows))
        return original_policy_batch_no_context(rows, *args, **kwargs)

    monkeypatch.setattr(
        rust,
        "policy_batch_no_context",
        count_policy_batch_no_context,
    )

    [traj] = rollout_episodes_batched(
        model,
        rust,
        [[opponent]],
        num_players=2,
        learner_seat=0,
        device="cpu",
        defer_log_prob=True,
        chunk_records=True,
    )

    assert traj.record_refs
    assert row_counts
    assert all(count == 2 for count in row_counts)


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

    calls = {"builtin_actions": 0, "enqueue_builtin_actions": 0}
    original_builtin_actions = rust.builtin_actions
    original_enqueue_builtin_actions = rust.enqueue_builtin_actions

    def count_builtin_actions(*args: Any, **kwargs: Any) -> Any:
        calls["builtin_actions"] += 1
        return original_builtin_actions(*args, **kwargs)

    def count_enqueue_builtin_actions(*args: Any, **kwargs: Any) -> Any:
        calls["enqueue_builtin_actions"] += 1
        return original_enqueue_builtin_actions(*args, **kwargs)

    monkeypatch.setattr(rust, "observation", fail_observation)
    monkeypatch.setattr(rust, "observations", fail_observation)
    monkeypatch.setattr(rust, "builtin_actions", count_builtin_actions)
    monkeypatch.setattr(rust, "enqueue_builtin_actions", count_enqueue_builtin_actions)

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
    assert calls["enqueue_builtin_actions"] > 0
    assert calls["builtin_actions"] == 0


@pytest.mark.skipif(shutil.which("cargo") is None, reason="cargo is not installed")
def test_rust_fast_rollout_behavior_override_disables_pending_learner_enqueue(monkeypatch):
    _build_rust_extension()
    from owars.agents.sniper import sniper_agent
    from owars.policies.config import OrbitPolicyConfig
    from owars.policies.model import OrbitPolicy
    from owars.training.league import OpponentSlot
    from owars.training.rust_env import RustVecEnv
    from owars.training.vec_rollout import rollout_episodes_batched

    rust = RustVecEnv(
        num_envs=1,
        num_players=2,
        episode_steps=4,
        ship_speed=6.0,
        random_seed=0,
    )
    enqueue_flags: list[bool] = []
    original_sample_batch_actions = rust.sample_batch_actions

    def count_sample_batch_actions(*args: Any, **kwargs: Any) -> Any:
        enqueue_flags.append(bool(kwargs.get("enqueue_actions", False)))
        return original_sample_batch_actions(*args, **kwargs)

    seen_behavior_steps: list[int] = []

    def behavior(obs: dict[str, Any]) -> list[list[float | int]]:
        seen_behavior_steps.append(int(obs["step"]))
        return []

    monkeypatch.setattr(rust, "sample_batch_actions", count_sample_batch_actions)

    model = OrbitPolicy(OrbitPolicyConfig(dim=16, ff_dim=32, depth=1, n_heads=2))
    opponent = OpponentSlot("sniper", agent=sniper_agent)
    [traj] = rollout_episodes_batched(
        model,
        rust,
        [[opponent]],
        num_players=2,
        learner_seat=0,
        device="cpu",
        learner_action_agent=behavior,
    )

    assert seen_behavior_steps
    assert enqueue_flags
    assert not any(enqueue_flags)
    assert traj.launch == []
    assert traj.target_idx == []
    assert traj.fraction == []


@pytest.mark.skipif(shutil.which("cargo") is None, reason="cargo is not installed")
def test_rust_fast_rollout_mixes_pending_native_and_python_flat_actions(monkeypatch):
    _build_rust_extension()
    from owars.agents.sniper import sniper_agent
    from owars.policies.config import OrbitPolicyConfig
    from owars.policies.model import OrbitPolicy
    from owars.training.league import LEARNER_NAME, OpponentSlot
    from owars.training.rust_env import RustVecEnv
    from owars.training.vec_rollout import rollout_episodes_batched

    rust = RustVecEnv(
        num_envs=1,
        num_players=4,
        episode_steps=4,
        ship_speed=6.0,
        random_seed=0,
    )
    pending_step_calls: list[tuple[int, int, int, bool]] = []
    original_step_pending = rust.step_subset_pending_actions
    enqueue_builtin_calls = 0
    original_enqueue_builtin = rust.enqueue_builtin_actions

    def count_step_pending(
        indices: list[int],
        env_rows: list[int],
        player_rows: list[int],
        actions: list[Any],
    ) -> Any:
        pending_step_calls.append(
            (len(indices), len(env_rows), len(player_rows), any(isinstance(act, list) for act in actions))
        )
        return original_step_pending(indices, env_rows, player_rows, actions)

    python_steps: list[int] = []

    def python_agent(obs: dict[str, Any]) -> list[list[float | int]]:
        python_steps.append(int(obs["step"]))
        return []

    def count_enqueue_builtin(*args: Any, **kwargs: Any) -> Any:
        nonlocal enqueue_builtin_calls
        enqueue_builtin_calls += 1
        return original_enqueue_builtin(*args, **kwargs)

    monkeypatch.setattr(rust, "step_subset_pending_actions", count_step_pending)
    monkeypatch.setattr(rust, "enqueue_builtin_actions", count_enqueue_builtin)

    model = OrbitPolicy(OrbitPolicyConfig(dim=16, ff_dim=32, depth=1, n_heads=2))
    opponents = [
        OpponentSlot(LEARNER_NAME, agent=None),
        OpponentSlot("sniper", agent=sniper_agent),
        OpponentSlot("python:no-op", agent=python_agent),
    ]
    trajs = rollout_episodes_batched(
        model,
        rust,
        [opponents],
        num_players=4,
        learner_seat=0,
        device="cpu",
    )
    # Seat 1 is a LEARNER_NAME self-play opponent, so it also records; this test
    # inspects the designated learner's trajectory (seat 0).
    traj = next(t for t in trajs if t.learner_seat == 0)

    assert traj.encoded
    assert python_steps
    assert enqueue_builtin_calls > 0
    assert pending_step_calls
    assert any(
        flat_env_rows == flat_player_rows and flat_env_rows > 0 and has_python_action
        for _, flat_env_rows, flat_player_rows, has_python_action in pending_step_calls
    )


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
    assert rust.sample_batch_actions(out, rows, deterministic=True) == sample_batch_actions_context(
        out, contexts, deterministic=True
    )
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
    assert rust.sample_batch_actions(out, rows, deterministic=True) == sample_batch_actions_raw(
        out, raw_observations, deterministic=True
    )
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


@pytest.mark.skipif(shutil.which("cargo") is None, reason="cargo is not installed")
def test_rust_crate_sources_tracks_library_inputs_without_dev_bins(tmp_path: Path):
    from owars.training.rust_env import _rust_crate_sources

    crate = tmp_path / "crate"
    (crate / "src" / "bin").mkdir(parents=True)
    (crate / "src" / "oracle").mkdir()
    (crate / "src" / "protocol" / "bin").mkdir(parents=True)
    for rel in (
        "Cargo.toml",
        "Cargo.lock",
        "build.rs",
        "src/lib.rs",
        "src/core.rs",
        "src/oracle/mod.rs",
        "src/protocol/bin/mod.rs",
        "src/oracle/tests.rs",
        "src/bin/bench.rs",
    ):
        path = crate / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("", encoding="utf-8")

    rel_sources = {path.relative_to(crate).as_posix() for path in _rust_crate_sources(crate)}

    assert {
        "Cargo.toml",
        "Cargo.lock",
        "build.rs",
        "src/lib.rs",
        "src/core.rs",
        "src/oracle/mod.rs",
        "src/protocol/bin/mod.rs",
    } <= rel_sources
    assert "src/oracle/tests.rs" not in rel_sources
    assert "src/bin/bench.rs" not in rel_sources


def _stepped_rust_env(*, num_players: int, seed: int, depth: int):
    """A RustVecEnv advanced `depth` no-op steps so orbiters have rotated and
    (for depth > 50) comets are live — the geometry that stresses the Python
    twins of the Rust legality/materialization path."""
    from owars.training.rust_env import RustVecEnv

    rust = RustVecEnv(
        num_envs=3,
        num_players=num_players,
        episode_steps=500,
        ship_speed=6.0,
        random_seed=seed,
    )
    rust.reset()
    noop = [[[] for _ in range(num_players)] for _ in range(3)]
    for _ in range(depth):
        rust.step_subset_fast(list(range(3)), noop)
    rows = [(e, p) for e in range(3) for p in range(num_players)]
    return rust, rows


@pytest.mark.skipif(shutil.which("cargo") is None, reason="cargo is not installed")
@pytest.mark.parametrize("num_players", [2, 4])
def test_python_legal_mask_matches_rust_state_legal_mask(num_players: int):
    """The submission bundle is pure Python and masks target logits with
    ``_target_legal_mask_from_planets``; training masks with the Rust env's
    ``legal_target_mask_from_state``. They must agree bit-for-bit across
    orbiting/comet geometry and launch fractions, or the served policy sees a
    different legal target set than it was trained against."""
    from owars.policies.sampling import _target_legal_mask_from_planets

    _build_rust_extension()
    for seed in range(4):
        for depth in (40, 120, 300):
            rust, rows = _stepped_rust_env(num_players=num_players, seed=seed, depth=depth)
            fast, _ = rust.policy_batch(rows, device="cpu")
            owned = fast.planet_owned_mask.numpy().astype(bool) & fast.planet_mask.numpy().astype(
                bool
            )
            pmask = fast.planet_mask.numpy().astype(bool)
            ids = fast.planet_ids.numpy()
            obs_list = rust.observations(rows)
            for frac_val in (0.1, 0.5, 0.75, 1.0):
                frac = np.full(fast.planet_mask.shape, frac_val, dtype=np.float32)
                rust_mask = np.asarray(
                    rust._core.legal_target_mask_from_state_active(rows, frac, owned)
                )
                py_mask = np.zeros_like(rust_mask)
                for r, obs in enumerate(obs_list):
                    py_mask[r] = np.asarray(
                        _target_legal_mask_from_planets(
                            frac[r].tolist(),
                            owned[r].tolist(),
                            pmask[r].tolist(),
                            ids[r].tolist(),
                            obs["planets"],
                            obs.get("angular_velocity", 0.0) or 0.0,
                            obs.get("comet_planet_ids", []),
                        ),
                        dtype=bool,
                    )
                assert np.array_equal(py_mask, rust_mask), (
                    f"legal-mask divergence seed={seed} depth={depth} "
                    f"players={num_players} frac={frac_val}: "
                    f"{int((py_mask != rust_mask).sum())} cells differ"
                )


@pytest.mark.skipif(shutil.which("cargo") is None, reason="cargo is not installed")
@pytest.mark.parametrize("num_players", [2, 4])
def test_python_materialize_matches_rust_materialize_actions(num_players: int):
    """``_build_moves_from_lists`` (submission) must produce the same
    ``[planet_id, angle, ships]`` triples as the Rust env's
    ``materialize_actions`` (training) for identical launch/target/fraction
    inputs. frac=0.5 deliberately lands on exact half-ship boundaries, which
    is where Python's banker's rounding used to send one fewer ship than
    Rust's round-half-away-from-zero."""
    from owars.game.observation import parse_observation
    from owars.policies.sampling import _build_moves_from_lists

    _build_rust_extension()
    for seed in range(4):
        for depth in (40, 150, 300):
            rust, rows = _stepped_rust_env(num_players=num_players, seed=seed, depth=depth)
            fast, _ = rust.policy_batch(rows, device="cpu")
            b, p = fast.planet_ids.shape
            owned = fast.planet_owned_mask.numpy().astype(bool) & fast.planet_mask.numpy().astype(
                bool
            )
            pmask = fast.planet_mask.numpy().astype(bool)
            ids = fast.planet_ids.numpy()
            obs_list = rust.observations(rows)
            for frac_val in (0.25, 0.5, 0.75, 1.0):
                frac = np.full((b, p), frac_val, dtype=np.float32)
                launch = owned.astype(np.float32)
                rust_mask = np.asarray(
                    rust._core.legal_target_mask_from_state_active(rows, frac, owned)
                )
                target_idx = np.zeros((b, p), dtype=np.int64)
                for r in range(b):
                    for s in range(p):
                        legal = np.flatnonzero(rust_mask[r, s])
                        if legal.size:
                            target_idx[r, s] = legal[0]
                        else:
                            launch[r, s] = 0.0
                rust_mat = rust._core.materialize_actions(
                    rows, launch, target_idx, frac, owned, pmask, ids, False
                )
                for r, obs in enumerate(obs_list):
                    moves, _ = _build_moves_from_lists(
                        launch[r].tolist(),
                        target_idx[r].tolist(),
                        frac[r].tolist(),
                        owned[r].tolist(),
                        pmask[r].tolist(),
                        ids[r].tolist(),
                        parse_observation(obs),
                    )
                    py = sorted(
                        (int(m.from_planet_id), int(m.num_ships), float(m.angle)) for m in moves
                    )
                    ru = sorted(
                        (int(a[0]), int(a[2]), float(a[1])) for a in rust_mat["actions"][r]
                    )
                    assert [(pl, sh) for (pl, sh, _a) in py] == [
                        (pl, sh) for (pl, sh, _a) in ru
                    ], (
                        f"launched (planet, ships) diverge seed={seed} depth={depth} "
                        f"players={num_players} frac={frac_val} row={r}: py={py} rust={ru}"
                    )
                    for (_pl, _sh, pa), (_rpl, _rsh, ra) in zip(py, ru, strict=True):
                        assert math.isclose(pa, ra, rel_tol=0.0, abs_tol=1e-9)


@pytest.mark.skipif(shutil.which("cargo") is None, reason="cargo is not installed")
@pytest.mark.parametrize("num_players", [2, 4])
def test_python_deterministic_actions_match_rust_native_sampler(num_players: int):
    """End-to-end usage parity: given identical policy logits and state, the
    Python deterministic sampler the submission bundle runs
    (``sample_batch_actions_raw``) must select the same launches as the Rust
    native sampler used during training. Angles may differ only by the
    observation's coordinate-serialization noise."""
    from owars.policies.model import PolicyOutput
    from owars.policies.sampling import sample_batch_actions_raw

    _build_rust_extension()

    def _make_out(fast, gen_seed: int) -> PolicyOutput:
        gen = torch.Generator().manual_seed(gen_seed)
        rb, rp = fast.planet_ids.shape
        return PolicyOutput(
            launch_logits=torch.randn(rb, rp, generator=gen) * 3.0,
            target_logits=torch.randn(rb, rp, rp, generator=gen) * 2.0,
            value=torch.zeros(rb),
            value_logits=torch.zeros(rb, 51),
            planet_owned_mask=fast.planet_owned_mask,
            planet_mask=fast.planet_mask,
            planet_ids=fast.planet_ids,
            action_logit_softcap=8.0,
            fraction_alpha=torch.rand(rb, rp, generator=gen) * 8 + 1,
            fraction_beta=torch.rand(rb, rp, generator=gen) * 8 + 1,
        )

    for seed in range(4):
        for depth in (35, 160, 320):
            rust, rows = _stepped_rust_env(num_players=num_players, seed=seed, depth=depth)
            fast, _ = rust.policy_batch(rows, device="cpu")
            out = _make_out(fast, seed * 1000 + depth)
            rust_actions = rust.sample_batch_actions(out, rows, deterministic=True)
            py_actions = sample_batch_actions_raw(out, rust.observations(rows), deterministic=True)
            for r in range(len(rows)):
                py = sorted((int(a[0]), int(a[2])) for a in py_actions[r])
                ru = sorted((int(a[0]), int(a[2])) for a in rust_actions[r])
                assert py == ru, (
                    f"deterministic launch set diverges seed={seed} depth={depth} "
                    f"players={num_players} row={r}: py={py} rust={ru}"
                )
