"""Tests for the Python exact fleet-destination oracle.

Two layers:

- crafted semantic tests that run without the Rust toolchain and pin the
  simulator-ordering rules (board > sun > pre-move vector order > sweep,
  strict-< hits, unknown comet boundary, horizon);
- differential tests that require cargo and assert bit-exact parity between
  the Python oracle and both Rust bindings (optimized + reference) on
  randomized observations with statics, orbiters, and comets.
"""

from __future__ import annotations

import math
import shutil
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from owars.game.destination_oracle import (
    STATUS_BOARD,
    STATUS_HORIZON,
    STATUS_NONE,
    STATUS_PLANET,
    STATUS_SUN,
    STATUS_UNKNOWN,
    infer_fleet_destinations,
)
from owars.game.observation import parse_observation

ROOT = Path(__file__).resolve().parents[1]
RUST_PY_CRATE = ROOT / "rust" / "owars_env_py"

requires_cargo = pytest.mark.skipif(shutil.which("cargo") is None, reason="cargo is not installed")


def _build_rust_extension() -> None:
    subprocess.run(
        ["cargo", "build", "--release"],
        cwd=RUST_PY_CRATE,
        check=True,
        capture_output=True,
        text=True,
    )


def _obs_dict(
    *,
    planets: list[list[Any]],
    fleets: list[list[Any]],
    step: int = 1,
    angular_velocity: float = 0.0,
    initial_planets: list[list[Any]] | None = None,
    comets: list[dict[str, Any]] | None = None,
    comet_planet_ids: list[int] | None = None,
) -> dict[str, Any]:
    return {
        "player": 0,
        "step": step,
        "planets": [row.copy() for row in planets],
        "fleets": [row.copy() for row in fleets],
        "angular_velocity": angular_velocity,
        "initial_planets": [
            row.copy() for row in (initial_planets if initial_planets is not None else planets)
        ],
        "next_fleet_id": len(fleets),
        "comets": comets or [],
        "comet_planet_ids": comet_planet_ids or [],
    }


def _python_oracle(obs_dict: dict[str, Any], *, ship_speed: float = 6.0):
    return infer_fleet_destinations(
        parse_observation(obs_dict),
        episode_steps=500,
        ship_speed=ship_speed,
    )


def test_python_oracle_static_direct_hit():
    obs = _obs_dict(
        planets=[
            [0, 0, 5.0, 10.0, 1.0, 10, 1],
            [1, 1, 20.0, 10.0, 1.0, 10, 1],
        ],
        fleets=[[0, 0, 15.0, 10.0, 0.0, 0, 1]],
    )
    dest, eta, status = _python_oracle(obs)
    assert status[0] == STATUS_PLANET
    assert dest[0] == 1
    assert eta[0] == 5.0


def test_python_oracle_board_before_sun():
    obs = _obs_dict(
        planets=[
            [0, 0, 95.0, 80.0, 1.0, 10, 1],
            [1, 1, 70.0, 50.0, 3.0, 10, 1],
        ],
        fleets=[[0, 0, 95.0, 50.0, math.pi, 0, 1000]],
    )
    dest, eta, status = _python_oracle(obs, ship_speed=100.0)
    assert status[0] == STATUS_BOARD
    assert dest[0] == -1
    assert eta[0] == 1.0


def test_python_oracle_sun_before_planet():
    obs = _obs_dict(
        planets=[
            [0, 0, 20.0, 80.0, 1.0, 10, 1],
            [1, 1, 70.0, 50.0, 3.0, 10, 1],
        ],
        fleets=[[0, 0, 20.0, 50.0, 0.0, 0, 100]],
    )
    dest, eta, status = _python_oracle(obs, ship_speed=100.0)
    assert status[0] == STATUS_SUN
    assert dest[0] == -1
    assert eta[0] == 1.0


def test_python_oracle_uses_planet_vector_order_for_ties():
    obs = _obs_dict(
        planets=[
            [5, 1, 30.0, 10.0, 2.0, 10, 1],
            [3, 1, 30.0, 10.0, 2.0, 10, 1],
            [0, 0, 10.0, 10.0, 1.0, 10, 1],
        ],
        fleets=[[0, 0, 25.0, 10.0, 0.0, 0, 1]],
    )
    dest, eta, status = _python_oracle(obs)
    assert status[0] == STATUS_PLANET
    assert dest[0] == 0
    assert eta[0] == 4.0


def test_python_oracle_strict_tangent_is_not_hit():
    obs = _obs_dict(
        planets=[
            [0, 0, 90.0, 10.0, 1.0, 10, 1],
            [1, 1, 96.0, 11.0, 1.0, 10, 1],
        ],
        fleets=[[0, 0, 95.0, 10.0, 0.0, 0, 1]],
        step=450,
    )
    dest, eta, status = _python_oracle(obs)
    assert status[0] == STATUS_BOARD
    assert dest[0] == -1
    assert eta[0] == 6.0


def test_python_oracle_moving_sweep_hit():
    obs = _obs_dict(
        planets=[
            [1, 1, 70.0, 50.0, 1.0, 10, 1],
            [0, 0, 20.0, 80.0, 1.0, 10, 1],
        ],
        fleets=[[0, 0, 70.0, 52.0, -math.pi / 2.0, 0, 1]],
        angular_velocity=0.1,
    )
    dest, eta, status = _python_oracle(obs)
    assert status[0] == STATUS_PLANET
    assert dest[0] == 0
    assert eta[0] == 1.0


def test_python_oracle_marks_future_comet_spawn_unknown():
    obs = _obs_dict(
        planets=[
            [0, 0, 5.0, 20.0, 1.0, 10, 1],
            [1, 1, 80.0, 80.0, 1.0, 10, 1],
        ],
        fleets=[[0, 0, 20.0, 20.0, 0.0, 0, 1]],
        step=48,
    )
    dest, eta, status = _python_oracle(obs)
    assert status[0] == STATUS_UNKNOWN
    assert dest[0] == -1
    assert eta[0] == 3.0


def test_python_oracle_zero_horizon_and_done():
    obs = _obs_dict(
        planets=[[0, 0, 5.0, 10.0, 1.0, 10, 1]],
        fleets=[[0, 0, 15.0, 10.0, 0.0, 0, 1]],
        step=499,
    )
    dest, eta, status = _python_oracle(obs)
    assert status[0] == STATUS_HORIZON
    assert dest[0] == -1
    assert eta[0] == 0.0

    dest, eta, status = infer_fleet_destinations(parse_observation(obs), done=True)
    assert status[0] == STATUS_NONE
    assert eta[0] == 0.0


def test_python_oracle_empty_fleets():
    obs = _obs_dict(planets=[[0, 0, 5.0, 10.0, 1.0, 10, 1]], fleets=[])
    dest, eta, status = _python_oracle(obs)
    assert dest.shape == eta.shape == status.shape == (0,)


def test_python_oracle_comet_negative_path_index_dwells_on_first_point():
    # path_index = -3: the simulator clamps the index, so the comet jumps to
    # path[0] on turn 1 (sweeping its chord) and dwells there until the index
    # catches up at turn 4.
    path = [[35.0 + 5.0 * i, 30.0] for i in range(6)]
    obs = _obs_dict(
        planets=[
            [0, 0, 90.0, 90.0, 1.0, 10, 1],
            [7, -1, 30.0, 30.0, 1.0, 5, 1],
        ],
        fleets=[
            [0, 0, 33.0, 31.5, -math.pi / 2.0, -1, 1],
            [1, 1, 35.0, 25.5, math.pi / 2.0, -1, 1],
        ],
        comets=[{"planet_ids": [7], "paths": [path], "path_index": -3}],
        comet_planet_ids=[7],
    )
    dest, eta, status = _python_oracle(obs)
    assert status[0] == STATUS_PLANET and dest[0] == 1 and eta[0] == 1.0
    assert status[1] == STATUS_PLANET and dest[1] == 1 and eta[1] == 4.0


def test_python_oracle_off_board_fleet_is_removed_at_turn_one():
    obs = _obs_dict(
        planets=[[0, 0, 90.0, 90.0, 1.0, 10, 1]],
        fleets=[
            [0, 0, -5.0, 50.0, 0.0, -1, 1],
            [1, 1, 104.0, 20.0, math.pi, -1, 1],
        ],
    )
    dest, eta, status = _python_oracle(obs)
    assert (status == STATUS_BOARD).all()
    assert (eta == 1.0).all()


def test_python_oracle_nonfinite_fleet_angle_is_skipped():
    obs = _obs_dict(
        planets=[[0, 0, 90.0, 90.0, 1.0, 10, 1]],
        fleets=[[0, 0, 50.0, 20.0, math.inf, -1, 1]],
    )
    dest, eta, status = _python_oracle(obs)
    assert status[0] == STATUS_NONE
    assert eta[0] == 0.0


def test_python_oracle_survivor_keeps_horizon():
    # A slow fleet aimed into empty space late in the game outlives the
    # lookahead without hitting anything.
    obs = _obs_dict(
        planets=[[0, 0, 5.0, 90.0, 1.0, 10, 1]],
        fleets=[[0, 0, 60.0, 90.0, 0.0, 0, 1]],
        step=460,
    )
    dest, eta, status = _python_oracle(obs)
    assert status[0] == STATUS_HORIZON
    assert dest[0] == -1
    assert eta[0] == 39.0


def _random_obs_dict(seed: int, step: int) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    planets: list[list[Any]] = []
    pid = 0
    for _ in range(int(rng.integers(3, 7))):
        production = int(rng.integers(1, 6))
        planets.append(
            [
                pid,
                int(rng.integers(-1, 2)),
                float(rng.uniform(3.0, 97.0)),
                float(rng.uniform(3.0, 97.0)),
                1.0 + math.log(production),
                int(rng.integers(1, 200)),
                production,
            ]
        )
        pid += 1
    for _ in range(int(rng.integers(2, 6))):
        production = int(rng.integers(1, 6))
        radius = 1.0 + math.log(production)
        orb_r = float(rng.uniform(12.0, 48.0 - radius))
        theta = float(rng.uniform(-math.pi, math.pi))
        planets.append(
            [
                pid,
                int(rng.integers(-1, 2)),
                50.0 + orb_r * math.cos(theta),
                50.0 + orb_r * math.sin(theta),
                radius,
                int(rng.integers(1, 200)),
                production,
            ]
        )
        pid += 1
    initial_planets = [row.copy() for row in planets]

    def random_path(length: int) -> list[list[float]]:
        return [
            [float(rng.uniform(-3.0, 103.0)), float(rng.uniform(-3.0, 103.0))]
            for _ in range(length)
        ]

    comet_ids: list[int] = []
    comet_rows: list[list[Any]] = []
    paths: list[list[list[float]]] = []
    n_comets = int(rng.integers(2, 5))
    # Negative indices exercise the simulator's max(0) clamp (dwell on path[0]).
    path_index = int(rng.choice([-3, -1, 0, 2, 5]))
    for i in range(n_comets):
        length = int(rng.integers(4, 18))
        path = random_path(length)
        comet_ids.append(100 + i)
        paths.append(path)
        if 0 <= path_index < length:
            x, y = path[path_index]
            if i == 0:
                # Stale stored position: turn-1 motion starts from here, not
                # from the recomputed path point.
                x += float(rng.uniform(-0.6, 0.6))
        else:
            x, y = -99.0, -99.0
        comet_rows.append([100 + i, -1, x, y, 1.0, 5, 1])
    # One comet id intentionally has no planet row: the simulator tolerates
    # this and the oracles must agree on skipping it.
    dropped = int(rng.integers(0, n_comets))
    planets.extend(row for i, row in enumerate(comet_rows) if i != dropped)

    # Second group with paths shorter than planet_ids: id 201 has no path
    # slot and must expire without sweeping on its first move.
    group_b_index = int(rng.choice([-1, 0, 3]))
    group_b_path = random_path(4)
    group_b = {"planet_ids": [200, 201], "paths": [group_b_path], "path_index": group_b_index}
    bx, by = (
        group_b_path[group_b_index] if 0 <= group_b_index < 4 else (-99.0, -99.0)
    )
    planets.append([200, -1, bx, by, 1.0, 5, 1])
    planets.append([201, -1, -99.0, -99.0, 1.0, 5, 1])

    fleets = [
        [
            i,
            i % 2,
            float(rng.uniform(1.0, 99.0)),
            float(rng.uniform(1.0, 99.0)),
            float(rng.uniform(-math.pi, math.pi)),
            -1,
            int(1 + rng.integers(0, 900)),
        ]
        for i in range(int(rng.integers(20, 60)))
    ]
    for x, y, angle in [(-5.0, 50.0, 0.0), (104.0, 20.0, math.pi), (50.0, -3.0, 1.2)]:
        fleets.append([len(fleets), 0, x, y, angle, -1, 10])
    return _obs_dict(
        planets=planets,
        fleets=fleets,
        step=step,
        angular_velocity=float(rng.uniform(0.025, 0.05)),
        initial_planets=initial_planets,
        comets=[
            {"planet_ids": comet_ids, "paths": paths, "path_index": path_index},
            group_b,
        ],
        comet_planet_ids=comet_ids + [200, 201],
    )


def _rust_core(obs: dict[str, Any]):
    from owars.training.rust_env import RustVecEnv

    rust = RustVecEnv(
        num_envs=1,
        num_players=2,
        episode_steps=500,
        ship_speed=6.0,
        random_seed=0,
    )
    rust._core.load_observation(0, obs)
    return rust._core


@requires_cargo
@pytest.mark.parametrize("step", [1, 48, 120, 460, 493])
@pytest.mark.parametrize("seed", range(5))
def test_python_oracle_matches_rust_bindings_bit_exactly(seed: int, step: int):
    _build_rust_extension()
    obs = _random_obs_dict(seed, step)
    core = _rust_core(obs)
    fast = core.fleet_destination_oracle([(0, 0)])
    reference = core.fleet_destination_oracle_reference([(0, 0)])
    dest, eta, status = _python_oracle(obs)

    n = len(obs["fleets"])
    for name, rust_out in (("fast", fast), ("reference", reference)):
        np.testing.assert_array_equal(rust_out["dest_idx"][0, :n], dest, err_msg=name)
        np.testing.assert_array_equal(rust_out["eta"][0, :n], eta, err_msg=name)
        np.testing.assert_array_equal(rust_out["status"][0, :n], status, err_msg=name)
        assert (rust_out["dest_idx"][0, n:] == -1).all()
        assert (rust_out["eta"][0, n:] == 0.0).all()
        assert (rust_out["status"][0, n:] == STATUS_NONE).all()


@requires_cargo
@pytest.mark.parametrize("episode_steps", [60, 120])
@pytest.mark.parametrize("seed", range(6))
def test_observation_episode_steps_threads_into_oracle(seed: int, episode_steps: int):
    # Regression for the v18 oracle-forecast sniper: the destination-oracle
    # lookahead is `episode_steps - 1 - step`, so the Python oracle must learn the
    # true episode length from the observation rather than assuming the default
    # 500. Deep into a short episode the horizon is tiny; if the obs did not carry
    # `episode_steps`, the Python oracle (horizon ~500) would diverge from the Rust
    # oracle (true short horizon) for any fleet whose impact lies beyond it.
    _build_rust_extension()
    from owars.training.rust_env import RustVecEnv

    step = episode_steps - 12  # deep enough that the horizon truncates far fleets
    obs = _random_obs_dict(seed, step)
    obs["episode_steps"] = episode_steps

    # The observation length flows through parse_observation.
    parsed = parse_observation(obs)
    assert parsed.episode_steps == episode_steps

    rust = RustVecEnv(
        num_envs=1,
        num_players=2,
        episode_steps=episode_steps,
        ship_speed=6.0,
        random_seed=0,
    )
    rust._core.load_observation(0, obs)
    rust_out = rust._core.fleet_destination_oracle([(0, 0)])

    n = len(obs["fleets"])
    dest, eta, status = infer_fleet_destinations(parsed, episode_steps=parsed.episode_steps)
    np.testing.assert_array_equal(rust_out["dest_idx"][0, :n], dest)
    np.testing.assert_array_equal(rust_out["eta"][0, :n], eta)
    np.testing.assert_array_equal(rust_out["status"][0, :n], status)


@requires_cargo
def test_python_oracle_matches_rust_with_empty_initial_planets():
    # The binding falls back to current planets when initial_planets is
    # empty; the Python oracle must rotate the same planets.
    _build_rust_extension()
    obs = _random_obs_dict(11, 120)
    obs["initial_planets"] = []
    core = _rust_core(obs)
    reference = core.fleet_destination_oracle_reference([(0, 0)])
    dest, eta, status = _python_oracle(obs)
    n = len(obs["fleets"])
    np.testing.assert_array_equal(reference["dest_idx"][0, :n], dest)
    np.testing.assert_array_equal(reference["eta"][0, :n], eta)
    np.testing.assert_array_equal(reference["status"][0, :n], status)


@requires_cargo
def test_rust_oracle_cache_invalidated_after_step():
    # The per-env cache has no staleness key; its whole safety story is that
    # every state mutation clears it. Step the env and require the oracle to
    # match a fresh Python run on the post-step observation.
    _build_rust_extension()
    from owars.training.rust_env import RustVecEnv

    rust = RustVecEnv(
        num_envs=1,
        num_players=2,
        episode_steps=500,
        ship_speed=6.0,
        random_seed=3,
    )
    rust.reset()
    rust._core.step_subset_fast([0], [[[], []]])  # initialize the board
    start_obs = rust.observation(0, 0)
    mine = [p for p in start_obs["planets"] if int(p[1]) == 0]
    launches = [[int(p[0]), 0.5 + 0.3 * i, max(1, int(p[5]) // 2)] for i, p in enumerate(mine)]
    rust._core.step_subset_fast([0], [[launches, []]])
    stale_check = rust._core.fleet_destination_oracle([(0, 0)])  # populate cache
    rust._core.step_subset_fast([0], [[[], []]])
    after = rust._core.fleet_destination_oracle([(0, 0)])
    post_obs = rust.observation(0, 0)
    dest, eta, status = _python_oracle(post_obs)
    n = len(post_obs["fleets"])
    assert n > 0
    np.testing.assert_array_equal(after["dest_idx"][0, :n], dest)
    np.testing.assert_array_equal(after["eta"][0, :n], eta)
    np.testing.assert_array_equal(after["status"][0, :n], status)
    # And the pre-step snapshot must genuinely differ in eta/status content
    # from the post-step one unless the board happens to be static.
    assert stale_check["eta"].shape == after["eta"].shape


@requires_cargo
def test_rust_oracle_duplicate_rows_share_cached_result():
    _build_rust_extension()
    obs = _random_obs_dict(7, 120)
    core = _rust_core(obs)
    first = core.fleet_destination_oracle([(0, 0), (0, 1)])
    second = core.fleet_destination_oracle([(0, 1)])
    np.testing.assert_array_equal(first["dest_idx"][0], first["dest_idx"][1])
    np.testing.assert_array_equal(first["eta"][0], first["eta"][1])
    np.testing.assert_array_equal(first["status"][0], first["status"][1])
    np.testing.assert_array_equal(second["dest_idx"][0], first["dest_idx"][0])
    np.testing.assert_array_equal(second["eta"][0], first["eta"][0])
    np.testing.assert_array_equal(second["status"][0], first["status"][0])
