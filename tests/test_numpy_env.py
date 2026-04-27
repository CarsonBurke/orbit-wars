from __future__ import annotations

import math
from typing import Any

import numpy as np

from owars.training.numpy_env import NumpyOrbitWarsEnv, NumpyVecEnv


def _official_env(num_players: int = 2, episode_steps: int = 120):
    from kaggle_environments import make

    return make(
        "orbit_wars",
        configuration={"episodeSteps": episode_steps, "shipSpeed": 6.0},
        debug=True,
    )


def _obs(state: Any, seat: int = 0) -> Any:
    return state[seat]["observation"]


def _simple_actions(obs: Any, num_players: int) -> list[list[list[float | int]]]:
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


def _assert_rows_close(actual: list[list[Any]], expected: list[list[Any]], int_cols: set[int]) -> None:
    assert len(actual) == len(expected)
    for a, e in zip(actual, expected, strict=True):
        assert len(a) == len(e)
        for idx, (av, ev) in enumerate(zip(a, e, strict=True)):
            if idx in int_cols:
                assert int(av) == int(ev)
            else:
                assert math.isclose(float(av), float(ev), rel_tol=0.0, abs_tol=1e-9)


def _assert_obs_close(actual: Any, expected: Any) -> None:
    assert int(actual["step"]) == int(expected["step"])
    assert float(actual["angular_velocity"]) == float(expected["angular_velocity"])
    assert int(actual["next_fleet_id"]) == int(expected["next_fleet_id"])
    assert list(actual["comet_planet_ids"]) == list(expected["comet_planet_ids"])
    _assert_rows_close(actual["planets"], expected["planets"], {0, 1, 5, 6})
    _assert_rows_close(actual["initial_planets"], expected["initial_planets"], {0, 1, 5, 6})
    _assert_rows_close(actual["fleets"], expected["fleets"], {0, 1, 5, 6})
    assert len(actual["comets"]) == len(expected["comets"])
    for a_group, e_group in zip(actual["comets"], expected["comets"], strict=True):
        assert list(a_group["planet_ids"]) == list(e_group["planet_ids"])
        assert int(a_group["path_index"]) == int(e_group["path_index"])
        assert len(a_group["paths"]) == len(e_group["paths"])
        for a_path, e_path in zip(a_group["paths"], e_group["paths"], strict=True):
            assert np.asarray(a_path, dtype=np.float64).shape == np.asarray(e_path, dtype=np.float64).shape
            assert np.allclose(
                np.asarray(a_path, dtype=np.float64),
                np.asarray(e_path, dtype=np.float64),
                rtol=0.0,
                atol=1e-12,
            )


def test_numpy_env_reset_mirrors_kaggle_empty_first_observation():
    env = NumpyOrbitWarsEnv(num_players=2, episode_steps=20)
    state = env.reset()
    assert state[0]["observation"]["step"] == 0
    assert state[0]["observation"]["planets"] == []
    assert state[0]["observation"]["fleets"] == []

    state = env.step([[], []])
    obs = state[0]["observation"]
    assert obs["step"] == 1
    assert len(obs["planets"]) >= 20
    assert obs["fleets"] == []
    assert obs["player"] == 0
    assert state[1]["observation"]["player"] == 1
    assert all(int(p[1]) == -1 for p in obs["initial_planets"])


def test_numpy_env_matches_official_from_loaded_observation_for_noops_before_spawn():
    off = _official_env(episode_steps=80)
    off_state = off.reset(num_agents=2)
    off_state = off.step([[], []])

    fast = NumpyOrbitWarsEnv.from_observation(
        _obs(off_state),
        num_players=2,
        episode_steps=80,
        ship_speed=6.0,
    )
    for _ in range(20):
        off_state = off.step([[], []])
        fast_state = fast.step([[], []])
        _assert_obs_close(fast_state[0]["observation"], _obs(off_state))
        assert fast.done == off.done


def test_numpy_env_matches_official_from_loaded_observation_with_launches_before_spawn():
    off = _official_env(episode_steps=90)
    off_state = off.reset(num_agents=2)
    off_state = off.step([[], []])
    fast = NumpyOrbitWarsEnv.from_observation(
        _obs(off_state),
        num_players=2,
        episode_steps=90,
        ship_speed=6.0,
    )

    for _ in range(24):
        actions = _simple_actions(_obs(off_state), 2)
        off_state = off.step(actions)
        fast_state = fast.step(actions)
        _assert_obs_close(fast_state[0]["observation"], _obs(off_state))
        assert [s["reward"] for s in fast_state] == [int(s.reward or 0) for s in off.steps[-1]]
        if off.done:
            break


def test_numpy_env_matches_official_existing_comet_motion():
    off = _official_env(episode_steps=320)
    off_state = off.reset(num_agents=2)
    while not off.done and not _obs(off_state)["comets"]:
        off_state = off.step([[], []])
    assert _obs(off_state)["comets"]

    fast = NumpyOrbitWarsEnv.from_observation(
        _obs(off_state),
        num_players=2,
        episode_steps=320,
        ship_speed=6.0,
    )
    for _ in range(5):
        off_state = off.step([[], []])
        fast_state = fast.step([[], []])
        _assert_obs_close(fast_state[0]["observation"], _obs(off_state))


def test_numpy_vec_env_uses_vecenv_subset_protocol():
    vec = NumpyVecEnv(num_envs=3, num_players=2, episode_steps=20, ship_speed=6.0)
    states = vec.reset()
    assert len(states) == 3
    results = vec.step_subset([0, 2], [[[], []], [[], []]])
    assert sorted(results) == [0, 2]
    assert results[0][0][0]["observation"]["step"] == 1
    assert results[2][0][0]["observation"]["step"] == 1


def test_numpy_vec_env_matches_scalar_fallback_with_launches():
    num_envs = 3
    scalar = [
        NumpyOrbitWarsEnv(
            num_players=2,
            episode_steps=120,
            ship_speed=6.0,
            random_seed=i,
        )
        for i in range(num_envs)
    ]
    scalar_states = [env.reset() for env in scalar]
    vec = NumpyVecEnv(
        num_envs=num_envs,
        num_players=2,
        episode_steps=120,
        ship_speed=6.0,
        random_seed=0,
    )
    vec_states = vec.reset()
    for env_idx in range(num_envs):
        _assert_obs_close(
            vec_states[env_idx][0]["observation"],
            scalar_states[env_idx][0]["observation"],
        )

    for _ in range(40):
        actions = []
        for env_idx, env in enumerate(scalar):
            action = _simple_actions(scalar_states[env_idx][0]["observation"], 2)
            actions.append(action)
            scalar_states[env_idx] = env.step(action)
        results = vec.step_subset(list(range(num_envs)), actions)
        for env_idx in range(num_envs):
            vec_states[env_idx] = results[env_idx][0]
            _assert_obs_close(
                vec_states[env_idx][0]["observation"],
                scalar_states[env_idx][0]["observation"],
            )
            assert results[env_idx][1] == scalar[env_idx].done


def test_numpy_vec_env_matches_scalar_subset_stepping():
    scalar = [
        NumpyOrbitWarsEnv(
            num_players=2,
            episode_steps=80,
            ship_speed=6.0,
            random_seed=i,
        )
        for i in range(3)
    ]
    scalar_states = [env.reset() for env in scalar]
    vec = NumpyVecEnv(
        num_envs=3,
        num_players=2,
        episode_steps=80,
        ship_speed=6.0,
        random_seed=0,
    )
    vec_states = vec.reset()

    active = [0, 2]
    actions = []
    for env_idx in active:
        action = _simple_actions(scalar_states[env_idx][0]["observation"], 2)
        actions.append(action)
        scalar_states[env_idx] = scalar[env_idx].step(action)
    results = vec.step_subset(active, actions)
    for env_idx in active:
        vec_states[env_idx] = results[env_idx][0]
        _assert_obs_close(
            vec_states[env_idx][0]["observation"],
            scalar_states[env_idx][0]["observation"],
        )
    assert (
        vec_states[1][0]["observation"]["step"]
        == scalar_states[1][0]["observation"]["step"]
        == 0
    )

    all_actions = []
    for env_idx, env in enumerate(scalar):
        action = _simple_actions(scalar_states[env_idx][0]["observation"], 2)
        all_actions.append(action)
        scalar_states[env_idx] = env.step(action)
    results = vec.step_subset([0, 1, 2], all_actions)
    for env_idx in range(3):
        vec_states[env_idx] = results[env_idx][0]
        _assert_obs_close(
            vec_states[env_idx][0]["observation"],
            scalar_states[env_idx][0]["observation"],
        )


def test_numpy_vec_env_matches_scalar_4p_noops():
    scalar = [
        NumpyOrbitWarsEnv(
            num_players=4,
            episode_steps=60,
            ship_speed=6.0,
            random_seed=i,
        )
        for i in range(2)
    ]
    scalar_states = [env.reset() for env in scalar]
    vec = NumpyVecEnv(
        num_envs=2,
        num_players=4,
        episode_steps=60,
        ship_speed=6.0,
        random_seed=0,
    )
    vec_states = vec.reset()
    for _ in range(20):
        actions = [[[], [], [], []] for _ in scalar]
        for env_idx, env in enumerate(scalar):
            scalar_states[env_idx] = env.step(actions[env_idx])
        results = vec.step_subset([0, 1], actions)
        for env_idx in range(2):
            vec_states[env_idx] = results[env_idx][0]
            _assert_obs_close(
                vec_states[env_idx][0]["observation"],
                scalar_states[env_idx][0]["observation"],
            )
