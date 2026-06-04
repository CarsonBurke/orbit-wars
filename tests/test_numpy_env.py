from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch

from owars.policies.features import encode_raw_observations
from owars.policies.model import PolicyOutput
from owars.policies.sampling import (
    sample_batch_actions_context,
    sample_batch_actions_raw,
)
from owars.training.numpy_env import NumpyOrbitWarsEnv, NumpyVecEnv
from owars.training.sharded_numpy_env import ShardedNumpyVecEnv


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


def test_numpy_vec_fast_policy_batch_matches_raw_observations():
    vec = NumpyVecEnv(
        num_envs=1,
        num_players=2,
        episode_steps=80,
        ship_speed=6.0,
        random_seed=0,
    )
    states = vec.reset()
    states[0] = vec.step_subset([0], [[[], []]])[0][0]
    raw = [states[0][0]["observation"], states[0][1]["observation"]]

    fast, contexts = vec.policy_batch([(0, 0), (0, 1)], device="cpu")
    expected = encode_raw_observations(raw, device="cpu")

    assert torch.allclose(fast.planet_feats, expected.planet_feats)
    assert torch.equal(fast.planet_mask, expected.planet_mask)
    assert torch.equal(fast.planet_owned_mask, expected.planet_owned_mask)
    assert torch.equal(fast.planet_ids, expected.planet_ids)
    assert torch.allclose(fast.fleet_feats, expected.fleet_feats)
    assert torch.equal(fast.fleet_mask, expected.fleet_mask)
    assert len(contexts) == 2

    b, p = fast.planet_ids.shape
    launch_logits = torch.full((b, p), -100.0)
    launch_logits[:, 0] = 100.0
    logits = torch.full((b, p, p), -100.0)
    logits[:, 0, :] = -100.0
    logits[:, 0, 1] = 100.0
    out = PolicyOutput(
        launch_logits=launch_logits,
        target_logits=logits,
        value=torch.zeros(b),
        value_logits=torch.zeros(b, 51),
        planet_owned_mask=fast.planet_owned_mask,
        planet_mask=fast.planet_mask,
        planet_ids=fast.planet_ids,
        fraction_alpha=torch.full((b, p), 20.0),
        fraction_beta=torch.full((b, p), 2.0),
    )
    assert sample_batch_actions_context(out, contexts, deterministic=True) == (
        sample_batch_actions_raw(out, raw, deterministic=True)
    )


def test_numpy_vec_fleet_target_metadata_matches_raw_observation_features():
    base_obs = {
        "player": 0,
        "step": 1,
        "planets": [
            [0, 0, 10.0, 10.0, 1.0, 50, 1],
            [1, 1, 90.0, 90.0, 1.0, 50, 1],
        ],
        "fleets": [],
        "angular_velocity": 0.0,
        "initial_planets": [
            [0, 0, 10.0, 10.0, 1.0, 50, 1],
            [1, 1, 90.0, 90.0, 1.0, 50, 1],
        ],
        "next_fleet_id": 0,
        "comets": [],
        "comet_planet_ids": [],
    }
    env = NumpyOrbitWarsEnv(num_players=2, episode_steps=20, ship_speed=6.0)
    env.load_observation(base_obs)
    vec = NumpyVecEnv(num_envs=1, num_players=2, episode_steps=20, ship_speed=6.0)
    vec.reset()
    vec._store_env(0, env)

    action = [[[0, 0.0, 10, 1, 3.0, 90.0, 90.0]], []]
    state = vec.step_subset([0], [action])[0][0]
    obs = state[0]["observation"]
    enemy_obs = state[1]["observation"]

    assert all(len(fleet) == 7 for fleet in obs["fleets"])
    assert obs["fleet_targets"] == {"0": [1, 2.0, 90.0, 90.0]}
    assert enemy_obs["fleet_targets"] == {}

    fast, _ = vec.policy_batch([(0, 0), (0, 1)], device="cpu")
    expected = encode_raw_observations([obs, enemy_obs], device="cpu")
    assert torch.allclose(fast.fleet_feats, expected.fleet_feats)
    row = fast.fleet_feats[0, 0].tolist()
    assert math.isclose(row[9], 1 / 128.0, abs_tol=1e-6)
    assert math.isclose(row[10], 2 / 500.0, abs_tol=1e-6)
    assert row[13] == 1.0
    assert fast.fleet_feats[1, 0, 9:14].tolist() == [0.0, 0.0, 0.0, 0.0, 0.0]

    state = vec.step_subset([0], [[[], []]])[0][0]
    assert state[0]["observation"]["fleet_targets"] == {"0": [1, 1.0, 90.0, 90.0]}
    state = vec.step_subset([0], [[[], []]])[0][0]
    assert state[0]["observation"]["fleet_targets"] == {}


def test_numpy_env_load_observation_ignores_non_official_fleet_columns():
    obs = {
        "player": 0,
        "step": 1,
        "planets": [
            [0, 0, 10.0, 10.0, 1.0, 50, 1],
            [1, 1, 90.0, 90.0, 1.0, 50, 1],
        ],
        "fleets": [[0, 0, 30.0, 30.0, 0.5, 0, 10, 1, 5.0, 90.0, 90.0]],
        "angular_velocity": 0.0,
        "initial_planets": [],
        "next_fleet_id": 1,
        "comets": [],
        "comet_planet_ids": [],
    }
    env = NumpyOrbitWarsEnv(num_players=2, episode_steps=20, ship_speed=6.0)
    env.load_observation(obs)
    visible = env._observation(0, env._observation_base())

    assert visible["fleets"] == [[0, 0, 30.0, 30.0, 0.5, 0, 10]]
    assert visible["fleet_targets"] == {}


def test_numpy_vec_fast_step_matches_materialized_step():
    normal = NumpyVecEnv(
        num_envs=2,
        num_players=2,
        episode_steps=80,
        ship_speed=6.0,
        random_seed=0,
    )
    fast = NumpyVecEnv(
        num_envs=2,
        num_players=2,
        episode_steps=80,
        ship_speed=6.0,
        random_seed=0,
    )
    normal_states = normal.reset()
    fast.reset()

    for _ in range(10):
        actions = [
            _simple_actions(normal_states[env_idx][0]["observation"], 2)
            for env_idx in range(2)
        ]
        normal_results = normal.step_subset([0, 1], actions)
        fast_results = fast.step_subset_fast([0, 1], actions)
        for env_idx in range(2):
            normal_states[env_idx] = normal_results[env_idx][0]
            assert fast_results[env_idx][0] is None or fast_results[env_idx][1]
            for seat in range(2):
                _assert_obs_close(
                    fast.observation(env_idx, seat),
                    normal_states[env_idx][seat]["observation"],
                )
            assert fast_results[env_idx][1] == normal_results[env_idx][1]


def test_numpy_vec_reset_subset_advances_seed():
    vec = NumpyVecEnv(
        num_envs=1,
        num_players=2,
        episode_steps=80,
        ship_speed=6.0,
        random_seed=0,
    )
    vec.reset()
    vec.step_subset_fast([0], [[[], []]])
    first = vec.observation(0, 0)

    vec.reset_subset([0])
    vec.step_subset_fast([0], [[[], []]])
    second = vec.observation(0, 0)

    assert first["planets"] != second["planets"]


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


def test_sharded_numpy_vec_env_matches_scalar_subset_stepping():
    scalar = [
        NumpyOrbitWarsEnv(
            num_players=2,
            episode_steps=50,
            ship_speed=6.0,
            random_seed=i,
        )
        for i in range(4)
    ]
    scalar_states = [env.reset() for env in scalar]
    with ShardedNumpyVecEnv(
        num_envs=4,
        num_players=2,
        episode_steps=50,
        ship_speed=6.0,
        random_seed=0,
        num_workers=2,
    ) as vec:
        vec_states = vec.reset()

        active = [0, 3]
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
        results = vec.step_subset([0, 1, 2, 3], all_actions)
        for env_idx in range(4):
            vec_states[env_idx] = results[env_idx][0]
            _assert_obs_close(
                vec_states[env_idx][0]["observation"],
                scalar_states[env_idx][0]["observation"],
            )


def test_sharded_numpy_vec_env_supports_fast_rollout_interface():
    baseline = NumpyVecEnv(
        num_envs=3,
        num_players=2,
        episode_steps=30,
        ship_speed=6.0,
        random_seed=0,
    )
    with ShardedNumpyVecEnv(
        num_envs=3,
        num_players=2,
        episode_steps=30,
        ship_speed=6.0,
        random_seed=0,
        num_workers=2,
    ) as vec, baseline:
        baseline_states = baseline.reset()
        states = vec.reset()

        assert vec.fast_rollout is True
        obs = vec.observation(0, 1)
        assert obs["player"] == 1
        assert obs == states[0][1]["observation"]
        assert vec.observations([(2, 1), (0, 0)]) == [
            states[2][1]["observation"],
            states[0][0]["observation"],
        ]
        _assert_obs_close(states[1][0]["observation"], baseline_states[1][0]["observation"])

        rows = [(2, 1), (0, 0), (1, 1)]
        encoded, contexts = vec.policy_batch(rows, device="cuda")
        expected, expected_contexts = baseline.policy_batch(rows, device="cpu")
        assert encoded.planet_feats.device.type == "cpu"
        assert encoded.planet_feats.shape[0] == len(rows)
        assert encoded.fleet_feats.shape[0] == len(rows)
        assert len(contexts) == len(rows)
        torch.testing.assert_close(encoded.planet_feats, expected.planet_feats)
        torch.testing.assert_close(encoded.planet_mask, expected.planet_mask)
        torch.testing.assert_close(encoded.planet_owned_mask, expected.planet_owned_mask)
        torch.testing.assert_close(encoded.planet_ids, expected.planet_ids)
        torch.testing.assert_close(encoded.planet_garrison, expected.planet_garrison)
        torch.testing.assert_close(encoded.fleet_feats, expected.fleet_feats)
        torch.testing.assert_close(encoded.fleet_mask, expected.fleet_mask)
        for actual, want in zip(contexts, expected_contexts, strict=True):
            assert actual.angular_velocity == want.angular_velocity
            assert list(actual.comet_planet_ids) == list(want.comet_planet_ids)
            assert np.allclose(actual.planets, want.planets, rtol=0.0, atol=0.0)

        results = vec.step_subset_fast([0, 2], [[[], []], [[], []]])
        assert set(results) == {0, 2}
        for state, done, final in results.values():
            assert done is False
            assert final is None
            assert state is None


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
