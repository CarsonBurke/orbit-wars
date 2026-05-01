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
    feature_mask = rust._core.legal_target_mask(
        rows,
        frac,
        fast.planet_owned_mask.numpy(),
        fast.planet_mask.numpy(),
        fast.planet_ids.numpy(),
    )

    assert state_mask.any()
    assert np.array_equal(state_mask, feature_mask)


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
    assert np.array_equal(
        state_materialized["materialized"],
        feature_materialized["materialized"],
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
        fraction_alpha=torch.full((b, p), 20.0),
        fraction_beta=torch.full((b, p), 2.0),
        value=torch.zeros(b),
        value_logits=torch.zeros(b, 51),
        planet_owned_mask=fast.planet_owned_mask,
        planet_mask=fast.planet_mask,
        planet_ids=fast.planet_ids,
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
    assert torch.equal(torch.isfinite(got_records.target_logits), torch.isfinite(expected_records.target_logits))


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
