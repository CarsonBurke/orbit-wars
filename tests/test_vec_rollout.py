from __future__ import annotations

import pytest
import torch

from owars.policies.features import EncodedObs
from owars.agents.learned import _FleetTargetTracker
from owars.policies.config import OrbitPolicyConfig
from owars.policies.model import OrbitPolicy
from owars.training.league import LEARNER_NAME, OpponentSlot
from owars.training.numpy_env import NumpyVecEnv
from owars.training.sharded_numpy_env import ShardedNumpyVecEnv
from owars.training.vec_env import (
    _annotate_state_with_fleet_targets,
    _record_action_sidecars,
    _strip_action_sidecars,
)
from owars.training.vec_rollout import (
    _normalize_learner_seats,
    _obs_reward_potential,
    _resolve_seat_agents,
    _trim_fleets_for_forward,
    _state_reward_potential,
    alternating_learner_seats,
    rollout_episodes_batched,
)


def test_alternating_learner_seats_by_env():
    assert alternating_learner_seats(6, 2) == [0, 1, 0, 1, 0, 1]
    assert alternating_learner_seats(6, 2, offset=1) == [1, 0, 1, 0, 1, 0]
    assert alternating_learner_seats(7, 4) == [0, 1, 2, 3, 0, 1, 2]


def test_resolve_seat_agents_uses_per_env_learner_seat():
    snapshot = OpponentSlot("frozen:a", agent=lambda _obs: [])
    self_play = OpponentSlot(LEARNER_NAME, agent=None)
    opponents_per_env = [
        [snapshot],
        [snapshot],
        [self_play],
    ]
    seats = _resolve_seat_agents(opponents_per_env, num_players=2, learner_seats=[0, 1, 0])

    assert seats[0][0] is None
    assert seats[0][1] is snapshot
    assert seats[1][0] is snapshot
    assert seats[1][1] is None
    assert seats[2][0] is None
    assert seats[2][1] is self_play


def test_normalize_learner_seats_validates_length_and_range():
    assert _normalize_learner_seats(1, num_envs=3, num_players=2) == [1, 1, 1]
    assert _normalize_learner_seats([0, 1, 0], num_envs=3, num_players=2) == [0, 1, 0]
    with pytest.raises(ValueError):
        _normalize_learner_seats([0, 1], num_envs=3, num_players=2)
    with pytest.raises(ValueError):
        _normalize_learner_seats([0, 2, 1], num_envs=3, num_players=2)


def test_trim_fleets_for_forward_keeps_planet_tensors_and_trims_fleets():
    feats = EncodedObs(
        planet_feats=torch.zeros(2, 64, 19),
        planet_mask=torch.ones(2, 64, dtype=torch.bool),
        planet_owned_mask=torch.zeros(2, 64, dtype=torch.bool),
        planet_ids=torch.arange(64).expand(2, -1),
        planet_garrison=torch.zeros(2, 64),
        fleet_feats=torch.zeros(2, 384, 20),
        fleet_mask=torch.zeros(2, 384, dtype=torch.bool),
    )
    feats.fleet_mask[0, 3] = True
    feats.fleet_mask[1, 18] = True

    trimmed = _trim_fleets_for_forward(feats)

    assert trimmed.fleet_feats.shape[1] == 19
    assert trimmed.planet_feats.data_ptr() == feats.planet_feats.data_ptr()
    assert trimmed.planet_mask.data_ptr() == feats.planet_mask.data_ptr()


def test_projected_population_potential_uses_best_enemy_and_remaining_horizon():
    obs = {
        "player": 0,
        "step": 10,
        "planets": [
            [0, 0, 0.0, 0.0, 1.0, 10, 2],
            [1, 1, 0.0, 0.0, 1.0, 20, 1],
            [2, 2, 0.0, 0.0, 1.0, 5, 4],
        ],
        "fleets": [[10, 0, 0.0, 0.0, 0.0, 0, 5]],
    }
    phi = _obs_reward_potential(
        obs, player=0, num_players=3, episode_steps=100, production_weight=1.0
    )
    own = 15 + 90 * 2
    best_enemy = max(20 + 90 * 1, 5 + 90 * 4)
    assert phi == pytest.approx(own - best_enemy)

    terminal = {**obs, "step": 100}
    terminal_phi = _obs_reward_potential(
        terminal,
        player=0,
        num_players=3,
        episode_steps=100,
        production_weight=1.0,
    )
    assert terminal_phi == pytest.approx(15 - 20)

    done_state = [
        {"status": "DONE", "observation": obs},
        {"status": "DONE", "observation": {**obs, "player": 1}},
        {"status": "DONE", "observation": {**obs, "player": 2}},
    ]
    early_finish_phi = _state_reward_potential(
        done_state, player=0, num_players=3, episode_steps=100, production_weight=1.0
    )
    assert early_finish_phi == pytest.approx(phi)


def test_kaggle_vecenv_helpers_preserve_policy_target_sidecars():
    trackers = [_FleetTargetTracker(), _FleetTargetTracker()]
    state0 = [
        {
            "observation": {
                "player": 0,
                "step": 0,
                "planets": [
                    [0, 0, 10.0, 10.0, 1.0, 50, 3],
                    [1, 1, 90.0, 90.0, 1.0, 10, 2],
                ],
                "fleets": [],
            }
        },
        {"observation": {"player": 1, "step": 0, "planets": [], "fleets": []}},
    ]
    actions = [
        [[0, 0.5, 10, 1, 2.0, 90.0, 90.0]],
        [],
    ]

    annotated0 = _annotate_state_with_fleet_targets(trackers, state0)
    _record_action_sidecars(trackers, annotated0, actions)
    state1 = [
        {
            "observation": {
                "player": 0,
                "step": 1,
                "planets": state0[0]["observation"]["planets"],
                "fleets": [[42, 0, 11.0, 11.0, 0.5, 0, 10]],
            }
        },
        {"observation": {"player": 1, "step": 1, "planets": [], "fleets": []}},
    ]

    assert _strip_action_sidecars(actions) == [[[0, 0.5, 10]], []]
    annotated1 = _annotate_state_with_fleet_targets(trackers, state1)
    assert annotated1[0]["observation"]["fleet_targets"] == {
        "42": [1, 1.0, 90.0, 90.0]
    }


def test_numpy_fast_rollout_records_configured_learner_seats():
    model = OrbitPolicy(OrbitPolicyConfig(dim=16, ff_dim=32, depth=1, n_heads=2))
    opponent = OpponentSlot("noop", agent=lambda _obs: [])
    vec = NumpyVecEnv(
        num_envs=2,
        num_players=2,
        episode_steps=12,
        ship_speed=6.0,
        random_seed=0,
    )

    with vec:
        trajs = rollout_episodes_batched(
            model,
            vec,
            [[opponent], [opponent]],
            num_players=2,
            learner_seat=[0, 1],
            device="cpu",
        )

    assert [traj.learner_seat for traj in trajs] == [0, 1]
    assert all(traj.seat_rewards for traj in trajs)
    assert all(traj.encoded for traj in trajs)
    for traj in trajs:
        assert any(mask.any() for mask in traj.owned_mask)
        for owned, obs in zip(traj.owned_mask, traj.encoded, strict=True):
            assert owned.equal(obs.planet_owned_mask)


def test_numpy_reward_potentials_match_materialized_observations():
    vec = NumpyVecEnv(
        num_envs=2,
        num_players=2,
        episode_steps=12,
        ship_speed=6.0,
        random_seed=0,
    )
    with vec:
        states = vec.reset()
        rows = [(0, 0), (1, 1)]
        potentials = vec.reward_potentials(rows, production_weight=1.0)
        expected = [
            _obs_reward_potential(
                states[env_idx][player]["observation"],
                player=player,
                num_players=2,
                episode_steps=12,
                production_weight=1.0,
            )
            for env_idx, player in rows
        ]

    assert potentials.tolist() == pytest.approx(expected)


def test_sharded_numpy_fast_rollout_records_configured_learner_seats():
    model = OrbitPolicy(OrbitPolicyConfig(dim=16, ff_dim=32, depth=1, n_heads=2))
    opponent = OpponentSlot("noop", agent=lambda _obs: [])
    vec = ShardedNumpyVecEnv(
        num_envs=4,
        num_players=2,
        episode_steps=12,
        ship_speed=6.0,
        random_seed=0,
        num_workers=2,
    )

    with vec:
        trajs = rollout_episodes_batched(
            model,
            vec,
            [[opponent], [opponent], [opponent], [opponent]],
            num_players=2,
            learner_seat=[0, 1, 0, 1],
            device="cpu",
        )

    assert [traj.learner_seat for traj in trajs] == [0, 1, 0, 1]
    assert all(traj.seat_rewards for traj in trajs)
    assert all(traj.encoded for traj in trajs)
    for traj in trajs:
        assert any(mask.any() for mask in traj.owned_mask)
        for owned, obs in zip(traj.owned_mask, traj.encoded, strict=True):
            assert owned.equal(obs.planet_owned_mask)
