from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from owars.agents.learned import _FleetTargetTracker
from owars.policies.config import OrbitPolicyConfig
from owars.policies.features import EncodedObs
from owars.policies.model import OrbitPolicy
from owars.training import train as train_mod
from owars.training.config import RewardCfg
from owars.training.league import LEARNER_NAME, OpponentSlot
from owars.training.numpy_env import NumpyVecEnv
from owars.training.sharded_numpy_env import ShardedNumpyVecEnv
from owars.training.vec_env import (
    _annotate_state_with_fleet_targets,
    _record_action_sidecars,
    _strip_action_sidecars,
)
from owars.training.vec_rollout import (
    _bucket_fleets_for_graph,
    _capped_graph_rows,
    _empty_traj,
    _finalize_trajectory,
    _flush_scoped_timings,
    _materialize_records_cpu,
    _normalize_learner_seats,
    _obs_reward_potential,
    _resolve_seat_agents,
    _reward_potentials,
    _scoped_timings,
    _snapshot_rollout_graph_rows,
    _state_reward_potential,
    _trim_fleets_for_forward,
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


def test_trim_fleets_for_forward_keeps_planet_tensors_and_buckets_fleets():
    feats = EncodedObs(
        planet_feats=torch.zeros(2, 64, 19),
        planet_mask=torch.ones(2, 64, dtype=torch.bool),
        planet_owned_mask=torch.zeros(2, 64, dtype=torch.bool),
        planet_ids=torch.arange(64).expand(2, -1),
        planet_garrison=torch.zeros(2, 64),
        fleet_feats=torch.zeros(2, 513, 20),
        fleet_mask=torch.zeros(2, 513, dtype=torch.bool),
    )
    feats.fleet_mask[0, 3] = True
    feats.fleet_mask[1, 18] = True

    trimmed = _trim_fleets_for_forward(feats)

    assert trimmed.fleet_feats.shape[1] == 64
    assert trimmed.planet_feats.data_ptr() == feats.planet_feats.data_ptr()
    assert trimmed.planet_mask.data_ptr() == feats.planet_mask.data_ptr()


def test_bucket_fleets_for_graph_pads_to_static_bucket():
    feats = EncodedObs(
        planet_feats=torch.zeros(2, 64, 19),
        planet_mask=torch.ones(2, 64, dtype=torch.bool),
        planet_owned_mask=torch.zeros(2, 64, dtype=torch.bool),
        planet_ids=torch.arange(64).expand(2, -1),
        planet_garrison=torch.zeros(2, 64),
        fleet_feats=torch.zeros(2, 20, 20),
        fleet_mask=torch.zeros(2, 20, dtype=torch.bool),
        fleet_target_planet_idx=torch.full((2, 20), -1, dtype=torch.long),
    )
    feats.fleet_mask[0, 19] = True
    feats.fleet_target_planet_idx[0, 19] = 3

    padded = _bucket_fleets_for_graph(feats)

    assert padded.fleet_feats.shape[1] == 64
    assert padded.fleet_mask[0, 19]
    assert int(padded.fleet_target_planet_idx[0, 19]) == 3
    assert not bool(padded.fleet_mask[:, 20:].any())
    assert torch.all(padded.fleet_target_planet_idx[:, 20:] == -1)
    assert padded.planet_feats.data_ptr() == feats.planet_feats.data_ptr()


def test_bucket_fleets_for_graph_fixed_width_does_not_truncate_used_fleets():
    feats = EncodedObs(
        planet_feats=torch.zeros(2, 64, 19),
        planet_mask=torch.ones(2, 64, dtype=torch.bool),
        planet_owned_mask=torch.zeros(2, 64, dtype=torch.bool),
        planet_ids=torch.arange(64).expand(2, -1),
        planet_garrison=torch.zeros(2, 64),
        fleet_feats=torch.zeros(2, 1100, 20),
        fleet_mask=torch.zeros(2, 1100, dtype=torch.bool),
        fleet_target_planet_idx=torch.full((2, 1100), -1, dtype=torch.long),
    )
    feats.fleet_mask[0, 1099] = True
    feats.fleet_target_planet_idx[0, 1099] = 7

    padded = _bucket_fleets_for_graph(feats, fixed_width=1024)

    assert padded.fleet_feats.shape[1] == 2048
    assert padded.fleet_mask[0, 1099]
    assert int(padded.fleet_target_planet_idx[0, 1099]) == 7


def test_bucket_fleets_for_graph_fixed_width_pads_underfilled_batches():
    feats = EncodedObs(
        planet_feats=torch.zeros(2, 64, 19),
        planet_mask=torch.ones(2, 64, dtype=torch.bool),
        planet_owned_mask=torch.zeros(2, 64, dtype=torch.bool),
        planet_ids=torch.arange(64).expand(2, -1),
        planet_garrison=torch.zeros(2, 64),
        fleet_feats=torch.zeros(2, 20, 20),
        fleet_mask=torch.zeros(2, 20, dtype=torch.bool),
        fleet_target_planet_idx=torch.full((2, 20), -1, dtype=torch.long),
    )

    padded = _bucket_fleets_for_graph(feats, fixed_width=1024)

    assert padded.fleet_feats.shape[1] == 1024
    assert not bool(padded.fleet_mask.any())
    assert torch.all(padded.fleet_target_planet_idx == -1)


def test_bucket_fleets_for_graph_uses_inbound_summary_without_fleet_padding():
    feats = EncodedObs(
        planet_feats=torch.zeros(2, 64, 19),
        planet_mask=torch.ones(2, 64, dtype=torch.bool),
        planet_owned_mask=torch.zeros(2, 64, dtype=torch.bool),
        planet_ids=torch.arange(64).expand(2, -1),
        planet_garrison=torch.zeros(2, 64),
        fleet_feats=torch.zeros(2, 20, 20),
        fleet_mask=torch.ones(2, 20, dtype=torch.bool),
        fleet_target_planet_idx=torch.full((2, 20), -1, dtype=torch.long),
        planet_inbound_feats=torch.zeros(2, 64, 13),
    )

    padded = _bucket_fleets_for_graph(feats, fixed_width=1024)

    assert padded.fleet_feats.shape[1] == 0
    assert padded.planet_inbound_feats is not None
    assert padded.planet_inbound_feats.shape == (2, 64, 13)


def test_snapshot_rollout_graph_rows_use_fixed_capacity_when_compiled():
    assert _capped_graph_rows(44, max_rows=64) == 64
    assert _capped_graph_rows(65, max_rows=64) == 128
    assert _snapshot_rollout_graph_rows(
        7,
        snapshot_compile_rows=64,
        compile_mode="reduce-overhead",
    ) == 64
    assert _snapshot_rollout_graph_rows(
        65,
        snapshot_compile_rows=64,
        compile_mode="reduce-overhead",
    ) == 128
    assert _snapshot_rollout_graph_rows(
        7,
        snapshot_compile_rows=64,
        compile_mode=None,
    ) == 8


def test_scoped_timings_preserve_aggregate_and_prefix():
    timings = {"native_action_s": 1.0}
    scoped = _scoped_timings(timings, "current_sample")

    assert scoped is not None
    scoped["native_action_s"] = 2.5
    _flush_scoped_timings(timings, scoped, "current_sample")

    assert timings["native_action_s"] == pytest.approx(3.5)
    assert timings["current_sample/native_action_s"] == pytest.approx(2.5)


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


def test_reward_potentials_can_use_production_margin_signal():
    obs = {
        "player": 0,
        "step": 10,
        "planets": [
            [0, 0, 0.0, 0.0, 1.0, 10, 2],
            [1, 1, 0.0, 0.0, 1.0, 20, 1],
            [2, 2, 0.0, 0.0, 1.0, 5, 4],
        ],
        "fleets": [[10, 0, 0.0, 0.0, 0.0, 0, 999]],
    }
    state = [
        {"status": "ACTIVE", "observation": obs},
        {"status": "ACTIVE", "observation": {**obs, "player": 1}},
        {"status": "ACTIVE", "observation": {**obs, "player": 2}},
    ]
    values = _reward_potentials(
        vec=object(),
        states=[state],
        rows=[(0, 0)],
        num_players=3,
        episode_steps=100,
        reward_cfg=RewardCfg(signal="production_margin"),
    )

    assert values == pytest.approx([2 - 4])


def test_win_terminal_signal_has_no_dense_potential_and_applies_terminal_outcome():
    state = [
        {
            "status": "ACTIVE",
            "observation": {
                "player": 0,
                "step": 10,
                "planets": [[0, 0, 0.0, 0.0, 1.0, 10, 2]],
                "fleets": [],
            },
        }
    ]
    reward_cfg = RewardCfg(
        signal="win_terminal",
        win_value=1.0,
        loss_value=-1.0,
        draw_value=0.0,
    )

    assert _reward_potentials(
        vec=object(),
        states=[state],
        rows=[(0, 0)],
        num_players=2,
        episode_steps=100,
        reward_cfg=reward_cfg,
    ) == [0.0]

    traj = _empty_traj()
    traj.reward.append(0.0)
    _finalize_trajectory(
        traj,
        [SimpleNamespace(score=12.0, reward=12.0), SimpleNamespace(score=7.0, reward=7.0)],
        learner_seat=0,
        reward_cfg=reward_cfg,
    )

    assert traj.won is True
    assert traj.final_score == pytest.approx(5.0)
    assert traj.reward == pytest.approx([1.0])


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
            assert obs.global_feats is not None
            assert obs.global_feats.shape == (model.cfg.global_features,)


def test_numpy_fast_rollout_can_defer_log_prob_storage():
    model = OrbitPolicy(OrbitPolicyConfig(dim=16, ff_dim=32, depth=1, n_heads=2))
    opponent = OpponentSlot("noop", agent=lambda _obs: [])
    vec = NumpyVecEnv(
        num_envs=1,
        num_players=2,
        episode_steps=8,
        ship_speed=6.0,
        random_seed=0,
    )

    with vec:
        [traj] = rollout_episodes_batched(
            model,
            vec,
            [[opponent]],
            num_players=2,
            learner_seat=0,
            device="cpu",
            defer_log_prob=True,
        )

    assert traj.encoded
    assert len(traj.launch) == len(traj.target_idx) == len(traj.fraction) == len(traj.reward)
    assert traj.log_prob == []


def test_numpy_fast_rollout_can_record_chunked_ppo_records():
    model = OrbitPolicy(
        OrbitPolicyConfig(
            dim=16,
            ff_dim=32,
            depth=1,
            n_heads=2,
            encoder_backend="destination_conditioned",
        )
    )
    opponent = OpponentSlot("noop", agent=lambda _obs: [])
    vec = NumpyVecEnv(
        num_envs=2,
        num_players=2,
        episode_steps=8,
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
            defer_log_prob=True,
            chunk_records=True,
        )

    assert [traj.learner_seat for traj in trajs] == [0, 1]
    assert all(traj.record_refs for traj in trajs)
    assert all(len(traj.record_refs) == len(traj.reward) for traj in trajs)
    for traj in trajs:
        assert traj.encoded == []
        assert traj.launch == []
        assert traj.target_idx == []
        assert traj.fraction == []
        assert traj.log_prob == []
        assert traj.value == []
        assert traj.owned_mask == []
        assert traj.target_legal_mask == []
        assert all(ref.chunk["log_prob"] is None for ref in traj.record_refs)
        assert all(ref.chunk["planet_inbound_feats"] is not None for ref in traj.record_refs)
        assert all(ref.chunk["fleet_target_planet_idx"] is not None for ref in traj.record_refs)

    batch = train_mod._stack_trajectories(
        trajs,
        gamma=1.0,
        gae_lambda=1.0,
        include_old_log_prob=False,
    )

    assert int(batch["launch"].shape[0]) == sum(len(traj.reward) for traj in trajs)
    assert batch["planet_inbound_feats"] is not None
    assert batch["fleet_target_planet_idx"] is not None
    assert torch.equal(batch["old_log_prob"], torch.zeros_like(batch["launch"]))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_materialize_records_cpu_handles_mixed_record_devices():
    batch = 2
    planets = 3
    cpu_stacked = EncodedObs(
        planet_feats=torch.randn(batch, planets, 4),
        planet_mask=torch.ones(batch, planets, dtype=torch.bool),
        planet_owned_mask=torch.tensor([[True, False, True], [False, True, True]]),
        planet_ids=torch.arange(planets).expand(batch, -1),
        planet_garrison=torch.ones(batch, planets),
        fleet_feats=torch.zeros(batch, 0, 2),
        fleet_mask=torch.zeros(batch, 0, dtype=torch.bool),
        global_feats=torch.randn(batch, 2),
    )
    stacked = cpu_stacked.to("cuda")
    records = SimpleNamespace(
        launch=torch.tensor(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            device="cuda",
        ),
        target_idx=torch.tensor([[2, 0, 0], [0, 2, 0]], device="cuda"),
        fraction=torch.full((batch, planets), 0.5, device="cuda"),
        log_prob=torch.full((batch, planets), -0.25),
        target_legal_mask=torch.ones(batch, planets, planets, dtype=torch.bool),
    )
    out = SimpleNamespace(value=torch.tensor([1.5, -0.5], device="cuda"))

    got = _materialize_records_cpu(
        stacked,
        cpu_stacked,
        out,
        records,
        torch.arange(batch, device="cuda"),
        [0, 1],
        include_log_prob=True,
    )

    assert got["target_idx"].device.type == "cpu"
    assert torch.equal(got["target_idx"], records.target_idx.cpu())
    assert torch.allclose(got["launch"], records.launch.cpu())
    assert torch.allclose(got["fraction"], records.fraction.cpu())
    assert torch.allclose(got["log_prob"], records.log_prob)
    assert torch.equal(got["target_legal_mask"], records.target_legal_mask)
    assert torch.allclose(got["value"], out.value.cpu())


def test_numpy_fast_rollout_behavior_override_records_value_only_sidecars():
    model = OrbitPolicy(OrbitPolicyConfig(dim=16, ff_dim=32, depth=1, n_heads=2))
    opponent = OpponentSlot("noop", agent=lambda _obs: [])
    seen_steps: list[int] = []

    def behavior(obs):
        seen_steps.append(int(obs["step"]))
        return []

    vec = NumpyVecEnv(
        num_envs=1,
        num_players=2,
        episode_steps=4,
        ship_speed=6.0,
        random_seed=0,
    )

    with vec:
        [traj] = rollout_episodes_batched(
            model,
            vec,
            [[opponent]],
            num_players=2,
            learner_seat=0,
            device="cpu",
            learner_action_agent=behavior,
        )

    assert seen_steps
    assert len(traj.encoded) == len(traj.value) == len(traj.reward)
    assert traj.encoded
    assert traj.launch == []
    assert traj.target_idx == []
    assert traj.fraction == []
    assert traj.log_prob == []
    assert traj.owned_mask == []
    assert traj.target_legal_mask == []


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
