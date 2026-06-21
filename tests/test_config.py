import math
from pathlib import Path

import pytest
import yaml

from owars.training.config import RunConfig, deep_override


def test_load_default():
    cfg = RunConfig.from_dict({})
    assert cfg.run.name == "default"
    assert cfg.game.num_players == 2
    assert cfg.game.train_num_players == [2]
    assert cfg.model.depth == 3
    assert cfg.opponents.fixed_opponents == ["sniper_v18"]
    assert cfg.sac.builtin_opponents == ["random", "sniper_v18", "heuristic"]


def test_unknown_key_raises():
    with pytest.raises(KeyError):
        RunConfig.from_dict({"model": {"not_a_real_field": 1}})


def test_invalid_gae_lambda_raises():
    with pytest.raises(ValueError):
        RunConfig.from_dict({"ppo": {"gae_lambda": 1.5}})


def test_invalid_gamma_raises():
    with pytest.raises(ValueError):
        RunConfig.from_dict({"ppo": {"gamma": 1.01}})


def test_train_num_players_loads_2p_4p_mix():
    cfg = RunConfig.from_dict({"game": {"num_players": 2, "train_num_players": [2, 4]}})

    assert cfg.game.train_num_players == [2, 4]


def test_train_num_players_rejects_invalid_format():
    with pytest.raises(ValueError, match="train_num_players"):
        RunConfig.from_dict({"game": {"train_num_players": [3]}})


def test_invalid_advantage_transform_raises():
    with pytest.raises(ValueError):
        RunConfig.from_dict({"ppo": {"advantage_transform": "zscoreish"}})


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("kl_lr_target", math.nan),
        ("kl_lr_target", math.inf),
        ("kl_lr_ema_half_life", math.nan),
        ("kl_lr_ema_half_life", math.inf),
        ("kl_lr_min_scale", math.nan),
        ("kl_lr_min_scale", math.inf),
        ("kl_lr_max_scale", math.nan),
        ("kl_lr_max_scale", math.inf),
    ],
)
def test_non_finite_kl_lr_config_raises(key: str, value: float):
    with pytest.raises(ValueError, match=key):
        RunConfig.from_dict({"optim": {key: value}})


def test_invalid_reward_signal_raises():
    with pytest.raises(ValueError, match="reward.signal"):
        RunConfig.from_dict({"reward": {"signal": "not_a_signal"}})


def test_win_terminal_reward_normalizes_terminal_only_mc_defaults():
    cfg = RunConfig.from_dict({"reward": {"signal": "win_terminal"}})

    assert cfg.reward.potential_weight == 0.0
    assert cfg.reward.win_value == 1.0
    assert cfg.reward.loss_value == -1.0
    assert cfg.reward.draw_value == 0.0
    assert cfg.ppo.gamma == 1.0
    assert cfg.ppo.gae_lambda == 0.95
    assert cfg.ppo.value_gae_lambda == 1.0
    assert cfg.model.value_min == -1.0
    assert cfg.model.value_max == 1.0
    assert cfg.model.value_num_bins == 41
    assert cfg.model.value_symlog is False
    assert cfg.model.value_bucket == "legacy"
    assert cfg.ppo.critic_return_norm == "none"
    assert not cfg.reward.uses_dense_potential()


def test_win_terminal_reward_preserves_explicit_value_support():
    cfg = RunConfig.from_dict(
        {
            "reward": {"signal": "win_terminal"},
            "model": {
                "value_min": -2.0,
                "value_max": 2.0,
                "value_num_bins": 81,
                "value_symlog": True,
                "value_bucket": "legacy",
            },
        }
    )

    assert cfg.model.value_min == -2.0
    assert cfg.model.value_max == 2.0
    assert cfg.model.value_num_bins == 81
    assert cfg.model.value_symlog is True
    assert cfg.model.value_bucket == "legacy"


def test_win_terminal_reward_rejects_discounted_ppo_returns():
    with pytest.raises(ValueError, match="ppo.gamma=1.0"):
        RunConfig.from_dict(
            {"reward": {"signal": "win_terminal"}, "ppo": {"gamma": 0.997}}
        )
    with pytest.raises(ValueError, match="ppo.value_gae_lambda=1.0"):
        RunConfig.from_dict(
            {
                "reward": {"signal": "win_terminal"},
                "ppo": {"value_gae_lambda": 0.95},
            }
        )


def test_invalid_value_gae_lambda_raises():
    with pytest.raises(ValueError, match="ppo.value_gae_lambda"):
        RunConfig.from_dict({"ppo": {"value_gae_lambda": 1.5}})


def test_fixed_opponent_mode_loads():
    cfg = RunConfig.from_dict(
        {"opponents": {"mode": "fixed", "fixed_opponents": ["sniper"]}}
    )
    assert cfg.opponents.mode == "fixed"
    assert cfg.opponents.fixed_opponents == ["sniper"]


def test_fixed_opponent_mode_requires_known_builtin():
    with pytest.raises(ValueError, match="opponents.fixed_opponents"):
        RunConfig.from_dict(
            {"opponents": {"mode": "fixed", "fixed_opponents": ["not_real"]}}
        )


def test_fixed_opponent_mode_requires_non_empty_opponents():
    with pytest.raises(ValueError, match="requires non-empty"):
        RunConfig.from_dict(
            {"opponents": {"mode": "fixed", "fixed_opponents": []}}
        )


def test_no_builtins_opponent_mode_loads():
    cfg = RunConfig.from_dict(
        {
            "opponents": {
                "mode": "no_builtins",
                "snapshot_every": 1,
                "active_pool_size": 12,
                "historical_training_archive_size": 64,
            }
        }
    )

    assert cfg.opponents.mode == "no_builtins"
    assert cfg.opponents.current_learner_prob == 0.4
    assert cfg.opponents.active_pool_prob == 0.3
    assert cfg.opponents.historical_archive_prob == 0.3
    assert cfg.opponents.active_sample_panel_size == 8
    assert cfg.opponents.historical_sample_panel_size == 8
    assert cfg.opponents.fixed_opponents == ["sniper_v18"]


def test_no_builtins_opponent_mode_rejects_builtin_value_pretraining():
    with pytest.raises(ValueError, match="pretrain_updates=0"):
        RunConfig.from_dict(
            {
                "opponents": {"mode": "no_builtins"},
                "ppo": {"pretrain_updates": 1},
            }
        )


@pytest.mark.parametrize(
    "opponents_cfg",
    [
        {"current_learner_prob": -0.1},
        {"current_learner_prob": math.nan},
        {"active_difficulty_weight": math.inf},
        {
            "current_learner_prob": 0.0,
            "active_pool_prob": 0.0,
            "historical_archive_prob": 0.0,
        },
        {"active_pool_size": 0},
        {"active_sample_panel_size": 0},
        {"historical_training_archive_size": 0},
        {"active_recency_half_life_updates": 0.0},
        {"min_games_before_active_eviction": -1},
        {"active_stats_ema_decay": 1.0},
        {"historical_sample_panel_size": 0},
        {"historical_agent_cache_size": 0},
        {"recent_eviction_archive_size": -1},
        {"notable_archive_size": -1},
        {
            "historical_training_archive_size": 4,
            "recent_eviction_archive_size": 3,
            "notable_archive_size": 2,
        },
        {
            "historical_training_archive_size": 8,
            "recent_eviction_archive_size": 7,
        },
    ],
)
def test_invalid_no_builtins_pool_config_raises(opponents_cfg):
    with pytest.raises(ValueError):
        RunConfig.from_dict({"opponents": {"mode": "no_builtins", **opponents_cfg}})


def test_invalid_rollout_compile_width_raises():
    with pytest.raises(ValueError, match="compile_fleet_width"):
        RunConfig.from_dict({"rollout": {"compile_fleet_width": 0}})
    cfg = RunConfig.from_dict(
        {
            "model": {"encoder_backend": "destination_conditioned"},
            "rollout": {"compile_fleet_width": 0},
        }
    )
    assert cfg.rollout.compile_fleet_width == 0
    with pytest.raises(ValueError, match="snapshot_compile_rows"):
        RunConfig.from_dict({"rollout": {"snapshot_compile_rows": 0}})


def test_rollout_compile_defaults_enabled():
    cfg = RunConfig.from_dict({})

    assert cfg.rollout.compile_policy is True
    assert cfg.rollout.compile_fleet_width == 1024
    assert cfg.rollout.snapshot_compile_rows == 64
    assert cfg.rollout.detail_timing is False
    assert cfg.rollout.sample_detail_timing is False

    timed = RunConfig.from_dict(
        {"rollout": {"detail_timing": True, "sample_detail_timing": True}}
    )
    assert timed.rollout.detail_timing is True
    assert timed.rollout.sample_detail_timing is True


def test_critic_return_norm_config_loads():
    cfg = RunConfig.from_dict(
        {
            "ppo": {
                "critic_return_norm": "none",
                "critic_return_norm_gamma": 0.9,
                "critic_return_norm_clip": None,
                "critic_return_norm_epsilon": 1.0e-6,
            }
        }
    )

    assert cfg.ppo.critic_return_norm == "none"
    assert cfg.ppo.critic_return_norm_gamma == 0.9
    assert cfg.ppo.critic_return_norm_clip is None
    assert cfg.ppo.critic_return_norm_epsilon == 1.0e-6


@pytest.mark.parametrize(
    "ppo_cfg",
    [
        {"critic_return_norm": "zscore"},
        {"critic_return_norm_gamma": 0.0},
        {"critic_return_norm_gamma": 1.1},
        {"critic_return_norm_clip": 0.0},
        {"critic_return_norm_epsilon": 0.0},
    ],
)
def test_invalid_critic_return_norm_config_raises(ppo_cfg):
    with pytest.raises(ValueError):
        RunConfig.from_dict({"ppo": ppo_cfg})


@pytest.mark.parametrize(
    "ppo_cfg",
    [
        {"clip_coef": 0.0},
        {"clip_coef_high": 0.0},
        {"clip_coef": 0.3, "clip_coef_high": 0.2},
    ],
)
def test_invalid_clip_coef_raises(ppo_cfg):
    with pytest.raises(ValueError):
        RunConfig.from_dict({"ppo": ppo_cfg})


@pytest.mark.parametrize(
    "model_cfg",
    [
        {"planet_rope_fraction": -0.1},
        {"planet_rope_fraction": 1.1},
        {"planet_rope_base": 0.0},
    ],
)
def test_invalid_planet_rope_config_raises(model_cfg):
    with pytest.raises(ValueError):
        RunConfig.from_dict({"model": model_cfg})


@pytest.mark.parametrize(
    "model_cfg",
    [
        {"n_heads": 0},
        {"n_kv_heads": 0},
        {"n_heads": 4, "n_kv_heads": 3},
        {"dim": 31, "n_heads": 4},
    ],
)
def test_invalid_attention_head_config_raises(model_cfg):
    with pytest.raises(ValueError):
        RunConfig.from_dict({"model": model_cfg})


def test_invalid_value_support_raises():
    with pytest.raises(ValueError):
        RunConfig.from_dict({"model": {"value_min": 1.0, "value_max": 1.0}})


def test_invalid_value_sigma_to_bin_ratio_raises():
    with pytest.raises(ValueError, match="value_sigma_to_bin_ratio"):
        RunConfig.from_dict({"model": {"value_sigma_to_bin_ratio": 0.0}})


def test_invalid_value_bucket_raises():
    with pytest.raises(ValueError, match="value_bucket"):
        RunConfig.from_dict({"model": {"value_bucket": "not_real"}})


def test_dreamer3_value_bucket_requires_odd_bins_and_symmetric_bounds():
    with pytest.raises(ValueError, match="odd"):
        RunConfig.from_dict({"model": {"value_bucket": "dreamer3", "value_num_bins": 510}})
    with pytest.raises(ValueError, match="symmetric"):
        RunConfig.from_dict(
            {
                "model": {
                    "value_bucket": "dreamer3",
                    "value_min": -8.0,
                    "value_max": 9.0,
                }
            }
        )
    with pytest.raises(ValueError, match="value_symlog"):
        RunConfig.from_dict({"model": {"value_bucket": "dreamer3", "value_symlog": True}})


def test_invalid_critic_mtp_horizon_raises():
    with pytest.raises(ValueError, match="critic_mtp_horizon"):
        RunConfig.from_dict({"model": {"critic_mtp_horizon": 0}})


def test_deep_override_merges():
    base = {"a": {"x": 1, "y": 2}, "b": 3}
    over = {"a": {"y": 20}, "c": 5}
    out = deep_override(base, over)
    assert out["a"]["x"] == 1 and out["a"]["y"] == 20
    assert out["b"] == 3 and out["c"] == 5


def test_real_yaml_loads():
    with open("configs/ppo_base.yaml") as f:
        cfg = RunConfig.from_dict(yaml.safe_load(f))
    assert cfg.run.name == "ppo_base"
    assert cfg.game.episode_steps == 500
    assert 0.0 <= cfg.opponents.self_play_prob <= 1.0
    assert cfg.opponents.top_k > 0
    assert cfg.ppo.gamma == 0.997
    assert cfg.ppo.gae_lambda == 0.95


def test_learned_configs_use_conventional_gae():
    paths = sorted(Path("configs").glob("ppo*.yaml"))
    assert paths
    for path in paths:
        with open(path) as f:
            cfg = RunConfig.from_dict(yaml.safe_load(f))
        assert cfg.ppo.gamma == 0.997
        assert cfg.ppo.gae_lambda == 0.95


def test_sniper_training_configs_load():
    expected = {
        "configs/ppo_sniper.yaml": ["sniper_v18"],
        "configs/ppo_sniper_oldblock.yaml": ["sniper_v17"],
        "configs/ppo_vs_sniper.yaml": ["sniper"],
        "configs/sac_vs_sniper.yaml": ["sniper_v18"],
    }
    for path, fixed_opponents in expected.items():
        with open(path) as f:
            cfg = RunConfig.from_dict(yaml.safe_load(f))
        assert cfg.opponents.mode == "fixed"
        assert cfg.opponents.fixed_opponents == fixed_opponents


def test_ablation_yaml_keys_load():
    for matrix_path in Path("ablations").glob("*.yaml"):
        matrix = yaml.safe_load(matrix_path.read_text())
        base = yaml.safe_load(Path(matrix["base"]).read_text()) or {}
        for cell in matrix["cells"]:
            nested = {}
            for key, value in cell.items():
                cursor = nested
                parts = key.split(".")
                for part in parts[:-1]:
                    cursor = cursor.setdefault(part, {})
                cursor[parts[-1]] = value
            try:
                RunConfig.from_dict(deep_override(base, nested))
            except (KeyError, ValueError) as exc:
                raise AssertionError(f"{matrix_path} cell {cell} does not load") from exc
