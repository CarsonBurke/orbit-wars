from pathlib import Path

import pytest
import yaml

from owars.training.config import RunConfig, deep_override


def test_load_default():
    cfg = RunConfig.from_dict({})
    assert cfg.run.name == "default"
    assert cfg.game.num_players == 2
    assert cfg.model.depth == 3


def test_unknown_key_raises():
    with pytest.raises(KeyError):
        RunConfig.from_dict({"model": {"not_a_real_field": 1}})


def test_invalid_gae_lambda_raises():
    with pytest.raises(ValueError):
        RunConfig.from_dict({"ppo": {"gae_lambda": 1.5}})


def test_invalid_gamma_raises():
    with pytest.raises(ValueError):
        RunConfig.from_dict({"ppo": {"gamma": 1.01}})


def test_invalid_advantage_transform_raises():
    with pytest.raises(ValueError):
        RunConfig.from_dict({"ppo": {"advantage_transform": "zscoreish"}})


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


@pytest.mark.parametrize(
    "ppo_cfg",
    [
        {"spo_eps_low": 0.0},
        {"spo_eps_high": 0.0},
        {"spo_eps_low": 0.3, "spo_eps_high": 0.2},
    ],
)
def test_invalid_spo_eps_raises(ppo_cfg):
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
    for path in ("configs/ppo_vs_sniper.yaml", "configs/sac_vs_sniper.yaml"):
        with open(path) as f:
            cfg = RunConfig.from_dict(yaml.safe_load(f))
        assert cfg.opponents.mode == "fixed"
        assert cfg.opponents.fixed_opponents == ["sniper"]


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
