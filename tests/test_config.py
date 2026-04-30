from pathlib import Path

import yaml

from owars.training.config import RunConfig, deep_override


def test_load_default():
    cfg = RunConfig.from_dict({})
    assert cfg.run.name == "default"
    assert cfg.game.num_players == 2
    assert cfg.model.depth == 3


def test_unknown_key_raises():
    try:
        RunConfig.from_dict({"model": {"not_a_real_field": 1}})
    except KeyError:
        return
    raise AssertionError("expected KeyError for unknown model field")


def test_non_positive_lambda_policy_alpha_raises():
    try:
        RunConfig.from_dict({"ppo": {"lambda_policy_alpha": 0.0}})
    except ValueError:
        return
    raise AssertionError("expected ValueError for non-positive lambda_policy_alpha")


def test_deep_override_merges():
    base = {"a": {"x": 1, "y": 2}, "b": 3}
    over = {"a": {"y": 20}, "c": 5}
    out = deep_override(base, over)
    assert out["a"]["x"] == 1 and out["a"]["y"] == 20
    assert out["b"] == 3 and out["c"] == 5


def test_real_yaml_loads():
    cfg = RunConfig.from_dict(yaml.safe_load(open("configs/ppo_base.yaml")))
    assert cfg.run.name == "ppo_base"
    assert cfg.game.episode_steps == 500
    assert 0.0 <= cfg.opponents.self_play_prob <= 1.0
    assert cfg.opponents.top_k > 0
    assert cfg.ppo.lambda_policy_alpha == 0.05


def test_learned_configs_use_vapo_length_adaptive_gae():
    for path in ("configs/ppo_base.yaml", "configs/ppo_4p.yaml"):
        cfg = RunConfig.from_dict(yaml.safe_load(open(path)))
        assert cfg.ppo.lambda_policy_alpha == 0.05


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
            RunConfig.from_dict(deep_override(base, nested))
