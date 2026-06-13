from __future__ import annotations

import inspect
from types import SimpleNamespace

from owars.policies.config import OrbitPolicyConfig
from owars.training import evaluate as evaluate_mod


class _FakePolicy:
    def __init__(self, cfg):
        self.cfg = cfg

    def to(self, _device):
        return self

    def load_state_dict(self, _state):
        return None

    def eval(self):
        return self


def test_evaluate_defaults_to_rust_backend():
    default = inspect.signature(evaluate_mod.evaluate_ckpt).parameters[
        "env_backend"
    ].default

    assert default == "rust"


def test_evaluate_uses_deterministic_actions_by_default(monkeypatch):
    seen: list[bool] = []

    monkeypatch.setattr(
        evaluate_mod.torch,
        "load",
        lambda *_args, **_kwargs: {
            "config": OrbitPolicyConfig(dim=32, ff_dim=64, depth=1, n_heads=2).to_dict(),
            "model": {},
        },
    )
    monkeypatch.setattr(evaluate_mod, "OrbitPolicy", _FakePolicy)

    def fake_rollout_episode(*_args, deterministic: bool, **_kwargs):
        seen.append(deterministic)
        return SimpleNamespace(won=True, drawn=False, final_score=1.0)

    monkeypatch.setattr(evaluate_mod, "rollout_episode", fake_rollout_episode)

    evaluate_mod.evaluate_ckpt(
        "dummy.pt",
        n_games=1,
        baselines=("random",),
        num_envs=1,
        env_backend="kaggle",
    )

    assert seen == [True]


def test_evaluate_can_request_stochastic_actions(monkeypatch):
    seen: list[bool] = []

    monkeypatch.setattr(
        evaluate_mod.torch,
        "load",
        lambda *_args, **_kwargs: {
            "config": OrbitPolicyConfig(dim=32, ff_dim=64, depth=1, n_heads=2).to_dict(),
            "model": {},
        },
    )
    monkeypatch.setattr(evaluate_mod, "OrbitPolicy", _FakePolicy)

    def fake_rollout_episode(*_args, deterministic: bool, **_kwargs):
        seen.append(deterministic)
        return SimpleNamespace(won=True, drawn=False, final_score=1.0)

    monkeypatch.setattr(evaluate_mod, "rollout_episode", fake_rollout_episode)

    evaluate_mod.evaluate_ckpt(
        "dummy.pt",
        n_games=1,
        baselines=("random",),
        num_envs=1,
        env_backend="kaggle",
        deterministic=False,
    )

    assert seen == [False]


def test_vectorized_evaluate_forwards_deterministic_flag(monkeypatch):
    seen: list[bool] = []

    monkeypatch.setattr(
        evaluate_mod.torch,
        "load",
        lambda *_args, **_kwargs: {
            "config": OrbitPolicyConfig(dim=32, ff_dim=64, depth=1, n_heads=2).to_dict(),
            "model": {},
        },
    )
    monkeypatch.setattr(evaluate_mod, "OrbitPolicy", _FakePolicy)

    class FakeVec:
        def __init__(self, **kwargs):
            self.num_envs = kwargs["num_envs"]

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

    def fake_rollout_batched(*_args, deterministic: bool, **_kwargs):
        seen.append(deterministic)
        return [SimpleNamespace(won=True, drawn=False, final_score=1.0)]

    monkeypatch.setattr(evaluate_mod, "NumpyVecEnv", FakeVec)
    monkeypatch.setattr(evaluate_mod, "rollout_episodes_batched", fake_rollout_batched)

    evaluate_mod.evaluate_ckpt(
        "dummy.pt",
        n_games=1,
        baselines=("random",),
        num_envs=2,
        env_backend="numpy",
    )

    assert seen == [True]
