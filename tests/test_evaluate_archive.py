from __future__ import annotations

import inspect
import math
from pathlib import Path
from types import SimpleNamespace

import pytest

from owars.policies.config import OrbitPolicyConfig
from owars.training import evaluate_archive as archive_mod


class _FakePolicy:
    def __init__(self, cfg):
        self.cfg = cfg

    def to(self, _device):
        return self

    def load_state_dict(self, _state):
        return None

    def eval(self):
        return self


def test_load_validation_manifest_resolves_paths_and_ratings(tmp_path: Path):
    manifest_path = tmp_path / "archive.yaml"
    manifest_path.write_text(
        """
version: val_archive_v1
created_update: 12
members:
  - name: early
    path: checkpoints/early.pt
    weight: 2.5
    bucket: early
    rating: 1010
  - name: late
    path: /abs/late.pt
    fixed_rating: 1200
"""
    )

    manifest = archive_mod.load_validation_manifest(manifest_path)

    assert manifest.version == "val_archive_v1"
    assert manifest.created_update == 12
    assert manifest.members[0].path == tmp_path / "checkpoints" / "early.pt"
    assert manifest.members[0].weight == 2.5
    assert manifest.members[0].bucket == "early"
    assert manifest.members[0].rating == 1010.0
    assert manifest.members[1].path == Path("/abs/late.pt")
    assert manifest.members[1].rating == 1200.0


def test_load_validation_manifest_rejects_duplicate_members(tmp_path: Path):
    manifest_path = tmp_path / "archive.yaml"
    manifest_path.write_text(
        """
version: val_archive_v1
members:
  - name: dup
    path: a.pt
  - name: dup
    path: b.pt
"""
    )

    with pytest.raises(ValueError, match="duplicate"):
        archive_mod.load_validation_manifest(manifest_path)


@pytest.mark.parametrize(
    "field",
    [
        "weight: .nan",
        "weight: .inf",
        "rating: .nan",
        "rating: .inf",
    ],
)
def test_load_validation_manifest_rejects_non_finite_values(
    tmp_path: Path,
    field: str,
):
    manifest_path = tmp_path / "archive.yaml"
    manifest_path.write_text(
        f"""
version: val_archive_v1
members:
  - name: bad
    path: bad.pt
    {field}
"""
    )

    with pytest.raises(ValueError, match="finite"):
        archive_mod.load_validation_manifest(manifest_path)


@pytest.mark.parametrize(
    "bucket",
    [
        "[]",
        "{}",
        "''",
    ],
)
def test_load_validation_manifest_rejects_malformed_bucket(
    tmp_path: Path,
    bucket: str,
):
    manifest_path = tmp_path / "archive.yaml"
    manifest_path.write_text(
        f"""
version: val_archive_v1
members:
  - name: bad
    path: bad.pt
    bucket: {bucket}
"""
    )

    with pytest.raises(ValueError, match="bucket"):
        archive_mod.load_validation_manifest(manifest_path)


def test_aggregate_archive_metrics_weighted_buckets_and_pseudo_elo():
    manifest = archive_mod.ValidationManifest(
        version="val_archive_v1",
        members=(
            archive_mod.ValidationMember(
                name="early",
                path=Path("early.pt"),
                weight=1.0,
                bucket="early",
                rating=1000.0,
            ),
            archive_mod.ValidationMember(
                name="late",
                path=Path("late.pt"),
                weight=3.0,
                bucket="late",
                rating=1200.0,
            ),
        ),
    )
    metrics = [
        archive_mod.MemberMetrics(
            name="early",
            bucket="early",
            weight=1.0,
            rating=1000.0,
            n_games=4,
            wins=1,
            draws=0,
            losses=3,
            win_rate=0.25,
            draw_rate=0.0,
            score_rate=0.25,
            mean_margin=-10.0,
            std_margin=1.0,
        ),
        archive_mod.MemberMetrics(
            name="late",
            bucket="late",
            weight=3.0,
            rating=1200.0,
            n_games=4,
            wins=3,
            draws=0,
            losses=1,
            win_rate=0.75,
            draw_rate=0.0,
            score_rate=0.75,
            mean_margin=30.0,
            std_margin=2.0,
        ),
    ]

    result = archive_mod.aggregate_archive_metrics(manifest, metrics)

    assert result["metric_prefix"] == "validation/val_archive_v1"
    assert result["weighted_win_rate"] == pytest.approx(0.625)
    assert result["weighted_margin"] == pytest.approx(20.0)
    assert result["bucket_win_rate_min"] == pytest.approx(0.25)
    assert result["bucket_win_rate_p10"] == pytest.approx(0.30)
    assert result["bucket_margin_min"] == pytest.approx(-10.0)
    assert result["per_bucket"]["early"]["win_rate"] == pytest.approx(0.25)
    assert result["per_bucket"]["late"]["mean_margin"] == pytest.approx(30.0)
    assert 1000.0 < result["pseudo_elo"] < 1400.0


def test_aggregate_archive_metrics_ignores_zero_weight_members():
    manifest = archive_mod.ValidationManifest(
        version="val_archive_v1",
        members=(
            archive_mod.ValidationMember(
                name="active",
                path=Path("active.pt"),
                weight=1.0,
                bucket="active",
            ),
            archive_mod.ValidationMember(
                name="inactive",
                path=Path("inactive.pt"),
                weight=0.0,
                bucket="inactive",
            ),
        ),
    )
    metrics = [
        archive_mod.MemberMetrics(
            name="active",
            bucket="active",
            weight=1.0,
            rating=None,
            n_games=4,
            wins=2,
            draws=0,
            losses=2,
            win_rate=0.5,
            draw_rate=0.0,
            score_rate=0.5,
            mean_margin=3.0,
            std_margin=1.0,
        )
    ]

    result = archive_mod.aggregate_archive_metrics(manifest, metrics)

    assert result["n_members"] == 1
    assert result["n_manifest_members"] == 2
    assert result["n_games"] == 4
    assert set(result["members"]) == {"active"}
    assert set(result["per_bucket"]) == {"active"}


def test_fixed_panel_pseudo_elo_is_centered_for_even_score():
    rating = archive_mod.solve_fixed_panel_pseudo_elo(
        [1000.0, 1200.0],
        [1.0, 1.0],
        0.5,
        scale=400.0 / math.log(10.0),
    )

    assert rating == pytest.approx(1100.0)


def test_archive_eval_env_count_uses_largest_exact_divisor():
    assert archive_mod._exact_eval_envs(20, 16) == 10
    assert archive_mod._exact_eval_envs(32, 16) == 16
    assert archive_mod._exact_eval_envs(17, 16) == 1


@pytest.mark.parametrize("scale", [0.0, math.nan, math.inf])
def test_fixed_panel_pseudo_elo_rejects_invalid_scale(scale: float):
    with pytest.raises(ValueError, match="scale"):
        archive_mod.solve_fixed_panel_pseudo_elo(
            [1000.0],
            [1.0],
            0.5,
            scale=scale,
        )


def test_aggregate_archive_metrics_requires_all_ratings_for_pseudo_elo():
    manifest = archive_mod.ValidationManifest(
        version="val_archive_v1",
        members=(
            archive_mod.ValidationMember(
                name="rated",
                path=Path("rated.pt"),
                weight=1.0,
                bucket="main",
                rating=1000.0,
            ),
            archive_mod.ValidationMember(
                name="unrated",
                path=Path("unrated.pt"),
                weight=1.0,
                bucket="main",
            ),
        ),
    )
    metrics = [
        archive_mod.MemberMetrics(
            name="rated",
            bucket="main",
            weight=1.0,
            rating=1000.0,
            n_games=2,
            wins=1,
            draws=0,
            losses=1,
            win_rate=0.5,
            draw_rate=0.0,
            score_rate=0.5,
            mean_margin=0.0,
            std_margin=1.0,
        ),
        archive_mod.MemberMetrics(
            name="unrated",
            bucket="main",
            weight=1.0,
            rating=None,
            n_games=2,
            wins=1,
            draws=0,
            losses=1,
            win_rate=0.5,
            draw_rate=0.0,
            score_rate=0.5,
            mean_margin=0.0,
            std_margin=1.0,
        ),
    ]

    result = archive_mod.aggregate_archive_metrics(manifest, metrics)

    assert "pseudo_elo" not in result


def test_evaluate_archive_defaults_to_rust_and_deterministic():
    signature = inspect.signature(archive_mod.evaluate_archive_ckpt)

    assert signature.parameters["env_backend"].default == "rust"
    assert signature.parameters["deterministic"].default is True
    assert signature.parameters["opponent_deterministic"].default is True


def test_evaluate_archive_runs_candidate_vs_validation_only(monkeypatch):
    seen = {"agents": [], "rollouts": []}

    monkeypatch.setattr(
        archive_mod.torch,
        "load",
        lambda *_args, **_kwargs: {
            "config": OrbitPolicyConfig(dim=32, ff_dim=64, depth=1, n_heads=2).to_dict(),
            "model": {},
        },
    )
    monkeypatch.setattr(archive_mod, "OrbitPolicy", _FakePolicy)

    class FakeLearnedAgent:
        def __init__(self, ckpt_path, *, device, deterministic, compile_mode):
            self.ckpt_path = Path(ckpt_path)
            self.device = device
            self.deterministic = deterministic
            self.compile_mode = compile_mode
            self.model = None
            seen["agents"].append(self)

    class FakeVec:
        def __init__(self, **kwargs):
            self.num_envs = kwargs["num_envs"]

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

    def fake_rollout_batched(
        _model,
        _vec,
        opponents_per_env,
        *,
        deterministic,
        **_kwargs,
    ):
        seen["rollouts"].append((opponents_per_env, deterministic))
        assert all(
            slot.name == "validation:member_a"
            for env_slots in opponents_per_env
            for slot in env_slots
        )
        return [
            SimpleNamespace(won=True, drawn=False, final_score=5.0),
            SimpleNamespace(won=False, drawn=False, final_score=-1.0),
        ]

    monkeypatch.setattr(archive_mod, "LearnedAgent", FakeLearnedAgent)
    monkeypatch.setattr(archive_mod, "NumpyVecEnv", FakeVec)
    monkeypatch.setattr(archive_mod, "rollout_episodes_batched", fake_rollout_batched)

    manifest = archive_mod.ValidationManifest(
        version="val_archive_v1",
        members=(
            archive_mod.ValidationMember(
                name="member_a",
                path=Path("member_a.pt"),
                weight=1.0,
                bucket="main",
            ),
            archive_mod.ValidationMember(
                name="member_disabled",
                path=Path("member_disabled.pt"),
                weight=0.0,
                bucket="disabled",
            ),
        ),
    )

    result = archive_mod.evaluate_archive_ckpt(
        "candidate.pt",
        manifest,
        games_per_member=2,
        num_envs=2,
        env_backend="numpy",
    )

    assert [agent.ckpt_path for agent in seen["agents"]] == [Path("member_a.pt")]
    assert seen["agents"][0].deterministic is True
    assert [deterministic for _opponents, deterministic in seen["rollouts"]] == [True]
    assert result["weighted_win_rate"] == pytest.approx(0.5)
    assert result["weighted_margin"] == pytest.approx(2.0)


def test_evaluate_archive_uses_exact_batch_size(monkeypatch):
    seen = {"vec_envs": [], "rollout_envs": []}

    monkeypatch.setattr(
        archive_mod.torch,
        "load",
        lambda *_args, **_kwargs: {
            "config": OrbitPolicyConfig(dim=32, ff_dim=64, depth=1, n_heads=2).to_dict(),
            "model": {},
        },
    )
    monkeypatch.setattr(archive_mod, "OrbitPolicy", _FakePolicy)

    class FakeLearnedAgent:
        def __init__(self, *_args, **_kwargs):
            self.model = None

    class FakeVec:
        def __init__(self, **kwargs):
            self.num_envs = kwargs["num_envs"]
            seen["vec_envs"].append(self.num_envs)

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

    def fake_rollout_batched(_model, _vec, opponents_per_env, **_kwargs):
        seen["rollout_envs"].append(len(opponents_per_env))
        return [
            SimpleNamespace(won=True, drawn=False, final_score=1.0)
            for _ in opponents_per_env
        ]

    monkeypatch.setattr(archive_mod, "LearnedAgent", FakeLearnedAgent)
    monkeypatch.setattr(archive_mod, "NumpyVecEnv", FakeVec)
    monkeypatch.setattr(archive_mod, "rollout_episodes_batched", fake_rollout_batched)

    manifest = archive_mod.ValidationManifest(
        version="val_archive_v1",
        members=(
            archive_mod.ValidationMember(
                name="member_a",
                path=Path("member_a.pt"),
                weight=1.0,
                bucket="main",
            ),
        ),
    )

    result = archive_mod.evaluate_archive_ckpt(
        "candidate.pt",
        manifest,
        games_per_member=20,
        num_envs=16,
        env_backend="numpy",
    )

    assert seen["vec_envs"] == [10]
    assert seen["rollout_envs"] == [10, 10]
    assert result["n_games"] == 20
