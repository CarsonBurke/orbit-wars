"""Evaluate a checkpoint against a frozen validation archive.

The validation archive is a versioned manifest of learned checkpoints. Routine
evaluation only schedules candidate-vs-archive-member games; archive members are
never paired against each other here.
"""

from __future__ import annotations

import argparse
import json
import math
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from ..agents.learned import LearnedAgent
from ..policies.config import OrbitPolicyConfig
from ..policies.model import OrbitPolicy, restore_fp32_params
from .league import OpponentSlot
from .numpy_env import NumpyVecEnv
from .rollout import rollout_episode
from .sharded_numpy_env import ShardedNumpyVecEnv
from .vec_env import VecEnv
from .vec_rollout import alternating_learner_seats, rollout_episodes_batched


@dataclass(frozen=True)
class ValidationMember:
    name: str
    path: Path
    weight: float = 1.0
    bucket: str = "default"
    rating: float | None = None


@dataclass(frozen=True)
class ValidationManifest:
    version: str
    members: tuple[ValidationMember, ...]
    created_update: int | None = None
    source_path: Path | None = None


@dataclass(frozen=True)
class MemberMetrics:
    name: str
    bucket: str
    weight: float
    rating: float | None
    n_games: int
    wins: int
    draws: int
    losses: int
    win_rate: float
    draw_rate: float
    score_rate: float
    mean_margin: float
    std_margin: float


def _required_str(raw: dict[str, Any], key: str, context: str) -> str:
    value = raw.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{context} requires non-empty string field {key!r}")
    return value.strip()


def _optional_rating(raw: dict[str, Any]) -> float | None:
    for key in ("rating", "elo", "fixed_rating"):
        if key in raw and raw[key] is not None:
            rating = float(raw[key])
            if not math.isfinite(rating):
                raise ValueError(f"validation member rating {key!r} must be finite")
            return rating
    return None


def _optional_bucket(raw: dict[str, Any], context: str) -> str:
    value = raw.get("bucket", "default")
    if value is None:
        return "default"
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{context} bucket must be a non-empty string")
    return value.strip()


def load_validation_manifest(path: str | Path) -> ValidationManifest:
    manifest_path = Path(path)
    raw = yaml.safe_load(manifest_path.read_text()) or {}
    if not isinstance(raw, dict):
        raise ValueError("validation manifest must be a mapping")

    version = _required_str(raw, "version", "validation manifest")
    created_update = raw.get("created_update")
    if created_update is not None:
        created_update = int(created_update)

    members_raw = raw.get("members")
    if not isinstance(members_raw, list) or not members_raw:
        raise ValueError("validation manifest requires a non-empty members list")

    names: set[str] = set()
    members: list[ValidationMember] = []
    for idx, item in enumerate(members_raw):
        if not isinstance(item, dict):
            raise ValueError(f"validation member {idx} must be a mapping")
        context = f"validation member {idx}"
        name = _required_str(item, "name", context)
        if name in names:
            raise ValueError(f"duplicate validation member name: {name!r}")
        names.add(name)

        member_path = Path(_required_str(item, "path", context)).expanduser()
        if not member_path.is_absolute():
            member_path = manifest_path.parent / member_path
        weight = float(item.get("weight", 1.0))
        if not math.isfinite(weight) or weight < 0.0:
            raise ValueError(
                f"validation member {name!r} weight must be finite and non-negative"
            )
        members.append(
            ValidationMember(
                name=name,
                path=member_path,
                weight=weight,
                bucket=_optional_bucket(item, context),
                rating=_optional_rating(item),
            )
        )

    if sum(member.weight for member in members) <= 0.0:
        raise ValueError("validation manifest must have positive total member weight")

    return ValidationManifest(
        version=version,
        created_update=created_update,
        members=tuple(members),
        source_path=manifest_path,
    )


def _load_model(ckpt_path: str | Path, device: str) -> OrbitPolicy:
    state = torch.load(ckpt_path, map_location=device)
    cfg = OrbitPolicyConfig(**state["config"])
    model = OrbitPolicy(cfg).to(device)
    if torch.device(device).type == "cuda":
        model.bfloat16()
        restore_fp32_params(model)
    model.load_state_dict(state["model"])
    model.eval()
    return model


def _make_vec(
    env_backend: str,
    *,
    num_envs: int,
    num_players: int,
    episode_steps: int,
    ship_speed: float,
    num_workers: int,
) -> VecEnv:
    vec_kwargs = dict(
        num_envs=num_envs,
        num_players=num_players,
        episode_steps=episode_steps,
        ship_speed=ship_speed,
        replay_env_idx=None,
    )
    if env_backend == "numpy":
        return NumpyVecEnv(**vec_kwargs)
    if env_backend == "numpy_mp":
        return ShardedNumpyVecEnv(**vec_kwargs, num_workers=num_workers)
    if env_backend == "rust":
        from .rust_env import RustVecEnv

        return RustVecEnv(**vec_kwargs)
    if env_backend == "kaggle":
        return VecEnv(**vec_kwargs)
    raise ValueError(f"unknown env_backend: {env_backend!r}")


def _exact_eval_envs(games_per_member: int, num_envs: int) -> int:
    limit = max(1, min(games_per_member, num_envs))
    for envs in range(limit, 0, -1):
        if games_per_member % envs == 0:
            return envs
    return 1


def _member_metrics(
    member: ValidationMember,
    *,
    n_games: int,
    wins: int,
    draws: int,
    margins: list[float],
) -> MemberMetrics:
    losses = n_games - wins - draws
    return MemberMetrics(
        name=member.name,
        bucket=member.bucket,
        weight=member.weight,
        rating=member.rating,
        n_games=n_games,
        wins=wins,
        draws=draws,
        losses=losses,
        win_rate=wins / max(1, n_games),
        draw_rate=draws / max(1, n_games),
        score_rate=(wins + 0.5 * draws) / max(1, n_games),
        mean_margin=float(np.mean(margins)) if margins else 0.0,
        std_margin=float(np.std(margins)) if margins else 0.0,
    )


def _evaluate_member(
    model: OrbitPolicy,
    member: ValidationMember,
    *,
    games_per_member: int,
    num_players: int,
    episode_steps: int,
    ship_speed: float,
    device: str,
    num_envs: int,
    env_backend: str,
    num_workers: int,
    deterministic: bool,
    opponent_deterministic: bool,
    compile_mode: str | None,
) -> MemberMetrics:
    envs = _exact_eval_envs(games_per_member, num_envs)
    wins = draws = 0
    margins: list[float] = []

    if envs == 1 and env_backend == "kaggle":
        if compile_mode is not None:
            warnings.warn(
                "compile_mode is only used by vectorized archive evaluation; "
                "use --num-envs > 1 or a fast env backend to benchmark compiled CUDA inference.",
                RuntimeWarning,
                stacklevel=2,
            )
        opponents = [
            LearnedAgent(
                member.path,
                device=device,
                deterministic=opponent_deterministic,
                compile_mode=compile_mode,
            )
            for _ in range(num_players - 1)
        ]
        for game_idx in range(games_per_member):
            traj = rollout_episode(
                model,
                opponents,
                num_players=num_players,
                episode_steps=episode_steps,
                ship_speed=ship_speed,
                device=device,
                deterministic=deterministic,
                learner_seat=game_idx % num_players,
            )
            wins += int(traj.won)
            draws += int(traj.drawn)
            margins.append(traj.final_score)
        return _member_metrics(
            member,
            n_games=games_per_member,
            wins=wins,
            draws=draws,
            margins=margins,
        )

    agent = LearnedAgent(
        member.path,
        device=device,
        deterministic=opponent_deterministic,
        compile_mode=compile_mode,
    )
    slot = OpponentSlot(name=f"validation:{member.name}", agent=agent)
    opponents_per_env = [[slot] * (num_players - 1) for _ in range(envs)]
    vec = _make_vec(
        env_backend,
        num_envs=envs,
        num_players=num_players,
        episode_steps=episode_steps,
        ship_speed=ship_speed,
        num_workers=num_workers,
    )
    with vec:
        batch_idx = 0
        while len(margins) < games_per_member:
            learner_seats = alternating_learner_seats(
                envs, num_players, offset=batch_idx
            )
            trajs = rollout_episodes_batched(
                model,
                vec,
                opponents_per_env,
                num_players=num_players,
                learner_seat=learner_seats,
                device=device,
                deterministic=deterministic,
                record_trajectories=False,
                compile_mode=compile_mode,
                policy_graph_rows=envs,
            )
            for traj in trajs[: games_per_member - len(margins)]:
                wins += int(traj.won)
                draws += int(traj.drawn)
                margins.append(traj.final_score)
            batch_idx += 1

    return _member_metrics(
        member,
        n_games=games_per_member,
        wins=wins,
        draws=draws,
        margins=margins,
    )


def _weighted_mean(metrics: list[MemberMetrics], attr: str) -> float:
    total = sum(metric.weight for metric in metrics)
    if total <= 0.0:
        return 0.0
    return float(sum(metric.weight * float(getattr(metric, attr)) for metric in metrics) / total)


def _sigmoid(x: float) -> float:
    if x >= 0.0:
        return 1.0 / (1.0 + math.exp(-x))
    z = math.exp(x)
    return z / (1.0 + z)


def solve_fixed_panel_pseudo_elo(
    ratings: list[float],
    weights: list[float],
    observed_score: float,
    *,
    scale: float = 400.0 / math.log(10.0),
    eps: float = 1e-6,
) -> float:
    """Solve `sum_i w_i * sigmoid((R - rating_i) / scale) = observed_score`."""
    if len(ratings) != len(weights) or not ratings:
        raise ValueError("ratings and weights must be non-empty and same length")
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError("pseudo-Elo scale must be positive")
    total_weight = sum(float(w) for w in weights)
    if total_weight <= 0.0:
        raise ValueError("pseudo-Elo weights must have positive total")
    norm = [float(w) / total_weight for w in weights]
    target = min(max(float(observed_score), eps), 1.0 - eps)

    def expected(rating: float) -> float:
        return sum(
            weight * _sigmoid((rating - member_rating) / scale)
            for member_rating, weight in zip(ratings, norm, strict=True)
        )

    low = min(ratings) - 20.0 * scale
    high = max(ratings) + 20.0 * scale
    while expected(low) > target:
        low -= 10.0 * scale
    while expected(high) < target:
        high += 10.0 * scale
    for _ in range(80):
        mid = 0.5 * (low + high)
        if expected(mid) < target:
            low = mid
        else:
            high = mid
    return float(0.5 * (low + high))


def aggregate_archive_metrics(
    manifest: ValidationManifest,
    member_metrics: list[MemberMetrics],
    *,
    pseudo_elo_scale: float = 400.0 / math.log(10.0),
) -> dict[str, Any]:
    by_name = {metric.name: metric for metric in member_metrics}
    scored_members = [member for member in manifest.members if member.weight > 0.0]
    missing = [member.name for member in scored_members if member.name not in by_name]
    if missing:
        raise ValueError(f"missing metrics for validation members: {missing}")

    ordered = [by_name[member.name] for member in scored_members]
    if not ordered:
        raise ValueError("cannot aggregate archive metrics with zero total weight")

    buckets: dict[str, list[MemberMetrics]] = {}
    for metric in ordered:
        buckets.setdefault(metric.bucket, []).append(metric)

    per_bucket: dict[str, dict[str, float | int]] = {}
    for bucket, bucket_metrics in sorted(buckets.items()):
        per_bucket[bucket] = {
            "weight": float(sum(metric.weight for metric in bucket_metrics)),
            "n_games": int(sum(metric.n_games for metric in bucket_metrics)),
            "win_rate": _weighted_mean(bucket_metrics, "win_rate"),
            "draw_rate": _weighted_mean(bucket_metrics, "draw_rate"),
            "score_rate": _weighted_mean(bucket_metrics, "score_rate"),
            "mean_margin": _weighted_mean(bucket_metrics, "mean_margin"),
        }

    bucket_win_rates = [float(row["win_rate"]) for row in per_bucket.values()]
    bucket_margins = [float(row["mean_margin"]) for row in per_bucket.values()]
    weighted_score_rate = _weighted_mean(ordered, "score_rate")

    result: dict[str, Any] = {
        "archive_version": manifest.version,
        "metric_prefix": f"validation/{manifest.version}",
        "created_update": manifest.created_update,
        "n_members": len(scored_members),
        "n_manifest_members": len(manifest.members),
        "n_games": int(sum(metric.n_games for metric in ordered)),
        "weighted_win_rate": _weighted_mean(ordered, "win_rate"),
        "weighted_draw_rate": _weighted_mean(ordered, "draw_rate"),
        "weighted_score_rate": weighted_score_rate,
        "weighted_margin": _weighted_mean(ordered, "mean_margin"),
        "bucket_win_rate_min": min(bucket_win_rates),
        "bucket_win_rate_p10": float(np.percentile(bucket_win_rates, 10)),
        "bucket_margin_min": min(bucket_margins),
        "bucket_margin_p10": float(np.percentile(bucket_margins, 10)),
        "per_bucket": per_bucket,
        "members": {metric.name: asdict(metric) for metric in ordered},
    }

    if all(metric.rating is not None for metric in ordered):
        result["pseudo_elo"] = solve_fixed_panel_pseudo_elo(
            [float(metric.rating) for metric in ordered if metric.rating is not None],
            [metric.weight for metric in ordered],
            weighted_score_rate,
            scale=pseudo_elo_scale,
        )
        result["pseudo_elo_scale"] = pseudo_elo_scale
        result["pseudo_elo_score_rate"] = weighted_score_rate

    return result


def evaluate_archive_ckpt(
    ckpt_path: str | Path,
    manifest: str | Path | ValidationManifest,
    *,
    games_per_member: int = 20,
    num_players: int = 2,
    episode_steps: int = 500,
    ship_speed: float = 6.0,
    device: str = "cpu",
    num_envs: int = 16,
    env_backend: str = "rust",
    num_workers: int = 0,
    compile_mode: str | None = "reduce-overhead",
    deterministic: bool = True,
    opponent_deterministic: bool = True,
    pseudo_elo_scale: float = 400.0 / math.log(10.0),
) -> dict[str, Any]:
    if games_per_member <= 0:
        raise ValueError("games_per_member must be positive")
    if num_envs <= 0:
        raise ValueError("num_envs must be positive")
    if num_players not in {2, 4}:
        raise ValueError("num_players must be 2 or 4")
    if env_backend not in {"kaggle", "numpy", "numpy_mp", "rust"}:
        raise ValueError(f"unknown env_backend: {env_backend!r}")

    archive = (
        load_validation_manifest(manifest)
        if isinstance(manifest, str | Path)
        else manifest
    )
    model = _load_model(ckpt_path, device)
    compile_mode = compile_mode if torch.device(device).type == "cuda" else None

    metrics = [
        _evaluate_member(
            model,
            member,
            games_per_member=games_per_member,
            num_players=num_players,
            episode_steps=episode_steps,
            ship_speed=ship_speed,
            device=device,
            num_envs=num_envs,
            env_backend=env_backend,
            num_workers=num_workers,
            deterministic=deterministic,
            opponent_deterministic=opponent_deterministic,
            compile_mode=compile_mode,
        )
        for member in archive.members
        if member.weight > 0.0
    ]
    result = aggregate_archive_metrics(
        archive,
        metrics,
        pseudo_elo_scale=pseudo_elo_scale,
    )
    result["ckpt"] = str(ckpt_path)
    result["games_per_member"] = games_per_member
    result["num_players"] = num_players
    result["env_backend"] = env_backend
    return result


def _print_human(results: dict[str, Any]) -> None:
    prefix = results["metric_prefix"]
    print(f"{prefix}/weighted_win_rate: {results['weighted_win_rate']:.4f}")
    print(f"{prefix}/weighted_margin: {results['weighted_margin']:.3f}")
    print(f"{prefix}/bucket_win_rate_min: {results['bucket_win_rate_min']:.4f}")
    print(f"{prefix}/bucket_win_rate_p10: {results['bucket_win_rate_p10']:.4f}")
    print(f"{prefix}/bucket_margin_min: {results['bucket_margin_min']:.3f}")
    if "pseudo_elo" in results:
        print(f"{prefix}/pseudo_elo: {results['pseudo_elo']:.1f}")
    for bucket, metrics in results["per_bucket"].items():
        print(
            f"{prefix}/bucket/{bucket}: "
            f"win_rate={metrics['win_rate']:.4f} "
            f"margin={metrics['mean_margin']:.3f} "
            f"games={metrics['n_games']}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--games-per-member", type=int, default=20)
    parser.add_argument("--num-players", type=int, default=2, choices=(2, 4))
    parser.add_argument("--episode-steps", type=int, default=500)
    parser.add_argument("--ship-speed", type=float, default=6.0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--num-envs", type=int, default=16)
    parser.add_argument(
        "--env-backend", choices=("kaggle", "numpy", "numpy_mp", "rust"), default="rust"
    )
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="Evaluate candidate with rollout-style stochastic sampling instead of deterministic deployment actions.",
    )
    parser.add_argument(
        "--opponent-stochastic",
        action="store_true",
        help="Run validation archive members stochastically instead of deterministic deployment actions.",
    )
    parser.add_argument(
        "--compile-mode",
        default="reduce-overhead",
        help="CUDA torch.compile mode for batched policy forwards; use 'none' to disable.",
    )
    parser.add_argument(
        "--pseudo-elo-scale",
        type=float,
        default=400.0 / math.log(10.0),
        help="Scale in sigmoid((R - rating_i) / scale) for fixed-panel pseudo-Elo.",
    )
    parser.add_argument("--json", action="store_true", help="Print full JSON results.")
    args = parser.parse_args()
    compile_mode = None if args.compile_mode.lower() == "none" else args.compile_mode

    results = evaluate_archive_ckpt(
        args.ckpt,
        args.manifest,
        games_per_member=args.games_per_member,
        num_players=args.num_players,
        episode_steps=args.episode_steps,
        ship_speed=args.ship_speed,
        device=args.device,
        num_envs=args.num_envs,
        env_backend=args.env_backend,
        num_workers=args.num_workers,
        compile_mode=compile_mode,
        deterministic=not args.stochastic,
        opponent_deterministic=not args.opponent_stochastic,
        pseudo_elo_scale=args.pseudo_elo_scale,
    )
    if args.json:
        print(json.dumps(results, indent=2, sort_keys=True))
    else:
        _print_human(results)


if __name__ == "__main__":
    main()
