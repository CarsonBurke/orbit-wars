#!/usr/bin/env python
"""Benchmark PPO rollout collection without running a PPO update.

Examples:
    PYTHONPATH=src python scripts/benchmark_rollout.py --config configs/ppo_base.yaml --num-envs 64
    PYTHONPATH=src python scripts/benchmark_rollout.py --config configs/ppo_base.yaml --num-envs 128 --sample-detail-timing
"""

from __future__ import annotations

import argparse
import random
from contextlib import ExitStack
from dataclasses import dataclass
from time import perf_counter
from types import SimpleNamespace

import torch

from owars.policies.model import normalize_matrices, restore_fp32_params
from owars.training.config import load_config
from owars.training.league import (
    LEARNER_NAME,
    FixedOpponentPool,
    OpponentSlot,
    _copy_model_without_compile_caches,
)
from owars.training.train import (
    _build_model,
    _build_training_vec,
    _format_episode_counts,
    _policy_compile_rows_for_sampled_rollout,
    _rollout_compile_mode_for_model,
    _train_num_players,
    _training_vec_counts,
    set_seed,
)
from owars.training.vec_rollout import alternating_learner_seats, rollout_episodes_batched


def _current_only_opponents(num_envs: int, num_players: int) -> list[list[OpponentSlot]]:
    current = OpponentSlot(name=LEARNER_NAME, agent=None)
    return [[current for _ in range(num_players - 1)] for _ in range(num_envs)]


def _fixed_opponents(
    pool: FixedOpponentPool,
    *,
    num_envs: int,
    num_players: int,
) -> list[list[OpponentSlot]]:
    return [pool.sample(num_players - 1) for _ in range(num_envs)]


@dataclass
class _SyntheticLearnedPanel:
    active: list[OpponentSlot]
    historical: list[OpponentSlot]


def _synthetic_learned_panel(
    cfg,
    model,
    *,
    device: torch.device,
    separate_models: bool,
    compile_snapshots: bool,
) -> _SyntheticLearnedPanel:

    def snapshot_agent() -> SimpleNamespace:
        snapshot_model = _copy_model_without_compile_caches(model).eval() if separate_models else model
        for param in snapshot_model.parameters():
            param.requires_grad_(False)
        return SimpleNamespace(
            model=snapshot_model,
            device=str(device),
            deterministic=False,
            compile_mode=_rollout_compile_mode_for_model(model, cfg)
            if compile_snapshots
            else None,
        )

    active = [
        OpponentSlot(name=f"synthetic_active:{idx}", agent=snapshot_agent())
        for idx in range(max(1, int(cfg.opponents.active_sample_panel_size)))
    ]
    historical = [
        OpponentSlot(name=f"synthetic_historical:{idx}", agent=snapshot_agent())
        for idx in range(max(1, int(cfg.opponents.historical_sample_panel_size)))
    ]
    return _SyntheticLearnedPanel(active=active, historical=historical)


def _configured_synthetic_opponents(
    cfg,
    panel: _SyntheticLearnedPanel,
    *,
    rng: random.Random,
    num_envs: int,
    num_players: int,
) -> list[list[OpponentSlot]]:
    current = OpponentSlot(name=LEARNER_NAME, agent=None)
    sources = [
        ("current", float(cfg.opponents.current_learner_prob)),
        ("active", float(cfg.opponents.active_pool_prob)),
        ("historical", float(cfg.opponents.historical_archive_prob)),
    ]
    total = sum(max(0.0, weight) for _name, weight in sources)
    if total <= 0.0:
        return _current_only_opponents(num_envs, num_players)

    def sample_slot() -> OpponentSlot:
        draw = rng.random() * total
        acc = 0.0
        for name, weight in sources:
            acc += max(0.0, weight)
            if draw > acc:
                continue
            if name == "current":
                return current
            if name == "active":
                return rng.choice(panel.active)
            return rng.choice(panel.historical)
        return current

    return [[sample_slot() for _ in range(num_players - 1)] for _ in range(num_envs)]


def _print_timings(timings: dict[str, float]) -> None:
    for key in sorted(timings):
        print(f"rollout_detail/{key}: {timings[key]:.6f}")


def _sync_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--num-envs", type=int, default=None)
    parser.add_argument("--episode-steps", type=int, default=None)
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument(
        "--warmup-iterations",
        type=int,
        default=1,
        help="Rollout iterations to run before timed measurement.",
    )
    parser.add_argument(
        "--opponents",
        choices=("configured", "current", "fixed"),
        default="configured",
        help=(
            "Opponent source. 'configured' uses synthetic learned snapshots with "
            "the config's current/active/historical probabilities; 'current' "
            "stress-tests all-current self-play rollout mechanics."
        ),
    )
    parser.add_argument("--rollout-detail-timing", action="store_true")
    parser.add_argument("--sample-detail-timing", action="store_true")
    parser.add_argument(
        "--active-panel-size",
        type=int,
        default=None,
        help="Override opponents.active_sample_panel_size for configured synthetic opponents.",
    )
    parser.add_argument(
        "--historical-panel-size",
        type=int,
        default=None,
        help="Override opponents.historical_sample_panel_size for configured synthetic opponents.",
    )
    parser.add_argument(
        "--snapshot-compile-rows",
        type=int,
        default=None,
        help="Override rollout.snapshot_compile_rows for learned snapshot forwards.",
    )
    parser.add_argument(
        "--shared-synthetic-snapshot-model",
        action="store_true",
        help=(
            "Use one model object for all synthetic learned snapshots. Faster to "
            "benchmark, but less representative of real snapshot cache fragmentation."
        ),
    )
    parser.add_argument(
        "--disable-synthetic-snapshot-compile",
        action="store_true",
        help="Benchmark configured learned snapshots with eager snapshot forwards.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.num_envs is not None:
        cfg.rollout.num_envs = int(args.num_envs)
    if args.episode_steps is not None:
        cfg.game.episode_steps = int(args.episode_steps)
    if args.active_panel_size is not None:
        cfg.opponents.active_sample_panel_size = int(args.active_panel_size)
    if args.historical_panel_size is not None:
        cfg.opponents.historical_sample_panel_size = int(args.historical_panel_size)
    if args.snapshot_compile_rows is not None:
        cfg.rollout.snapshot_compile_rows = int(args.snapshot_compile_rows)
    cfg.rollout.detail_timing = bool(args.rollout_detail_timing or args.sample_detail_timing)
    cfg.rollout.sample_detail_timing = bool(args.sample_detail_timing)

    set_seed(cfg.run.seed)
    if cfg.run.device != "cuda":
        raise ValueError("benchmark_rollout.py expects a CUDA training config")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    device = torch.device("cuda")
    torch.set_float32_matmul_precision("high")
    model = _build_model(cfg).to(device)
    model.bfloat16()
    restore_fp32_params(model)
    normalize_matrices(model)

    train_num_players = _train_num_players(cfg)
    total_rollout_s = 0.0
    total_games = 0
    aggregate_timings: dict[str, float] = {}

    with ExitStack() as stack:
        vecs = {
            num_players: stack.enter_context(_build_training_vec(cfg, num_players, num_envs))
            for num_players, num_envs in _training_vec_counts(cfg).items()
            if num_envs > 0
        }
        for vec in vecs.values():
            vec.set_recording(False)

        timed_iterations = max(1, int(args.iterations))
        warmup_iterations = max(0, int(args.warmup_iterations))
        format_rng = random.Random(cfg.run.seed + 0x5E1F)
        opponent_rng = random.Random(cfg.run.seed)
        synthetic_panel = (
            _synthetic_learned_panel(
                cfg,
                model,
                device=device,
                separate_models=not args.shared_synthetic_snapshot_model,
                compile_snapshots=not args.disable_synthetic_snapshot_compile,
            )
            if args.opponents == "configured"
            else None
        )
        for phase, iterations in (("warmup", warmup_iterations), ("timed", timed_iterations)):
            fixed_pool = (
                FixedOpponentPool(cfg.opponents.fixed_opponents, rng=random.Random(cfg.run.seed))
                if args.opponents == "fixed"
                else None
            )
            timed = phase == "timed"
            for iteration in range(iterations):
                iteration_games = cfg.rollout.num_envs * cfg.rollout.games_per_env_per_update
                episode_counts = _format_episode_counts(
                    iteration_games,
                    train_num_players,
                    format_rng,
                )
                episode_cursor = iteration * iteration_games
                format_order = list(train_num_players)
                format_rng.shuffle(format_order)
                for num_players in format_order:
                    remaining = episode_counts.get(num_players, 0)
                    vec = vecs[num_players]
                    while remaining > 0:
                        rollout_envs = min(vec.num_envs, remaining)
                        remaining -= rollout_envs
                        if args.opponents == "fixed":
                            assert fixed_pool is not None
                            opponents_per_env = _fixed_opponents(
                                fixed_pool,
                                num_envs=rollout_envs,
                                num_players=num_players,
                            )
                        elif args.opponents == "current":
                            opponents_per_env = _current_only_opponents(
                                rollout_envs,
                                num_players,
                            )
                        else:
                            assert synthetic_panel is not None
                            opponents_per_env = _configured_synthetic_opponents(
                                cfg,
                                synthetic_panel,
                                rng=opponent_rng,
                                num_envs=rollout_envs,
                                num_players=num_players,
                            )
                        learner_seats = alternating_learner_seats(
                            rollout_envs,
                            num_players,
                            offset=episode_cursor,
                        )
                        policy_graph_rows = _policy_compile_rows_for_sampled_rollout(
                            opponents_per_env,
                            learner_seats=learner_seats,
                            num_players=num_players,
                        )
                        timings = (
                            {}
                            if cfg.rollout.detail_timing or cfg.rollout.sample_detail_timing
                            else None
                        )
                        sample_timings = timings if cfg.rollout.sample_detail_timing else None
                        _sync_cuda(device)
                        start = perf_counter()
                        trajs = rollout_episodes_batched(
                            model,
                            vec,
                            opponents_per_env,
                            num_players=num_players,
                            learner_seat=learner_seats,
                            device=str(device),
                            reward_cfg=cfg.reward,
                            compile_mode=_rollout_compile_mode_for_model(model, cfg),
                            compile_fleet_width=cfg.rollout.compile_fleet_width,
                            policy_graph_rows=policy_graph_rows,
                            snapshot_compile_rows=cfg.rollout.snapshot_compile_rows,
                            defer_log_prob=True,
                            chunk_records=True,
                            timings=timings,
                            sample_timings=sample_timings,
                        )
                        _sync_cuda(device)
                        elapsed = perf_counter() - start
                        if timed:
                            total_rollout_s += elapsed
                            total_games += len(trajs)
                        if timed and timings is not None:
                            for key, value in timings.items():
                                aggregate_timings[key] = aggregate_timings.get(key, 0.0) + value
                        episode_cursor += rollout_envs
                        print(
                            f"{phase}={iteration} players={num_players} games={len(trajs)} "
                            f"rollout_s={elapsed:.6f} games_per_s={len(trajs) / max(elapsed, 1e-9):.2f}"
                        )

    print(
        f"summary games={total_games} rollout_s={total_rollout_s:.6f} "
        f"games_per_s={total_games / max(total_rollout_s, 1e-9):.2f}"
    )
    if aggregate_timings:
        _print_timings(aggregate_timings)


if __name__ == "__main__":
    main()
