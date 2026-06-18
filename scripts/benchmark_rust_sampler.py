#!/usr/bin/env python
"""Benchmark Rust legality and action materialization hot paths.

By default this exercises the Rust env binding directly on CPU, without policy
forward or CUDA transfers. Use ``--device cuda`` to include the CUDA tensor
staging path used by rollout sampling.

Examples:
    PYTHONPATH=src python scripts/benchmark_rust_sampler.py
    PYTHONPATH=src python scripts/benchmark_rust_sampler.py --num-envs 256 --steps 80
"""

from __future__ import annotations

import argparse
from time import perf_counter

import numpy as np
import torch

from owars.policies.model import PolicyOutput
from owars.training.rust_env import RustVecEnv


def _noop_actions(num_envs: int, num_players: int) -> list[list[list]]:
    return [[[] for _ in range(num_players)] for _ in range(num_envs)]


def _make_rows(num_envs: int, num_players: int) -> list[tuple[int, int]]:
    return [(env_idx, player) for env_idx in range(num_envs) for player in range(num_players)]


def _builtin_actions(
    vec: RustVecEnv,
    rows: list[tuple[int, int]],
    *,
    num_envs: int,
    num_players: int,
    name: str,
) -> list[list[object]]:
    flat = vec.builtin_actions(name, rows, native_actions=True)
    grouped = [[None for _ in range(num_players)] for _ in range(num_envs)]
    for (env_idx, player), actions in zip(rows, flat, strict=True):
        grouped[env_idx][player] = actions
    return grouped


def _advance_env(
    vec: RustVecEnv,
    *,
    steps: int,
    num_players: int,
    setup_policy: str,
) -> None:
    active = list(range(vec.num_envs))
    rows = _make_rows(vec.num_envs, num_players)
    for _ in range(steps):
        if setup_policy == "noop":
            actions = _noop_actions(vec.num_envs, num_players)
        else:
            actions = _builtin_actions(
                vec,
                rows,
                num_envs=vec.num_envs,
                num_players=num_players,
                name=setup_policy,
            )
        vec.step_subset_fast(active, actions)


def _action_fields_from_legal(
    legal: np.ndarray,
    frac: np.ndarray,
) -> np.ndarray:
    has_target = legal.any(axis=-1)
    target_idx = legal.argmax(axis=-1).astype(np.float32)
    fields = np.empty((*has_target.shape, 3), dtype=np.float32)
    fields[..., 0] = has_target.astype(np.float32)
    fields[..., 1] = target_idx
    fields[..., 2] = frac
    return fields


def _dense_from_compact(
    compact: dict[str, np.ndarray],
    *,
    batch: int,
    planets: int,
) -> np.ndarray:
    dense = np.zeros((batch, planets, planets), dtype=bool)
    if compact["row_idx"].size:
        dense[compact["row_idx"], compact["source_idx"]] = compact["mask"]
    return dense


def _bench_call(name: str, warmup: int, iters: int, fn) -> float:
    for _ in range(warmup):
        fn()
    start = perf_counter()
    for _ in range(iters):
        fn()
    elapsed = perf_counter() - start
    ms = elapsed * 1000.0 / max(1, iters)
    print(f"{name}: {ms:.3f} ms/call")
    return ms


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-envs", type=int, default=128)
    parser.add_argument("--num-players", type=int, choices=(2, 4), default=4)
    parser.add_argument("--episode-steps", type=int, default=500)
    parser.add_argument("--ship-speed", type=float, default=6.0)
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda"),
        default="cpu",
        help="Device for the synthetic PolicyOutput passed to RustVecEnv.sample_batch_actions.",
    )
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="Benchmark stochastic action sampling instead of deterministic argmax sampling.",
    )
    parser.add_argument(
        "--setup-policy",
        default="sniper",
        help="Native builtin used to advance to the benchmark state; use 'noop' for no actions.",
    )
    parser.add_argument(
        "--strict-target-legality",
        action="store_true",
        help="Use exact route blocker masks instead of the faster simulator-valid support.",
    )
    args = parser.parse_args()

    with RustVecEnv(
        num_envs=args.num_envs,
        num_players=args.num_players,
        episode_steps=args.episode_steps,
        ship_speed=args.ship_speed,
        random_seed=args.seed,
        strict_target_legality=args.strict_target_legality,
    ) as vec:
        vec.reset()
        _advance_env(
            vec,
            steps=args.steps,
            num_players=args.num_players,
            setup_policy=args.setup_policy,
        )
        rows = _make_rows(args.num_envs, args.num_players)
        feats, _contexts = vec.policy_batch_no_context(
            rows,
            device="cpu",
            include_fleet_targets=True,
        )
        batch, planets = feats.planet_mask.shape
        frac = np.full((batch, planets), 0.5, dtype=np.float32)
        active_fields = np.empty((batch, planets, 2), dtype=np.float32)
        active_fields[..., 0] = frac
        active_fields[..., 1] = 1.0

        legal = vec._core.legal_target_mask_from_state_active_fields(
            rows,
            active_fields,
        )
        fields = _action_fields_from_legal(legal, frac)
        launch_logits = torch.where(
            feats.planet_owned_mask,
            torch.full((batch, planets), 8.0),
            torch.full((batch, planets), -8.0),
        )
        device = torch.device(args.device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise SystemExit("CUDA requested but torch.cuda.is_available() is false")
        policy_out = PolicyOutput(
            launch_logits=launch_logits,
            target_logits=torch.zeros((batch, planets, planets)),
            value=torch.zeros(batch),
            value_logits=torch.zeros(batch, 51),
            planet_owned_mask=feats.planet_owned_mask,
            planet_mask=feats.planet_mask,
            planet_ids=feats.planet_ids,
            action_logit_softcap=8.0,
            fraction_alpha=torch.full((batch, planets), 8.0),
            fraction_beta=torch.full((batch, planets), 2.0),
        )
        if device.type != "cpu":
            policy_out = PolicyOutput(
                launch_logits=policy_out.launch_logits.to(device),
                target_logits=policy_out.target_logits.to(device),
                value=policy_out.value.to(device),
                value_logits=policy_out.value_logits.to(device),
                planet_owned_mask=policy_out.planet_owned_mask.to(device),
                planet_mask=policy_out.planet_mask.to(device),
                planet_ids=policy_out.planet_ids.to(device),
                action_logit_softcap=policy_out.action_logit_softcap,
                launch_log_std=policy_out.launch_log_std,
                fraction_alpha=policy_out.fraction_alpha.to(device),
                fraction_beta=policy_out.fraction_beta.to(device),
                fraction_mean=(
                    None
                    if policy_out.fraction_mean is None
                    else policy_out.fraction_mean.to(device)
                ),
                fraction_log_std=(
                    None
                    if policy_out.fraction_log_std is None
                    else policy_out.fraction_log_std.to(device)
                ),
                launch_prob_floor=policy_out.launch_prob_floor,
            )

        print(
            f"rows={len(rows)} envs={args.num_envs} players={args.num_players} "
            f"planets={planets} steps={args.steps} setup_policy={args.setup_policy} "
            f"device={device.type} deterministic={not args.stochastic}"
        )
        print(
            f"legal_true_frac={float(legal.mean()):.4f} "
            f"launch_frac={float(fields[..., 0].mean()):.4f}"
        )

        def legal_call() -> np.ndarray:
            return vec._core.legal_target_mask_from_state_active_fields(
                rows,
                active_fields,
            )

        def compact_legal_call() -> dict[str, np.ndarray]:
            return vec._core.compact_legal_target_mask_from_state_active_fields(
                rows,
                active_fields,
            )

        def feature_call() -> object:
            return vec.policy_batch_no_context(
                rows,
                device="cpu",
                include_fleet_targets=True,
            )

        def builtin_call() -> object:
            return vec.builtin_actions(
                args.setup_policy,
                rows,
                native_actions=True,
            )

        def materialize_call() -> object:
            return vec._core.materialize_masked_action_fields_from_state(
                rows,
                fields,
                True,
            )

        def combined_call() -> object:
            legal_now = legal_call()
            fields_now = _action_fields_from_legal(legal_now, frac)
            return vec._core.materialize_masked_action_fields_from_state(
                rows,
                fields_now,
                True,
            )

        def sampler_call() -> object:
            return vec.sample_batch_actions(
                policy_out,
                rows,
                deterministic=not args.stochastic,
                native_actions=True,
            )

        record_rows = [
            idx for idx, (_env_idx, player) in enumerate(rows) if player == 0
        ]
        record_source_mask = (
            feats.planet_owned_mask[record_rows] & feats.planet_mask[record_rows]
        ).numpy()
        def recorded_sampler_call() -> object:
            return vec.sample_batch_with_records(
                policy_out,
                rows,
                deterministic=not args.stochastic,
                record_rows=record_rows,
                record_source_mask=record_source_mask,
                native_actions=True,
                compute_log_prob=False,
            )

        active_envs = list(range(args.num_envs))

        def recorded_enqueue_step_call() -> object:
            vec.sample_batch_with_records(
                policy_out,
                rows,
                deterministic=not args.stochastic,
                record_rows=record_rows,
                record_source_mask=record_source_mask,
                native_actions=True,
                enqueue_actions=True,
                compute_log_prob=False,
                compact_legal_records=True,
            )
            return vec.step_subset_pending_actions(active_envs, [], [], [])

        feature_ms = _bench_call("policy_batch", args.warmup, args.iters, feature_call)
        builtin_ms = 0.0
        if args.setup_policy != "noop":
            builtin_ms = _bench_call(
                "builtin_actions",
                args.warmup,
                args.iters,
                builtin_call,
            )
        legal_ms = _bench_call("legal_mask", args.warmup, args.iters, legal_call)
        compact_legal_ms = 0.0
        if hasattr(vec._core, "compact_legal_target_mask_from_state_active_fields"):
            compact_legal_ms = _bench_call(
                "compact_legal_mask_rust",
                args.warmup,
                args.iters,
                compact_legal_call,
            )
            compact = compact_legal_call()
            compact_dense = _dense_from_compact(
                compact,
                batch=batch,
                planets=planets,
            )
            if not np.array_equal(compact_dense, legal):
                raise AssertionError("compact legal mask did not match dense legal mask")
        materialize_ms = _bench_call(
            "materialize",
            args.warmup,
            args.iters,
            materialize_call,
        )
        combined_ms = _bench_call("combined", args.warmup, args.iters, combined_call)
        sampler_ms = _bench_call(
            "sample_batch_actions",
            args.warmup,
            args.iters,
            sampler_call,
        )
        recorded_label = (
            "sample_batch_with_records_deferred"
            if device.type == "cuda"
            else "sample_batch_with_records_native_recorded"
        )
        recorded_sampler_ms = _bench_call(
            recorded_label,
            args.warmup,
            args.iters,
            recorded_sampler_call,
        )
        recorded_enqueue_step_ms = _bench_call(
            "sample_batch_with_records_enqueue_step",
            args.warmup,
            args.iters,
            recorded_enqueue_step_call,
        )
        print(
            "summary "
            f"policy_batch_ms={feature_ms:.3f} builtin_ms={builtin_ms:.3f} "
            f"legal_ms={legal_ms:.3f} compact_legal_rust_ms={compact_legal_ms:.3f} "
            f"materialize_ms={materialize_ms:.3f} "
            f"combined_ms={combined_ms:.3f} sampler_ms={sampler_ms:.3f} "
            f"{recorded_label}_ms={recorded_sampler_ms:.3f} "
            f"recorded_enqueue_step_ms={recorded_enqueue_step_ms:.3f}"
        )


if __name__ == "__main__":
    main()
