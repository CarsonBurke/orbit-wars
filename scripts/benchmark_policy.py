#!/usr/bin/env python
"""Benchmark OrbitPolicy CUDA forward latency."""

from __future__ import annotations

import argparse
from time import perf_counter

import torch

from owars.policies.config import OrbitPolicyConfig
from owars.policies.features import encode_raw_observations
from owars.policies.model import OrbitPolicy, restore_fp32_params
from owars.training.rust_env import RustVecEnv


def _sync() -> None:
    torch.cuda.synchronize()


def _group_actions(
    active: list[int],
    rows: list[tuple[int, int]],
    flat_actions: list[object],
    *,
    num_players: int,
) -> list[list[object]]:
    row_to_active = {env_idx: pos for pos, env_idx in enumerate(active)}
    actions = [[[] for _ in range(num_players)] for _ in active]
    for (env_idx, player), acts in zip(rows, flat_actions, strict=True):
        actions[row_to_active[env_idx]][player] = acts
    return actions


def _make_raw_observations(
    batch_size: int,
    env_step: int,
    *,
    num_players: int,
    setup_policy: str,
) -> list[object]:
    with RustVecEnv(
        num_envs=batch_size,
        num_players=num_players,
        episode_steps=500,
        ship_speed=6.0,
        random_seed=123,
    ) as vec:
        vec.reset()
        active = list(range(batch_size))
        for _ in range(env_step):
            if not active:
                break
            rows = [
                (env_idx, player)
                for env_idx in active
                for player in range(num_players)
            ]
            if setup_policy == "noop":
                actions = [[[] for _ in range(num_players)] for _ in active]
            else:
                flat = vec.builtin_actions(setup_policy, rows, native_actions=True)
                actions = _group_actions(
                    active,
                    rows,
                    flat,
                    num_players=num_players,
                )
            result = vec.step_subset_fast(active, actions)
            active = [env_idx for env_idx in active if not bool(result[env_idx][1])]
        return vec.observations([(env_idx, 0) for env_idx in range(batch_size)])


def _make_features(
    raw: list[object],
    *,
    include_fleet_targets: bool,
) -> tuple[object, float]:
    for _ in range(3):
        encode_raw_observations(
            raw,
            device="cuda",
            pin_memory=False,
            include_fleet_targets=include_fleet_targets,
        )
    _sync()
    start = perf_counter()
    feats = encode_raw_observations(
        raw,
        device="cuda",
        pin_memory=False,
        include_fleet_targets=include_fleet_targets,
    )
    _sync()
    return feats, (perf_counter() - start) * 1000.0


def _bench(
    backend: str,
    feats: object,
    *,
    dim: int,
    ff_dim: int,
    depth: int,
    n_heads: int,
    n_kv_heads: int | None,
    num_fleet_latents: int,
    warmup: int,
    iters: int,
    compile_model: bool,
) -> tuple[float, int]:
    cfg = OrbitPolicyConfig(
        dim=dim,
        ff_dim=ff_dim,
        depth=depth,
        n_heads=n_heads,
        n_kv_heads=n_kv_heads,
        encoder_backend=backend,  # type: ignore[arg-type]
        num_fleet_latents=num_fleet_latents,
    )
    model = OrbitPolicy(cfg).cuda().eval()
    model.bfloat16()
    restore_fp32_params(model)
    if compile_model:
        model = torch.compile(model, mode="reduce-overhead")  # type: ignore[assignment]
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        _h, full_mask, _pm, _fm, _rope, _planet_slice, _p, _f = (
            model._embed_tokens(feats)
        )
    effective_tokens = int(full_mask.shape[1])
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for _ in range(warmup):
            model(feats)
        _sync()
        start = perf_counter()
        for _ in range(iters):
            model(feats)
        _sync()
    return (perf_counter() - start) * 1000.0 / iters, effective_tokens


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--env-step", type=int, default=50)
    parser.add_argument("--num-players", type=int, choices=(2, 4), default=4)
    parser.add_argument(
        "--setup-policy",
        default="sniper",
        help="Native builtin policy used to advance benchmark states; use 'noop' for empty early boards.",
    )
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--ff-dim", type=int, default=256)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument(
        "--n-kv-heads",
        type=int,
        default=None,
        help="KV heads for GQA/MQA; omit for ordinary MHA.",
    )
    parser.add_argument("--num-fleet-latents", type=int, default=64)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument(
        "--backend",
        choices=("dense", "fleet_latent", "destination_conditioned", "all"),
        default="all",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("benchmark_policy.py requires CUDA")
    backends = (
        ("dense", "fleet_latent", "destination_conditioned")
        if args.backend == "all"
        else (args.backend,)
    )
    raw = _make_raw_observations(
        args.batch_size,
        args.env_step,
        num_players=args.num_players,
        setup_policy=args.setup_policy,
    )
    print(
        f"batch={args.batch_size} players={args.num_players} "
        f"env_step={args.env_step} setup_policy={args.setup_policy} "
        f"compiled={args.compile}"
    )
    for backend in backends:
        feats, feature_ms = _make_features(
            raw,
            include_fleet_targets=backend == "destination_conditioned",
        )
        real_tokens = (
            feats.planet_mask.sum(dim=1) + feats.fleet_mask.sum(dim=1) + 3
        ).float()
        active_fleet_width = int(feats.fleet_mask.sum(dim=1).max().item())
        inbound_nonzero = (
            0
            if feats.planet_inbound_feats is None
            else int((feats.planet_inbound_feats.abs().sum(dim=-1) > 0).sum().item())
        )
        ms, effective_tokens = _bench(
            backend,
            feats,
            dim=args.dim,
            ff_dim=args.ff_dim,
            depth=args.depth,
            n_heads=args.n_heads,
            n_kv_heads=args.n_kv_heads,
            num_fleet_latents=args.num_fleet_latents,
            warmup=args.warmup,
            iters=args.iters,
            compile_model=args.compile,
        )
        print(
            f"{backend:>24}: feature={feature_ms:.3f} ms "
            f"forward={ms:.3f} ms effective_tokens={effective_tokens} "
            f"real_tokens_mean={real_tokens.mean().item():.1f} "
            f"real_tokens_max={int(real_tokens.max().item())} "
            f"fleet_width_max={active_fleet_width} "
            f"inbound_nonzero={inbound_nonzero}"
        )


if __name__ == "__main__":
    main()
