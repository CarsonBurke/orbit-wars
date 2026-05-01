#!/usr/bin/env python
"""Benchmark OrbitPolicy CUDA forward latency."""

from __future__ import annotations

import argparse
from time import perf_counter

import torch

from owars.policies.config import OrbitPolicyConfig
from owars.policies.features import encode_raw_observations
from owars.policies.model import OrbitPolicy, restore_fp32_params
from owars.training.numpy_env import NumpyVecEnv


def _sync() -> None:
    torch.cuda.synchronize()


def _make_features(batch_size: int, env_step: int) -> object:
    vec = NumpyVecEnv(
        num_envs=batch_size,
        num_players=2,
        episode_steps=500,
        ship_speed=6.0,
        random_seed=123,
    )
    states = vec.reset()
    active = list(range(batch_size))
    for _ in range(env_step):
        result = vec.step_subset(active, [[[], []] for _ in active])
        states = [result[i][0] for i in active]
    raw = [state[0]["observation"] for state in states]
    return encode_raw_observations(raw, device="cuda", pin_memory=False)


def _bench(
    backend: str,
    feats: object,
    *,
    dim: int,
    ff_dim: int,
    depth: int,
    n_heads: int,
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
        encoder_backend=backend,  # type: ignore[arg-type]
        num_fleet_latents=num_fleet_latents,
    )
    model = OrbitPolicy(cfg).cuda().eval()
    model.bfloat16()
    restore_fp32_params(model)
    if compile_model:
        model = torch.compile(model, mode="reduce-overhead")  # type: ignore[assignment]
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for _ in range(warmup):
            model(feats)
        _sync()
        start = perf_counter()
        for _ in range(iters):
            model(feats)
        _sync()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        _h, full_mask, _pm, _fm, _rope, _planet_slice, _p, _f = (
            model._embed_tokens(feats)
        )
    return (perf_counter() - start) * 1000.0 / iters, int(full_mask.shape[1])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--env-step", type=int, default=50)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--ff-dim", type=int, default=256)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--num-fleet-latents", type=int, default=64)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument(
        "--backend",
        choices=("dense", "fleet_latent", "all"),
        default="all",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("benchmark_policy.py requires CUDA")
    feats = _make_features(args.batch_size, args.env_step)
    real_tokens = (
        feats.planet_mask.sum(dim=1) + feats.fleet_mask.sum(dim=1) + 2
    ).float()
    backends = (
        ("dense", "fleet_latent")
        if args.backend == "all"
        else (args.backend,)
    )
    print(
        f"batch={args.batch_size} env_step={args.env_step} "
        f"tokens_mean={real_tokens.mean().item():.1f} "
        f"tokens_max={int(real_tokens.max().item())} "
        f"compiled={args.compile}"
    )
    for backend in backends:
        ms, effective_tokens = _bench(
            backend,
            feats,
            dim=args.dim,
            ff_dim=args.ff_dim,
            depth=args.depth,
            n_heads=args.n_heads,
            num_fleet_latents=args.num_fleet_latents,
            warmup=args.warmup,
            iters=args.iters,
            compile_model=args.compile,
        )
        print(f"{backend:>12}: {ms:.3f} ms/forward effective_tokens={effective_tokens}")


if __name__ == "__main__":
    main()
