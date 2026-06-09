#!/usr/bin/env python
"""Benchmark scalar vs grouped Muon optimizer steps.

This isolates the optimizer backend from PPO/env work by cloning the real
OrbitPolicy Muon-owned block-matrix shapes, assigning fixed synthetic
gradients, and timing `Muon.step()`.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from time import perf_counter

import torch

from owars.training.config import load_config
from owars.training.muon import Muon
from owars.training.train import _build_model, _split_params


@dataclass
class BenchResult:
    ms_per_step: float
    peak_mb: float


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _make_param_groups(config_path: str, device: torch.device) -> list[dict]:
    cfg = load_config(config_path)
    model = _build_model(cfg).to(device)
    muon_blocks, _adamw_default, _adamw_control, _adamw_head = _split_params(model)

    def clone_params(params: list[torch.nn.Parameter]) -> list[torch.nn.Parameter]:
        return [
            torch.nn.Parameter(torch.randn_like(p, device=device))
            for p in params
        ]

    return [{"params": clone_params(muon_blocks), "lr": cfg.optim.muon_lr}]


def _clone_groups(groups: list[dict]) -> list[dict]:
    return [
        {
            **{k: v for k, v in group.items() if k != "params"},
            "params": [
                torch.nn.Parameter(p.detach().clone())
                for p in group["params"]
            ],
        }
        for group in groups
    ]


def _all_params(groups: list[dict]) -> list[torch.nn.Parameter]:
    return [p for group in groups for p in group["params"]]


def _make_grads(
    params: list[torch.nn.Parameter],
    *,
    steps: int,
    seed: int,
) -> list[list[torch.Tensor]]:
    gen = torch.Generator(device=params[0].device)
    gen.manual_seed(seed)
    return [
        [torch.randn(p.shape, device=p.device, dtype=p.dtype, generator=gen) for p in params]
        for _ in range(steps)
    ]


def _assign_grads(params: list[torch.nn.Parameter], grads: list[torch.Tensor]) -> None:
    for p, g in zip(params, grads, strict=True):
        p.grad = g


def _bench(
    groups: list[dict],
    grads_by_step: list[list[torch.Tensor]],
    *,
    fused: bool,
    warmup: int,
    iters: int,
    momentum: float,
    backend_steps: int,
    normuon: bool,
) -> BenchResult:
    params = _all_params(groups)
    device = params[0].device
    opt = Muon(
        groups,
        lr=groups[0]["lr"],
        momentum=momentum,
        backend_steps=backend_steps,
        normuon=normuon,
        fused=fused,
    )
    total_steps = warmup + iters
    if len(grads_by_step) < total_steps:
        raise ValueError("not enough precomputed gradients")

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for step in range(warmup):
        _assign_grads(params, grads_by_step[step])
        opt.step()
    _sync(device)

    start = perf_counter()
    for step in range(warmup, total_steps):
        _assign_grads(params, grads_by_step[step])
        opt.step()
    _sync(device)

    peak_mb = (
        torch.cuda.max_memory_allocated(device) / 1024 / 1024
        if device.type == "cuda"
        else 0.0
    )
    return BenchResult((perf_counter() - start) * 1000.0 / iters, peak_mb)


def _trajectory_check(
    groups: list[dict],
    grads_by_step: list[list[torch.Tensor]],
    *,
    momentum: float,
    backend_steps: int,
    normuon: bool,
) -> float:
    scalar_groups = _clone_groups(groups)
    fused_groups = _clone_groups(groups)
    scalar_params = _all_params(scalar_groups)
    fused_params = _all_params(fused_groups)
    scalar = Muon(
        scalar_groups,
        lr=scalar_groups[0]["lr"],
        momentum=momentum,
        backend_steps=backend_steps,
        normuon=normuon,
        fused=False,
    )
    fused = Muon(
        fused_groups,
        lr=fused_groups[0]["lr"],
        momentum=momentum,
        backend_steps=backend_steps,
        normuon=normuon,
        fused=True,
    )

    max_abs = 0.0
    for grads in grads_by_step:
        _assign_grads(scalar_params, grads)
        _assign_grads(fused_params, grads)
        scalar.step()
        fused.step()
        for a, b in zip(scalar_params, fused_params, strict=True):
            max_abs = max(max_abs, float((a - b).abs().max().item()))
    return max_abs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/ppo_base.yaml")
    parser.add_argument("--iters", type=int, default=1000)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--backend-steps", type=int, default=5)
    parser.add_argument("--momentum", type=float, default=0.95)
    parser.add_argument("--no-normuon", action="store_true")
    parser.add_argument("--trajectory-steps", type=int, default=20)
    args = parser.parse_args()

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")

    base_groups = _make_param_groups(args.config, device)
    params = _all_params(base_groups)
    shape_counts: dict[tuple[int, ...], int] = {}
    for p in params:
        shape_counts[tuple(p.shape)] = shape_counts.get(tuple(p.shape), 0) + 1
    print(
        f"device={device} params={len(params)} "
        f"grouped_shapes={sum(1 for n in shape_counts.values() if n > 1)}"
    )

    total_steps = args.warmup + args.iters
    grads = _make_grads(params, steps=total_steps, seed=args.seed)
    scalar = _bench(
        _clone_groups(base_groups),
        grads,
        fused=False,
        warmup=args.warmup,
        iters=args.iters,
        momentum=args.momentum,
        backend_steps=args.backend_steps,
        normuon=not args.no_normuon,
    )
    fused = _bench(
        _clone_groups(base_groups),
        grads,
        fused=True,
        warmup=args.warmup,
        iters=args.iters,
        momentum=args.momentum,
        backend_steps=args.backend_steps,
        normuon=not args.no_normuon,
    )
    speedup = scalar.ms_per_step / fused.ms_per_step
    print(f"scalar: {scalar.ms_per_step:.4f} ms/step peak={scalar.peak_mb:.1f} MiB")
    print(f" fused: {fused.ms_per_step:.4f} ms/step peak={fused.peak_mb:.1f} MiB")
    print(f"speedup: {speedup:.2f}x")

    traj_grads = _make_grads(params, steps=args.trajectory_steps, seed=args.seed + 1)
    max_abs = _trajectory_check(
        base_groups,
        traj_grads,
        momentum=args.momentum,
        backend_steps=args.backend_steps,
        normuon=not args.no_normuon,
    )
    print(f"trajectory_max_abs_param_delta_vs_scalar: {max_abs:.3e}")


if __name__ == "__main__":
    main()
