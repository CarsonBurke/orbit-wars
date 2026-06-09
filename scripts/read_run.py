#!/usr/bin/env python
"""Read training-health scalars from a run's TensorBoard events.

Examples:
    python scripts/read_run.py                  # latest run, curated health snapshot
    python scripts/read_run.py --trace          # trajectory (strided) for each tag
    python scripts/read_run.py --tags kl --trace # only tags matching /kl/
    python scripts/read_run.py --run runs/ppo_sniper/20260609-001436 --all
    python scripts/read_run.py --watch 30        # re-read every 30s until interrupted

Defaults to the most recently modified `runs/*/*` directory. The curated tag
set is the handful of scalars that actually diagnose PPO trust-region health
(KL, entropy, ratio-clip, the shared-trunk grad balance, win-rate, margin).
"""

from __future__ import annotations

import argparse
import re
import time
from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# Curated PPO/nGPT training-health scalars, in display order. Anything matching
# `--tags <regex>` overrides this; `--all` prints every scalar tag.
HEALTH_TAGS = (
    "kl/approx",
    "kl/log_ratio_abs_max",
    "kl/log_ratio_abs_mean",
    "kl/ratio_clip_frac",
    "kl/ratio_clip_frac_high",
    "losses/entropy",
    "losses/policy_loss",
    "losses/value_loss",
    "losses/actor_shared_grad_norm",
    "losses/critic_shared_grad_norm",
    "losses/actor_shared_raw_grad_norm",
    "losses/critic_shared_raw_grad_norm",
    "rollout/win_rate",
    "rollout/margin",
)


def latest_run(root: Path) -> Path:
    runs = [p for p in root.glob("runs/*/*") if p.is_dir()]
    if not runs:
        raise SystemExit(f"no run directories under {root / 'runs'}")
    return max(runs, key=lambda p: p.stat().st_mtime)


def load(run_dir: Path) -> EventAccumulator:
    acc = EventAccumulator(str(run_dir), size_guidance={"scalars": 0})
    acc.Reload()
    return acc


def select_tags(acc: EventAccumulator, args: argparse.Namespace) -> list[str]:
    available = acc.Tags().get("scalars", [])
    if args.all:
        return sorted(available)
    if args.tags:
        pattern = re.compile(args.tags, re.IGNORECASE)
        return [t for t in available if pattern.search(t)]
    return [t for t in HEALTH_TAGS if t in available]


def max_step(acc: EventAccumulator, tags: list[str]) -> int:
    best = 0
    for tag in tags:
        events = acc.Scalars(tag)
        if events:
            best = max(best, events[-1].step)
    return best


def report(acc: EventAccumulator, args: argparse.Namespace) -> None:
    tags = select_tags(acc, args)
    if not tags:
        print("  (no matching scalar tags)")
        return
    width = max(len(t) for t in tags)
    print(f"max step: {max_step(acc, tags)}  |  tags: {len(tags)}")
    for tag in tags:
        events = acc.Scalars(tag)
        if not events:
            continue
        if args.trace:
            points = events[:: max(1, args.every)]
            if points and points[-1].step != events[-1].step:
                points = [*points, events[-1]]
            trail = " ".join(f"{e.step}:{e.value:+.3f}" for e in points)
            print(f"  {tag:<{width}}  {trail}")
        else:
            last = events[-1]
            print(f"  {tag:<{width}}  step={last.step:>4}  {last.value:+.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=None, help="run dir (default: latest under runs/)")
    parser.add_argument("--tags", default=None, help="regex; show only matching scalar tags")
    parser.add_argument("--all", action="store_true", help="show every scalar tag")
    parser.add_argument("--trace", action="store_true", help="print strided trajectory per tag")
    parser.add_argument("--every", type=int, default=4, help="trajectory stride for --trace")
    parser.add_argument("--watch", type=float, default=None, help="re-read every N seconds")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    run_dir = args.run if args.run else latest_run(root)
    if not run_dir.is_absolute():
        run_dir = root / run_dir
    print(f"run: {run_dir}")

    while True:
        report(load(run_dir), args)
        if args.watch is None:
            return
        print("-" * 60, flush=True)
        time.sleep(args.watch)


if __name__ == "__main__":
    main()
