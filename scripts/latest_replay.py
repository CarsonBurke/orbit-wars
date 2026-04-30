#!/usr/bin/env python
"""Render one replay from the newest training checkpoint.

Default behavior:
  1. Pick the newest TensorBoard run directory under runs/<name>/<timestamp>.
  2. Pick final.pt under checkpoints/<name>/, falling back to the newest
     non-init checkpoint if the run has not written final weights yet.
  3. Run final learned weights against themselves in the official Kaggle
     environment.
  4. Write an HTML replay under the run's eval_replays/ directory.
"""

from __future__ import annotations

import argparse
import re
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import torch

from owars.agents import HeuristicAgent, random_agent, sniper_agent
from owars.agents.learned import LearnedAgent


@dataclass(frozen=True)
class RunChoice:
    name: str
    path: Path | None


def _latest_run(runs_root: Path) -> RunChoice:
    event_files = list(runs_root.glob("*/*/events.out.tfevents.*"))
    if not event_files:
        return _latest_checkpoint_run(Path("checkpoints"))
    newest = max(event_files, key=lambda p: p.stat().st_mtime)
    run_dir = newest.parent
    return RunChoice(name=run_dir.parent.name, path=run_dir)


def _latest_checkpoint_run(ckpt_root: Path) -> RunChoice:
    candidates = [
        p
        for p in ckpt_root.glob("*/*.pt")
        if p.name != "snapshot_init.pt" and p.parent.is_dir()
    ]
    if not candidates:
        raise FileNotFoundError("no checkpoints found")
    newest = max(candidates, key=lambda p: p.stat().st_mtime)
    return RunChoice(name=newest.parent.name, path=None)


def _snapshot_num(path: Path) -> int:
    match = re.fullmatch(r"snapshot_(\d+)\.pt", path.name)
    return int(match.group(1)) if match else -1


def _latest_checkpoint(ckpt_root: Path, run_name: str) -> Path:
    run_dir = ckpt_root / run_name
    if not run_dir.exists():
        raise FileNotFoundError(f"checkpoint directory not found: {run_dir}")
    final = run_dir / "final.pt"
    if final.is_file():
        return final
    candidates = [
        p for p in run_dir.glob("*.pt")
        if p.name != "snapshot_init.pt" and p.is_file()
    ]
    if not candidates:
        raise FileNotFoundError(f"no usable checkpoints found in {run_dir}")
    return max(candidates, key=lambda p: (p.stat().st_mtime, _snapshot_num(p)))


def _agent_factory(
    name: str,
    ckpt: Path,
    *,
    device: str,
    deterministic: bool,
) -> Callable[[Any], list[list]]:
    if name == "learned":
        return LearnedAgent(
            ckpt,
            device=device,
            deterministic=deterministic,
        )
    if name == "heuristic":
        return HeuristicAgent()
    if name in {"sniper", "bot"}:
        return sniper_agent
    if name == "random":
        return random_agent
    raise ValueError(f"unknown agent: {name!r}")


def _kaggle_agent(agent: Callable[[Any], list[list]]) -> Callable[..., list[list]]:
    def wrapped(obs: Any, *_args: Any) -> list[list]:
        return agent(obs)

    return wrapped


def _default_out(run: RunChoice, ckpt: Path, opponent: str) -> Path:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    base = run.path / "eval_replays" if run.path is not None else Path("replays")
    return base / f"{ckpt.stem}_vs_{opponent}_{stamp}.html"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-root", type=Path, default=Path("runs"))
    parser.add_argument("--ckpt-root", type=Path, default=Path("checkpoints"))
    parser.add_argument("--run", default=None, help="Run name under checkpoints/")
    parser.add_argument("--ckpt", type=Path, default=None)
    parser.add_argument(
        "--opponent",
        choices=("heuristic", "sniper", "bot", "random", "learned"),
        default="learned",
        help="Opponent agent. 'bot' is an alias for the competition sniper starter bot.",
    )
    parser.add_argument("--opponent-ckpt", type=Path, default=None)
    parser.add_argument("--num-players", type=int, choices=(2, 4), default=2)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--ship-speed", type=float, default=6.0)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--stochastic", action="store_true")
    parser.add_argument("--out", type=Path, default=None)
    return parser


def main() -> None:
    parser = _parser()
    args = parser.parse_args()

    from kaggle_environments import make  # type: ignore[import-not-found]

    run = (
        RunChoice(name=args.run, path=None)
        if args.run is not None
        else _latest_run(args.runs_root)
    )
    ckpt = args.ckpt or _latest_checkpoint(args.ckpt_root, run.name)
    opponent_ckpt = args.opponent_ckpt or ckpt
    out = args.out or _default_out(run, ckpt, args.opponent)
    out.parent.mkdir(parents=True, exist_ok=True)

    learned = _agent_factory(
        "learned",
        ckpt,
        device=args.device,
        deterministic=not args.stochastic,
    )
    opponent = _agent_factory(
        args.opponent,
        opponent_ckpt,
        device=args.device,
        deterministic=not args.stochastic,
    )

    env = make(
        "orbit_wars",
        configuration={
            "episodeSteps": args.steps,
            "shipSpeed": args.ship_speed,
            **({} if args.seed is None else {"randomSeed": args.seed}),
        },
        debug=True,
    )
    env.run([_kaggle_agent(learned), *([_kaggle_agent(opponent)] * (args.num_players - 1))])
    rewards = [float(s.reward or 0.0) for s in env.steps[-1]]
    statuses = [str(s.status) for s in env.steps[-1]]
    out.write_text(env.render(mode="html"))

    print(f"run: {run.name}")
    if run.path is not None:
        print(f"run_dir: {run.path}")
    print(f"ckpt: {ckpt}")
    print(f"opponent: {args.opponent}")
    print(f"rewards: {rewards}")
    print(f"statuses: {statuses}")
    print(f"replay: {out.resolve()}")


if __name__ == "__main__":
    main()
