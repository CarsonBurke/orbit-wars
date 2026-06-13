#!/usr/bin/env python
"""Render one replay from the newest training checkpoint.

Default behavior:
  1. Pick the newest TensorBoard run directory under runs/<name>/<timestamp>.
  2. Pick the newest checkpoint under checkpoints/<name>/, ignoring
     snapshot_init.pt. A stale final.pt from an older architecture should not
     beat fresh snapshots from the active run.
  3. Run the selected learned weights against themselves in the official Kaggle
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

from owars.agents import HeuristicAgent, random_agent, sniper_agent
from owars.agents.learned import LearnedAgent
from owars.agents.sac_agent import SACAgent


@dataclass(frozen=True)
class RunChoice:
    name: str
    path: Path | None


def _latest_run(
    runs_root: Path,
    ckpt_root: Path,
    *,
    allow_stale_fallback: bool = False,
) -> RunChoice:
    event_files = sorted(
        runs_root.glob("*/*/events.out.tfevents.*"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    newest_missing: RunChoice | None = None
    seen: set[str] = set()
    for event_file in event_files:
        run_dir = event_file.parent
        run_name = run_dir.parent.name
        if run_name in seen:
            continue
        seen.add(run_name)
        if _usable_checkpoints(ckpt_root / run_name):
            return RunChoice(name=run_name, path=run_dir)
        if newest_missing is None:
            newest_missing = RunChoice(name=run_name, path=run_dir)
        if not allow_stale_fallback:
            break
    if newest_missing is not None and not allow_stale_fallback:
        raise FileNotFoundError(
            f"newest run {newest_missing.name!r} at {newest_missing.path} has no usable "
            f"checkpoint under {ckpt_root / newest_missing.name}; pass --run/--ckpt "
            "explicitly or rerun with --allow-stale-fallback"
        )
    return _latest_checkpoint_run(ckpt_root)


def _usable_checkpoints(run_dir: Path) -> list[Path]:
    if not run_dir.is_dir():
        return []
    return [
        p for p in run_dir.glob("*.pt")
        if p.name != "snapshot_init.pt" and p.is_file()
    ]


def _latest_checkpoint_run(ckpt_root: Path) -> RunChoice:
    candidates = [
        ckpt
        for run_dir in ckpt_root.glob("*")
        for ckpt in _usable_checkpoints(run_dir)
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
    candidates = _usable_checkpoints(run_dir)
    if not candidates:
        raise FileNotFoundError(f"no usable checkpoints found in {run_dir}")
    return max(candidates, key=lambda p: (p.stat().st_mtime, _snapshot_num(p)))


def _is_sac_checkpoint(ckpt: Path) -> bool:
    """A SAC checkpoint stores `actor`/`actor_cfg`; PPO stores `model`/`config`."""
    import torch

    state = torch.load(ckpt, map_location="cpu", weights_only=False)
    return "actor" in state and "actor_cfg" in state


def _agent_factory(
    name: str,
    ckpt: Path,
    *,
    device: str,
    deterministic: bool,
) -> Callable[[Any], list[list]]:
    if name == "learned":
        # Auto-detect SAC vs PPO checkpoint format so one flag works for both.
        if _is_sac_checkpoint(ckpt):
            return SACAgent(ckpt, device=device, deterministic=deterministic)
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


def _counting_kaggle_agent(
    agent: Callable[[Any], list[list]],
    counts: dict[str, int],
) -> Callable[..., list[list]]:
    def wrapped(obs: Any, *_args: Any) -> list[list]:
        actions = agent(obs)
        counts["turns"] += 1
        counts["actions"] += len(actions)
        if actions:
            counts["nonempty_turns"] += 1
        return actions

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
        "--allow-stale-fallback",
        action="store_true",
        help="If the newest TensorBoard run has no checkpoint, fall back to an older checkpointed run.",
    )
    parser.add_argument(
        "--opponent",
        choices=("heuristic", "sniper", "bot", "random", "learned"),
        default="learned",
        help="Opponent agent. 'bot' is an alias for the competition sniper starter bot.",
    )
    parser.add_argument("--opponent-ckpt", type=Path, default=None)
    parser.add_argument("--num-players", type=int, choices=(2, 4), default=2)
    parser.add_argument(
        "--learner-seat",
        type=int,
        default=0,
        help="Seat index for the learned agent; use paired replays to check seat bias.",
    )
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--ship-speed", type=float, default=6.0)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Use deployment-style deterministic action selection. Default replay is stochastic to match PPO rollouts.",
    )
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--out", type=Path, default=None)
    return parser


def main() -> None:
    parser = _parser()
    args = parser.parse_args()
    if not 0 <= args.learner_seat < args.num_players:
        parser.error("--learner-seat must be in [0, num_players)")

    from kaggle_environments import make  # type: ignore[import-not-found]

    run = (
        RunChoice(name=args.run, path=None)
        if args.run is not None
        else _latest_run(
            args.runs_root,
            args.ckpt_root,
            allow_stale_fallback=args.allow_stale_fallback,
        )
    )
    ckpt = args.ckpt or _latest_checkpoint(args.ckpt_root, run.name)
    opponent_ckpt = args.opponent_ckpt or ckpt
    out = args.out or _default_out(run, ckpt, args.opponent)
    out.parent.mkdir(parents=True, exist_ok=True)

    learned = _agent_factory(
        "learned",
        ckpt,
        device=args.device,
        deterministic=args.deterministic and not args.stochastic,
    )
    opponent = _agent_factory(
        args.opponent,
        opponent_ckpt,
        device=args.device,
        deterministic=args.deterministic and not args.stochastic,
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
    learner_counts = {"turns": 0, "actions": 0, "nonempty_turns": 0}
    agents = []
    for seat in range(args.num_players):
        if seat == args.learner_seat:
            agents.append(_counting_kaggle_agent(learned, learner_counts))
        else:
            agents.append(_kaggle_agent(opponent))
    env.run(agents)
    rewards = [float(s.reward or 0.0) for s in env.steps[-1]]
    statuses = [str(s.status) for s in env.steps[-1]]
    learner_reward = rewards[args.learner_seat]
    opponent_best = max(
        (reward for seat, reward in enumerate(rewards) if seat != args.learner_seat),
        default=0.0,
    )
    out.write_text(env.render(mode="html"))

    print(f"run: {run.name}")
    if run.path is not None:
        print(f"run_dir: {run.path}")
    print(f"ckpt: {ckpt}")
    print(f"opponent: {args.opponent}")
    print(f"learner_seat: {args.learner_seat}")
    print(f"device: {args.device}")
    print(
        "policy_mode: "
        f"{'deterministic' if args.deterministic and not args.stochastic else 'stochastic'}"
    )
    print(
        "learner_actions: "
        f"{learner_counts['actions']} over {learner_counts['turns']} turns "
        f"({learner_counts['nonempty_turns']} nonempty turns)"
    )
    print(f"rewards: {rewards}")
    print(f"learner_won: {learner_reward > opponent_best}")
    print(f"statuses: {statuses}")
    print(f"replay: {out.resolve()}")


if __name__ == "__main__":
    main()
