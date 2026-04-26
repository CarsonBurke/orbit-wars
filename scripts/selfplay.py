#!/usr/bin/env python
"""Run a single Orbit Wars match between two arbitrary agents (for debugging
and sanity checks; not used by the training loop)."""

from __future__ import annotations

import argparse

from owars.agents import HeuristicAgent, random_agent, sniper_agent
from owars.agents.learned import LearnedAgent

REGISTRY = {
    "random": lambda _: random_agent,
    "sniper": lambda _: sniper_agent,
    "heuristic": lambda _: HeuristicAgent(),
    "learned": lambda ckpt: LearnedAgent(ckpt),
}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--p0", default="heuristic")
    p.add_argument("--p1", default="sniper")
    p.add_argument("--p0-ckpt", default=None)
    p.add_argument("--p1-ckpt", default=None)
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--steps", type=int, default=500)
    p.add_argument("--num-players", type=int, default=2, choices=(2, 4))
    args = p.parse_args()

    from kaggle_environments import make  # type: ignore[import-not-found]

    a0 = REGISTRY[args.p0](args.p0_ckpt)
    a1 = REGISTRY[args.p1](args.p1_ckpt)
    others = [a1] * (args.num_players - 1)

    wins = 0
    for _ in range(args.episodes):
        env = make("orbit_wars", configuration={"episodeSteps": args.steps})
        env.run([a0, *others])
        rewards = [float(s.reward or 0.0) for s in env.steps[-1]]
        if rewards[0] >= max(rewards):
            wins += 1
    print(f"{args.p0} vs {args.p1}: {wins}/{args.episodes} wins")


if __name__ == "__main__":
    main()
