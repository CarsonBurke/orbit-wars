"""Self-play opponent pool with Elo-driven snapshot retention.

Sampling rule (per opponent slot, independently):
  - With probability `self_play_prob` (default 0.8): play "self" (a
    no-grad copy of the current learner).
  - Otherwise: play a uniformly random snapshot from the pool's top-K
    by Elo. If the pool is empty, fall back to "self".

Eviction: when a new snapshot pushes the pool past `top_k`, drop the
member with the *lowest UCB* (upper-confidence bound on Elo, see
`EloTracker.ucb`). Few-games snapshots get a wide interval that lifts
their UCB above battle-tested peers' UCBs at the same point estimate, so
they're protected during their first sample; only snapshots that are
*both* weak and well-measured drop to the bottom and get culled.

Heuristic baselines (random/sniper/heuristic) intentionally aren't in this
pool — they live in `evaluate.py` for benchmark runs and in the value-
pretraining behavior policy. Their fixed Elo would just clog the top-K.
"""

from __future__ import annotations

import copy
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import torch

from ..agents.heuristic import heuristic_agent
from ..agents.learned import LearnedAgent
from ..agents.random_agent import random_agent
from ..agents.sniper import sniper_agent
from ..policies.model import OrbitPolicy
from .elo import EloTracker

AgentFn = Callable[[Any], list[list]]

LEARNER_NAME = "learner"

BUILTIN: dict[str, AgentFn] = {
    "random": random_agent,
    "sniper": sniper_agent,
    "heuristic": heuristic_agent,
}


@dataclass
class OpponentSlot:
    """One filled opponent seat — `name` is the Elo identity.

    `agent` is None for self-play seats (`name == LEARNER_NAME`): the
    vectorized rollout batches those obs into the learner forward, so it
    never invokes a per-seat callable. For snapshot seats `agent` is the
    `LearnedAgent` to call on each obs.
    """

    name: str
    agent: AgentFn | None


class OpponentPool:
    """Top-K snapshot pool, ranked by Elo."""

    def __init__(
        self,
        elo: EloTracker,
        top_k: int = 8,
        self_play_prob: float = 0.8,
        rng: random.Random | None = None,
    ):
        self.elo = elo
        self.top_k = top_k
        self.self_play_prob = self_play_prob
        self.rng = rng or random.Random()
        self._frozen: dict[str, AgentFn] = {}

    # --- snapshot management -------------------------------------------------

    def add_snapshot(
        self,
        label: str,
        model: OrbitPolicy,
        ckpt_path: str | Path,
        seed_rating: float | None = None,
    ) -> str:
        """Save a frozen copy of `model` to `ckpt_path` and add it to the pool.

        New snapshots inherit `seed_rating` (default: the learner's current
        Elo). After insertion, the pool is trimmed to top-K by current Elo.
        Returns the snapshot's identity name (e.g. "frozen:0050").
        """
        snap = copy.deepcopy(model).eval()
        for p in snap.parameters():
            p.requires_grad_(False)
        ckpt_path = Path(ckpt_path)
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"model": snap.state_dict(), "config": snap.cfg.to_dict()}, ckpt_path
        )
        name = f"frozen:{label}"
        self._frozen[name] = LearnedAgent(ckpt_path, deterministic=False)
        if seed_rating is None:
            seed_rating = self.elo.get(LEARNER_NAME)
        self.elo.set(name, float(seed_rating))
        self._trim_to_top_k()
        return name

    def _trim_to_top_k(self) -> None:
        if len(self._frozen) <= self.top_k:
            return
        # Sort by UCB so a snapshot that's weak *and* well-measured is the
        # one to go — not a freshly-added snapshot whose first game happened
        # to be a loss against a strong opponent. Lowest-UCB == lowest-rating
        # under high certainty == confident this one's actually weak.
        ranked = sorted(self._frozen.keys(), key=lambda n: self.elo.ucb(n), reverse=True)
        for name in ranked[self.top_k :]:
            self._frozen.pop(name, None)
            # Leave the rating in EloTracker — it's history; cheap to keep.

    def snapshot_names(self) -> list[str]:
        """Currently-alive snapshot identities, ordered by Elo desc."""
        return sorted(self._frozen.keys(), key=lambda n: self.elo.get(n), reverse=True)

    # --- sampling ------------------------------------------------------------

    def sample(self, k: int) -> list[OpponentSlot]:
        """Sample `k` opponent slots independently per slot."""
        return [self._sample_one() for _ in range(k)]

    def _sample_one(self) -> OpponentSlot:
        use_self = (not self._frozen) or (self.rng.random() < self.self_play_prob)
        if use_self:
            # No agent callable: vec_rollout batches self-play seats into
            # the learner forward via identity == LEARNER_NAME.
            return OpponentSlot(name=LEARNER_NAME, agent=None)
        name = self.rng.choice(self.snapshot_names())
        return OpponentSlot(name=name, agent=self._frozen[name])
