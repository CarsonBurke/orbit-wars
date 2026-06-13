"""Self-play opponent pools.

`OpponentPool` is the original Elo-driven snapshot pool. Its behavior is kept
stable for old configs:
  - With probability `self_play_prob` (default 0.8): play "self" (a
    no-grad copy of the current learner).
  - Otherwise: play a uniformly random snapshot from the pool's top-K
    by Elo. If the pool is empty, fall back to "self".

`NoBuiltinTrainingPool` is the learned-policy-only pool described in
docs/selfplay_pool_and_validation_spec.md. It samples the current learner,
an active utility-retained snapshot pool, and a stratified historical archive.
"""

from __future__ import annotations

import copy
import math
import random
from collections import OrderedDict
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from ..agents.heuristic import heuristic_agent
from ..agents.learned import LearnedAgent
from ..agents.random_agent import random_agent
from ..agents.sniper import (
    sniper_agent,
    sniper_v2_agent,
    sniper_v3_agent,
    sniper_v4_agent,
    sniper_v5_agent,
    sniper_v6_agent,
    sniper_v7_agent,
    sniper_v8_agent,
    sniper_v9_agent,
    sniper_v10_agent,
    sniper_v11_agent,
    sniper_v12_agent,
    sniper_v13_agent,
    sniper_v14_agent,
    sniper_v15_agent,
    sniper_v16_agent,
    sniper_v17_agent,
)
from ..policies.model import OrbitPolicy
from .elo import EloTracker

AgentFn = Callable[[Any], list[list]]

LEARNER_NAME = "learner"

BUILTIN: dict[str, AgentFn] = {
    "random": random_agent,
    "sniper": sniper_agent,
    "sniper_v2": sniper_v2_agent,
    "sniper_v3": sniper_v3_agent,
    "sniper_v4": sniper_v4_agent,
    "sniper_v5": sniper_v5_agent,
    "sniper_v6": sniper_v6_agent,
    "sniper_v7": sniper_v7_agent,
    "sniper_v8": sniper_v8_agent,
    "sniper_v9": sniper_v9_agent,
    "sniper_v10": sniper_v10_agent,
    "sniper_v11": sniper_v11_agent,
    "sniper_v12": sniper_v12_agent,
    "sniper_v13": sniper_v13_agent,
    "sniper_v14": sniper_v14_agent,
    "sniper_v15": sniper_v15_agent,
    "sniper_v16": sniper_v16_agent,
    "sniper_v17": sniper_v17_agent,
    "heuristic": heuristic_agent,
}


def _copy_model_without_compile_caches(model: OrbitPolicy) -> OrbitPolicy:
    cache_names = (
        "_owars_minibatch_kernel_cache",
        "_owars_rollout_kernel_cache",
    )
    stashed = {
        name: model.__dict__.pop(name)
        for name in cache_names
        if name in model.__dict__
    }
    try:
        return copy.deepcopy(model)
    finally:
        model.__dict__.update(stashed)


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


@dataclass(frozen=True)
class OpponentSamplePanel:
    """Bounded opponent identities for one rollout wave.

    Source probabilities are still sampled per opponent seat, but non-current
    draws are restricted to these identities. That keeps self-training from
    fragmenting rollout inference across every retained snapshot in the pool.
    """

    active: tuple[str, ...] = ()
    historical: tuple[str, ...] = ()


class LazyLearnedAgent:
    """LearnedAgent wrapper that loads retained archive snapshots on demand.

    Active snapshots stay hot because they are sampled often and need fast
    batched inference. Historical snapshots can be numerous; keeping them as
    checkpoint paths plus a small LRU avoids retaining the full archive on the
    training GPU.
    """

    _cache: OrderedDict[tuple[str, str, bool, str | None], LearnedAgent] = OrderedDict()

    def __init__(
        self,
        ckpt_path: str | Path,
        *,
        device: str,
        deterministic: bool,
        compile_mode: str | None = None,
        cache_size: int = 8,
    ):
        self.ckpt_path = Path(ckpt_path)
        self.device = device
        self.deterministic = deterministic
        self.compile_mode = compile_mode
        self.cache_size = max(1, int(cache_size))

    def _agent(self) -> LearnedAgent:
        key = (
            str(self.ckpt_path.resolve()),
            self.device,
            self.deterministic,
            self.compile_mode,
        )
        agent = self._cache.get(key)
        if agent is not None:
            self._cache.move_to_end(key)
            return agent
        agent = LearnedAgent(
            self.ckpt_path,
            device=self.device,
            deterministic=self.deterministic,
            compile_mode=self.compile_mode,
        )
        self._cache[key] = agent
        self._cache.move_to_end(key)
        while len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)
        return agent

    def __call__(self, obs: Any) -> list[list]:
        return self._agent()(obs)

    def act_batch(self, obs_list: list[Any]) -> list[list[list]]:
        return self._agent().act_batch(obs_list)

    @property
    def model(self) -> OrbitPolicy:
        return self._agent().model


class OpponentPool:
    """Top-K snapshot pool, ranked by Elo."""

    def __init__(
        self,
        elo: EloTracker,
        top_k: int = 8,
        self_play_prob: float = 0.8,
        device: str = "cpu",
        rng: random.Random | None = None,
    ):
        self.elo = elo
        self.top_k = top_k
        self.self_play_prob = self_play_prob
        self.device = device
        self.rng = rng or random.Random()
        self._frozen: dict[str, AgentFn] = {}
        self._frozen_paths: dict[str, Path] = {}

    # --- snapshot management -------------------------------------------------

    def _copy_model_without_compile_caches(self, model: OrbitPolicy) -> OrbitPolicy:
        return _copy_model_without_compile_caches(model)

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
        snap = self._copy_model_without_compile_caches(model).eval()
        for p in snap.parameters():
            p.requires_grad_(False)
        ckpt_path = Path(ckpt_path)
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"model": snap.state_dict(), "config": snap.cfg.to_dict()}, ckpt_path
        )
        name = f"frozen:{label}"
        self._frozen[name] = LearnedAgent(
            ckpt_path,
            device=self.device,
            deterministic=False,
        )
        self._frozen_paths[name] = ckpt_path
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
            path = self._frozen_paths.pop(name, None)
            if path is not None:
                path.unlink(missing_ok=True)
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


class FixedOpponentPool:
    """Static builtin-opponent sampler with the `OpponentPool` rollout API."""

    def __init__(
        self,
        opponent_names: Sequence[str],
        rng: random.Random | None = None,
    ):
        unknown = set(opponent_names) - set(BUILTIN)
        if unknown:
            raise ValueError(
                f"unknown fixed opponents {sorted(unknown)}; "
                f"valid: {sorted(BUILTIN)}"
            )
        if not opponent_names:
            raise ValueError("fixed opponent mode requires at least one opponent")
        self.opponent_names = list(opponent_names)
        self.rng = rng or random.Random()

    def sample(self, k: int) -> list[OpponentSlot]:
        return [self._sample_one() for _ in range(k)]

    def _sample_one(self) -> OpponentSlot:
        name = self.rng.choice(self.opponent_names)
        return OpponentSlot(name=name, agent=BUILTIN[name])

    def snapshot_names(self) -> list[str]:
        return []


@dataclass
class TrainingSnapshot:
    """Learned snapshot metadata and learner-relative active-pool statistics.

    `wins/draws/losses_vs_current` are from the snapshot's perspective against
    the live learner. `effective_result_ema` is intentionally from the learner's
    perspective because active-pool utility needs the learner win rate.
    """

    name: str
    agent: AgentFn
    path: Path
    created_update: int
    entered_active_update: int | None = None
    games_vs_current: int = 0
    wins_vs_current: int = 0
    draws_vs_current: int = 0
    losses_vs_current: int = 0
    mean_margin_vs_current: float = 0.0
    last_sampled_update: int = -1
    effective_result_ema: float = 0.5
    effective_margin_ema: float = 0.0
    effective_games: float = 0.0
    evicted_update: int | None = None
    previous_best: bool = False
    notable: bool = False

    def record_vs_current(
        self,
        *,
        learner_score: float,
        snapshot_score: float,
        current_update: int,
        ema_decay: float,
    ) -> None:
        margin = float(snapshot_score) - float(learner_score)
        result_for_learner: float
        if snapshot_score > learner_score:
            self.wins_vs_current += 1
            result_for_learner = 0.0
        elif snapshot_score < learner_score:
            self.losses_vs_current += 1
            result_for_learner = 1.0
        else:
            self.draws_vs_current += 1
            result_for_learner = 0.5
        self.games_vs_current += 1
        self.mean_margin_vs_current += (
            margin - self.mean_margin_vs_current
        ) / self.games_vs_current
        if self.effective_games <= 0.0:
            self.effective_result_ema = result_for_learner
            self.effective_margin_ema = margin
        else:
            keep = ema_decay
            add = 1.0 - ema_decay
            self.effective_result_ema = (
                keep * self.effective_result_ema + add * result_for_learner
            )
            self.effective_margin_ema = (
                keep * self.effective_margin_ema + add * margin
            )
        self.effective_games = ema_decay * self.effective_games + 1.0
        self.last_sampled_update = current_update

    def learner_win_rate(self) -> float:
        if self.effective_games > 0.0:
            return self.effective_result_ema
        if self.games_vs_current > 0:
            return (
                self.losses_vs_current + 0.5 * self.draws_vs_current
            ) / self.games_vs_current
        return 0.5


class NoBuiltinTrainingPool:
    """Learned-only self-play pool from the self-play pool spec.

    Sampling is per opponent slot:
      - 40% current learner
      - 30% active training pool
      - 30% historical training archive

    Empty learned-policy sources are removed from the draw and the remaining
    weights are renormalized. The active pool is retained by learner-relative
    utility, not Elo. Historical archive sampling is stratified by log-spaced
    milestones, recent active evictions, and notable/previous-best snapshots.
    """

    HISTORICAL_BUCKET_WEIGHTS: dict[str, float] = {
        "log": 0.50,
        "recent_eviction": 0.25,
        "notable": 0.25,
    }

    def __init__(
        self,
        *,
        active_pool_size: int = 16,
        active_sample_panel_size: int = 2,
        historical_training_archive_size: int = 128,
        current_learner_prob: float = 0.4,
        active_pool_prob: float = 0.3,
        historical_archive_prob: float = 0.3,
        difficulty_weight: float = 1.0,
        uncertainty_weight: float = 0.25,
        recency_weight: float = 0.25,
        hardness_weight: float = 0.5,
        recency_half_life_updates: float = 50.0,
        min_games_before_eviction: int = 16,
        stats_ema_decay: float = 0.95,
        historical_sample_panel_size: int = 2,
        historical_agent_cache_size: int = 8,
        recent_eviction_archive_size: int | None = None,
        notable_archive_size: int | None = None,
        device: str = "cpu",
        rng: random.Random | None = None,
    ):
        if active_pool_size <= 0:
            raise ValueError("active_pool_size must be positive")
        if active_sample_panel_size <= 0:
            raise ValueError("active_sample_panel_size must be positive")
        if historical_training_archive_size <= 0:
            raise ValueError("historical_training_archive_size must be positive")
        probs = (current_learner_prob, active_pool_prob, historical_archive_prob)
        if (
            any((not math.isfinite(p)) or p < 0.0 for p in probs)
            or sum(probs) <= 0.0
        ):
            raise ValueError("sampling probabilities must be non-negative and non-zero")
        weights = (
            difficulty_weight,
            uncertainty_weight,
            recency_weight,
            hardness_weight,
        )
        if (
            any((not math.isfinite(weight)) or weight < 0.0 for weight in weights)
            or sum(weights) <= 0.0
        ):
            raise ValueError("utility weights must be non-negative and non-zero")
        if (
            not math.isfinite(recency_half_life_updates)
            or recency_half_life_updates <= 0.0
        ):
            raise ValueError("recency_half_life_updates must be positive")
        if min_games_before_eviction < 0:
            raise ValueError("min_games_before_eviction must be non-negative")
        if not math.isfinite(stats_ema_decay) or not 0.0 <= stats_ema_decay < 1.0:
            raise ValueError("stats_ema_decay must be in [0, 1)")
        if historical_sample_panel_size <= 0:
            raise ValueError("historical_sample_panel_size must be positive")
        if historical_agent_cache_size <= 0:
            raise ValueError("historical_agent_cache_size must be positive")
        default_bucket_cap = historical_training_archive_size // 4
        recent_eviction_archive_size = (
            default_bucket_cap
            if recent_eviction_archive_size is None
            else recent_eviction_archive_size
        )
        notable_archive_size = (
            default_bucket_cap
            if notable_archive_size is None
            else notable_archive_size
        )
        if recent_eviction_archive_size < 0:
            raise ValueError("recent_eviction_archive_size must be non-negative")
        if notable_archive_size < 0:
            raise ValueError("notable_archive_size must be non-negative")
        if (
            recent_eviction_archive_size + notable_archive_size
            > historical_training_archive_size
        ):
            raise ValueError(
                "recent/notable archive caps must fit within "
                "historical_training_archive_size"
            )

        self.active_pool_size = active_pool_size
        self.active_sample_panel_size = active_sample_panel_size
        self.historical_training_archive_size = historical_training_archive_size
        self.current_learner_prob = current_learner_prob
        self.active_pool_prob = active_pool_prob
        self.historical_archive_prob = historical_archive_prob
        self.difficulty_weight = difficulty_weight
        self.uncertainty_weight = uncertainty_weight
        self.recency_weight = recency_weight
        self.hardness_weight = hardness_weight
        self.recency_half_life_updates = recency_half_life_updates
        self.min_games_before_eviction = min_games_before_eviction
        self.stats_ema_decay = stats_ema_decay
        self.historical_sample_panel_size = historical_sample_panel_size
        self.historical_agent_cache_size = historical_agent_cache_size
        self.recent_eviction_archive_size = recent_eviction_archive_size
        self.notable_archive_size = notable_archive_size
        self.device = device
        self.rng = rng or random.Random()
        self.current_update = 0
        self._snapshots: dict[str, TrainingSnapshot] = {}
        self._active: set[str] = set()
        self._historical_buckets: dict[str, set[str]] = {
            "log": set(),
            "recent_eviction": set(),
            "notable": set(),
        }
        self._log_archive: set[str] = set()
        self._recent_evictions: list[str] = []
        self._historical_panel: set[str] = set()
        self._historical_panel_update: int | None = None

    # --- snapshot management -------------------------------------------------

    def add_snapshot(
        self,
        label: str,
        model: OrbitPolicy,
        ckpt_path: str | Path,
        *,
        created_update: int | None = None,
        enter_active: bool = True,
        previous_best: bool = False,
        notable: bool = False,
    ) -> str:
        """Save a frozen learned snapshot and route it into active/archive pools."""
        created_update = self.current_update if created_update is None else created_update
        snap = _copy_model_without_compile_caches(model).eval()
        for p in snap.parameters():
            p.requires_grad_(False)
        ckpt_path = Path(ckpt_path)
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"model": snap.state_dict(), "config": snap.cfg.to_dict()}, ckpt_path
        )
        name = f"frozen:{label}"
        if name in self._snapshots:
            raise ValueError(f"snapshot {name!r} already exists")
        agent: AgentFn = (
            LearnedAgent(
                ckpt_path,
                device=self.device,
                deterministic=False,
            )
            if enter_active
            else self._lazy_agent(ckpt_path)
        )
        record = TrainingSnapshot(
            name=name,
            agent=agent,
            path=ckpt_path,
            created_update=int(created_update),
            previous_best=previous_best,
            notable=notable,
        )
        self._snapshots[name] = record
        if enter_active:
            record.entered_active_update = self.current_update
            record.evicted_update = None
            self._active.add(name)
            self._trim_active()
        self.rebuild_historical_archive()
        return name

    def mark_previous_best(self, name: str) -> None:
        self._snapshots[name].previous_best = True
        self.rebuild_historical_archive()

    def mark_notable(self, name: str) -> None:
        self._snapshots[name].notable = True
        self.rebuild_historical_archive()

    def set_current_update(self, update: int) -> None:
        self.current_update = int(update)
        self.rebuild_historical_archive()

    def _trim_active(self) -> None:
        while len(self._active) > self.active_pool_size:
            exposed = [
                name
                for name in self._active
                if self._snapshots[name].games_vs_current
                >= self.min_games_before_eviction
            ]
            if exposed:
                victim = min(exposed, key=self.active_utility)
            else:
                victim = min(
                    self._active,
                    key=lambda n: (
                        self._snapshots[n].entered_active_update
                        if self._snapshots[n].entered_active_update is not None
                        else self._snapshots[n].created_update,
                        self._snapshots[n].created_update,
                    ),
                )
            self._evict_active(victim)

    def _evict_active(self, name: str) -> None:
        self._active.remove(name)
        record = self._snapshots[name]
        record.evicted_update = self.current_update
        record.agent = self._lazy_agent(record.path)
        self._recent_evictions.append(name)

    def rebuild_historical_archive(self) -> None:
        non_active = [
            name
            for name, record in self._snapshots.items()
            if name not in self._active and record.created_update <= self.current_update
        ]
        log_cap = max(
            0,
            self.historical_training_archive_size
            - self.recent_eviction_archive_size
            - self.notable_archive_size,
        )
        self._log_archive = self._select_log_archive(non_active, log_cap)
        buckets = {
            "log": set(self._log_archive),
            "recent_eviction": set(self._select_recent_evictions(non_active)),
            "notable": set(self._select_notables(non_active)),
        }
        self._historical_buckets = buckets
        self._discard_unretained()
        self._historical_panel_update = None

    def _lazy_agent(self, ckpt_path: str | Path) -> LazyLearnedAgent:
        return LazyLearnedAgent(
            ckpt_path,
            device=self.device,
            deterministic=False,
            cache_size=self.historical_agent_cache_size,
        )

    def _select_log_archive(self, names: Sequence[str], cap: int) -> set[str]:
        if cap <= 0:
            return set()
        candidates = [
            name
            for name in names
            if self.current_update - self._snapshots[name].created_update >= 1
        ]
        return self._select_log_spaced_by_age(
            candidates,
            cap,
            age_fn=lambda n: self.current_update - self._snapshots[n].created_update,
            tie_fn=lambda n: self._snapshots[n].created_update,
        )

    def _log_age_bucket(self, name: str) -> int:
        age = max(1, self.current_update - self._snapshots[name].created_update)
        return int(math.log2(age))

    def _select_log_spaced_by_age(
        self,
        names: Sequence[str],
        cap: int,
        *,
        age_fn: Callable[[str], int],
        tie_fn: Callable[[str], int],
        prefer_oldest_in_bucket: bool = True,
    ) -> set[str]:
        """Pick one online representative per logarithmic age bucket.

        Retained snapshots are deleted from disk, so this selector must preserve
        the oldest member of each bucket; otherwise a snapshot can be discarded
        at age 3 and never survive to represent age 4/8/16 later.
        """
        if cap <= 0:
            return set()
        by_bucket: dict[int, list[str]] = {}
        for name in names:
            age = max(0, int(age_fn(name)))
            bucket = 0 if age <= 1 else int(math.log2(age))
            by_bucket.setdefault(bucket, []).append(name)
        bucket_ids = sorted(by_bucket)
        selected_by_bucket: dict[int, str] = {}
        for bucket in bucket_ids:
            if prefer_oldest_in_bucket:
                selected_by_bucket[bucket] = max(
                    by_bucket[bucket],
                    key=lambda n: (
                        max(0, int(age_fn(n))),
                        tie_fn(n),
                    ),
                )
            else:
                selected_by_bucket[bucket] = min(
                    by_bucket[bucket],
                    key=lambda n: (
                        max(0, int(age_fn(n))),
                        -tie_fn(n),
                    ),
                )
        if len(bucket_ids) <= cap:
            return set(selected_by_bucket.values())
        keep_buckets = self._evenly_spaced_values(bucket_ids, cap)
        return {selected_by_bucket[bucket] for bucket in keep_buckets}

    @staticmethod
    def _evenly_spaced_values(values: Sequence[int], count: int) -> list[int]:
        if count <= 0:
            return []
        if len(values) <= count:
            return list(values)
        if count == 1:
            return [values[-1]]
        last = len(values) - 1
        indexes = sorted(
            {
                round(i * last / (count - 1))
                for i in range(count)
            }
        )
        return [values[i] for i in indexes]

    def _select_recent_evictions(self, names: Sequence[str]) -> list[str]:
        if self.recent_eviction_archive_size <= 0:
            return []
        allowed = set(names)
        candidates = [
            name
            for name in dict.fromkeys(reversed(self._recent_evictions))
            if name in allowed
            and name in self._snapshots
            and self._snapshots[name].evicted_update is not None
        ]
        selected = self._select_log_spaced_by_age(
            candidates,
            self.recent_eviction_archive_size,
            age_fn=lambda n: self.current_update
            - int(
                self._snapshots[n].evicted_update
                if self._snapshots[n].evicted_update is not None
                else self.current_update
            ),
            tie_fn=lambda n: int(
                self._snapshots[n].evicted_update
                if self._snapshots[n].evicted_update is not None
                else 0
            ),
            prefer_oldest_in_bucket=False,
        )
        return sorted(
            selected,
            key=lambda n: int(
                self._snapshots[n].evicted_update
                if self._snapshots[n].evicted_update is not None
                else 0
            ),
            reverse=True,
        )

    def _select_notables(self, names: Sequence[str]) -> list[str]:
        allowed = set(names)
        candidates = [
            name
            for name in allowed
            if self._snapshots[name].previous_best or self._snapshots[name].notable
        ]
        return sorted(
            candidates,
            key=lambda n: (
                not self._snapshots[n].previous_best,
                not self._snapshots[n].notable,
                -self._snapshots[n].created_update,
            ),
        )[: self.notable_archive_size]

    def _discard_unretained(self) -> None:
        retained = set(self._active) | self._historical_names()
        for name in list(self._snapshots):
            if name in retained:
                continue
            record = self._snapshots.pop(name)
            record.path.unlink(missing_ok=True)
        self._recent_evictions = [
            name for name in self._recent_evictions if name in self._snapshots
        ]
        self._log_archive = {name for name in self._log_archive if name in self._snapshots}
        self._historical_panel = {
            name for name in self._historical_panel if name in self._snapshots
        }

    # --- statistics / utility ------------------------------------------------

    def record_game(
        self,
        seats: Sequence[tuple[str, float]],
        *,
        current_update: int | None = None,
    ) -> None:
        """Update active stats only for active snapshots that faced the learner."""
        current_update = self.current_update if current_update is None else current_update
        learner_scores = [float(score) for name, score in seats if name == LEARNER_NAME]
        if not learner_scores:
            return
        learner_score = sum(learner_scores) / len(learner_scores)
        for name, score in seats:
            if name == LEARNER_NAME or name not in self._active:
                continue
            self.record_result_vs_current(
                name,
                learner_score=learner_score,
                snapshot_score=float(score),
                current_update=current_update,
            )

    def record_result_vs_current(
        self,
        name: str,
        *,
        learner_score: float,
        snapshot_score: float,
        current_update: int | None = None,
    ) -> None:
        """Record one current-learner-vs-active-snapshot result."""
        if name not in self._active:
            return
        update = self.current_update if current_update is None else int(current_update)
        self._snapshots[name].record_vs_current(
            learner_score=learner_score,
            snapshot_score=snapshot_score,
            current_update=update,
            ema_decay=self.stats_ema_decay,
        )

    def active_utility(self, name: str) -> float:
        record = self._snapshots[name]
        learner_win_rate = record.learner_win_rate()
        difficulty = max(0.0, 1.0 - 2.0 * abs(learner_win_rate - 0.5))
        uncertainty = 1.0 / math.sqrt(1.0 + record.games_vs_current)
        age = max(0, self.current_update - record.created_update)
        recency = math.exp(-age / self.recency_half_life_updates)
        hardness = min(max(0.5 - learner_win_rate, 0.0), 0.5)
        return (
            self.difficulty_weight * difficulty
            + self.uncertainty_weight * uncertainty
            + self.recency_weight * recency
            + self.hardness_weight * hardness
        )

    def snapshot_stats(self, name: str) -> TrainingSnapshot:
        return self._snapshots[name]

    # --- sampling ------------------------------------------------------------

    def sample(
        self,
        k: int,
        *,
        current_update: int | None = None,
        panel: OpponentSamplePanel | None = None,
    ) -> list[OpponentSlot]:
        if current_update is not None and int(current_update) != self.current_update:
            self.set_current_update(int(current_update))
        panel = self._sanitize_panel(panel)
        update = self.current_update
        slots = [self._sample_one(update, panel) for _ in range(k)]
        self._diversify_if_all_same_non_current(slots, update, panel)
        return slots

    def sample_panel(
        self,
        *,
        current_update: int | None = None,
    ) -> OpponentSamplePanel:
        """Sample a bounded learned-opponent panel for one rollout wave."""
        if current_update is not None and int(current_update) != self.current_update:
            self.set_current_update(int(current_update))
        active = self._sample_name_panel(
            self.active_snapshot_names(),
            self.active_sample_panel_size,
        )
        historical = self._choose_historical_panel()
        return OpponentSamplePanel(
            active=tuple(active),
            historical=tuple(sorted(historical)),
        )

    def _sample_name_panel(self, names: Sequence[str], size: int) -> list[str]:
        if len(names) <= size:
            return list(names)
        return self.rng.sample(list(names), size)

    def _sanitize_panel(
        self,
        panel: OpponentSamplePanel | None,
    ) -> OpponentSamplePanel | None:
        if panel is None:
            return None
        active = tuple(name for name in panel.active if name in self._active)
        historical = tuple(
            name for name in panel.historical if name in self._historical_names()
        )
        return OpponentSamplePanel(active=active, historical=historical)

    def _sample_one(
        self,
        current_update: int,
        panel: OpponentSamplePanel | None = None,
    ) -> OpponentSlot:
        source = self._sample_source(panel)
        if source == "current":
            return OpponentSlot(name=LEARNER_NAME, agent=None)
        if source == "active":
            active_names = (
                list(panel.active) if panel is not None else self.active_snapshot_names()
            )
            name = self.rng.choice(active_names)
        else:
            name = self._sample_historical_name(panel)
        self._snapshots[name].last_sampled_update = current_update
        return OpponentSlot(name=name, agent=self._snapshots[name].agent)

    def _sample_source(self, panel: OpponentSamplePanel | None = None) -> str:
        weighted: list[tuple[str, float]] = []
        active_available = bool(panel.active) if panel is not None else bool(self._active)
        historical_available = (
            bool(panel.historical)
            if panel is not None
            else bool(self._historical_names())
        )
        if self.current_learner_prob > 0.0:
            weighted.append(("current", self.current_learner_prob))
        if active_available and self.active_pool_prob > 0.0:
            weighted.append(("active", self.active_pool_prob))
        if historical_available and self.historical_archive_prob > 0.0:
            weighted.append(("historical", self.historical_archive_prob))
        if not weighted:
            return "current"
        return self._weighted_choice(weighted)

    def _sample_historical_name(
        self,
        panel: OpponentSamplePanel | None = None,
    ) -> str:
        if panel is None:
            self._ensure_historical_panel()
        weighted = [
            (bucket, weight)
            for bucket, weight in self.HISTORICAL_BUCKET_WEIGHTS.items()
            if self._historical_bucket_panel(bucket, panel)
        ]
        bucket = self._weighted_choice(weighted)
        return self.rng.choice(sorted(self._historical_bucket_panel(bucket, panel)))

    def _ensure_historical_panel(self) -> None:
        names = self._historical_names()
        if self._historical_panel_update == self.current_update:
            self._historical_panel &= names
            return
        self._historical_panel = self._choose_historical_panel()
        self._historical_panel_update = self.current_update

    def _choose_historical_panel(self) -> set[str]:
        names = self._historical_names()
        if len(names) <= self.historical_sample_panel_size:
            return set(names)
        panel: set[str] = set()
        non_empty_buckets = [
            bucket
            for bucket in self.HISTORICAL_BUCKET_WEIGHTS
            if self._historical_buckets[bucket]
        ]
        if len(non_empty_buckets) > self.historical_sample_panel_size:
            bucket_order = self._sample_bucket_panel(
                non_empty_buckets,
                self.historical_sample_panel_size,
            )
        else:
            bucket_order = non_empty_buckets
        for bucket in bucket_order:
            bucket_names = sorted(self._historical_buckets[bucket])
            panel.add(self.rng.choice(bucket_names))
        while len(panel) < self.historical_sample_panel_size:
            weighted = [
                (bucket, weight)
                for bucket, weight in self.HISTORICAL_BUCKET_WEIGHTS.items()
                if sorted(self._historical_buckets[bucket] - panel)
            ]
            if not weighted:
                break
            bucket = self._weighted_choice(weighted)
            panel.add(self.rng.choice(sorted(self._historical_buckets[bucket] - panel)))
        return panel

    def _sample_bucket_panel(self, buckets: Sequence[str], size: int) -> list[str]:
        remaining = list(buckets)
        selected: list[str] = []
        while remaining and len(selected) < size:
            bucket = self._weighted_choice(
                [
                    (name, self.HISTORICAL_BUCKET_WEIGHTS[name])
                    for name in remaining
                ]
            )
            selected.append(bucket)
            remaining.remove(bucket)
        return selected

    def _historical_bucket_panel(
        self,
        bucket: str,
        panel: OpponentSamplePanel | None = None,
    ) -> set[str]:
        names = self._historical_buckets[bucket]
        if not names:
            return set()
        panel_names = set(panel.historical) if panel is not None else self._historical_panel
        if not panel_names:
            return names
        return names & panel_names

    def _weighted_choice(self, weighted: Sequence[tuple[str, float]]) -> str:
        total = sum(weight for _, weight in weighted)
        r = self.rng.random() * total
        upto = 0.0
        for name, weight in weighted:
            upto += weight
            if r <= upto:
                return name
        return weighted[-1][0]

    def _diversify_if_all_same_non_current(
        self,
        slots: list[OpponentSlot],
        current_update: int,
        panel: OpponentSamplePanel | None = None,
    ) -> None:
        if len(slots) < 2:
            return
        non_current = [slot for slot in slots if slot.name != LEARNER_NAME]
        if len(non_current) != len(slots):
            return
        first = non_current[0].name
        if any(slot.name != first for slot in non_current):
            return
        if first in self._active:
            same_source = set(panel.active) if panel is not None else set(
                self.active_snapshot_names()
            )
        elif first in self._historical_names():
            same_source = set(panel.historical) if panel is not None else (
                self._historical_names()
            )
        else:
            same_source = set()
        alternatives = sorted(same_source - {first})
        if not alternatives:
            return
        replacement = self.rng.choice(alternatives)
        slots[-1] = OpponentSlot(
            name=replacement,
            agent=self._snapshots[replacement].agent,
        )
        self._snapshots[replacement].last_sampled_update = current_update

    # --- views ---------------------------------------------------------------

    def active_snapshot_names(self) -> list[str]:
        return sorted(self._active, key=self.active_utility, reverse=True)

    def historical_snapshot_names(self, bucket: str | None = None) -> list[str]:
        if bucket is not None:
            if bucket not in self._historical_buckets:
                raise ValueError(
                    f"unknown historical bucket {bucket!r}; "
                    f"valid: {sorted(self._historical_buckets)}"
                )
            names = self._historical_buckets[bucket]
        else:
            names = self._historical_names()
        return sorted(names, key=lambda n: self._snapshots[n].created_update)

    def snapshot_names(self) -> list[str]:
        """Currently active snapshot identities, ordered by utility desc."""
        return self.active_snapshot_names()

    def all_snapshot_names(self) -> list[str]:
        return sorted(self._snapshots)

    def _historical_names(self) -> set[str]:
        return set().union(*self._historical_buckets.values())
