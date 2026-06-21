"""Vectorized rollouts: N parallel envs + one batched policy forward per step.

Per env-step the orchestrator does:

  1. Walk every alive (env, seat) and bucket the obs by *agent identity*
     — `LEARNER_NAME` for the live model (covers the learner's seat *and*
     any self-play opponent seats), or `frozen:<label>` for snapshots.
  2. Encode + stack the learner bucket into one big batch and run a
     single `model(...)` forward (this is where the wall-clock win comes
     from — at 16 envs × 2 seats with 80% self-play, the learner bucket
     averages ~25 obs per step instead of doing 25 separate B=1 forwards).
  3. For each snapshot bucket, batch that snapshot's rows separately.
     Different snapshots are different models, so they cannot share a
     forward, but each snapshot still uses one batched call for its rows.
  4. Distribute the resulting moves back into per-env action lists, then
     step the alive envs in parallel via `VecEnv.step_subset`.
  5. Trajectories are recorded *only for the learner_seat* of each env —
     self-play opponent seats are batched into the forward for compute,
     but their per-step records aren't collected (PPO trains on the
     learner's transitions only).
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Sequence
from time import perf_counter
from typing import Any

import numpy as np
import torch

from ..policies.features import (
    MAX_PLANETS,
    PLANET_INBOUND_FEAT_DIM,
    EncodedObs,
    active_fleet_width,
    bucket_encoded_fleet_width,
    bucket_fleet_width,
    encode_raw_observations,
    fleet_target_planet_idx_or_empty,
    planet_inbound_feats_or_empty,
    slice_encoded_fleet_width,
)
from ..policies.model import OrbitPolicy, PolicyOutput
from ..policies.sampling import (
    ActionContext,
    sample_batch_actions_context,
    sample_batch_actions_raw,
    sample_batch_with_records_context,
    sample_batch_with_records_raw,
)
from .config import RewardCfg
from .league import LEARNER_NAME, OpponentSlot
from .rollout import Trajectory, TrajectoryRecordRef, _obs_production_margin
from .vec_env import VecEnv

_SOURCE_MAJOR_COMPILE_ROW_CAP = 32


def _mark_cuda_graph_step(device: torch.device) -> None:
    if device.type != "cuda":
        return
    mark = getattr(torch.compiler, "cudagraph_mark_step_begin", None)
    if callable(mark):
        mark()


def _add_timing(timings: dict[str, float] | None, key: str, seconds: float) -> None:
    if timings is not None:
        timings[key] = timings.get(key, 0.0) + float(seconds)


def _sync_cuda_timing(device: torch.device, enabled: bool) -> None:
    if enabled and device.type == "cuda":
        torch.cuda.synchronize(device)


def _scoped_timings(
    timings: dict[str, float] | None,
    prefix: str | None,
) -> dict[str, float] | None:
    if timings is None or prefix is None:
        return timings
    return {}


def _flush_scoped_timings(
    timings: dict[str, float] | None,
    scoped: dict[str, float] | None,
    prefix: str | None,
) -> None:
    if timings is None or scoped is None or prefix is None:
        return
    for key, value in scoped.items():
        _add_timing(timings, key, value)
        _add_timing(timings, f"{prefix}/{key}", value)


def _policy_batch_kwargs(policy_batch: Any, **kwargs: Any) -> dict[str, Any]:
    owner = getattr(policy_batch, "__self__", None)
    if hasattr(owner, "_feature_stage"):
        kwargs["reuse_pinned_buffers"] = True
    return kwargs


def _next_power_of_two(n: int) -> int:
    n = max(1, int(n))
    return 1 << (n - 1).bit_length()


def _capped_graph_rows(rows: int, *, max_rows: int) -> int:
    """Use static row buckets with power-of-two overflow."""
    rows = max(1, int(rows))
    cap = max(1, int(max_rows))
    return cap if rows <= cap else _next_power_of_two(rows)


def _snapshot_graph_rows(rows: int, *, max_rows: int = 64) -> int:
    """Use small static row buckets for frozen snapshots.

    Snapshot buckets are often tiny; padding each snapshot to the full learner
    row count burns GPU work and graph memory for rows that do not exist.
    """
    rows = max(1, int(rows))
    cap = max(1, int(max_rows))
    bucket = _next_power_of_two(rows)
    return min(bucket, cap) if rows <= cap else bucket


def _source_graph_rows(
    source_count: int,
    *,
    graph_rows: int,
    planets: int,
    max_sources_per_row: int,
) -> int:
    del source_count, max_sources_per_row
    row_capacity = min(int(planets), _SOURCE_MAJOR_COMPILE_ROW_CAP)
    return max(1, int(graph_rows) * row_capacity)


def _snapshot_rollout_graph_rows(
    rows: int,
    *,
    snapshot_compile_rows: int,
    compile_mode: str | None,
) -> int:
    if compile_mode is not None:
        return _capped_graph_rows(rows, max_rows=snapshot_compile_rows)
    return _snapshot_graph_rows(rows)


def _kernel_cache(model: torch.nn.Module) -> dict:
    cache = model.__dict__.get("_owars_rollout_kernel_cache")
    if cache is None:
        cache = {}
        model.__dict__["_owars_rollout_kernel_cache"] = cache
    return cache


class _DenseRolloutForwardKernel(torch.nn.Module):
    def __init__(
        self,
        model: torch.nn.Module,
        *,
        autocast_enabled: bool,
        include_value: bool,
    ) -> None:
        super().__init__()
        self.model = model
        self.autocast_enabled = bool(autocast_enabled)
        self.include_value = bool(include_value)

    def forward(
        self,
        global_feats: torch.Tensor,
        planet_feats: torch.Tensor,
        planet_mask: torch.Tensor,
        planet_owned_mask: torch.Tensor,
        planet_ids: torch.Tensor,
        planet_garrison: torch.Tensor,
        fleet_feats: torch.Tensor,
        fleet_mask: torch.Tensor,
        fleet_target_planet_idx: torch.Tensor,
        planet_inbound_feats: torch.Tensor,
    ) -> PolicyOutput:
        feats = EncodedObs(
            planet_feats=planet_feats,
            planet_mask=planet_mask,
            planet_owned_mask=planet_owned_mask,
            planet_ids=planet_ids,
            planet_garrison=planet_garrison,
            fleet_feats=fleet_feats,
            fleet_mask=fleet_mask,
            global_feats=global_feats,
            fleet_target_planet_idx=fleet_target_planet_idx,
            planet_inbound_feats=planet_inbound_feats,
        )
        with torch.autocast(
            device_type="cuda", dtype=torch.bfloat16, enabled=self.autocast_enabled
        ):
            return self.model(feats, include_value=self.include_value)


class _SourceMajorRolloutForwardKernel(torch.nn.Module):
    def __init__(
        self,
        model: torch.nn.Module,
        *,
        autocast_enabled: bool,
        include_value: bool,
        target_planets: int,
    ) -> None:
        super().__init__()
        self.model = model
        self.autocast_enabled = bool(autocast_enabled)
        self.include_value = bool(include_value)
        self.target_planets = int(target_planets)

    def forward(
        self,
        global_feats: torch.Tensor,
        planet_feats: torch.Tensor,
        planet_mask: torch.Tensor,
        planet_owned_mask: torch.Tensor,
        planet_ids: torch.Tensor,
        planet_garrison: torch.Tensor,
        fleet_feats: torch.Tensor,
        fleet_mask: torch.Tensor,
        fleet_target_planet_idx: torch.Tensor,
        planet_inbound_feats: torch.Tensor,
        actor_source_rows: torch.Tensor,
        actor_source_cols: torch.Tensor,
        actor_source_valid: torch.Tensor,
    ) -> PolicyOutput:
        feats = EncodedObs(
            planet_feats=planet_feats,
            planet_mask=planet_mask,
            planet_owned_mask=planet_owned_mask,
            planet_ids=planet_ids,
            planet_garrison=planet_garrison,
            fleet_feats=fleet_feats,
            fleet_mask=fleet_mask,
            global_feats=global_feats,
            fleet_target_planet_idx=fleet_target_planet_idx,
            planet_inbound_feats=planet_inbound_feats,
        )
        with torch.autocast(
            device_type="cuda", dtype=torch.bfloat16, enabled=self.autocast_enabled
        ):
            return self.model(
                feats,
                include_value=self.include_value,
                actor_source_rows=actor_source_rows,
                actor_source_cols=actor_source_cols,
                actor_source_valid=actor_source_valid,
                target_planets=self.target_planets,
            )


def _get_rollout_kernel(
    model: torch.nn.Module,
    device: torch.device,
    compile_mode: str | None,
    *,
    include_value: bool,
    source_major_actor: bool = False,
    target_planets: int = 0,
    shape_key: tuple[int, ...],
) -> torch.nn.Module:
    mode = compile_mode if device.type == "cuda" else None
    key = ("rollout", mode, bool(include_value), bool(source_major_actor), shape_key)
    cache = _kernel_cache(model)
    cached = cache.get(key)
    if cached is not None:
        return cached
    if source_major_actor:
        kernel = _SourceMajorRolloutForwardKernel(
            model,
            autocast_enabled=device.type == "cuda",
            include_value=include_value,
            target_planets=target_planets,
        )
    else:
        kernel = _DenseRolloutForwardKernel(
            model,
            autocast_enabled=device.type == "cuda",
            include_value=include_value,
        )
    if mode is not None:
        kernel = torch.compile(
            kernel,
            dynamic=False,
            fullgraph=True,
            mode=mode,
        )
    cache[key] = kernel
    return kernel


def _pad_rows(
    t: torch.Tensor,
    rows: int,
    *,
    fill: int | float | bool = 0,
) -> torch.Tensor:
    current = t.shape[0]
    if current >= rows:
        return t
    out = t.new_full((rows, *t.shape[1:]), fill)
    out[:current] = t
    return out


def _pad_encoded_rows(feats: EncodedObs, rows: int) -> EncodedObs:
    if feats.planet_feats.shape[0] >= rows:
        return feats
    return EncodedObs(
        planet_feats=_pad_rows(feats.planet_feats, rows),
        planet_mask=_pad_rows(feats.planet_mask, rows, fill=False),
        planet_owned_mask=_pad_rows(feats.planet_owned_mask, rows, fill=False),
        planet_ids=_pad_rows(feats.planet_ids, rows, fill=-1),
        planet_garrison=_pad_rows(feats.planet_garrison, rows),
        fleet_feats=_pad_rows(feats.fleet_feats, rows),
        fleet_mask=_pad_rows(feats.fleet_mask, rows, fill=False),
        global_feats=None
        if feats.global_feats is None
        else _pad_rows(feats.global_feats, rows),
        fleet_target_planet_idx=None
        if feats.fleet_target_planet_idx is None
        else _pad_rows(feats.fleet_target_planet_idx, rows, fill=-1),
        planet_inbound_feats=None
        if feats.planet_inbound_feats is None
        else _pad_rows(feats.planet_inbound_feats, rows),
        compact_source_rows=feats.compact_source_rows,
        compact_source_cols=feats.compact_source_cols,
        compact_target_planets=feats.compact_target_planets,
    )


def _narrow_encoded_rows(feats: EncodedObs, start: int, length: int) -> EncodedObs:
    compact_source_rows = feats.compact_source_rows
    compact_source_cols = feats.compact_source_cols
    if compact_source_rows is not None and compact_source_cols is not None:
        keep = (compact_source_rows >= start) & (compact_source_rows < start + length)
        compact_source_rows = (compact_source_rows[keep] - start).astype("int64", copy=False)
        compact_source_cols = compact_source_cols[keep].astype("int64", copy=False)
    return EncodedObs(
        planet_feats=feats.planet_feats.narrow(0, start, length),
        planet_mask=feats.planet_mask.narrow(0, start, length),
        planet_owned_mask=feats.planet_owned_mask.narrow(0, start, length),
        planet_ids=feats.planet_ids.narrow(0, start, length),
        planet_garrison=feats.planet_garrison.narrow(0, start, length),
        fleet_feats=feats.fleet_feats.narrow(0, start, length),
        fleet_mask=feats.fleet_mask.narrow(0, start, length),
        global_feats=None
        if feats.global_feats is None
        else feats.global_feats.narrow(0, start, length),
        fleet_target_planet_idx=None
        if feats.fleet_target_planet_idx is None
        else feats.fleet_target_planet_idx.narrow(0, start, length),
        planet_inbound_feats=None
        if feats.planet_inbound_feats is None
        else feats.planet_inbound_feats.narrow(0, start, length),
        compact_source_rows=compact_source_rows,
        compact_source_cols=compact_source_cols,
        compact_target_planets=feats.compact_target_planets,
    )


def _global_feats_or_empty(feats: EncodedObs) -> torch.Tensor:
    if feats.global_feats is not None:
        return feats.global_feats
    batch = feats.planet_feats.shape[0]
    return feats.planet_feats.new_zeros(batch, 0)


def _trim_fleets_for_forward(feats: EncodedObs) -> EncodedObs:
    if feats.planet_inbound_feats is not None and feats.planet_inbound_feats.shape[-2] > 0:
        return slice_encoded_fleet_width(feats, 0)
    return bucket_encoded_fleet_width(feats)


def _bucket_fleets_for_graph(
    feats: EncodedObs,
    *,
    fixed_width: int | None = None,
) -> EncodedObs:
    if feats.planet_inbound_feats is not None and feats.planet_inbound_feats.shape[-2] > 0:
        return slice_encoded_fleet_width(feats, 0)
    used = active_fleet_width(feats.fleet_mask)
    width = (
        int(fixed_width)
        if fixed_width is not None and used <= int(fixed_width)
        else bucket_fleet_width(used)
    )
    return slice_encoded_fleet_width(feats, width)


def _assert_destination_compiled_shape(feats: EncodedObs) -> None:
    inbound = feats.planet_inbound_feats
    if inbound is None:
        raise RuntimeError("destination-conditioned compiled rollout requires inbound summaries")
    if tuple(inbound.shape[-2:]) != (MAX_PLANETS, PLANET_INBOUND_FEAT_DIM):
        raise RuntimeError(
            "destination-conditioned compiled rollout requires inbound summary "
            f"shape (*, {MAX_PLANETS}, {PLANET_INBOUND_FEAT_DIM}), got {tuple(inbound.shape)}"
        )
    if int(feats.fleet_feats.shape[-2]) != 0:
        raise RuntimeError("destination-conditioned compiled rollout must use fleet width 0")


def _slice_policy_output(out: PolicyOutput, rows: int) -> PolicyOutput:
    source_major = out.actor_source_rows is not None
    if not source_major and out.launch_logits.shape[0] == rows:
        return out
    return PolicyOutput(
        launch_logits=out.launch_logits if source_major else out.launch_logits[:rows],
        target_logits=out.target_logits if source_major else out.target_logits[:rows],
        value=out.value[:rows],
        value_logits=out.value_logits[:rows],
        planet_owned_mask=out.planet_owned_mask[:rows],
        planet_mask=out.planet_mask[:rows],
        planet_ids=out.planet_ids[:rows],
        actor_source_rows=out.actor_source_rows,
        actor_source_cols=out.actor_source_cols,
        actor_source_valid=out.actor_source_valid,
        target_planets=out.target_planets,
        action_logit_softcap=out.action_logit_softcap,
        launch_log_std=(
            None
            if out.launch_log_std is None
            else out.launch_log_std
            if source_major
            else out.launch_log_std[:rows]
        ),
        launch_prob_floor=out.launch_prob_floor,
        fraction_alpha=(
            None
            if out.fraction_alpha is None
            else out.fraction_alpha
            if source_major
            else out.fraction_alpha[:rows]
        ),
        fraction_beta=(
            None
            if out.fraction_beta is None
            else out.fraction_beta
            if source_major
            else out.fraction_beta[:rows]
        ),
        fraction_mean=None if out.fraction_mean is None else out.fraction_mean[:rows],
        fraction_log_std=(
            None if out.fraction_log_std is None else out.fraction_log_std[:rows]
        ),
    )


def _empty_traj() -> Trajectory:
    return Trajectory(
        encoded=[], launch=[], target_idx=[], fraction=[],
        log_prob=[], value=[], reward=[], owned_mask=[],
        target_legal_mask=[],
    )


def alternating_learner_seats(
    num_envs: int, num_players: int, *, offset: int = 0
) -> list[int]:
    return [(env_idx + offset) % num_players for env_idx in range(num_envs)]


def _normalize_learner_seats(
    learner_seat: int | Sequence[int],
    num_envs: int,
    num_players: int,
) -> list[int]:
    if isinstance(learner_seat, int):
        seats = [learner_seat] * num_envs
    else:
        seats = [int(seat) for seat in learner_seat]
        if len(seats) != num_envs:
            raise ValueError(
                f"learner_seat has {len(seats)} entries for {num_envs} envs"
            )
    bad = [seat for seat in seats if not 0 <= seat < num_players]
    if bad:
        raise ValueError(f"invalid learner seat(s) for {num_players} players: {bad}")
    return seats


def _resolve_seat_agents(
    opponents_per_env: list[list[OpponentSlot]],
    num_players: int,
    learner_seats: Sequence[int],
) -> list[list[OpponentSlot | None]]:
    """For each env, return a list indexed by seat. `None` marks the
    learner's own seat; self-play opponent seats keep an OpponentSlot
    whose name is LEARNER_NAME (so they batch into the learner forward).
    """
    out: list[list[OpponentSlot | None]] = []
    for env_idx, slots in enumerate(opponents_per_env):
        learner_seat = int(learner_seats[env_idx])
        per_seat: list[OpponentSlot | None] = [None] * num_players
        op_ix = 0
        for seat in range(num_players):
            if seat == learner_seat:
                continue  # already None
            per_seat[seat] = slots[op_ix]
            op_ix += 1
        out.append(per_seat)
    return out


def _finalize_trajectory(
    traj: Trajectory,
    final: Any,
    learner_seat: int,
    reward_cfg: RewardCfg,
) -> None:
    """Write seat_rewards / outcome onto a finished trajectory."""
    if final is None:
        return
    seat_scores = [float(getattr(s, "score", s.reward or 0.0)) for s in final]
    learner_score = seat_scores[learner_seat]
    others = [r for i, r in enumerate(seat_scores) if i != learner_seat]
    margin = learner_score - max(others) if others else learner_score
    traj.final_score = margin
    traj.won = learner_score > max(others) if others else True
    traj.drawn = bool(others) and learner_score == max(others)
    traj.seat_rewards = seat_scores
    traj.learner_seat = learner_seat

    if traj.won:
        outcome = reward_cfg.win_value
    elif traj.drawn:
        outcome = reward_cfg.draw_value
    else:
        outcome = reward_cfg.loss_value
    if traj.reward:
        traj.reward[-1] += outcome + reward_cfg.margin_scale * margin


def _obs_reward_potential(
    obs: Any,
    player: int,
    num_players: int,
    episode_steps: int,
    production_weight: float,
) -> float:
    get = obs.get if isinstance(obs, dict) else lambda key, default=None: getattr(obs, key, default)
    ships = [0.0] * num_players
    production = [0.0] * num_players
    for planet in get("planets", []) or []:
        owner = int(planet[1])
        if owner != -1:
            ships[owner] += float(planet[5])
            production[owner] += float(planet[6])
    for fleet in get("fleets", []) or []:
        owner = int(fleet[1])
        if owner != -1:
            ships[owner] += float(fleet[6])
    step = int(get("step", 0) or 0)
    turns_left = max(0.0, float(episode_steps - step))
    projected = [
        ships[p] + production_weight * turns_left * production[p]
        for p in range(num_players)
    ]
    own = projected[player]
    enemy = max((projected[p] for p in range(num_players) if p != player), default=0.0)
    return own - enemy


def _state_reward_potential(
    state: Any,
    player: int,
    num_players: int,
    episode_steps: int,
    production_weight: float,
) -> float:
    slot = state[player]
    obs = slot["observation"] if isinstance(slot, dict) else slot.observation
    return _obs_reward_potential(
        obs,
        player,
        num_players,
        episode_steps,
        production_weight,
    )


def _state_production_margin(
    state: Any,
    player: int,
    num_players: int,
) -> float:
    slot = state[player]
    obs = slot["observation"] if isinstance(slot, dict) else slot.observation
    return _obs_production_margin(obs, player, num_players)


def _reward_potentials(
    vec: Any,
    states: Sequence[Any],
    rows: list[tuple[int, int]],
    num_players: int,
    episode_steps: int,
    reward_cfg: RewardCfg,
) -> list[float]:
    if not rows:
        return []
    if reward_cfg.signal == "win_terminal":
        return [0.0] * len(rows)
    if reward_cfg.signal == "production_margin":
        native_production = getattr(vec, "production_margins", None)
        if callable(native_production):
            values = native_production(rows)
            return [float(v) for v in values]
        return [
            _state_production_margin(
                states[env_idx],
                player,
                num_players,
            )
            for env_idx, player in rows
        ]
    native = getattr(vec, "reward_potentials", None)
    if callable(native):
        values = native(rows, production_weight=reward_cfg.production_weight)
        return [float(v) for v in values]
    return [
        _state_reward_potential(
            states[env_idx],
            player,
            num_players,
            episode_steps,
            reward_cfg.production_weight,
        )
        for env_idx, player in rows
    ]


def rollout_episodes_batched(
    model: OrbitPolicy,
    vec: VecEnv,
    opponents_per_env: list[list[OpponentSlot]],
    *,
    num_players: int,
    device: str = "cpu",
    learner_seat: int | Sequence[int] = 0,
    deterministic: bool = False,
    reward_cfg: RewardCfg | None = None,
    record_trajectories: bool = True,
    compile_mode: str | None = None,
    compile_fleet_width: int | None = None,
    policy_graph_rows: int | None = None,
    snapshot_compile_rows: int = 64,
    learner_action_agent: Callable[[Any], list[list]] | None = None,
    defer_log_prob: bool = False,
    chunk_records: bool = False,
    timings: dict[str, float] | None = None,
    sample_timings: dict[str, float] | None = None,
) -> list[Trajectory]:
    """Play `len(opponents_per_env)` episodes in parallel; one Trajectory per
    *live-learner seat*.

    A fixed-opponent / evaluation rollout records only each env's designated
    learner seat, so the result is one trajectory per env in env order. A
    self-play rollout (opponent seats whose slot name is LEARNER_NAME) records
    those seats too — each from its own perspective — so an env can contribute
    several trajectories. `Trajectory.env_index` identifies the source env for
    per-game bookkeeping; the designated learner seat satisfies
    `traj.learner_seat == learner_seat[env_index]`.

    `vec` is a long-lived `VecEnv` owned by the caller — we just call
    `vec.reset()` here. Spawning subprocess workers per call would burn
    seconds of pure startup time on each PPO update; the caller creates
    the pool once and reuses it.

    `opponents_per_env[i]` is a list of `num_players-1` `OpponentSlot`s for
    env `i` (the seat order skips that env's learner seat). `learner_seat`
    can be a scalar for legacy single-seat rollouts or a per-env list. The
    opponent assignment is fixed for the whole episode — sampled once by the
    caller before the rollout. `learner_action_agent` is for value pretraining:
    the model still records learner-seat values/actions, but the env executes
    the configured behavior policy's actions for those learner seats.
    """
    num_envs = len(opponents_per_env)
    assert 0 < num_envs <= vec.num_envs, (
        f"opponents_per_env has {num_envs} entries but vec has {vec.num_envs} workers"
    )
    if reward_cfg is None:
        reward_cfg = RewardCfg()
    learner_seats = _normalize_learner_seats(learner_seat, num_envs, num_players)
    seat_agents = _resolve_seat_agents(opponents_per_env, num_players, learner_seats)

    # Record EVERY live-learner seat, not just the designated learner seat: in
    # self-play the opponent seats are the live model too (slot.name ==
    # LEARNER_NAME) and their transitions are equally valid on-policy training
    # data. `recorded_keys` is ordered (env, then seat); for fixed/eval rollouts
    # (no LEARNER_NAME opponents) it degenerates to one designated seat per env,
    # so the returned list stays one-trajectory-per-env in env order.
    recorded_keys = [
        (env_idx, seat)
        for env_idx in range(num_envs)
        for seat in range(num_players)
        if seat_agents[env_idx][seat] is None
        or seat_agents[env_idx][seat].name == LEARNER_NAME
    ]
    trajectories: dict[tuple[int, int], Trajectory] = {
        key: _empty_traj() for key in recorded_keys
    }
    finals: list[Any] = [None] * num_envs

    states = vec.reset()
    dones = [False] * num_envs
    active_envs = list(range(num_envs))
    episode_steps = int(getattr(vec, "episode_steps", 500))
    dense_potential = record_trajectories and reward_cfg.uses_dense_potential()
    previous_potential: dict[tuple[int, int], float] = {key: 0.0 for key in recorded_keys}
    if dense_potential:
        previous_potential = dict(
            zip(
                recorded_keys,
                _reward_potentials(
                    vec,
                    states,
                    recorded_keys,
                    num_players,
                    episode_steps,
                    reward_cfg,
                ),
                strict=True,
            )
        )
    fast_policy_batch = getattr(vec, "policy_batch", None)
    fast_policy_batch_no_context = getattr(vec, "policy_batch_no_context", None)
    fast_observation = getattr(vec, "observation", None)
    fast_observations = getattr(vec, "observations", None)
    fast_step_subset = getattr(vec, "step_subset_fast", None)
    fast_step_flat = getattr(vec, "step_subset_flat_actions", None)
    fast_step_pending = getattr(vec, "step_subset_pending_actions", None)
    fast_builtin_actions = getattr(vec, "builtin_actions", None)
    fast_enqueue_builtin_actions = getattr(vec, "enqueue_builtin_actions", None)
    native_builtin_opponents = set(getattr(vec, "native_builtin_opponents", ()))
    use_fast_numpy_path = (
        bool(getattr(vec, "fast_rollout", False))
        and callable(fast_policy_batch)
        and callable(fast_observation)
        and callable(fast_step_subset)
    )
    use_pending_step = use_fast_numpy_path and callable(fast_step_pending)
    use_flat_step = use_fast_numpy_path and (use_pending_step or callable(fast_step_flat))
    max_policy_rows = max(1, num_envs * num_players)

    while active_envs:
        phase_t0 = perf_counter()
        # 1. Bucket (env, seat, obs) tuples by agent identity.
        learner_bucket: list[tuple[int, int, Any]] = []
        learner_obs_requests: list[tuple[int, tuple[int, int]]] = []
        opp_buckets: dict[str, list[tuple[int, int, Any, OpponentSlot]]] = (
            defaultdict(list)
        )
        pending_opp_obs: list[tuple[int, int]] = []
        pending_opp_slots: list[tuple[int, int, OpponentSlot]] = []
        for env_idx in active_envs:
            state = None if use_fast_numpy_path else states[env_idx]
            for seat in range(num_players):
                slot = seat_agents[env_idx][seat]
                if slot is None or slot.name == LEARNER_NAME:
                    needs_behavior_obs = (
                        learner_action_agent is not None
                        and seat == learner_seats[env_idx]
                    )
                    if use_fast_numpy_path:
                        obs = None
                        if needs_behavior_obs:
                            learner_obs_requests.append(
                                (len(learner_bucket), (env_idx, seat))
                            )
                    else:
                        obs = state[seat]["observation"]
                    learner_bucket.append((env_idx, seat, obs))
                else:
                    can_fast_snapshot = (
                        use_fast_numpy_path
                        and callable(fast_policy_batch_no_context)
                        and callable(getattr(vec, "sample_batch_actions", None))
                        and getattr(slot.agent, "model", None) is not None
                    )
                    if can_fast_snapshot or (
                        use_fast_numpy_path
                        and callable(fast_builtin_actions)
                        and slot.name in native_builtin_opponents
                    ):
                        opp_buckets[slot.name].append((env_idx, seat, None, slot))
                    elif use_fast_numpy_path:
                        pending_opp_obs.append((env_idx, seat))
                        pending_opp_slots.append((env_idx, seat, slot))
                    else:
                        obs = state[seat]["observation"]
                        opp_buckets[slot.name].append((env_idx, seat, obs, slot))

        if pending_opp_obs:
            if callable(fast_observations):
                opponent_obs = fast_observations(pending_opp_obs)
            else:
                opponent_obs = [
                    fast_observation(env_idx, seat)
                    for env_idx, seat in pending_opp_obs
                ]
            for (env_idx, seat, slot), obs in zip(
                pending_opp_slots, opponent_obs, strict=True
            ):
                opp_buckets[slot.name].append((env_idx, seat, obs, slot))

        if learner_obs_requests:
            rows = [row for _bucket_idx, row in learner_obs_requests]
            if callable(fast_observations):
                learner_obs = fast_observations(rows)
            else:
                learner_obs = [
                    fast_observation(env_idx, seat)
                    for env_idx, seat in rows
                ]
            for (bucket_idx, _row), obs in zip(
                learner_obs_requests, learner_obs, strict=True
            ):
                env_idx, seat, _old_obs = learner_bucket[bucket_idx]
                learner_bucket[bucket_idx] = (env_idx, seat, obs)
        _add_timing(timings, "bucket_s", perf_counter() - phase_t0)

        if record_trajectories and len(learner_bucket) > 1:
            learner_bucket.sort(
                key=lambda row: 0 if row[1] == learner_seats[row[0]] else 1
            )

        actions_per_env: dict[int, list[Any]] | None = None
        flat_env_rows: list[int] | None = None
        flat_player_rows: list[int] | None = None
        flat_actions: list[Any] | None = None
        if use_flat_step:
            flat_env_rows = []
            flat_player_rows = []
            flat_actions = []
        else:
            actions_per_env = {
                i: [None] * num_players for i in active_envs
            }

        prefetched_learner: EncodedObs | None = None
        prefetched_snapshots: dict[str, EncodedObs] = {}
        can_prefetch_features = (
            use_fast_numpy_path
            and callable(fast_policy_batch_no_context)
            and learner_action_agent is None
        )
        if can_prefetch_features:
            include_fleet_targets = (
                getattr(getattr(model, "cfg", None), "encoder_backend", None)
                == "destination_conditioned"
            )
            feature_requests: list[tuple[str, list[tuple[int, int, Any]]]] = []
            if learner_bucket:
                feature_requests.append((LEARNER_NAME, learner_bucket))
            for name, bucket in opp_buckets.items():
                agent_model = getattr(bucket[0][3].agent, "model", None)
                if (
                    agent_model is not None
                    and callable(getattr(vec, "sample_batch_actions", None))
                    and getattr(getattr(agent_model, "cfg", None), "encoder_backend", None)
                    == getattr(getattr(model, "cfg", None), "encoder_backend", None)
                ):
                    feature_requests.append(
                        (
                            name,
                            [
                                (env_idx, seat, None)
                                for env_idx, seat, _obs, _slot in bucket
                            ],
                        )
                    )
            if len(feature_requests) > 1:
                phase_t0 = perf_counter()
                all_rows: list[tuple[int, int]] = []
                spans: dict[str, tuple[int, int]] = {}
                for key, bucket in feature_requests:
                    start = len(all_rows)
                    all_rows.extend((env_idx, seat) for env_idx, seat, _obs in bucket)
                    spans[key] = (start, len(bucket))
                prefetched, _contexts = fast_policy_batch_no_context(
                    all_rows,
                    **_policy_batch_kwargs(
                        fast_policy_batch_no_context,
                        device="cpu",
                        pin_memory=torch.device(device).type == "cuda",
                        include_fleet_targets=include_fleet_targets,
                    ),
                )
                _add_timing(timings, "policy_feature_s", perf_counter() - phase_t0)
                for key, (start, length) in spans.items():
                    narrowed = _narrow_encoded_rows(prefetched, start, length)
                    if key == LEARNER_NAME:
                        prefetched_learner = narrowed
                    else:
                        prefetched_snapshots[key] = narrowed

        # 2. One batched forward for the learner identity.
        if learner_bucket:
            phase_t0 = perf_counter()
            _step_learner_bucket(
                model,
                learner_bucket,
                actions_per_env,
                flat_env_rows,
                flat_player_rows,
                flat_actions,
                trajectories,
                device,
                deterministic,
                record_trajectories,
                (
                    fast_policy_batch_no_context
                    if use_fast_numpy_path and callable(fast_policy_batch_no_context)
                    else fast_policy_batch if use_fast_numpy_path else None
                ),
                compile_mode,
                compile_fleet_width,
                policy_graph_rows or max_policy_rows,
                learner_action_agent,
                defer_log_prob=defer_log_prob,
                chunk_records=chunk_records,
                enqueue_native_actions=use_pending_step and learner_action_agent is None,
                timings=timings,
                sample_timings=sample_timings,
                sample_timing_prefix="current_sample",
                preencoded_cpu=prefetched_learner,
            )
            _add_timing(timings, "learner_bucket_s", perf_counter() - phase_t0)

        # 3. Per-snapshot inference. Learned snapshots expose `act_batch`;
        # builtin Python baselines stay on the scalar callable path.
        for _name, bucket in opp_buckets.items():
            agent = bucket[0][3].agent
            if (
                use_fast_numpy_path
                and callable(fast_builtin_actions)
                and _name in native_builtin_opponents
            ):
                phase_t0 = perf_counter()
                rows = [(env_idx, seat) for env_idx, seat, _obs, _slot in bucket]
                if use_pending_step and callable(fast_enqueue_builtin_actions):
                    fast_enqueue_builtin_actions(_name, rows)
                    _add_timing(timings, "builtin_policy_s", perf_counter() - phase_t0)
                    continue
                batched_actions = fast_builtin_actions(
                    _name,
                    rows,
                    native_actions=True,
                )
                for (env_idx, seat, _obs, _slot), acts in zip(
                    bucket, batched_actions, strict=True
                ):
                    if actions_per_env is not None:
                        actions_per_env[env_idx][seat] = acts
                    if flat_env_rows is not None and flat_player_rows is not None and flat_actions is not None:
                        flat_env_rows.append(env_idx)
                        flat_player_rows.append(seat)
                        flat_actions.append(acts)
                _add_timing(timings, "builtin_policy_s", perf_counter() - phase_t0)
                continue
            agent_model = getattr(agent, "model", None)
            if (
                use_fast_numpy_path
                and agent_model is not None
                and callable(fast_policy_batch_no_context)
                and callable(getattr(vec, "sample_batch_actions", None))
            ):
                snapshot_bucket = [
                    (env_idx, seat, None)
                    for env_idx, seat, _obs, _slot in bucket
                ]
                snapshot_compile_mode = getattr(agent, "compile_mode", None)
                snapshot_graph_rows = _snapshot_rollout_graph_rows(
                    len(snapshot_bucket),
                    snapshot_compile_rows=snapshot_compile_rows,
                    compile_mode=snapshot_compile_mode,
                )
                phase_t0 = perf_counter()
                _step_learner_bucket(
                    agent_model,
                    snapshot_bucket,
                    actions_per_env,
                    flat_env_rows,
                    flat_player_rows,
                    flat_actions,
                    trajectories,
                    str(getattr(agent, "device", device)),
                    bool(getattr(agent, "deterministic", deterministic)),
                    False,
                    fast_policy_batch_no_context,
                    snapshot_compile_mode,
                    None,
                    snapshot_graph_rows,
                    enqueue_native_actions=use_pending_step,
                    timings=timings,
                    sample_timings=sample_timings,
                    sample_timing_prefix="snapshot_sample",
                    preencoded_cpu=prefetched_snapshots.get(_name),
                )
                _add_timing(timings, "snapshot_bucket_s", perf_counter() - phase_t0)
                continue
            phase_t0 = perf_counter()
            act_batch = getattr(agent, "act_batch", None)
            if callable(act_batch):
                obs_list = [obs for _env_idx, _seat, obs, _slot in bucket]
                batched_actions = act_batch(obs_list)
                for (env_idx, seat, _obs, _slot), acts in zip(
                    bucket, batched_actions, strict=True
                ):
                    if actions_per_env is not None:
                        actions_per_env[env_idx][seat] = acts
                    if flat_env_rows is not None and flat_player_rows is not None and flat_actions is not None:
                        flat_env_rows.append(env_idx)
                        flat_player_rows.append(seat)
                        flat_actions.append(acts)
            else:
                for env_idx, seat, obs, slot in bucket:
                    acts = slot.agent(obs)
                    if actions_per_env is not None:
                        actions_per_env[env_idx][seat] = acts
                    if flat_env_rows is not None and flat_player_rows is not None and flat_actions is not None:
                        flat_env_rows.append(env_idx)
                        flat_player_rows.append(seat)
                        flat_actions.append(acts)
            _add_timing(timings, "python_policy_s", perf_counter() - phase_t0)

        # 4. Step alive envs in parallel.
        active = active_envs
        phase_t0 = perf_counter()
        if use_flat_step:
            if flat_env_rows is None or flat_player_rows is None or flat_actions is None:
                raise RuntimeError("flat rollout step missing action rows")
            if use_pending_step:
                results = fast_step_pending(active, flat_env_rows, flat_player_rows, flat_actions)
            else:
                results = fast_step_flat(active, flat_env_rows, flat_player_rows, flat_actions)
        else:
            if actions_per_env is None:
                raise RuntimeError("nested rollout step missing action rows")
            actions_list = [actions_per_env[i] for i in active]
            step_subset = fast_step_subset if use_fast_numpy_path else vec.step_subset
            results = step_subset(active, actions_list)
        _add_timing(timings, "env_step_s", perf_counter() - phase_t0)
        for i, (state, done, final) in results.items():
            if state is not None:
                states[i] = state
            if done:
                dones[i] = True
                finals[i] = final
        active_envs = [idx for idx in active_envs if not dones[idx]]
        if dense_potential:
            phase_t0 = perf_counter()
            active_set = set(active)
            rows = [
                (env_idx, seat)
                for (env_idx, seat) in recorded_keys
                if env_idx in active_set and trajectories[(env_idx, seat)].reward
            ]
            current_potential = _reward_potentials(
                vec,
                states,
                rows,
                num_players,
                episode_steps,
                reward_cfg,
            )
            for (env_idx, seat), phi in zip(rows, current_potential, strict=True):
                key = (env_idx, seat)
                trajectories[key].reward[-1] += reward_cfg.potential_weight * (
                    phi - previous_potential[key]
                )
                previous_potential[key] = phi
            _add_timing(timings, "reward_s", perf_counter() - phase_t0)

    # 5. Apply terminal reward + record seat_rewards on each trajectory. Each
    # recorded seat is finalized from its OWN perspective (margin/outcome), so a
    # self-play game contributes one trajectory per live-learner seat.
    for env_idx, seat in recorded_keys:
        traj = trajectories[(env_idx, seat)]
        traj.env_index = env_idx
        _finalize_trajectory(traj, finals[env_idx], seat, reward_cfg)

    return [trajectories[key] for key in recorded_keys]


def _step_learner_bucket(
    model: OrbitPolicy,
    bucket: list[tuple[int, int, Any]],
    actions_per_env: dict[int, list[Any]] | None,
    flat_env_rows: list[int] | None,
    flat_player_rows: list[int] | None,
    flat_actions: list[Any] | None,
    trajectories: dict[tuple[int, int], Trajectory],
    device: str,
    deterministic: bool,
    record_trajectories: bool,
    policy_batch: Any | None = None,
    compile_mode: str | None = None,
    compile_fleet_width: int | None = None,
    graph_rows: int | None = None,
    learner_action_agent: Callable[[Any], list[list]] | None = None,
    defer_log_prob: bool = False,
    chunk_records: bool = False,
    enqueue_native_actions: bool = False,
    timings: dict[str, float] | None = None,
    sample_timings: dict[str, float] | None = None,
    sample_timing_prefix: str | None = None,
    preencoded_cpu: EncodedObs | None = None,
) -> None:
    """Encode + batch-forward the learner identity across (env, seat) pairs.

    Self-play opponent seats batch in here too for compute efficiency, but
    only each env's configured learner seat gets recorded into its trajectory
    — PPO trains on the learner-seat transitions, not the self-play side's.
    """
    raw_obs_list = [obs for _, _, obs in bucket]
    target_device = torch.device(device)
    record_on_cpu = record_trajectories and target_device.type == "cuda"
    graph_enabled = target_device.type == "cuda" and compile_mode is not None
    fixed_graph_fleet_width = (
        int(compile_fleet_width)
        if graph_enabled and compile_fleet_width is not None
        else None
    )
    if graph_enabled and graph_rows is not None:
        graph_rows = _capped_graph_rows(len(bucket), max_rows=int(graph_rows))
    else:
        graph_rows = max(int(graph_rows or len(bucket)), len(bucket))
    include_fleet_targets = (
        getattr(getattr(model, "cfg", None), "encoder_backend", None)
        == "destination_conditioned"
    )
    if graph_enabled and include_fleet_targets:
        fixed_graph_fleet_width = 0
    sync_timing = sample_timings is not None
    action_contexts: list[ActionContext] | None = None
    policy_rows: list[tuple[int, int]] | None = None
    source_index_stacked: EncodedObs | None = None
    fast_sampler = getattr(getattr(policy_batch, "__self__", None), "sample_batch_with_records", None)
    fast_actions_sampler = getattr(getattr(policy_batch, "__self__", None), "sample_batch_actions", None)
    phase_t0 = perf_counter()
    if preencoded_cpu is not None:
        if preencoded_cpu.planet_feats.shape[0] != len(bucket):
            raise ValueError("preencoded rollout batch row count must match bucket")
        if callable(policy_batch):
            policy_rows = [(env_idx, seat) for env_idx, seat, _obs in bucket]
        if target_device.type == "cuda":
            cpu_stacked = preencoded_cpu
            source_index_stacked = cpu_stacked
            device_source = (
                _bucket_fleets_for_graph(cpu_stacked, fixed_width=fixed_graph_fleet_width)
                if graph_enabled
                else _trim_fleets_for_forward(cpu_stacked)
            )
            if graph_enabled:
                device_source = _pad_encoded_rows(device_source, graph_rows)
            stacked = _encoded_to_device(device_source, target_device)
            if not record_on_cpu:
                cpu_stacked = None
        else:
            stacked = preencoded_cpu
            if stacked.planet_feats.device != target_device:
                stacked = _encoded_to_device(stacked, target_device)
            cpu_stacked = stacked if record_trajectories else None
            if stacked.planet_feats.device.type == "cpu":
                source_index_stacked = stacked
            if graph_enabled:
                stacked = _bucket_fleets_for_graph(
                    stacked,
                    fixed_width=fixed_graph_fleet_width,
                )
                stacked = _pad_encoded_rows(stacked, graph_rows)
            else:
                stacked = _trim_fleets_for_forward(stacked)
    elif callable(policy_batch):
        policy_rows = [(env_idx, seat) for env_idx, seat, _obs in bucket]
        if target_device.type == "cuda":
            cpu_stacked, action_contexts = policy_batch(
                policy_rows,
                **_policy_batch_kwargs(
                    policy_batch,
                    device="cpu",
                    pin_memory=True,
                    include_fleet_targets=include_fleet_targets,
                ),
            )
            source_index_stacked = cpu_stacked
            device_source = (
                _bucket_fleets_for_graph(cpu_stacked, fixed_width=fixed_graph_fleet_width)
                if graph_enabled
                else _trim_fleets_for_forward(cpu_stacked)
            )
            if graph_enabled:
                device_source = _pad_encoded_rows(device_source, graph_rows)
            stacked = _encoded_to_device(device_source, target_device)
            if not record_on_cpu:
                cpu_stacked = None
        else:
            stacked, action_contexts = policy_batch(
                policy_rows,
                **_policy_batch_kwargs(
                    policy_batch,
                    device=device,
                    pin_memory=target_device.type == "cuda",
                    include_fleet_targets=include_fleet_targets,
                ),
            )
            if stacked.planet_feats.device != target_device:
                stacked = _encoded_to_device(stacked, target_device)
            cpu_stacked = stacked if record_trajectories else None
            if stacked.planet_feats.device.type == "cpu":
                source_index_stacked = stacked
            if not graph_enabled:
                stacked = _trim_fleets_for_forward(stacked)
    else:
        cpu_stacked = (
            encode_raw_observations(
                raw_obs_list,
                device="cpu",
                include_fleet_targets=include_fleet_targets,
            )
            if record_on_cpu
            else None
        )
        if cpu_stacked is not None:
            source_index_stacked = cpu_stacked
            device_source = (
                _bucket_fleets_for_graph(cpu_stacked, fixed_width=fixed_graph_fleet_width)
                if graph_enabled
                else _trim_fleets_for_forward(cpu_stacked)
            )
            if graph_enabled:
                device_source = _pad_encoded_rows(device_source, graph_rows)
            stacked = _encoded_to_device(device_source, target_device)
        else:
            stacked = encode_raw_observations(
                raw_obs_list,
                device=device,
                pin_memory=target_device.type == "cuda",
                include_fleet_targets=include_fleet_targets,
            )
            cpu_stacked = (
                stacked
                if record_trajectories and target_device.type == "cpu"
                else None
            )
            if stacked.planet_feats.device.type == "cpu":
                source_index_stacked = stacked
            if graph_enabled:
                stacked = _bucket_fleets_for_graph(
                    stacked,
                    fixed_width=fixed_graph_fleet_width,
                )
                stacked = _pad_encoded_rows(stacked, graph_rows)
            else:
                stacked = _trim_fleets_for_forward(stacked)
    _sync_cuda_timing(target_device, sync_timing)
    _add_timing(timings, "policy_feature_s", perf_counter() - phase_t0)
    if cpu_stacked is not None:
        real_rows = cpu_stacked.planet_feats.shape[0]
    else:
        real_rows = len(bucket)
        if stacked.planet_feats.shape[0] < real_rows:
            raise RuntimeError("encoded rollout batch has fewer rows than bucket")
    learner_rows: list[int] = []
    learner_keys: list[tuple[int, int]] = []
    if record_trajectories:
        # The learner bucket holds ONLY live-learner seats (the designated learner
        # seat plus any LEARNER_NAME self-play opponents), so every row is recorded
        # into its own (env, seat)-keyed trajectory.
        learner_rows = list(range(len(bucket)))
        learner_keys = [(bucket[k][0], bucket[k][1]) for k in learner_rows]
    record_source_mask_full = None
    compact_source_rows = None
    compact_source_cols = None
    compact_target_planets = None
    compact_max_sources_per_row = 1
    if (
        source_index_stacked is not None
        and source_index_stacked.planet_feats.device.type == "cpu"
        and policy_rows is not None
    ):
        record_source_mask_full = (
            source_index_stacked.planet_owned_mask & source_index_stacked.planet_mask
        )
        source_rows, source_cols = torch.nonzero(record_source_mask_full, as_tuple=True)
        compact_source_rows = source_rows.numpy()
        compact_source_cols = source_cols.numpy()
        if record_source_mask_full.shape[0] > 0:
            compact_max_sources_per_row = max(
                1,
                int(record_source_mask_full.sum(dim=1).max().item()),
            )
        compact_target_planets = source_index_stacked.compact_target_planets
        if compact_target_planets is None:
            compact_target_planets = int(source_index_stacked.planet_mask.shape[1])
            live_planet_cols = torch.nonzero(
                source_index_stacked.planet_mask.any(dim=0),
                as_tuple=False,
            )
            if live_planet_cols.numel():
                compact_target_planets = int(live_planet_cols[-1].item()) + 1
    record_value_only = record_trajectories and learner_action_agent is not None
    record_values = record_trajectories and (record_value_only or not defer_log_prob)

    # CUDA rollout uses a fixed padded batch so Inductor can reuse one static
    # graph even as envs finish and the real learner bucket shrinks.
    graph_stacked = _pad_encoded_rows(stacked, graph_rows) if graph_enabled else stacked
    if graph_enabled and include_fleet_targets:
        _assert_destination_compiled_shape(graph_stacked)
    source_major_actor = (
        policy_rows is not None
        and compact_source_rows is not None
        and compact_source_cols is not None
        and callable(fast_sampler)
        and getattr(model.cfg, "action_logit_softcap", None) is not None
        and record_trajectories
        and not record_value_only
        and defer_log_prob
        and (
            not graph_enabled
            or compact_max_sources_per_row <= _SOURCE_MAJOR_COMPILE_ROW_CAP
        )
    )
    target_planets_for_actor = int(
        compact_target_planets
        if compact_target_planets is not None
        else graph_stacked.planet_feats.shape[1]
    )
    if graph_enabled:
        target_planets_for_actor = int(graph_stacked.planet_feats.shape[1])
    actor_source_rows_t = torch.empty(0, dtype=torch.long, device=target_device)
    actor_source_cols_t = torch.empty(0, dtype=torch.long, device=target_device)
    actor_source_valid_t = torch.empty(0, dtype=torch.bool, device=target_device)
    if source_major_actor:
        source_count = int(np.asarray(compact_source_rows).shape[0])
        source_rows_np = np.asarray(compact_source_rows, dtype=np.int64)
        source_cols_np = np.asarray(compact_source_cols, dtype=np.int64)
        source_rows_t = torch.as_tensor(source_rows_np, dtype=torch.long)
        source_cols_t = torch.as_tensor(source_cols_np, dtype=torch.long)
        source_valid_t = torch.ones(source_count, dtype=torch.bool)
        if graph_enabled:
            source_rows_cap = _source_graph_rows(
                source_count,
                graph_rows=int(graph_stacked.planet_feats.shape[0]),
                planets=int(graph_stacked.planet_feats.shape[1]),
                max_sources_per_row=compact_max_sources_per_row,
            )
            if source_count < source_rows_cap:
                pad = source_rows_cap - source_count
                source_rows_t = torch.cat(
                    (source_rows_t, torch.zeros(pad, dtype=torch.long)),
                    dim=0,
                )
                source_cols_t = torch.cat(
                    (source_cols_t, torch.zeros(pad, dtype=torch.long)),
                    dim=0,
                )
                source_valid_t = torch.cat(
                    (source_valid_t, torch.zeros(pad, dtype=torch.bool)),
                    dim=0,
                )
        actor_source_rows_t = source_rows_t.to(target_device, non_blocking=True)
        actor_source_cols_t = source_cols_t.to(target_device, non_blocking=True)
        actor_source_valid_t = source_valid_t.to(target_device, non_blocking=True)
    kernel = _get_rollout_kernel(
        model,
        target_device,
        compile_mode if graph_enabled else None,
        include_value=record_values,
        source_major_actor=source_major_actor,
        target_planets=target_planets_for_actor if source_major_actor else 0,
        shape_key=(
            int(graph_stacked.planet_feats.shape[0]),
            int(graph_stacked.planet_feats.shape[1]),
            int(graph_stacked.fleet_feats.shape[1]),
            int(_global_feats_or_empty(graph_stacked).shape[1]),
            int(planet_inbound_feats_or_empty(graph_stacked).shape[-2]),
            int(planet_inbound_feats_or_empty(graph_stacked).shape[-1]),
            int(actor_source_rows_t.shape[0]) if source_major_actor else 0,
            target_planets_for_actor if source_major_actor else 0,
        ),
    )
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            if graph_enabled:
                _mark_cuda_graph_step(target_device)
            _sync_cuda_timing(target_device, sync_timing)
            phase_t0 = perf_counter()
            kernel_args = (
                _global_feats_or_empty(graph_stacked),
                graph_stacked.planet_feats,
                graph_stacked.planet_mask,
                graph_stacked.planet_owned_mask,
                graph_stacked.planet_ids,
                graph_stacked.planet_garrison,
                graph_stacked.fleet_feats,
                graph_stacked.fleet_mask,
                fleet_target_planet_idx_or_empty(graph_stacked),
                planet_inbound_feats_or_empty(graph_stacked),
            )
            if source_major_actor:
                out = kernel(
                    *kernel_args,
                    actor_source_rows_t,
                    actor_source_cols_t,
                    actor_source_valid_t,
                )
            else:
                out = kernel(*kernel_args)
            _sync_cuda_timing(target_device, sync_timing)
            _add_timing(timings, "policy_forward_s", perf_counter() - phase_t0)
    finally:
        model.train(was_training)
    out = _slice_policy_output(out, real_rows)

    scoped_sample_timings = _scoped_timings(sample_timings, sample_timing_prefix)
    phase_t0 = perf_counter()
    if record_trajectories and not record_value_only:
        if callable(fast_sampler) and policy_rows is not None:
            record_source_mask = None
            if record_source_mask_full is not None and learner_rows:
                record_source_mask = record_source_mask_full[learner_rows].numpy()
            actions_list, records = fast_sampler(
                out,
                policy_rows,
                deterministic=deterministic,
                record_rows=learner_rows,
                record_source_mask=record_source_mask,
                native_actions=True,
                enqueue_actions=enqueue_native_actions,
                compute_log_prob=not defer_log_prob,
                compact_legal_records=chunk_records,
                compact_source_rows=compact_source_rows,
                compact_source_cols=compact_source_cols,
                compact_target_planets=compact_target_planets,
                timings=scoped_sample_timings,
            )
        elif action_contexts is not None:
            actions_list, records = sample_batch_with_records_context(
                out,
                action_contexts,
                deterministic=deterministic,
                record_rows=learner_rows,
            )
        else:
            actions_list, records = sample_batch_with_records_raw(
                out,
                raw_obs_list,
                deterministic=deterministic,
                record_rows=learner_rows,
            )
    else:
        if callable(fast_actions_sampler) and policy_rows is not None:
            actions_list = fast_actions_sampler(
                out,
                policy_rows,
                deterministic=deterministic,
                native_actions=True,
                enqueue_actions=enqueue_native_actions,
                timings=scoped_sample_timings,
                compact_source_rows=compact_source_rows,
                compact_source_cols=compact_source_cols,
                compact_target_planets=compact_target_planets,
            )
        elif action_contexts is not None:
            actions_list = sample_batch_actions_context(
                out,
                action_contexts,
                deterministic=deterministic,
            )
        else:
            actions_list = sample_batch_actions_raw(
                out,
                raw_obs_list,
                deterministic=deterministic,
            )
        records = []
    _flush_scoped_timings(sample_timings, scoped_sample_timings, sample_timing_prefix)
    _sync_cuda_timing(target_device, sync_timing)
    _add_timing(timings, "policy_sample_s", perf_counter() - phase_t0)

    if learner_action_agent is not None and learner_rows:
        phase_t0 = perf_counter()
        override_obs = [raw_obs_list[k] for k in learner_rows]
        act_batch = getattr(learner_action_agent, "act_batch", None)
        if callable(act_batch):
            override_actions = act_batch(override_obs)
        else:
            override_actions = [learner_action_agent(obs) for obs in override_obs]
        for row, acts in zip(learner_rows, override_actions, strict=True):
            actions_list[row] = acts
        _add_timing(timings, "policy_override_s", perf_counter() - phase_t0)

    if record_trajectories and learner_rows:
        phase_t0 = perf_counter()
        row_idx = torch.as_tensor(
            learner_rows, device=out.value.device, dtype=torch.long
        )
        if record_value_only:
            rec = _materialize_value_records_cpu(
                stacked,
                cpu_stacked,
                out,
                row_idx,
                learner_rows,
            )
            for j, (env_idx, seat) in enumerate(learner_keys):
                traj = trajectories[(env_idx, seat)]
                traj.encoded.append(
                    EncodedObs(
                        planet_feats=rec["planet_feats"][j],
                        planet_mask=rec["planet_mask"][j],
                        planet_owned_mask=rec["planet_owned_mask"][j],
                        planet_ids=rec["planet_ids"][j],
                        planet_garrison=rec["planet_garrison"][j],
                        fleet_feats=rec["fleet_feats"][j],
                        fleet_mask=rec["fleet_mask"][j],
                        global_feats=rec["global_feats"][j],
                        fleet_target_planet_idx=None
                        if rec["fleet_target_planet_idx"] is None
                        else rec["fleet_target_planet_idx"][j],
                        planet_inbound_feats=None
                        if rec["planet_inbound_feats"] is None
                        else rec["planet_inbound_feats"][j],
                    )
                )
                traj.value.append(rec["value"][j])
                traj.reward.append(0.0)
        else:
            rec = _materialize_records_cpu(
                stacked,
                cpu_stacked,
                out,
                records,
                row_idx,
                learner_rows,
                include_log_prob=not defer_log_prob,
                include_value=record_values,
                timings=timings,
            )
            if chunk_records:
                for j, (env_idx, seat) in enumerate(learner_keys):
                    traj = trajectories[(env_idx, seat)]
                    traj.record_refs.append(TrajectoryRecordRef(rec, j))
                    traj.reward.append(0.0)
            else:
                for j, (env_idx, seat) in enumerate(learner_keys):
                    traj = trajectories[(env_idx, seat)]
                    traj.encoded.append(
                        EncodedObs(
                            planet_feats=rec["planet_feats"][j],
                            planet_mask=rec["planet_mask"][j],
                            planet_owned_mask=rec["planet_owned_mask"][j],
                            planet_ids=rec["planet_ids"][j],
                            planet_garrison=rec["planet_garrison"][j],
                            fleet_feats=rec["fleet_feats"][j],
                            fleet_mask=rec["fleet_mask"][j],
                            global_feats=rec["global_feats"][j],
                            fleet_target_planet_idx=None
                            if rec["fleet_target_planet_idx"] is None
                            else rec["fleet_target_planet_idx"][j],
                            planet_inbound_feats=None
                            if rec["planet_inbound_feats"] is None
                            else rec["planet_inbound_feats"][j],
                        )
                    )
                    traj.launch.append(rec["launch"][j])
                    traj.target_idx.append(rec["target_idx"][j])
                    traj.fraction.append(rec["fraction"][j])
                    if rec["log_prob"] is not None:
                        traj.log_prob.append(rec["log_prob"][j])
                    if rec["value"] is not None:
                        traj.value.append(rec["value"][j])
                    traj.owned_mask.append(rec["planet_owned_mask"][j] & rec["planet_mask"][j])
                    traj.target_legal_mask.append(rec["target_legal_mask"][j])
                    traj.reward.append(0.0)
        _add_timing(timings, "policy_record_s", perf_counter() - phase_t0)

    if actions_list is not None:
        for k, (env_idx, seat, _obs) in enumerate(bucket):
            acts = actions_list[k]
            if actions_per_env is not None:
                actions_per_env[env_idx][seat] = acts
            if (
                flat_env_rows is not None
                and flat_player_rows is not None
                and flat_actions is not None
            ):
                flat_env_rows.append(env_idx)
                flat_player_rows.append(seat)
                flat_actions.append(acts)


def _materialize_value_records_cpu(
    stacked: EncodedObs,
    cpu_stacked: EncodedObs | None,
    out: Any,
    row_idx: torch.Tensor,
    rows: list[int],
) -> dict[str, torch.Tensor]:
    """Copy value-pretrain rollout records to CPU without action sidecars."""
    feature_source = cpu_stacked if cpu_stacked is not None else stacked
    feature_rows = _feature_row_index(feature_source, row_idx, rows)
    value_cpu = out.value.index_select(0, row_idx).detach().cpu()
    return {
        "planet_feats": _materialize_feature_field(feature_source.planet_feats, feature_rows),
        "planet_mask": _materialize_feature_field(feature_source.planet_mask, feature_rows),
        "planet_owned_mask": _materialize_feature_field(
            feature_source.planet_owned_mask,
            feature_rows,
        ),
        "planet_ids": _materialize_feature_field(feature_source.planet_ids, feature_rows),
        "planet_garrison": _materialize_feature_field(
            feature_source.planet_garrison,
            feature_rows,
        ),
        "fleet_feats": _materialize_feature_field(feature_source.fleet_feats, feature_rows),
        "fleet_mask": _materialize_feature_field(feature_source.fleet_mask, feature_rows),
        "fleet_target_planet_idx": None
        if feature_source.fleet_target_planet_idx is None
        else _materialize_feature_field(
            feature_source.fleet_target_planet_idx,
            feature_rows,
        ),
        "planet_inbound_feats": None
        if feature_source.planet_inbound_feats is None
        else _materialize_feature_field(feature_source.planet_inbound_feats, feature_rows),
        "global_feats": _materialize_feature_field(
            _global_feats_or_empty(feature_source),
            feature_rows,
        ),
        "value": value_cpu,
    }


def _feature_row_index(
    feature_source: EncodedObs,
    row_idx: torch.Tensor,
    rows: list[int],
) -> tuple[int, int] | torch.Tensor:
    if feature_source.planet_feats.device.type != "cpu":
        return row_idx
    if rows:
        start = int(rows[0])
        if all(int(row) == start + idx for idx, row in enumerate(rows)):
            return (start, len(rows))
    return torch.as_tensor(rows, dtype=torch.long)


def _materialize_feature_field(
    field: torch.Tensor,
    rows: tuple[int, int] | torch.Tensor,
) -> torch.Tensor:
    if isinstance(rows, tuple):
        start, length = rows
        selected = field.narrow(0, start, length)
        if field.device.type == "cpu":
            return selected.detach().clone()
        return selected.detach().cpu()
    return field.index_select(0, rows).detach().cpu()


def _materialize_records_cpu(
    stacked: EncodedObs,
    cpu_stacked: EncodedObs | None,
    out: Any,
    records: list[Any],
    row_idx: torch.Tensor,
    rows: list[int],
    *,
    include_log_prob: bool = True,
    include_value: bool = True,
    timings: dict[str, float] | None = None,
) -> dict[str, torch.Tensor]:
    """Copy learner rollout records to CPU once per field per env step.

    Keeping every per-step feature tensor on CUDA caps rollout parallelism and
    leaves thousands of small device allocations alive until PPO batching.
    The action sampler already synchronizes for Python env actions, so this
    moves trajectory storage off VRAM at the same loop boundary.
    """
    feature_source = cpu_stacked if cpu_stacked is not None else stacked
    feature_rows = _feature_row_index(feature_source, row_idx, rows)
    if isinstance(records, list):
        launch = torch.stack([records[k].launch for k in rows])
        target_idx = torch.stack([records[k].target_idx for k in rows])
        fraction = torch.stack([records[k].fraction for k in rows])
        log_prob = torch.stack([records[k].log_prob for k in rows])
        target_legal_mask = torch.stack([records[k].target_legal_mask for k in rows])
    else:
        launch = records.launch
        target_idx = records.target_idx
        fraction = records.fraction
        log_prob = records.log_prob
        target_legal_mask = records.target_legal_mask
    compact_target_legal = hasattr(records, "target_legal_source_mask")
    if target_legal_mask is None and not compact_target_legal:
        raise RuntimeError("rollout records are missing dense target legality")
    keep_log_prob = include_log_prob or bool(
        getattr(records, "old_log_prob_computed", False)
    )
    b, p = target_idx.shape
    value = out.value.index_select(0, row_idx) if include_value else None
    launch_is_cpu = launch.device.type == "cpu"
    fraction_is_cpu = fraction.device.type == "cpu"
    mask_is_cpu = target_legal_mask is None or target_legal_mask.device.type == "cpu"
    log_prob_is_cpu = log_prob.device.type == "cpu"
    record_is_cpu = target_idx.device.type == "cpu"
    phase_t0 = perf_counter()
    if record_is_cpu:
        value_cpu = value.detach().cpu() if value is not None else None
        target_idx_cpu = target_idx.detach().cpu().long()
        launch_cpu = launch.detach().cpu()
        fraction_cpu = fraction.detach().cpu()
        log_prob_cpu = log_prob.detach().cpu() if keep_log_prob else None
        target_legal_mask_cpu = (
            None if target_legal_mask is None else target_legal_mask.detach().cpu()
        )
    else:
        flat_parts = [target_idx.float()]
        if not launch_is_cpu:
            flat_parts.append(launch.float())
        if not fraction_is_cpu:
            flat_parts.append(fraction.float())
        if value is not None:
            flat_parts.append(value.float().unsqueeze(1))
        if keep_log_prob and not log_prob_is_cpu:
            flat_parts.append(log_prob.float())
        if target_legal_mask is not None and not mask_is_cpu:
            flat_parts.append(target_legal_mask.float().reshape(b, -1))
        flat = torch.cat(tuple(flat_parts), dim=1).detach().cpu()
        pos = 0
        target_idx_cpu = flat[:, pos : pos + p].long()
        pos += p
        if launch_is_cpu:
            launch_cpu = launch.detach().cpu()
        else:
            launch_cpu = flat[:, pos : pos + p]
            pos += p
        if fraction_is_cpu:
            fraction_cpu = fraction.detach().cpu()
        else:
            fraction_cpu = flat[:, pos : pos + p]
            pos += p
        if value is None:
            value_cpu = None
        else:
            value_cpu = flat[:, pos]
            pos += 1
        if keep_log_prob and log_prob_is_cpu:
            log_prob_cpu = log_prob.detach().cpu()
        elif keep_log_prob:
            log_prob_cpu = flat[:, pos : pos + p]
            pos += p
        else:
            log_prob_cpu = None
        if target_legal_mask is None:
            target_legal_mask_cpu = None
        elif mask_is_cpu:
            target_legal_mask_cpu = target_legal_mask.detach().cpu()
        else:
            target_legal_width = p * p
            target_legal_mask_cpu = flat[:, pos : pos + target_legal_width].reshape(
                b,
                p,
                p,
            ).bool()
    _add_timing(timings, "policy_record/action_fields_s", perf_counter() - phase_t0)

    phase_t0 = perf_counter()
    out = {
        "planet_feats": _materialize_feature_field(feature_source.planet_feats, feature_rows),
        "planet_mask": _materialize_feature_field(feature_source.planet_mask, feature_rows),
        "planet_owned_mask": _materialize_feature_field(
            feature_source.planet_owned_mask,
            feature_rows,
        ),
        "planet_ids": _materialize_feature_field(feature_source.planet_ids, feature_rows),
        "planet_garrison": _materialize_feature_field(
            feature_source.planet_garrison,
            feature_rows,
        ),
        "fleet_feats": _materialize_feature_field(feature_source.fleet_feats, feature_rows),
        "fleet_mask": _materialize_feature_field(feature_source.fleet_mask, feature_rows),
        "fleet_target_planet_idx": None
        if feature_source.fleet_target_planet_idx is None
        else _materialize_feature_field(
            feature_source.fleet_target_planet_idx,
            feature_rows,
        ),
        "planet_inbound_feats": None
        if feature_source.planet_inbound_feats is None
        else _materialize_feature_field(feature_source.planet_inbound_feats, feature_rows),
        "global_feats": _materialize_feature_field(
            _global_feats_or_empty(feature_source),
            feature_rows,
        ),
        "target_idx": target_idx_cpu,
        "launch": launch_cpu,
        "fraction": fraction_cpu,
        "log_prob": log_prob_cpu,
        "value": value_cpu,
        "target_legal_mask": target_legal_mask_cpu,
    }
    _add_timing(timings, "policy_record/features_s", perf_counter() - phase_t0)
    out["old_log_prob_computed"] = bool(
        getattr(records, "old_log_prob_computed", False)
    )
    out["values_computed"] = bool(include_value)
    if compact_target_legal:
        phase_t0 = perf_counter()
        out["target_legal_row_idx"] = records.target_legal_row_idx.detach().cpu().long()
        out["target_legal_source_idx"] = (
            records.target_legal_source_idx.detach().cpu().long()
        )
        out["target_legal_source_mask"] = (
            records.target_legal_source_mask.detach().cpu().bool()
        )
        _add_timing(timings, "policy_record/compact_legal_s", perf_counter() - phase_t0)
    if hasattr(records, "source_row_idx"):
        phase_t0 = perf_counter()
        out["source_row_idx"] = records.source_row_idx.detach().cpu().long()
        out["source_col_idx"] = records.source_col_idx.detach().cpu().long()
        out["source_row_offsets"] = records.source_row_offsets.detach().cpu().long()
        out["source_launch"] = records.source_launch.detach().cpu().float()
        out["source_raw_launch"] = records.source_raw_launch.detach().cpu().float()
        out["source_target_idx"] = records.source_target_idx.detach().cpu().long()
        out["source_fraction"] = records.source_fraction.detach().cpu().float()
        if hasattr(records, "source_log_prob"):
            out["source_log_prob"] = records.source_log_prob.detach().cpu().float()
        if hasattr(records, "source_target_legal_mask"):
            out["source_target_legal_mask"] = (
                records.source_target_legal_mask.detach().cpu().bool()
            )
        _add_timing(timings, "policy_record/source_records_s", perf_counter() - phase_t0)
    return out


def _encoded_to_device(feats: EncodedObs, device: torch.device) -> EncodedObs:
    if device.type != "cuda":
        return feats.to(device)

    def move(t: torch.Tensor) -> torch.Tensor:
        if t.device == device:
            return t
        if t.device.type == "cpu":
            if not t.is_pinned():
                t = t.pin_memory()
            return t.to(device, non_blocking=True)
        return t.to(device, non_blocking=True)

    return EncodedObs(
        planet_feats=move(feats.planet_feats),
        planet_mask=move(feats.planet_mask),
        planet_owned_mask=move(feats.planet_owned_mask),
        planet_ids=move(feats.planet_ids),
        planet_garrison=move(feats.planet_garrison),
        fleet_feats=move(feats.fleet_feats),
        fleet_mask=move(feats.fleet_mask),
        global_feats=None if feats.global_feats is None else move(feats.global_feats),
        fleet_target_planet_idx=None
        if feats.fleet_target_planet_idx is None
        else move(feats.fleet_target_planet_idx),
        planet_inbound_feats=None
        if feats.planet_inbound_feats is None
        else move(feats.planet_inbound_feats),
        compact_source_rows=feats.compact_source_rows,
        compact_source_cols=feats.compact_source_cols,
        compact_target_planets=feats.compact_target_planets,
    )
