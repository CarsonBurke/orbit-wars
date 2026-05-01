"""Vectorized rollouts: N parallel envs + one batched policy forward per step.

Per env-step the orchestrator does:

  1. Walk every alive (env, seat) and bucket the obs by *agent identity*
     — `LEARNER_NAME` for the live model (covers the learner's seat *and*
     any self-play opponent seats), or `frozen:<label>` for snapshots.
  2. Encode + stack the learner bucket into one big batch and run a
     single `model(...)` forward (this is where the wall-clock win comes
     from — at 16 envs × 2 seats with 80% self-play, the learner bucket
     averages ~25 obs per step instead of doing 25 separate B=1 forwards).
  3. For each snapshot bucket, run the snapshot's `LearnedAgent` per
     element (small batches; we don't bother batching across snapshots
     because a different snapshot = a different model).
  4. Distribute the resulting moves back into per-env action lists, then
     step the alive envs in parallel via `VecEnv.step_subset`.
  5. Trajectories are recorded *only for the learner_seat* of each env —
     self-play opponent seats are batched into the forward for compute,
     but their per-step records aren't collected (PPO trains on the
     learner's transitions only).
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from typing import Any

import torch

from ..policies.features import EncodedObs, encode_raw_observations
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
from .rollout import Trajectory
from .vec_env import VecEnv


def _mark_cuda_graph_step(device: torch.device) -> None:
    if device.type != "cuda":
        return
    mark = getattr(torch.compiler, "cudagraph_mark_step_begin", None)
    if callable(mark):
        mark()


def _kernel_cache(model: torch.nn.Module) -> dict:
    cache = model.__dict__.get("_owars_rollout_kernel_cache")
    if cache is None:
        cache = {}
        model.__dict__["_owars_rollout_kernel_cache"] = cache
    return cache


class _RolloutForwardKernel(torch.nn.Module):
    def __init__(self, model: torch.nn.Module, *, autocast_enabled: bool) -> None:
        super().__init__()
        self.model = model
        self.autocast_enabled = bool(autocast_enabled)

    def forward(
        self,
        planet_feats: torch.Tensor,
        planet_mask: torch.Tensor,
        planet_owned_mask: torch.Tensor,
        planet_ids: torch.Tensor,
        planet_garrison: torch.Tensor,
        fleet_feats: torch.Tensor,
        fleet_mask: torch.Tensor,
    ) -> PolicyOutput:
        feats = EncodedObs(
            planet_feats=planet_feats,
            planet_mask=planet_mask,
            planet_owned_mask=planet_owned_mask,
            planet_ids=planet_ids,
            planet_garrison=planet_garrison,
            fleet_feats=fleet_feats,
            fleet_mask=fleet_mask,
        )
        with torch.autocast(
            device_type="cuda", dtype=torch.bfloat16, enabled=self.autocast_enabled
        ):
            return self.model(feats)


def _get_rollout_kernel(
    model: torch.nn.Module,
    device: torch.device,
    compile_mode: str | None,
) -> torch.nn.Module:
    mode = compile_mode if device.type == "cuda" else None
    key = ("rollout", mode)
    cache = _kernel_cache(model)
    cached = cache.get(key)
    if cached is not None:
        return cached
    kernel = _RolloutForwardKernel(model, autocast_enabled=device.type == "cuda")
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
    )


def _slice_policy_output(out: PolicyOutput, rows: int) -> PolicyOutput:
    if out.launch_logits.shape[0] == rows:
        return out
    return PolicyOutput(
        launch_logits=out.launch_logits[:rows],
        target_logits=out.target_logits[:rows],
        fraction_alpha=out.fraction_alpha[:rows],
        fraction_beta=out.fraction_beta[:rows],
        value=out.value[:rows],
        value_logits=out.value_logits[:rows],
        planet_owned_mask=out.planet_owned_mask[:rows],
        planet_mask=out.planet_mask[:rows],
        planet_ids=out.planet_ids[:rows],
    )


def _empty_traj() -> Trajectory:
    return Trajectory(
        encoded=[], launch=[], target_idx=[], fraction=[],
        log_prob=[], value=[], reward=[], owned_mask=[],
        old_launch_logits=[],
        old_target_logits=[],
        old_fraction_alpha=[],
        old_fraction_beta=[],
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
    policy_graph_rows: int | None = None,
) -> list[Trajectory]:
    """Play `len(opponents_per_env)` episodes in parallel; one Trajectory per env.

    `vec` is a long-lived `VecEnv` owned by the caller — we just call
    `vec.reset()` here. Spawning subprocess workers per call would burn
    seconds of pure startup time on each PPO update; the caller creates
    the pool once and reuses it.

    `opponents_per_env[i]` is a list of `num_players-1` `OpponentSlot`s for
    env `i` (the seat order skips that env's learner seat). `learner_seat`
    can be a scalar for legacy single-seat rollouts or a per-env list. The
    opponent assignment is fixed for the whole episode — sampled once by the
    caller before the rollout.
    """
    num_envs = len(opponents_per_env)
    assert num_envs == vec.num_envs, (
        f"opponents_per_env has {num_envs} entries but vec has {vec.num_envs} workers"
    )
    if reward_cfg is None:
        reward_cfg = RewardCfg()
    learner_seats = _normalize_learner_seats(learner_seat, num_envs, num_players)
    seat_agents = _resolve_seat_agents(opponents_per_env, num_players, learner_seats)

    trajectories = [_empty_traj() for _ in range(num_envs)]
    finals: list[Any] = [None] * num_envs

    states = vec.reset()
    dones = [False] * num_envs
    episode_steps = int(getattr(vec, "episode_steps", 500))
    dense_potential = record_trajectories and reward_cfg.potential_weight != 0.0
    previous_potential = [0.0] * num_envs
    if dense_potential:
        previous_potential = _reward_potentials(
            vec,
            states,
            [(idx, learner_seats[idx]) for idx in range(num_envs)],
            num_players,
            episode_steps,
            reward_cfg,
        )
    fast_policy_batch = getattr(vec, "policy_batch", None)
    fast_policy_batch_no_context = getattr(vec, "policy_batch_no_context", None)
    fast_observation = getattr(vec, "observation", None)
    fast_observations = getattr(vec, "observations", None)
    fast_step_subset = getattr(vec, "step_subset_fast", None)
    use_fast_numpy_path = (
        bool(getattr(vec, "fast_rollout", False))
        and callable(fast_policy_batch)
        and callable(fast_observation)
        and callable(fast_step_subset)
    )

    while not all(dones):
        # 1. Bucket (env, seat, obs) tuples by agent identity.
        learner_bucket: list[tuple[int, int, Any]] = []
        opp_buckets: dict[str, list[tuple[int, int, Any, OpponentSlot]]] = (
            defaultdict(list)
        )
        pending_opp_obs: list[tuple[int, int]] = []
        pending_opp_slots: list[tuple[int, int, OpponentSlot]] = []
        for env_idx in range(num_envs):
            if dones[env_idx]:
                continue
            state = None if use_fast_numpy_path else states[env_idx]
            for seat in range(num_players):
                slot = seat_agents[env_idx][seat]
                if slot is None or slot.name == LEARNER_NAME:
                    obs = None if use_fast_numpy_path else state[seat]["observation"]
                    learner_bucket.append((env_idx, seat, obs))
                else:
                    if use_fast_numpy_path:
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

        actions_per_env: dict[int, list[Any]] = {
            i: [None] * num_players for i in range(num_envs) if not dones[i]
        }

        # 2. One batched forward for the learner identity.
        if learner_bucket:
            _step_learner_bucket(
                model,
                learner_bucket,
                actions_per_env,
                trajectories,
                learner_seats,
                device,
                deterministic,
                record_trajectories,
                (
                    fast_policy_batch_no_context
                    if use_fast_numpy_path and callable(fast_policy_batch_no_context)
                    else fast_policy_batch if use_fast_numpy_path else None
                ),
                compile_mode,
                policy_graph_rows or (num_envs * num_players),
            )

        # 3. Per-snapshot inference. Learned snapshots expose `act_batch`;
        # builtin Python baselines stay on the scalar callable path.
        for _name, bucket in opp_buckets.items():
            agent = bucket[0][3].agent
            act_batch = getattr(agent, "act_batch", None)
            if callable(act_batch):
                obs_list = [obs for _env_idx, _seat, obs, _slot in bucket]
                batched_actions = act_batch(obs_list)
                for (env_idx, seat, _obs, _slot), acts in zip(
                    bucket, batched_actions, strict=True
                ):
                    actions_per_env[env_idx][seat] = acts
            else:
                for env_idx, seat, obs, slot in bucket:
                    actions_per_env[env_idx][seat] = slot.agent(obs)

        # 4. Step alive envs in parallel.
        active = [i for i in range(num_envs) if not dones[i]]
        actions_list = [actions_per_env[i] for i in active]
        step_subset = fast_step_subset if use_fast_numpy_path else vec.step_subset
        results = step_subset(active, actions_list)
        for i, (state, done, final) in results.items():
            if state is not None:
                states[i] = state
            if done:
                dones[i] = True
                finals[i] = final
        if dense_potential:
            rows = [
                (idx, learner_seats[idx])
                for idx in active
                if trajectories[idx].reward
            ]
            current_potential = _reward_potentials(
                vec,
                states,
                rows,
                num_players,
                episode_steps,
                reward_cfg,
            )
            for (env_idx, _seat), phi in zip(rows, current_potential, strict=True):
                trajectories[env_idx].reward[-1] += reward_cfg.potential_weight * (
                    phi - previous_potential[env_idx]
                )
                previous_potential[env_idx] = phi

    # 5. Apply terminal reward + record seat_rewards on each trajectory.
    for env_idx in range(num_envs):
        _finalize_trajectory(
            trajectories[env_idx], finals[env_idx], learner_seats[env_idx], reward_cfg
        )

    return trajectories


def _step_learner_bucket(
    model: OrbitPolicy,
    bucket: list[tuple[int, int, Any]],
    actions_per_env: dict[int, list[Any]],
    trajectories: list[Trajectory],
    learner_seats: Sequence[int],
    device: str,
    deterministic: bool,
    record_trajectories: bool,
    policy_batch: Any | None = None,
    compile_mode: str | None = None,
    graph_rows: int | None = None,
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
    graph_rows = max(int(graph_rows or len(bucket)), len(bucket))
    action_contexts: list[ActionContext] | None = None
    policy_rows: list[tuple[int, int]] | None = None
    fast_sampler = getattr(getattr(policy_batch, "__self__", None), "sample_batch_with_records", None)
    fast_actions_sampler = getattr(getattr(policy_batch, "__self__", None), "sample_batch_actions", None)
    if callable(policy_batch):
        policy_rows = [(env_idx, seat) for env_idx, seat, _obs in bucket]
        if record_on_cpu:
            cpu_stacked, action_contexts = policy_batch(
                policy_rows, device="cpu", pin_memory=False
            )
            device_source = (
                _pad_encoded_rows(cpu_stacked, graph_rows)
                if graph_enabled
                else cpu_stacked
            )
            stacked = _encoded_to_device(device_source, target_device)
        else:
            stacked, action_contexts = policy_batch(
                policy_rows,
                device=device,
                pin_memory=target_device.type == "cuda",
            )
            if stacked.planet_feats.device != target_device:
                stacked = _encoded_to_device(stacked, target_device)
            cpu_stacked = None
    else:
        cpu_stacked = (
            encode_raw_observations(raw_obs_list, device="cpu")
            if record_on_cpu
            else None
        )
        if cpu_stacked is not None:
            device_source = (
                _pad_encoded_rows(cpu_stacked, graph_rows)
                if graph_enabled
                else cpu_stacked
            )
            stacked = _encoded_to_device(device_source, target_device)
        else:
            stacked = encode_raw_observations(
                raw_obs_list,
                device=device,
                pin_memory=target_device.type == "cuda",
            )
    if cpu_stacked is not None:
        real_rows = cpu_stacked.planet_feats.shape[0]
    else:
        real_rows = len(bucket)
        if stacked.planet_feats.shape[0] < real_rows:
            raise RuntimeError("encoded rollout batch has fewer rows than bucket")
    learner_rows: list[int] = []
    learner_envs: list[int] = []
    if record_trajectories:
        learner_rows = [
            k
            for k, (env_idx, seat, _obs) in enumerate(bucket)
            if seat == learner_seats[env_idx]
        ]
        learner_envs = [bucket[k][0] for k in learner_rows]
    # CUDA rollout uses a fixed padded batch so Inductor can reuse one static
    # graph even as envs finish and the real learner bucket shrinks.
    graph_stacked = _pad_encoded_rows(stacked, graph_rows) if graph_enabled else stacked
    kernel = _get_rollout_kernel(
        model,
        target_device,
        compile_mode if graph_enabled else None,
    )
    with torch.no_grad():
        _mark_cuda_graph_step(target_device)
        out = kernel(
            graph_stacked.planet_feats,
            graph_stacked.planet_mask,
            graph_stacked.planet_owned_mask,
            graph_stacked.planet_ids,
            graph_stacked.planet_garrison,
            graph_stacked.fleet_feats,
            graph_stacked.fleet_mask,
        )
    out = _slice_policy_output(out, real_rows)

    if record_trajectories:
        if callable(fast_sampler) and policy_rows is not None:
            actions_list, records = fast_sampler(
                out,
                policy_rows,
                deterministic=deterministic,
                record_rows=learner_rows,
                native_actions=True,
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

    if record_trajectories and learner_rows:
        row_idx = torch.as_tensor(
            learner_rows, device=out.value.device, dtype=torch.long
        )
        rec = _materialize_records_cpu(
            stacked,
            cpu_stacked,
            out,
            records,
            row_idx,
            learner_rows,
        )
        for j, env_idx in enumerate(learner_envs):
            traj = trajectories[env_idx]
            traj.encoded.append(
                EncodedObs(
                    planet_feats=rec["planet_feats"][j],
                    planet_mask=rec["planet_mask"][j],
                    planet_owned_mask=rec["planet_owned_mask"][j],
                    planet_ids=rec["planet_ids"][j],
                    planet_garrison=rec["planet_garrison"][j],
                    fleet_feats=rec["fleet_feats"][j],
                    fleet_mask=rec["fleet_mask"][j],
                )
            )
            traj.launch.append(rec["launch"][j])
            traj.target_idx.append(rec["target_idx"][j])
            traj.fraction.append(rec["fraction"][j])
            traj.log_prob.append(rec["log_prob"][j])
            traj.value.append(rec["value"][j])
            traj.owned_mask.append(rec["owned_mask"][j])
            traj.old_launch_logits.append(rec["old_launch_logits"][j])
            traj.old_target_logits.append(rec["old_target_logits"][j])
            traj.old_fraction_alpha.append(rec["old_fraction_alpha"][j])
            traj.old_fraction_beta.append(rec["old_fraction_beta"][j])
            traj.reward.append(0.0)

    for k, (env_idx, seat, _obs) in enumerate(bucket):
        actions_per_env[env_idx][seat] = actions_list[k]


def _materialize_records_cpu(
    stacked: EncodedObs,
    cpu_stacked: EncodedObs | None,
    out: Any,
    records: list[Any],
    row_idx: torch.Tensor,
    rows: list[int],
) -> dict[str, torch.Tensor]:
    """Copy learner rollout records to CPU once per field per env step.

    Keeping every per-step feature tensor on CUDA caps rollout parallelism and
    leaves thousands of small device allocations alive until PPO batching.
    The action sampler already synchronizes for Python env actions, so this
    moves trajectory storage off VRAM at the same loop boundary.
    """
    feature_source = cpu_stacked if cpu_stacked is not None else stacked
    feature_rows = (
        torch.as_tensor(rows, dtype=torch.long)
        if feature_source.planet_feats.device.type == "cpu"
        else row_idx
    )
    if isinstance(records, list):
        launch = torch.stack([records[k].launch for k in rows])
        target_idx = torch.stack([records[k].target_idx for k in rows])
        fraction = torch.stack([records[k].fraction for k in rows])
        log_prob = torch.stack([records[k].log_prob for k in rows])
        old_launch_logits = torch.stack([records[k].launch_logits for k in rows])
        old_target_logits = torch.stack([records[k].target_logits for k in rows])
        old_fraction_alpha = torch.stack([records[k].fraction_alpha for k in rows])
        old_fraction_beta = torch.stack([records[k].fraction_beta for k in rows])
    else:
        launch = records.launch
        target_idx = records.target_idx
        fraction = records.fraction
        log_prob = records.log_prob
        old_launch_logits = records.launch_logits
        old_target_logits = records.target_logits
        old_fraction_alpha = records.fraction_alpha
        old_fraction_beta = records.fraction_beta
    owned_mask = out.planet_owned_mask.index_select(0, row_idx)
    value = out.value.index_select(0, row_idx)
    b, p = target_idx.shape
    flat = torch.cat(
        (
            target_idx.float(),
            launch.float(),
            fraction.float(),
            log_prob.float(),
            value.float().unsqueeze(1),
            owned_mask.float(),
            old_launch_logits.float(),
            old_target_logits.float().reshape(b, -1),
            old_fraction_alpha.float(),
            old_fraction_beta.float(),
        ),
        dim=1,
    ).detach().cpu()
    pos = 0
    target_idx_cpu = flat[:, pos : pos + p].long()
    pos += p
    launch_cpu = flat[:, pos : pos + p]
    pos += p
    fraction_cpu = flat[:, pos : pos + p]
    pos += p
    log_prob_cpu = flat[:, pos : pos + p]
    pos += p
    value_cpu = flat[:, pos]
    pos += 1
    owned_mask_cpu = flat[:, pos : pos + p].bool()
    pos += p
    old_launch_logits_cpu = flat[:, pos : pos + p]
    pos += p
    old_logits_width = p * p
    old_target_logits_cpu = flat[:, pos : pos + old_logits_width].reshape(b, p, p)
    pos += old_logits_width
    old_fraction_alpha_cpu = flat[:, pos : pos + p]
    pos += p
    old_fraction_beta_cpu = flat[:, pos : pos + p]

    return {
        "planet_feats": feature_source.planet_feats.index_select(0, feature_rows)
        .detach()
        .cpu(),
        "planet_mask": feature_source.planet_mask.index_select(0, feature_rows)
        .detach()
        .cpu(),
        "planet_owned_mask": feature_source.planet_owned_mask.index_select(0, feature_rows)
        .detach()
        .cpu(),
        "planet_ids": feature_source.planet_ids.index_select(0, feature_rows)
        .detach()
        .cpu(),
        "planet_garrison": feature_source.planet_garrison.index_select(0, feature_rows)
        .detach()
        .cpu(),
        "fleet_feats": feature_source.fleet_feats.index_select(0, feature_rows)
        .detach()
        .cpu(),
        "fleet_mask": feature_source.fleet_mask.index_select(0, feature_rows)
        .detach()
        .cpu(),
        "target_idx": target_idx_cpu,
        "launch": launch_cpu,
        "fraction": fraction_cpu,
        "log_prob": log_prob_cpu,
        "value": value_cpu,
        "owned_mask": owned_mask_cpu,
        "old_launch_logits": old_launch_logits_cpu,
        "old_target_logits": old_target_logits_cpu,
        "old_fraction_alpha": old_fraction_alpha_cpu,
        "old_fraction_beta": old_fraction_beta_cpu,
    }


def _encoded_to_device(feats: EncodedObs, device: torch.device) -> EncodedObs:
    if device.type != "cuda":
        return feats.to(device)

    def move(t: torch.Tensor) -> torch.Tensor:
        if t.device == device:
            return t
        if t.device.type == "cpu":
            return t.pin_memory().to(device, non_blocking=True)
        return t.to(device, non_blocking=True)

    return EncodedObs(
        planet_feats=move(feats.planet_feats),
        planet_mask=move(feats.planet_mask),
        planet_owned_mask=move(feats.planet_owned_mask),
        planet_ids=move(feats.planet_ids),
        planet_garrison=move(feats.planet_garrison),
        fleet_feats=move(feats.fleet_feats),
        fleet_mask=move(feats.fleet_mask),
    )
