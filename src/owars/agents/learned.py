"""Wraps a trained `OrbitPolicy` in the agent callable interface.

The runtime path is:
  `obs -> action-derived fleet sidecar -> features -> policy(features) -> Move list`

This module is intentionally light on numpy/torch imports at module top so
the submission shell can lazy-load weights only once per process.
"""

from __future__ import annotations

import math
from dataclasses import fields
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from ..policies.features import (
    MAX_FLEETS,
    MAX_PLANETS,
    EncodedObs,
    encode_raw_observations,
)
from ..policies.model import (
    OrbitPolicy,
    OrbitPolicyConfig,
    PolicyOutput,
    restore_fp32_params,
)
from ..policies.sampling import sample_batch_actions_raw


def _policy_config_from_checkpoint(raw: dict[str, Any]) -> OrbitPolicyConfig:
    allowed = {field.name for field in fields(OrbitPolicyConfig)}
    return OrbitPolicyConfig(**{k: v for k, v in raw.items() if k in allowed})


def _angle_delta(a: float, b: float) -> float:
    return abs((a - b + math.pi) % (2.0 * math.pi) - math.pi)


def _tracker_key(obs: Any) -> tuple[Any, ...]:
    if not isinstance(obs, dict):
        return ("unknown",)
    player = int(obs.get("player", 0) or 0)
    planets = obs.get("initial_planets") or obs.get("planets") or []
    planet_key = tuple(
        (int(p[0]), round(float(p[2]), 4), round(float(p[3]), 4))
        for p in planets[:64]
        if len(p) >= 4
    )
    return (
        player,
        round(float(obs.get("angular_velocity", 0.0) or 0.0), 6),
        planet_key,
    )


class _FleetTargetTracker:
    def __init__(self) -> None:
        self.by_fleet_id: dict[int, list[float]] = {}
        self.pending: list[dict[str, float | int]] = []
        self.last_step: int | None = None

    def reset(self) -> None:
        self.by_fleet_id.clear()
        self.pending.clear()
        self.last_step = None

    def annotate(self, obs: Any) -> Any:
        if not isinstance(obs, dict):
            return obs
        player = int(obs.get("player", 0) or 0)
        step = int(obs.get("step", 0) or 0)
        if self.last_step is not None and step < self.last_step:
            self.reset()
        if self.last_step is not None:
            delta = max(0, step - self.last_step)
            if delta:
                for meta in self.by_fleet_id.values():
                    meta[1] = max(0.0, float(meta[1]) - float(delta))
                self.by_fleet_id = {
                    fid: meta
                    for fid, meta in self.by_fleet_id.items()
                    if float(meta[1]) > 0.0
                }
        self.last_step = step

        fleets = obs.get("fleets") or []
        live_ids = {int(f[0]) for f in fleets if len(f) >= 7}
        self.by_fleet_id = {
            fid: meta for fid, meta in self.by_fleet_id.items() if fid in live_ids
        }

        unmatched = [
            f
            for f in fleets
            if len(f) >= 7
            and int(f[1]) == player
            and int(f[0]) not in self.by_fleet_id
        ]
        used: set[int] = set()
        kept_pending: list[dict[str, float | int]] = []
        for pending in self.pending:
            elapsed = max(0, step - int(pending["step"]))
            match_idx = None
            for idx, fleet in enumerate(unmatched):
                if idx in used:
                    continue
                if int(fleet[5]) != int(pending["from_id"]):
                    continue
                if int(fleet[6]) != int(pending["ships"]):
                    continue
                if _angle_delta(float(fleet[4]), float(pending["angle"])) > 1e-6:
                    continue
                match_idx = idx
                break
            if match_idx is None:
                if elapsed <= 1:
                    kept_pending.append(pending)
                continue
            used.add(match_idx)
            fleet_id = int(unmatched[match_idx][0])
            self.by_fleet_id[fleet_id] = [
                int(pending["target_id"]),
                max(0.0, float(pending["eta"]) - float(elapsed)),
                float(pending["target_x"]),
                float(pending["target_y"]),
            ]
        self.pending = kept_pending

        annotated = dict(obs)
        annotated["fleet_targets"] = {
            str(fid): meta
            for fid, meta in self.by_fleet_id.items()
            if float(meta[1]) > 0.0
        }
        return annotated

    def record(self, obs: Any, actions: list[list]) -> None:
        if not isinstance(obs, dict):
            return
        step = int(obs.get("step", 0) or 0)
        for action in actions:
            if len(action) < 7:
                continue
            self.pending.append(
                {
                    "step": step,
                    "from_id": int(action[0]),
                    "angle": float(action[1]),
                    "ships": int(action[2]),
                    "target_id": int(action[3]),
                    "eta": float(action[4]),
                    "target_x": float(action[5]),
                    "target_y": float(action[6]),
                }
            )


class _InferenceForwardKernel(nn.Module):
    def __init__(
        self,
        model: OrbitPolicy,
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
            device_type="cuda",
            dtype=torch.bfloat16,
            enabled=self.autocast_enabled,
        ):
            return self.model(feats, include_value=self.include_value)


def _mark_cuda_graph_step(device: torch.device) -> None:
    if device.type != "cuda":
        return
    mark = getattr(torch.compiler, "cudagraph_mark_step_begin", None)
    if callable(mark):
        mark()


def _next_power_of_two(n: int) -> int:
    return 1 << max(0, int(n - 1).bit_length())


def _pad_rows(t: torch.Tensor, rows: int, fill: int | float | bool = 0) -> torch.Tensor:
    if t.shape[0] >= rows:
        return t
    pad_shape = (rows - t.shape[0], *t.shape[1:])
    pad = t.new_full(pad_shape, fill)
    return torch.cat((t, pad), dim=0)


def _pad_encoded(feats: EncodedObs, rows: int) -> EncodedObs:
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
    return PolicyOutput(
        launch_logits=out.launch_logits[:rows],
        target_logits=out.target_logits[:rows],
        value=out.value[:rows],
        value_logits=out.value_logits[:rows],
        planet_owned_mask=out.planet_owned_mask[:rows],
        planet_mask=out.planet_mask[:rows],
        planet_ids=out.planet_ids[:rows],
        action_logit_softcap=out.action_logit_softcap,
        launch_log_std=None if out.launch_log_std is None else out.launch_log_std[:rows],
        launch_prob_floor=out.launch_prob_floor,
        fraction_alpha=None if out.fraction_alpha is None else out.fraction_alpha[:rows],
        fraction_beta=None if out.fraction_beta is None else out.fraction_beta[:rows],
        fraction_mean=None if out.fraction_mean is None else out.fraction_mean[:rows],
        fraction_log_std=(
            None if out.fraction_log_std is None else out.fraction_log_std[:rows]
        ),
    )


class LearnedAgent:
    def __init__(
        self,
        ckpt_path: str | Path,
        device: str = "cpu",
        deterministic: bool = True,
        compile_mode: str | None = None,
        compile_graph_rows: int | None = None,
    ):
        state = torch.load(ckpt_path, map_location=device)
        cfg = _policy_config_from_checkpoint(state["config"])
        self.model = OrbitPolicy(cfg).to(device)
        # Match the training-time fp32-master pattern so loaded checkpoints
        # cast cleanly under autocast on CUDA. CPU load (kaggle submission
        # shell) stays fp32 — no FA-2 there anyway.
        if torch.device(device).type == "cuda":
            self.model.bfloat16()
            restore_fp32_params(self.model)
        self.model.load_state_dict(state["model"], strict=True)
        self.model.eval()
        self.device = device
        self.deterministic = deterministic
        self.compile_mode = compile_mode if torch.device(device).type == "cuda" else None
        self.compile_graph_rows = (
            int(compile_graph_rows)
            if self.compile_mode is not None and compile_graph_rows is not None
            else None
        )
        self._forward_kernels: dict[tuple[int, bool], nn.Module] = {}
        self._tracker = _FleetTargetTracker()
        self._batch_trackers: dict[tuple[Any, ...], _FleetTargetTracker] = {}
        if self.compile_graph_rows is not None:
            self._warmup_forward_kernel(self.compile_graph_rows)

    def _warmup_forward_kernel(self, rows: int) -> None:
        rows = max(1, int(rows))
        device = torch.device(self.device)
        feats = EncodedObs(
            planet_feats=torch.zeros(
                rows,
                MAX_PLANETS,
                self.model.cfg.planet_features,
                device=device,
            ),
            planet_mask=torch.zeros(rows, MAX_PLANETS, dtype=torch.bool, device=device),
            planet_owned_mask=torch.zeros(
                rows,
                MAX_PLANETS,
                dtype=torch.bool,
                device=device,
            ),
            planet_ids=torch.full(
                (rows, MAX_PLANETS),
                -1,
                dtype=torch.long,
                device=device,
            ),
            planet_garrison=torch.zeros(rows, MAX_PLANETS, device=device),
            fleet_feats=torch.zeros(
                rows,
                MAX_FLEETS,
                self.model.cfg.fleet_features,
                device=device,
            ),
            fleet_mask=torch.zeros(rows, MAX_FLEETS, dtype=torch.bool, device=device),
        )
        with torch.inference_mode():
            self._forward(feats, rows, include_value=False)

    def _forward(
        self,
        feats: EncodedObs,
        rows: int,
        *,
        include_value: bool = True,
    ) -> PolicyOutput:
        device = torch.device(self.device)
        if self.compile_graph_rows is not None:
            graph_rows = (
                self.compile_graph_rows
                if rows <= self.compile_graph_rows
                else _next_power_of_two(rows)
            )
        elif self.compile_mode is not None:
            graph_rows = _next_power_of_two(rows)
        else:
            graph_rows = rows
        graph_feats = _pad_encoded(feats, graph_rows) if graph_rows != rows else feats
        kernel_key = (graph_rows, bool(include_value))
        kernel = self._forward_kernels.get(kernel_key)
        if kernel is None:
            kernel = _InferenceForwardKernel(
                self.model,
                autocast_enabled=device.type == "cuda",
                include_value=include_value,
            )
            if self.compile_mode is not None:
                kernel = torch.compile(
                    kernel,
                    dynamic=False,
                    fullgraph=True,
                    mode=self.compile_mode,
                )
            self._forward_kernels[kernel_key] = kernel
        if self.compile_mode is not None:
            _mark_cuda_graph_step(device)
        out = kernel(
            graph_feats.planet_feats,
            graph_feats.planet_mask,
            graph_feats.planet_owned_mask,
            graph_feats.planet_ids,
            graph_feats.planet_garrison,
            graph_feats.fleet_feats,
            graph_feats.fleet_mask,
        )
        return _slice_policy_output(out, rows) if graph_rows != rows else out

    @torch.inference_mode()
    def __call__(self, obs: Any) -> list[list]:
        annotated = self._tracker.annotate(obs)
        feats = encode_raw_observations([annotated], device=self.device)
        out = self._forward(feats, 1, include_value=False)
        actions = sample_batch_actions_raw(
            out,
            [annotated],
            deterministic=self.deterministic,
        )[0]
        self._tracker.record(obs, actions)
        return [move[:3] for move in actions]

    @torch.inference_mode()
    def act_batch(self, obs_list: list[Any]) -> list[list[list]]:
        if not obs_list:
            return []
        keys = [_tracker_key(obs) for obs in obs_list]
        annotated = [
            self._batch_trackers.setdefault(key, _FleetTargetTracker()).annotate(obs)
            if isinstance(obs, dict) and "fleet_targets" not in obs
            else obs
            for key, obs in zip(keys, obs_list, strict=True)
        ]
        feats = encode_raw_observations(
            annotated,
            device=self.device,
            pin_memory=torch.device(self.device).type == "cuda",
        )
        out = self._forward(feats, len(annotated), include_value=False)
        actions_list = sample_batch_actions_raw(
            out,
            annotated,
            deterministic=self.deterministic,
        )
        live_keys = set(keys)
        self._batch_trackers = {
            key: tracker
            for key, tracker in self._batch_trackers.items()
            if key in live_keys
        }
        for key, obs, actions in zip(keys, obs_list, actions_list, strict=True):
            if isinstance(obs, dict) and "fleet_targets" not in obs:
                self._batch_trackers[key].record(obs, actions)
        return [[move[:3] for move in actions] for actions in actions_list]
