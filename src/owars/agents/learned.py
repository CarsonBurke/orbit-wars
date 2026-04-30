"""Wraps a trained `OrbitPolicy` in the agent callable interface.

The runtime path is:
  `obs -> action-derived fleet sidecar -> features -> policy(features) -> Move list`

This module is intentionally light on numpy/torch imports at module top so
the submission shell can lazy-load weights only once per process.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import torch

from ..policies.features import encode_raw_observations
from ..policies.model import OrbitPolicy, OrbitPolicyConfig, restore_fp32_params
from ..policies.sampling import sample_batch_actions_raw


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


class LearnedAgent:
    def __init__(
        self,
        ckpt_path: str | Path,
        device: str = "cpu",
        deterministic: bool = True,
    ):
        state = torch.load(ckpt_path, map_location=device)
        cfg = OrbitPolicyConfig(**state["config"])
        self.model = OrbitPolicy(cfg).to(device)
        # Match the training-time fp32-master pattern so loaded checkpoints
        # cast cleanly under autocast on CUDA. CPU load (kaggle submission
        # shell) stays fp32 — no FA-2 there anyway.
        if torch.device(device).type == "cuda":
            self.model.bfloat16()
            restore_fp32_params(self.model)
        self.model.load_state_dict(state["model"])
        self.model.eval()
        self.device = device
        self.deterministic = deterministic
        self._tracker = _FleetTargetTracker()
        self._batch_trackers: dict[tuple[Any, ...], _FleetTargetTracker] = {}

    @torch.inference_mode()
    def __call__(self, obs: Any) -> list[list]:
        annotated = self._tracker.annotate(obs)
        feats = encode_raw_observations([annotated], device=self.device)
        autocast_enabled = torch.device(self.device).type == "cuda"
        with torch.autocast(
            device_type="cuda", dtype=torch.bfloat16, enabled=autocast_enabled
        ):
            out = self.model(feats)
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
        autocast_enabled = torch.device(self.device).type == "cuda"
        with torch.autocast(
            device_type="cuda", dtype=torch.bfloat16, enabled=autocast_enabled
        ):
            out = self.model(feats)
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
