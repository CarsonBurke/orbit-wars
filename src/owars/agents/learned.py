"""Wraps a trained `OrbitPolicy` in the agent callable interface.

The runtime path is:
  `obs -> parse_observation -> features -> policy(features) -> Move list`

This module is intentionally light on numpy/torch imports at module top so
the submission shell can lazy-load weights only once per process.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from ..game import parse_observation
from ..policies.features import encode_observation, encode_observations
from ..policies.model import OrbitPolicy, OrbitPolicyConfig, restore_fp32_params
from ..policies.sampling import sample_actions, sample_batch_actions


class LearnedAgent:
    def __init__(
        self,
        ckpt_path: str | Path,
        device: str = "cpu",
        deterministic: bool = True,
        max_moves_per_turn: int = 16,
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
        self.max_moves_per_turn = max_moves_per_turn

    @torch.inference_mode()
    def __call__(self, obs: Any) -> list[list]:
        o = parse_observation(obs)
        feats = encode_observation(o, device=self.device)
        autocast_enabled = torch.device(self.device).type == "cuda"
        with torch.autocast(
            device_type="cuda", dtype=torch.bfloat16, enabled=autocast_enabled
        ):
            out = self.model(feats)
        moves = sample_actions(
            out,
            o,
            deterministic=self.deterministic,
            max_moves=self.max_moves_per_turn,
        )
        return [m.as_list() for m in moves]

    @torch.inference_mode()
    def act_batch(self, obs_list: list[Any]) -> list[list[list]]:
        if not obs_list:
            return []
        parsed = [parse_observation(obs) for obs in obs_list]
        feats = encode_observations(
            parsed,
            device=self.device,
            pin_memory=torch.device(self.device).type == "cuda",
        )
        autocast_enabled = torch.device(self.device).type == "cuda"
        with torch.autocast(
            device_type="cuda", dtype=torch.bfloat16, enabled=autocast_enabled
        ):
            out = self.model(feats)
        moves_list = sample_batch_actions(
            out,
            parsed,
            deterministic=self.deterministic,
            max_moves=self.max_moves_per_turn,
        )
        return [[m.as_list() for m in moves] for moves in moves_list]
