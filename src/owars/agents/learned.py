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
from ..policies.features import encode_observation
from ..policies.model import OrbitPolicy, OrbitPolicyConfig
from ..policies.sampling import sample_actions


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
        self.model.load_state_dict(state["model"])
        self.model.eval()
        self.device = device
        self.deterministic = deterministic
        self.max_moves_per_turn = max_moves_per_turn

    @torch.no_grad()
    def __call__(self, obs: Any) -> list[list]:
        o = parse_observation(obs)
        feats = encode_observation(o, device=self.device)
        out = self.model(feats)
        moves = sample_actions(
            out,
            o,
            deterministic=self.deterministic,
            max_moves=self.max_moves_per_turn,
        )
        return [m.as_list() for m in moves]
