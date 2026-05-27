"""Wraps a trained SAC `SACActor` in the agent callable interface.

Mirrors `LearnedAgent` (the PPO `OrbitPolicy` wrapper) but for the SAC test
branch: SAC checkpoints store `actor` / `actor_cfg` (not `model` / `config`).
The SAC actor now uses the SAME factored launch/target/fraction action PPO does
— the launch angle is solved analytically from the chosen target — so
deterministic inference goes through the shared `sac_sampling` path, which in
turn reuses `sampling.py`'s lead-intercept geometry. The `log_std` bounds are
irrelevant at play time (deterministic uses the squashed mean) but kept for
faithful reconstruction / training resume.

Like `LearnedAgent`, this annotates each observation with a per-agent
`_FleetTargetTracker` so encoded fleet features match training.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from ..policies.config import OrbitPolicyConfig
from ..policies.features import encode_raw_observations
from ..policies.sac_model import SACActor
from ..policies.sac_sampling import sac_sample_actions
from .learned import _FleetTargetTracker


class SACAgent:
    """Callable agent backed by a trained `SACActor` checkpoint."""

    def __init__(
        self,
        ckpt_path: str | Path,
        device: str = "cpu",
        deterministic: bool = True,
        episode_steps: int = 500,
    ):
        state = torch.load(ckpt_path, map_location=device, weights_only=False)
        if "actor" not in state or "actor_cfg" not in state:
            raise ValueError(
                f"{ckpt_path} is not a SAC checkpoint (missing 'actor'/'actor_cfg'); "
                "use LearnedAgent for PPO checkpoints"
            )
        cfg = OrbitPolicyConfig(**state["actor_cfg"])
        self.model = SACActor(
            cfg,
            log_std_min=state.get("log_std_min", -5.0),
            log_std_max=state.get("log_std_max", 2.0),
        ).to(device)
        self.model.load_state_dict(state["actor"])
        self.model.eval()
        self.device = device
        self.deterministic = deterministic
        # The checkpoint records the training horizon (the encoder FiLM is
        # conditioned on step/episode_steps); prefer it so play-time time_feat
        # matches training, falling back to the arg for legacy checkpoints.
        self.episode_steps = int(state.get("episode_steps", episode_steps))
        self._tracker = _FleetTargetTracker()

    @torch.inference_mode()
    def __call__(self, obs: Any) -> list[list]:
        annotated = self._tracker.annotate(obs)
        feats = encode_raw_observations([annotated], device=self.device)
        # Game-clock scalar ∈ [0,1] for the encoder FiLM — must match training so
        # the policy reproduces its endgame behavior.
        get = obs.get if isinstance(obs, dict) else lambda k, d=None: getattr(obs, k, d)
        step = float(get("step", 0) or 0)
        time_feat = torch.tensor(
            [min(1.0, max(0.0, step / float(self.episode_steps)))],
            dtype=torch.float32,
            device=self.device,
        )
        actions = sac_sample_actions(
            self.model,
            feats,
            annotated,
            deterministic=self.deterministic,
            time_feat=time_feat,
        )
        self._tracker.record(obs, actions)
        return [move[:3] for move in actions]
