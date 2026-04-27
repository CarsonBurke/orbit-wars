"""Policy/value-net config (mirrors hull-tactical's ModelCfg pattern)."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass
class OrbitPolicyConfig:
    # Per-token feature widths (set by `features.encode_observation`).
    planet_features: int = 19
    fleet_features: int = 15

    # Set-transformer encoder hyperparameters.
    dim: int = 96
    ff_dim: int = 256
    depth: int = 3
    n_heads: int = 4
    dropout: float = 0.0

    # Action factorization. For each owned planet we emit:
    #   - logits over (target_planet | no-op)
    #   - a tanh-squashed Normal(μ, σ) on the fraction of garrison to send.
    # Launch angle is derived from the chosen target via an iterative
    # lead-intercept solver in `sampling.py` — no learned angle component.

    # Value head — always a single scalar predicting expected score margin.
    value_hidden: int = 64

    def to_dict(self) -> dict:
        return asdict(self)
