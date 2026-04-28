"""Policy/value-net config (mirrors hull-tactical's ModelCfg pattern)."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal


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
    # Encoder dispatch. `dense` is the CUDA rollout default because PyTorch's
    # nested-jagged subclass overhead dominates at our small rollout batch sizes.
    # `nested` is retained for A/B benchmarks and larger-token experiments.
    encoder_backend: Literal["dense", "nested"] = "dense"

    # Action factorization. For each owned planet we emit:
    #   - logits over (target_planet | no-op)
    #   - a soft-capped Beta(α, β) on the fraction of garrison to send.
    # Launch angle is derived from the chosen target via an iterative
    # lead-intercept solver in `sampling.py` — no learned angle component.

    # Distributional value head — predicts a categorical over `value_num_bins`
    # bins on `[value_min, value_max]`, trained with HL-Gauss CE (dreamer4
    # `dreamer4.py:722–805`). With γ=1 the *unshaped* return is exactly the
    # terminal outcome ∈ [-1, 1], but `RewardCfg.margin_scale > 0` adds
    # `margin_scale · margin` to the terminal reward (margin ≤ 2 in 2P). We
    # default to [-2, 2] to give 1.0 headroom on each side — covers
    # `margin_scale ≤ 0.5` exactly. Configs with more aggressive margin
    # shaping (or non-terminal shaping bonuses, currently dead) should
    # widen further. 51 bins → ~0.08 resolution. The scalar value used for
    # advantage is the expectation E[V] = Σ p_i · c_i recovered via
    # `HLGaussLoss.bins_to_scalar`. Targets outside the support are clipped
    # to the boundary bin in `target_probs`, so the head degrades gracefully
    # instead of going NaN if returns exceed the range.
    value_hidden: int = 64
    value_num_bins: int = 51
    value_min: float = -2.0
    value_max: float = 2.0

    def to_dict(self) -> dict:
        return asdict(self)
