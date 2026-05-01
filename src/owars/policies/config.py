"""Policy/value-net config (mirrors hull-tactical's ModelCfg pattern)."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal


@dataclass
class OrbitPolicyConfig:
    # Per-token feature widths (set by `features.encode_observation`).
    planet_features: int = 19
    fleet_features: int = 20

    # Set-transformer encoder hyperparameters.
    dim: int = 96
    ff_dim: int = 256
    depth: int = 3
    n_heads: int = 4
    dropout: float = 0.0
    # Apply 2D RoPE to this fraction of each self-attention head for physical
    # planet tokens. Summary/fleet tokens keep the ordinary content attention.
    # The actual rotated width is rounded to a valid 2D pair count.
    planet_rope_fraction: float = 0.25
    planet_rope_base: float = 10000.0
    # Encoder dispatch. `fleet_latent` is the default: raw planet tokens are
    # preserved for the action vocabulary, while raw fleets are compressed.
    encoder_backend: Literal["dense", "fleet_latent"] = "fleet_latent"
    # Perceiver-style fleet tokenizer. Raw planet tokens are preserved because
    # they define the source/target action vocabulary; raw fleet tokens are
    # compressed into this fixed latent set before the main policy encoder.
    num_fleet_latents: int = 64
    fleet_tokenizer_depth: int = 1

    # Action factorization. For each owned planet we emit:
    #   - a Bernoulli launch logit
    #   - masked categorical target logits, conditional on launching
    #   - a mode+concentration Beta(α, β) fraction, conditional on launching.
    # Launch angle is derived from the chosen target via an iterative
    # lead-intercept solver in `sampling.py` — no learned angle component.

    # Distributional value head — predicts a categorical over `value_num_bins`
    # bins, trained with Dreamer4-style symlog HL-Gauss CE. `value_min/max`
    # are raw return bounds; with `value_symlog=True`, the library transforms
    # them to symlog support endpoints for bucket placement and decodes scalar
    # predictions with symexp. For example, raw ±100k becomes symlog support
    # ±log(100001) ≈ ±11.51. Targets outside the raw support are clipped to
    # the boundary bin in `target_probs`.
    value_hidden: int = 64
    value_num_bins: int = 153
    value_min: float = -100_000.0
    value_max: float = 100_000.0
    value_symlog: bool = True

    def to_dict(self) -> dict:
        return asdict(self)
