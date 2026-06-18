"""Policy/value-net config (mirrors hull-tactical's ModelCfg pattern)."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal


def normalize_attention_config(
    dim: int,
    n_heads: int,
    n_kv_heads: int | None,
) -> tuple[int, int, int]:
    """Return normalized `(n_kv_heads, head_dim, kv_dim)` for attention."""
    if n_heads <= 0:
        raise ValueError("n_heads must be positive")
    if dim % n_heads != 0:
        raise ValueError(f"dim {dim} not divisible by n_heads {n_heads}")
    if n_kv_heads is None:
        n_kv_heads = n_heads
    if n_kv_heads <= 0:
        raise ValueError("n_kv_heads must be positive")
    if n_heads % n_kv_heads != 0:
        raise ValueError("n_heads must be divisible by n_kv_heads")
    head_dim = dim // n_heads
    kv_dim = n_kv_heads * head_dim
    return n_kv_heads, head_dim, kv_dim


@dataclass
class OrbitPolicyConfig:
    # Per-token feature widths (set by `features.encode_observation`).
    planet_features: int = 19
    fleet_features: int = 20
    global_features: int = 27

    # Set-transformer encoder hyperparameters.
    dim: int = 96
    ff_dim: int = 256
    depth: int = 3
    n_heads: int = 4
    # Grouped-query attention. `None` uses ordinary MHA (`n_heads` KV heads);
    # set to `1` for MQA or another divisor of `n_heads` for GQA.
    n_kv_heads: int | None = None
    dropout: float = 0.0
    # nGPT residual / attention init profile. Defaults are the faithful nGPT
    # values (gentle cold start: each sublayer contributes a small geodesic
    # step, soft attention). The "old-block" residual profile is opt-in:
    #   - eigen_alpha_init=0.5  — full-strength residual. α=0.5 makes the eigen
    #     step `justnorm(x̂ + α(â−x̂))` equal `justnorm(x̂ + â)`, i.e. the old
    #     Euclidean `x + sublayer` add projected back onto the sphere.
    #   - qk_gain_init=5.0      — sharp attention. Sets the effective per-channel
    #     `sqk_q` scale on the unit-norm queries to 5, reproducing the old
    #     parameter-golf `q_gain=5` logit magnitude under the √head_dim softmax
    #     scale (the unit-norm vs unit-RMS difference cancels exactly).
    #   - block_skip=True       — learnable per-channel U-net skip toward the
    #     block-stack input (the embedding), the on-sphere analog of the old
    #     `resid_mix` x0 term. Zero-init ⇒ identity at start, learns from there.
    eigen_alpha_init: float = 0.05
    qk_gain_init: float = 1.0
    block_skip: bool = False
    # Apply 2D RoPE to this fraction of each self-attention head for physical
    # planet tokens. Summary/fleet tokens keep the ordinary content attention.
    # The actual rotated width is rounded to a valid 2D pair count.
    planet_rope_fraction: float = 0.25
    planet_rope_base: float = 10000.0
    # Encoder dispatch. `fleet_latent` is the default: raw planet tokens are
    # preserved for the action vocabulary, while raw fleets are compressed.
    encoder_backend: Literal["dense", "fleet_latent", "destination_conditioned"] = "fleet_latent"
    # Perceiver-style fleet tokenizer. Raw planet tokens are preserved because
    # they define the source/target action vocabulary; raw fleet tokens are
    # compressed into this fixed latent set before the main policy encoder.
    num_fleet_latents: int = 64
    fleet_tokenizer_depth: int = 1

    # Action factorization. For each owned planet we emit:
    #   - one masked categorical over [noop, target_0, ..., target_P]
    #   - pg-style softcapped logits for the categorical distribution
    #   - a unimodal Beta fraction, conditional on launching.
    # Launch angle is derived from the chosen target via an iterative
    # lead-intercept solver in `sampling.py` — no learned angle component.
    action_logit_softcap: float = 8.0

    # Distributional critic. The first horizon predicts V(s_t); additional
    # MTP horizons predict future-row lambda returns from the same critic token
    # and are masked at episode tails. `dreamer3` uses a CleanRL v162-style
    # coordinate-space symlog bucket: symmetric odd bins in coord space,
    # symexp centers for scalar decode, and Gaussian CDF target projection.
    # PPO defaults fit the normalized clipped-return envelope:
    # symlog(10 * (1 - 0.997**500) / (1 - 0.997)) ~= 7.86, so [-8, 8]
    # covers the hard horizon while preserving resolution near zero.
    value_hidden: int = 64
    value_num_bins: int = 255
    value_sigma_to_bin_ratio: float = 0.75
    critic_mtp_horizon: int = 6
    value_min: float = -8.0
    value_max: float = 8.0
    value_symlog: bool = False
    value_bucket: Literal["dreamer3", "legacy"] = "dreamer3"
    # Real-units bound on the per-planet SAC advantage heads: each head emits
    # `adv_scale·tanh(raw/adv_scale)`, so a single planet's launch/no-launch
    # advantage is confined to ±adv_scale ship-margin units and the summed
    # advantage to ±n_owned·adv_scale. The tanh bound (rather than logit-space
    # tilt) is what anchors the dueling V/A split: V carries the state baseline,
    # adv the bounded action-dependent residual. Sized for headroom over the
    # per-capture margin tail (~30 ships); raise toward 60 if `adv_spread`
    # saturates near the bound.
    adv_scale: float = 40.0

    def to_dict(self) -> dict:
        return asdict(self)
