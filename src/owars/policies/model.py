"""OrbitPolicy — set-transformer encoder + factored action heads.

Architecture:
  [ACTOR] [CRITIC] planet_tokens fleet_tokens
              │
              ▼
  [N × Transformer block] ─► token reps
              │
              ├─► h_actor (broadcast)  ─► concat onto each planet rep ─►
              │                             launch/target/fraction heads
              ├─► h_critic             ─► distributional value head (HL-Gauss)
              ├─► planet_h             ─► target_key (per-planet rep stays d-dim)
              └─► fleet_h              ─► (consumed only by encoder cross-attention)

Two learnable summary tokens (Set-Transformer-style PMA) prepend the input
set. The critic token gives the value head a *learned* aggregator instead
of a mean-pool over up to 64+384 tokens, where decision-relevant tokens
otherwise drown in the average. The actor token is concatenated as global
context onto each per-planet rep before the action heads — the per-planet
target/fraction heads see "what's the joint plan look like" without us
having to go autoregressive.

We deliberately do **not** down-project after the actor concat: target_query
and the Beta fraction heads take 2d-wide inputs and project to their natural
output dim (d for query/key, 1 for each concentration head). Down-projecting
`[planet_h || h_actor]` back to d would discard exactly the global-context
capacity the extra token was added to provide.

**Action factorization.** Per source planet, the actor emits a Bernoulli
launch decision, a masked categorical target distribution conditional on
launching, and a unimodal Beta fraction distribution conditional on launching. This
keeps "should this planet act?" independent of the number of legal target
planets; target count should affect *where* probability mass goes, not whether
the source launches at all.

**No angle head.** The launch angle is computed exactly via an iterative
lead-intercept solver in `sampling.py`.

**Fraction head is a unimodal Beta** on native fraction support `(0, 1)`,
following the CleanRL IterThink v24 / Dreamer4 path:
`alpha = 1 + softplus(raw_alpha)`, `beta = 1 + softplus(raw_beta)`.
PPO stores the native Beta sample that the simulator executes, so log-prob
recomputation evaluates the same bounded-support sample directly; no tanh
squash, inverse, or Jacobian correction is involved.

**Distributional value head (HL-Gauss).** dreamer4 (`dreamer4.py:722–805`)
predicts a categorical over a fixed bin support and trains it with
cross-entropy against a Gaussian-kernel-smoothed target distribution
(Imani et al. 2018, Farebrother et al. 2024). The expected value
E[V|s] = Σ p_i · center_i is the recovered scalar; the learning gradient
is bounded by 1 (CE) instead of unbounded (MSE on a possibly-misscaled
scalar), which makes the critic dramatically more robust to early
mis-prediction.

We use Dreamer4's symlog HL-Gauss value encoding over a wide raw support by
default. Raw projected-margin rewards can move by tens of thousands over an
episode, but symlog bucket placement preserves resolution near zero while
still representing decisive endgame margins. σ = 0.5 × bin_size (Dreamer4
default), and `target_probs` clips out-of-range targets to the boundary bin so
the head degrades gracefully rather than producing NaN.

**Critic still shares the encoder backbone.** Value-loss gradients flow
through the same transformer the actor uses. This is tamed by
(a) value-pretraining before PPO, (b) `value_coef ≤ 0.5`, and (c) the
critic now reading from its own dedicated [CRITIC] token, which gives it
a dimension to specialize without competing with per-planet actor reps.

Why a transformer over set tokens (vs a CNN on a rasterized board): planets
and fleets are intrinsically a *small set* of typed entities, not pixels.
Attention naturally handles variable counts and lets each owned planet
attend to every other planet/fleet to decide where to send ships. A
rasterization would discretize positions and lose the precise angles the
action space wants.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812
from hl_gauss_pytorch import HLGaussLoss as _LibraryHLGaussLoss

from .config import OrbitPolicyConfig, normalize_attention_config
from .features import EncodedObs

_PLANET_XY_SCALE: float = 100.0
_PLANET_XY_OFFSET: float = 50.0


class SquaredReLU(nn.Module):
    """Activation: `relu(x)²` — Primer / modded-nanogpt / parameter-golf
    `train_gpt.py:619`.

      • x ≥ 0 → x²
      • x < 0 → 0  (rectifier semantics — feature is "off" for negatives)

    Picked over the `leaky_relu(x, 0.5)²` variant in pg's `sota_train_gpt.py`
    because the leaky² curve is a U-shape (negative inputs still produce
    `0.25·x²` activation, with a sign-flipped gradient on the negative
    branch). The rectifier semantics here are the more conservative choice
    and what modded-nanogpt and Primer use; in practice the empirical gap
    on bounded-input regimes (RMSNorm + QK-norm + zero-init proj) is small
    and ReLU² has the cleaner "feature on/off" interpretation.

    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(x).square()


class CastedLinear(nn.Linear):
    """Linear with `forward` that casts the fp32-master weight (and bias) to
    the input dtype on each call (parameter-golf `sota_train_gpt.py:80`).

    The store-master / cast-on-forward pattern is the same precision regime
    you'd get from `torch.autocast(bf16)` over a vanilla `nn.Linear`, but
    explicit: the dtype boundary is in this method, no autocast cache,
    no implicit interaction with `torch.compile` tracing.
    Combined with `model.bfloat16()` + `restore_fp32_params(model)` (which
    walks the module tree and `.float()`-s every `CastedLinear` plus all
    `ndim<2` params and named control tensors), this gives:

      • bf16 storage for activations/embeddings (memory)
      • fp32 master for *every* weight matrix Muon and AdamW update
        against (precision in the optimizer's per-element accumulators)
      • bf16 compute path on every Linear forward (speed)

    All without relying on autocast for the bf16 boundary.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.weight.to(x.dtype)
        b = self.bias.to(x.dtype) if self.bias is not None else None
        return F.linear(x, w, b)


# Names of control-tensor parameters that must stay in fp32 even after
# `model.bfloat16()` — small tensors with high-precision-sensitive
# updates (per-channel residual scales, attention temperature, summary
# tokens). Matches parameter-golf's `CONTROL_TENSOR_NAME_PATTERNS`.
_FP32_NAME_SUBSTRINGS: tuple[str, ...] = (
    "attn_scale",
    "ff_scale",
    "resid_mix",
    "q_gain",
    "target_q_gain",
    "actor_token",
    "critic_token",
    "fleet_latents",
)


def restore_fp32_params(model: nn.Module) -> None:
    """After `model.bfloat16()`, restore fp32 master copies for the params
    that need precision: every `CastedLinear` weight/bias, every `ndim<2`
    param (biases, RMSNorm/LayerNorm gains, scalars), and every named
    control tensor (`_FP32_NAME_SUBSTRINGS`).

    Lifted from parameter-golf's `restore_fp32_params`. The end-state
    invariant: control tensors and biases are fp32, Linear weights are
    fp32 master (bf16 compute via `CastedLinear`), other ndim≥2 buffers/
    params (none in our model — embeddings are CastedLinear, not
    `nn.Embedding`) stay bf16.
    """
    for module in model.modules():
        if isinstance(module, CastedLinear):
            module.float()
    for name, param in model.named_parameters():
        wants_fp32 = param.ndim < 2 or any(
            sub in name for sub in _FP32_NAME_SUBSTRINGS
        )
        if wants_fp32 and param.dtype != torch.float32:
            param.data = param.data.float()


class Rotary2D(nn.Module):
    """Shared partial 2D RoPE for physical board tokens.

    Positions are continuous board coordinates, not sequence indices. The
    rotated subspace is split evenly between x and y axes; remaining head
    channels stay unrotated. Callers apply it only to the contiguous planet
    token slice; summary/fleet tokens never enter this module.
    """

    def __init__(
        self,
        head_dim: int,
        fraction: float = 0.25,
        base: float = 10000.0,
    ):
        super().__init__()
        if not 0.0 <= fraction <= 1.0:
            raise ValueError("planet_rope_fraction must be in [0, 1]")
        if base <= 0.0:
            raise ValueError("planet_rope_base must be positive")
        self.base = float(base)
        target = int(round(head_dim * fraction))
        # 2D RoPE needs one even-width pair group for x and one for y, so the
        # total rotated width must be a multiple of 4.
        rotate_dim = 4 * int(round(target / 4))
        if fraction > 0.0:
            rotate_dim = max(4, rotate_dim)
        rotate_dim = min(head_dim - (head_dim % 4), rotate_dim)
        self.rotate_dim = rotate_dim
        self.axis_dim = rotate_dim // 2
        self.register_buffer(
            "inv_freq",
            self._make_inv_freq(torch.device("cpu")),
            persistent=False,
        )

    def _make_inv_freq(self, device: torch.device) -> torch.Tensor:
        if self.axis_dim == 0:
            return torch.empty(0, dtype=torch.float32, device=device)
        return 1.0 / (
            self.base
            ** (
                torch.arange(
                    0,
                    self.axis_dim,
                    2,
                    dtype=torch.float32,
                    device=device,
                )
                / max(1, self.axis_dim)
            )
        )

    def _apply(self, fn):  # type: ignore[no-untyped-def]
        out = super()._apply(fn)
        # `model.bfloat16()` casts buffers. RoPE frequencies are tiny control
        # data; recompute them in fp32 on the transformed device.
        self.inv_freq = self._make_inv_freq(self.inv_freq.device)
        return out

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]
        return torch.stack((-x_odd, x_even), dim=-1).flatten(-2)

    def cache(
        self,
        positions: torch.Tensor,
        dtype: torch.dtype,
    ) -> Rotary2DCache | None:
        """Precompute per-forward sin/cos for reuse by every encoder block."""
        if self.rotate_dim == 0:
            return None
        pos_x = positions[..., 0].float().unsqueeze(-1)
        pos_y = positions[..., 1].float().unsqueeze(-1)
        inv = self.inv_freq
        cos_x = torch.repeat_interleave((pos_x * inv).cos(), 2, dim=-1).to(dtype)
        sin_x = torch.repeat_interleave((pos_x * inv).sin(), 2, dim=-1).to(dtype)
        cos_y = torch.repeat_interleave((pos_y * inv).cos(), 2, dim=-1).to(dtype)
        sin_y = torch.repeat_interleave((pos_y * inv).sin(), 2, dim=-1).to(dtype)
        return Rotary2DCache(cos_x, sin_x, cos_y, sin_y)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        cache: Rotary2DCache | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.rotate_dim == 0 or cache is None:
            return q, k
        axis = self.axis_dim
        q_rot = q[..., : self.rotate_dim]
        k_rot = k[..., : self.rotate_dim]
        q_x, q_y = q_rot[..., :axis], q_rot[..., axis:]
        k_x, k_y = k_rot[..., :axis], k_rot[..., axis:]
        cos_x = cache.cos_x.unsqueeze(2)
        sin_x = cache.sin_x.unsqueeze(2)
        cos_y = cache.cos_y.unsqueeze(2)
        sin_y = cache.sin_y.unsqueeze(2)
        q_new = torch.cat(
            (
                q_x * cos_x + self._rotate_half(q_x) * sin_x,
                q_y * cos_y + self._rotate_half(q_y) * sin_y,
            ),
            dim=-1,
        )
        k_new = torch.cat(
            (
                k_x * cos_x + self._rotate_half(k_x) * sin_x,
                k_y * cos_y + self._rotate_half(k_y) * sin_y,
            ),
            dim=-1,
        )
        return q_new, k_new


@dataclass(frozen=True)
class Rotary2DCache:
    cos_x: torch.Tensor
    sin_x: torch.Tensor
    cos_y: torch.Tensor
    sin_y: torch.Tensor


def _splice_rope(
    full: torch.Tensor,
    rotated: torch.Tensor,
    rope_slice: slice,
    rotate_dim: int,
) -> torch.Tensor:
    """Out-of-place splice of `rotated` into `full[:, rope_slice, :, :rotate_dim]`.

    The previous in-place assignment (`full[:, rope_slice, :, :rotate_dim] =
    rotated`) bumps `full`'s autograd version, which is fine when `full` is
    consumed by exactly one backward pass after the mutation (PPO's per-
    minibatch forward/backward) but breaks under SAC's actor update where the
    encoder is forwarded twice within a single update — the second forward's
    backward sees an "expected version 0, got version 1" error. This helper
    rebuilds the tensor via `torch.cat`, which keeps the autograd graph
    monotonic across multi-backward training loops.
    """
    pre = full[:, : rope_slice.start]
    mid_keep = full[:, rope_slice, :, rotate_dim:]
    new_mid = torch.cat([rotated, mid_keep], dim=-1)
    post = full[:, rope_slice.stop :]
    return torch.cat([pre, new_mid, post], dim=1)


class SelfAttention(nn.Module):
    """Multi-head self-attention with QK-norm + per-head q_gain.

    parameter-golf pattern (`sota_train_gpt.py:CausalSelfAttention`):
      1. Project x → Q, K, V.
      2. **RMSNorm Q and K per-head** along the head-dim. Pins ‖q_h‖ and
         ‖k_h‖ to fixed magnitude so attention-logit magnitude does not
         drift as the QKV projections move under Muon (or any optimizer).
      3. Multiply Q by per-head learnable `q_gain` (init=5). This is the
         *attention temperature*: high gain → sharp softmax, low → flat.
      4. SDPA on dense padded tokens with a key mask.
      5. Output projection (zero-init for cold-start identity).
    """

    def __init__(
        self,
        dim: int,
        n_heads: int,
        qk_gain_init: float = 5.0,
        *,
        n_kv_heads: int | None = None,
    ):
        super().__init__()
        n_kv_heads, head_dim, kv_dim = normalize_attention_config(
            dim, n_heads, n_kv_heads
        )
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        # Three separate Q/K/V projections (parameter-golf style).
        self.c_q = CastedLinear(dim, dim, bias=False)
        self.c_k = CastedLinear(dim, kv_dim, bias=False)
        self.c_v = CastedLinear(dim, kv_dim, bias=False)
        self.out_proj = CastedLinear(dim, dim, bias=False)
        # Per-head scalar gain on Q after RMSNorm — sets the attention
        # softmax temperature. Init 5.0 ≈ parameter-golf's `qk_gain_init`.
        self.q_gain = nn.Parameter(torch.full((n_heads,), float(qk_gain_init)))
        # Orthogonal init for Q/K/V projections (parameter-golf
        # `_init_weights`, sota_train_gpt.py:146 — applied to every Linear
        # with both dims ≥64 that isn't `_zero_init`). At spectral norm 1
        # the relative perturbation per Muon NS5 step is `lr` (= 0.02 in
        # our default), vs. 2× larger from PyTorch's kaiming_uniform_(a=√5)
        # whose spectral norm is ~0.5 at dim=128 — so orthogonal init
        # halves first-step log-prob drift.
        nn.init.orthogonal_(self.c_q.weight, gain=0.1)
        nn.init.orthogonal_(self.c_k.weight, gain=0.1)
        nn.init.orthogonal_(self.c_v.weight, gain=0.1)
        # Zero-init output projection — see block-level docstring on
        # cold-start identity. Wins over orthogonal because it's *after*.
        # No bias (parameter-golf: every Linear is `bias=False`); the bias
        # would be a per-channel drift channel with no upside since the
        # residual already adds zero contribution at cold-start.
        nn.init.zeros_(self.out_proj.weight)

    def forward(
        self,
        x: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
        *,
        rope: Rotary2D | None = None,
        rope_cache: Rotary2DCache | None = None,
        rope_slice: slice | None = None,
    ) -> torch.Tensor:
        # `x` is a dense padded tensor [B, T, D] with `valid_mask=True` for
        # real tokens.
        q = self.c_q(x).unflatten(-1, (self.n_heads, self.head_dim))
        k = self.c_k(x).unflatten(-1, (self.n_kv_heads, self.head_dim))
        v = self.c_v(x).unflatten(-1, (self.n_kv_heads, self.head_dim))
        q = F.rms_norm(q, (self.head_dim,))
        k = F.rms_norm(k, (self.head_dim,))
        if rope is not None and rope_cache is not None and rope_slice is not None:
            q_planet, k_planet = rope(q[:, rope_slice], k[:, rope_slice], rope_cache)
            q = _splice_rope(q, q_planet, rope_slice, rope.rotate_dim)
            k = _splice_rope(k, k_planet, rope_slice, rope.rotate_dim)
        q = q * self.q_gain.to(q.dtype)[None, None, :, None]
        # SDPA expects [B, H, j, head_dim]. Caller is responsible for
        # bf16 autocast on CUDA — that's what enables FA-2 dispatch.
        # Putting an inner autocast or `sdpa_kernel` here breaks AOT
        # autograd under `torch.compile` (see parameter-golf
        # `sota_train_gpt.py`: outer autocast around the whole training step,
        # no inner contexts).
        # On CPU SDPA dispatches to the math kernel — used only by tests.
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        attn_mask = None
        if valid_mask is not None:
            attn_mask = valid_mask[:, None, None, :]
        o = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            is_causal=False,
            enable_gqa=self.n_kv_heads != self.n_heads,
        )
        o = o.transpose(1, 2).flatten(-2)  # [B, T, H*head_dim]
        return self.out_proj(o)


class CrossAttention(nn.Module):
    """Cross-attention where a fixed latent set queries a masked source set."""

    def __init__(
        self,
        dim: int,
        n_heads: int,
        qk_gain_init: float = 1.0,
        *,
        n_kv_heads: int | None = None,
    ):
        super().__init__()
        n_kv_heads, head_dim, kv_dim = normalize_attention_config(
            dim, n_heads, n_kv_heads
        )
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.c_q = CastedLinear(dim, dim, bias=False)
        self.c_k = CastedLinear(dim, kv_dim, bias=False)
        self.c_v = CastedLinear(dim, kv_dim, bias=False)
        self.out_proj = CastedLinear(dim, dim, bias=False)
        self.q_gain = nn.Parameter(torch.full((n_heads,), float(qk_gain_init)))

        nn.init.orthogonal_(self.c_q.weight, gain=0.1)
        nn.init.orthogonal_(self.c_k.weight, gain=0.1)
        nn.init.orthogonal_(self.c_v.weight, gain=0.1)
        nn.init.zeros_(self.out_proj.weight)

    def forward(
        self,
        queries: torch.Tensor,
        keys_values: torch.Tensor,
        key_mask: torch.Tensor,
    ) -> torch.Tensor:
        # Rows with no fleets get one synthetic zero key. That keeps SDPA away
        # from all-masked rows while still contributing no fleet information.
        empty = ~key_mask.any(dim=-1, keepdim=True)
        first_key = torch.zeros_like(key_mask)
        first_key[:, :1] = True
        safe_mask = key_mask | (empty & first_key)
        keys_values = keys_values.masked_fill(~key_mask.unsqueeze(-1), 0.0)

        q = self.c_q(queries).unflatten(-1, (self.n_heads, self.head_dim))
        k = self.c_k(keys_values).unflatten(-1, (self.n_kv_heads, self.head_dim))
        v = self.c_v(keys_values).unflatten(-1, (self.n_kv_heads, self.head_dim))
        q = F.rms_norm(q, (self.head_dim,))
        k = F.rms_norm(k, (self.head_dim,))
        q = q * self.q_gain.to(q.dtype)[None, None, :, None]

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        attn_mask = safe_mask[:, None, None, :]
        o = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            enable_gqa=self.n_kv_heads != self.n_heads,
        )
        o = o.transpose(1, 2).flatten(-2)
        return self.out_proj(o)


class TransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        ff_dim: int,
        n_heads: int,
        dropout: float = 0.0,
        *,
        n_kv_heads: int | None = None,
        layer_idx: int = 0,
    ):
        super().__init__()
        # RMSNorm without learnable affine — parameter-golf moves all
        # per-channel scale into explicit `attn_scale`/`ff_scale`/`resid_mix`
        # so the norm itself is a pure unit-RMS projection. LayerNorm's γ
        # would otherwise be a free per-channel scale that competes with
        # those explicit params and isn't routed to the scalar AdamW group.
        self.ln1 = nn.RMSNorm(dim, elementwise_affine=False)
        self.ln2 = nn.RMSNorm(dim, elementwise_affine=False)
        self.attn = SelfAttention(dim, n_heads, n_kv_heads=n_kv_heads)
        # Depth attenuation disabled (no-op constant). Was `1/√(layer+1)`
        # following parameter-golf `Block.ln_scale_factor`, but with ortho
        # init at gain=0.1 the residual-stream growth is already bounded
        # by tiny step-0 weights — the depth scale was over-suppressing
        # gradient flow into deeper layers and giving worse KL than a
        # plain pre-norm block. Kept as an attribute (=1.0) so re-enabling
        # is a one-line change.
        self.ln_scale_factor = 1.0
        self.ff = nn.Sequential(
            CastedLinear(dim, ff_dim, bias=False),
            SquaredReLU(),
            CastedLinear(ff_dim, dim, bias=False),
        )
        self.drop = nn.Dropout(dropout)
        # Orthogonal init for the input projection (both dims ≥64 →
        # parameter-golf init policy). Output projection is zero-init
        # for the cold-start identity property and wins over orthogonal
        # because the zeros_ runs after.
        nn.init.orthogonal_(self.ff[0].weight, gain=0.1)
        # Zero-init the FF output projection. Attn already zero-inits its
        # own out_proj inside `SelfAttention.__init__`. Combined with
        # `resid_mix=(1, 0)` and `attn_scale=ff_scale=1`, each block is an
        # exact identity at step 0 — the residual stream carries
        # `embed_norm(embeds)` through unchanged.
        nn.init.zeros_(self.ff[-1].weight)
        # Per-channel learnable residual scaling on each branch
        # (parameter-golf `Block.attn_scale/mlp_scale`, sota_train_gpt.py:111).
        # Ones-init → identity to a vanilla pre-norm block at step 0, but the
        # optimizer can learn to attenuate each residual branch per channel.
        self.attn_scale = nn.Parameter(torch.ones(dim))
        self.ff_scale = nn.Parameter(torch.ones(dim))
        # Per-channel learnable mix between the current residual stream `x`
        # and the original block-stack input `x0` (parameter-golf `resid_mix`,
        # sota_train_gpt.py:111). Initialized to (1, 0) so the block sees `x`
        # unchanged at step 0. Lets each block independently re-mix the
        # un-transformed embedding stream when middle layers dominate norms.
        self.resid_mix = nn.Parameter(
            torch.stack([torch.ones(dim), torch.zeros(dim)])
        )

    def forward(
        self,
        x: torch.Tensor,
        x0: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
        *,
        rope: Rotary2D | None = None,
        rope_cache: Rotary2DCache | None = None,
        rope_slice: slice | None = None,
    ) -> torch.Tensor:
        # `x` and `x0` are dense padded [B, T, D]. Cast scale/mix params to
        # activation dtype to keep the bf16 residual path bf16 (see
        # parameter-golf `sota_train_gpt.py`).
        dt = x.dtype
        mix = self.resid_mix.to(dt)
        x_in = mix[0] * x + mix[1] * x0
        a = self.attn(
            self.ln1(x_in) * self.ln_scale_factor,
            valid_mask,
            rope=rope,
            rope_cache=rope_cache,
            rope_slice=rope_slice,
        )
        if self.drop.p:
            a = self.drop(a)
        x = x_in + self.attn_scale.to(dt) * a
        ff = self.ff(self.ln2(x) * self.ln_scale_factor)
        if self.drop.p:
            ff = self.drop(ff)
        x = x + self.ff_scale.to(dt) * ff
        return x


class FleetLatentBlock(nn.Module):
    """One Perceiver-style fleet compression block.

    Fleet latents cross-attend to raw fleet tokens, then self-attend among
    themselves. This costs O(L·F + L²) instead of fleet-fleet O(F²).
    """

    def __init__(
        self,
        dim: int,
        ff_dim: int,
        n_heads: int,
        dropout: float = 0.0,
        *,
        n_kv_heads: int | None = None,
        layer_idx: int = 0,
    ):
        super().__init__()
        self.latent_norm = nn.RMSNorm(dim, elementwise_affine=False)
        self.fleet_norm = nn.RMSNorm(dim, elementwise_affine=False)
        self.cross_attn = CrossAttention(dim, n_heads, n_kv_heads=n_kv_heads)
        self.cross_scale = nn.Parameter(torch.ones(dim))
        self.self_block = TransformerBlock(
            dim,
            ff_dim,
            n_heads,
            dropout,
            n_kv_heads=n_kv_heads,
            layer_idx=layer_idx,
        )

    def forward(
        self,
        latents: torch.Tensor,
        fleets: torch.Tensor,
        fleet_mask: torch.Tensor,
        x0: torch.Tensor,
    ) -> torch.Tensor:
        dt = latents.dtype
        latents = latents + self.cross_scale.to(dt) * self.cross_attn(
            self.latent_norm(latents),
            self.fleet_norm(fleets),
            fleet_mask,
        )
        return self.self_block(latents, x0)


class FleetLatentTokenizer(nn.Module):
    """Compress padded fleet tokens into a fixed learned latent set."""

    def __init__(
        self,
        dim: int,
        ff_dim: int,
        n_heads: int,
        num_latents: int,
        depth: int,
        dropout: float = 0.0,
        *,
        n_kv_heads: int | None = None,
    ):
        super().__init__()
        if num_latents < 1:
            raise ValueError("num_fleet_latents must be >= 1")
        if depth < 1:
            raise ValueError("fleet_tokenizer_depth must be >= 1")
        self.num_latents = num_latents
        self.fleet_latents = nn.Parameter(torch.zeros(num_latents, dim))
        nn.init.trunc_normal_(self.fleet_latents, std=0.02)
        self.input_norm = nn.RMSNorm(dim, elementwise_affine=False)
        self.layers = nn.ModuleList(
            [
                FleetLatentBlock(
                    dim,
                    ff_dim,
                    n_heads,
                    dropout,
                    n_kv_heads=n_kv_heads,
                    layer_idx=i,
                )
                for i in range(depth)
            ]
        )
        self.final_norm = nn.RMSNorm(dim, elementwise_affine=False)

    def forward(self, fleets: torch.Tensor, fleet_mask: torch.Tensor) -> torch.Tensor:
        b = fleets.shape[0]
        fleets = self.input_norm(fleets)
        fleets = fleets.masked_fill(~fleet_mask.unsqueeze(-1), 0.0)
        latents = self.fleet_latents.view(1, self.num_latents, -1).expand(b, -1, -1)
        x0 = latents
        for layer in self.layers:
            latents = layer(latents, fleets, fleet_mask, x0)
        return self.final_norm(latents)


# Beta samples live on an open interval. Clamp sampled fractions off exactly
# 0/1 before log-prob evaluation or simulator action materialization.
BETA_SAMPLE_EPS: float = 1e-6


class HLGaussLoss(nn.Module):
    """Histogram-loss-Gaussian distributional regression head.

    Thin adapter around `hl_gauss_pytorch.HLGaussLoss`, the same package used
    by Dreamer4's `SymExpHLGauss` wrapper. The adapter preserves the small API
    the rest of this repo expects: scalar target encoding via `target_probs`
    and scalar value recovery via `bins_to_scalar`.

    Forward semantics:
      - `target_probs(value)` encodes a scalar to a per-bin probability
        vector via the truncated-Gaussian CDF over the bin support, with
        renormalization so the truncated tails don't bias the target.
      - `bins_to_scalar(logits)` recovers E[V] = Σ softmax(logits)_i · c_i.
      - `loss(logits, target_probs)` is just F.cross_entropy on a per-element
        basis (caller is responsible for masking/reduction).

    With `symlog=True`, `min_value` / `max_value` are raw values. The library
    transforms those endpoints to symlog space for the histogram support and
    applies symexp when decoding scalar predictions.
    """

    def __init__(
        self,
        min_value: float = -1.0,
        max_value: float = 1.0,
        num_bins: int = 41,
        sigma_to_bin_ratio: float = 0.5,
        symlog: bool = False,
    ):
        super().__init__()
        if num_bins < 2:
            raise ValueError(f"num_bins must be ≥ 2, got {num_bins}")
        if min_value >= max_value:
            raise ValueError("min_value must be less than max_value")
        self.num_bins = num_bins
        self.min_value = min_value
        self.max_value = max_value
        self.symlog = bool(symlog)
        transform = _symlog if self.symlog else None
        inverse_transform = _symexp if self.symlog else None
        self.encoder = _LibraryHLGaussLoss(
            min_value=min_value,
            max_value=max_value,
            num_bins=num_bins,
            sigma_to_bin_ratio=sigma_to_bin_ratio,
            clamp_to_range=True,
            transform=transform,
            inverse_transform=inverse_transform,
        )

    def _apply(self, fn):  # type: ignore[no-untyped-def]
        out = super()._apply(fn)
        # `model.bfloat16()` casts buffers too. The histogram support defines
        # target placement and scalar decoding, so keep it in fp32 like the
        # previous local implementation did.
        for name, buffer in self.encoder.named_buffers(recurse=False):
            if buffer.is_floating_point() and buffer.dtype != torch.float32:
                self.encoder._buffers[name] = buffer.float()
        return out

    def target_probs(self, values: torch.Tensor) -> torch.Tensor:
        """Encode scalar `values` (any leading shape) into [..., num_bins]
        probabilities under a Gaussian centered at each value.

        We set `clamp_to_range=True` on the library encoder, so out-of-support
        targets concentrate mass at the boundary instead of producing a
        near-zero target distribution and a silent value-loss no-op.
        """
        return self.encoder.transform_to_probs(values.float())

    def bins_to_scalar(self, logits: torch.Tensor) -> torch.Tensor:
        return self.encoder(logits.float()).clamp(self.min_value, self.max_value)


def _symlog(x: torch.Tensor) -> torch.Tensor:
    return x.sign() * torch.log1p(x.abs())


def _symexp(x: torch.Tensor) -> torch.Tensor:
    return x.sign() * torch.expm1(x.abs())

@dataclass
class PolicyOutput:
    launch_logits: torch.Tensor       # [B, P] Bernoulli logits
    target_logits: torch.Tensor       # [B, P, P] masked target categorical logits
    value: torch.Tensor               # [B] — scalar value E[V] recovered from value_logits
    value_logits: torch.Tensor        # [B, num_bins] — distributional value head logits
    planet_owned_mask: torch.Tensor   # [B, P] bool
    planet_mask: torch.Tensor         # [B, P] bool
    planet_ids: torch.Tensor          # [B, P] long
    fraction_alpha: torch.Tensor | None = None  # [B, P] Beta concentration α
    fraction_beta: torch.Tensor | None = None   # [B, P] Beta concentration β
    # SAC's actor still adapts through this shared sampler as a squashed
    # Normal. PPO OrbitPolicy leaves these as None and uses the Beta fields.
    fraction_mean: torch.Tensor | None = None       # [B, P] Normal mean
    fraction_log_std: torch.Tensor | None = None    # [B, P] Normal log std


def _match_feature_width(x: torch.Tensor, expected: int) -> torch.Tensor:
    actual = x.shape[-1]
    if actual > expected:
        return x[..., :expected]
    return F.pad(x, (0, expected - actual))


class OrbitPolicy(nn.Module):
    def __init__(self, cfg: OrbitPolicyConfig):
        super().__init__()
        if cfg.encoder_backend not in {"dense", "fleet_latent"}:
            raise ValueError(f"unknown encoder_backend: {cfg.encoder_backend!r}")
        self.cfg = cfg
        self.planet_embed = CastedLinear(cfg.planet_features, cfg.dim)
        self.fleet_embed = CastedLinear(cfg.fleet_features, cfg.dim)
        self.fleet_tokenizer = (
            FleetLatentTokenizer(
                dim=cfg.dim,
                ff_dim=cfg.ff_dim,
                n_heads=cfg.n_heads,
                n_kv_heads=cfg.n_kv_heads,
                num_latents=cfg.num_fleet_latents,
                depth=cfg.fleet_tokenizer_depth,
                dropout=cfg.dropout,
            )
            if cfg.encoder_backend == "fleet_latent"
            else None
        )
        # Two learnable summary tokens (PMA-style). Initialized small so
        # they don't dominate the encoder at step 0 — gradient flow alone
        # will scale them up as the heads start using their output.
        # Store summary tokens flat. AOTAutograd can reduce broadcasted
        # `[1, 1, d]` parameters to `[d]` gradients in compiled backward at
        # larger PPO batch sizes; making the parameter itself `[d]` keeps the
        # expected gradient shape aligned with the reduction.
        self.actor_token = nn.Parameter(torch.zeros(cfg.dim))
        self.critic_token = nn.Parameter(torch.zeros(cfg.dim))
        nn.init.trunc_normal_(self.actor_token, std=0.02)
        nn.init.trunc_normal_(self.critic_token, std=0.02)
        # Embed-LN normalizes the residual-stream entry point. The per-token
        # embeddings (planet_embed, fleet_embed) and the two summary tokens
        # have heterogeneous scales, so a single LN here gives every block
        # the same input regime and stabilizes resid_mix's `x0` reference.
        self.embed_norm = nn.RMSNorm(cfg.dim, elementwise_affine=False)
        head_dim = cfg.dim // cfg.n_heads
        self.planet_rope = Rotary2D(
            head_dim,
            fraction=cfg.planet_rope_fraction,
            base=cfg.planet_rope_base,
        )
        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    cfg.dim,
                    cfg.ff_dim,
                    cfg.n_heads,
                    cfg.dropout,
                    n_kv_heads=cfg.n_kv_heads,
                    layer_idx=i,
                )
                for i in range(cfg.depth)
            ]
        )
        # Final-LN — standard for pre-norm transformers. Without it the
        # residual stream's scale grows ~unbounded (every block adds; nothing
        # normalizes the running sum), and downstream linear heads see
        # depth-dependent activation magnitudes.
        self.final_norm = nn.RMSNorm(cfg.dim, elementwise_affine=False)
        # `target_query` consumes [planet_h || h_actor] → 2·dim. `target_key`
        # stays at `dim` because adding the same h_actor to every key shifts
        # all `q·k` row-wise by a constant and cancels in the softmax — it
        # can't change the relative ranking of targets. Putting it on the
        # query side only is what gives the actor token bite.
        self.target_query = CastedLinear(2 * cfg.dim, cfg.dim, bias=False)
        self.target_key = CastedLinear(cfg.dim, cfg.dim, bias=False)
        # Target attention temperature (per parameter-golf `q_gain` pattern,
        # `sota_train_gpt.py:101,104`). We RMS-norm Q & K below, then scale
        # by this learnable scalar before the softmax. Pins the target-logit
        # magnitude regardless of how `target_query`/`target_key` weights
        # drift under Muon — the readout had been the only unprotected
        # attention path in the model.
        # Init=1.0 (not pg's 5.0): pg's softmax is over a 50k-token vocab
        # where each per-token log_prob is small enough that gain=5 is fine,
        # but our target softmax is over ≤17 slots — gain=5 here makes the
        # cold-start distribution sharp, so first-update Δlog_prob (and
        # approx_kl) blows up. Init=1 reproduces the standard sqrt(d)-scaled
        # attention temperature; the optimizer can sharpen via the control-LR
        # group as training proceeds.
        self.target_q_gain = nn.Parameter(torch.tensor(1.0))
        # Orthogonal init at gain=1 (parameter-golf `_init_weights` policy
        # for Linears with both dims ≥64). Spectral norm 1 means a Muon
        # NS5 update of magnitude `muon_lr` produces a `muon_lr`-fraction
        # *relative* perturbation in outputs — predictable first-step
        # behavior, no dependence on the kaiming_uniform fan-in scale.
        # An earlier Xavier(gain=1) attempt blew up first-step KL because
        # Xavier's variance is *larger* than kaiming default; orthogonal at
        # gain=1 has unit spectral norm but still bounded variance, which
        # is what we actually want for log-prob stability.
        # Mild reduction vs trunk's 0.1 — but largely vacuous: the einsum
        # output is QK-RMSNormed and tempered by `target_q_gain`, so init
        # weight magnitude on these doesn't translate to logit magnitude.
        nn.init.orthogonal_(self.target_query.weight, gain=0.05)
        nn.init.orthogonal_(self.target_key.weight, gain=0.05)
        # Per-planet launch Bernoulli. It sees the same local+global context
        # as the fraction head, but is independent of target count; target
        # selection is a separate conditional categorical below. Bias negative
        # so cold-start prefers not launching until advantage says otherwise.
        self.launch_head = CastedLinear(2 * cfg.dim, 1)
        nn.init.zeros_(self.launch_head.weight)
        nn.init.constant_(self.launch_head.bias, -1.5)
        # 2·dim input for the same reason as target_query. Two independent
        # readouts parameterize a native-support Beta fraction distribution:
        # alpha,beta = 1 + softplus(raw). The +1 enforces the Dreamer4
        # unimodal path and avoids boundary-seeking exploration.
        self.fraction_alpha_head = CastedLinear(2 * cfg.dim, 1)
        self.fraction_beta_head = CastedLinear(2 * cfg.dim, 1)
        # gain=0.01 — CleanRL PPO's canonical actor-readout init
        # (`ppo_continuous_action.py:127`). Zero biases make alpha == beta at
        # init, so the deterministic fraction starts at 0.5.
        nn.init.orthogonal_(self.fraction_alpha_head.weight, gain=0.01)
        nn.init.orthogonal_(self.fraction_beta_head.weight, gain=0.01)
        nn.init.zeros_(self.fraction_alpha_head.bias)
        nn.init.zeros_(self.fraction_beta_head.bias)
        # Distributional value head — emits logits over `value_num_bins`
        # bins. Scalar V is recovered from these via `HLGaussLoss.bins_to_scalar`.
        # The default uses Dreamer4-style symlog buckets over a wide raw
        # projected-margin support.
        self.value_encoder = HLGaussLoss(
            min_value=cfg.value_min,
            max_value=cfg.value_max,
            num_bins=cfg.value_num_bins,
            symlog=cfg.value_symlog,
        )
        self.value_head = nn.Sequential(
            CastedLinear(cfg.dim, cfg.value_hidden, bias=False),
            SquaredReLU(),
            CastedLinear(cfg.value_hidden, cfg.value_num_bins, bias=False),
        )
        # Orthogonal init for the value-head input projection (both dims
        # ≥64 if `value_hidden ≥ 64`).
        if (
            self.value_head[0].weight.shape[0] >= 64
            and self.value_head[0].weight.shape[1] >= 64
        ):
            nn.init.orthogonal_(self.value_head[0].weight, gain=0.1)
        # Zero-init the value head's last layer so V(s) is the *uniform*
        # distribution over bins at step 0 — same parameter-golf lever as
        # the action heads. Recovered scalar starts at 0 (mean of bin
        # centers on the symmetric [-1, 1] support) so cold-start advantage
        # is not corrupted by a random bin distribution.
        nn.init.zeros_(self.value_head[-1].weight)
        # Cached on-device self-target mask. P is bounded by MAX_PLANETS,
        # so we allocate once at module init and slice per-forward instead
        # of allocating a fresh `torch.eye` every step (~num_envs × episode
        # _steps allocations per PPO iteration on GPU).
        from .features import MAX_PLANETS

        self.register_buffer(
            "_self_target_mask",
            torch.eye(MAX_PLANETS, dtype=torch.bool),
            persistent=False,
        )

    def _embed_tokens(
        self, feats: EncodedObs
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        Rotary2DCache | None,
        slice,
        int,
        int,
    ]:
        # Promote to batch dim if not already.
        if feats.planet_feats.dim() == 2:
            planet_feats = feats.planet_feats.unsqueeze(0)
            planet_mask = feats.planet_mask.unsqueeze(0)
            fleet_feats = feats.fleet_feats.unsqueeze(0)
            fleet_mask = feats.fleet_mask.unsqueeze(0)
        else:
            planet_feats = feats.planet_feats
            planet_mask = feats.planet_mask
            fleet_feats = feats.fleet_feats
            fleet_mask = feats.fleet_mask

        b, p, _ = planet_feats.shape
        f = fleet_feats.shape[1]
        if planet_feats.shape[-1] != self.planet_embed.in_features:
            planet_feats = _match_feature_width(
                planet_feats, self.planet_embed.in_features
            )
        if fleet_feats.shape[-1] != self.fleet_embed.in_features:
            fleet_feats = _match_feature_width(fleet_feats, self.fleet_embed.in_features)

        h_p = self.planet_embed(planet_feats)
        h_f = self.fleet_embed(fleet_feats)
        if self.fleet_tokenizer is not None:
            h_f = self.fleet_tokenizer(h_f, fleet_mask)
            fleet_mask = torch.ones(
                b,
                h_f.shape[1],
                dtype=torch.bool,
                device=fleet_mask.device,
            )
            f = h_f.shape[1]
        # Prepend the two summary tokens, broadcast to batch dim. Parameters
        # are stored flat so compiled backward's broadcast reduction returns
        # `[d]`, matching the actual parameter shape.
        actor_t = self.actor_token.view(1, 1, -1).expand(b, 1, -1)
        critic_t = self.critic_token.view(1, 1, -1).expand(b, 1, -1)
        h = torch.cat([actor_t, critic_t, h_p, h_f], dim=1)
        # Normalize the residual-stream entry point. embed_norm runs on
        # padded `[B, T, D]` since LN is per-token — padded positions are
        # normalized too; masks make padded tokens inert in attention.
        h = self.embed_norm(h)
        summary_mask = torch.ones(b, 2, dtype=torch.bool, device=planet_mask.device)
        full_mask = torch.cat([summary_mask, planet_mask, fleet_mask], dim=1)
        # Planet features store centered normalized coordinates:
        # ((x - 50) / 100, (y - 50) / 100). RoPE uses physical board
        # coordinates so the phase scale is meaningful on the 100x100 map.
        planet_xy = planet_feats[..., :2] * _PLANET_XY_SCALE + _PLANET_XY_OFFSET
        rope_cache = self.planet_rope.cache(planet_xy, h.dtype)
        planet_slice = slice(2, 2 + p)
        return h, full_mask, planet_mask, fleet_mask, rope_cache, planet_slice, p, f

    def _split_encoded(
        self,
        h: torch.Tensor,
        planet_mask: torch.Tensor,
        fleet_mask: torch.Tensor,
        p: int,
        f: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        h_actor = h[:, 0]                   # [B, d]
        h_critic = h[:, 1]                  # [B, d]
        planet_h = h[:, 2 : 2 + p]          # [B, P, d]
        fleet_h = h[:, 2 + p : 2 + p + f]   # [B, F, d]
        token_mask = torch.cat([planet_mask, fleet_mask], dim=1)
        return planet_h, fleet_h, h_actor, h_critic, token_mask

    def _encode_dense(
        self,
        h: torch.Tensor,
        full_mask: torch.Tensor,
        planet_mask: torch.Tensor,
        fleet_mask: torch.Tensor,
        rope_cache: Rotary2DCache | None,
        planet_slice: slice,
        p: int,
        f: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Graphable padded encoder.

        This is the default CUDA path. It intentionally spends a little extra
        attention work on padded slots to avoid NestedTensor Python subclass
        dispatch and jagged pack/unpack overhead.
        """
        x0 = h
        for layer in self.layers:
            h = layer(
                h,
                x0,
                full_mask,
                rope=self.planet_rope,
                rope_cache=rope_cache,
                rope_slice=planet_slice,
            )
        h = self.final_norm(h)
        return self._split_encoded(h, planet_mask, fleet_mask, p, f)

    def encode(
        self, feats: EncodedObs
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run the transformer over [actor, critic, planets..., fleets...].

        Returns `(planet_h, fleet_h, h_actor, h_critic, token_mask)` where
        `token_mask` is the planets+fleet-context mask. Dense padded CUDA is
        the default because rollout profiling showed NestedTensor dispatch,
        not attention compute, dominating model latency.
        """
        h, full_mask, planet_mask, fleet_mask, rope_cache, planet_slice, p, f = (
            self._embed_tokens(feats)
        )
        return self._encode_dense(
            h,
            full_mask,
            planet_mask,
            fleet_mask,
            rope_cache,
            planet_slice,
            p,
            f,
        )

    def forward(self, feats: EncodedObs, *, include_value: bool = True) -> PolicyOutput:
        planet_h, _fleet_h, h_actor, h_critic, _token_mask = self.encode(feats)
        b, p, d = planet_h.shape

        # Promote scalar mask/garrison/id to batch dim if not already.
        def _b(t: torch.Tensor) -> torch.Tensor:
            return t.unsqueeze(0) if t.dim() == 1 else t

        planet_owned = _b(feats.planet_owned_mask)
        planet_mask = _b(feats.planet_mask)
        planet_ids = _b(feats.planet_ids)

        # Concatenate the actor token onto each per-planet rep — global
        # context for the action heads. Broadcast: [B,1,d] → [B,P,d].
        actor_ctx = h_actor.unsqueeze(1).expand(-1, p, -1)
        planet_with_ctx = torch.cat([planet_h, actor_ctx], dim=-1)  # [B, P, 2d]

        # Target attention: query carries actor context (2d→d), key stays
        # plain (d→d). Putting the actor concat on the *key* side too would
        # add a column-constant term to `q·k` that cancels in the softmax.
        q = self.target_query(planet_with_ctx)
        k = self.target_key(planet_h)
        # QK-RMSNorm + learnable gain — same pattern as `SelfAttention`.
        # `F.rms_norm` along the last dim pins ‖q‖ and ‖k‖ to √d regardless
        # of weight magnitude, so the post-softmax target distribution has
        # a magnitude bound that doesn't drift with the projection norms.
        q = F.rms_norm(q, (q.size(-1),))
        k = F.rms_norm(k, (k.size(-1),))
        q = q * self.target_q_gain.to(q.dtype)
        # [B, P, P] — keep the canonical 1/√d divisor; `target_q_gain`
        # multiplies on top, mirroring how trunk SDPA's auto-scale plus
        # `q_gain` compose.
        logits = torch.einsum("bid,bjd->bij", q, k) / (d**0.5)
        # Mask out padded *targets*.
        logits = logits.masked_fill(~planet_mask.unsqueeze(1), float("-inf"))
        # Mask self-targets (diagonal). The simulator silently no-ops a
        # send-to-self anyway; without this mask the policy can put
        # probability mass on a meaningless action and the entropy term
        # rewards it. Buffer is preallocated; slice for the actual P.
        logits = logits.masked_fill(
            self._self_target_mask[:p, :p].unsqueeze(0), float("-inf")
        )
        # Per-planet launch logits are a Bernoulli sibling of target/fraction,
        # not an extra target slot. Rows with no legal target get a very low
        # launch logit; the target categorical is still sanitized downstream
        # for distribution APIs, but such rows are behaviorally no-launch.
        valid_target_count = (planet_mask.sum(dim=-1, keepdim=True) - 1).clamp_min(0)
        launch_logits = self.launch_head(planet_with_ctx).squeeze(-1)
        launch_logits = launch_logits.masked_fill(valid_target_count <= 0, -20.0)
        target_logits = logits  # [B, P, P]

        # Fraction head: native-support unimodal Beta, matching the CleanRL
        # IterThink v24 / Dreamer4 beta path.
        fraction_alpha = 1.0 + F.softplus(
            self.fraction_alpha_head(planet_with_ctx).squeeze(-1).float()
        )
        fraction_beta = 1.0 + F.softplus(
            self.fraction_beta_head(planet_with_ctx).squeeze(-1).float()
        )

        if include_value:
            # Value: distributional head over the dedicated critic token.
            # Logits are returned for distributional CE loss + value clipping;
            # the scalar `value` is recovered via E[V] = Σ p_i · center_i.
            value_logits = self.value_head(h_critic)  # [B, num_bins]
            value = self.value_encoder.bins_to_scalar(value_logits)
        else:
            value = h_critic.new_empty((b,), dtype=torch.float32)
            value_logits = h_critic.new_empty((b, 0), dtype=torch.float32)

        return PolicyOutput(
            launch_logits=launch_logits,
            target_logits=target_logits,
            value=value,
            value_logits=value_logits,
            planet_owned_mask=planet_owned,
            planet_mask=planet_mask,
            planet_ids=planet_ids,
            fraction_alpha=fraction_alpha,
            fraction_beta=fraction_beta,
        )
