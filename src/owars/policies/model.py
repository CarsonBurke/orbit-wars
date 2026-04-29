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
and fraction_head take 2d-wide inputs and project to their natural output
dim (d for query/key, 2 for the Beta's mean/concentration heads). Down-projecting
`[planet_h || h_actor]` back to d would discard exactly the global-context
capacity the extra token was added to provide.

**Action factorization.** Per source planet, the actor emits a Bernoulli
launch decision, a masked categorical target distribution conditional on
launching, and a Beta fraction distribution conditional on launching. This
keeps "should this planet act?" independent of the number of legal target
planets; target count should affect *where* probability mass goes, not whether
the source launches at all.

**No angle head.** The launch angle is computed exactly via an iterative
lead-intercept solver in `sampling.py`.

**Fraction head is a Beta(α, β)** on [0, 1] — the natural distribution for
"fraction of garrison to send." The network predicts Beta mode μ and extra
concentration c, then derives:
    α = 1 + c · μ
    β = 1 + c · (1 − μ)
This keeps α,β ≥ 1 (unimodal/concave; Chou et al. 2017) without letting the
max-spread floor become an absorbing state. If concentration bottoms out,
the mode still has a direct gradient through both α and β. Concentration is
bounded, giving a persistent exploration floor without an entropy bonus.

**Distributional value head (HL-Gauss).** dreamer4 (`dreamer4.py:722–805`)
predicts a categorical over a fixed bin support and trains it with
cross-entropy against a Gaussian-kernel-smoothed target distribution
(Imani et al. 2018, Farebrother et al. 2024). The expected value
E[V|s] = Σ p_i · center_i is the recovered scalar; the learning gradient
is bounded by 1 (CE) instead of unbounded (MSE on a possibly-misscaled
scalar), which makes the critic dramatically more robust to early
mis-prediction.

We deliberately drop dreamer4's symlog transform — our reward is
terminal-only ±1 with γ=1, so unshaped returns live in [-1, 1]. We
default the bin support to [-2, 2] (51 bins) so margin shaping
(`RewardCfg.margin_scale > 0` adds up to ±2·margin_scale) has 1.0 of
headroom on each side; configs that go past that should widen via
`OrbitPolicyConfig.value_min/max`. σ = 0.5 × bin_size (dreamer4
default), and `target_probs` clips out-of-range targets to the
boundary bin so the head degrades gracefully rather than producing NaN.

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

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import OrbitPolicyConfig
from .features import EncodedObs


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

    NJT-safe: `relu` has a nested-jagged kernel, unlike `leaky_relu`.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(x).square()


class CastedLinear(nn.Linear):
    """Linear with `forward` that casts the fp32-master weight (and bias) to
    the input dtype on each call (parameter-golf `sota_train_gpt.py:80`).

    The store-master / cast-on-forward pattern is the same precision regime
    you'd get from `torch.autocast(bf16)` over a vanilla `nn.Linear`, but
    explicit: the dtype boundary is in this method, no autocast cache,
    no implicit interaction with `torch.compile` or nested-jagged tracing.
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

class SelfAttention(nn.Module):
    """Multi-head self-attention with QK-norm + per-head q_gain on a
    nested-jagged input — strict-flash dispatch.

    parameter-golf pattern (`sota_train_gpt.py:CausalSelfAttention`):
      1. Project x → Q, K, V.
      2. **RMSNorm Q and K per-head** along the head-dim. Pins ‖q_h‖ and
         ‖k_h‖ to fixed magnitude so attention-logit magnitude does not
         drift as the QKV projections move under Muon (or any optimizer).
      3. Multiply Q by per-head learnable `q_gain` (init=5). This is the
         *attention temperature*: high gain → sharp softmax, low → flat.
      4. SDPA on a `torch.nested` jagged tensor under `sdpa_kernel(
         [SDPBackend.FLASH_ATTENTION])` — packed valid tokens, no
         attn_mask, no padding wasted in the kernel. Real FA-2.
      5. Output projection (zero-init for cold-start identity).

    Variable-length set masking is handled *outside* the kernel: the
    encoder packs valid tokens into a jagged tensor (`OrbitPolicy.encode`)
    and unpacks back to padded `[B, T, D]` after the block stack. SDPA's
    flash backend rejects any non-causal mask, and PyTorch silently falls
    back to mem-efficient when `attn_mask` is non-None — using nested-jagged
    is the only way to *guarantee* FA-2 for bidirectional variable-length
    attention.
    """

    def __init__(self, dim: int, n_heads: int, qk_gain_init: float = 5.0):
        super().__init__()
        if dim % n_heads != 0:
            raise ValueError(f"dim {dim} not divisible by n_heads {n_heads}")
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        # Three separate Q/K/V projections (parameter-golf style). NestedTensor
        # supports `unbind(dim=0)` only, so a fused-QKV → unbind doesn't work.
        # Splitting at the projection level is also more idiomatic in modern
        # PyTorch attention impls.
        self.c_q = CastedLinear(dim, dim, bias=False)
        self.c_k = CastedLinear(dim, dim, bias=False)
        self.c_v = CastedLinear(dim, dim, bias=False)
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
        self, x: torch.Tensor, valid_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        # `x` is either a nested-jagged tensor of shape [B, j, D] where j
        # varies per batch, or a dense padded tensor [B, T, D] with
        # `valid_mask=True` for real tokens.
        q = self.c_q(x).unflatten(-1, (self.n_heads, self.head_dim))
        k = self.c_k(x).unflatten(-1, (self.n_heads, self.head_dim))
        v = self.c_v(x).unflatten(-1, (self.n_heads, self.head_dim))
        q = F.rms_norm(q, (self.head_dim,))
        k = F.rms_norm(k, (self.head_dim,))
        q = q * self.q_gain.to(q.dtype)[None, None, :, None]
        # SDPA expects [B, H, j, head_dim]. Caller is responsible for
        # bf16 autocast on CUDA — that's what enables FA-2 dispatch.
        # Putting an inner autocast or `sdpa_kernel` here breaks AOT
        # autograd's nested-jagged subclass accounting under
        # `torch.compile` (see parameter-golf `sota_train_gpt.py`: outer
        # autocast around the whole training step, no inner contexts).
        # On CPU SDPA dispatches to the math kernel — used only by tests.
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        attn_mask = None
        if valid_mask is not None:
            attn_mask = valid_mask[:, None, None, :]
        o = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, is_causal=False)
        o = o.transpose(1, 2).flatten(-2)  # [B, j, H*head_dim] nested
        return self.out_proj(o)


class TransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        ff_dim: int,
        n_heads: int,
        dropout: float = 0.0,
        *,
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
        self.attn = SelfAttention(dim, n_heads)
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
    ) -> torch.Tensor:
        # `x` and `x0` are nested-jagged [B, j, D] or dense padded
        # [B, T, D]. Cast scale/mix params to activation dtype to keep the
        # bf16 residual path bf16 (see parameter-golf `sota_train_gpt.py`).
        dt = x.dtype
        mix = self.resid_mix.to(dt)
        x_in = mix[0] * x + mix[1] * x0
        a = self.attn(self.ln1(x_in) * self.ln_scale_factor, valid_mask)
        if self.drop.p:
            a = self.drop(a)
        x = x_in + self.attn_scale.to(dt) * a
        ff = self.ff(self.ln2(x) * self.ln_scale_factor)
        if self.drop.p:
            ff = self.drop(ff)
        x = x + self.ff_scale.to(dt) * ff
        return x


# Bounds for the mode+concentration Beta fraction head. Extra concentration
# c is bounded to [min, max]; α=1+c·μ and β=1+c·(1−μ), so α,β stay ≥1.
FRACTION_MEAN_EPS: float = 1e-4
FRACTION_CONCENTRATION_MIN: float = 1.0
FRACTION_CONCENTRATION_INIT: float = 1.4
FRACTION_CONCENTRATION_MAX: float = 20.0


class HLGaussLoss(nn.Module):
    """Histogram-loss-Gaussian distributional regression head.

    Lifted in spirit from dreamer4 (`dreamer4.py:722–805` `SymExpHLGauss`)
    and the underlying Imani et al. 2018 / Farebrother et al. 2024 design.
    For our [-1, 1] return range we drop the symlog transform.

    Forward semantics:
      - `target_probs(value)` encodes a scalar to a per-bin probability
        vector via the truncated-Gaussian CDF over the bin support, with
        renormalization so the truncated tails don't bias the target.
      - `bins_to_scalar(logits)` recovers E[V] = Σ softmax(logits)_i · c_i.
      - `loss(logits, target_probs)` is just F.cross_entropy on a per-element
        basis (caller is responsible for masking/reduction).

    All math casts to fp32 internally — the support buffer is bf16 after
    `model.bfloat16()` but bin edges in [-1, 1] need more than 7 mantissa
    bits to be useful as targets.
    """

    def __init__(
        self,
        min_value: float = -1.0,
        max_value: float = 1.0,
        num_bins: int = 41,
        sigma_to_bin_ratio: float = 0.5,
    ):
        super().__init__()
        if num_bins < 2:
            raise ValueError(f"num_bins must be ≥ 2, got {num_bins}")
        self.num_bins = num_bins
        self.min_value = min_value
        self.max_value = max_value
        self.bin_size = (max_value - min_value) / num_bins
        self.sigma = sigma_to_bin_ratio * self.bin_size
        self.sigma_times_sqrt_two = (2.0 ** 0.5) * self.sigma

    def _support(self, ref: torch.Tensor) -> torch.Tensor:
        """Bin-edge tensor `[num_bins+1]` in fp32 on `ref`'s device.

        We deliberately do *not* `register_buffer` the support — `model.bfloat16()`
        casts buffers to bf16, but bin edges in [-2, 2] need fp32 (bf16 step
        ≈ 0.016 vs 0.08 bin spacing → ~20% bin shift). Recomputing per call
        on the input device is trivial (tens of floats).
        """
        return torch.linspace(
            self.min_value,
            self.max_value,
            self.num_bins + 1,
            dtype=torch.float32,
            device=ref.device,
        )

    def _centers(self, support: torch.Tensor) -> torch.Tensor:
        return (support[:-1] + support[1:]) / 2

    def target_probs(self, values: torch.Tensor) -> torch.Tensor:
        """Encode scalar `values` (any leading shape) into [..., num_bins]
        probabilities under a Gaussian centered at each value.

        We deliberately diverge from dreamer4's library default
        (`hl_gauss_pytorch/hl_gauss.py:131-146`, `clamp_to_range=False`)
        and clamp the target to `[min_value, max_value]` upfront. Their
        255-bin symlog [-20, 20] support corresponds to ±5e8 in raw units
        — effectively never out-of-range. Our compact [-2, 2] support
        could be exceeded if a config dials margin shaping past 0.5;
        without the clamp the truncated-Gaussian collapses (cdf saturates
        on both ends, z ≈ 0, divide by `clamp_min(1e-10)` yields an
        all-zero target distribution → zero CE gradient). The clamp
        concentrates mass at the boundary bin instead, degrading
        gracefully rather than silently no-op'ing the value loss.
        """
        v = values.float().clamp(self.min_value, self.max_value)
        support = self._support(v)  # [num_bins+1]
        # cdf[..., i] = erf((support_i − v) / (σ√2))
        cdf = torch.erf((support - v.unsqueeze(-1)) / self.sigma_times_sqrt_two)
        bin_probs = cdf[..., 1:] - cdf[..., :-1]
        z = (cdf[..., -1] - cdf[..., 0]).clamp_min(1e-10)
        return bin_probs / z.unsqueeze(-1)

    def bins_to_scalar(self, logits: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(logits.float(), dim=-1)
        support = self._support(probs)
        centers = self._centers(support)
        return (probs * centers).sum(dim=-1)

def _logit(p: float) -> float:
    return math.log(p / (1.0 - p))


def _fraction_beta_params(
    mode_logit: torch.Tensor,
    concentration_logit: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Map raw fraction heads to Beta(α, β) through mode+concentration.

    Mode uses a sigmoid range instead of a clamp, preserving gradients until
    true logit saturation. Concentration may saturate at its floor, but μ can
    still move α and β, so the lower-concentration state is not absorbing.
    """
    mode = FRACTION_MEAN_EPS + (1.0 - 2.0 * FRACTION_MEAN_EPS) * torch.sigmoid(
        mode_logit
    )
    span = FRACTION_CONCENTRATION_MAX - FRACTION_CONCENTRATION_MIN
    concentration = FRACTION_CONCENTRATION_MIN + span * torch.sigmoid(
        concentration_logit
    )
    alpha = 1.0 + concentration * mode
    beta = 1.0 + concentration * (1.0 - mode)
    return alpha, beta


@dataclass
class PolicyOutput:
    launch_logits: torch.Tensor       # [B, P] Bernoulli logits
    target_logits: torch.Tensor       # [B, P, P] masked target categorical logits
    fraction_alpha: torch.Tensor      # [B, P]  Beta α from mode+concentration
    fraction_beta: torch.Tensor       # [B, P]  Beta β from mode+concentration
    value: torch.Tensor               # [B] — scalar value E[V] recovered from value_logits
    value_logits: torch.Tensor        # [B, num_bins] — distributional value head logits
    planet_owned_mask: torch.Tensor   # [B, P] bool
    planet_mask: torch.Tensor         # [B, P] bool
    planet_ids: torch.Tensor          # [B, P] long


def _match_feature_width(x: torch.Tensor, expected: int) -> torch.Tensor:
    actual = x.shape[-1]
    if actual > expected:
        return x[..., :expected]
    return F.pad(x, (0, expected - actual))


class OrbitPolicy(nn.Module):
    def __init__(self, cfg: OrbitPolicyConfig):
        super().__init__()
        if cfg.encoder_backend not in {"dense", "nested"}:
            raise ValueError(f"unknown encoder_backend: {cfg.encoder_backend!r}")
        self.cfg = cfg
        self.planet_embed = CastedLinear(cfg.planet_features, cfg.dim)
        self.fleet_embed = CastedLinear(cfg.fleet_features, cfg.dim)
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
        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    cfg.dim,
                    cfg.ff_dim,
                    cfg.n_heads,
                    cfg.dropout,
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
        # 2·dim input for the same reason as target_query. Outputs raw
        # (mode_logit, concentration_logit) for the Beta fraction distribution.
        self.fraction_head = CastedLinear(2 * cfg.dim, 2)
        # gain=0.01 — cleanrl PPO's canonical actor-readout init
        # (`ppo_continuous_action_pmpo_d4_beta_relusq_v3.py:182`,
        # `ppo_continuous_action.py:127`). Bias the concentration head so
        # init is Beta(1.7, 1.7), matching the old near-uniform cold start.
        nn.init.orthogonal_(self.fraction_head.weight, gain=0.01)
        nn.init.zeros_(self.fraction_head.bias)
        conc_p = (
            (FRACTION_CONCENTRATION_INIT - FRACTION_CONCENTRATION_MIN)
            / (FRACTION_CONCENTRATION_MAX - FRACTION_CONCENTRATION_MIN)
        )
        nn.init.constant_(self.fraction_head.bias[1], _logit(conc_p))
        # Distributional value head — emits logits over `value_num_bins`
        # bins on the [-1, 1] support. Scalar V is recovered from these
        # via `HLGaussLoss.bins_to_scalar`. dreamer4 uses 255 bins on a
        # symlog'd support; with our terminal-only ±1 reward (returns ∈
        # [-1, 1]) we use a coarser linear bin set.
        self.value_encoder = HLGaussLoss(
            min_value=cfg.value_min,
            max_value=cfg.value_max,
            num_bins=cfg.value_num_bins,
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
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int, int]:
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
        # Prepend the two summary tokens, broadcast to batch dim. Parameters
        # are stored flat so compiled backward's broadcast reduction returns
        # `[d]`, matching the actual parameter shape.
        actor_t = self.actor_token.view(1, 1, -1).expand(b, 1, -1)
        critic_t = self.critic_token.view(1, 1, -1).expand(b, 1, -1)
        h = torch.cat([actor_t, critic_t, h_p, h_f], dim=1)
        # Normalize the residual-stream entry point. embed_norm runs on
        # padded `[B, T, D]` since LN is per-token — padded positions are
        # normalized too but get dropped by the nested-jagged pack below.
        h = self.embed_norm(h)
        summary_mask = torch.ones(b, 2, dtype=torch.bool, device=planet_mask.device)
        full_mask = torch.cat([summary_mask, planet_mask, fleet_mask], dim=1)
        return h, full_mask, planet_mask, fleet_mask, p, f

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
            h = layer(h, x0, full_mask)
        h = self.final_norm(h)
        return self._split_encoded(h, planet_mask, fleet_mask, p, f)

    @torch.compiler.disable
    def _encode_nested(
        self,
        h: torch.Tensor,
        full_mask: torch.Tensor,
        planet_mask: torch.Tensor,
        fleet_mask: torch.Tensor,
        p: int,
        f: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Nested-jagged encoder retained for A/B benchmarks.

        This can dispatch SDPA to flash on valid packed tokens, but the PyTorch
        NestedTensor subclass overhead is slower than dense masked attention in
        our rollout profiles.
        """
        lengths = full_mask.sum(dim=-1)  # [B]
        b = int(lengths.shape[0])
        offsets = torch.zeros(b + 1, dtype=torch.int64, device=h.device)
        offsets[1:] = lengths.cumsum(0)
        # Boolean indexing flattens valid tokens row-major across the batch.
        values = h[full_mask]
        h_nt = torch.nested.nested_tensor_from_jagged(values, offsets)
        # `x0` is the post-embed-norm residual stream; each block's resid_mix
        # mixes against this fixed reference.
        x0_nt = h_nt
        for layer in self.layers:
            h_nt = layer(h_nt, x0_nt)
        h_nt = self.final_norm(h_nt)
        # Unpack: scatter valid tokens back to original padded positions.
        out_values = h_nt.values()  # [total_valid, D]
        h = torch.zeros_like(h)
        h[full_mask] = out_values
        return self._split_encoded(h, planet_mask, fleet_mask, p, f)

    def encode(
        self, feats: EncodedObs
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run the transformer over [actor, critic, planets..., fleets...].

        Returns `(planet_h, fleet_h, h_actor, h_critic, token_mask)` where
        `token_mask` is the planets+fleets mask. Dense padded CUDA is the
        default because rollout profiling showed NestedTensor dispatch, not
        attention compute, dominating model latency.
        """
        h, full_mask, planet_mask, fleet_mask, p, f = self._embed_tokens(feats)
        if self.cfg.encoder_backend == "nested":
            return self._encode_nested(h, full_mask, planet_mask, fleet_mask, p, f)
        return self._encode_dense(h, full_mask, planet_mask, fleet_mask, p, f)

    def forward(self, feats: EncodedObs) -> PolicyOutput:
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

        # Fraction head: raw (mode_logit, concentration_logit) → Beta(α, β).
        # Mode remains trainable even when concentration hits its floor.
        offs = self.fraction_head(planet_with_ctx).float()
        fraction_alpha, fraction_beta = _fraction_beta_params(
            offs[..., 0], offs[..., 1]
        )

        # Value: distributional head over the dedicated critic token.
        # Logits are returned for distributional CE loss + value clipping;
        # the scalar `value` is recovered via E[V] = Σ p_i · center_i.
        value_logits = self.value_head(h_critic)  # [B, num_bins]
        value = self.value_encoder.bins_to_scalar(value_logits)

        return PolicyOutput(
            launch_logits=launch_logits,
            target_logits=target_logits,
            fraction_alpha=fraction_alpha,
            fraction_beta=fraction_beta,
            value=value,
            value_logits=value_logits,
            planet_owned_mask=planet_owned,
            planet_mask=planet_mask,
            planet_ids=planet_ids,
        )
