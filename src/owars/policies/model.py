"""OrbitPolicy — set-transformer encoder + factored action heads.

Architecture:
  [ACTOR] [CRITIC] [GLOBAL] planet_tokens fleet_tokens
              │
              ▼
  [N × Transformer block] ─► token reps
              │
              ├─► h_actor (broadcast)  ─► concat onto each planet rep ─►
              │                             launch/target/fraction heads
              ├─► h_critic             ─► distributional value head (HL-Gauss)
              ├─► h_global             ─► observation-level context in trunk
              ├─► planet_h             ─► target_key (per-planet rep stays d-dim)
              └─► fleet_h              ─► (consumed only by encoder cross-attention)

Learnable prefix tokens (Set-Transformer-style PMA) prepend the input set.
The critic token gives the value head a *learned* aggregator instead
of a mean-pool over many planet/fleet tokens, where decision-relevant tokens
otherwise drown in the average. The actor token is concatenated as global
context onto each per-planet rep before the action heads — the per-planet
target/fraction heads see "what's the joint plan look like" without us
having to go autoregressive. The global token carries scalar observation
context such as game clock while remaining outside the planet action vocabulary.

We deliberately do **not** down-project after the actor concat: target_query
and the Beta fraction heads take 2d-wide inputs and project to their natural
output dim (d for query/key, 1 for each concentration head). Down-projecting
`[planet_h || h_actor]` back to d would discard exactly the global-context
capacity the extra token was added to provide.

**Action factorization.** Per source planet, the actor emits one masked
categorical over `[noop, target_0, ..., target_P]` with pg-style softcapped
logits, plus a unimodal Beta fraction distribution conditional on launching.

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

try:
    from torch.nn.attention.flex_attention import create_block_mask, flex_attention
except Exception:  # pragma: no cover - optional on older torch builds
    create_block_mask = None
    flex_attention = None

from .config import OrbitPolicyConfig, normalize_attention_config
from .features import EncodedObs

_PLANET_XY_SCALE: float = 100.0
_PLANET_XY_OFFSET: float = 50.0

# Hypersphere-normalization epsilon. nGPT's `justnorm` divides by the raw L2
# norm with no floor (model.py:103-106), which is safe for a dense LM where
# every token is a real embedding. Our set-transformer carries *padded* token
# slots (planet_mask / fleet_mask) whose pre-norm vectors can be ~zero, so we
# clamp the norm to avoid a 0/0 → NaN that would otherwise sit in padded rows
# and poison any global op that touches them. Valid tokens are far from this
# floor, so the projection is unaffected for them.
_JUSTNORM_EPS: float = 1e-6


def justnorm(x: torch.Tensor, dim: int = -1, eps: float = _JUSTNORM_EPS) -> torch.Tensor:
    """Project `x` onto the unit hypersphere along `dim` (nGPT `justnorm`).

    nGPT does the reduction in fp32 (`ngpt/model.py:103`) regardless of the
    activation dtype, because the residual stream's geometry is the whole point
    of the method and bf16 norm accumulation is too coarse. We mirror that and
    cast back to the input dtype, with the padded-slot eps floor described
    above.
    """
    dtype = x.dtype
    x = x.float()
    return (x / x.norm(p=2, dim=dim, keepdim=True).clamp_min(eps)).to(dtype)


def eigen_residual(
    h: torch.Tensor, update: torch.Tensor, alpha: torch.Tensor
) -> torch.Tensor:
    """nGPT hypersphere residual step (`ngpt/model.py:148-153`).

    Both the running stream `h` and the sublayer output `update` are projected
    to the sphere, then `h` is moved a per-channel *eigen learning rate* `alpha`
    of the way toward `update` along the chord and re-projected:

        h ← justnorm( norm(h) + alpha · (norm(update) − norm(h)) )

    This replaces the additive pre-norm residual (`h + scale · sublayer`). At
    small `alpha` the step is a near-geodesic interpolation toward the sublayer
    direction, which is what keeps every block's output on the unit sphere.
    `alpha` is the already-rescaled effective LR (see `_EigenAlpha`).
    """
    a = justnorm(h)
    b = justnorm(update)
    return justnorm(a + alpha * (b - a))


class _EigenAlpha(nn.Module):
    """Per-channel eigen learning rate with nGPT's stored/effective split.

    nGPT stores `alpha` at scale `base_scale = 1/√dim` so AdamW sees a unit-ish
    gradient regime, then rescales to the *effective* value
    `|alpha · (init_value / base_scale)|` in the forward (`ngpt/model.py:88,
    145-146`). The `abs` keeps the LR non-negative. We keep the same trick so
    these params live in the scalar/control optimizer group at a sane scale.
    """

    def __init__(self, dim: int, init_value: float, base_scale: float):
        super().__init__()
        self.init_value = float(init_value)
        self.base_scale = float(base_scale)
        self.alpha = nn.Parameter(torch.full((dim,), float(base_scale)))

    def forward(self) -> torch.Tensor:
        return (self.alpha * (self.init_value / self.base_scale)).abs()


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
# nGPT hypersphere controls (`alpha` eigen LRs, `sqk` QK scale, `suv` MLP
# scale) are all 1D, so the `ndim < 2` rule already keeps them fp32; they are
# listed here for intent. `fleet_latents` is the only 2D control tensor that
# needs the explicit name match. `q_gain`/`target_q_gain` remain for the target
# readout's attention temperature, which is not part of the hypersphere trunk.
_FP32_NAME_SUBSTRINGS: tuple[str, ...] = (
    "alpha",
    "sqk",
    "suv",
    "q_gain",
    "target_q_gain",
    "actor_token",
    "critic_token",
    "global_token",
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


def normalize_matrices(model: nn.Module) -> None:
    """Re-project every hypersphere-trunk matrix onto the unit sphere.

    nGPT keeps its weight matrices on the hypersphere by renormalizing them
    once at init and after every optimizer step (`ngpt/train.py:410-411,
    499-500`). Each `SelfAttention` / `CrossAttention` / `TransformerBlock`
    owns the row/column axis convention for its own matrices via
    `normalize_weights`; we walk the module tree and invoke them. Task heads
    (the target-attention readout, the Beta-fraction and distributional value
    heads, the input feature embeddings) are deliberately NOT normalized — the
    hypersphere is a property of the encoder trunk, not the readouts.

    Operates in-place on the fp32 master weights (`CastedLinear` keeps weights
    fp32 after `restore_fp32_params`), so it composes with the bf16-compute
    path. Cheap: a handful of small `norm` reductions, no matmuls. Call it
    after each `optimizer.step()` in the training loop.
    """
    for module in model.modules():
        normalize = getattr(module, "normalize_weights", None)
        if callable(normalize):
            normalize()


@torch.no_grad()
def ngpt_control_stats(model: nn.Module) -> dict[str, float]:
    """Effective magnitudes of the nGPT control scalars, for TensorBoard.

    These are the parameters that set the model's function-space sensitivity
    to a given trunk rotation: the eigen LRs (`alpha`, expected to *grow* over
    training — nGPT's design — which on a fixed optimizer lr means growing
    KL per PPO update), the QK scales (`sqk`, attention sharpness), the MLP
    pre-activation scale (`suv`), and the target-readout temperature.

    Units: `eigen_alpha_*` and `sqk_*` are *effective* values (post
    stored/effective rescale — init is `eigen_alpha_init` for alphas,
    `qk_gain_init` / 1.0 for sqk_q / sqk_k). `suv_mean` is the stored
    parameter (init 1.0); the forward's `×√dim` is a fixed constant, so
    drift reads identically either way.
    """
    eigen: list[torch.Tensor] = []
    sqk_q: list[torch.Tensor] = []
    sqk_k: list[torch.Tensor] = []
    suv: list[torch.Tensor] = []
    for module in model.modules():
        if isinstance(module, _EigenAlpha):
            eigen.append(module().flatten())
        elif isinstance(module, (SelfAttention, CrossAttention, DestinationFleetCrossAttention)):
            inv = 1.0 / module.base_scale
            sqk_q.append((module.sqk_q * inv).flatten())
            sqk_k.append((module.sqk_k * inv).flatten())
        elif isinstance(module, TransformerBlock):
            suv.append(module.suv.flatten())
    stats: dict[str, float] = {}
    values: list[torch.Tensor] = []
    names: list[str] = []
    if eigen:
        eig = torch.cat(eigen).float()
        names.extend(("eigen_alpha_mean", "eigen_alpha_max"))
        values.extend((eig.mean(), eig.max()))
    if sqk_q:
        names.append("sqk_q_eff_mean")
        values.append(torch.cat(sqk_q).float().mean())
    if sqk_k:
        names.append("sqk_k_eff_mean")
        values.append(torch.cat(sqk_k).float().mean())
    if suv:
        names.append("suv_mean")
        values.append(torch.cat(suv).float().mean())
    q_gain = getattr(model, "target_q_gain", None)
    if isinstance(q_gain, torch.Tensor):
        names.append("target_q_gain")
        values.append(q_gain.float().mean())
    if values:
        host_values = torch.stack(values).detach().cpu().tolist()
        stats.update(zip(names, (float(v) for v in host_values), strict=True))
    return stats


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
    """Multi-head self-attention with nGPT hypersphere QK normalization.

    nGPT pattern (`ngpt/model.py:128-136`):
      1. Project x → Q, K, V.
      2. Apply RoPE to Q/K (here: only the physical planet-token slice).
      3. **Unit-normalize Q and K per head** onto the hypersphere
         (`justnorm`), then scale per channel by the learnable `sqk`. Pins
         ‖q_h‖,‖k_h‖ to a learned magnitude so logit scale doesn't drift as
         the QKV matrices (kept on the sphere by `normalize_matrices`) move.
      4. SDPA with `softmax_scale = √head_dim` (nGPT inverts the usual
         `1/√head_dim`: with unit-norm q,k the raw dot is in [−1,1], so the
         softmax needs scaling *up*, not down).
      5. Output projection — normalized like every trunk matrix; cold-start
         smallness comes from the eigen-LR `alpha`, not a zero-init here.
    """

    def __init__(
        self,
        dim: int,
        n_heads: int,
        *,
        n_kv_heads: int | None = None,
        qk_gain_init: float = 1.0,
    ):
        super().__init__()
        n_kv_heads, head_dim, kv_dim = normalize_attention_config(
            dim, n_heads, n_kv_heads
        )
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.base_scale = dim**-0.5
        # Three separate Q/K/V projections (nGPT keeps them split).
        self.c_q = CastedLinear(dim, dim, bias=False)
        self.c_k = CastedLinear(dim, kv_dim, bias=False)
        self.c_v = CastedLinear(dim, kv_dim, bias=False)
        self.out_proj = CastedLinear(dim, dim, bias=False)
        # Per-channel QK scale on the unit-normed Q/K (nGPT `sqk`, stored at
        # `base_scale` so the effective value is `stored / base_scale`). The
        # query scale carries the sharp-attention gain: effective `sqk_q` =
        # `qk_gain_init` (1.0 = nGPT-soft, 5.0 = old `q_gain`), applied to Q
        # only — exactly like the old block — so logits scale by the gain, not
        # its square. K stays at effective 1.0. GQA gives Q and K different
        # head counts, so they carry separate scales.
        self.sqk_q = nn.Parameter(
            torch.full((dim,), float(self.base_scale * qk_gain_init))
        )
        self.sqk_k = nn.Parameter(torch.full((kv_dim,), float(self.base_scale)))
        # Init is largely vacuous: `normalize_matrices` projects every row/col
        # of these matrices onto the unit sphere at build and after each
        # optimizer step, so only the direction (not the gain) survives.
        nn.init.orthogonal_(self.c_q.weight)
        nn.init.orthogonal_(self.c_k.weight)
        nn.init.orthogonal_(self.c_v.weight)
        nn.init.orthogonal_(self.out_proj.weight)

    def normalize_weights(self) -> None:
        """Project the four projection matrices back onto the hypersphere.

        Q/K/V read the unit-norm stream, so each output row's input-weight
        vector is normed along the input dim (`dim=1`, nGPT
        `train.py:402-404`). The output projection writes the stream, so each
        input column is normed along the output dim (`dim=0`,
        `train.py:405`).
        """
        with torch.no_grad():
            self.c_q.weight.copy_(justnorm(self.c_q.weight, dim=1))
            self.c_k.weight.copy_(justnorm(self.c_k.weight, dim=1))
            self.c_v.weight.copy_(justnorm(self.c_v.weight, dim=1))
            self.out_proj.weight.copy_(justnorm(self.out_proj.weight, dim=0))

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
        # real tokens, already on the unit hypersphere.
        q = self.c_q(x).unflatten(-1, (self.n_heads, self.head_dim))
        k = self.c_k(x).unflatten(-1, (self.n_kv_heads, self.head_dim))
        v = self.c_v(x).unflatten(-1, (self.n_kv_heads, self.head_dim))
        # RoPE first (nGPT order): it's a per-pair rotation, so it preserves
        # the per-head norm that `justnorm` then pins.
        if rope is not None and rope_cache is not None and rope_slice is not None:
            q_planet, k_planet = rope(q[:, rope_slice], k[:, rope_slice], rope_cache)
            q = _splice_rope(q, q_planet, rope_slice, rope.rotate_dim)
            k = _splice_rope(k, k_planet, rope_slice, rope.rotate_dim)
        # Hypersphere QK: unit-norm per head, then per-channel `sqk` scale.
        sqk_q = (self.sqk_q * (1.0 / self.base_scale)).to(q.dtype).view(
            1, 1, self.n_heads, self.head_dim
        )
        sqk_k = (self.sqk_k * (1.0 / self.base_scale)).to(k.dtype).view(
            1, 1, self.n_kv_heads, self.head_dim
        )
        q = sqk_q * justnorm(q)
        k = sqk_k * justnorm(k)
        # SDPA expects [B, H, j, head_dim]. Caller is responsible for bf16
        # autocast on CUDA — that's what keeps the FlashAttention-2 /
        # mem-efficient backend in play (the key-padding mask routes to the
        # mem-efficient kernel; flash proper only fires for unmasked rows).
        # No inner autocast/`sdpa_kernel` context — that breaks AOT autograd
        # under `torch.compile`. On CPU SDPA falls to the math kernel (tests).
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
            scale=self.head_dim**0.5,
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
        *,
        n_kv_heads: int | None = None,
        qk_gain_init: float = 1.0,
    ):
        super().__init__()
        n_kv_heads, head_dim, kv_dim = normalize_attention_config(
            dim, n_heads, n_kv_heads
        )
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.base_scale = dim**-0.5
        self.c_q = CastedLinear(dim, dim, bias=False)
        self.c_k = CastedLinear(dim, kv_dim, bias=False)
        self.c_v = CastedLinear(dim, kv_dim, bias=False)
        self.out_proj = CastedLinear(dim, dim, bias=False)
        # Sharp-attention gain on the query scale only (see SelfAttention).
        self.sqk_q = nn.Parameter(
            torch.full((dim,), float(self.base_scale * qk_gain_init))
        )
        self.sqk_k = nn.Parameter(torch.full((kv_dim,), float(self.base_scale)))

        nn.init.orthogonal_(self.c_q.weight)
        nn.init.orthogonal_(self.c_k.weight)
        nn.init.orthogonal_(self.c_v.weight)
        nn.init.orthogonal_(self.out_proj.weight)

    def normalize_weights(self) -> None:
        """Project the projection matrices onto the hypersphere (see
        `SelfAttention.normalize_weights`)."""
        with torch.no_grad():
            self.c_q.weight.copy_(justnorm(self.c_q.weight, dim=1))
            self.c_k.weight.copy_(justnorm(self.c_k.weight, dim=1))
            self.c_v.weight.copy_(justnorm(self.c_v.weight, dim=1))
            self.out_proj.weight.copy_(justnorm(self.out_proj.weight, dim=0))

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
        # Hypersphere QK: unit-norm per head, then per-channel `sqk` scale.
        sqk_q = (self.sqk_q * (1.0 / self.base_scale)).to(q.dtype).view(
            1, 1, self.n_heads, self.head_dim
        )
        sqk_k = (self.sqk_k * (1.0 / self.base_scale)).to(k.dtype).view(
            1, 1, self.n_kv_heads, self.head_dim
        )
        q = sqk_q * justnorm(q)
        k = sqk_k * justnorm(k)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        attn_mask = safe_mask[:, None, None, :]
        o = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            scale=self.head_dim**0.5,
            enable_gqa=self.n_kv_heads != self.n_heads,
        )
        o = o.transpose(1, 2).flatten(-2)
        return self.out_proj(o)


class DestinationFleetCrossAttention(nn.Module):
    """Planet-query cross-attention over fleets scoped by exact destination.

    Query rows are planets and key/value rows are all current fleet tokens. The
    destination mask is arbitrary at element granularity, so the CUDA path uses
    FlexAttention. There is no per-planet fleet cap: batching remains `[B,P,F]`
    and `dest_idx == p` decides membership.
    """

    def __init__(
        self,
        dim: int,
        n_heads: int,
        *,
        n_kv_heads: int | None = None,
        qk_gain_init: float = 1.0,
    ):
        super().__init__()
        n_kv_heads, head_dim, kv_dim = normalize_attention_config(
            dim, n_heads, n_kv_heads
        )
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.base_scale = dim**-0.5
        self.c_q = CastedLinear(dim, dim, bias=False)
        self.c_k = CastedLinear(dim, kv_dim, bias=False)
        self.c_v = CastedLinear(dim, kv_dim, bias=False)
        self.out_proj = CastedLinear(dim, dim, bias=False)
        self.sqk_q = nn.Parameter(
            torch.full((dim,), float(self.base_scale * qk_gain_init))
        )
        self.sqk_k = nn.Parameter(torch.full((kv_dim,), float(self.base_scale)))

        nn.init.orthogonal_(self.c_q.weight)
        nn.init.orthogonal_(self.c_k.weight)
        nn.init.orthogonal_(self.c_v.weight)
        nn.init.orthogonal_(self.out_proj.weight)

    def normalize_weights(self) -> None:
        with torch.no_grad():
            self.c_q.weight.copy_(justnorm(self.c_q.weight, dim=1))
            self.c_k.weight.copy_(justnorm(self.c_k.weight, dim=1))
            self.c_v.weight.copy_(justnorm(self.c_v.weight, dim=1))
            self.out_proj.weight.copy_(justnorm(self.out_proj.weight, dim=0))

    def forward(
        self,
        planets: torch.Tensor,
        fleets: torch.Tensor,
        planet_mask: torch.Tensor,
        fleet_mask: torch.Tensor,
        fleet_target_planet_idx: torch.Tensor,
    ) -> torch.Tensor:
        b, p, _ = planets.shape
        f = fleets.shape[1]
        if f == 0:
            return torch.zeros_like(planets)

        valid_dest = (
            fleet_mask
            & (fleet_target_planet_idx >= 0)
            & (fleet_target_planet_idx < p)
        )
        dest_idx = torch.where(
            valid_dest,
            fleet_target_planet_idx,
            torch.full_like(fleet_target_planet_idx, -1),
        )

        q = self.c_q(planets).unflatten(-1, (self.n_heads, self.head_dim))
        k = self.c_k(fleets).unflatten(-1, (self.n_kv_heads, self.head_dim))
        v = self.c_v(fleets).unflatten(-1, (self.n_kv_heads, self.head_dim))
        sqk_q = (self.sqk_q * (1.0 / self.base_scale)).to(q.dtype).view(
            1, 1, self.n_heads, self.head_dim
        )
        sqk_k = (self.sqk_k * (1.0 / self.base_scale)).to(k.dtype).view(
            1, 1, self.n_kv_heads, self.head_dim
        )
        q = sqk_q * justnorm(q)
        k = sqk_k * justnorm(k)
        v = v.masked_fill(~fleet_mask.unsqueeze(-1).unsqueeze(-1), 0.0)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        if q.is_cuda and flex_attention is not None and create_block_mask is not None:
            def mask_mod(batch, _head, q_idx, kv_idx):  # type: ignore[no-untyped-def]
                return planet_mask[batch, q_idx] & (dest_idx[batch, kv_idx] == q_idx)

            block_mask = create_block_mask(
                mask_mod,
                b,
                None,
                p,
                f,
                device=q.device,
                BLOCK_SIZE=(64, 64),
            )
            o = flex_attention(
                q,
                k,
                v,
                block_mask=block_mask,
                scale=self.head_dim**0.5,
                enable_gqa=self.n_kv_heads != self.n_heads,
            )
        else:
            arange_p = torch.arange(p, device=planets.device)
            attn_mask = (
                planet_mask[:, :, None]
                & (dest_idx[:, None, :] == arange_p[None, :, None])
            )
            o = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=attn_mask[:, None],
                scale=self.head_dim**0.5,
                enable_gqa=self.n_kv_heads != self.n_heads,
            )
        o = o.transpose(1, 2).flatten(-2)
        return self.out_proj(o)


class DestinationFleetConditioner(nn.Module):
    """Apply exact-destination fleet conditioning to planet tokens.

    The destination-conditioned backend has a hard contract: callers must pass
    `fleet_target_planet_idx` with the same shape as `fleet_mask`. Missing or
    mismatched sidecars are integration bugs, not a no-op fallback.
    """

    def __init__(
        self,
        dim: int,
        n_heads: int,
        *,
        n_kv_heads: int | None = None,
        qk_gain_init: float = 1.0,
    ) -> None:
        super().__init__()
        self.cross_attn = DestinationFleetCrossAttention(
            dim,
            n_heads,
            n_kv_heads=n_kv_heads,
            qk_gain_init=qk_gain_init,
        )
        self.mod = CastedLinear(dim, 2 * dim)
        nn.init.zeros_(self.mod.weight)
        nn.init.zeros_(self.mod.bias)

    def forward(
        self,
        planets: torch.Tensor,
        fleets: torch.Tensor,
        planet_mask: torch.Tensor,
        fleet_mask: torch.Tensor,
        fleet_target_planet_idx: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if fleet_target_planet_idx is None:
            raise ValueError(
                "destination_conditioned encoder requires fleet_target_planet_idx"
            )
        fleet_target_planet_idx = fleet_target_planet_idx.to(
            device=fleet_mask.device,
            dtype=torch.long,
        )
        if tuple(fleet_target_planet_idx.shape) != tuple(fleet_mask.shape):
            raise ValueError(
                "fleet_target_planet_idx shape must match fleet_mask shape; "
                f"got {tuple(fleet_target_planet_idx.shape)} vs {tuple(fleet_mask.shape)}"
            )

        b, p, _ = planets.shape
        h_p = justnorm(planets)
        h_f = justnorm(fleets).masked_fill(~fleet_mask.unsqueeze(-1), 0.0)
        fleet_ctx = self.cross_attn(
            h_p,
            h_f,
            planet_mask,
            fleet_mask,
            fleet_target_planet_idx,
        )
        valid_dest = (
            fleet_mask
            & (fleet_target_planet_idx >= 0)
            & (fleet_target_planet_idx < p)
        )
        has_inbound_i = torch.zeros(
            b,
            p,
            dtype=torch.int8,
            device=fleet_target_planet_idx.device,
        )
        has_inbound_i.scatter_reduce_(
            1,
            fleet_target_planet_idx.masked_fill(~valid_dest, 0),
            valid_dest.to(torch.int8),
            reduce="amax",
            include_self=True,
        )
        gamma, beta = self.mod(fleet_ctx).chunk(2, dim=-1)
        conditioned = justnorm(h_p * (1.0 + gamma) + beta)
        h_p = torch.where(has_inbound_i.bool().unsqueeze(-1), conditioned, h_p)
        h_f = h_p.new_zeros(b, 0, h_p.shape[-1])
        fleet_mask = torch.zeros(b, 0, dtype=torch.bool, device=fleet_mask.device)
        return h_p, h_f, fleet_mask


class TransformerBlock(nn.Module):
    """nGPT normalized-transformer block (`ngpt/model.py:108-179`).

    The residual stream lives on the unit hypersphere. There is no pre-norm
    RMSNorm: each sublayer reads the already-unit-norm stream directly, and the
    eigen-LR `eigen_residual` step both injects the sublayer output and
    re-projects to the sphere — so the additive `attn_scale`/`ff_scale`/
    `resid_mix` levers and the zero-init cold-start identity are gone. The FF
    is nGPT's SwiGLU (`u · silu(v)`, `ngpt/model.py:158-164`), with nGPT's
    per-channel `suv` scale restoring O(1) pre-activation magnitude after the
    unit-norm input is read by the normalized `c_fc` rows.
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
        eigen_alpha_init: float = 0.05,
        qk_gain_init: float = 1.0,
        block_skip: bool = False,
    ):
        super().__init__()
        self.base_scale = dim**-0.5
        self.attn = SelfAttention(
            dim, n_heads, n_kv_heads=n_kv_heads, qk_gain_init=qk_gain_init
        )
        # nGPT SwiGLU FF (`ngpt/model.py:158-164`): one `c_fc` projects to
        # 2·ff_dim, split into gate `u` and value `v`; activation is
        # `u · silu(v)`; `mlp_proj` reads the ff_dim-wide gated result.
        self.c_fc = CastedLinear(dim, 2 * ff_dim, bias=False)
        self.mlp_proj = CastedLinear(ff_dim, dim, bias=False)
        self.drop = nn.Dropout(dropout)
        # Init is vacuous post-`normalize_weights` (rows/cols are re-projected
        # onto the sphere); orthogonal just gives a well-conditioned direction.
        nn.init.orthogonal_(self.c_fc.weight)
        nn.init.orthogonal_(self.mlp_proj.weight)
        # Per-channel pre-activation scale (nGPT `suv`, init 1.0 via the ×√dim
        # split), applied to the full 2·ff_dim `uv` before the gate split. The
        # √dim factor undoes the 1/√dim shrink from a unit-norm input hitting a
        # unit-norm `c_fc` row, so the SwiGLU pre-activation is O(1).
        self.suv = nn.Parameter(torch.ones(2 * ff_dim))
        self.suv_scale = float(dim**0.5)
        # Eigen learning rates — per-channel geodesic step size for each
        # sublayer's hypersphere residual update (nGPT init 0.05; 0.5 gives the
        # full-strength residual add, see OrbitPolicyConfig).
        self.attn_alpha = _EigenAlpha(
            dim, init_value=eigen_alpha_init, base_scale=self.base_scale
        )
        self.mlp_alpha = _EigenAlpha(
            dim, init_value=eigen_alpha_init, base_scale=self.base_scale
        )
        # Optional learnable per-channel U-net skip toward the block-stack input
        # `x0` (the on-sphere analog of parameter-golf `resid_mix`). Applied at
        # block input as a third eigen step `justnorm(x̂ + β(x̂₀−x̂))`. Plain
        # zero-init (not the `_EigenAlpha` stored/effective split, which is
        # degenerate at init 0) ⇒ identity at start, signed β learnable from
        # layer 1 on (at layer 0 `x == x0` so the step is an exact no-op).
        self.block_skip = bool(block_skip)
        if self.block_skip:
            self.skip_alpha = nn.Parameter(torch.zeros(dim))

    def normalize_weights(self) -> None:
        """Project the FF matrices onto the hypersphere. `c_fc` reads the
        unit-norm stream (norm along input `dim=1`); `mlp_proj` writes it (norm
        along output `dim=0`). The attention matrices normalize themselves via
        `self.attn.normalize_weights()`."""
        with torch.no_grad():
            self.c_fc.weight.copy_(justnorm(self.c_fc.weight, dim=1))
            self.mlp_proj.weight.copy_(justnorm(self.mlp_proj.weight, dim=0))

    def forward(
        self,
        x: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
        *,
        x0: torch.Tensor | None = None,
        rope: Rotary2D | None = None,
        rope_cache: Rotary2DCache | None = None,
        rope_slice: slice | None = None,
    ) -> torch.Tensor:
        # `x` is a dense padded [B, T, D] on the unit hypersphere.
        # Optional U-net skip toward the block-stack input before the sublayers.
        if self.block_skip and x0 is not None:
            x = eigen_residual(x, x0, self.skip_alpha.to(x.dtype))
        a = self.attn(
            x,
            valid_mask,
            rope=rope,
            rope_cache=rope_cache,
            rope_slice=rope_slice,
        )
        if self.drop.p:
            a = self.drop(a)
        x = eigen_residual(x, a, self.attn_alpha().to(x.dtype))
        suv = (self.suv * self.suv_scale).to(x.dtype)
        uv = suv * self.c_fc(x)
        u, v = uv.chunk(2, dim=-1)
        ff = self.mlp_proj(u * F.silu(v))
        if self.drop.p:
            ff = self.drop(ff)
        x = eigen_residual(x, ff, self.mlp_alpha().to(x.dtype))
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
        eigen_alpha_init: float = 0.05,
        qk_gain_init: float = 1.0,
        block_skip: bool = False,
    ):
        super().__init__()
        self.base_scale = dim**-0.5
        self.cross_attn = CrossAttention(
            dim, n_heads, n_kv_heads=n_kv_heads, qk_gain_init=qk_gain_init
        )
        # Eigen-LR for the cross-attention hypersphere residual (nGPT init 0.05;
        # 0.5 gives the full-strength residual add).
        self.cross_alpha = _EigenAlpha(
            dim, init_value=eigen_alpha_init, base_scale=self.base_scale
        )
        self.self_block = TransformerBlock(
            dim,
            ff_dim,
            n_heads,
            dropout,
            n_kv_heads=n_kv_heads,
            layer_idx=layer_idx,
            eigen_alpha_init=eigen_alpha_init,
            qk_gain_init=qk_gain_init,
            block_skip=block_skip,
        )

    def forward(
        self,
        latents: torch.Tensor,
        fleets: torch.Tensor,
        fleet_mask: torch.Tensor,
        *,
        latents0: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Latents and fleets are already on the hypersphere; cross-attention
        # justnorms Q/K internally, then the eigen-LR step injects the result
        # and re-projects the latents onto the sphere. `latents0` is the
        # tokenizer's block-stack input, fed to the self-block's U-net skip.
        cross = self.cross_attn(latents, fleets, fleet_mask)
        latents = eigen_residual(latents, cross, self.cross_alpha().to(latents.dtype))
        return self.self_block(latents, x0=latents0)


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
        eigen_alpha_init: float = 0.05,
        qk_gain_init: float = 1.0,
        block_skip: bool = False,
    ):
        super().__init__()
        if num_latents < 1:
            raise ValueError("num_fleet_latents must be >= 1")
        if depth < 1:
            raise ValueError("fleet_tokenizer_depth must be >= 1")
        self.num_latents = num_latents
        self.fleet_latents = nn.Parameter(torch.zeros(num_latents, dim))
        nn.init.trunc_normal_(self.fleet_latents, std=0.02)
        self.layers = nn.ModuleList(
            [
                FleetLatentBlock(
                    dim,
                    ff_dim,
                    n_heads,
                    dropout,
                    n_kv_heads=n_kv_heads,
                    layer_idx=i,
                    eigen_alpha_init=eigen_alpha_init,
                    qk_gain_init=qk_gain_init,
                    block_skip=block_skip,
                )
                for i in range(depth)
            ]
        )

    def forward(self, fleets: torch.Tensor, fleet_mask: torch.Tensor) -> torch.Tensor:
        b = fleets.shape[0]
        # Put fleet tokens and latents on the hypersphere before the Perceiver
        # blocks; padded fleet slots are zeroed (justnorm's eps floor keeps the
        # projection finite for near-zero pad vectors first).
        fleets = justnorm(fleets)
        fleets = fleets.masked_fill(~fleet_mask.unsqueeze(-1), 0.0)
        latents = justnorm(self.fleet_latents).view(1, self.num_latents, -1).expand(
            b, -1, -1
        )
        # Block-stack input for the per-block U-net skip (no-op unless block_skip).
        latents0 = latents
        for layer in self.layers:
            latents = layer(latents, fleets, fleet_mask, latents0=latents0)
        return justnorm(latents)


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
      - `bins_to_scalar(logits)` recovers the expected raw scalar over bin
        centers. With symlog support this is E[symexp(c_i)], matching the
        CleanRL v149 Bellman scalar path, not symexp(E[c_i]).
      - `loss(logits, target_probs)` is just F.cross_entropy on a per-element
        basis (caller is responsible for masking/reduction).

    With `symlog=True`, `min_value` / `max_value` are raw values. The library
    transforms those endpoints to symlog space for the histogram support and
    applies symlog when encoding scalar targets. Scalar decode maps each
    symlog-space bin center back to raw space before taking the expectation.
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
        probs = logits.float().softmax(dim=-1)
        centers = self.encoder.centers.float()
        if self.symlog:
            centers = _symexp(centers)
        return (probs * centers).sum(dim=-1).clamp(self.min_value, self.max_value)


def _symlog(x: torch.Tensor) -> torch.Tensor:
    return x.sign() * torch.log1p(x.abs())


def _symexp(x: torch.Tensor) -> torch.Tensor:
    return x.sign() * torch.expm1(x.abs())

@dataclass
class PolicyOutput:
    launch_logits: torch.Tensor       # [B, P] noop column logits for PPO / legacy launch logits for SAC adapters
    target_logits: torch.Tensor       # [B, P, P] masked target categorical logits
    value: torch.Tensor               # [B] — scalar value E[V] recovered from value_logits
    value_logits: torch.Tensor        # [B,H,num_bins] — distributional critic logits; H=critic_mtp_horizon
    planet_owned_mask: torch.Tensor   # [B, P] bool
    planet_mask: torch.Tensor         # [B, P] bool
    planet_ids: torch.Tensor          # [B, P] long
    action_logit_softcap: float | None = None
    launch_log_std: torch.Tensor | None = None  # [B, P] state-dependent Normal log std
    launch_prob_floor: float = 0.0
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
        if cfg.encoder_backend not in {"dense", "fleet_latent", "destination_conditioned"}:
            raise ValueError(f"unknown encoder_backend: {cfg.encoder_backend!r}")
        self.cfg = cfg
        self.global_embed = CastedLinear(cfg.global_features, cfg.dim)
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
                eigen_alpha_init=cfg.eigen_alpha_init,
                qk_gain_init=cfg.qk_gain_init,
                block_skip=cfg.block_skip,
            )
            if cfg.encoder_backend == "fleet_latent"
            else None
        )
        self.destination_fleet_conditioner = (
            DestinationFleetConditioner(
                cfg.dim,
                cfg.n_heads,
                n_kv_heads=cfg.n_kv_heads,
                qk_gain_init=cfg.qk_gain_init,
            )
            if cfg.encoder_backend == "destination_conditioned"
            else None
        )
        # Learnable prefix tokens (PMA-style). Initialized small so
        # they don't dominate the encoder at step 0 — gradient flow alone
        # will scale them up as the heads start using their output.
        # Store prefix tokens flat. AOTAutograd can reduce broadcasted
        # `[1, 1, d]` parameters to `[d]` gradients in compiled backward at
        # larger PPO batch sizes; making the parameter itself `[d]` keeps the
        # expected gradient shape aligned with the reduction.
        self.actor_token = nn.Parameter(torch.zeros(cfg.dim))
        self.critic_token = nn.Parameter(torch.zeros(cfg.dim))
        self.global_token = nn.Parameter(torch.zeros(cfg.dim))
        nn.init.trunc_normal_(self.actor_token, std=0.02)
        nn.init.trunc_normal_(self.critic_token, std=0.02)
        nn.init.trunc_normal_(self.global_token, std=0.02)
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
                    eigen_alpha_init=cfg.eigen_alpha_init,
                    qk_gain_init=cfg.qk_gain_init,
                    block_skip=cfg.block_skip,
                )
                for i in range(cfg.depth)
            ]
        )
        # `target_query` consumes [planet_h || h_actor] → 2·dim. `target_key`
        # stays at `dim` because adding the same h_actor to every key shifts
        # all `q·k` row-wise by a constant and cancels in the softmax — it
        # can't change the relative ranking of targets. Putting it on the
        # query side only is what gives the actor token bite.
        self.target_query = CastedLinear(2 * cfg.dim, cfg.dim, bias=False)
        self.target_key = CastedLinear(cfg.dim, cfg.dim, bias=False)
        self.target_noop_key = nn.Parameter(torch.zeros(cfg.dim))
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
        nn.init.trunc_normal_(self.target_noop_key, std=0.02)
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
            sigma_to_bin_ratio=cfg.value_sigma_to_bin_ratio,
            symlog=cfg.value_symlog,
        )
        self.value_head = nn.Sequential(
            CastedLinear(cfg.dim, cfg.value_hidden, bias=False),
            SquaredReLU(),
            CastedLinear(
                cfg.value_hidden,
                cfg.critic_mtp_horizon * cfg.value_num_bins,
                bias=False,
            ),
        )
        # Orthogonal init for the value-head input projection (both dims
        # ≥64 if `value_hidden ≥ 64`).
        if (
            self.value_head[0].weight.shape[0] >= 64
            and self.value_head[0].weight.shape[1] >= 64
        ):
            nn.init.orthogonal_(self.value_head[0].weight, gain=0.1)
        # Zero-init the bias-free value head. With symmetric supports, zero
        # logits are a neutral scalar prior, matching CleanRL v149's
        # nocriticbias critic.
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
        # Put the trunk matrices on the hypersphere at init (nGPT
        # `train.py:410-411`); the training loop re-normalizes after each step.
        normalize_matrices(self)

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
            fleet_target_planet_idx = (
                None
                if feats.fleet_target_planet_idx is None
                else feats.fleet_target_planet_idx.unsqueeze(0)
            )
            if feats.global_feats is None:
                global_feats = None
            elif feats.global_feats.dim() == 1:
                global_feats = feats.global_feats.unsqueeze(0)
            else:
                global_feats = feats.global_feats
        else:
            planet_feats = feats.planet_feats
            planet_mask = feats.planet_mask
            fleet_feats = feats.fleet_feats
            fleet_mask = feats.fleet_mask
            fleet_target_planet_idx = feats.fleet_target_planet_idx
            global_feats = feats.global_feats

        b, p, _ = planet_feats.shape
        f = fleet_feats.shape[1]
        if fleet_target_planet_idx is not None:
            fleet_target_planet_idx = fleet_target_planet_idx.to(
                device=fleet_mask.device,
                dtype=torch.long,
            )
        if global_feats is None:
            global_feats = planet_feats.new_zeros(b, self.global_embed.in_features)
        elif global_feats.dim() == 1:
            global_feats = global_feats.unsqueeze(0)
        global_feats = global_feats.to(device=planet_feats.device, dtype=planet_feats.dtype)
        if global_feats.shape[-1] != self.global_embed.in_features:
            global_feats = _match_feature_width(
                global_feats,
                self.global_embed.in_features,
            )
        if planet_feats.shape[-1] != self.planet_embed.in_features:
            planet_feats = _match_feature_width(
                planet_feats, self.planet_embed.in_features
            )
        if fleet_feats.shape[-1] != self.fleet_embed.in_features:
            fleet_feats = _match_feature_width(fleet_feats, self.fleet_embed.in_features)

        h_g = self.global_embed(global_feats)
        h_p = self.planet_embed(planet_feats)
        h_f = self.fleet_embed(fleet_feats)
        if self.destination_fleet_conditioner is not None:
            h_p, h_f, fleet_mask = self.destination_fleet_conditioner(
                h_p,
                h_f,
                planet_mask,
                fleet_mask,
                fleet_target_planet_idx,
            )
            f = 0
        elif self.fleet_tokenizer is not None:
            h_f = self.fleet_tokenizer(h_f, fleet_mask)
            fleet_mask = torch.ones(
                b,
                h_f.shape[1],
                dtype=torch.bool,
                device=fleet_mask.device,
            )
            f = h_f.shape[1]
        # Prepend the three prefix tokens, broadcast to batch dim. Parameters
        # are stored flat so compiled backward's broadcast reduction returns
        # `[d]`, matching the actual parameter shape.
        actor_t = self.actor_token.view(1, 1, -1).expand(b, 1, -1)
        critic_t = self.critic_token.view(1, 1, -1).expand(b, 1, -1)
        global_t = self.global_token.view(1, 1, -1).expand(b, 1, -1)
        global_t = global_t + h_g.unsqueeze(1)
        h = torch.cat([actor_t, critic_t, global_t, h_p, h_f], dim=1)
        # Project the residual-stream entry point onto the unit hypersphere
        # (nGPT). Runs on padded `[B, T, D]` — padded slots are normed too
        # (eps-safe) but masks keep them inert in attention.
        h = justnorm(h)
        summary_mask = torch.ones(b, 3, dtype=torch.bool, device=planet_mask.device)
        full_mask = torch.cat([summary_mask, planet_mask, fleet_mask], dim=1)
        # Planet features store centered normalized coordinates:
        # ((x - 50) / 100, (y - 50) / 100). RoPE uses physical board
        # coordinates so the phase scale is meaningful on the 100x100 map.
        planet_xy = planet_feats[..., :2] * _PLANET_XY_SCALE + _PLANET_XY_OFFSET
        rope_cache = self.planet_rope.cache(planet_xy, h.dtype)
        planet_slice = slice(3, 3 + p)
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
        planet_h = h[:, 3 : 3 + p]          # [B, P, d]
        fleet_h = h[:, 3 + p : 3 + p + f]   # [B, F, d]
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
        # Block-stack input for the per-block U-net skip (no-op unless block_skip).
        x0 = h
        for layer in self.layers:
            h = layer(
                h,
                full_mask,
                x0=x0,
                rope=self.planet_rope,
                rope_cache=rope_cache,
                rope_slice=planet_slice,
            )
        # Each block already ends on the sphere; a final justnorm makes the
        # head inputs explicitly unit-norm regardless of accumulated drift.
        h = justnorm(h)
        return self._split_encoded(h, planet_mask, fleet_mask, p, f)

    def encode(
        self, feats: EncodedObs
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run the transformer over [actor, critic, global, planets..., fleets...].

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

    def forward(
        self,
        feats: EncodedObs,
        *,
        include_value: bool = True,
        include_actor: bool = True,
    ) -> PolicyOutput:
        planet_h, _fleet_h, h_actor, h_critic, _token_mask = self.encode(feats)
        b, p, d = planet_h.shape

        # Promote scalar mask/garrison/id to batch dim if not already.
        def _b(t: torch.Tensor) -> torch.Tensor:
            return t.unsqueeze(0) if t.dim() == 1 else t

        planet_owned = _b(feats.planet_owned_mask)
        planet_mask = _b(feats.planet_mask)
        planet_ids = _b(feats.planet_ids)

        if include_actor:
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
            noop_k = F.rms_norm(
                self.target_noop_key.to(dtype=k.dtype, device=k.device),
                (k.size(-1),),
            )
            noop_logits = torch.einsum("bid,d->bi", q, noop_k) / (d**0.5)
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
            target_logits = logits  # [B, P, P]

            # Fraction head: native-support unimodal Beta, matching the CleanRL
            # IterThink v24 / Dreamer4 beta path.
            #
            # RMS-norm the input first. The nGPT encoder exits on the unit L2
            # sphere (per-element RMS ≈ 1/√d), but these single-linear heads
            # (gain=0.01) were tuned for the old RMSNorm exit's per-element-RMS-1
            # scale. Without restoring it the raw logit collapses by ≈√d ≈ 11×,
            # pinning the Beta at α=β=1+softplus(0)≈1.69 (a dead, state-independent
            # fraction) with an ≈11×-attenuated gradient it can't escape. The
            # target categorical above is immune because it RMS-norms q/k post
            # -projection; the value head recovers via its 2-layer MLP; this
            # single-linear head cannot, so it needs the scale restored here.
            # Mechanically this is a constant ×√d on the already-unit-L2 input
            # (no learnable weight, no per-token-varying divisor for valid
            # tokens) — it is NOT nGPT's `sz` (a learnable per-output-channel
            # scale applied *after* the logits). We don't need `sz` here: these
            # heads are excluded from `normalize_matrices`, so their free weight
            # rows can absorb any output gain directly; rms_norm only restores
            # the head-input scale the cold-start init was tuned for.
            ctx_n = F.rms_norm(planet_with_ctx, (planet_with_ctx.size(-1),))
            fraction_alpha = 1.0 + F.softplus(
                self.fraction_alpha_head(ctx_n).squeeze(-1).float()
            )
            fraction_beta = 1.0 + F.softplus(
                self.fraction_beta_head(ctx_n).squeeze(-1).float()
            )
        else:
            noop_logits = h_actor.new_empty((b, p), dtype=torch.float32)
            target_logits = h_actor.new_empty((b, p, 0), dtype=torch.float32)
            fraction_alpha = None
            fraction_beta = None

        if include_value:
            # Value: distributional head over the dedicated critic token.
            # Horizon 0 is V(s_t); later horizons are critic-only MTP targets.
            # RMS-norm the unit-L2 critic token back to the per-element-RMS-1
            # scale the head's gain was tuned for (see the fraction head above).
            # The MLP critic recovers without this, but restoring the scale
            # makes its cold-start well-conditioned rather than ~11× under-driven.
            value_logits = self.value_head(F.rms_norm(h_critic, (h_critic.size(-1),))).view(
                b,
                int(self.cfg.critic_mtp_horizon),
                int(self.cfg.value_num_bins),
            )
            value = self.value_encoder.bins_to_scalar(value_logits[:, 0])
        else:
            value = h_critic.new_empty((b,), dtype=torch.float32)
            value_logits = h_critic.new_empty(
                (b, int(self.cfg.critic_mtp_horizon), 0),
                dtype=torch.float32,
            )

        return PolicyOutput(
            launch_logits=noop_logits,
            target_logits=target_logits,
            value=value,
            value_logits=value_logits,
            planet_owned_mask=planet_owned,
            planet_mask=planet_mask,
            planet_ids=planet_ids,
            action_logit_softcap=float(self.cfg.action_logit_softcap),
            fraction_alpha=fraction_alpha,
            fraction_beta=fraction_beta,
        )
