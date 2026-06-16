"""SAC actor + twin soft-Q networks for Orbit Wars.

Architecture
------------
Three independent set-transformer networks (actor, qf1, qf2) plus EMA target
copies of qf1/qf2. cleanrl convention: no shared encoder between actor and
critics.

The set encoder is a stripped variant of `OrbitPolicy.encode`:
  - one learnable summary token ([SUMMARY] only — actor uses it as global
    context concatenated onto each planet rep; critic uses it for the dueling
    state-value head and as global context for the launch-advantage attention)
  - reuses the same transformer block, fleet-latent tokenizer, 2D RoPE, and
    embed/final RMSNorms
  - the critic encoder is **state-only** (no action conditioning at the input):
    the action enters *after* the encoder via the factored advantage heads, so
    per-planet discrete options can be enumerated cheaply without re-encoding.

Hybrid factored action (mirrors PPO `OrbitPolicy`)
--------------------------------------------------
Per owned source planet i, the actor emits the SAME factored action PPO uses:
  - launch_i:   Bernoulli(σ(launch_logit_i))
  - target_i:   masked Categorical over P planet slots (legal target support)
  - fraction_i: tanh-squashed Normal on (0, 1) — the only continuous dim

The launch *angle* is NOT learned; it is solved analytically by
`sampling._lead_solution` from the chosen target (see `sac_sampling.py`).

Factored (dueling) critic + closed-form discrete SAC
----------------------------------------------------
`SACSoftQ` is **factored**: `Q(s, a) = V(s) + Σ_i A_i(s, a_i)` over owned source
planets, with each planet's per-option advantage enumerable:
  - `A0_i(s)`        — the "no-launch" advantage (scalar per planet)
  - `AL_i[t](s, f_i)` — the "launch to target t with fraction f_i" advantage,
    a K-basis fraction-conditioned target attention (cubic in f_i so an interior
    optimal fraction exists; a monotone AL would collapse f→0/1).
This makes the soft-V expectation `Σ_a π(a)·Q(s,a)` tractable in closed form
(no REINFORCE): the discrete policy gradient is exact and baseline-free, the
fraction uses the pathwise/reparam gradient. The split is in REAL ship-margin
units: `Q(s,a) = V(s) + adv(s,a)`. `V(s) = bins_to_scalar(nV)` is decoded from a
**distributional** HL-Gauss two-hot head over symlog bins (clamped to
`[value_min, value_max]`, which structurally bounds the bootstrap — the
scalar+symlog-MSE critic it replaces had a loss whose gradient vanished at large
|Q|, letting the deadly-triad bootstrap run away). The per-planet advantages are
scalar and tanh-bounded to ±`adv_scale`, so `adv = Σ_i A_i` is bounded and linear
in the policy probs (the closed-form expectation survives). The tanh bound — not
a logit-space tilt — is what anchors the dueling V/A split.

`get_action` returns the differentiable closed-form quantities the trainer
needs: `launch_p`, `target_probs`, the reparam `fraction` + its per-planet
`logp_frac_per`, and the closed-form entropies `H_disc` / `H_cont`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812

from .config import OrbitPolicyConfig
from .features import MAX_PLANETS, EncodedObs
from .model import (
    CastedLinear,
    DestinationFleetConditioner,
    FleetLatentTokenizer,
    HLGaussLoss,
    Rotary2D,
    SquaredReLU,
    TransformerBlock,
    justnorm,
    normalize_matrices,
)

_PLANET_XY_SCALE: float = 100.0
_PLANET_XY_OFFSET: float = 50.0
FRACTION_LOG_STD_INIT: float = 0.0

# Numerical floors mirrored from `sampling.py` so train-time log-probs match
# the rollout-time squashed-Normal log-probs the buffer was filled with.
SQUASH_EPS: float = 1e-6

# Number of fraction-basis functions for the launch-advantage head. φ(f) =
# [1, f, f², f³] gives a cubic dependence on the fraction so the advantage has
# interior optima (a monotone advantage would push the optimal fraction to a
# boundary, reproducing the fraction-collapse failure).
_FRAC_BASIS_K: int = 4


def _match_feature_width(x: torch.Tensor, expected: int) -> torch.Tensor:
    actual = x.shape[-1]
    if actual > expected:
        return x[..., :expected]
    return F.pad(x, (0, expected - actual))


def _fraction_basis(fraction: torch.Tensor) -> torch.Tensor:
    """[1, f, f², f³] stacked on a trailing dim → [..., K]."""
    f = fraction
    return torch.stack([torch.ones_like(f), f, f * f, f * f * f], dim=-1)


def _squashed_normal_log_prob(
    mean: torch.Tensor,
    log_std: torch.Tensor,
    fraction: torch.Tensor,
) -> torch.Tensor:
    """SAC log-prob for fraction = 0.5·(tanh(z) + 1).

    `|df/dz| = 0.5·(1 − u²)` with `u = tanh(z) = 2f − 1`, so the true normalized
    log-density is `logN(z) − log(1 − u²) − log(0.5)`. We keep the `+log2`
    (`= −log(0.5)`) half-range term — unlike the PPO ratio use where it cancels —
    because SAC tunes the entropy *value* (α target), not just log-prob ratios,
    so the constant matters. Carries grad through `fraction` (no atanh detach).
    """
    u = (2.0 * fraction.float() - 1.0).clamp(-1.0 + SQUASH_EPS, 1.0 - SQUASH_EPS)
    z = torch.atanh(u)
    log_std = log_std.float()
    inv_std = torch.exp(-log_std)
    log_prob_z = (
        -0.5 * ((z - mean.float()) * inv_std).square()
        - log_std
        - 0.5 * math.log(2.0 * math.pi)
    )
    squash_correction = torch.log(1.0 - u.square() + SQUASH_EPS)
    return log_prob_z - squash_correction + math.log(2.0)


class SACEncoder(nn.Module):
    """Set-transformer encoder shared in shape between SAC actor and critic.

    Two outputs per forward:
      - per-planet token reps `[B, P, d]`
      - one summary-token rep `[B, d]`

    Both actor and critic use the state-only form (no action conditioning at the
    input). `extra_planet_dim` is retained for flexibility but unused by the
    current critic (the action enters via the factored advantage heads instead).
    """

    def __init__(self, cfg: OrbitPolicyConfig, *, extra_planet_dim: int = 0):
        super().__init__()
        if cfg.encoder_backend not in {"dense", "fleet_latent", "destination_conditioned"}:
            raise ValueError(f"unknown encoder_backend: {cfg.encoder_backend!r}")
        self.cfg = cfg
        self.extra_planet_dim = int(extra_planet_dim)
        self.planet_embed = CastedLinear(
            cfg.planet_features + self.extra_planet_dim, cfg.dim
        )
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
        self.summary_token = nn.Parameter(torch.zeros(cfg.dim))
        nn.init.trunc_normal_(self.summary_token, std=0.02)
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
        # Global time FiLM. The reward is a production-margin LEVEL, so the
        # value-to-go = Σ_{k≥t} γ^{k−t}·prod_margin shrinks as the remaining
        # horizon shortens — a dependence on the game clock that is NOT in the
        # per-planet/fleet tokens. Without it the critic sees identical
        # encodings for boards at t=50 vs t=450 and can only fit the
        # step-averaged value. We condition the post-encoder reps on a scalar
        # time feature ∈ [0,1] via FiLM (γ,β per channel). Last layer is
        # zero-init so γ=β=0 at start ⇒ identity (no perturbation to the
        # carefully-tuned head inits); the optimizer grows it as the endgame
        # signal demands. A multiplicative γ(t) is exactly the right form to
        # represent value ≈ horizon_factor(t) · production_margin(s).
        self.time_film = nn.Sequential(
            CastedLinear(1, cfg.dim, bias=True),
            SquaredReLU(),
            CastedLinear(cfg.dim, 2 * cfg.dim, bias=True),
        )
        nn.init.zeros_(self.time_film[-1].weight)
        nn.init.zeros_(self.time_film[-1].bias)
        # Put the encoder trunk matrices on the hypersphere at init (nGPT);
        # the SAC trainer re-normalizes after each optimizer step.
        normalize_matrices(self)

    def _embed(
        self,
        feats: EncodedObs,
        action_feats: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, slice, int, int, object]:
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
            planet_inbound_feats = (
                None
                if feats.planet_inbound_feats is None
                else feats.planet_inbound_feats.unsqueeze(0)
            )
        else:
            planet_feats = feats.planet_feats
            planet_mask = feats.planet_mask
            fleet_feats = feats.fleet_feats
            fleet_mask = feats.fleet_mask
            fleet_target_planet_idx = feats.fleet_target_planet_idx
            planet_inbound_feats = feats.planet_inbound_feats

        b, p, _ = planet_feats.shape
        f = fleet_feats.shape[1]
        if fleet_target_planet_idx is not None:
            fleet_target_planet_idx = fleet_target_planet_idx.to(
                device=fleet_mask.device,
                dtype=torch.long,
            )

        if planet_feats.shape[-1] != self.cfg.planet_features:
            planet_feats = _match_feature_width(planet_feats, self.cfg.planet_features)
        if fleet_feats.shape[-1] != self.cfg.fleet_features:
            fleet_feats = _match_feature_width(fleet_feats, self.cfg.fleet_features)

        if self.extra_planet_dim:
            if action_feats is None:
                raise RuntimeError(
                    "SACEncoder built with extra_planet_dim>0 requires action_feats"
                )
            if action_feats.dim() == 2:
                action_feats = action_feats.unsqueeze(0)
            planet_feats = torch.cat(
                [planet_feats, action_feats.to(planet_feats.dtype)], dim=-1
            )

        h_p = self.planet_embed(planet_feats)
        if self.destination_fleet_conditioner is not None:
            has_planet_inbound_feats = (
                planet_inbound_feats is not None
                and planet_inbound_feats.shape[-2] > 0
            )
            if not has_planet_inbound_feats:
                h_f = self.fleet_embed(fleet_feats)
            else:
                h_f = h_p.new_zeros(b, 0, h_p.shape[-1])
            h_p, h_f, fleet_mask = self.destination_fleet_conditioner(
                h_p,
                planet_mask,
                fleet_mask,
                fleet_target_planet_idx,
                fleet_feats,
                planet_inbound_feats,
                h_f if not has_planet_inbound_feats else None,
            )
        if self.fleet_tokenizer is not None:
            if self.destination_fleet_conditioner is None:
                h_f = self.fleet_embed(fleet_feats)
            h_f = self.fleet_tokenizer(h_f, fleet_mask)
            fleet_mask = torch.ones(
                b, h_f.shape[1], dtype=torch.bool, device=fleet_mask.device
            )
        elif self.destination_fleet_conditioner is None:
            h_f = self.fleet_embed(fleet_feats)
        f = h_f.shape[1]

        summary_t = self.summary_token.view(1, 1, -1).expand(b, 1, -1)
        h = torch.cat([summary_t, h_p, h_f], dim=1)
        # Project the residual-stream entry point onto the unit hypersphere
        # (nGPT). Padded slots are normed too (eps-safe) but masked in attention.
        h = justnorm(h)
        summary_mask = torch.ones(b, 1, dtype=torch.bool, device=planet_mask.device)
        full_mask = torch.cat([summary_mask, planet_mask, fleet_mask], dim=1)

        planet_xy = planet_feats[..., :2] * _PLANET_XY_SCALE + _PLANET_XY_OFFSET
        rope_cache = self.planet_rope.cache(planet_xy, h.dtype)
        planet_slice = slice(1, 1 + p)
        return h, full_mask, planet_mask, fleet_mask, planet_slice, p, f, rope_cache

    def forward(
        self,
        feats: EncodedObs,
        *,
        action_feats: torch.Tensor | None = None,
        time_feat: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        h, full_mask, planet_mask, fleet_mask, planet_slice, p, _f, rope_cache = (
            self._embed(feats, action_feats)
        )
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
        # Each block already ends on the sphere; final justnorm makes the head
        # inputs explicitly unit-norm.
        h = justnorm(h)
        summary_h = h[:, 0]
        planet_h = h[:, 1 : 1 + p]
        if time_feat is not None:
            # FiLM-condition the post-encoder reps on the game clock (identity
            # at init via the zero-init head). γ,β are per-channel; broadcast γ
            # over the planet axis. `x·(1+γ)+β` so γ=0 ⇒ no change.
            tf = time_feat.reshape(-1, 1).to(dtype=summary_h.dtype, device=summary_h.device)
            gamma, beta = self.time_film(tf).chunk(2, dim=-1)  # each [B, d]
            summary_h = summary_h * (1.0 + gamma) + beta
            planet_h = planet_h * (1.0 + gamma).unsqueeze(1) + beta.unsqueeze(1)
        return planet_h, summary_h


@dataclass
class SACAction:
    """One forward pass of `SACActor.get_action`.

    Stored in the replay buffer (samples, non-owned slots zeroed): `launch`,
    `target_idx`, `fraction`, `legal_mask`. The fraction is the reparam sample
    gated by *ownership only* (NOT by the sampled launch), because the
    closed-form critic evaluates the launch-advantage at this fraction weighted
    by the launch *probability*.

    The remaining fields are the differentiable closed-form quantities the
    trainer's factored soft-value needs:
      - `launch_p` [B,P], `target_probs` [B,P,P]: the discrete policy (grad)
      - `logp_frac_per` [B,P]: per-planet squashed-Normal log-prob of `fraction`
      - `H_disc` [B]: Σ_i g_i·[H_bern(p_i) + p_i·H_cat(π_t)]  (joint disc entropy)
      - `H_cont` [B]: Σ_i g_i·p_i·(−logp_frac_i)  (p-weighted continuous entropy)
      - `cont_logp_sum` [B]: Σ_i g_i·p_i·logp_frac_i  (= −H_cont; for α_cont tuning)
      - `eff_cont_dim` [B] = Σ_i g_i·p_i (expected #launches; scales the α_cont
        target so it matches the p-weighting of `cont_logp_sum`)
      - `n_owned` [B], `h_disc_max` [B]: per-state α-target bookkeeping, where
        `h_disc_max = Σ_i g_i·log(n_legal_i+1)` is the max attainable discrete
        entropy (uniform over the n_legal+1 options) summed over owned sources.
    """

    launch: torch.Tensor            # [B, P] float 0/1 Bernoulli sample (owned-gated)
    target_idx: torch.Tensor        # [B, P] long in [0, P)
    fraction: torch.Tensor          # [B, P] reparam sample in (eps, 1-eps), owned-gated
    launch_p: torch.Tensor          # [B, P] σ(launch_logit), impossible→~0
    target_probs: torch.Tensor      # [B, P, P] softmax over legal support
    fraction_mean: torch.Tensor     # [B, P] pre-squash Normal mean
    fraction_log_std: torch.Tensor  # [B, P] pre-squash Normal log std
    logp_frac_per: torch.Tensor     # [B, P] squashed-Normal log-prob of fraction
    H_disc: torch.Tensor            # [B] joint discrete entropy (summed)
    H_cont: torch.Tensor            # [B] p-weighted continuous entropy (summed)
    cont_logp_sum: torch.Tensor     # [B] Σ g·p·logp_frac (= −H_cont)
    owned_mask: torch.Tensor        # [B, P] bool (owned & alive source gate)
    legal_mask: torch.Tensor        # [B, P, P] bool legal-target support
    n_owned: torch.Tensor           # [B] float Σ_i g_i
    eff_cont_dim: torch.Tensor      # [B] float Σ_i g_i·p_i (expected #launches)
    h_disc_max: torch.Tensor        # [B] float Σ_i g_i·log(n_legal_i+1)


class SACActor(nn.Module):
    """Hybrid factored actor: Bernoulli launch + Categorical target + squashed
    Normal fraction, mirroring `OrbitPolicy.forward`'s action heads.

    Heads (per planet, all consuming `[planet_h || summary_h]` = 2d):
      - `launch_head`: 2d → 1   — Bernoulli logit, bias −1.5 (cold-start no-op)
      - `target_query`/`target_key`/`target_q_gain`: target attention → [B,P,P]
      - `fraction_head`: 2d → 1 — pre-squash Normal mean; `fraction_log_std` is
        a single learned scalar expanded over planets (cleanrl idiom)
    """

    def __init__(
        self,
        cfg: OrbitPolicyConfig,
        *,
        log_std_min: float = -5.0,
        log_std_max: float = 2.0,
    ):
        super().__init__()
        if log_std_min >= log_std_max:
            raise ValueError("log_std_min must be < log_std_max")
        self.cfg = cfg
        self.log_std_min = float(log_std_min)
        self.log_std_max = float(log_std_max)
        self.encoder = SACEncoder(cfg)
        # Target attention (mirror OrbitPolicy): query carries the global actor
        # context (2d→d), key stays plain (d→d) — adding context to keys shifts
        # every q·k row-wise by a constant and cancels in the softmax.
        self.target_query = CastedLinear(2 * cfg.dim, cfg.dim, bias=False)
        self.target_key = CastedLinear(cfg.dim, cfg.dim, bias=False)
        self.target_q_gain = nn.Parameter(torch.tensor(1.0))
        nn.init.orthogonal_(self.target_query.weight, gain=0.05)
        nn.init.orthogonal_(self.target_key.weight, gain=0.05)
        # Per-planet launch Bernoulli; bias negative so cold-start prefers
        # not launching until advantage says otherwise.
        self.launch_head = CastedLinear(2 * cfg.dim, 1)
        nn.init.zeros_(self.launch_head.weight)
        nn.init.constant_(self.launch_head.bias, -1.5)
        # Fraction: pre-squash Normal mean + one direct learned log std scalar
        # expanded over planets, matching the cleanrl squashed-Gaussian idiom.
        self.fraction_head = CastedLinear(2 * cfg.dim, 1)
        self.fraction_log_std = nn.Parameter(torch.tensor(FRACTION_LOG_STD_INIT))
        nn.init.orthogonal_(self.fraction_head.weight, gain=0.01)
        nn.init.zeros_(self.fraction_head.bias)
        # Cached self-target mask; sliced per forward (allocate once, not every
        # step). P is bounded by MAX_PLANETS.
        self.register_buffer(
            "_self_target_mask",
            torch.eye(MAX_PLANETS, dtype=torch.bool),
            persistent=False,
        )

    def _heads(
        self, feats: EncodedObs, *, time_feat: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
        """Return `(launch_logits, target_logits, fraction_mean,
        fraction_log_std, p)` before any legality masking is applied.

        Self-targets are masked (diagonal) and the launch logit on
        no-legal-self-target-count rows is suppressed, mirroring
        `OrbitPolicy.forward`. External callers apply the lead-intercept legal
        mask via `sampling`-style helpers.
        """
        planet_h, summary_h = self.encoder(feats, time_feat=time_feat)
        b, p, d = planet_h.shape
        actor_ctx = summary_h.unsqueeze(1).expand(-1, p, -1)
        planet_with_ctx = torch.cat([planet_h, actor_ctx], dim=-1)  # [B, P, 2d]

        def _b(t: torch.Tensor) -> torch.Tensor:
            return t.unsqueeze(0) if t.dim() == 1 else t

        planet_mask = _b(feats.planet_mask)

        q = self.target_query(planet_with_ctx)
        k = self.target_key(planet_h)
        q = F.rms_norm(q, (q.size(-1),))
        k = F.rms_norm(k, (k.size(-1),))
        q = q * self.target_q_gain.to(q.dtype)
        logits = torch.einsum("bid,bjd->bij", q, k) / (d**0.5)
        logits = logits.masked_fill(~planet_mask.unsqueeze(1), float("-inf"))
        logits = logits.masked_fill(
            self._self_target_mask[:p, :p].unsqueeze(0), float("-inf")
        )

        valid_target_count = (planet_mask.sum(dim=-1, keepdim=True) - 1).clamp_min(0)
        launch_logits = self.launch_head(planet_with_ctx).squeeze(-1).float()
        launch_logits = launch_logits.masked_fill(valid_target_count <= 0, -20.0)

        fraction_mean = self.fraction_head(planet_with_ctx).squeeze(-1).float()
        log_std_raw = self.fraction_log_std.float()
        # tanh-remap into [log_std_min, log_std_max] (SpinUp / cleanrl idiom).
        fraction_log_std = self.log_std_min + 0.5 * (
            self.log_std_max - self.log_std_min
        ) * (torch.tanh(log_std_raw) + 1.0)
        fraction_log_std = fraction_log_std.expand_as(fraction_mean)
        return launch_logits, logits.float(), fraction_mean, fraction_log_std, p

    def _owned_mask(self, feats: EncodedObs) -> torch.Tensor:
        owned = feats.planet_owned_mask
        pmask = feats.planet_mask
        if owned.dim() == 1:
            owned = owned.unsqueeze(0)
            pmask = pmask.unsqueeze(0)
        return owned & pmask

    def get_action(
        self,
        feats: EncodedObs,
        legal_mask: torch.Tensor,
        *,
        deterministic: bool = False,
        time_feat: torch.Tensor | None = None,
    ) -> SACAction:
        """Sample the factored action under the provided legal-target mask and
        compute the closed-form differentiable quantities the trainer needs.

        `legal_mask` is `[B, P, P]` (or `[P, P]` for a single env) — the
        lead-intercept-feasible + route-clear target support. Sources with no
        legal target are forced to no-op (launch=0) and contribute zero
        log-prob / entropy.
        """
        launch_logits, target_logits, fraction_mean, fraction_log_std, p = self._heads(
            feats, time_feat=time_feat
        )
        # Finite guard: a divergent update can NaN outputs and tanh(NaN) flows
        # into the C++ env (which aborts with no Python traceback). Sanitize so
        # a stray non-finite value degrades to a safe action.
        launch_logits = torch.nan_to_num(launch_logits)
        fraction_mean = torch.nan_to_num(fraction_mean)
        fraction_log_std = torch.nan_to_num(fraction_log_std)

        if legal_mask.dim() == 2:
            legal_mask = legal_mask.unsqueeze(0)
        legal_mask = legal_mask.to(device=target_logits.device, dtype=torch.bool)

        gate = self._owned_mask(feats)  # [B, P] bool
        gate_f = gate.float()

        # Apply legality: restrict target support to legal slots on source rows,
        # leave non-source rows untouched (they're gated out below anyway).
        source = gate.unsqueeze(-1)
        masked_targets = target_logits.masked_fill(~legal_mask, float("-inf"))
        target_logits = torch.where(source, masked_targets, target_logits)
        # Force no-op on sources with no legal target.
        has_legal = legal_mask.any(dim=-1)  # [B, P]
        impossible = gate & ~has_legal
        launch_logits = launch_logits.masked_fill(impossible, -20.0)

        safe_target_logits = _safe_target_logits(target_logits)
        target_log_probs = F.log_softmax(safe_target_logits, dim=-1)  # [B, P, P]
        target_probs = target_log_probs.exp()

        # ---- sample (for env execution + buffer) ----
        launch_p = torch.sigmoid(launch_logits)  # [B, P] (grad-carrying)
        if deterministic:
            launch = (launch_logits > 0.0).float()
            target_idx = safe_target_logits.argmax(dim=-1)
            z = fraction_mean
        else:
            launch = torch.bernoulli(launch_p.detach())
            uniform = torch.rand_like(safe_target_logits).clamp_(
                SQUASH_EPS, 1.0 - SQUASH_EPS
            )
            gumbel = -torch.log(-torch.log(uniform))
            target_idx = (safe_target_logits + gumbel).argmax(dim=-1)
            eps = torch.randn_like(fraction_mean)
            z = fraction_mean + fraction_log_std.exp() * eps  # reparam (grad)
        launch = launch * gate_f  # zero non-source launches
        fraction = (0.5 * (torch.tanh(z) + 1.0)).clamp(SQUASH_EPS, 1.0 - SQUASH_EPS)

        # ---- per-planet squashed-Normal log-prob of the reparam fraction ----
        # Pathwise through `fraction` (carries grad in the stochastic branch).
        logp_frac_per = _squashed_normal_log_prob(
            fraction_mean, fraction_log_std, fraction
        )  # [B, P]

        # ---- closed-form discrete entropy (joint = summed over owned sources)
        # H_launch_i = -[p·log p + (1-p)·log(1-p)]; 0 for forced-no-op sources.
        p_clamp = launch_p.clamp(SQUASH_EPS, 1.0 - SQUASH_EPS)
        h_launch = -(p_clamp * p_clamp.log() + (1.0 - p_clamp) * (1.0 - p_clamp).log())
        h_launch = h_launch.masked_fill(impossible, 0.0)
        # H_target_i = -Σ_j π_t(j)logπ_t(j); masked slots have logπ=-inf, so
        # replace with finite zeros first (0·log0=0 convention, clean backward).
        safe_log_probs = torch.where(
            torch.isfinite(target_log_probs),
            target_log_probs,
            torch.zeros_like(target_log_probs),
        )
        h_target = -(target_probs * safe_log_probs).sum(-1)  # [B, P]
        h_target = torch.where(has_legal, h_target, torch.zeros_like(h_target))
        h_disc = (gate_f * (h_launch + launch_p * h_target)).sum(-1)  # [B]

        # ---- p-weighted continuous entropy (fraction only emitted on launch) ----
        cont_logp_sum = (gate_f * launch_p * logp_frac_per).sum(-1)  # [B] = -H_cont
        h_cont = -cont_logp_sum  # [B]

        # ---- α-target bookkeeping ----
        n_owned = gate_f.sum(-1)  # [B]
        # Expected number of launches Σ g·p. cont_logp_sum is p-weighted (a
        # fraction is only emitted on launch), so the α_cont target must scale by
        # this — NOT by n_owned — else a low-launch policy drives α_cont → 0
        # regardless of the actual fraction entropy.
        eff_cont_dim = (gate_f * launch_p).sum(-1)  # [B]
        n_legal_count = legal_mask.sum(-1).float()  # [B, P] number legal per src
        # Per-source max attainable discrete entropy. The launch Bernoulli ×
        # masked categorical (target only when launch=1) is a distribution over
        # the (n_legal + 1) joint options {noop} ∪ {launch→t}; its maximum is the
        # uniform `log(n_legal + 1)`, attained at p* = n_legal/(n_legal+1) with a
        # uniform target. 0 when no legal target.
        per_src_max = torch.where(
            has_legal,
            torch.log(n_legal_count.clamp_min(1.0) + 1.0),
            torch.zeros_like(n_legal_count),
        )
        h_disc_max = (gate_f * per_src_max).sum(-1)  # [B]

        # Zero stored action on non-owned planets so buffer/executed agree.
        launch = launch * gate_f
        fraction = fraction * gate_f

        return SACAction(
            launch=launch,
            target_idx=target_idx,
            fraction=fraction,
            launch_p=launch_p,
            target_probs=target_probs,
            fraction_mean=fraction_mean,
            fraction_log_std=fraction_log_std,
            logp_frac_per=logp_frac_per,
            H_disc=h_disc,
            H_cont=h_cont,
            cont_logp_sum=cont_logp_sum,
            owned_mask=gate,
            legal_mask=legal_mask,
            n_owned=n_owned,
            eff_cont_dim=eff_cont_dim,
            h_disc_max=h_disc_max,
        )


def _safe_target_logits(target_logits: torch.Tensor) -> torch.Tensor:
    """Make target categorical rows finite even when no legal target exists.

    A row of all `-inf` (no legal target) becomes all-zeros so `log_softmax` is
    finite. Such rows are gated out of every sum, so the substituted uniform
    distribution never leaks into the objective.
    """
    finite = torch.isfinite(target_logits).any(dim=-1, keepdim=True)
    return torch.where(finite, target_logits, torch.zeros_like(target_logits))


@dataclass
class QComponents:
    """Per-twin factored critic outputs for the distributional dueling critic.

    `nV` is the dueling state-value DISTRIBUTION as bin logits; `nA0` the
    per-planet no-launch advantage (scalar, tanh-bounded); `nAL` the
    per-(source, target) launch advantage (scalar, tanh-bounded) at the supplied
    fraction. `taken_advantage`/`expected_advantage` reduce them to the real-units
    scalar `adv(s,a) = Σ_i g_i[…]`, linear in the policy probs (that linearity is
    the closed-form `E_a[adv]`). The real-units Q is `bins_to_scalar(nV) + adv`:
    the decoded V is structurally bounded to `[value_min, value_max]` and adv to
    ±n_owned·adv_scale, so the bootstrap cannot diverge. The `n` prefix is
    historical (not normalized).
    """

    nV: torch.Tensor    # [B, num_bins]  value-distribution logits
    nA0: torch.Tensor   # [B, P]         scalar no-launch advantage
    nAL: torch.Tensor   # [B, P, P]      scalar launch advantage (at the fraction)


class SACSoftQ(nn.Module):
    """Factored (dueling) distributional soft-Q: `Q(s, a) = V(s) + Σ_i A_i(s, a_i)`.

    State-only set-transformer encoder (no action conditioning at the input) →
    per-planet reps + summary. The dueling state value is DISTRIBUTIONAL — an
    HL-Gauss head emits logits over `value_num_bins` symlog-spaced bins (Dreamer
    two-hot encoding); the advantages are scalar and tanh-bounded:
      - `value_head(summary)` → nV  [B, num_bins]  (value-distribution logits)
      - `noop_head(planet_h)` → nA0 [B, P]   (scalar no-launch adv, ±adv_scale)
      - K-basis fraction target attention → nAL [B, P, P] (launch adv, ±adv_scale)

    `taken_advantage`/`expected_advantage` reduce them to the real-units scalar
    `adv(s, a) = Σ_i g_i[(1-l_i)nA0_i + l_i nAL_i[t_i]]`, linear in the policy
    probs (the closed-form `E_a[adv]`, no REINFORCE). The real-units Q is
    `V(s) + adv` with `V(s) = bins_to_scalar(nV)`: V learns the full TD target via
    HL-Gauss CE, adv learns the residual `(y − V)` (V detached) via a Huber loss,
    and the tanh bound on the per-planet heads anchors the V/A split (no
    centering). Decoded V is clamped to `[value_min, value_max]` and adv to
    ±n_owned·adv_scale, so the bootstrap cannot diverge (this replaces the old
    symlog-MSE scalar Q, whose loss vanished at large |Q| and let the deadly triad
    run away).

    The launch advantage is `nAL_i[t] = Σ_k S_k(i, t)·φ_k(f_i)` with
    `φ(f) = [1, f, f², f³]` (cubic in the fraction so an interior optimal
    fraction exists). `S_k` are K independent attention score maps (query from
    `[planet_h‖summary]`, key from `planet_h`).
    """

    def __init__(self, cfg: OrbitPolicyConfig):
        super().__init__()
        self.cfg = cfg
        self.encoder = SACEncoder(cfg)
        d = cfg.dim
        self.k_basis = _FRAC_BASIS_K
        # Real-units bound on each per-planet advantage head (tanh saturation).
        self.adv_scale = float(cfg.adv_scale)

        # Distributional value support (symlog two-hot bins over the raw range).
        self.hlgauss = HLGaussLoss(
            min_value=cfg.value_min,
            max_value=cfg.value_max,
            num_bins=cfg.value_num_bins,
            symlog=cfg.value_symlog,
        )

        # Dueling state-value DISTRIBUTION (logits over the bins).
        self.value_head = nn.Sequential(
            CastedLinear(d, d, bias=False),
            SquaredReLU(),
            CastedLinear(d, cfg.value_num_bins, bias=False),
        )
        nn.init.orthogonal_(self.value_head[0].weight, gain=0.1)
        # Zero-init the logit layer ⇒ cold-start uniform dist ⇒ value ≈ 0.
        nn.init.zeros_(self.value_head[-1].weight)

        # No-launch advantage (per planet, scalar).
        self.noop_head = nn.Sequential(
            CastedLinear(d, d, bias=False),
            SquaredReLU(),
            CastedLinear(d, 1, bias=False),
        )
        nn.init.orthogonal_(self.noop_head[0].weight, gain=0.1)
        nn.init.zeros_(self.noop_head[-1].weight)

        # K-basis launch-advantage attention. Query carries summary context
        # (2d→K·d), key is plain (d→K·d). Zero-init so cold-start nAL ≈ 0.
        self.adv_query = CastedLinear(2 * d, self.k_basis * d, bias=False)
        self.adv_key = CastedLinear(d, self.k_basis * d, bias=False)
        nn.init.zeros_(self.adv_query.weight)
        nn.init.orthogonal_(self.adv_key.weight, gain=0.05)

    def encode(
        self, feats: EncodedObs, *, time_feat: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """State-only encode → (planet_h [B,P,d], summary_h [B,d])."""
        return self.encoder(feats, time_feat=time_feat)

    def value(self, summary_h: torch.Tensor) -> torch.Tensor:
        """Dueling state-value distribution logits nV [B, num_bins]."""
        return self.value_head(summary_h).float()

    def noop_adv(self, planet_h: torch.Tensor) -> torch.Tensor:
        """Scalar no-launch advantage nA0 [B, P], tanh-bounded to ±adv_scale
        (real ship-margin units). Bounding the PER-PLANET head (not the summed
        advantage) preserves the linear-in-probs closed form, so the summed
        advantage stays exactly enumerable while confined to ±n_owned·adv_scale.
        """
        a = self.noop_head(planet_h).squeeze(-1).float()
        return self.adv_scale * torch.tanh(a / self.adv_scale)

    def bins_to_scalar(self, logits: torch.Tensor) -> torch.Tensor:
        """Decode value-distribution logits → REAL value, clamped to the support
        `[value_min, value_max]` (this clamp is what bounds Q)."""
        return self.hlgauss.bins_to_scalar(logits)

    def launch_adv(
        self,
        planet_h: torch.Tensor,
        summary_h: torch.Tensor,
        fraction: torch.Tensor,
    ) -> torch.Tensor:
        """Scalar launch advantage nAL [B, P, P] at per-source `fraction` [B, P].

        `nAL[b, i, t] = Σ_k S_k(i, t)·φ_k(f_i)` with cubic fraction basis. The
        gradient w.r.t. `fraction` flows through φ (the pathwise/reparam path).
        """
        b, pl, d = planet_h.shape
        ctx = summary_h.unsqueeze(1).expand(-1, pl, -1)
        q = self.adv_query(torch.cat([planet_h, ctx], dim=-1))  # [B,P,K*d]
        k = self.adv_key(planet_h)  # [B,P,K*d]
        q = q.view(b, pl, self.k_basis, d)
        k = k.view(b, pl, self.k_basis, d)
        q = F.rms_norm(q, (d,))
        k = F.rms_norm(k, (d,))
        # S[b,i,t,k] = (q[b,i,k]·k[b,t,k]) / sqrt(d)
        scores = torch.einsum("bikd,btkd->bitk", q, k) / (d**0.5)  # [B,P,P,K]
        phi = _fraction_basis(fraction)  # [B,P,K]
        nAL = torch.einsum("bitk,bik->bit", scores, phi).float()  # [B,P,P]
        # Same tanh bound as noop_adv: per-(source,target) head confined to
        # ±adv_scale, so the launch advantage shares the no-launch head's units.
        return self.adv_scale * torch.tanh(nAL / self.adv_scale)

    def components(
        self,
        feats: EncodedObs,
        fraction: torch.Tensor,
        *,
        time_feat: torch.Tensor | None = None,
    ) -> QComponents:
        """Factored components at the supplied per-source `fraction`: value
        distribution logits + tanh-bounded scalar advantages.

        `taken_advantage` / `expected_advantage` reduce these to the real-units
        scalar advantage `adv = Σ_i A_i` (linear in the policy probs ⇒ closed-form
        expectation). The real-units Q is `bins_to_scalar(nV) + adv`.
        """
        planet_h, summary_h = self.encode(feats, time_feat=time_feat)
        if fraction.dim() == 1:
            fraction = fraction.unsqueeze(0)
        return QComponents(
            nV=self.value(summary_h),
            nA0=self.noop_adv(planet_h),
            nAL=self.launch_adv(planet_h, summary_h, fraction),
        )


def taken_advantage(
    comp: QComponents,
    launch: torch.Tensor,
    target_idx: torch.Tensor,
    owned_mask: torch.Tensor,
) -> torch.Tensor:
    """Scalar advantage `adv = Σ_i g_i[(1-l_i)nA0_i + l_i·nAL_i[t_i]]` [B] for the
    taken action — the action-dependent part of Q (the V baseline is
    action-independent). `comp.nAL` must be built at the action's fraction;
    `launch` is 0/1, `target_idx` indexes the chosen target per source.
    """
    if launch.dim() == 1:
        launch = launch.unsqueeze(0)
        target_idx = target_idx.unsqueeze(0)
        owned_mask = owned_mask.unsqueeze(0)
    p = comp.nA0.shape[1]
    al_taken = comp.nAL.gather(
        -1, target_idx.clamp(0, p - 1).unsqueeze(-1)
    ).squeeze(-1)  # [B, P]
    g = owned_mask.float()
    return (g * ((1.0 - launch) * comp.nA0 + launch * al_taken)).sum(-1)  # [B]


def expected_advantage(
    comp: QComponents,
    launch_p: torch.Tensor,
    target_probs: torch.Tensor,
    owned_mask: torch.Tensor,
) -> torch.Tensor:
    """Scalar expected advantage `adv = Σ_i g_i[(1-p_i)nA0_i + p_i·Σ_t π_t(i)·
    nAL_i[t]]` [B], LINEAR in the policy probs (the closed-form `E_a[adv]`, no
    REINFORCE). In real ship-margin units (the heads are tanh-bounded), this is the
    action-dependent part of Q the actor ascends directly: `Q = V(s) + adv`, so a
    single real-units α balances the value and entropy terms. `comp.nAL` must be
    built at the policy's reparam fraction; the caller controls detachment (nA0
    detached as a baseline, nAL live for the pathwise fraction grad,
    launch_p/target_probs live for the discrete grad).
    """
    if launch_p.dim() == 1:
        launch_p = launch_p.unsqueeze(0)
        target_probs = target_probs.unsqueeze(0)
        owned_mask = owned_mask.unsqueeze(0)
    al_exp = (target_probs * comp.nAL).sum(-1)  # [B, P]  Σ_t π_t·nAL[t]
    g = owned_mask.float()
    return (g * ((1.0 - launch_p) * comp.nA0 + launch_p * al_exp)).sum(-1)  # [B]


def make_targets(qf: SACSoftQ) -> SACSoftQ:
    """Build an EMA target network that mirrors `qf`'s parameters exactly.

    `SACSoftQ(qf.cfg)` reconstructs the non-persistent HLGauss support buffer
    identically by config, so it needs no copying; `load_state_dict` copies the
    learned parameters. `polyak_update` walks only parameters, so that buffer (the
    same constant in online and target nets) is correctly left untouched.
    """
    target = SACSoftQ(qf.cfg)
    target.load_state_dict(qf.state_dict())
    for p in target.parameters():
        p.requires_grad_(False)
    return target


def polyak_update(target: nn.Module, source: nn.Module, tau: float) -> None:
    """In-place soft update: `target ← (1-tau)·target + tau·source`."""
    with torch.no_grad():
        for tp, sp in zip(target.parameters(), source.parameters(), strict=True):
            tp.data.mul_(1.0 - tau).add_(sp.data, alpha=tau)


__all__ = [
    "MAX_PLANETS",
    "QComponents",
    "SACAction",
    "SACActor",
    "SACEncoder",
    "SACSoftQ",
    "expected_advantage",
    "make_targets",
    "polyak_update",
    "taken_advantage",
]
