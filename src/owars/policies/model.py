"""OrbitPolicy — set-transformer encoder + factored action heads.

Architecture:
  [ACTOR] [CRITIC] planet_tokens fleet_tokens
              │
              ▼
  [N × Transformer block] ─► token reps
              │
              ├─► h_actor (broadcast)  ─► concat onto each planet rep ─►
              │                             target attention + fraction head
              ├─► h_critic             ─► value MLP (replaces mean-pool)
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
dim (d for query/key, 2 for Beta params). Down-projecting `[planet_h ||
h_actor]` back to d would discard exactly the global-context capacity the
extra token was added to provide.

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
import torch.nn.functional as F

from .config import OrbitPolicyConfig
from .features import EncodedObs


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, ff_dim: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.ln2 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            dim, n_heads, dropout=dropout, batch_first=True
        )
        self.ff = nn.Sequential(
            nn.Linear(dim, ff_dim),
            nn.GELU(),
            nn.Linear(ff_dim, dim),
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor | None) -> torch.Tensor:
        h = self.ln1(x)
        a, _ = self.attn(h, h, h, key_padding_mask=key_padding_mask, need_weights=False)
        x = x + self.drop(a)
        x = x + self.drop(self.ff(self.ln2(x)))
        return x


@dataclass
class PolicyOutput:
    target_logits: torch.Tensor  # [B, P, P+1]  +1 = no-op slot
    fraction_alpha: torch.Tensor  # [B, P]
    fraction_beta: torch.Tensor   # [B, P]
    angle_alpha: torch.Tensor     # [B, P]
    angle_beta: torch.Tensor      # [B, P]
    value: torch.Tensor           # [B]
    planet_owned_mask: torch.Tensor  # [B, P] bool
    planet_mask: torch.Tensor        # [B, P] bool
    planet_ids: torch.Tensor          # [B, P] long


class OrbitPolicy(nn.Module):
    def __init__(self, cfg: OrbitPolicyConfig):
        super().__init__()
        self.cfg = cfg
        self.planet_embed = nn.Linear(cfg.planet_features, cfg.dim)
        self.fleet_embed = nn.Linear(cfg.fleet_features, cfg.dim)
        # Two learnable summary tokens (PMA-style). Initialized small so
        # they don't dominate the encoder at step 0 — gradient flow alone
        # will scale them up as the heads start using their output.
        self.actor_token = nn.Parameter(torch.zeros(1, 1, cfg.dim))
        self.critic_token = nn.Parameter(torch.zeros(1, 1, cfg.dim))
        nn.init.trunc_normal_(self.actor_token, std=0.02)
        nn.init.trunc_normal_(self.critic_token, std=0.02)
        self.layers = nn.ModuleList(
            [
                TransformerBlock(cfg.dim, cfg.ff_dim, cfg.n_heads, cfg.dropout)
                for _ in range(cfg.depth)
            ]
        )
        # `target_query` consumes [planet_h || h_actor] → 2·dim. `target_key`
        # stays at `dim` because adding the same h_actor to every key shifts
        # all `q·k` row-wise by a constant and cancels in the softmax — it
        # can't change the relative ranking of targets. Putting it on the
        # query side only is what gives the actor token bite.
        self.target_query = nn.Linear(2 * cfg.dim, cfg.dim, bias=False)
        self.target_key = nn.Linear(cfg.dim, cfg.dim, bias=False)
        self.noop_logit = nn.Parameter(torch.zeros(1))
        # 2·dim input for the same reason as target_query.
        self.fraction_head = nn.Linear(2 * cfg.dim, 2)  # log-alpha, log-beta offsets
        # Per-planet Beta(α, β) on a Δangle residual added to the analytic
        # lead-intercept in `sampling.py`. The simulator's actual arrival
        # step depends on planet motion *during travel*, so the analytic
        # prior is approximate; the residual lets the policy learn to
        # over/under-shoot per planet (e.g. consistently aim a bit ahead
        # for fast-orbiting targets). Per-(planet, target) would be more
        # expressive but quadratic; the per-target physics is already in
        # the analytic baseline, so per-planet residual suffices.
        self.angle_head = nn.Linear(2 * cfg.dim, 2)
        self.value_head = nn.Sequential(
            nn.Linear(cfg.dim, cfg.value_hidden),
            nn.GELU(),
            nn.Linear(cfg.value_hidden, 1),
        )
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

    def encode(
        self, feats: EncodedObs
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run the transformer over [actor, critic, planets..., fleets...].

        Returns `(planet_h, fleet_h, h_actor, h_critic, token_mask)` where
        `token_mask` is the planets+fleets mask (excludes the two summary
        tokens, which are always real and only consumed via the dedicated
        `h_actor` / `h_critic` slices).
        """
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

        h_p = self.planet_embed(planet_feats)
        h_f = self.fleet_embed(fleet_feats)
        # Prepend the two summary tokens, broadcast to batch dim.
        actor_t = self.actor_token.expand(b, -1, -1)
        critic_t = self.critic_token.expand(b, -1, -1)
        h = torch.cat([actor_t, critic_t, h_p, h_f], dim=1)
        # The two summary tokens are always real (never masked out).
        summary_mask = torch.ones(b, 2, dtype=torch.bool, device=planet_mask.device)
        full_mask = torch.cat([summary_mask, planet_mask, fleet_mask], dim=1)
        # `key_padding_mask=True` means *masked out* in MultiheadAttention.
        kpm = ~full_mask
        for layer in self.layers:
            h = layer(h, key_padding_mask=kpm)
        h_actor = h[:, 0]                 # [B, d]
        h_critic = h[:, 1]                # [B, d]
        planet_h = h[:, 2 : 2 + p]        # [B, P, d]
        fleet_h = h[:, 2 + p : 2 + p + f]  # [B, F, d]
        # Caller-facing mask is planets+fleets only.
        token_mask = torch.cat([planet_mask, fleet_mask], dim=1)
        return planet_h, fleet_h, h_actor, h_critic, token_mask

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
        # [B, P, P]
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
        # Concat the per-planet no-op logit on the last axis.
        noop = self.noop_logit.view(1, 1, 1).expand(b, p, 1)
        target_logits = torch.cat([logits, noop], dim=-1)  # [B, P, P+1]

        # Fraction head — interpret outputs as log-offsets on a base concentration.
        base = float(self.cfg.fraction_concentration)
        offs = self.fraction_head(planet_with_ctx)
        alpha = base * F.softplus(offs[..., 0]) + 1e-3
        beta = base * F.softplus(offs[..., 1]) + 1e-3

        # Angle residual head — same parameterization, separate weights.
        a_offs = self.angle_head(planet_with_ctx)
        angle_alpha = base * F.softplus(a_offs[..., 0]) + 1e-3
        angle_beta = base * F.softplus(a_offs[..., 1]) + 1e-3

        # Value: dedicated critic token (replaces mean-pool).
        value = self.value_head(h_critic).squeeze(-1)

        return PolicyOutput(
            target_logits=target_logits,
            fraction_alpha=alpha,
            fraction_beta=beta,
            angle_alpha=angle_alpha,
            angle_beta=angle_beta,
            value=value,
            planet_owned_mask=planet_owned,
            planet_mask=planet_mask,
            planet_ids=planet_ids,
        )
