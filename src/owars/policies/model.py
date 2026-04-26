"""OrbitPolicy — set-transformer encoder + factored action heads.

Architecture:
  planet tokens ─┐
                ├─► linear ─► [N × Transformer block] ─► token reps
  fleet tokens  ─┘                                            │
                                                              ├─► policy:
                                                              │     for each owned planet,
                                                              │     attend over all planets
                                                              │     to score targets;
                                                              │     plus a Beta head for
                                                              │     fraction-of-garrison.
                                                              └─► value: pool over real
                                                                    tokens, MLP scalar.

**The critic shares the transformer backbone with the actor.** Both heads
read from the same `[N × Transformer block]` output: the policy uses per-
planet token reps for target attention; the value head pools all real
tokens (planets + fleets) and MLPs to a scalar. This is the cheap default
— ~half the params of a 2-network setup, and the value-loss gradient
helps the encoder learn position-relevant features. The trade-off is that
value-loss spikes can destabilize the policy; in practice that's tamed by
(a) value-pretraining the encoder before PPO turns on, and (b) keeping
`value_coef ≤ 0.5`.

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
        self.layers = nn.ModuleList(
            [
                TransformerBlock(cfg.dim, cfg.ff_dim, cfg.n_heads, cfg.dropout)
                for _ in range(cfg.depth)
            ]
        )
        self.target_query = nn.Linear(cfg.dim, cfg.dim, bias=False)
        self.target_key = nn.Linear(cfg.dim, cfg.dim, bias=False)
        self.noop_logit = nn.Parameter(torch.zeros(1))
        self.fraction_head = nn.Linear(cfg.dim, 2)  # log-alpha, log-beta offsets
        self.value_head = nn.Sequential(
            nn.Linear(cfg.dim, cfg.value_hidden),
            nn.GELU(),
            nn.Linear(cfg.value_hidden, 1),
        )

    def encode(self, feats: EncodedObs) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
        h = torch.cat([h_p, h_f], dim=1)
        token_mask = torch.cat([planet_mask, fleet_mask], dim=1)
        # `key_padding_mask=True` means *masked out* in MultiheadAttention.
        kpm = ~token_mask
        for layer in self.layers:
            h = layer(h, key_padding_mask=kpm)
        return h[:, :p], h[:, p : p + f], token_mask

    def forward(self, feats: EncodedObs) -> PolicyOutput:
        planet_h, _fleet_h, token_mask = self.encode(feats)
        b, p, d = planet_h.shape

        # Promote scalar mask/garrison/id to batch dim if not already.
        def _b(t: torch.Tensor) -> torch.Tensor:
            return t.unsqueeze(0) if t.dim() == 1 else t

        planet_owned = _b(feats.planet_owned_mask)
        planet_mask = _b(feats.planet_mask)
        planet_ids = _b(feats.planet_ids)

        # Target attention: each planet emits a query; every planet is a key.
        q = self.target_query(planet_h)
        k = self.target_key(planet_h)
        # [B, P, P]
        logits = torch.einsum("bid,bjd->bij", q, k) / (d**0.5)
        # Mask out padded *targets*.
        logits = logits.masked_fill(~planet_mask.unsqueeze(1), float("-inf"))
        # Mask self-targets (diagonal). The simulator silently no-ops a
        # send-to-self anyway; without this mask the policy can put
        # probability mass on a meaningless action and the entropy term
        # rewards it.
        diag = torch.eye(p, dtype=torch.bool, device=logits.device).unsqueeze(0)
        logits = logits.masked_fill(diag, float("-inf"))
        # Concat the per-planet no-op logit on the last axis.
        noop = self.noop_logit.view(1, 1, 1).expand(b, p, 1)
        target_logits = torch.cat([logits, noop], dim=-1)  # [B, P, P+1]

        # Fraction head — interpret outputs as log-offsets on a base concentration.
        offs = self.fraction_head(planet_h)
        base = float(self.cfg.fraction_concentration)
        alpha = base * F.softplus(offs[..., 0]) + 1e-3
        beta = base * F.softplus(offs[..., 1]) + 1e-3

        # Value: mean-pool over real tokens.
        token_h = torch.cat([planet_h, _fleet_h], dim=1)
        denom = token_mask.float().sum(dim=1, keepdim=True).clamp_min(1.0)
        pooled = (token_h * token_mask.unsqueeze(-1).float()).sum(dim=1) / denom
        value = self.value_head(pooled).squeeze(-1)

        return PolicyOutput(
            target_logits=target_logits,
            fraction_alpha=alpha,
            fraction_beta=beta,
            value=value,
            planet_owned_mask=planet_owned,
            planet_mask=planet_mask,
            planet_ids=planet_ids,
        )
