"""End-to-end smoke test for `ppo_update` after the tanh-Gaussian rewrite.

Exercises the new `frac_z` recompute path: builds a tiny synthetic batch
that mirrors the keys `_stack_trajectories` produces, runs one PPO update,
and asserts the returned metrics are finite. Mainly a regression guard
against the old Beta-keys leaking back in (`fraction`, `angle_offset`).
"""

from __future__ import annotations

import math

import torch

from owars.policies.config import OrbitPolicyConfig
from owars.policies.features import EncodedObs, MAX_FLEETS, MAX_PLANETS
from owars.policies.model import OrbitPolicy
from owars.policies.sampling import sample_batch_with_records
from owars.training.ppo import ppo_update


def _toy_batch(model: OrbitPolicy, B: int, P: int = MAX_PLANETS, F: int = MAX_FLEETS) -> dict[str, torch.Tensor]:
    """Forward the model once on synthetic obs, sample, and pack a PPO batch.

    We use the *real* sampler to get a `frac_z` whose log_prob exactly
    matches what `ppo_update` will recompute — i.e. ratio ≈ 1 at step 0,
    which is the only invariant we want to assert at this scale.
    """
    torch.manual_seed(0)
    planet_feats = torch.randn(B, P, 19) * 0.3
    planet_mask = torch.zeros(B, P, dtype=torch.bool)
    planet_mask[:, :8] = True
    planet_owned = torch.zeros(B, P, dtype=torch.bool)
    planet_owned[:, :3] = True
    planet_ids = torch.full((B, P), -1, dtype=torch.long)
    planet_ids[:, :8] = torch.arange(8)
    planet_garrison = torch.zeros(B, P)
    planet_garrison[:, :8] = 50
    fleet_feats = torch.zeros(B, F, 15)
    fleet_mask = torch.zeros(B, F, dtype=torch.bool)
    feats = EncodedObs(
        planet_feats=planet_feats, planet_mask=planet_mask,
        planet_owned_mask=planet_owned, planet_ids=planet_ids,
        planet_garrison=planet_garrison,
        fleet_feats=fleet_feats, fleet_mask=fleet_mask,
    )
    with torch.no_grad():
        out = model(feats)
    # Build a SampleRecord per env via the real batched sampler so frac_z and
    # log_prob are mutually consistent with the model's current heads.
    from owars.game.observation import Observation
    from owars.game.types import Planet
    obs = [
        Observation(
            player=0, step=0,
            planets=[Planet(i, 0, 0.0, 0.0, 1.0, 10, 1) for i in range(8)],
            fleets=[], angular_velocity=0.04, initial_planets=[],
            comet_planet_ids=set(), comets=[], remaining_overage_time=60.0,
        ) for _ in range(B)
    ]
    _, records = sample_batch_with_records(out, obs, deterministic=False)
    target_idx = torch.stack([r.target_idx for r in records])
    frac_z = torch.stack([r.frac_z for r in records])
    log_prob = torch.stack([r.log_prob for r in records])
    old_target_logits = torch.stack([r.target_logits for r in records])
    old_fraction_mu = torch.stack([r.fraction_mu for r in records])
    old_fraction_log_sigma = torch.stack([r.fraction_log_sigma for r in records])

    return {
        "planet_feats": planet_feats,
        "planet_mask": planet_mask,
        "planet_owned_mask": planet_owned,
        "planet_ids": planet_ids,
        "planet_garrison": planet_garrison,
        "fleet_feats": fleet_feats,
        "fleet_mask": fleet_mask,
        "target_idx": target_idx,
        "frac_z": frac_z,
        "old_log_prob": log_prob,
        "owned_mask": planet_owned,
        "advantage": torch.randn(B),
        "return": torch.randn(B),
        "old_target_logits": old_target_logits,
        "old_fraction_mu": old_fraction_mu,
        "old_fraction_log_sigma": old_fraction_log_sigma,
    }


def test_ppo_update_runs_and_returns_finite_metrics():
    cfg = OrbitPolicyConfig(dim=32, ff_dim=64, depth=2, n_heads=2)
    model = OrbitPolicy(cfg)
    batch = _toy_batch(model, B=8)
    optim = torch.optim.AdamW(model.parameters(), lr=3e-4)

    log = ppo_update(
        model, optim, batch,
        clip_eps_low=0.2, clip_eps_high=0.28,
        value_coef=0.5, entropy_coef=0.01, pmpo_kl_coef=0.3,
        epochs=2, minibatch_size=4, grad_clip=0.5,
    )

    for name in ("policy_loss", "value_loss", "entropy", "approx_kl", "clip_frac", "pmpo_kl"):
        v = getattr(log, name)
        assert math.isfinite(v), f"{name}={v!r}"
    # On epoch 0 the new policy ≡ old policy, so KL must start at exactly 0.
    # The reported number is the running mean across all (epoch, minibatch)
    # updates, so we just sanity-check non-negativity here.
    assert log.pmpo_kl >= 0.0, log.pmpo_kl


def test_log_prob_recompute_matches_sample_time():
    """Re-evaluating log_prob at the recorded `frac_z` with the same
    parameters (no gradient step yet) must reproduce the recorded `log_prob`
    — this is the invariant that makes PPO's importance ratio ≈ 1 on epoch 0
    and the actual unit test for the tanh-Jacobian sign convention.
    """
    import math
    cfg = OrbitPolicyConfig(dim=32, ff_dim=64, depth=2, n_heads=2)
    model = OrbitPolicy(cfg)
    batch = _toy_batch(model, B=4)

    # Re-run forward (no-grad) and recompute log_prob exactly the way
    # ppo_update does — without taking any optimizer step.
    from owars.policies.features import EncodedObs
    feats = EncodedObs(
        planet_feats=batch["planet_feats"], planet_mask=batch["planet_mask"],
        planet_owned_mask=batch["planet_owned_mask"], planet_ids=batch["planet_ids"],
        planet_garrison=batch["planet_garrison"],
        fleet_feats=batch["fleet_feats"], fleet_mask=batch["fleet_mask"],
    )
    with torch.no_grad():
        out = model(feats)
    target = batch["target_idx"]
    p = out.target_logits.shape[1]
    target_log_probs = torch.log_softmax(out.target_logits, dim=-1)
    target_lp = target_log_probs.gather(-1, target.unsqueeze(-1)).squeeze(-1)
    z = batch["frac_z"]
    sigma = out.fraction_log_sigma.exp()
    normal_lp = (
        -0.5 * ((z - out.fraction_mu) / sigma).pow(2)
        - out.fraction_log_sigma
        - 0.5 * math.log(2.0 * math.pi)
    )
    tanh_correction = 2.0 * (math.log(2.0) - z - torch.nn.functional.softplus(-2.0 * z))
    frac_lp = normal_lp - tanh_correction + math.log(2.0)
    move_mask = (target != p).float()
    chosen = target_lp + move_mask * frac_lp

    owned = batch["owned_mask"].float()
    diff = (chosen - batch["old_log_prob"]) * owned
    # Should be exactly zero up to numerical noise — same params, same z.
    assert diff.abs().max().item() < 1e-4, diff.abs().max().item()


