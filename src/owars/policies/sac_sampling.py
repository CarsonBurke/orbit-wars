"""Translate the hybrid factored SAC actor into Orbit Wars action lists.

The SAC actor emits the SAME factored action PPO does — per owned source
planet a Bernoulli launch, a masked Categorical target, and a tanh-squashed
Normal fraction — and the launch *angle* is solved analytically from the
chosen target by the lead-intercept solver. We therefore reuse `sampling.py`
verbatim for rollout move-building + legal-target masking: this module only
adapts the actor's heads into a `PolicyOutput` so the `sampling` helpers can do
the geometry.

`sac_sample_with_record` returns the legal Kaggle action list plus a
`SampleRecord` whose `(raw_launch, target_idx, fraction, target_legal_mask)`
is what the SAC replay buffer stores — `raw_launch` (the masked Bernoulli
sample), not the materialized launch, because the actor optimizes that raw
launch and the critic must condition on the same action. The returned actions
carry the
7-element tracker payload (`[id, angle, ships, target_id, eta, x, y]`), so the
caller's `_FleetTargetTracker.record` keeps encoded fleet features in sync the
same way the PPO rollout does. `sac_sample_actions` is the moves-only
deterministic path for inference; `uniform_random_factored_actions` produces
max-entropy warmup actions during the `learning_starts` window.
"""

from __future__ import annotations

from typing import Any

import torch

from .features import EncodedObs, fleet_target_planet_idx_or_empty
from .model import PolicyOutput
from .sac_model import SACActor
from .sampling import (
    SampleRecord,
    sample_batch_actions_raw,
    sample_batch_with_records_raw,
)


def _mark_cuda_graph_step(device: torch.device) -> None:
    """Delimit a cudagraph replay (no-op off CUDA / when not graphed)."""
    if device.type != "cuda":
        return
    mark = getattr(torch.compiler, "cudagraph_mark_step_begin", None)
    if callable(mark):
        mark()


def _heads_to_policy_output(
    launch_logits: torch.Tensor,
    target_logits: torch.Tensor,
    fraction_mean: torch.Tensor,
    fraction_log_std: torch.Tensor,
    feats: EncodedObs,
) -> PolicyOutput:
    """Pack actor heads + planet masks into a `PolicyOutput` for `sampling.py`.

    `sampling`'s rollout helpers only read the head logits and the planet
    masks/ids — they own legality masking, Bernoulli/Categorical/fraction
    sampling, and the lead-intercept geometry. The unmasked head logits are
    what they expect (they apply their own `_apply_target_legal_mask`), so we
    hand back the raw heads with dummy value fields.
    """

    def _b(t: torch.Tensor) -> torch.Tensor:
        return t.unsqueeze(0) if t.dim() == 1 else t

    b = launch_logits.shape[0]
    return PolicyOutput(
        launch_logits=launch_logits,
        target_logits=target_logits,
        fraction_mean=fraction_mean,
        fraction_log_std=fraction_log_std,
        value=launch_logits.new_zeros((b,)),
        value_logits=launch_logits.new_zeros((b, 0)),
        planet_owned_mask=_b(feats.planet_owned_mask),
        planet_mask=_b(feats.planet_mask),
        planet_ids=_b(feats.planet_ids),
    )


def _actor_policy_output(
    actor: SACActor, feats: EncodedObs, *, time_feat: torch.Tensor | None = None
) -> PolicyOutput:
    launch_logits, target_logits, fraction_mean, fraction_log_std, _p = (
        actor._heads(feats, time_feat=time_feat)
    )
    return _heads_to_policy_output(
        launch_logits, target_logits, fraction_mean, fraction_log_std, feats
    )


def sac_sample_with_record(
    actor: SACActor,
    feats: EncodedObs,
    raw_obs: Any,
    *,
    deterministic: bool = False,
    time_feat: torch.Tensor | None = None,
) -> tuple[list[list], SampleRecord]:
    """Stochastic rollout for one env: legal Kaggle actions + the buffer record.

    Reuses `sampling.sample_batch_with_records_raw` (batch-1), which builds the
    lead-solved 7-element action lists and the
    `SampleRecord(launch, target_idx, fraction, target_legal_mask)` the SAC
    buffer stores.
    """
    out = _actor_policy_output(actor, feats, time_feat=time_feat)
    actions_list, records = sample_batch_with_records_raw(
        out, [raw_obs], deterministic=deterministic
    )
    return actions_list[0], records[0]


def sac_sample_actions(
    actor: SACActor,
    feats: EncodedObs,
    raw_obs: Any,
    *,
    deterministic: bool = True,
    time_feat: torch.Tensor | None = None,
) -> list[list]:
    """Moves-only path for inference / opponent self-play.

    `time_feat` is the game-clock scalar the encoder FiLM conditions on; pass
    `step/episode_steps` at play time so the policy is endgame-aware.

    Reuses `sampling.sample_batch_actions_raw` (batch-1), which applies the
    deterministic launch-if-idle fallback so a confident-but-sub-0.5 launch
    still acts at play time.
    """
    out = _actor_policy_output(actor, feats, time_feat=time_feat)
    return sample_batch_actions_raw(out, [raw_obs], deterministic=deterministic)[0]


def uniform_random_factored_actions(
    actor: SACActor,
    feats: EncodedObs,
    raw_obs: Any,
    *,
    time_feat: torch.Tensor | None = None,
) -> tuple[list[list], SampleRecord]:
    """Uniform-random factored warmup action for the `learning_starts` window.

    cleanrl samples uniformly over the action space before `learning_starts`.
    Here that means a uniform launch Bernoulli, a uniform target over the legal
    support, and a wide fraction — produced by flattening the actor's head
    logits / mean so the sampler draws from the maximum-entropy factored
    distribution, then running the same `sampling` geometry so warmup
    transitions are legal and on-distribution with learned rollouts.
    """
    launch_logits, target_logits, fraction_mean, fraction_log_std, _p = (
        actor._heads(feats, time_feat=time_feat)
    )
    # Uniform launch (logit 0 → p=0.5), uniform target (flat logits over the
    # legal support `sampling` masks in — keep -inf on padded/self targets so
    # the geometry isn't asked to solve impossible routes), wide fraction.
    flat_launch = torch.zeros_like(launch_logits)
    flat_targets = torch.where(
        torch.isfinite(target_logits),
        torch.zeros_like(target_logits),
        target_logits,
    )
    flat_mean = torch.zeros_like(fraction_mean)
    wide_log_std = torch.zeros_like(fraction_log_std)  # std=1 pre-squash
    out = _heads_to_policy_output(
        flat_launch, flat_targets, flat_mean, wide_log_std, feats
    )
    actions_list, records = sample_batch_with_records_raw(
        out, [raw_obs], deterministic=False
    )
    return actions_list[0], records[0]


# ---------------------------------------------------------------------------
# Compiled + batched rollout inference
# ---------------------------------------------------------------------------
#
# The single-env `sac_sample_*` wrappers above run the actor's `_heads` eagerly
# at batch 1. The vectorized SAC rollout instead batches every alive env's
# observation into one forward, run through a `torch.compile`d kernel whose
# *outer* `torch.autocast(bf16)` dispatches the encoder's SDPA to
# FlashAttention-2 — the SAME idiom PPO's `vec_rollout._RolloutForwardKernel`
# uses. The compiled kernel returns only the (unmasked) head logits packed into
# a `PolicyOutput`; the legality + lead-intercept geometry stays eager in
# `sampling.py`, outside the graph.


class _SACHeadsKernel(torch.nn.Module):
    """Compiled actor-`_heads` forward for batched rollout inference."""

    def __init__(self, actor: SACActor, *, autocast_enabled: bool) -> None:
        super().__init__()
        self.actor = actor
        self.autocast_enabled = bool(autocast_enabled)

    def forward(
        self,
        planet_feats: torch.Tensor,
        planet_mask: torch.Tensor,
        planet_owned_mask: torch.Tensor,
        planet_ids: torch.Tensor,
        planet_garrison: torch.Tensor,
        fleet_feats: torch.Tensor,
        fleet_mask: torch.Tensor,
        fleet_target_planet_idx: torch.Tensor,
        time_feat: torch.Tensor,
    ) -> PolicyOutput:
        feats = EncodedObs(
            planet_feats=planet_feats,
            planet_mask=planet_mask,
            planet_owned_mask=planet_owned_mask,
            planet_ids=planet_ids,
            planet_garrison=planet_garrison,
            fleet_feats=fleet_feats,
            fleet_mask=fleet_mask,
            fleet_target_planet_idx=fleet_target_planet_idx,
        )
        with torch.autocast(
            device_type="cuda", dtype=torch.bfloat16, enabled=self.autocast_enabled
        ):
            launch_logits, target_logits, fraction_mean, fraction_log_std, _p = (
                self.actor._heads(feats, time_feat=time_feat)
            )
        return _heads_to_policy_output(
            launch_logits, target_logits, fraction_mean, fraction_log_std, feats
        )


def _heads_kernel_cache(actor: SACActor) -> dict:
    cache = actor.__dict__.get("_owars_sac_heads_kernel_cache")
    if cache is None:
        cache = {}
        actor.__dict__["_owars_sac_heads_kernel_cache"] = cache
    return cache


def get_sac_heads_kernel(
    actor: SACActor, *, device: torch.device, compile_mode: str | None
) -> torch.nn.Module:
    """Build (and cache) the rollout heads kernel; `torch.compile`d on CUDA."""
    mode = compile_mode if device.type == "cuda" else None
    key = ("heads", mode)
    cache = _heads_kernel_cache(actor)
    cached = cache.get(key)
    if cached is not None:
        return cached
    kernel: torch.nn.Module = _SACHeadsKernel(
        actor, autocast_enabled=device.type == "cuda"
    )
    if mode is not None:
        kernel = torch.compile(kernel, dynamic=False, fullgraph=True, mode=mode)
    cache[key] = kernel
    return kernel


def run_sac_heads(
    kernel: torch.nn.Module,
    feats: EncodedObs,
    *,
    device: torch.device,
    time_feat: torch.Tensor,
) -> PolicyOutput:
    """Batched, no-grad actor-heads forward (marks the cudagraph step first).

    `time_feat` is the per-env game-clock scalar ([B] in [0,1]) the encoder's
    FiLM conditions on. The returned `PolicyOutput` may alias cudagraph static
    buffers under `reduce-overhead`, so the caller must consume it (sample +
    materialize any kept tensors to CPU) before the next `run_sac_heads` call.
    """
    _mark_cuda_graph_step(device)
    with torch.no_grad():
        return kernel(
            feats.planet_feats,
            feats.planet_mask,
            feats.planet_owned_mask,
            feats.planet_ids,
            feats.planet_garrison,
            feats.fleet_feats,
            feats.fleet_mask,
            fleet_target_planet_idx_or_empty(feats),
            time_feat,
        )


def flatten_policy_output(out: PolicyOutput) -> PolicyOutput:
    """Max-entropy ("uniform") factored policy for the `learning_starts` warmup.

    Batched analogue of `uniform_random_factored_actions`: launch logit 0
    (p=0.5), flat logits over the legal target support (keep -inf on
    padded/self slots so the geometry isn't handed impossible routes), and a
    zero-mean unit-std fraction. Masks/ids pass through unchanged.
    """
    flat_targets = torch.where(
        torch.isfinite(out.target_logits),
        torch.zeros_like(out.target_logits),
        out.target_logits,
    )
    return PolicyOutput(
        launch_logits=torch.zeros_like(out.launch_logits),
        target_logits=flat_targets,
        fraction_mean=torch.zeros_like(out.fraction_mean),
        fraction_log_std=torch.zeros_like(out.fraction_log_std),
        value=out.value,
        value_logits=out.value_logits,
        planet_owned_mask=out.planet_owned_mask,
        planet_mask=out.planet_mask,
        planet_ids=out.planet_ids,
    )


def record_to_cpu(record: SampleRecord) -> SampleRecord:
    """Detach + move a rollout record off the (cudagraph) device buffers.

    Must run before the next `run_sac_heads` overwrites the buffers the record
    tensors are derived from.
    """
    return SampleRecord(
        launch=record.launch.detach().cpu(),
        raw_launch=record.raw_launch.detach().cpu(),
        target_idx=record.target_idx.detach().cpu(),
        fraction=record.fraction.detach().cpu(),
        log_prob=record.log_prob.detach().cpu(),
        target_legal_mask=record.target_legal_mask.detach().cpu(),
    )


__all__ = [
    "flatten_policy_output",
    "get_sac_heads_kernel",
    "record_to_cpu",
    "run_sac_heads",
    "sac_sample_actions",
    "sac_sample_with_record",
    "sample_batch_actions_raw",
    "sample_batch_with_records_raw",
    "uniform_random_factored_actions",
]
