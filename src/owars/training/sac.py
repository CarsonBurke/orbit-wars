"""SAC trainer for Orbit Wars.

Hybrid discrete+continuous variant — the actor emits the SAME factored action
PPO uses: a per-source-planet Bernoulli launch, a masked Categorical target,
and a tanh-squashed Normal fraction. The launch *angle* is solved analytically
from the chosen target by the lead-intercept solver (see `sac_sampling.py`),
so the policy never has to learn a raw absolute angle.

SAC treatment:
  * discrete-SAC (closed-form Bernoulli+categorical entropy) for launch+target;
  * cleanrl reparameterized tanh-Normal for the fraction.
The critic is **factored (dueling)**: Q(s, a) = V(s) + Σ_i A_i(s, a_i) over owned
source planets, with each planet's per-option advantage enumerable (`SACSoftQ`).
This makes the soft-value expectation Σ_a π(a)·Q(s, a) tractable in *closed
form*: the discrete launch+target gradient is the exact, baseline-free policy
gradient (no REINFORCE / no (Q−b) variance), and the fraction uses the
pathwise/reparam gradient. Two entropy temperatures are tuned independently:
`alpha_discrete` (launch+target) and `alpha_continuous` (fraction).

Value scale: the factored critic is dueling in REAL ship-margin units,
`Q(s,a) = V(s) + adv(s,a)`. The state value `V = bins_to_scalar(nV)` is decoded
from an HL-Gauss two-hot distribution over symlog-spaced bins (clamped to
`[value_min, value_max]`, bounding the bootstrap — this replaces the earlier
scalar symlog-MSE critic, whose loss gradient vanished at large |Q| and let the
deadly triad run the value away). The per-planet advantages are tanh-bounded
(±`adv_scale`), so `adv = Σ_i A_i` is bounded and linear in the policy probs. The
TD loss splits the heads: V learns the full target via HL-Gauss CE, adv the
residual `(y − V)` (V detached) via Huber. The reward is the
per-step production-margin delta (O(±10²); see `RewardCfg` and
`rollout._obs_production_margin`), and the encoder is FiLM-conditioned on a
global time feature so the value-to-go can depend on the remaining horizon.

Orbit Wars-specific adaptations vs cleanrl `sac_continuous_action.py`:

* Replay-buffer state is the structured `EncodedObs` (planet/fleet token
  tensors + masks), not a flat vector. The stored action is factored
  (`raw_launch[P]` — the policy's raw Bernoulli sample, i.e. the action the
  actor optimizes and the critic conditions on, NOT the env-materialized
  launch — `target_idx[P]`, `fraction[P]`) plus the legal-target masks for s
  and s' (the q-step samples a'~π(·|s') and must mask to s' legal support).
* Self-play uses the LIVE model only: in league mode the opponent seat is
  either a builtin baseline (prob `builtin_prob`) or the current learner
  itself. In fixed mode every opponent is sampled from
  `opponents.fixed_opponents`. No frozen snapshot copies are kept. `league.py`
  is used only for Elo bookkeeping. Only learner-seat transitions are pushed to
  replay.
"""

from __future__ import annotations

import math
import random
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F  # noqa: N812
from torch.utils.tensorboard import SummaryWriter

from ..policies.config import OrbitPolicyConfig
from ..policies.features import (
    MAX_FLEETS,
    MAX_PLANETS,
    EncodedObs,
    encode_raw_observations,
)
from ..policies.sac_model import (
    QComponents,
    SACActor,
    SACSoftQ,
    expected_advantage,
    make_targets,
    polyak_update,
    taken_advantage,
)
from ..policies.sac_sampling import (
    flatten_policy_output,
    get_sac_heads_kernel,
    record_to_cpu,
    run_sac_heads,
    sample_batch_actions_raw,
    sample_batch_with_records_raw,
)
from ..policies.sampling import (
    SampleBatchRecord,
    SampleRecord,
    sample_batch_with_records_context,
)
from .config import RunConfig
from .elo import EloTracker
from .league import BUILTIN, LEARNER_NAME, OpponentPool
from .numpy_env import NumpyVecEnv
from .rollout import _obs_production_margin
from .sharded_numpy_env import ShardedNumpyVecEnv
from .vec_env import VecEnv


def _owned_gate(feats: EncodedObs) -> torch.Tensor:
    """Owned & alive planet mask `[B, P]` float for assembling factored Q."""
    owned = feats.planet_owned_mask
    pmask = feats.planet_mask
    if owned.dim() == 1:
        owned = owned.unsqueeze(0)
        pmask = pmask.unsqueeze(0)
    return (owned & pmask).float()


# ---------------------------------------------------------------------------
# Replay buffer
# ---------------------------------------------------------------------------


@dataclass
class SACBatch:
    """One sampled minibatch, all tensors on the trainer device."""

    feats: EncodedObs
    next_feats: EncodedObs
    launch: torch.Tensor               # [B, P] float 0/1
    target_idx: torch.Tensor           # [B, P] int64
    fraction: torch.Tensor             # [B, P] float
    target_legal_mask: torch.Tensor    # [B, P, P] bool — legal support at s
    next_target_legal_mask: torch.Tensor  # [B, P, P] bool — legal support at s'
    reward: torch.Tensor               # [B]
    done: torch.Tensor                 # [B] float (0/1)
    time: torch.Tensor                 # [B] game-clock ∈ [0,1] at s (FiLM cond)
    next_time: torch.Tensor            # [B] game-clock ∈ [0,1] at s'


class ReplayBuffer:
    """Circular buffer of factored-action transitions over `EncodedObs` states.

    Storage lives on `device` (CPU by default to spare GPU memory and let the
    buffer scale into system RAM). `sample` moves the gathered minibatch to the
    train device per update — the standard SAC replay pattern.

    Per transition we store the factored action (`launch`, `target_idx`,
    `fraction`) and the legal-target mask for BOTH s and s'. The next-state mask
    is needed because the soft Bellman target samples a'~π(·|s') and must mask
    the target categorical to s' legal support. Footprint is ~83 KB/transition,
    dominated by `fleet_feats [MAX_FLEETS=384, 20] f32` for s and s' (≈61 KB);
    the two `[64, 64]` bool masks add 8 KB. So capacity × 83 KB (50k ≈ 4.2 GB
    of RAM). At large `batch_size × UTD` the per-update H2D move of these obs is
    the dominant learn-phase cost, so `sample` supports a pinned `non_blocking`
    transfer that `_ReplayPrefetcher` overlaps with compute on a side stream.
    """

    def __init__(
        self,
        capacity: int,
        *,
        planet_features: int,
        fleet_features: int,
        device: str | torch.device = "cpu",
    ):
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.capacity = int(capacity)
        self.device = torch.device(device)
        f32 = torch.float32
        i64 = torch.int64

        def state_block() -> dict[str, torch.Tensor]:
            return {
                "planet_feats": torch.zeros(
                    (capacity, MAX_PLANETS, planet_features), dtype=f32, device=self.device
                ),
                "planet_mask": torch.zeros(
                    (capacity, MAX_PLANETS), dtype=torch.bool, device=self.device
                ),
                "planet_owned_mask": torch.zeros(
                    (capacity, MAX_PLANETS), dtype=torch.bool, device=self.device
                ),
                "planet_ids": torch.full(
                    (capacity, MAX_PLANETS), -1, dtype=i64, device=self.device
                ),
                "planet_garrison": torch.zeros(
                    (capacity, MAX_PLANETS), dtype=f32, device=self.device
                ),
                "fleet_feats": torch.zeros(
                    (capacity, MAX_FLEETS, fleet_features), dtype=f32, device=self.device
                ),
                "fleet_mask": torch.zeros(
                    (capacity, MAX_FLEETS), dtype=torch.bool, device=self.device
                ),
            }

        self.state = state_block()
        self.next_state = state_block()
        self.launch = torch.zeros(
            (capacity, MAX_PLANETS), dtype=f32, device=self.device
        )
        self.target_idx = torch.zeros(
            (capacity, MAX_PLANETS), dtype=i64, device=self.device
        )
        self.fraction = torch.zeros(
            (capacity, MAX_PLANETS), dtype=f32, device=self.device
        )
        self.target_legal_mask = torch.zeros(
            (capacity, MAX_PLANETS, MAX_PLANETS), dtype=torch.bool, device=self.device
        )
        self.next_target_legal_mask = torch.zeros(
            (capacity, MAX_PLANETS, MAX_PLANETS), dtype=torch.bool, device=self.device
        )
        self.reward = torch.zeros((capacity,), dtype=f32, device=self.device)
        self.done = torch.zeros((capacity,), dtype=f32, device=self.device)
        self.time = torch.zeros((capacity,), dtype=f32, device=self.device)
        self.next_time = torch.zeros((capacity,), dtype=f32, device=self.device)

        self.ptr = 0
        self.size = 0

    def __len__(self) -> int:
        return self.size

    def add(
        self,
        feats: EncodedObs,
        launch: torch.Tensor,
        target_idx: torch.Tensor,
        fraction: torch.Tensor,
        target_legal_mask: torch.Tensor,
        next_target_legal_mask: torch.Tensor,
        reward: float,
        done: bool,
        next_feats: EncodedObs,
        time: float,
        next_time: float,
    ) -> None:
        """Insert one transition. All tensors must be unbatched (no leading B)."""
        idx = self.ptr

        def _check_unbatched(t: torch.Tensor, name: str, want_dim: int) -> torch.Tensor:
            if t.dim() == want_dim + 1 and t.shape[0] == 1:
                return t.squeeze(0)
            if t.dim() != want_dim:
                raise ValueError(f"{name} expected {want_dim}D, got {tuple(t.shape)}")
            return t

        def _write_state(block: dict[str, torch.Tensor], s: EncodedObs) -> None:
            block["planet_feats"][idx].copy_(
                _check_unbatched(s.planet_feats, "planet_feats", 2).to(
                    self.device, dtype=torch.float32, non_blocking=False
                )
            )
            block["planet_mask"][idx].copy_(
                _check_unbatched(s.planet_mask, "planet_mask", 1).to(self.device)
            )
            block["planet_owned_mask"][idx].copy_(
                _check_unbatched(s.planet_owned_mask, "planet_owned_mask", 1).to(
                    self.device
                )
            )
            block["planet_ids"][idx].copy_(
                _check_unbatched(s.planet_ids, "planet_ids", 1).to(
                    self.device, dtype=torch.int64
                )
            )
            block["planet_garrison"][idx].copy_(
                _check_unbatched(s.planet_garrison, "planet_garrison", 1).to(
                    self.device, dtype=torch.float32
                )
            )
            block["fleet_feats"][idx].copy_(
                _check_unbatched(s.fleet_feats, "fleet_feats", 2).to(
                    self.device, dtype=torch.float32
                )
            )
            block["fleet_mask"][idx].copy_(
                _check_unbatched(s.fleet_mask, "fleet_mask", 1).to(self.device)
            )

        _write_state(self.state, feats)
        _write_state(self.next_state, next_feats)
        self.launch[idx].copy_(
            _check_unbatched(launch, "launch", 1).to(self.device, dtype=torch.float32)
        )
        self.target_idx[idx].copy_(
            _check_unbatched(target_idx, "target_idx", 1).to(
                self.device, dtype=torch.int64
            )
        )
        self.fraction[idx].copy_(
            _check_unbatched(fraction, "fraction", 1).to(
                self.device, dtype=torch.float32
            )
        )
        self.target_legal_mask[idx].copy_(
            _check_unbatched(target_legal_mask, "target_legal_mask", 2).to(self.device)
        )
        self.next_target_legal_mask[idx].copy_(
            _check_unbatched(
                next_target_legal_mask, "next_target_legal_mask", 2
            ).to(self.device)
        )
        self.reward[idx] = float(reward)
        self.done[idx] = float(bool(done))
        self.time[idx] = float(time)
        self.next_time[idx] = float(next_time)

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(
        self,
        batch_size: int,
        *,
        device: str | torch.device,
        non_blocking: bool = False,
    ) -> SACBatch:
        if self.size < batch_size:
            raise ValueError(
                f"requested batch_size {batch_size} > buffer size {self.size}"
            )
        idx = torch.randint(
            0, self.size, (batch_size,), dtype=torch.long, device=self.device
        )
        out_device = torch.device(device)
        # CPU→CUDA: route the gathered rows through pinned host memory so the H2D
        # is true DMA and `non_blocking` actually overlaps (the copy source must
        # be page-locked, or the flag is ignored and the copy is synchronous).
        # PyTorch's caching host allocator recycles the pinned blocks and gates
        # reuse on copy completion, so no manual double-buffering is needed; the
        # transfer/compute overlap is driven by `_ReplayPrefetcher`. `index_select`
        # produces a fresh (pageable) tensor, so we pin that result, not the
        # buffer storage (pinning the buffer wouldn't survive the gather).
        pin = non_blocking and out_device.type == "cuda" and self.device.type == "cpu"

        def _move(t: torch.Tensor) -> torch.Tensor:
            gathered = t.index_select(0, idx)
            if pin:
                return gathered.pin_memory().to(out_device, non_blocking=True)
            return gathered.to(out_device)

        def _gather_state(block: dict[str, torch.Tensor]) -> EncodedObs:
            return EncodedObs(
                planet_feats=_move(block["planet_feats"]),
                planet_mask=_move(block["planet_mask"]),
                planet_owned_mask=_move(block["planet_owned_mask"]),
                planet_ids=_move(block["planet_ids"]),
                planet_garrison=_move(block["planet_garrison"]),
                fleet_feats=_move(block["fleet_feats"]),
                fleet_mask=_move(block["fleet_mask"]),
            )

        return SACBatch(
            feats=_gather_state(self.state),
            next_feats=_gather_state(self.next_state),
            launch=_move(self.launch),
            target_idx=_move(self.target_idx),
            fraction=_move(self.fraction),
            target_legal_mask=_move(self.target_legal_mask),
            next_target_legal_mask=_move(self.next_target_legal_mask),
            reward=_move(self.reward),
            done=_move(self.done),
            time=_move(self.time),
            next_time=_move(self.next_time),
        )


def _record_batch_stream(batch: SACBatch, stream: torch.cuda.Stream) -> None:
    """Tell the caching allocator every tensor in `batch` is consumed on
    `stream`, so the side-stream-allocated GPU memory isn't recycled until the
    compute that reads it has finished (the NVIDIA/timm prefetch idiom). Without
    this, the allocator could hand the block back to a later `sample` while the
    current update is still reading it.
    """

    def _rec(t: torch.Tensor) -> None:
        t.record_stream(stream)

    for obs in (batch.feats, batch.next_feats):
        _rec(obs.planet_feats)
        _rec(obs.planet_mask)
        _rec(obs.planet_owned_mask)
        _rec(obs.planet_ids)
        _rec(obs.planet_garrison)
        _rec(obs.fleet_feats)
        _rec(obs.fleet_mask)
    for t in (
        batch.launch,
        batch.target_idx,
        batch.fraction,
        batch.target_legal_mask,
        batch.next_target_legal_mask,
        batch.reward,
        batch.done,
        batch.time,
        batch.next_time,
    ):
        _rec(t)


class _ReplayPrefetcher:
    """1-deep async minibatch prefetcher: overlaps the CPU→GPU H2D copy of the
    NEXT minibatch with the compute on the CURRENT one. The replay stays
    CPU-resident; each `next()` returns a fresh uniform-random minibatch already
    on the train device.

    On CUDA: samples on a dedicated side stream with pinned, `non_blocking` H2D,
    so the DMA runs concurrently with the compute stream's kernels.
    `wait_stream` serializes the consume after the copy completes, and
    `_record_batch_stream` keeps the side-stream allocation alive until the
    compute stream is done reading it. On CPU/eager (no CUDA stream) it degrades
    to a plain synchronous `replay.sample`, so the same call site works in tests.

    NOTE — UNVERIFIED ON GPU HERE: the side-stream H2D + `record_stream`
    interaction with `torch.compile` `reduce-overhead` cudagraph trees is novel
    for this repo (no GPU in this environment). The compiled kernels copy these
    batch tensors into their own static input buffers — a normal op on the
    compute stream — so each prefetched tensor is read once on the compute
    stream and `record_stream(compute)` covers that lifetime. Validate
    throughput and correctness on a GPU before relying on the overlap.
    """

    def __init__(
        self, replay: ReplayBuffer, batch_size: int, device: torch.device
    ) -> None:
        self._replay = replay
        self._batch_size = batch_size
        self._device = device
        self._use_stream = device.type == "cuda" and replay.device.type == "cpu"
        self._stream = torch.cuda.Stream(device) if self._use_stream else None
        self._next: SACBatch | None = None
        if self._use_stream:
            self._preload()

    def _preload(self) -> None:
        with torch.cuda.stream(self._stream):
            self._next = self._replay.sample(
                self._batch_size, device=self._device, non_blocking=True
            )

    def next(self) -> SACBatch:
        if not self._use_stream:
            return self._replay.sample(self._batch_size, device=self._device)
        compute = torch.cuda.current_stream(self._device)
        # Block the compute stream until the prefetched H2D copy lands, then keep
        # its allocation alive across the compute that reads it.
        compute.wait_stream(self._stream)
        batch = self._next
        assert batch is not None  # always primed: __init__/next() re-preload
        _record_batch_stream(batch, compute)
        self._preload()
        return batch


# ---------------------------------------------------------------------------
# Rollout helpers
# ---------------------------------------------------------------------------


def _time_feat_from_obs(
    obs_list: list[Any], episode_steps: int, device: torch.device | str
) -> torch.Tensor:
    """Per-env game-clock scalar ∈ [0,1] (`step / episode_steps`) for FiLM.

    The production-margin reward's value-to-go shrinks with the remaining
    horizon, so the encoder conditions on this clock to make the endgame value
    learnable (see `SACEncoder.time_film`).
    """
    vals = []
    for o in obs_list:
        get = o.get if isinstance(o, dict) else lambda k, d=None: getattr(o, k, d)
        step = float(get("step", 0) or 0)
        vals.append(min(1.0, max(0.0, step / float(episode_steps))))
    return torch.tensor(vals, dtype=torch.float32, device=device)


# ---------------------------------------------------------------------------
# SAC update step
# ---------------------------------------------------------------------------


@dataclass
class SACUpdateOutput:
    qf1_loss: float
    qf2_loss: float
    qf1_value: float
    qf2_value: float
    qf_grad_norm: float
    boot_value: float          # hard (entropy-free) bootstrap value, real units
    boot_entropy_bonus: float  # entropy contribution to the soft target, real units
    explained_var: float       # EV of taken-Q vs bootstrap target (critic health)
    adv_explained_var: float   # EV of the action-dependent advantage vs target residual
    adv_spread: float          # std of the decoded advantage contribution (real units)
    alpha_disc: float
    alpha_cont: float
    # Actor-side metrics; `None` on a logging tick where no actor update ran
    # (possible only when num_envs*gradient_steps < policy_frequency).
    actor_loss: float | None
    actor_grad_norm: float | None
    alpha_disc_loss: float | None
    alpha_cont_loss: float | None
    h_disc_mean: float | None          # achieved discrete entropy, summed over owned (batch mean)
    h_disc_per_planet: float | None    # achieved per-owned-planet discrete entropy
    target_h_disc: float | None        # per-owned-planet target entropy α_disc drives toward
    logp_frac_mean: float | None       # p-weighted Σ g·p·logp_frac (= -H_cont)
    target_h_cont: float | None        # launch-mass-scaled target entropy α_cont drives toward


# ---------------------------------------------------------------------------
# Compiled update kernels (autocast → FA-2; torch.compile fusion)
# ---------------------------------------------------------------------------
#
# Mirrors the PPO minibatch-kernel idiom (`ppo.py`): a flat-tensor `nn.Module`
# whose forward wraps the network passes in `torch.autocast(bf16)` — the *outer*
# autocast is what dispatches the encoder's `scaled_dot_product_attention` to
# FlashAttention-2 (see `model.py` SDPA note) — then computes the loss. The
# whole forward is `torch.compile`d (`dynamic=False, fullgraph=True`) so inductor
# fuses the ~10⁴ eager ops into a handful of kernels. Backward, optimizer steps,
# and the (scalar) alpha updates stay eager.
#
# The two alphas are passed as tensor inputs so the graph stays static; `gamma`
# is a compile-time constant baked into __init__.


def _encoded_args(feats: EncodedObs) -> tuple[torch.Tensor, ...]:
    """EncodedObs → its seven tensors in field order (flat kernel inputs)."""
    return (
        feats.planet_feats,
        feats.planet_mask,
        feats.planet_owned_mask,
        feats.planet_ids,
        feats.planet_garrison,
        feats.fleet_feats,
        feats.fleet_mask,
    )


def _encoded_row(feats: EncodedObs, i: int) -> EncodedObs:
    """One row of a batched `EncodedObs` as an unbatched state (for replay)."""
    return EncodedObs(
        planet_feats=feats.planet_feats[i],
        planet_mask=feats.planet_mask[i],
        planet_owned_mask=feats.planet_owned_mask[i],
        planet_ids=feats.planet_ids[i],
        planet_garrison=feats.planet_garrison[i],
        fleet_feats=feats.fleet_feats[i],
        fleet_mask=feats.fleet_mask[i],
    )


def _encoded_from_args(
    planet_feats: torch.Tensor,
    planet_mask: torch.Tensor,
    planet_owned_mask: torch.Tensor,
    planet_ids: torch.Tensor,
    planet_garrison: torch.Tensor,
    fleet_feats: torch.Tensor,
    fleet_mask: torch.Tensor,
) -> EncodedObs:
    return EncodedObs(
        planet_feats=planet_feats,
        planet_mask=planet_mask,
        planet_owned_mask=planet_owned_mask,
        planet_ids=planet_ids,
        planet_garrison=planet_garrison,
        fleet_feats=fleet_feats,
        fleet_mask=fleet_mask,
    )


def _compile_kernel(
    kernel: torch.nn.Module, *, device: torch.device, compile_mode: str | None
) -> torch.nn.Module:
    """`torch.compile` the kernel on CUDA (PPO settings); identity otherwise."""
    if device.type != "cuda" or not compile_mode:
        return kernel
    return torch.compile(kernel, dynamic=False, fullgraph=True, mode=compile_mode)


def _mark_cuda_graph_step(device: torch.device) -> None:
    """Open a new CUDA-graph-trees step at the start of one learner iteration.

    The SAC update path compiles TWO kernels (`q_kernel`, `actor_kernel`) that
    SHARE the `qf1`/`qf2` modules and runs them interleaved, each with an eager
    `.backward()`. CUDA-graph trees (`reduce-overhead`) record one memory pool
    across a tree of graph executions; everything between two
    `cudagraph_mark_step_begin()` calls is ONE step with coherent buffer-liveness
    tracking. The mark must therefore bracket a *whole* learner iteration — the
    critic update, the actor burst, and the polyak — as a single step.

    Marking before *each* kernel instead (the natural-looking choice) splits the
    critic and actor into separate steps, so the tree reclaims the shared-critic
    pool slot mid-lineage and the actor backward reads a buffer a later
    critic-graph replay overwrote: "accessing tensor output of CUDAGraphs that
    has been overwritten by a subsequent run", raised inside
    `actor_loss.backward()`. One mark per iteration keeps the shared `qf1`/`qf2`
    forward activations valid through both backwards. (Empirically verified; see
    `ppo.py`, whose single-kernel loop marks once per minibatch for the same
    reason.) No-op off CUDA / pre-2.x torch.
    """
    if device.type != "cuda":
        return
    mark = getattr(torch.compiler, "cudagraph_mark_step_begin", None)
    if callable(mark):
        mark()


class _SACQKernel(torch.nn.Module):
    """Twin-Q soft-Bellman loss as one compiled graph (REAL ship-margin units).

    Resamples a'~π(·|s') (no grad), forms the per-twin real Q' = V'(s') + E_a[adv'],
    takes the pessimistic twin-min, adds the real-units entropy bonus
    `α_d·H' + α_c·H'`, and bootstraps the REAL target `y = r + γ(1−done)·V_soft'`.
    Then each twin's heads are trained with a SPLIT loss: the value head learns the
    full target `y` via HL-Gauss cross-entropy (`−Σ target_probs(y)·log_softmax(nV)`,
    decoded V clamped to `[value_min, value_max]` ⇒ bounded bootstrap), and the
    advantage heads learn the residual `(y − V)` (V detached) via a Huber loss —
    orthogonal heads, the tanh bound anchoring the V/A split (no centering). One α
    in real units balances entropy in both the actor and this bootstrap. Returns
    `(q_loss, qf1_loss, qf2_loss, q1_scalar, q2_scalar, …)` (q scalars REAL).
    """

    def __init__(
        self,
        actor: SACActor,
        qf1: SACSoftQ,
        qf2: SACSoftQ,
        qf1_target: SACSoftQ,
        qf2_target: SACSoftQ,
        *,
        gamma: float,
        autocast_enabled: bool,
    ) -> None:
        super().__init__()
        self.actor = actor
        self.qf1 = qf1
        self.qf2 = qf2
        self.qf1_target = qf1_target
        self.qf2_target = qf2_target
        self.gamma = float(gamma)
        self.autocast_enabled = bool(autocast_enabled)

    def forward(
        self,
        pf: torch.Tensor,
        pm: torch.Tensor,
        pom: torch.Tensor,
        pid: torch.Tensor,
        pg: torch.Tensor,
        ff: torch.Tensor,
        fm: torch.Tensor,
        npf: torch.Tensor,
        npm: torch.Tensor,
        npom: torch.Tensor,
        npid: torch.Tensor,
        npg: torch.Tensor,
        nff: torch.Tensor,
        nfm: torch.Tensor,
        launch: torch.Tensor,
        target_idx: torch.Tensor,
        fraction: torch.Tensor,
        legal: torch.Tensor,
        next_legal: torch.Tensor,
        reward: torch.Tensor,
        done: torch.Tensor,
        time: torch.Tensor,
        next_time: torch.Tensor,
        alpha_d: torch.Tensor,
        alpha_c: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        feats = _encoded_from_args(pf, pm, pom, pid, pg, ff, fm)
        next_feats = _encoded_from_args(npf, npm, npom, npid, npg, nff, nfm)
        gate = (feats.planet_owned_mask & feats.planet_mask).float()
        next_gate = (next_feats.planet_owned_mask & next_feats.planet_mask).float()

        with torch.no_grad():
            with torch.autocast(
                device_type="cuda", dtype=torch.bfloat16, enabled=self.autocast_enabled
            ):
                next_action = self.actor.get_action(
                    next_feats, next_legal, deterministic=False, time_feat=next_time
                )
                # Per-twin REAL next-state Q' = V'(s') + E_a[adv'](s'): decode the
                # value distribution to real ship-margin units, add the closed-form
                # expected advantage (bounded ±n_owned·adv_scale, same units). The
                # twin-min is taken on the real Q'.
                q_next_twin = []
                v_only_next_twin = []  # V' alone (no advantage), diagnostic only
                for qt in (self.qf1_target, self.qf2_target):
                    comp = qt.components(
                        next_feats, next_action.fraction, time_feat=next_time
                    )
                    v_next = qt.bins_to_scalar(comp.nV)
                    adv_next = expected_advantage(
                        comp,
                        next_action.launch_p,
                        next_action.target_probs,
                        next_gate,
                    )
                    q_next_twin.append(v_next + adv_next)
                    v_only_next_twin.append(v_next)
                hard_v_next = torch.minimum(q_next_twin[0], q_next_twin[1])
                v_only_next = torch.minimum(v_only_next_twin[0], v_only_next_twin[1])
            # Entropy bonus is now ADDED IN REAL UNITS (cleanrl's `min_q − α·log π`,
            # sign-flipped to +α·H), the SAME real-units coordinate the actor's
            # `adv + α·H` lives in and where α is tuned — one α, no symlog-Jacobian
            # decoupling. SIGNED: H_disc ≥ 0 but the continuous differential H_cont
            # can be negative for a confident low-spread fraction, so a sharply-tuned
            # policy can make the bonus slightly negative — standard max-ent.
            ent_bonus = alpha_d * next_action.H_disc + alpha_c * next_action.H_cont
            soft_v_next = hard_v_next + ent_bonus
            y = reward + (1.0 - done) * self.gamma * soft_v_next

        with torch.autocast(
            device_type="cuda", dtype=torch.bfloat16, enabled=self.autocast_enabled
        ):
            comp1 = self.qf1.components(feats, fraction, time_feat=time)
            comp2 = self.qf2.components(feats, fraction, time_feat=time)
        # Split per-twin loss against the SAME real target `y`:
        #   value head → HL-Gauss CE on the full target (decoded V bounded to the
        #     support; `target_probs` applies symlog + clamps internally), and
        #   advantage heads → Huber on the residual `(y − V)` with V DETACHED.
        # These are orthogonal projections of the same target onto V and adv (not
        # an alternating two-target drift): V owns the state baseline, the bounded
        # adv owns the action-dependent residual. The Huber tail gradient does not
        # vanish (unlike a CE through a tilt), so the tanh-bounded adv keeps a live
        # signal. Run in fp32 — the CE over symlog-spaced bins is precision-
        # sensitive and the encoder forces fp32 buffers anyway. The twins share an
        # identical HL-Gauss config, so `target_probs(y)` is the same; compute once.
        y_f = y.float()
        tp = self.qf1.hlgauss.target_probs(y_f)  # [B, num_bins]
        v1 = self.qf1.bins_to_scalar(comp1.nV).float()
        v2 = self.qf2.bins_to_scalar(comp2.nV).float()
        adv1 = taken_advantage(comp1, launch, target_idx, gate).float()
        adv2 = taken_advantage(comp2, launch, target_idx, gate).float()
        v_ce_1 = -(tp * F.log_softmax(comp1.nV.float(), dim=-1)).sum(-1).mean()
        v_ce_2 = -(tp * F.log_softmax(comp2.nV.float(), dim=-1)).sum(-1).mean()
        adv_hub_1 = F.smooth_l1_loss(adv1, (y_f - v1).detach(), beta=1.0)
        adv_hub_2 = F.smooth_l1_loss(adv2, (y_f - v2).detach(), beta=1.0)
        qf1_loss = v_ce_1 + adv_hub_1
        qf2_loss = v_ce_2 + adv_hub_2
        q_loss = qf1_loss + qf2_loss
        # Bootstrap diagnostics (real units): the hard (entropy-free) twin-min value
        # and the SIGNED entropy bonus actually injected into y (usually positive;
        # slightly negative when the continuous differential entropy is). Now that
        # the bonus is added in real units it is literally `α_d·H + α_c·H` —
        # bootstrap/entropy_bonus_frac confirms α·H is a meaningful fraction of the
        # value, not collapsed to ~0 (the near-hard critic this refactor fixes).
        boot_value = hard_v_next.mean().detach()
        boot_entropy_bonus = (soft_v_next - hard_v_next).mean().detach()
        # Explained variance of the taken-action Q vs the bootstrap target across
        # the batch: ev = 1 − Var(y − Q_taken)/Var(y). A critic that only predicts
        # the marginal (the failure mode at the old value scale) sits at ev≈0; a
        # critic that tracks state-conditional value approaches 1. This is the
        # decisive health check for the value-scale / observability fix.
        q1_full = v1 + adv1
        explained_var = (
            1.0 - (y_f - q1_full).var() / y_f.var().clamp_min(1e-8)
        ).detach()
        # Advantage EV: does the critic's ACTION-DEPENDENT part explain the
        # action-relevant residual of the target, or is `explained_var` above just
        # the critic predicting its own (action-independent) state value? adv_contrib
        # is the taken-action advantage directly (= Q_taken − V, the dueling
        # baseline carries zero policy gradient); strip the discounted next-state V
        # baseline from y to isolate the residual the advantage SHOULD predict
        # (reward + value change + the next-state entropy bonus). If the advantages
        # are inert (the suspected failure — Q≈V, entropy_bonus_frac≈0),
        # adv_contrib≈const ⇒ adv_explained_var≈0 AND adv_spread≈0 even while
        # explained_var≈1, proving the headline EV is vacuous self-prediction.
        # adv_spread (real-units std of adv_contrib) disambiguates inert (≈0) from
        # noisy-but-uncorrelated (large spread, low EV).
        adv_contrib = adv1
        y_resid = y_f - self.gamma * (1.0 - done.float()) * v_only_next.float()
        adv_explained_var = (
            1.0 - (y_resid - adv_contrib).var() / y_resid.var().clamp_min(1e-8)
        ).detach()
        adv_spread = adv_contrib.std().detach()
        return (
            q_loss,
            qf1_loss.detach(),
            qf2_loss.detach(),
            q1_full.mean().detach(),  # REAL value units
            (v2 + adv2).mean().detach(),
            boot_value,
            boot_entropy_bonus,
            explained_var,
            adv_explained_var,
            adv_spread,
        )


class _SACActorKernel(torch.nn.Module):
    """Closed-form actor objective as one compiled graph.

    Resamples a~π(·|s); assembles the per-twin closed-form expected SCALAR
    advantage `E_a[adv_j]` (A0 detached as a baseline, AL live for the pathwise
    fraction grad, launch_p/target_probs live for the discrete grad); the loss is
    `-(min_j E_a[adv_j] + α_d·H_disc + α_c·H_cont)`. The actor ascends the scalar
    advantage DIRECTLY — the state value V is action-independent (zero policy
    gradient), so only adv carries the gradient. adv is now in REAL ship-margin
    units (the heads are tanh-bounded), the SAME units as the entropy bonus and the
    bootstrap, so a single global α (per discrete/continuous) balances value vs
    entropy identically in the actor and the critic target. Returns the actor loss
    plus the detached entropy bookkeeping the eager dual-alpha step consumes.
    """

    def __init__(
        self,
        actor: SACActor,
        qf1: SACSoftQ,
        qf2: SACSoftQ,
        *,
        autocast_enabled: bool,
    ) -> None:
        super().__init__()
        self.actor = actor
        self.qf1 = qf1
        self.qf2 = qf2
        self.autocast_enabled = bool(autocast_enabled)

    def forward(
        self,
        pf: torch.Tensor,
        pm: torch.Tensor,
        pom: torch.Tensor,
        pid: torch.Tensor,
        pg: torch.Tensor,
        ff: torch.Tensor,
        fm: torch.Tensor,
        legal: torch.Tensor,
        time: torch.Tensor,
        alpha_d: torch.Tensor,
        alpha_c: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        feats = _encoded_from_args(pf, pm, pom, pid, pg, ff, fm)
        gate = (feats.planet_owned_mask & feats.planet_mask).float()

        with torch.autocast(
            device_type="cuda", dtype=torch.bfloat16, enabled=self.autocast_enabled
        ):
            action = self.actor.get_action(
                feats, legal, deterministic=False, time_feat=time
            )
            adv_per_twin = []
            for qf in (self.qf1, self.qf2):
                comp = qf.components(feats, action.fraction, time_feat=time)
                # nA0 detached (no-launch baseline), nAL live (pathwise fraction
                # grad). nV is unused by the scalar advantage — the actor ascends
                # `adv`, the action-dependent part of Q; V is the fixed baseline.
                comp_actor = QComponents(
                    nV=comp.nV.detach(), nA0=comp.nA0.detach(), nAL=comp.nAL
                )
                adv_per_twin.append(
                    expected_advantage(
                        comp_actor, action.launch_p, action.target_probs, gate
                    )
                )
        # Pessimistic over twins on the SCALAR advantage (the action-dependent part
        # of Q; the V baseline contributes no policy gradient). adv is in real
        # ship-margin units (tanh-bounded heads), the same units as α·H, so a single
        # global α balances the value/entropy trade-off everywhere.
        adv_min = torch.minimum(adv_per_twin[0], adv_per_twin[1])
        soft_v = adv_min + alpha_d * action.H_disc + alpha_c * action.H_cont
        actor_loss = -soft_v.mean()
        return (
            actor_loss,
            action.H_disc.detach(),
            action.h_disc_max.detach(),
            action.cont_logp_sum.detach(),
            action.n_owned.detach(),
            action.eff_cont_dim.detach(),
        )


def _clip_optimizer_grads(
    optimizer: torch.optim.Optimizer, grad_clip: float
) -> torch.Tensor:
    """Clip the optimizer's grad-norm to `grad_clip` (no clip when ≤0) and return
    the PRE-clip total L2 norm.

    The norm is returned as a GPU scalar and NOT materialized — the caller
    `.item()`s it (with the rest of the metrics) only on logging ticks, so the
    grad-norm logging costs no per-update CUDA sync. `max_norm=inf` measures
    without rescaling (clip coef clamps to 1.0).
    """
    params = [p for group in optimizer.param_groups for p in group["params"]]
    max_norm = grad_clip if grad_clip > 0.0 else float("inf")
    return torch.nn.utils.clip_grad_norm_(params, max_norm)


@dataclass
class _QStats:
    """Critic-step metrics as un-materialized GPU scalars (`.item()` at log time).

    Tensor fields (not floats) so the hot loop never CUDA-syncs. `_q_step`
    `.clone()`s the cudagraph-output fields before wrapping them here, so these
    references survive arbitrary later kernel replays (see `_q_step`).
    """

    qf1_loss: torch.Tensor
    qf2_loss: torch.Tensor
    qf1_value: torch.Tensor
    qf2_value: torch.Tensor
    grad_norm: torch.Tensor
    boot_value: torch.Tensor          # hard (entropy-free) bootstrap value, real units
    boot_entropy_bonus: torch.Tensor  # entropy contribution to soft_v_next, real units
    explained_var: torch.Tensor       # EV of taken-Q vs bootstrap target (critic health)
    adv_explained_var: torch.Tensor   # EV of the action-dependent advantage vs target residual
    adv_spread: torch.Tensor          # std of the decoded advantage contribution (real units)


@dataclass
class _ActorStats:
    """Actor/alpha-step metrics as un-materialized GPU scalars (`.item()` at log time)."""

    actor_loss: torch.Tensor
    alpha_disc_loss: torch.Tensor
    alpha_cont_loss: torch.Tensor
    grad_norm: torch.Tensor
    h_disc: torch.Tensor             # achieved discrete entropy, summed over owned (batch mean)
    h_disc_per_planet: torch.Tensor  # achieved per-owned-planet discrete entropy (α_disc target quantity)
    target_h_disc: torch.Tensor      # per-owned-planet target entropy α_disc drives toward
    cont_logp: torch.Tensor          # p-weighted Σ g·p·logp_frac (= -H_cont)
    target_h_cont: torch.Tensor      # launch-mass-scaled target entropy α_cont drives toward


def _q_step(
    *,
    q_kernel: torch.nn.Module,
    q_optimizer: torch.optim.Optimizer,
    log_alpha_disc: torch.Tensor,
    log_alpha_cont: torch.Tensor,
    batch: SACBatch,
    grad_clip: float,
) -> _QStats:
    """Twin-Q TD update against the soft Bellman target (factored critic).

      Q'_j    = bins_to_scalar(nV_j(s')) + E_a[adv_j](s')      # REAL units
      soft_V' = min_j Q'_j + α_d·H' + α_c·H'                   # real-units bonus
      y       = r + (1-d)·γ·soft_V'                            # REAL soft target
      loss_j  = CE( target_probs(y), nV_j(s) )                 # value head
              + Huber( adv_j(s, a_buffer), sg[y − V_j(s)] )    # advantage head

    The entropy bonus enters in REAL units (cleanrl's `min_q + α·H`), the same
    units as the actor's `adv + α·H`, so α is "value-per-nat" consistently in both
    the critic bootstrap and the actor objective — a single, well-scaled α. The
    forward+loss runs inside the compiled `q_kernel` (autocast → FA-2); the
    gradient clip and optimizer step stay eager. The value head is DISTRIBUTIONAL
    (HL-Gauss two-hot over symlog bins, decoded V clamped to
    `[value_min, value_max]` ⇒ bounded bootstrap); the advantage heads are scalar,
    tanh-bounded, and fit the residual `(y − V)` with V detached.
    """
    alpha_d = log_alpha_disc.exp().detach()
    alpha_c = log_alpha_cont.exp().detach()
    (
        q_loss,
        qf1_loss,
        qf2_loss,
        q1_scalar,
        q2_scalar,
        boot_value,
        boot_ent,
        explained_var,
        adv_explained_var,
        adv_spread,
    ) = q_kernel(
        *_encoded_args(batch.feats),
        *_encoded_args(batch.next_feats),
        batch.launch,
        batch.target_idx,
        batch.fraction,
        batch.target_legal_mask,
        batch.next_target_legal_mask,
        batch.reward,
        batch.done,
        batch.time,
        batch.next_time,
        alpha_d,
        alpha_c,
    )

    q_optimizer.zero_grad(set_to_none=True)
    q_loss.backward()
    grad_norm = _clip_optimizer_grads(q_optimizer, grad_clip)
    q_optimizer.step()
    # Return un-materialized GPU scalars; the caller `.item()`s on logging ticks
    # only, so the hot loop never CUDA-syncs. The kernel outputs alias the
    # cudagraph static output buffers (reduce-overhead) and the caller HOLDS the
    # latest stats across subsequent iterations (each a new cudagraph step that
    # overwrites those buffers), so `.clone()` them — PyTorch's documented remedy
    # for retained cudagraph outputs. A 0-dim clone is async (no CPU sync).
    # grad_norm is an eager (non-cudagraph) tensor, already safe to hold.
    return _QStats(
        qf1_loss=qf1_loss.clone(),
        qf2_loss=qf2_loss.clone(),
        qf1_value=q1_scalar.clone(),
        qf2_value=q2_scalar.clone(),
        grad_norm=grad_norm,
        boot_value=boot_value.clone(),
        boot_entropy_bonus=boot_ent.clone(),
        explained_var=explained_var.clone(),
        adv_explained_var=adv_explained_var.clone(),
        adv_spread=adv_spread.clone(),
    )


def _actor_alpha_step(
    *,
    actor_kernel: torch.nn.Module,
    actor_optimizer: torch.optim.Optimizer,
    alpha_disc_optimizer: torch.optim.Optimizer,
    alpha_cont_optimizer: torch.optim.Optimizer,
    log_alpha_disc: torch.Tensor,
    log_alpha_cont: torch.Tensor,
    disc_target_entropy_ratio: float,
    target_entropy_per_dim: float,
    batch: SACBatch,
    grad_clip: float,
) -> _ActorStats:
    """Actor + dual-alpha update (closed-form, no REINFORCE).

    The closed-form soft-value objective (with the detach routing — nV/nA0
    detached, nAL live for the pathwise fraction grad, launch_p/target_probs live
    for the exact discrete grad) runs inside the compiled `actor_kernel`
    (autocast → FA-2). The actor optimizer step and the two scalar entropy-
    temperature updates stay eager.

    Returns un-materialized GPU scalars (`_ActorStats`); the caller `.item()`s
    them on logging ticks only.
    """
    alpha_d = log_alpha_disc.exp().detach()
    alpha_c = log_alpha_cont.exp().detach()
    (
        actor_loss,
        h_disc_d,
        h_disc_max_d,
        cont_logp_d,
        n_owned,
        eff_cont_dim,
    ) = actor_kernel(
        *_encoded_args(batch.feats),
        batch.target_legal_mask,
        batch.time,
        alpha_d,
        alpha_c,
    )

    actor_optimizer.zero_grad(set_to_none=True)
    actor_loss.backward()
    actor_grad_norm = _clip_optimizer_grads(actor_optimizer, grad_clip)
    actor_optimizer.step()

    # ---- dual-alpha updates (per-state targets; kernel pre-detached all) ----
    denom = n_owned.clamp_min(1.0)

    # Discrete: tune on the PER-PLANET average entropy so the target does not
    # scale with the owned-planet count (the old summed target ran α to 1.6).
    # cleanrl `sac_atari.py` form `Σ_a p_a·(-α·(logp_a + H_target)) = α·(H - H_t)`
    # with α = exp(log_α) LIVE (here H, H_t are the achieved / target entropies
    # already expectation-weighted in the kernel). The α-weighting self-damps,
    # so α_disc equilibrates at achieved == target (a reachable fraction of max)
    # by policy feedback — the old `log_α·(...)` form needed a ceiling clamp to
    # stop upward runaway; the atari form doesn't.
    achieved_disc = h_disc_d / denom  # avg per-owned-planet discrete entropy
    target_disc = disc_target_entropy_ratio * (h_disc_max_d / denom)
    alpha_disc_loss = (
        log_alpha_disc.exp() * (achieved_disc - target_disc)
    ).mean()  # ↑α when achieved < target
    alpha_disc_optimizer.zero_grad(set_to_none=True)
    alpha_disc_loss.backward()
    alpha_disc_optimizer.step()

    # Continuous: cont_logp is p-weighted (Σ g·p·logp_frac), so the target must
    # scale by the expected launch count Σ g·p — matching that weighting. Scaling
    # by the full n_owned instead drives α_cont → 0 whenever launches are rare
    # (the conservative early policy), starving the fraction of entropy pressure.
    #
    # Use cleanrl's α-weighted loss `-(α·(logp + target))` with α = exp(log_α)
    # LIVE (not `-(log_α·(...))`): its d/d(log_α) = -α·residual self-damps as α
    # shrinks, so a transiently-negative residual (the bounded fraction's natural
    # entropy can exceed the target while the cold critic hasn't concentrated it
    # yet) no longer collapses α_cont exponentially — it recovers once the critic
    # pushes logp past the target. This is the Haarnoja (2018) dual objective.
    h_target_frac = target_entropy_per_dim * eff_cont_dim  # [B]
    alpha_cont_loss = -(log_alpha_cont.exp() * (cont_logp_d + h_target_frac)).mean()
    alpha_cont_optimizer.zero_grad(set_to_none=True)
    alpha_cont_loss.backward()
    alpha_cont_optimizer.step()

    # Un-materialized GPU scalars: achieved/target entropies ride along for the
    # achieved-vs-target α diagnostic. `actor_loss` aliases a cudagraph output
    # buffer (reduce-overhead) and the caller holds it across later iterations
    # that overwrite the buffer, so `.clone()` it (0-dim clone is async, no CPU
    # sync). Every other field is a fresh eager tensor (`.mean()` / eager
    # arithmetic / grad-norm), already safe to hold.
    return _ActorStats(
        actor_loss=actor_loss.detach().clone(),
        alpha_disc_loss=alpha_disc_loss.detach(),
        alpha_cont_loss=alpha_cont_loss.detach(),
        grad_norm=actor_grad_norm,
        h_disc=h_disc_d.mean(),
        h_disc_per_planet=achieved_disc.mean(),
        target_h_disc=target_disc.mean(),
        cont_logp=cont_logp_d.mean(),
        target_h_cont=h_target_frac.mean(),
    )


def _target_step(
    qf1: SACSoftQ,
    qf2: SACSoftQ,
    qf1_target: SACSoftQ,
    qf2_target: SACSoftQ,
    tau: float,
) -> None:
    polyak_update(qf1_target, qf1, tau)
    polyak_update(qf2_target, qf2, tau)


def _build_update_output(
    q: _QStats,
    a: _ActorStats | None,
    alpha_disc: float,
    alpha_cont: float,
) -> SACUpdateOutput:
    """Materialize the held GPU-scalar stats to CPU floats — the ONLY place that
    `.item()`s update metrics, and called only on logging ticks (so the hot loop
    never CUDA-syncs). The held tensors carry the last update's values."""
    out = SACUpdateOutput(
        qf1_loss=q.qf1_loss.item(),
        qf2_loss=q.qf2_loss.item(),
        qf1_value=q.qf1_value.item(),
        qf2_value=q.qf2_value.item(),
        qf_grad_norm=q.grad_norm.item(),
        boot_value=q.boot_value.item(),
        boot_entropy_bonus=q.boot_entropy_bonus.item(),
        explained_var=q.explained_var.item(),
        adv_explained_var=q.adv_explained_var.item(),
        adv_spread=q.adv_spread.item(),
        alpha_disc=alpha_disc,
        alpha_cont=alpha_cont,
        actor_loss=None,
        actor_grad_norm=None,
        alpha_disc_loss=None,
        alpha_cont_loss=None,
        h_disc_mean=None,
        h_disc_per_planet=None,
        target_h_disc=None,
        logp_frac_mean=None,
        target_h_cont=None,
    )
    if a is not None:
        out.actor_loss = a.actor_loss.item()
        out.actor_grad_norm = a.grad_norm.item()
        out.alpha_disc_loss = a.alpha_disc_loss.item()
        out.alpha_cont_loss = a.alpha_cont_loss.item()
        out.h_disc_mean = a.h_disc.item()
        out.h_disc_per_planet = a.h_disc_per_planet.item()
        out.target_h_disc = a.target_h_disc.item()
        out.logp_frac_mean = a.cont_logp.item()
        out.target_h_cont = a.target_h_cont.item()
    return out


# ---------------------------------------------------------------------------
# Main training entry point
# ---------------------------------------------------------------------------


@dataclass
class SACState:
    """Lightweight container for everything the trainer keeps across steps."""

    actor: SACActor
    qf1: SACSoftQ
    qf2: SACSoftQ
    qf1_target: SACSoftQ
    qf2_target: SACSoftQ
    q_kernel: torch.nn.Module
    actor_kernel: torch.nn.Module
    compile_mode: str | None  # None ⇒ eager kernels (skip cudagraph step marks)
    q_optimizer: torch.optim.Optimizer
    actor_optimizer: torch.optim.Optimizer
    log_alpha_disc: torch.Tensor
    log_alpha_cont: torch.Tensor
    alpha_disc_optimizer: torch.optim.Optimizer
    alpha_cont_optimizer: torch.optim.Optimizer
    rng: random.Random
    pool: OpponentPool
    writer: SummaryWriter
    ckpt_dir: Path
    disc_target_entropy_ratio: float
    target_entropy_per_dim: float
    builtin_agents: dict[str, Any]
    builtin_prob: float
    # Training horizon — recorded into checkpoints so the agent replays with the
    # same step/episode_steps the encoder FiLM was conditioned on.
    episode_steps: int


def _build_policy_cfg(cfg: RunConfig) -> OrbitPolicyConfig:
    m = cfg.model
    pcfg = OrbitPolicyConfig(
        dim=m.dim,
        ff_dim=m.ff_dim,
        depth=m.depth,
        n_heads=m.n_heads,
        n_kv_heads=m.n_kv_heads,
        dropout=m.dropout,
        planet_rope_fraction=m.planet_rope_fraction,
        planet_rope_base=m.planet_rope_base,
        encoder_backend=m.encoder_backend,
        num_fleet_latents=m.num_fleet_latents,
        fleet_tokenizer_depth=m.fleet_tokenizer_depth,
        value_hidden=m.value_hidden,
        value_num_bins=m.value_num_bins,
        value_min=m.value_min,
        value_max=m.value_max,
        value_symlog=m.value_symlog,
        action_logit_softcap=m.action_logit_softcap,
        adv_scale=m.adv_scale,
    )
    return pcfg


def _builtin_opponent_slate(cfg: RunConfig) -> tuple[list[str], float]:
    """Return SAC builtin opponent names and sampling probability."""
    if cfg.opponents.mode == "fixed":
        return list(cfg.opponents.fixed_opponents), 1.0
    return list(cfg.sac.builtin_opponents), cfg.sac.builtin_prob


def _make_vec_env(cfg: RunConfig) -> Any:
    vec_kwargs = dict(
        num_envs=max(1, cfg.rollout.num_envs),
        num_players=cfg.game.num_players,
        episode_steps=cfg.game.episode_steps,
        ship_speed=cfg.game.ship_speed,
    )
    backend = cfg.rollout.env_backend
    if backend == "numpy":
        return NumpyVecEnv(
            **vec_kwargs,
            replay_env_idx=None,
            random_seed=cfg.run.seed,
        )
    if backend == "numpy_mp":
        return ShardedNumpyVecEnv(
            **vec_kwargs,
            replay_env_idx=None,
            num_workers=cfg.rollout.num_workers,
            random_seed=cfg.run.seed,
        )
    if backend == "rust":
        from .rust_env import RustVecEnv

        return RustVecEnv(
            **vec_kwargs,
            replay_env_idx=None,
            random_seed=cfg.run.seed,
        )
    if backend == "kaggle":
        return VecEnv(**vec_kwargs, replay_env_idx=None)
    raise ValueError(f"unknown rollout.env_backend: {backend!r}")


def _observations_for_rows(
    vec: Any,
    states: list[Any],
    rows: list[tuple[int, int]],
) -> list[Any]:
    observations = getattr(vec, "observations", None)
    if callable(observations):
        return observations(rows)
    return [states[env_idx][seat]["observation"] for env_idx, seat in rows]


def _encoded_to(feats: EncodedObs, device: torch.device | str) -> EncodedObs:
    return EncodedObs(
        planet_feats=feats.planet_feats.to(device),
        planet_mask=feats.planet_mask.to(device),
        planet_owned_mask=feats.planet_owned_mask.to(device),
        planet_ids=feats.planet_ids.to(device),
        planet_garrison=feats.planet_garrison.to(device),
        fleet_feats=feats.fleet_feats.to(device),
        fleet_mask=feats.fleet_mask.to(device),
    )


def _policy_inputs_for_rows(
    vec: Any,
    states: list[Any],
    rows: list[tuple[int, int]],
    *,
    episode_steps: int,
    device: torch.device,
) -> tuple[list[Any], EncodedObs, list[Any] | None, torch.Tensor]:
    obs = _observations_for_rows(vec, states, rows)
    policy_batch = getattr(vec, "policy_batch", None)
    if callable(policy_batch):
        enc, contexts = policy_batch(rows, device=str(device), pin_memory=True)
        enc = _encoded_to(enc, device)
        return obs, enc, contexts, _time_feat_from_obs(obs, episode_steps, device)
    enc = encode_raw_observations(obs, device=device)
    return obs, enc, None, _time_feat_from_obs(obs, episode_steps, device)


def _record_batch_to_list(records: SampleBatchRecord) -> list[SampleRecord]:
    out: list[SampleRecord] = []
    for i in range(records.launch.shape[0]):
        out.append(
            SampleRecord(
                launch=records.launch[i],
                raw_launch=records.raw_launch[i],
                target_idx=records.target_idx[i],
                fraction=records.fraction[i],
                log_prob=records.log_prob[i],
                target_legal_mask=records.target_legal_mask[i],
            )
        )
    return out


def _records_to_cpu_list(
    records: list[SampleRecord] | SampleBatchRecord,
) -> list[SampleRecord]:
    if isinstance(records, SampleBatchRecord):
        records = _record_batch_to_list(records)
    return [record_to_cpu(r) for r in records]


def _sample_actions_with_records(
    vec: Any,
    out: Any,
    rows: list[tuple[int, int]],
    obs: list[Any],
    contexts: list[Any] | None,
    *,
    deterministic: bool = False,
) -> tuple[list[Any], list[SampleRecord]]:
    native_sampler = getattr(vec, "sample_batch_with_records", None)
    if callable(native_sampler):
        actions, records = native_sampler(out, rows, deterministic=deterministic)
        return actions, _records_to_cpu_list(records)
    if contexts is not None:
        actions, records = sample_batch_with_records_context(
            out, contexts, deterministic=deterministic
        )
        return actions, _records_to_cpu_list(records)
    actions, records = sample_batch_with_records_raw(
        out, obs, deterministic=deterministic
    )
    return actions, _records_to_cpu_list(records)


def _build_state(cfg: RunConfig, device: torch.device) -> SACState:
    pcfg = _build_policy_cfg(cfg)
    sac = cfg.sac

    actor = SACActor(
        pcfg,
        log_std_min=sac.log_std_min,
        log_std_max=sac.log_std_max,
    ).to(device)
    qf1 = SACSoftQ(pcfg).to(device)
    qf2 = SACSoftQ(pcfg).to(device)
    qf1_target = make_targets(qf1).to(device)
    qf2_target = make_targets(qf2).to(device)

    # Compiled update kernels (autocast → FA-2; inductor fusion + CUDA-graphs).
    # `compile_mode` is the shared RunConfig knob PPO uses; on CPU it's a no-op
    # identity. Both kernels keep the configured cudagraph mode; the interleaved
    # shared-qf1/qf2 backward is made cudagraph-safe by marking ONE step per
    # learner iteration in `_run_updates` (see `_mark_cuda_graph_step`).
    autocast_enabled = device.type == "cuda"
    compile_mode = cfg.run.compile_mode or None
    q_kernel = _compile_kernel(
        _SACQKernel(
            actor,
            qf1,
            qf2,
            qf1_target,
            qf2_target,
            gamma=sac.gamma,
            autocast_enabled=autocast_enabled,
        ),
        device=device,
        compile_mode=compile_mode,
    )
    actor_kernel = _compile_kernel(
        _SACActorKernel(actor, qf1, qf2, autocast_enabled=autocast_enabled),
        device=device,
        compile_mode=compile_mode,
    )

    q_optimizer = torch.optim.Adam(
        list(qf1.parameters()) + list(qf2.parameters()),
        lr=sac.q_lr,
        weight_decay=sac.weight_decay,
    )
    actor_optimizer = torch.optim.Adam(
        actor.parameters(),
        lr=sac.policy_lr,
        weight_decay=sac.weight_decay,
    )

    # Two independent entropy temperatures: discrete (launch+target) and
    # continuous (fraction). Each has its own log_alpha scalar + Adam.
    log_alpha_disc = torch.tensor(
        math.log(max(1e-8, sac.alpha_discrete)),
        device=device,
        dtype=torch.float32,
        requires_grad=True,
    )
    log_alpha_cont = torch.tensor(
        math.log(max(1e-8, sac.alpha_continuous)),
        device=device,
        dtype=torch.float32,
        requires_grad=True,
    )
    alpha_disc_optimizer = torch.optim.Adam(
        [log_alpha_disc], lr=sac.alpha_discrete_lr
    )
    alpha_cont_optimizer = torch.optim.Adam(
        [log_alpha_cont], lr=sac.alpha_continuous_lr
    )

    run_dir = Path(cfg.run.log_root) / cfg.run.name / time.strftime("%Y%m%d-%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(run_dir.as_posix())
    ckpt_dir = Path(cfg.run.ckpt_root) / cfg.run.name
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    elo = EloTracker(
        initial_rating=cfg.opponents.initial_rating,
        k_factor=cfg.opponents.k_factor,
    )
    elo.set(LEARNER_NAME, cfg.opponents.initial_rating)
    builtin_names, builtin_prob = _builtin_opponent_slate(cfg)
    builtin_agents = {name: BUILTIN[name] for name in builtin_names}
    for name in builtin_agents:
        elo.set(name, cfg.opponents.initial_rating)
    pool_device = (
        cfg.run.device
        if cfg.opponents.snapshot_device == "train"
        else cfg.opponents.snapshot_device
    )
    # SAC keeps no frozen opponent snapshots. In league mode the opponent seat
    # is either the current actor or a mixed-in builtin; in fixed mode
    # `builtin_prob=1.0` forces every rematch to use a static builtin. The pool
    # is kept purely for Elo bookkeeping symmetry with the PPO trainer.
    pool = OpponentPool(
        elo=elo,
        top_k=cfg.opponents.top_k,
        self_play_prob=cfg.opponents.self_play_prob,
        device=pool_device,
    )

    rng = random.Random(cfg.run.seed)
    return SACState(
        actor=actor,
        qf1=qf1,
        qf2=qf2,
        qf1_target=qf1_target,
        qf2_target=qf2_target,
        q_kernel=q_kernel,
        actor_kernel=actor_kernel,
        compile_mode=compile_mode,
        q_optimizer=q_optimizer,
        actor_optimizer=actor_optimizer,
        log_alpha_disc=log_alpha_disc,
        log_alpha_cont=log_alpha_cont,
        alpha_disc_optimizer=alpha_disc_optimizer,
        alpha_cont_optimizer=alpha_cont_optimizer,
        rng=rng,
        pool=pool,
        writer=writer,
        ckpt_dir=ckpt_dir,
        disc_target_entropy_ratio=sac.disc_target_entropy_ratio,
        target_entropy_per_dim=sac.target_entropy_per_dim,
        builtin_agents=builtin_agents,
        builtin_prob=builtin_prob,
        episode_steps=cfg.game.episode_steps,
    )


def _save_checkpoint(state: SACState, step: int, *, label: str = "latest") -> Path:
    path = state.ckpt_dir / f"sac_{label}.pt"
    payload: dict[str, Any] = {
        "step": step,
        "actor": state.actor.state_dict(),
        "qf1": state.qf1.state_dict(),
        "qf2": state.qf2.state_dict(),
        "qf1_target": state.qf1_target.state_dict(),
        "qf2_target": state.qf2_target.state_dict(),
        "actor_cfg": state.actor.cfg.to_dict(),
        "log_std_min": state.actor.log_std_min,
        "log_std_max": state.actor.log_std_max,
        "log_alpha_disc": state.log_alpha_disc.detach().cpu(),
        "log_alpha_cont": state.log_alpha_cont.detach().cpu(),
        # The encoder FiLM conditions on step/episode_steps, so the agent must
        # replay with the SAME horizon it trained on; ship it in the checkpoint
        # rather than relying on the SACAgent default.
        "episode_steps": state.episode_steps,
    }
    torch.save(payload, path)
    return path


def _select_opponent_for_episode(state: SACState) -> tuple[str, Any]:
    """Pick one opponent for the upcoming 2-player episode.

    With probability `builtin_prob` (and when any builtins are configured) the
    seat is a uniformly-chosen builtin baseline; otherwise it's self-play
    (`(LEARNER_NAME, None)`, the loop drives the live learner actor on that
    seat). Fixed-opponent mode is represented by `builtin_prob=1.0`.
    """
    if state.builtin_agents and state.rng.random() < state.builtin_prob:
        name = state.rng.choice(list(state.builtin_agents.keys()))
        return name, state.builtin_agents[name]
    return LEARNER_NAME, None


def _run_updates(
    state: SACState,
    prefetcher: "_ReplayPrefetcher",
    sac: Any,
    device: torch.device,
    *,
    num_envs: int,
    learn_step: int,
    tick: int,
    global_step: int,
    start_time: float,
    critic_updates: int | None = None,
) -> int:
    """One learning phase, interleaved exactly like cleanrl SAC.

    Each tick collected `num_envs` transitions, so we run `num_envs *
    gradient_steps` critic updates (UTD = `gradient_steps` updates per collected
    transition; `gradient_steps=1` ⇒ cleanrl's UTD=1). A PERSISTENT `learn_step`
    counter drives the cadence so it is continuous across ticks — every
    `policy_frequency`-th critic update runs `policy_frequency` delayed
    actor+alpha updates (cleanrl's compensation loop ⇒ net 1:1 actor:critic),
    and every `target_network_frequency`-th critic update runs the target
    polyak. Returns the advanced `learn_step`.

    Performance: the step functions return un-materialized GPU scalars; we only
    keep the LATEST `_QStats`/`_ActorStats` and `.item()` them (in
    `_build_update_output`) on logging ticks. So the hot loop does ZERO GPU→CPU
    syncs — the logged value is the tick's last update (a snapshot, standard for
    SAC dashboards). The step functions `.clone()` their cudagraph-output scalars
    before returning, so the held stats are safe to read after later replays.

    Minibatches come from `prefetcher.next()`, which keeps one minibatch's
    pinned, async H2D copy in flight on a side stream so it overlaps the previous
    update's compute (the CPU-resident replay never lives on the GPU).
    """
    q_stats: _QStats | None = None
    a_stats: _ActorStats | None = None

    n_critic = (
        max(0, int(critic_updates))
        if critic_updates is not None
        else max(0, int(round(num_envs * float(sac.gradient_steps))))
    )
    for _ in range(n_critic):
        # One CUDA-graph step per learner iteration: brackets the critic update,
        # the actor burst, and the polyak as a single step so the shared qf1/qf2
        # forward activations stay valid through both backwards (see
        # `_mark_cuda_graph_step` — marking per-kernel instead crashes). Gated on
        # compile_mode like PPO: eager kernels have no cudagraph to mark.
        if state.compile_mode is not None:
            _mark_cuda_graph_step(device)
        learn_step += 1
        batch = prefetcher.next()
        q_stats = _q_step(
            q_kernel=state.q_kernel,
            q_optimizer=state.q_optimizer,
            log_alpha_disc=state.log_alpha_disc,
            log_alpha_cont=state.log_alpha_cont,
            batch=batch,
            grad_clip=sac.grad_clip,
        )

        # Delayed actor+alpha: every policy_frequency-th critic step, run
        # policy_frequency updates (cleanrl compensation ⇒ net 1:1 actor:critic).
        if learn_step % sac.policy_frequency == 0:
            for _ in range(sac.policy_frequency):
                batch = prefetcher.next()
                a_stats = _actor_alpha_step(
                    actor_kernel=state.actor_kernel,
                    actor_optimizer=state.actor_optimizer,
                    alpha_disc_optimizer=state.alpha_disc_optimizer,
                    alpha_cont_optimizer=state.alpha_cont_optimizer,
                    log_alpha_disc=state.log_alpha_disc,
                    log_alpha_cont=state.log_alpha_cont,
                    disc_target_entropy_ratio=state.disc_target_entropy_ratio,
                    target_entropy_per_dim=state.target_entropy_per_dim,
                    batch=batch,
                    grad_clip=sac.grad_clip,
                )

        # Target polyak per cleanrl cadence (once every target_network_frequency
        # critic updates, NOT once per tick).
        if learn_step % sac.target_network_frequency == 0:
            _target_step(
                state.qf1, state.qf2, state.qf1_target, state.qf2_target, sac.tau
            )

    # Materialize metrics to CPU only on logging ticks (the one place we sync).
    if tick % sac.log_metrics_every == 0 and q_stats is not None:
        metrics = _build_update_output(
            q_stats,
            a_stats,
            float(state.log_alpha_disc.exp().item()),
            float(state.log_alpha_cont.exp().item()),
        )
        _log_metrics(state.writer, metrics, global_step, start_time)

    return learn_step


def train(cfg: RunConfig) -> None:
    sac = cfg.sac
    if cfg.game.num_players != 2:
        raise NotImplementedError(
            "SAC test branch supports 2-player games only; got num_players="
            f"{cfg.game.num_players}"
        )
    device = torch.device(cfg.run.device)
    if device.type == "cuda":
        # TF32 tensor cores for the residual *float32* matmuls only — the value
        # decode / critic-head math that runs OUTSIDE autocast. The encoder and
        # attention run under bf16 autocast (SDPA → FlashAttention-2), so they
        # stay bf16 and are untouched by this; it just upgrades the leftover
        # fp32 GEMMs from full precision to TF32 (the warning's recommendation).
        torch.set_float32_matmul_precision("high")
    if cfg.run.torch_num_threads > 0:
        torch.set_num_threads(cfg.run.torch_num_threads)
    torch.manual_seed(cfg.run.seed)

    state = _build_state(cfg, device)
    pcfg = _build_policy_cfg(cfg)
    # CPU-resident buffer (the standard SAC pattern): keeps the large replay off
    # the GPU (system RAM scales further than VRAM), and each sampled minibatch
    # is moved to the train device per update in `ReplayBuffer.sample`.
    replay = ReplayBuffer(
        sac.buffer_size,
        planet_features=pcfg.planet_features,
        fleet_features=pcfg.fleet_features,
        device="cpu",
    )

    num_players = cfg.game.num_players
    num_envs = max(1, cfg.rollout.num_envs)
    total_env_steps = int(cfg.run.total_updates)  # env-step (transition) budget

    # Compiled + autocast(bf16) actor-heads kernel for batched rollout inference
    # (FA-2 via SDPA on CUDA; eager identity on CPU). Same knob PPO uses.
    compile_mode = cfg.run.compile_mode or None
    heads_kernel = get_sac_heads_kernel(
        state.actor, device=device, compile_mode=compile_mode
    )

    def production_margin(obs: Any, seat: int) -> float:
        return _obs_production_margin(obs, seat, num_players)

    # Per-env episode bookkeeping. Seats start alternated across envs so the
    # learner sees both initial conditions from step 0 (symmetry paranoia).
    vec = _make_vec_env(cfg)
    fast_step_subset = getattr(vec, "step_subset_fast", None)
    step_subset = fast_step_subset if callable(fast_step_subset) else vec.step_subset
    start_time = time.time()
    games_vs: dict[str, int] = defaultdict(int)
    wins_vs: dict[str, float] = defaultdict(float)
    margin_vs: dict[str, float] = defaultdict(float)

    try:
        states = vec.reset()
        learner_seat = [e % num_players for e in range(num_envs)]
        opp_seat = [(s + 1) % num_players for s in learner_seat]
        opponents = [_select_opponent_for_episode(state) for _ in range(num_envs)]
        learner_rows = [(e, learner_seat[e]) for e in range(num_envs)]
        learner_obs = _observations_for_rows(vec, states, learner_rows)
        previous_prod_margin = [
            production_margin(obs, seat)
            for obs, (_env_idx, seat) in zip(learner_obs, learner_rows, strict=True)
        ]
        episode_return = [0.0] * num_envs
        episode_length = [0] * num_envs

        global_step = 0
        tick = 0
        learn_step = 0  # persistent critic-update counter (drives cleanrl cadence)
        update_credit = 0.0
        last_snapshot_step = 0
        last_latest_step = 0
        # Async minibatch prefetcher (pinned, side-stream, overlapped H2D). Built
        # lazily on the first learn tick — `sample` needs len(replay) >=
        # batch_size, which is guaranteed by the same gate that runs updates.
        prefetcher: _ReplayPrefetcher | None = None
        # Materialization-gap diagnostic (defect C): running fraction of raw
        # policy launches the env no-ops (no move could be built). The replay
        # stores the raw launch, so the critic learns the true no-op value of
        # these; this tracks how big that correction is.
        materialize_gap_sum = 0.0
        materialize_raw_sum = 0.0
        materialize_log_tick = 0

        while global_step < total_env_steps:
            warmup = global_step < sac.learning_starts

            # ---- act: one batched forward per identity (learner / opponent) ----
            # Each forward's PolicyOutput may alias cudagraph buffers, so we
            # sample + materialize records to CPU before the next forward. Both
            # warmup and learned paths share `sampling.py`'s lead-solved geometry
            # so stored (launch, target_idx, fraction, mask) and executed moves
            # always agree.
            learner_rows = [(e, learner_seat[e]) for e in range(num_envs)]
            learner_obs, learner_enc, learner_contexts, learner_time = (
                _policy_inputs_for_rows(
                    vec,
                    states,
                    learner_rows,
                    episode_steps=cfg.game.episode_steps,
                    device=device,
                )
            )
            learner_out = run_sac_heads(
                heads_kernel, learner_enc, device=device, time_feat=learner_time
            )
            if warmup:
                learner_out = flatten_policy_output(learner_out)
            learner_actions, learner_records = _sample_actions_with_records(
                vec,
                learner_out,
                learner_rows,
                learner_obs,
                learner_contexts,
                deterministic=False,
            )

            if not warmup:
                materialize_raw_sum += sum(
                    float(r.raw_launch.sum()) for r in learner_records
                )
                materialize_gap_sum += sum(
                    float(r.raw_launch.sum() - r.launch.sum())
                    for r in learner_records
                )
                materialize_log_tick += 1
                if materialize_log_tick >= sac.log_metrics_every:
                    if materialize_raw_sum > 0.0:
                        state.writer.add_scalar(
                            "rollout/materialize_gap_frac",
                            materialize_gap_sum / materialize_raw_sum,
                            global_step,
                        )
                    materialize_gap_sum = 0.0
                    materialize_raw_sum = 0.0
                    materialize_log_tick = 0

            opp_actions: list[Any] = [None] * num_envs
            builtin_rows = [
                (e, opp_seat[e]) for e in range(num_envs) if opponents[e][1] is not None
            ]
            builtin_obs = _observations_for_rows(vec, states, builtin_rows)
            for obs, (e, _seat) in zip(builtin_obs, builtin_rows, strict=True):
                _name, agent = opponents[e]
                if agent is None:
                    raise RuntimeError("builtin row resolved to self-play opponent")
                opp_actions[e] = agent(obs)

            self_play_envs = [e for e in range(num_envs) if opponents[e][1] is None]
            if self_play_envs:
                opp_rows = [(e, opp_seat[e]) for e in self_play_envs]
                opp_obs, opp_enc, opp_contexts, opp_time = _policy_inputs_for_rows(
                    vec,
                    states,
                    opp_rows,
                    episode_steps=cfg.game.episode_steps,
                    device=device,
                )
                opp_out = run_sac_heads(
                    heads_kernel, opp_enc, device=device, time_feat=opp_time
                )
                if opp_contexts is not None:
                    sampled_opp_actions, _opp_records = (
                        sample_batch_with_records_context(
                            opp_out, opp_contexts, deterministic=False, record_rows=[]
                        )
                    )
                else:
                    sampled_opp_actions = sample_batch_actions_raw(
                        opp_out, opp_obs, deterministic=False
                    )
                for e, acts in zip(self_play_envs, sampled_opp_actions, strict=True):
                    opp_actions[e] = acts
            for e in range(num_envs):
                if opp_actions[e] is None:
                    raise RuntimeError(f"missing opponent action for env {e}")

            # ---- step every env in parallel ----
            # 7-element learner/self-play actions carry the tracker payload the
            # worker uses to keep encoded fleet features in sync; the worker
            # strips sidecars before stepping the official env.
            actions_list: list[list[Any]] = []
            for e in range(num_envs):
                acts: list[Any] = [None] * num_players
                acts[learner_seat[e]] = learner_actions[e]
                acts[opp_seat[e]] = opp_actions[e]
                actions_list.append(acts)
            results = step_subset(list(range(num_envs)), actions_list)
            next_states: list[Any] = [None] * num_envs
            dones = [False] * num_envs
            finals: list[Any] = [None] * num_envs
            for e, (st, dn, fn) in results.items():
                next_states[e] = st
                dones[e] = dn
                finals[e] = fn
                episode_length[e] += 1

            # ---- s' legal-target support for the q-step's a'~π(·|s') resample ----
            next_learner_obs, next_learner_enc, next_contexts, next_learner_time = (
                _policy_inputs_for_rows(
                    vec,
                    next_states,
                    learner_rows,
                    episode_steps=cfg.game.episode_steps,
                    device=device,
                )
            )
            next_out = run_sac_heads(
                heads_kernel, next_learner_enc, device=device, time_feat=next_learner_time
            )
            _next_actions, next_records = _sample_actions_with_records(
                vec,
                next_out,
                learner_rows,
                next_learner_obs,
                next_contexts,
                deterministic=False,
            )

            # ---- reward, terminal flag, replay insert, episode logging ----
            for e in range(num_envs):
                # Reward = DELTA of the production margin: the agent is rewarded
                # the step it grows its own production (a capture) and penalized
                # the step the enemy grows theirs. r = pw·(Φ(s') − Φ(s)) with
                # Φ = own_production − max_opponent_production — non-zero exactly
                # on capture/loss events, so credit lands on the action that
                # caused the swing. See `_obs_production_margin`.
                cur_prod_margin = production_margin(
                    next_learner_obs[e], learner_seat[e]
                )
                step_reward = cfg.reward.potential_weight * (
                    cur_prod_margin - previous_prod_margin[e]
                )
                previous_prod_margin[e] = cur_prod_margin
                # Every episode end is a genuine TERMINAL here, so it zeroes the
                # bootstrap. Orbit Wars is a finite-horizon, time-limited task (the
                # game ends at `episode_steps`; the objective is ships at the end)
                # and the policy/critic OBSERVE the game clock (the FiLM time
                # feature), so this is a time-aware MDP: both episode ends —
                # elimination and the step-limit timeout — are true terminals with
                # no t>1 future to bootstrap from (Pardo et al., "Time Limits in
                # RL"). Bootstrapping the timeout would read γ·V of the auto-reset
                # next_obs (a FRESH game) and inflate the value — the qf1_value≫
                # realized-return overestimation. Hence terminal == done.
                terminal = bool(dones[e])

                if dones[e]:
                    opp_name = opponents[e][0]
                    seat_scores = [
                        float(getattr(s, "score", s.reward or 0.0))
                        for s in finals[e]
                    ]
                    ours = seat_scores[learner_seat[e]]
                    theirs = seat_scores[opp_seat[e]]
                    won = ours > theirs
                    drawn = ours == theirs
                    outcome = (
                        cfg.reward.win_value if won
                        else cfg.reward.draw_value if drawn
                        else cfg.reward.loss_value
                    )
                    margin = ours - theirs
                    step_reward += outcome + cfg.reward.margin_scale * margin

                    outcome_for_elo = 1.0 if won else 0.5 if drawn else 0.0
                    state.pool.elo.update_pair(LEARNER_NAME, opp_name, outcome_for_elo)
                    games_vs[opp_name] += 1
                    wins_vs[opp_name] += outcome_for_elo
                    margin_vs[opp_name] += margin
                    n_vs = games_vs[opp_name]

                    # Distinct x per finishing env within this tick's step range.
                    log_step = global_step + e
                    return_total = episode_return[e] + step_reward
                    state.writer.add_scalar(
                        "episode/return", return_total, log_step
                    )
                    state.writer.add_scalar(
                        "charts/episodic_return", return_total, log_step
                    )
                    state.writer.add_scalar(
                        "charts/episodic_length", episode_length[e], log_step
                    )
                    state.writer.add_scalar("episode/win_rate", float(won), log_step)
                    state.writer.add_scalar("episode/margin", margin, log_step)
                    state.writer.add_scalar(
                        f"winrate/{opp_name}", float(won), log_step
                    )
                    state.writer.add_scalar(
                        f"winrate_cumulative/{opp_name}",
                        wins_vs[opp_name] / n_vs,
                        log_step,
                    )
                    state.writer.add_scalar(
                        f"margin_cumulative/{opp_name}",
                        margin_vs[opp_name] / n_vs,
                        log_step,
                    )
                    state.writer.add_scalar(
                        "league/elo_learner",
                        state.pool.elo.get(LEARNER_NAME),
                        log_step,
                    )
                    for _bname in state.builtin_agents:
                        state.writer.add_scalar(
                            f"league/elo_{_bname}",
                            state.pool.elo.get(_bname),
                            log_step,
                        )

                replay.add(
                    _encoded_row(learner_enc, e),
                    # The MDP action is the policy's RAW launch (masked only for
                    # unowned / no-legal sources), NOT the materialized launch:
                    # the actor optimizes launch_p and the critic conditions on
                    # the taken launch, so both must see the same action. The
                    # env no-ops launches that can't build a move; that shows up
                    # as the realized reward + next state, which is exactly the
                    # value the critic should learn for the raw action.
                    learner_records[e].raw_launch,
                    learner_records[e].target_idx,
                    learner_records[e].fraction,
                    learner_records[e].target_legal_mask,
                    next_records[e].target_legal_mask,
                    step_reward,
                    terminal,
                    _encoded_row(next_learner_enc, e),
                    float(learner_time[e].item()),
                    float(next_learner_time[e].item()),
                )
                episode_return[e] += step_reward

            global_step += num_envs
            tick += 1

            # ---- advance live envs; reset+rematch finished ones ----
            for e in range(num_envs):
                if not dones[e]:
                    states[e] = next_states[e]
            done_envs = [e for e in range(num_envs) if dones[e]]
            if done_envs:
                reset_states = vec.reset_subset(done_envs)
                reset_rows: list[tuple[int, int]] = []
                for e in done_envs:
                    states[e] = reset_states[e]
                    learner_seat[e] = (learner_seat[e] + 1) % num_players  # flip seat
                    opp_seat[e] = (learner_seat[e] + 1) % num_players
                    opponents[e] = _select_opponent_for_episode(state)
                    reset_rows.append((e, learner_seat[e]))
                    episode_return[e] = 0.0
                    episode_length[e] = 0
                reset_obs = _observations_for_rows(vec, states, reset_rows)
                for obs, (e, seat) in zip(reset_obs, reset_rows, strict=True):
                    previous_prod_margin[e] = production_margin(obs, seat)

            # ---- learn ----
            if global_step >= sac.learning_starts and len(replay) >= sac.batch_size:
                update_credit += num_envs * float(sac.gradient_steps)
                critic_updates = int(update_credit)
                update_credit -= critic_updates
                if critic_updates > 0:
                    if prefetcher is None:
                        prefetcher = _ReplayPrefetcher(replay, sac.batch_size, device)
                    learn_step = _run_updates(
                        state,
                        prefetcher,
                        sac,
                        device,
                        num_envs=num_envs,
                        learn_step=learn_step,
                        tick=tick,
                        global_step=global_step,
                        start_time=start_time,
                        critic_updates=critic_updates,
                    )

            # ---- snapshot / rolling-latest (threshold-based: global_step jumps
            # by num_envs per tick, so exact modulo is unreliable) ----
            if (
                sac.snapshot_every > 0
                and global_step >= sac.learning_starts
                and global_step - last_snapshot_step >= sac.snapshot_every
            ):
                _save_checkpoint(state, global_step, label=f"{global_step:08d}")
                last_snapshot_step = global_step

            if (
                sac.latest_ckpt_every > 0
                and global_step >= sac.learning_starts
                and global_step - last_latest_step >= sac.latest_ckpt_every
            ):
                _save_checkpoint(state, global_step, label="latest")
                last_latest_step = global_step

        _save_checkpoint(state, global_step, label="final")
    finally:
        vec.close()
        state.writer.close()


def _log_metrics(
    writer: SummaryWriter,
    m: SACUpdateOutput,
    step: int,
    start_time: float,
) -> None:
    writer.add_scalar("losses/qf1_loss", m.qf1_loss, step)
    writer.add_scalar("losses/qf2_loss", m.qf2_loss, step)
    writer.add_scalar("losses/qf1_value", m.qf1_value, step)
    writer.add_scalar("losses/qf2_value", m.qf2_value, step)
    writer.add_scalar("losses/explained_variance", m.explained_var, step)
    # Action-dependent EV + advantage spread: distinguishes a critic that tracks
    # state-conditional value (high adv_explained_var) from one that only predicts
    # its own state value (explained_variance≈1 while adv_explained_var≈0). When
    # adv_spread≈0 the advantages are inert and the actor gets ~no action gradient.
    writer.add_scalar("losses/adv_explained_variance", m.adv_explained_var, step)
    writer.add_scalar("losses/adv_spread", m.adv_spread, step)
    writer.add_scalar("losses/qf_grad_norm", m.qf_grad_norm, step)
    writer.add_scalar("losses/alpha_disc", m.alpha_disc, step)
    writer.add_scalar("losses/alpha_cont", m.alpha_cont, step)
    # Soft-value scale consistency: the real-units entropy bonus α·H injected into
    # the bootstrap target, vs the hard value it rides on. entropy_bonus_frac near
    # 0 ⇒ a near-hard critic (the failure the real-units soft value fixes); a
    # healthy fraction confirms the soft target carries meaningful entropy.
    writer.add_scalar("bootstrap/value", m.boot_value, step)
    writer.add_scalar("bootstrap/entropy_bonus", m.boot_entropy_bonus, step)
    writer.add_scalar(
        "bootstrap/entropy_bonus_frac",
        abs(m.boot_entropy_bonus) / (abs(m.boot_value) + 1.0),
        step,
    )
    # Actor-side metrics (absent on a logging tick with no actor update).
    if m.actor_loss is not None:
        writer.add_scalar("losses/actor_loss", m.actor_loss, step)
    if m.actor_grad_norm is not None:
        writer.add_scalar("losses/actor_grad_norm", m.actor_grad_norm, step)
    if m.alpha_disc_loss is not None:
        writer.add_scalar("losses/alpha_disc_loss", m.alpha_disc_loss, step)
    if m.alpha_cont_loss is not None:
        writer.add_scalar("losses/alpha_cont_loss", m.alpha_cont_loss, step)
    # Entropy diagnostics: achieved vs the per-state target each α drives toward.
    if m.h_disc_mean is not None:
        writer.add_scalar("entropy/H_disc", m.h_disc_mean, step)
    if m.h_disc_per_planet is not None:
        writer.add_scalar("entropy/H_disc_per_planet", m.h_disc_per_planet, step)
    if m.target_h_disc is not None:
        writer.add_scalar("entropy/target_H_disc", m.target_h_disc, step)
    if m.logp_frac_mean is not None:
        # p-weighted Σ g·p·logp_frac (= −H_cont, summed per state).
        writer.add_scalar("entropy/cont_logp", m.logp_frac_mean, step)
        writer.add_scalar("entropy/H_cont", -m.logp_frac_mean, step)
    if m.target_h_cont is not None:
        writer.add_scalar("entropy/target_H_cont", m.target_h_cont, step)
    elapsed = max(1e-6, time.time() - start_time)
    writer.add_scalar("charts/SPS", step / elapsed, step)


__all__ = [
    "ReplayBuffer",
    "SACBatch",
    "SACState",
    "SACUpdateOutput",
    "train",
]
