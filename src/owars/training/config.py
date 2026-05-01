"""YAML-driven run config (mirrors hull-tactical's RunConfig pattern)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml


@dataclass
class GameCfg:
    num_players: int = 2          # 2 or 4
    episode_steps: int = 500
    act_timeout: float = 1.0
    ship_speed: float = 6.0


@dataclass
class ModelCfg:
    # head_dim = dim / n_heads = 32 — the smallest setting in FA-2's eligible
    # set {16, 32, 64, 128, 256} that gives bf16 SDPA the flash kernel.
    # `dim=96` (head_dim=24) silently falls back to math/mem-efficient.
    dim: int = 128
    ff_dim: int = 256
    depth: int = 3
    n_heads: int = 4
    dropout: float = 0.0
    planet_rope_fraction: float = 0.25
    planet_rope_base: float = 10000.0
    encoder_backend: Literal["dense", "fleet_latent"] = "fleet_latent"
    num_fleet_latents: int = 64
    fleet_tokenizer_depth: int = 1
    value_hidden: int = 64
    value_num_bins: int = 153
    value_min: float = -100_000.0
    value_max: float = 100_000.0
    value_symlog: bool = True


@dataclass
class OptimCfg:
    """Optimizer hyperparameters.

    We use a parameter-golf-style dual-optimizer setup: **Muon (with row
    normalization, "normuon")** for 2D matrix weights inside transformer
    blocks, and fused **AdamW** for everything else (input projections,
    action/value readouts, biases, summary tokens, and control tensors like
    `attn_scale`/`ff_scale`/`resid_mix`).

    Muon orthogonalizes the gradient via Newton-Schulz iteration, producing
    updates with bounded spectral norm regardless of the gradient's input
    magnitude. This kills the cold-start first-update KL spike — AdamW's
    first step takes a full-lr step in the gradient direction with no
    running variance to scale against, whereas Muon's first step is
    pre-normalized to unit-spectral-norm before the lr multiply.
    """

    # Muon (matrix-2D weights inside transformer blocks).
    # `muon_lr` is naturally ~50–100× larger than an AdamW lr because Muon
    # updates are bounded after orthogonalization; parameter-golf uses 0.022
    # for matrix params on a 512-dim transformer.
    muon_lr: float = 0.022
    muon_momentum: float = 0.95
    muon_backend_steps: int = 5
    muon_row_normalize: bool = True
    # Group same-shaped matrices so Muon's row-normalization and
    # Newton-Schulz backend run as batched/foreach operations instead of a
    # Python loop of tiny per-parameter matmuls.
    muon_fused: bool = True
    muon_weight_decay: float = 0.0
    # Linear ramp `momentum_warmup_start` → `muon_momentum` over the first
    # `muon_momentum_warmup_steps` optimizer-step calls. Prevents the
    # momentum buffer from baking in noise from the first few gradient
    # samples on a freshly-initialized policy — those gradients are
    # atypically noisy and a 0.95 buffer would carry them for ~20 steps.
    # (parameter-golf uses 1500 steps, 0.92 → 0.99; we ramp faster because
    # PPO's first updates land before that long.)
    muon_momentum_warmup_steps: int = 100
    muon_momentum_warmup_start: float = 0.85
    # AdamW (default group: input projections, biases, summary tokens).
    # PPO-canonical 3e-4.
    lr: float = 3e-4
    # AdamW for task readouts (`target_query/key`, launch/fraction heads,
    # and value head). parameter-golf keeps output heads out of Muon; using
    # fused AdamW here is both cheaper than Newton-Schulz on tiny matrices
    # and avoids turning a large readout gradient into a full spectral step.
    head_lr: float = 3e-4
    # AdamW (control-tensor group: per-channel residual scales `attn_scale`,
    # `ff_scale`, `resid_mix`, and per-head attention temperature `q_gain`).
    # parameter-golf runs `scalar_lr ≈ matrix_lr` (0.02 vs 0.022) — equal
    # update magnitudes between Muon-driven matrices and AdamW-driven
    # scalars. With `lr=3e-4` the scalars move 67× slower than the matrices
    # and can't damp residual contribution fast enough to compensate for
    # actor drift, so KL accumulates. Match `muon_lr` to restore parity.
    control_lr: float = 0.02
    weight_decay: float = 1e-4
    grad_clip: float = 0.5
    minibatch_size: int = 4096
    epochs_per_update: int = 4


@dataclass
class PPOCfg:
    """SPO-asym policy objective + distributional critic.

    Dense potential rewards make the per-step signal informative, so use the
    conventional PPO GAE setup: discounted returns with one fixed lambda for
    both actor advantages and critic targets.
    """

    gamma: float = 0.997
    gae_lambda: float = 0.95
    norm_advantage: bool = True
    # CleanRL SPO asym: quadratic ratio penalty with a looser bound when
    # ratio drift agrees with the advantage sign.
    spo_eps_low: float = 0.2
    spo_eps_high: float = 0.28
    # Distributional CE gradients are naturally bounded (per-bin
    # `softmax − target_probs` has ‖∇‖ ~ O(1)), unlike MSE which blew
    # up under bad predictions. dreamer4 effectively runs the equivalent
    # of `value_coef=1.0` (separate `value_optim`, `dreamer4.py:4543`).
    value_coef: float = 1.0
    # Categorical target entropy has no structural floor, so keep it from
    # collapsing. The Beta fraction already has a concentration floor; do not
    # reward collapse toward max-entropy Beta(1,1).
    target_entropy_coef: float = 0.01
    fraction_entropy_coef: float = 0.0
    # No value clipping. dreamer4-style clipping (`max(ce, ce_of_clipped_v)`)
    # only behaves sensibly when the HL-Gauss σ is wide enough that the
    # re-encoded clipped scalar overlaps with the return-encoded target.
    # Our σ = 0.5·bin_size is single-bin-narrow, so the clipped CE saturates
    # at `−log(eps) ≈ 46` whenever `|clipped_v − return| ≳ σ` (i.e. always at
    # cold start). Distributional CE has bounded per-element gradients
    # (`softmax_i − target_i ∈ [-1, 1]`), so the safety rationale for
    # clipping doesn't apply the way it does to scalar MSE.
    # --- Value pretraining (cold-start the critic before PPO turns on). ---
    pretrain_updates: int = 0
    pretrain_episodes: int = 64
    pretrain_lr: float = 1.0e-3
    pretrain_behavior: str = "heuristic"  # which agent to roll behavior with


@dataclass
class RolloutCfg:
    """Per-update rollout settings.

    `num_envs` is the total rollout parallelism: each PPO update plays
    exactly this many episodes and batches policy forwards across all alive
    envs each step.
    `rust` is the default fast training backend. `numpy` uses the in-process
    Python/NumPy parity path. `numpy_mp` shards that path across CPU worker
    processes. `num_workers` is only used by `numpy_mp`; the official Kaggle
    backend already runs one worker per env.
    """

    num_envs: int = 128
    num_workers: int = 0  # 0 => backend default; set to physical cores for rollout-heavy runs
    env_backend: str = "rust"  # "rust", "numpy", "numpy_mp", or "kaggle"


@dataclass
class OpponentsCfg:
    """Self-play matchmaking + Elo-pruned snapshot pool.

    Per opponent slot, with probability `self_play_prob` we play "self" (a
    no-grad copy of the current learner); otherwise we pick uniformly from
    the live snapshot pool. The pool keeps the top-K snapshots by Elo —
    weak snapshots get evicted instead of aging out by FIFO.
    """

    snapshot_every: int = 25      # save a frozen snapshot for the pool every N updates
    top_k: int = 10               # max live snapshots; lowest-Elo evicted past this
    self_play_prob: float = 0.8   # P(opponent slot = current learner) per slot
    snapshot_device: str = "train"  # "cpu", "cuda", or "train" to mirror run.device
    initial_rating: float = 1500.0
    k_factor: float = 32.0


@dataclass
class RewardCfg:
    """Dense potential reward aligned with the terminal scoring rule.

    Per learner step, reward is:

        potential_weight * (Phi(s_next) - Phi(s))

    where Phi is raw projected population margin against the strongest enemy.
    Population is current ships on owned planets plus ships in owned fleets.
    Production is converted into projected future population by
    `production_weight * turns_left`.

    Terminal outcome fields default to zero because the dense potential
    replaces the old ±1 terminal-only reward. They remain configurable for
    ablations that want to mix outcome reward back in.
    """

    potential_weight: float = 1.0
    production_weight: float = 1.0
    win_value: float = 0.0
    loss_value: float = 0.0
    draw_value: float = 0.0
    margin_scale: float = 0.0


@dataclass
class RunCfg:
    name: str = "default"
    seed: int = 0
    device: str = "cuda"
    compile_mode: str = "reduce-overhead"
    # Small batched rollout forwards are slower with large CPU thread pools.
    # 0 leaves PyTorch's process default unchanged.
    torch_num_threads: int = 8
    log_root: str = "runs"
    ckpt_root: str = "checkpoints"
    total_updates: int = 1000


@dataclass
class RunConfig:
    run: RunCfg = field(default_factory=RunCfg)
    game: GameCfg = field(default_factory=GameCfg)
    model: ModelCfg = field(default_factory=ModelCfg)
    optim: OptimCfg = field(default_factory=OptimCfg)
    ppo: PPOCfg = field(default_factory=PPOCfg)
    rollout: RolloutCfg = field(default_factory=RolloutCfg)
    opponents: OpponentsCfg = field(default_factory=OpponentsCfg)
    reward: RewardCfg = field(default_factory=RewardCfg)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> RunConfig:
        cfg = cls()
        for section, sub in d.items():
            if not hasattr(cfg, section):
                raise KeyError(f"unknown config section: {section!r}")
            target = getattr(cfg, section)
            for k, v in (sub or {}).items():
                if not hasattr(target, k):
                    raise KeyError(f"unknown {section}.{k}")
                setattr(target, k, v)
        if not 0.0 <= cfg.ppo.gae_lambda <= 1.0:
            raise ValueError("ppo.gae_lambda must be in [0, 1]")
        if not 0.0 < cfg.ppo.gamma <= 1.0:
            raise ValueError("ppo.gamma must be in (0, 1]")
        if cfg.ppo.spo_eps_low <= 0.0 or cfg.ppo.spo_eps_high <= 0.0:
            raise ValueError("ppo.spo_eps_low/high must be positive")
        if cfg.ppo.spo_eps_high < cfg.ppo.spo_eps_low:
            raise ValueError("ppo.spo_eps_high must be >= ppo.spo_eps_low")
        if not 0.0 <= cfg.model.planet_rope_fraction <= 1.0:
            raise ValueError("model.planet_rope_fraction must be in [0, 1]")
        if cfg.model.planet_rope_base <= 0.0:
            raise ValueError("model.planet_rope_base must be positive")
        if cfg.model.value_min >= cfg.model.value_max:
            raise ValueError("model.value_min must be less than model.value_max")
        if cfg.reward.production_weight < 0.0:
            raise ValueError("reward.production_weight must be non-negative")
        return cfg


def load_config(path: str | Path) -> RunConfig:
    with open(path) as f:
        return RunConfig.from_dict(yaml.safe_load(f) or {})


def deep_override(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge `override` into `base` and return a new dict."""
    out = {k: (dict(v) if isinstance(v, dict) else v) for k, v in base.items()}
    for k, v in override.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = deep_override(out[k], v)
        else:
            out[k] = v
    return out
