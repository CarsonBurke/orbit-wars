"""YAML-driven run config (mirrors hull-tactical's RunConfig pattern)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class GameCfg:
    num_players: int = 2          # 2 or 4
    episode_steps: int = 500
    act_timeout: float = 1.0
    ship_speed: float = 6.0


@dataclass
class ModelCfg:
    dim: int = 96
    ff_dim: int = 256
    depth: int = 3
    n_heads: int = 4
    dropout: float = 0.0


@dataclass
class OptimCfg:
    """Optimizer hyperparameters.

    We use a parameter-golf-style dual-optimizer setup: **Muon (with row
    normalization, "normuon")** for 2D matrix weights in the transformer
    blocks and projection heads, and **AdamW** for everything else
    (LayerNorm gains/biases, summary tokens, control tensors like
    `attn_scale`/`ff_scale`/`resid_mix`, and Linear biases).

    Muon orthogonalizes the gradient via Newton-Schulz iteration, producing
    updates with bounded spectral norm regardless of the gradient's input
    magnitude. This kills the cold-start first-update KL spike — AdamW's
    first step takes a full-lr step in the gradient direction with no
    running variance to scale against, whereas Muon's first step is
    pre-normalized to unit-spectral-norm before the lr multiply.
    """

    # Muon (matrix-2D weights in blocks + projection heads).
    # `muon_lr` is naturally ~50–100× larger than an AdamW lr because Muon
    # updates are bounded after orthogonalization; parameter-golf uses 0.022
    # for matrix params on a 512-dim transformer.
    muon_lr: float = 0.022
    # Slower Muon LR for the action-head readout matrices (target_query,
    # target_key, fraction_head, noop_head). parameter-golf gives the LM
    # head ~3× slower LR than the trunk (`head_lr=0.008` vs `matrix_lr=0.022`,
    # `sota_train_gpt.py:10,211`). The readout is the only path between
    # encoder shifts and policy logits/μ — slowing it dampens ratio drift
    # per step without slowing the trunk's ability to learn the value
    # function. value_head matrices stay at `muon_lr` (value loss isn't
    # on the KL critical path).
    muon_head_lr: float = 0.008
    muon_momentum: float = 0.95
    muon_backend_steps: int = 5
    muon_row_normalize: bool = True
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
    # AdamW (default group: biases, summary tokens). PPO-canonical 3e-4.
    lr: float = 3e-4
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
    minibatch_size: int = 1024
    epochs_per_update: int = 4


@dataclass
class PPOCfg:
    """PPO + VAPO-style decoupled critic.

    `gamma` defaults to 1.0 because Orbit Wars is finite-horizon (≤500 steps)
    with terminal-only reward — there's no infinite-horizon variance issue
    and discounting just decays the only signal we have.

    `lambda_critic = 1.0` makes the value target a Monte-Carlo return
    (unbiased; the cold-start regime where bootstrapping hurts most). The
    actor advantage uses `lambda_policy` for variance reduction. Setting
    `lambda_policy_alpha > 0` switches the actor to length-adaptive
    `λ = 1 − 1/(α·l)` (VAPO §4.2)."""

    gamma: float = 1.0
    lambda_critic: float = 1.0
    lambda_policy: float = 0.95
    lambda_policy_alpha: float = 0.0   # 0 ⇒ use fixed `lambda_policy`
    # Asymmetric PPO clipping (VAPO §4.4 / DAPO): a wider upper bound lets
    # the policy lean into beneficial moves while a tight lower bound limits
    # catastrophic policy jumps on negative advantage. The 0.2/0.28 default
    # is the VAPO recommendation; setting them equal recovers vanilla PPO.
    clip_eps_low: float = 0.2
    clip_eps_high: float = 0.28
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    # --- Value pretraining (cold-start the critic before PPO turns on). ---
    pretrain_updates: int = 0
    pretrain_episodes: int = 64
    pretrain_lr: float = 1.0e-3
    pretrain_behavior: str = "heuristic"  # which agent to roll behavior with


@dataclass
class RolloutCfg:
    """Per-update rollout settings.

    `num_envs` is the parallelism: each PPO update plays exactly this many
    episodes in parallel via subprocess workers, and the policy forward
    is batched across all alive envs each step. There's no separate
    "workers" knob — workers == envs, one each.
    """

    num_envs: int = 16
    max_moves_per_turn: int = 16
    env_backend: str = "numpy"  # "numpy" in-process or "kaggle" subprocesses


@dataclass
class OpponentsCfg:
    """Self-play matchmaking + Elo-pruned snapshot pool.

    Per opponent slot, with probability `self_play_prob` we play "self" (a
    no-grad copy of the current learner); otherwise we pick uniformly from
    the live snapshot pool. The pool keeps the top-K snapshots by Elo —
    weak snapshots get evicted instead of aging out by FIFO.
    """

    snapshot_every: int = 25      # save a frozen snapshot for the pool every N updates
    top_k: int = 8                # max live snapshots; lowest-Elo evicted past this
    self_play_prob: float = 0.8   # P(opponent slot = current learner) per slot
    snapshot_device: str = "cpu"  # "cpu", "cuda", or "train" to mirror run.device
    initial_rating: float = 1500.0
    k_factor: float = 32.0


@dataclass
class RewardCfg:
    """VAPO-style outcome reward by default: ±1 terminal, 0 mid-episode.

    Shaping fields default to 0 and exist only for ablation. The premise
    is that GAE with γ=0.997 and λ=0.95 over a 500-step horizon will
    bootstrap the value function back through the game from the terminal
    sign alone — adding shaping injects non-stationary noise into
    intermediate returns that fights the value head. If pure terminal
    fails to learn we'll know from the train/win_rate scalar; turn the
    shaping knobs on then."""

    win_value: float = 1.0
    loss_value: float = -1.0
    draw_value: float = 0.0
    # --- Shaping (default off; ablation only). ---
    capture_bonus: float = 0.0
    loss_penalty: float = 0.0
    sun_loss_penalty: float = 0.0
    margin_scale: float = 0.0


@dataclass
class RunCfg:
    name: str = "default"
    seed: int = 0
    device: str = "cuda"
    # Small batched rollout forwards are slower with large CPU thread pools.
    # 0 leaves PyTorch's process default unchanged.
    torch_num_threads: int = 8
    log_root: str = "runs"
    ckpt_root: str = "checkpoints"
    total_updates: int = 200


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
    def from_dict(cls, d: dict[str, Any]) -> "RunConfig":
        cfg = cls()
        for section, sub in d.items():
            if not hasattr(cfg, section):
                raise KeyError(f"unknown config section: {section!r}")
            target = getattr(cfg, section)
            for k, v in (sub or {}).items():
                if not hasattr(target, k):
                    raise KeyError(f"unknown {section}.{k}")
                setattr(target, k, v)
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
