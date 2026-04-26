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
    fraction_concentration: float = 4.0


@dataclass
class OptimCfg:
    lr: float = 3e-4
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
    clip_eps: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    # --- Value pretraining (cold-start the critic before PPO turns on). ---
    pretrain_updates: int = 0
    pretrain_episodes: int = 64
    pretrain_lr: float = 1.0e-3
    pretrain_behavior: str = "heuristic"  # which agent to roll behavior with


@dataclass
class RolloutCfg:
    episodes_per_update: int = 16
    parallel_workers: int = 1
    max_moves_per_turn: int = 16


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
