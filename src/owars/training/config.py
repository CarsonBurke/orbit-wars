"""YAML-driven run config (mirrors hull-tactical's RunConfig pattern)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml

from owars.policies.config import normalize_attention_config


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
    # Grouped-query attention. `None` uses ordinary MHA (`n_heads` KV heads);
    # set to `1` for MQA or another divisor of `n_heads` for GQA.
    n_kv_heads: int | None = None
    dropout: float = 0.0
    # nGPT residual / attention init profile (see OrbitPolicyConfig). Defaults
    # are faithful nGPT; the "old-block" profile (full-strength residual, sharp
    # attention, U-net skip) is opt-in via eigen_alpha_init=0.5 / qk_gain_init=5
    # / block_skip=true.
    eigen_alpha_init: float = 0.05
    qk_gain_init: float = 1.0
    block_skip: bool = False
    planet_rope_fraction: float = 0.25
    planet_rope_base: float = 10000.0
    encoder_backend: Literal["dense", "fleet_latent", "destination_conditioned"] = "fleet_latent"
    global_features: int = 27
    num_fleet_latents: int = 64
    fleet_tokenizer_depth: int = 1
    value_hidden: int = 64
    value_num_bins: int = 153
    critic_mtp_horizon: int = 6
    value_min: float = -100_000.0
    value_max: float = 100_000.0
    value_symlog: bool = True
    action_logit_softcap: float = 8.0
    # Real-units bound on the per-planet SAC advantage heads (see
    # OrbitPolicyConfig.adv_scale): each head emits adv_scale·tanh(raw/adv_scale).
    adv_scale: float = 40.0


@dataclass
class OptimCfg:
    """Optimizer hyperparameters.

    We use a parameter-golf-style dual-optimizer setup: **Muon (with NorMuon
    neuron-wise normalization)** for 2D matrix weights inside transformer
    blocks, and fused **AdamW** for everything else (input projections,
    action/value readouts, biases, summary tokens, and the nGPT hypersphere
    control tensors `attn_alpha`/`mlp_alpha`/`cross_alpha`/`sqk`/`suv`).

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
    # NorMuon (arXiv:2510.05491): per-output-neuron second-moment EMA applied
    # AFTER Newton-Schulz, with Frobenius-norm restoration so the step stays
    # lr-compatible with plain Muon. See `_normuon_normalize` in `muon.py`.
    muon_normuon: bool = True
    # Decay for the NorMuon per-neuron second-moment EMA (Adam-style beta2).
    muon_beta2: float = 0.95
    # Group same-shaped matrices so Muon's Newton-Schulz backend runs as
    # batched/foreach operations instead of a Python loop of tiny
    # per-parameter matmuls.
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
    # AdamW (control-tensor group: nGPT hypersphere controls — per-channel
    # eigen LRs `attn_alpha`/`mlp_alpha`/`cross_alpha`, QK scale `sqk`, MLP
    # scale `suv` — plus the target-readout temperature `q_gain`).
    # Reference-faithful: nGPT trains these 1-D control scalars at the SAME
    # base AdamW lr as the matrices (ngpt/model.py:305-319 — the ndim<2 group
    # shares `learning_rate`); there is no separate fast "control" group. The
    # old 0.02 (≈67× lr, to "match muon_lr") over-drove the gating scalars: an
    # orthogonalized, bounded Muon step is NOT comparable to an AdamW scalar
    # step, so matching muon_lr was a bug — the scalar gain ran away until the
    # PPO trust region snapped (KL blowup). Raise toward lr×2-3 only if the
    # scalars learn too slowly.
    control_lr: float = 3e-4
    # Linear LR warmup over the first `lr_warmup_steps` optimizer-step calls
    # (ramping every group's lr from ~0 → configured value). The nGPT port
    # needs this cold-start guard: a fresh policy has uncalibrated AdamW
    # second moments, so the first ~tens of minibatch steps take near-full
    # `lr`·sign() steps; without the ramp the trunk-gating scalars swing hard
    # across the first PPO update and spike the policy KL far outside the
    # frozen-`old_log_prob` trust region. ~2-3 PPO updates of ramp removes the
    # update-0 KL spike. 0 disables. Counted in optimizer steps (≈ epochs ×
    # minibatches per update), matching the Muon momentum warmup.
    lr_warmup_steps: int = 100
    # KL-feedback LR controller. PPO always runs the configured epoch count;
    # KL is used only to adapt the next update's LR and to expose explosions in
    # the logs. The signal is `PPOLog.per_planet_approx_kl`, smoothed by an EMA
    # with this half-life in PPO updates. The persistent LR scale is multiplied
    # by `kl_lr_target / kl_ema` after each update, then clamped.
    kl_lr_target: float = 0.025
    kl_lr_ema_half_life: float = 20.0
    kl_lr_min_scale: float = 0.1
    kl_lr_max_scale: float = 10.0
    weight_decay: float = 1e-4
    # PPO clips actor and critic flows separately. Each flow includes its task
    # readout head plus the shared trunk, then clipped shared gradients are
    # summed before the optimizer step.
    grad_clip: float = 1.0
    # If set, PPO divides the full rollout batch into exactly this many
    # shuffled minibatches per epoch, padding the tail with zero-weight rows so
    # every optimizer step has one stable shape.
    minibatch_count: int | None = None
    minibatch_size: int = 4096
    epochs_per_update: int = 4


@dataclass
class PPOCfg:
    """Asymmetric clip-higher policy objective + distributional critic.

    Dense potential rewards make the per-step signal informative, so the
    default is conventional PPO GAE with one lambda. Sparse terminal rewards
    can decouple the critic target with `value_gae_lambda=1.0` while keeping
    lower-variance policy advantages.
    """

    gamma: float = 0.997
    gae_lambda: float = 0.95
    value_gae_lambda: float | None = None
    # CleanRL IterThink v24 policy-advantage shaping. "rankgauss" maps the
    # full rollout's raw GAE advantages to empirical Gaussian quantiles before
    # minibatching; "none" keeps raw GAE.
    advantage_transform: Literal["rankgauss", "none"] = "rankgauss"
    norm_advantage: bool = True
    # Asymmetric PPO clip-higher (DAPO / cleanRL iterthink_v24_beta): the
    # surrogate ratio is clamped to [1-clip_coef, 1+clip_coef_high]. The upper
    # bound is deliberately looser so an under-weighted action can recover while
    # an already-favored one stays capped.
    clip_coef: float = 0.2
    clip_coef_high: float = 0.28
    # Distributional CE gradients are naturally bounded (per-bin
    # `softmax − target_probs` has ‖∇‖ ~ O(1)), unlike MSE which blew
    # up under bad predictions. dreamer4 effectively runs the equivalent
    # of `value_coef=1.0` (separate `value_optim`, `dreamer4.py:4543`).
    value_coef: float = 1.0
    # Categorical target entropy has no structural floor, so keep it from
    # collapsing. Fraction entropy uses the squashed-Gaussian Normal entropy
    # approximation and is usually left off for PPO.
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
class SACCfg:
    """Soft Actor-Critic (hybrid discrete+continuous variant).

    The actor emits the SAME factored action PPO uses: per owned source planet
    a Bernoulli launch, a masked Categorical target, and a tanh-squashed Normal
    fraction on (0, 1). The launch *angle* is solved analytically from the
    chosen target by the lead-intercept solver (not learned). The fraction is
    the only continuous action dim.

    SAC treatment: discrete-SAC (closed-form Bernoulli+categorical entropy) for
    launch+target; cleanrl reparameterized tanh-Normal for the fraction. The
    critic is FACTORED (dueling) in REAL ship-margin units,
    `Q(s, a) = V(s) + Σ_i A_i(s, a_i)`: V = bins_to_scalar of an HL-Gauss two-hot
    distribution over symlog-spaced bins, and the per-planet advantages A_i are
    scalar and tanh-bounded (±`adv_scale`). Because the scalar adv = Σ_i A_i is
    LINEAR in the policy probs, the soft-value expectation E_a[Q] stays
    closed-form — so the launch/target gradient is EXACT (no REINFORCE / no
    score-function baseline); only the fraction uses the pathwise/reparam
    gradient. The actor ascends the scalar advantage directly, treating V as an
    action-independent baseline.

    Two entropy temperatures are tuned independently: `alpha_discrete` for the
    launch+target factors and `alpha_continuous` for the fraction.

    Three independent set-transformer networks (actor, qf1, qf2) plus EMA
    target copies of qf1/qf2; no encoder sharing between actor and critics.
    """

    gamma: float = 0.99
    tau: float = 0.005                # Polyak coefficient for target net
    # Initial entropy temperatures. `alpha`/`autotune` are retained for
    # RunConfig parity; the hybrid actor always tunes the two split alphas.
    alpha: float = 0.2
    autotune: bool = True
    alpha_discrete: float = 0.2       # initial launch+target entropy temperature
    alpha_continuous: float = 0.2     # initial fraction entropy temperature
    alpha_discrete_lr: float = 1.0e-3
    alpha_continuous_lr: float = 1.0e-3
    # Discrete target entropy is tuned on the PER-PLANET AVERAGE: α_disc drives
    # `H_disc / n_owned` toward `ratio · (h_disc_max / n_owned)`, where
    # h_disc_max = Σ_i g_i·log(n_legal_i+1) is the max attainable launch+target
    # entropy (the joint (n_legal+1)-way choice {noop} ∪ {launch→t}). Averaging
    # keeps the target bounded so α_disc does not run away as the owned-planet
    # count grows. 0.7 keeps selection exploratory without forcing uniform.
    disc_target_entropy_ratio: float = 0.7
    # Per-launched-dim target entropy for the (single) continuous fraction,
    # scaled by the per-sample launch mass Σ g·p (see _actor_alpha_step). -1.0
    # nat/dim is the cleanrl default (= -dim). It pairs with the α-weighted alpha
    # loss `-(α·(logp + target))` (α = log_alpha.exp()) in _actor_alpha_step:
    # that loss self-damps as α shrinks, so a target temporarily below the
    # bounded fraction's natural entropy no longer collapses α_cont — a floor
    # clamp is not needed.
    target_entropy_per_dim: float = -1.0
    # No-op threshold for the post-squash fraction. Below this, the planet
    # silently skips its launch at action-translation time. The simulator
    # already coerces send=0 to no-op, but an explicit threshold keeps the
    # buffer's stored fraction comparable to what the env actually executed.
    no_op_fraction: float = 0.02
    # Standard SAC pre-squash log-std clamp (cleanrl uses SpinUp's tanh
    # remap of a Linear-output log_std into [-5, 2]).
    log_std_min: float = -5.0
    log_std_max: float = 2.0

    buffer_size: int = 100_000        # transitions; per-env-step pushes
    batch_size: int = 256
    learning_starts: int = 5_000      # uniform-random action until this many transitions
    # Cadence counts CRITIC UPDATES (continuous across rollout ticks): an
    # actor+alpha step every policy_frequency-th critic update, run
    # policy_frequency times (cleanrl compensation ⇒ net 1:1 actor:critic); a
    # target polyak every target_network_frequency-th critic update.
    policy_frequency: int = 2
    target_network_frequency: int = 1
    gradient_steps: float = 1.0       # UTD: critic updates per collected transition

    q_lr: float = 1.0e-3
    policy_lr: float = 3.0e-4
    alpha_lr: float = 1.0e-3
    weight_decay: float = 0.0
    grad_clip: float = 0.0            # 0 disables clipping (cleanrl default)

    log_metrics_every: int = 100
    snapshot_every: int = 25_000      # transitions between labeled archive checkpoints
    # Rolling `sac_latest.pt` overwrite cadence. Cheap single-file checkpoint so
    # `scripts/latest_replay.py` always finds a current policy to render while
    # training runs. 0 disables. Distinct from the labeled snapshot archive.
    latest_ckpt_every: int = 2_000

    # Opponent slate. Self-play uses the LIVE model only — just the active actor
    # and its replay buffer, no frozen snapshot copies. Per episode, with
    # probability `builtin_prob` the opponent seat is one of `builtin_opponents`
    # (chosen uniformly); otherwise it's self-play against the current learner.
    # Mixing in fixed baselines is the guard against self-play strategy collapse
    # (see AGENTS.md "Self-play strategy collapse").
    builtin_opponents: list[str] = field(
        default_factory=lambda: ["random", "sniper_v17", "heuristic"]
    )
    builtin_prob: float = 0.5


@dataclass
class RolloutCfg:
    """Per-update rollout settings.

    `num_envs` is the rollout parallelism. Each PPO update plays
    `num_envs * games_per_env_per_update` episodes and batches policy forwards
    across all alive envs each step.
    `rust` is the default fast training backend. `numpy` uses the in-process
    Python/NumPy parity path. `numpy_mp` shards that path across CPU worker
    processes. `num_workers` is only used by `numpy_mp`; the official Kaggle
    backend already runs one worker per env.
    """

    num_envs: int = 128
    games_per_env_per_update: int = 1
    num_workers: int = 0  # 0 => backend default; set to physical cores for rollout-heavy runs
    env_backend: str = "rust"  # "rust", "numpy", "numpy_mp", or "kaggle"


@dataclass
class OpponentsCfg:
    """Opponent matchmaking.

    ``mode="league"`` is the normal self-play league: per opponent slot, with
    probability `self_play_prob` we play "self" (a no-grad copy of the current
    learner); otherwise we pick uniformly from the live snapshot pool. The pool
    keeps the top-K snapshots by Elo.

    ``mode="fixed"`` samples static builtin opponents from `fixed_opponents`
    and never uses self-play or snapshots. This is the low-noise training mode
    for measuring learner progress against a stationary baseline.
    """

    mode: Literal["league", "fixed"] = "league"
    fixed_opponents: list[str] = field(default_factory=lambda: ["sniper_v17"])
    snapshot_every: int = 25      # save a frozen snapshot for the pool every N updates
    top_k: int = 10               # max live snapshots; lowest-Elo evicted past this
    self_play_prob: float = 0.8   # P(opponent slot = current learner) per slot
    snapshot_device: str = "train"  # "cpu", "cuda", or "train" to mirror run.device
    initial_rating: float = 1500.0
    k_factor: float = 32.0


@dataclass
class RewardCfg:
    """Reward configuration.

    `signal` selects the reward family:

      - `projected_margin`: PROJECTED population margin —
        current ships on owned planets + ships in owned fleets, plus production
        converted to projected future ships by `production_weight * turns_left`.
        O(±10^3).
      - `production_margin`: PRODUCTION-RATE margin only —
        Σ production over owned planets (comets included), NO ship counts and NO
        turns_left projection. O(±10^2), which keeps the distributional critic's
        value bounded (see configs/sac_base.yaml). `production_weight` is unused
        by this one.
      - `win_terminal`: terminal-only outcome reward. Dense potential deltas are
        disabled; configs that specify only `signal: win_terminal` default to
        `win_value=+1`, `loss_value=-1`, `draw_value=0`.

    The dense delta is the SOLE reward: the terminal outcome fields
    (win/loss/draw_value, margin_scale) default to zero. They stay configurable
    for ablations, but note that under gamma<1 over a 500-step horizon a sparse
    terminal term is discounted to near-zero for early actions, so it is a weak
    lever — the dense potential carries the signal.
    """

    signal: Literal[
        "projected_margin",
        "production_margin",
        "win_terminal",
    ] = "projected_margin"
    potential_weight: float = 1.0
    production_weight: float = 1.0
    win_value: float = 0.0
    loss_value: float = 0.0
    draw_value: float = 0.0
    margin_scale: float = 0.0

    def uses_dense_potential(self) -> bool:
        return self.signal != "win_terminal" and self.potential_weight != 0.0


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
    sac: SACCfg = field(default_factory=SACCfg)
    rollout: RolloutCfg = field(default_factory=RolloutCfg)
    opponents: OpponentsCfg = field(default_factory=OpponentsCfg)
    reward: RewardCfg = field(default_factory=RewardCfg)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> RunConfig:
        cfg = cls()
        reward_section = d.get("reward") or {}
        ppo_section = d.get("ppo") or {}
        model_section = d.get("model") or {}
        for section, sub in d.items():
            if not hasattr(cfg, section):
                raise KeyError(f"unknown config section: {section!r}")
            target = getattr(cfg, section)
            for k, v in (sub or {}).items():
                if not hasattr(target, k):
                    raise KeyError(f"unknown {section}.{k}")
                setattr(target, k, v)
        if cfg.reward.signal == "win_terminal":
            if "potential_weight" not in reward_section:
                cfg.reward.potential_weight = 0.0
            if "win_value" not in reward_section:
                cfg.reward.win_value = 1.0
            if "loss_value" not in reward_section:
                cfg.reward.loss_value = -1.0
            if "draw_value" not in reward_section:
                cfg.reward.draw_value = 0.0
            if "gamma" not in ppo_section:
                cfg.ppo.gamma = 1.0
            if "value_gae_lambda" not in ppo_section:
                cfg.ppo.value_gae_lambda = 1.0
            if "value_min" not in model_section:
                cfg.model.value_min = -1.0
            if "value_max" not in model_section:
                cfg.model.value_max = 1.0
            if "value_num_bins" not in model_section:
                cfg.model.value_num_bins = 41
            if "value_symlog" not in model_section:
                cfg.model.value_symlog = False
        if not 0.0 <= cfg.ppo.gae_lambda <= 1.0:
            raise ValueError("ppo.gae_lambda must be in [0, 1]")
        if cfg.ppo.value_gae_lambda is not None and not (
            0.0 <= cfg.ppo.value_gae_lambda <= 1.0
        ):
            raise ValueError("ppo.value_gae_lambda must be in [0, 1]")
        if not 0.0 < cfg.ppo.gamma <= 1.0:
            raise ValueError("ppo.gamma must be in (0, 1]")
        if cfg.reward.signal == "win_terminal" and cfg.ppo.gamma != 1.0:
            raise ValueError("reward.signal='win_terminal' requires ppo.gamma=1.0")
        if cfg.ppo.clip_coef <= 0.0 or cfg.ppo.clip_coef_high <= 0.0:
            raise ValueError("ppo.clip_coef/high must be positive")
        if (
            cfg.reward.signal == "win_terminal"
            and cfg.ppo.value_gae_lambda != 1.0
        ):
            raise ValueError(
                "reward.signal='win_terminal' requires ppo.value_gae_lambda=1.0"
            )
        if cfg.ppo.clip_coef_high < cfg.ppo.clip_coef:
            raise ValueError("ppo.clip_coef_high must be >= ppo.clip_coef")
        if cfg.ppo.advantage_transform not in {"rankgauss", "none"}:
            raise ValueError("ppo.advantage_transform must be 'rankgauss' or 'none'")
        if cfg.optim.minibatch_count is not None and cfg.optim.minibatch_count <= 0:
            raise ValueError("optim.minibatch_count must be positive when set")
        if cfg.optim.minibatch_size <= 0:
            raise ValueError("optim.minibatch_size must be positive")
        if cfg.optim.epochs_per_update <= 0:
            raise ValueError("optim.epochs_per_update must be positive")
        if cfg.optim.kl_lr_target <= 0.0:
            raise ValueError("optim.kl_lr_target must be positive")
        if cfg.optim.kl_lr_ema_half_life <= 0.0:
            raise ValueError("optim.kl_lr_ema_half_life must be positive")
        if cfg.optim.kl_lr_min_scale <= 0.0:
            raise ValueError("optim.kl_lr_min_scale must be positive")
        if cfg.optim.kl_lr_max_scale < cfg.optim.kl_lr_min_scale:
            raise ValueError(
                "optim.kl_lr_max_scale must be >= optim.kl_lr_min_scale"
            )
        if cfg.rollout.num_envs <= 0:
            raise ValueError("rollout.num_envs must be positive")
        if cfg.rollout.games_per_env_per_update <= 0:
            raise ValueError("rollout.games_per_env_per_update must be positive")
        if not 0.0 <= cfg.model.planet_rope_fraction <= 1.0:
            raise ValueError("model.planet_rope_fraction must be in [0, 1]")
        if cfg.model.planet_rope_base <= 0.0:
            raise ValueError("model.planet_rope_base must be positive")
        try:
            normalize_attention_config(
                cfg.model.dim,
                cfg.model.n_heads,
                cfg.model.n_kv_heads,
            )
        except ValueError as exc:
            raise ValueError(f"model.{exc}") from exc
        if cfg.model.value_min >= cfg.model.value_max:
            raise ValueError("model.value_min must be less than model.value_max")
        if cfg.model.critic_mtp_horizon <= 0:
            raise ValueError("model.critic_mtp_horizon must be positive")
        if cfg.model.adv_scale <= 0.0:
            raise ValueError("model.adv_scale must be positive")
        if cfg.model.action_logit_softcap <= 0.0:
            raise ValueError("model.action_logit_softcap must be positive")
        if cfg.reward.production_weight < 0.0:
            raise ValueError("reward.production_weight must be non-negative")
        valid_reward_signals = {
            "projected_margin",
            "production_margin",
            "win_terminal",
        }
        if cfg.reward.signal not in valid_reward_signals:
            raise ValueError(
                "reward.signal must be one of "
                f"{sorted(valid_reward_signals)}"
            )
        if not 0.0 <= cfg.sac.builtin_prob <= 1.0:
            raise ValueError("sac.builtin_prob must be in [0, 1]")
        if not 0.0 <= cfg.sac.disc_target_entropy_ratio <= 1.0:
            raise ValueError("sac.disc_target_entropy_ratio must be in [0, 1]")
        if cfg.sac.alpha_discrete <= 0.0 or cfg.sac.alpha_continuous <= 0.0:
            raise ValueError("sac.alpha_discrete/continuous must be positive")
        if cfg.sac.alpha_discrete_lr <= 0.0 or cfg.sac.alpha_continuous_lr <= 0.0:
            raise ValueError("sac.alpha_discrete_lr/continuous_lr must be positive")
        from .league import BUILTIN

        if cfg.opponents.mode not in {"league", "fixed"}:
            raise ValueError("opponents.mode must be 'league' or 'fixed'")
        if not 0.0 <= cfg.opponents.self_play_prob <= 1.0:
            raise ValueError("opponents.self_play_prob must be in [0, 1]")
        if cfg.opponents.top_k <= 0:
            raise ValueError("opponents.top_k must be positive")
        if cfg.opponents.snapshot_every <= 0:
            raise ValueError("opponents.snapshot_every must be positive")
        unknown_fixed = set(cfg.opponents.fixed_opponents) - set(BUILTIN)
        if unknown_fixed:
            raise ValueError(
                f"opponents.fixed_opponents has unknown agents "
                f"{sorted(unknown_fixed)}; valid: {sorted(BUILTIN)}"
            )
        if cfg.opponents.mode == "fixed" and not cfg.opponents.fixed_opponents:
            raise ValueError(
                "opponents.mode='fixed' requires non-empty opponents.fixed_opponents"
            )

        unknown = set(cfg.sac.builtin_opponents) - set(BUILTIN)
        if unknown:
            raise ValueError(
                f"sac.builtin_opponents has unknown agents {sorted(unknown)}; "
                f"valid: {sorted(BUILTIN)}"
            )
        if (
            cfg.opponents.mode == "league"
            and cfg.sac.builtin_prob > 0.0
            and not cfg.sac.builtin_opponents
        ):
            raise ValueError(
                "sac.builtin_prob > 0 requires a non-empty sac.builtin_opponents"
            )
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
