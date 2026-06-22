# Spec: 4-instance parallel learning league

**Status:** design / not yet implemented.
**Scope:** training only. No change to the model architecture, the action/obs
protocol, the Rust env, or the submission bundle. The submission is still one
`OrbitPolicy` checkpoint; this changes only how that checkpoint is *trained*.

## 1. Goal (the user's ask, verbatim intent)

> "Change in self learning strategy: have 4 separate weight paths/instances of
> the model that learn in parallel, like AlphaStar's league. Have them play
> against each other, never against themselves, all learning, 100% of the time.
> Random start positions and all that of course. Should have an elo for each,
> elo p1, elo p2, elo p3, elo p4."

Concretely:

- `N = 4` independent `OrbitPolicy` instances, each with its **own** weights and
  **own** optimizer state. All four train simultaneously and continuously.
- **100 %** of rollout games are inter-instance league games. There are **no**
  fixed builtin opponents in the learning signal and **no** frozen historical
  snapshots in the default configuration (both remain available as opt-in
  diversity knobs — see §9).
- Every seat in every game is filled by a **distinct** instance. Instance `i`
  never plays a copy of itself (not in 2p, not in 4p).
- Maps, orbital phases, seat assignment, comet timing are already randomized per
  episode by the Rust env (`random_seed` is per-env, and `alternating_learner_seats`
  rotates the recorded seat). We reuse that; no new randomization is needed.
- Each instance has a tracked Elo (`elo/p0`..`elo/p3`), updated from match
  outcomes and logged to TensorBoard.

This is the **simple symmetric** league (all four instances identical objective,
always learning), **not** AlphaStar's full main/main-exploiter/league-exploiter
taxonomy. The collapse risk that taxonomy was designed to fix is real here and is
addressed by recommendation, not by adding agent roles (§8, §9).

## 2. Why this is a real change, not a config flip

The entire rollout + PPO pipeline is built around exactly **one** live learner:

- `league.py` defines a single `LEARNER_NAME = "learner"`. Opponent seats are
  either that same identity (self-play, no agent callable — they fold into the
  learner forward) or a frozen `LearnedAgent` snapshot.
- `vec_rollout.rollout_episodes_batched` buckets every (env, seat) by *agent
  identity*; the learner bucket is whichever seats have `slot is None` or
  `slot.name == LEARNER_NAME` (`vec_rollout.py:756-762`, `:819-855`). It records
  trajectories **only** for those rows and runs them through **one** `model`
  forward via `_step_learner_bucket`.
- `train._ppo_loop` builds **one** `model`, **one** `optimizer`
  (`train.py:1648`, `:1704`), **one** `EloTracker` keyed on `LEARNER_NAME`
  (`:1706-1710`), **one** opponent `pool`, and calls `ppo_update(model,
  optimizer, batch, ...)` once per update (`:2086`). `_stack_trajectories`
  concatenates **all** recorded trajectories into a single batch.

For a 4-instance league, four *different* models each need their own recorded
trajectories, their own batch, and their own optimizer step. The hard part
(§5, §6) is generalizing the single-learner assumption in three places: the
rollout bucketing/recording, the per-instance trajectory partition, and the
update loop.

## 3. Architecture chosen

A new opponent mode **`opponents.mode = "league_parallel"`** plus a new config
block **`league:`** (§10). When active:

- `_ppo_loop` (or a new sibling `_league_loop`) holds a `list[OrbitPolicy]` of
  length `N`, a parallel `list[MultiOptimizer]`, a parallel list of
  reward/return normalizers, and one `EloTracker` whose identities are the
  instance names `p0..p{N-1}` (instead of `LEARNER_NAME` + snapshots).
- Each rollout wave: matchmaking (§4) assigns `num_players` **distinct**
  instances to each env's seats. The rollout runs **all** instances as live
  learners — every seat is a recorded learner row keyed by `(instance, env,
  seat)`, batched into that instance's forward.
- After the rollout, trajectories are partitioned by instance; each instance
  runs its own `compute_old_policy_dist` (PMPO) + `ppo_update` on **only** the
  seats it controlled.
- Elo is updated once per game from the seat→instance identities.
- The submission checkpoint is the highest-Elo instance (§7).

### Instance identity

Reuse the `LEARNER_NAME`-style string identity, but parameterize it:

```python
# league.py
def instance_name(i: int) -> str:
    return f"p{i}"          # "p0", "p1", "p2", "p3"
```

These strings flow through Elo exactly where `LEARNER_NAME` does today
(`_seat_names`, `elo.update_from_game`). The rollout's "is this a recorded
learner row" test changes from `slot.name == LEARNER_NAME` to `slot.name in
{p0..p{N-1}}` (§5).

## 4. Matchmaking

Per rollout wave, for each env we must pick `num_players` **distinct** instance
indices and assign them to seats. Two policies, behind
`league.pairing = "uniform" | "elo_matched"` (default `"uniform"`):

**Uniform (default, recommended to start):**
```python
def sample_match(n_instances, num_players, rng) -> list[int]:
    # distinct instance per seat; order *is* the seat assignment, already random
    return rng.sample(range(n_instances), num_players)
```
With `N = 4`:
- 4p: the assignment is a random permutation of `[0,1,2,3]` — all four play,
  every game, distinct seats. (`rng.sample(range(4), 4)` = a shuffle.)
- 2p: a random unordered pair, randomly seated — `C(4,2)=6` distinct matchups,
  each instance plays ~half the 2p games.

`rng.sample` already guarantees distinctness, so "never plays itself" is
structural, not a post-hoc filter.

**Elo-matched (opt-in):** for 2p, sample the first instance uniformly, then pick
the opponent with probability decreasing in `|elo_i - elo_j|` (softmax over
`-|Δelo|/spread`). For 4p, sample a quartet biased toward low intra-quartet Elo
variance. Keeps games informative once Elo spreads, at the cost of less coverage
of the win-rate matrix. Defer until uniform is shown healthy.

**Seat rotation / symmetry:** keep using `alternating_learner_seats(...,
offset=seat_offset)` semantics for *which* recorded seat is "primary" per env,
but note that in a full league **every** seat is recorded, so the
designated-primary concept is only needed for diagnostics (§6, "primary"). The
seat→instance map is the `sample_match` output; because that's a random
permutation/sample, seat assignment is already uniformly random across
instances, satisfying the "random start positions / seats" requirement and the
AGENTS.md symmetry-paranoia rule (an instance that's only strong from seat 0 will
show it in the per-seat Elo because seats rotate).

The matchmaker lives in a new `LeaguePool` class in `league.py` (§5), which
exposes `sample_match(num_players) -> list[int]` and owns the per-instance Elo
bookkeeping hooks. It replaces `NoBuiltinTrainingPool` for this mode; it does
**not** subclass it (the active/historical-utility machinery is about
learner-vs-frozen-snapshot, which has no analogue when all seats learn).

## 5. Rollout changes (the hard part #1)

`rollout_episodes_batched` and `_step_learner_bucket` must support **multiple
distinct live-learner identities in one wave**, each with its own model and its
own recorded trajectories.

Today's structure already has *almost* the right shape: it buckets seats by
identity and, for snapshot identities, runs a separate batched forward per
identity (`vec_rollout.py:996-1064`, the `for _name, bucket in opp_buckets`
loop, which calls `_step_learner_bucket(agent_model, ...)` with
`record_trajectories=False`). The change is to make **every** instance bucket a
*recorded* learner bucket against its own model.

### 5.1 New entry point: `rollout_league_episodes_batched`

Add a sibling to `rollout_episodes_batched` rather than overloading it (the
single-learner path is hot and used by eval/fixed/no_builtins; keep it
byte-for-byte). Signature:

```python
def rollout_league_episodes_batched(
    models: Sequence[OrbitPolicy],          # len N
    vec: VecEnv,
    seat_instances_per_env: list[list[int]],# [env][seat] -> instance idx, distinct per env
    *,
    num_players: int,
    device: str = "cpu",
    reward_cfg: RewardCfg | None = None,
    compile_mode: str | None = None,
    compile_fleet_width: int | None = None,
    snapshot_compile_rows: int = 64,
    defer_log_prob: bool = False,
    chunk_records: bool = False,
    timings: dict[str, float] | None = None,
    sample_timings: dict[str, float] | None = None,
) -> dict[int, list[Trajectory]]:           # instance idx -> its trajectories
```

`seat_instances_per_env[env]` is a length-`num_players` list of distinct
instance indices (the matchmaker output). There is no "learner seat" / "opponent
seat" distinction anymore — every seat is a learner seat for *some* instance.

### 5.2 Bucketing

Replace the identity test. For each `(env, seat)`:
```python
inst = seat_instances_per_env[env_idx][seat]
inst_buckets[inst].append((env_idx, seat, obs))   # obs via fast_observation, same as today
```
There is exactly one bucket per instance that appears in the wave (≤ N buckets).

### 5.3 Per-instance recorded forward

For each instance bucket, call `_step_learner_bucket(models[inst], bucket, ...,
record_trajectories=True)` writing into a per-instance trajectory dict keyed by
`(env, seat)`. This is the existing learner path, invoked N times with N
different models instead of once. Each call already:
- batches the bucket into one forward,
- records every row into its `(env, seat)` trajectory,
- supports the compiled CUDA-graph kernel (cached per-model via
  `_kernel_cache(model)` — keyed on the *model object*, so 4 models get 4
  independent kernel caches automatically; see §6.4).

The `record_trajectories=False` snapshot path and the builtin-agent path stay in
the function but are only reached if the opt-in diversity opponents are enabled
(§9); in the default pure-league config they're never hit.

### 5.4 Trajectory keys and finalization

Today trajectories are keyed `(env, seat)` and finalized from each recorded
seat's own perspective (`_finalize_trajectory`, `vec_rollout.py:1142-1145`). For
the league we key per instance:

```python
trajectories: dict[int, dict[tuple[int,int], Trajectory]]
```
Finalization is unchanged per trajectory (each is finalized from its seat's
perspective: margin = own_score − max(others)). The function returns
`{inst: [traj, ...]}`, where each instance's list is its recorded seats across
all envs it played in this wave.

Because seats rotate and pairings are random, an instance's trajectory list mixes
seat-0 and seat-1 (and 2/3) games — exactly the symmetry coverage we want.

## 6. Training-loop changes (the hard part #2)

A new `_league_loop` (parallel to `_ppo_loop`), or `_ppo_loop` branching on
`mode == "league_parallel"`. Recommendation: **new function**, because the
single-learner loop has substantial mode-specific bookkeeping
(`primary_trajs`, snapshot pool, KL-LR controller is shared) and forking it keeps
both paths readable. Shared helpers (`_stack_trajectories`, `_build_model`,
`_build_optimizer`, `compute_old_policy_dist`, `_refresh_batch_advantages_from_values`,
`ppo_update`, normalizer construction, metric helpers) are reused unchanged.

### 6.1 Setup

```python
models = [_build_model(cfg).to(device) for _ in range(N)]
for m in models:
    if device.type == "cuda":
        m.bfloat16(); restore_fp32_params(m)
    normalize_matrices(m)
optimizers = [_build_optimizer(m, cfg.optim) for m in models]
reward_norms = [_build_reward_normalizer(cfg) for _ in range(N)]   # per-instance
return_pct_norms = [_build_percentile_return_normalizer(cfg) for _ in range(N)]
elo = EloTracker(initial_rating=cfg.opponents.initial_rating, k_factor=cfg.opponents.k_factor)
for i in range(N):
    elo.ensure(instance_name(i))
pool = LeaguePool(n_instances=N, num_players_set=train_num_players,
                  pairing=cfg.league.pairing, elo=elo, rng=random.Random(cfg.run.seed))
```

Each instance keeps its **own** reward/return normalizer: the normalizers track a
running return RMS / percentile that's policy-specific, and four policies at
different skill produce different return scales. Sharing one would couple their
critic target scales.

KL-LR controller (`kl_lr_ema`, `kl_lr_scale`) becomes **per-instance** lists too,
since each instance's KL is its own. (Under PMPO the controller is inert anyway —
`train.py:2111-2120` — so for the default PMPO league this is bookkeeping that
stays constant; keep it per-instance for correctness if someone runs
`policy_objective: ppo`.)

### 6.2 Warm-start

`--load` currently loads one checkpoint into one model. For the league, support:
- `--load <ckpt>`: load the **same** checkpoint into **all** N instances, then
  perturb (§8) to break the symmetry that would otherwise make them identical
  and collapse matchmaking to a no-op.
- `league.init_paths: [ckptA, ckptB, ...]` (optional, len ≤ N): per-instance
  warm-start; unspecified instances are fresh-init. Recommended migration: warm
  all 4 from the current strong `ppo_selfplay_pmpo/final.pt` and rely
  on different `init_seed` per instance + rollout stochasticity for divergence
  (§8, §11).

### 6.3 Per-wave rollout + per-instance update

```python
for update in range(total_updates):
    # 1. matchmaking + rollout (all formats, like _format_episode_counts today)
    inst_trajs = {i: [] for i in range(N)}           # instance -> list[Trajectory]
    for num_players in format_order:                  # 2p / 4p mix, reuse _format_episode_counts
        for wave in waves(num_players):
            seat_instances_per_env = [
                pool.sample_match(num_players) for _ in range(rollout_envs)
            ]
            wave_trajs = rollout_league_episodes_batched(
                models, vec, seat_instances_per_env, num_players=num_players, ...)
            for i, trajs in wave_trajs.items():
                inst_trajs[i].extend(trajs)
            # Elo: one update per game from seat->instance identities
            for env_idx in range(rollout_envs):
                seats = [
                    (instance_name(seat_instances_per_env[env_idx][seat]),
                     <final score of that seat>)
                    for seat in range(num_players)
                ]
                elo.update_from_game(seats)            # §7

    # 2. per-instance PPO update
    for i in range(N):
        batch = _stack_trajectories(inst_trajs[i], gamma=..., gae_lambda=...,
                                    reward_normalizer=reward_norms[i],
                                    include_old_log_prob=True)
        batch = _trim_ppo_batch_fleet_width(batch, pad_to_bucket=...)
        if cfg.ppo.policy_objective == "pmpo":
            old_dist = compute_old_policy_dist(models[i], batch, ...)
            batch.update(old_dist into old_* keys)      # exactly as train.py:2026-2046
            _refresh_batch_advantages_from_values(batch, old_dist["value"], ...)
        log_i = ppo_update(models[i], optimizers[i], batch, ...)   # same kwargs as today
        logs[i] = log_i
```

The per-game "final score of that seat" comes from any one of the instance
trajectories for that env: `Trajectory.seat_rewards` is the full seat-score
vector (`_finalize_trajectory` sets `traj.seat_rewards = seat_scores`,
`vec_rollout.py:584`), identical across all seats of the same env, so read it
from the first recorded trajectory for that env.

### 6.4 Compile / CUDA-graph interaction

- `_get_rollout_kernel`/`_kernel_cache` key the compiled kernel on the **model
  object** (`model.__dict__["_owars_rollout_kernel_cache"]`, `vec_rollout.py:172-177`),
  so four distinct models get four independent compiled rollout kernels with no
  collision. Same for the PPO-update minibatch kernel inside `ppo_update`.
- Cost: 4× the compile time on the first update (each model traces its own
  `reduce-overhead` graph) and 4× the captured CUDA-graph memory. The model is
  tiny (dim 128, depth 3), so a captured graph is small; 4× is acceptable on an
  H100/A100. Quantify in §6.5.
- `_mark_cuda_graph_step` is called inside each `_step_learner_bucket`, so
  interleaving four models' captured graphs in one wave is fine — each replay is
  preceded by its own step-begin mark.
- Keep `cfg.rollout.compile_fleet_width` and `snapshot_compile_rows` shared; all
  four models are architecturally identical so their static shapes match.

### 6.5 Memory / compute budget (quantified)

- **Params:** checkpoint is ~2.6 MB (stated in AGENTS.md). 4 instances ≈ 10.4 MB
  of fp32/bf16 weights — negligible.
- **Optimizer state:** AdamW carries 2 moments/param; Muon carries momentum (+
  NorMuon 2nd-moment EMA). Call it ~3× param bytes per instance ⇒ ~4 × 3 × 2.6 MB
  ≈ 31 MB. Negligible.
- **Activations / rollout:** total recorded rows per update are **unchanged** —
  today one learner records ~`num_envs × num_players × seats-it-owns` rows; now
  the same total board-seats are split across 4 instances. Aggregate rollout
  compute is the same (every seat still does one forward); it's just bucketed
  into 4 forwards/wave instead of 1–2. The four PPO updates together process the
  same total row count as today's single update, so **per-update wall-clock for
  the optimization step is ~unchanged** (4 updates each on ~¼ the rows). The
  extra cost is (a) 4× kernel compile on update 0, (b) smaller per-instance
  batches ⇒ slightly worse GPU utilization per update, (c) 4× the
  `_stack_trajectories`/`compute_old_policy_dist` Python overhead. Expect a
  modest (~10–25 %) throughput hit, dominated by smaller batches and Python
  loop overhead, not by memory.
- **Mitigation for small batches:** bump `rollout.num_envs` so each instance's
  per-update batch stays in the same order as the current single-learner batch
  (`ppo_selfplay_pmpo.yaml` uses 128 envs locally; scale up on a bigger box).
  With N=4 and 2p, each instance sees ~half the envs'
  seats; with 4p, each sees ~one seat per env. Target ≥ the current
  `minibatch_size` (4096–8192) rows per instance per update by sizing
  `num_envs × num_players / N ≳ minibatch_size / mean_owned_planets`.

## 7. Per-instance Elo

Reuse `EloTracker` and `elo.update_from_game` unchanged — it already does exactly
what's needed:

- **2p:** one pairwise update, `K = k_factor` (the `M−1 = 1` divisor is a no-op),
  result by score comparison. This is textbook Elo (`elo.py:84-130`).
- **4p (ranking-based, defensible):** `update_from_game` runs **one pairwise
  update per distinct identity pair**, each at `K/(M−1) = k_factor/3`, scoring
  `sa ∈ {0, 0.5, 1}` by seat-score comparison, with all expectations computed
  against **pre-game** ratings and deltas applied after (`elo.py:111-130`). For
  4 distinct instances that's `C(4,2)=6` pairwise comparisons — i.e. a
  rank-based all-pairs (Elo "tournament") update. Each instance's total per-game
  K-budget is bounded to ~`k_factor`. This is a standard and defensible
  multiplayer Elo; no new code.

Because the league guarantees **distinct** identities per game, the
`len(identities) < 2` early-return (`elo.py:108-109`) never fires — unlike
self-play where all-self games are degenerate.

**TensorBoard keys:** the user asked for `elo p1..p4`. Use zero-based instance
names internally (`p0..p3`) but log under the requested human keys. Add to the
`"league"` scalar group (or a dedicated `"elo"` group):

```python
logger.scalars("elo", {f"p{i+1}": elo.get(instance_name(i)) for i in range(N)}, update)
```
(`elo/p1`..`elo/p4`, matching the user's wording). Also log `elo/spread`
(`max − min`) and `elo/std` as health metrics (§8).

`k_factor` and `initial_rating` come from the existing `OpponentsCfg.k_factor` /
`initial_rating` (default 32.0 / 1500.0), keeping the repo's conventions.

## 8. Diversity / collapse considerations

The user explicitly asked for the simple symmetric league, **not** AlphaStar's
main/exploiter/league-exploiter roles. We honor that, but flag the risk and
provide cheap diversity seeds:

**Risk:** four instances with identical objective, identical architecture, and
the same warm-start can converge to the **same** policy. If they do, every league
game is effectively self-play, Elo collapses toward equal (`elo/spread → 0`),
gradients narrow, and we lose the diversity benefit — the exact "self-play
strategy collapse" AGENTS.md warns about, just sharded four ways.

**Cheap diversity seeds (recommended defaults):**
1. **Different init seeds per instance.** Fresh-init instances use
   `cfg.run.seed + i` for parameter init; warm-started instances get a small
   Gaussian weight perturbation seeded by `i` (a "symmetry-break" jitter, e.g.
   `0.01 · ||w|| · N(0,1)` on trunk matrices, then `normalize_matrices`). Without
   this, four identical warm-starts produce identical gradients on identical data
   distributions and stay locked together. **This is the single most important
   anti-collapse measure and should be on by default.**
2. **Different rollout RNG streams** (already true — each env has its own seed,
   and matchmaking gives each instance a different opponent mix).
3. **Optional reward-shaping perturbation** per instance
   (`league.reward_jitter`): e.g. ± small `production_weight` offset, so
   instances pursue subtly different objectives (one slightly greedier on
   production, one on ships). Off by default; an open question whether this helps
   or just adds noise (§12).

**Health metrics that detect collapse (§9):** `elo/spread`, the pairwise
win-rate matrix off-diagonal mass, and a behavioral diversity proxy.

**Explicit non-goal:** we are **not** adding exploiter agents, frozen
main-agent checkpoints as obligatory opponents, or PFSP (prioritized fictitious
self-play) weighting in v1. Those are the documented next step if symmetric
diversity proves insufficient (§12).

## 9. Metrics (beyond Elo)

Log per update:
- `elo/p1..p4`, `elo/spread`, `elo/std` — primary skill + divergence signal.
- **Pairwise win-rate matrix** `winrate/p{i}_vs_p{j}`: accumulate per-pair
  game outcomes within the update (or an EMA across updates, since per-update
  pair counts are small). Off-diagonal near 0.5 *with* high Elo spread is
  contradictory and flags an Elo/aggregation bug; uniformly 0.5 with zero spread
  flags collapse. This is the league analogue of the per-opponent win-rate the
  single-learner loop logs.
- **Per-instance** `losses/*`, `policy/*`, `fraction/*` for each `i` (prefix
  scalars with `p{i}/`), so each instance's PPO health (KL, entropy, value EV) is
  visible. Reuse the existing scalar dicts from `ppo_update`'s `PPOLog`.
- **Strategy diversity proxy:** mean pairwise distance between instances'
  action distributions on a shared fixed probe batch (e.g. cache ~256 obs once;
  each update, run all 4 models on it and log mean pairwise JS-divergence of the
  per-planet target/launch distributions). Falling toward 0 ⇒ collapse. Cheap
  (256 rows × 4 forwards, no grad). Recommended but can be v1.1.
- Per-instance `rollout/win_rate` and `rollout/margin` are **not** meaningful in
  the symmetric-zero-sum league (they hover at 0.5 / 0 by construction, like
  self-play today — `train.py:1873-1877` notes this). Replace the headline
  health signal with Elo spread + win-rate matrix. Keep logging the global mean
  episode length and shaped return for sanity.

## 10. Config surface

New top-level block `league:` and a new `opponents.mode`. Keep configs as the
source of truth (repo convention). Add to `config.py`:

```python
@dataclass
class LeagueCfg:
    num_instances: int = 4
    pairing: Literal["uniform", "elo_matched"] = "uniform"
    # Symmetry-break perturbation applied to warm-started instances (and the
    # init seed offset for fresh ones). 0 disables (NOT recommended for a shared
    # warm-start — instances would stay identical).
    init_perturb_std: float = 0.01
    # Optional per-instance warm-start checkpoints (len <= num_instances).
    init_paths: list[str] = field(default_factory=list)
    # Optional per-instance reward-shaping jitter (off by default).
    reward_jitter: float = 0.0
    # Elo-matched pairing temperature (only used when pairing == "elo_matched").
    elo_match_spread: float = 200.0
    # Which instance is "the submission": "best_elo" (default) or a fixed index.
    submission_select: Literal["best_elo", "index"] = "best_elo"
    submission_index: int = 0
    # Probability a seat is replaced by a frozen historical snapshot / builtin
    # (opt-in diversity; 0 = pure inter-instance league per the user's ask).
    historical_prob: float = 0.0
    builtin_prob: float = 0.0
    builtin_opponents: list[str] = field(default_factory=list)
```

Wire `LeagueCfg` into `RunConfig` (a `league: LeagueCfg` field, default-factory),
add it to `from_dict`'s section dispatch (it already iterates `d.items()` and
`setattr`s — just add the dataclass), and add validation:
- `num_instances >= max(train_num_players)` (need ≥ `num_players` distinct
  instances to fill a game without repeats — for 4p that's `num_instances ≥ 4`).
- `0.0 <= init_perturb_std`, `len(init_paths) <= num_instances`,
  `submission_index < num_instances`.
- `opponents.mode == "league_parallel"` requires `ppo.pretrain_updates == 0`
  (same rule as `no_builtins`; value pretraining uses builtin behavior and there
  is no single learner to pretrain) — extend the existing check at
  `config.py:857`.
- Extend `opponents.mode` literal + the validation at `config.py:442` / `:749`
  to include `"league_parallel"`.

Provide a starter config `configs/league_4p_pmpo.yaml`: copy
`ppo_selfplay_pmpo.yaml`, set `opponents.mode: league_parallel`,
add the `league:` block with `num_instances: 4`, bump `rollout.num_envs` per
§6.5, and warm-start all 4 from the current self-play final checkpoint.

## 11. Checkpointing

- **Rolling latest:** every update, write `latest_p{i}.pt` for each instance via
  `_save_ppo_checkpoint(models[i], ..., reward_norms[i], return_pct_norms[i])`.
  These are `--load`-compatible single-model checkpoints.
- **The submission checkpoint:** every update, also write `latest.pt` = a copy of
  the current **highest-Elo** instance (or `submission_index` if
  `submission_select == "index"`). This is the file the bundler/eval consumes;
  the rest of the pipeline (`scripts/bundle.py`, `evaluate.py`) needs no change
  because it's a normal single-model checkpoint.
- **Final:** on completion write `final_p{i}.pt` for all, `final.pt` = best-Elo
  instance, and `elo.json` = `elo.snapshot_dict()` (already done at
  `train.py:2450`, now containing `p0..p{N-1}`).
- **Snapshots into a historical archive** are **not** written in pure-league mode
  (no frozen-snapshot pool). If `league.historical_prob > 0` is enabled later,
  each instance contributes snapshots to a shared archive at `snapshot_every`;
  out of scope for v1.

## 12. Migration / sequencing

Current state: `no_builtins` self-play, single learner, warm-started from the
sniper run, 2p then 2p+4p (`ppo_selfplay_pmpo*.yaml`).

Recommended sequencing:
1. **Land the plumbing** behind `mode: league_parallel` with `num_instances: 1`
   as a degenerate equivalence check — with one instance and `historical_prob =
   builtin_prob = 0` there are no legal opponents for 2p/4p, so this is only a
   wiring smoke test (assert it errors cleanly on "need ≥ num_players
   instances"). Real validation is `num_instances: 2`, 2p only: two instances,
   pure head-to-head, Elo should diverge then track relative skill.
2. **Warm-start all 4 from `ppo_selfplay_pmpo/final.pt`** with
   `init_perturb_std: 0.01`. Rationale: the current policy is the strongest
   asset; starting 3 fresh + 1 warm wastes compute re-learning basics and lets
   the warm one trivially dominate Elo early (low signal). Four perturbed copies
   of a strong policy diverge under different opponent mixes — the league's
   point.
   - **Alternative (open question §13):** 2 warm + 2 fresh, to inject more
     behavioral diversity at the cost of early Elo being dominated by the warm
     pair. Decide empirically.
3. **Start 2p-only**, then flip `train_num_players: [2, 4]` once 2p league Elo is
   healthy (spread grows, win-rate matrix sensible) — mirrors the existing
   2p→2p4p caution in the configs.
4. Keep the single-learner `no_builtins` path fully intact; `league_parallel` is
   additive. The two share `_build_model`, `_build_optimizer`, `ppo_update`,
   `compute_old_policy_dist`, `_stack_trajectories`, and all metric helpers.

## 13. Risks & open questions

**Risks:**
- **Collapse to identical policies** despite perturbation (§8). Mitigated by
  init perturbation + diversity metrics; if `elo/spread` stays ~0 and the
  diversity proxy decays, escalate to reward jitter or exploiter roles.
- **Throughput regression** from 4 smaller batches + 4× Python overhead (§6.5).
  Mitigated by raising `num_envs`. If still slow, consider a single batched
  forward across instances *only when shapes match* — but they're different
  weights, so it'd need a grouped/vmap forward (significant complexity; out of
  scope, noted as a future optimization).
- **Elo low-information at start:** with 4 equal warm-starts, early games are
  ~coin-flips and Elo barely moves until policies diverge. Expected; not a bug.
- **Non-stationarity:** every instance's opponents are also learning, so each
  faces a moving target (true of self-play too, but now 3 independent moving
  targets). PMPO's reverse-KL trust region (`pmpo_kl_coef`, `pmpo_target_kl`)
  already bounds per-update movement; no extra mechanism proposed.
- **Reward normalizer divergence:** four instances at different skill have
  different return scales; per-instance normalizers (§6.1) handle this, but a
  badly-diverged instance could get an ill-scaled critic. Monitor per-instance
  `reward_norm/*` and `explained_variance`.

**Open questions needing the user's decision:**
1. **Warm-start composition:** all-4-warm (recommended) vs 2-warm/2-fresh vs
   all-fresh? Affects early dynamics and diversity.
2. **Pairing policy:** uniform (recommended, full coverage) vs Elo-matched
   (informative games, less coverage)? Can start uniform and switch.
3. **`num_instances`:** the ask is 4. Confirm 4 is fixed, or do we want it
   configurable (the spec makes it configurable with `num_instances`, default 4).
4. **Diversity escalation policy:** if symmetric instances collapse, do we (a)
   add reward jitter, (b) add frozen-snapshot opponents (`historical_prob > 0`),
   or (c) go full AlphaStar exploiters? v1 ships the simple league; this decides
   the v1.1 lever.
5. **Submission selection:** highest-Elo instance (recommended) — confirm, vs
   always a fixed index, vs an ensemble/distillation of all four (out of scope
   but worth flagging as a possible v2).

## 14. File-by-file change list

- **`src/owars/training/config.py`**
  - Add `LeagueCfg` dataclass; add `league: LeagueCfg` field to `RunConfig`.
  - Extend `opponents.mode` `Literal` and the two `mode` validation blocks
    (`:442`, `:749`) to allow `"league_parallel"`.
  - Add `league` validation (instance count vs `train_num_players`, paths,
    perturb std, submission index) and the `pretrain_updates == 0` rule.
- **`src/owars/training/league.py`**
  - Add `instance_name(i)` helper.
  - Add `LeaguePool` class: `__init__(n_instances, num_players_set, pairing,
    elo, rng, ...)`, `sample_match(num_players) -> list[int]` (uniform /
    elo-matched), and (opt-in) snapshot/builtin seat replacement hooks.
- **`src/owars/training/vec_rollout.py`**
  - Add `rollout_league_episodes_batched(models, vec, seat_instances_per_env,
    ...) -> dict[int, list[Trajectory]]` (§5). Reuses `_step_learner_bucket`
    unchanged (call it once per instance bucket with that instance's model and
    `record_trajectories=True`).
- **`src/owars/training/train.py`**
  - Add `_league_loop(cfg, models, optimizers, elo, pool, logger, device, vecs,
    reward_norms, return_pct_norms)` (§6); branch `train_one_run` on
    `cfg.opponents.mode == "league_parallel"` to build the N-lists and call it.
  - Add `_init_league_models(cfg, device, load_weights)` (build N, warm-start /
    perturb per `LeagueCfg`).
  - Per-instance checkpoint writers (`latest_p{i}.pt`, best-Elo `latest.pt`,
    `final_p{i}.pt`); reuse `_save_ppo_checkpoint`.
  - Elo + TB logging (`elo/p1..p4`, `elo/spread`, pairwise win-rate matrix,
    per-instance `p{i}/losses/*`).
- **`configs/league_4p_pmpo.yaml`** (new): starter config (§10).
- **Tests:** `tests/` — a `LeaguePool.sample_match` distinctness/coverage test;
  a small `rollout_league_episodes_batched` smoke test on the numpy backend
  asserting each instance gets trajectories only from seats it controlled; an
  Elo 4p all-pairs update test (already partially covered by existing elo tests —
  extend to assert distinct-identity 4p behavior).

### Key function signatures that change or are added

```python
# league.py
def instance_name(i: int) -> str: ...
class LeaguePool:
    def sample_match(self, num_players: int) -> list[int]: ...   # distinct instances

# vec_rollout.py
def rollout_league_episodes_batched(
    models: Sequence[OrbitPolicy],
    vec: VecEnv,
    seat_instances_per_env: list[list[int]],
    *, num_players: int, device: str = "cpu",
    reward_cfg: RewardCfg | None = None,
    compile_mode: str | None = None,
    compile_fleet_width: int | None = None,
    snapshot_compile_rows: int = 64,
    defer_log_prob: bool = False, chunk_records: bool = False,
    timings: dict[str, float] | None = None,
    sample_timings: dict[str, float] | None = None,
) -> dict[int, list[Trajectory]]: ...

# train.py
def _league_loop(
    cfg: RunConfig,
    models: list[OrbitPolicy],
    optimizers: list[MultiOptimizer],
    elo: EloTracker,
    pool: LeaguePool,
    logger: TBLogger,
    device: torch.device,
    vecs: dict[int, VecEnv],
    reward_normalizers: list[DiscountedReturnNormalizer | None],
    return_pct_normalizers: list[PercentileReturnNormalizer | None],
) -> dict: ...
```

`ppo_update`, `compute_old_policy_dist`, `_stack_trajectories`,
`_build_optimizer`, `_build_model` are reused **unchanged** — they already take
`(model, optimizer, batch)` and are model-agnostic, so calling them in a loop
over instances is sufficient.
