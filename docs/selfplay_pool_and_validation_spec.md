# Self-Play Pool and Validation Spec

## Goal

Train one policy as fast as possible while avoiding self-play collapse, without
spending rollout budget on opponent-vs-opponent rating games. Progress should be
measured against a stable full-horizon opponent panel, not against the mutable
training pool.

This design intentionally uses no fixed or built-in bots in the training mix.
All training opponents are the current learner or snapshots of learned policies.

## Non-Goals

- Do not run active-pool members against each other just to maintain Elo.
- Do not use fixed/built-in agents as training opponents.
- Do not make validation depend on the current active training pool.
- Do not select validation archive members by top-K Elo.
- Do not use training return as the main full-horizon progress metric.

## Pools

There are four identities in the system, with different purposes.

### Current Learner

The live model being updated by PPO.

Purpose:
- Primary self-play pressure.
- Defines the current curriculum difficulty for snapshot measurement.

Sampling target:
- 40% of opponent slots.

### Active Training Pool

A mutable set of learned snapshots selected for training utility against the
current learner.

Purpose:
- Provide non-stationary but useful opponents.
- Keep the learner from overfitting to only its exact current policy.
- Prefer opponents near the learner's current skill, not necessarily the
historically strongest snapshots.

Sampling target:
- 30% of opponent slots.

Rating rule:
- Ratings/statistics are updated only from games where the current learner plays
  that snapshot.
- No snapshot-vs-snapshot games are scheduled for pool maintenance.

Retention should favor:
- snapshots with win rate near 50% against the current learner,
- snapshots with high uncertainty,
- recent snapshots,
- a small number of hard current opponents.

### Historical Training Archive

A mutable anti-forgetting archive of learned snapshots, retained mostly by
log-spaced age and milestone rules.

Purpose:
- Expose the learner to older strategies.
- Prevent forgetting and cyclic collapse.
- Keep long-horizon strategic diversity without requiring internal archive Elo.

Sampling target:
- 30% of opponent slots.

Retention is not top-K. It is stratified:
- log-spaced historical snapshots,
- recent active-pool evictions,
- previous training bests,
- notable collapse/recovery/specialist policies discovered organically.

The archive can change over time. It is a training device, not a validation
metric.

### Validation Archive

A stable, versioned panel of learned snapshots used only to measure progress.

Purpose:
- Full-horizon model-selection metric.
- Detect forgetting and non-transitive regressions.
- Provide a stable score that does not move with the active training curriculum.

Sampling target:
- 0% during normal training.

The validation archive must be frozen for long windows. When changed, it becomes
a new version and scores are not compared directly across versions without
labeling the archive version.

Retention is not top-K. It is fixed-panel and versioned:
- log-spaced snapshots from previous strong runs,
- previous submission candidates,
- previous validation winners,
- representative old strategies from different training phases.

## Training Opponent Mix

For each opponent seat, sample independently from:

```text
40% current learner
30% active training pool
30% historical training archive
```

If a pool is empty, redistribute its probability proportionally over the
non-empty learned-policy pools. Early in training this means mostly current
learner plus active snapshots once they exist.

For 4-player games, avoid filling all opponent seats with the same non-current
snapshot when enough alternatives exist. Diversity within a game is preferred,
but do not add expensive scheduling constraints.

## Snapshot Cadence

Create candidate snapshots frequently enough that the active pool can churn.

Recommended initial settings:

```text
snapshot_every_updates: 1
active_pool_size: 12-20
historical_training_archive_size: 64-256
validation_archive_size: 16-64
```

The active pool does not need every snapshot. Every snapshot may first enter a
candidate buffer, then either:
- enter the active pool,
- enter the historical training archive by log-spaced retention,
- be discarded.

## Active Pool Statistics

Each active snapshot tracks statistics only against the current learner:

```text
games_vs_current
wins_vs_current
draws_vs_current
losses_vs_current
mean_margin_vs_current
last_sampled_update
created_update
entered_active_update
```

The current learner changes after every PPO update, so these statistics are
non-stationary. Treat them as curriculum signals, not absolute strength.

Use an EMA or decay window so old results against older learner versions do not
dominate:

```text
effective_result_ema
effective_margin_ema
effective_games
```

## Active Pool Utility

Active-pool retention should maximize training utility, not Elo.

A simple utility score:

```text
win_rate = wins_vs_current / max(1, games_vs_current)
difficulty = 1 - 2 * abs(win_rate - 0.5)
uncertainty = 1 / sqrt(1 + games_vs_current)
recency = exp(-(current_update - created_update) / recency_half_life)
hardness = clamp(0.5 - win_rate, 0, 0.5)

utility =
    difficulty_weight * difficulty
  + uncertainty_weight * uncertainty
  + recency_weight * recency
  + hardness_weight * hardness
```

Interpretation:
- `difficulty` keeps opponents near 50%.
- `uncertainty` gives new snapshots enough exposure.
- `recency` keeps the pool from becoming stale.
- `hardness` preserves a few opponents the learner still struggles against.

Initial weights:

```text
difficulty_weight: 1.0
uncertainty_weight: 0.25
recency_weight: 0.25
hardness_weight: 0.5
recency_half_life_updates: 50
```

When the pool is over capacity, evict the lowest-utility snapshot, unless it is
protected by minimum exposure.

Minimum exposure:

```text
min_games_before_eviction: 16
```

If all over-capacity snapshots are under minimum exposure, evict the oldest
under-exposed snapshot.

## Historical Training Archive Retention

The historical training archive should be broad and cheap. It should not need
ratings among archive members.

Use log-spaced retention by update number. Keep snapshots closest to target
ages:

```text
age_targets_updates = [
  1, 2, 4, 8, 16, 32, 64, 128, 256, 512, ...
]
```

For each target age, retain the snapshot whose age is closest to that target.
Age is:

```text
age = current_update - snapshot_update
```

Also retain:
- recent active-pool evictions, up to a cap,
- previous validation winners,
- manually marked notable snapshots.

Suggested composition:

```text
50% log-spaced milestones
25% recent active-pool evictions
25% previous bests / notable policies
```

Sampling from the historical training archive should be stratified over those
buckets, not uniform over all files.

## Validation Archive

The validation archive is a stable panel.

It should be stored as a versioned manifest:

```yaml
version: val_archive_v1
created_update: 0
members:
  - name: run_a_u0001
    path: checkpoints/archive/run_a_u0001.pt
    weight: 1.0
    bucket: early
  - name: run_a_u0016
    path: checkpoints/archive/run_a_u0016.pt
    weight: 1.0
    bucket: mid
```

Weights define the validation opponent distribution. Keep them fixed within a
version.

Do not compare raw validation scores across different archive versions without
including the version in the metric name:

```text
validation/v1/weighted_win_rate
validation/v2/weighted_win_rate
```

## Validation Metrics

For each evaluated checkpoint, play only:

```text
candidate checkpoint vs validation archive members
```

Do not run validation archive members against each other during normal training.

Primary metric:

```text
weighted_validation_win_rate
```

Secondary metrics:

```text
weighted_validation_margin
bucket_win_rate_min
bucket_win_rate_p10
bucket_margin_min
per_bucket_win_rate
per_bucket_margin
```

The anti-collapse metric is:

```text
bucket_win_rate_p10
```

or, if buckets are few:

```text
bucket_win_rate_min
```

This prevents a checkpoint from looking good by farming one part of the archive
while forgetting another.

## Validation Elo

Validation Elo should be derived only from candidate-vs-validation games.

No new archive-vs-archive games are required for routine evaluation.

There are two acceptable forms:

### Fixed-Panel Pseudo-Elo

Given fixed archive member ratings from a prior offline calibration, solve for
candidate rating `R` such that:

```text
sum_i weight_i * sigmoid((R - rating_i) / scale)
  =
observed_weighted_score
```

This is one-dimensional root finding.

This number is only meaningful within a validation archive version.

### Cached Matrix Elo

Occasionally, offline, compute or update a cross-play matrix among validation
members and important historical candidates. Fit Bradley-Terry/Elo from the
cached matrix.

This is optional and should not be part of the hot training loop.

Routine checkpoint selection should use fixed-panel win rate and bucket lower
quantiles, not depend on fresh matrix Elo.

## Evaluation Cadence

Use two evaluation tiers.

Cheap validation:

```text
every_updates: 5
games_per_member: 4-8
archive_members: fixed subset or all with low games
```

Full validation:

```text
every_updates: 25
games_per_member: 16-64
archive_members: all
also_run_when: cheap_validation_near_best
```

Save best checkpoint by:

```text
primary: full_validation_weighted_win_rate
tie_break_1: bucket_win_rate_p10
tie_break_2: weighted_validation_margin
```

## Logging

Training pool health:

```text
pool/active_size
pool/archive_size
pool/current_sample_frac
pool/active_sample_frac
pool/historical_sample_frac
pool/active_median_age
pool/active_update_span
pool/archive_update_span
pool/evictions_per_100_updates
pool/active_mean_win_rate_vs_current
pool/active_mean_abs_win_rate_minus_half
```

Validation:

```text
validation/<version>/weighted_win_rate
validation/<version>/weighted_margin
validation/<version>/bucket_win_rate_min
validation/<version>/bucket_win_rate_p10
validation/<version>/pseudo_elo
validation/<version>/games
```

Per-bucket validation:

```text
validation/<version>/bucket/<bucket>/win_rate
validation/<version>/bucket/<bucket>/margin
validation/<version>/bucket/<bucket>/games
```

## Failure Modes and Responses

### Active Pool Stops Churning

Symptoms:

```text
active_update_span is small
evictions_per_100_updates near zero
active_median_age high
```

Responses:
- increase snapshot cadence,
- increase recency weight,
- reserve active slots for recent snapshots,
- lower minimum exposure.

### Learner Beats Active Pool Too Easily

Symptoms:

```text
active_mean_win_rate_vs_current > 0.7
training win rate high
validation flat or down
```

Responses:
- sample more historical archive,
- increase hardness weight,
- promote hard historical opponents into active pool.

### Validation Improves on Average but Forgets Buckets

Symptoms:

```text
weighted_validation_win_rate up
bucket_win_rate_min or p10 down
```

Responses:
- increase historical archive sampling,
- promote forgotten-bucket snapshots into historical training archive,
- do not select this checkpoint as best unless the primary objective explicitly
  tolerates the regression.

### Training Overfits Validation Archive

Symptoms:

```text
validation improves but new archive version or submission result does not
```

Responses:
- keep validation archive out of training,
- add a second hidden validation archive for final selection,
- rotate validation versions only at explicit milestones.

## Implementation Plan

1. Add snapshot metadata:

```text
snapshot_id
path
created_update
source_run
pool_state: candidate | active | historical_training | validation
tags
```

2. Replace current top-K-only opponent pool with:

```text
CurrentLearnerSlot
ActiveTrainingPool
HistoricalTrainingArchive
ValidationArchive
```

3. Add training sampler:

```text
sample_opponent_slot():
  bucket = categorical({current: .4, active: .3, historical: .3})
  return sampled opponent from bucket
```

4. Update active-pool stats only from current-vs-snapshot games already played
   during training.

5. Add active-pool eviction by utility score.

6. Add historical archive retention by log-spaced age targets.

7. Add validation archive manifest and evaluator.

8. Add checkpoint selection by validation archive score, not live training Elo.

9. Add TensorBoard logging for pool health and validation metrics.

## Default Config Sketch

```yaml
opponents:
  mode: no_builtins
  current_learner_prob: 0.40
  active_pool_prob: 0.30
  historical_archive_prob: 0.30

  snapshot_every: 1
  snapshot_device: train
  active_pool_size: 16
  historical_training_archive_size: 128
  historical_sample_panel_size: 8
  historical_agent_cache_size: 8

  min_games_before_active_eviction: 16
  active_recency_half_life_updates: 50
  active_difficulty_weight: 1.0
  active_uncertainty_weight: 0.25
  active_recency_weight: 0.25
  active_hardness_weight: 0.5
  active_stats_ema_decay: 0.95

  # Optional bucket caps. Defaults reserve 25% each for recent evictions and
  # notables, leaving the remaining 50% for pinned log-age landmarks.
  recent_eviction_archive_size: null
  notable_archive_size: null

```

Archive validation is intentionally a separate CLI, not a `RunConfig` section:

```bash
python scripts/evaluate_archive.py \
  --ckpt checkpoints/<run>/final.pt \
  --manifest configs/validation_archive_v1.yaml \
  --games-per-member 32
```

Primary model-selection fields:

```text
select_by: weighted_win_rate
tie_breakers:
  1. bucket_win_rate_p10
  2. weighted_margin
```
