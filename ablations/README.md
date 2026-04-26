# Ablations

Each YAML in this folder is an **ablation matrix**: a base config plus a list of cells that override it. Run a matrix with:

```bash
python scripts/ablate.py --matrix ablations/<name>.yaml
```

Each cell becomes its own run under `runs/<matrix_name>/<cell.run.name>/<timestamp>` and gets its own checkpoint under `checkpoints/<matrix_name>/<cell.run.name>/`. A `<matrix>.results.json` lands beside the matrix when the sweep finishes.

## What lives here

| Matrix | What it answers |
|---|---|
| [`headline.yaml`](headline.yaml) | Did the modeling work pay off? Heuristic vs sniper vs trained transformer (small) vs trained transformer (medium). **Run this first.** |
| [`policy_v0.yaml`](policy_v0.yaml) | Architecture sweep (one knob per cell): depth, width, heads, dropout. |
| [`opponents_v0.yaml`](opponents_v0.yaml) | Opponent-pool composition: pure self-play vs mixed vs heavy-heuristic — the dial that most controls strategy diversity. |
| [`reward_v0.yaml`](reward_v0.yaml) | Reward-shaping sweep: capture/loss bonuses, margin scale, terminal-only rewards. |
| [`ppo_v0.yaml`](ppo_v0.yaml) | PPO knobs: clip, gamma, lr, entropy coef, value coef. |

## Conventions

- **One seed per cell by default.** If a result lands close to a neighbor and the call matters, reseed *that one cell* by editing `run.seed` and rerunning — much cheaper than always multiplying every matrix by 3.
- **Run order**: `headline` → `policy_v0` → `opponents_v0` → `reward_v0` → `ppo_v0`. Architecture first to lock the model; then opponents (the biggest signal in this competition); then reward shaping; PPO knobs last because they move the score least.
- **One knob per cell** in every matrix except `headline`. Combine knobs only after you've identified the top 1–2 settings on each axis.
- **Name cells descriptively** (`depth_4`, `pool_self_only`) so tensorboard subdirs read like a story.
- **Record the conclusion** in `<matrix>.results.md` after a sweep: which cell won, by how much, against which opponents, and what to try next.

## Running a single config (no matrix)

```bash
python scripts/train.py --config configs/ppo_base.yaml
python scripts/evaluate.py --ckpt checkpoints/ppo_base/final.pt
```

## Reading results

```bash
tensorboard --logdir runs
```

The `train/win_rate` and `train/margin` scalars give you per-update progress; the per-baseline win-rates from `scripts/evaluate.py` tell you the absolute story.
