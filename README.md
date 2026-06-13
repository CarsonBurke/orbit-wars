# Orbit Wars

Code for the [Orbit Wars](https://www.kaggle.com/competitions/orbit-wars) Kaggle simulation competition. Build an agent that conquers planets rotating around a sun in a 2-player or 4-player real-time strategy game; ranking is by skill rating from games against other submitted bots.

See [`AGENTS.md`](./AGENTS.md) for the brief future agents need, and [`STRATEGY.md`](./STRATEGY.md) for the modeling plan.

## Layout

```
src/owars/        package code (game, agents, policies, training, utils, submission)
configs/          YAML configs for runs and ablations
scripts/          entry points (train, evaluate, ablate, bundle, selfplay)
ablations/        ablation matrix definitions + results
runs/             tensorboard event files (gitignored)
checkpoints/      model checkpoints (gitignored)
data/             starter kit + cached competition files (gitignored)
submission/       runtime entry point (main.py) for Kaggle
tests/            unit tests
notebooks/        EDA + replay analysis notebooks
```

## Quickstart

```bash
# 1. Install (editable) into a venv
python -m venv .venv && source .venv/bin/activate
pip install -e '.[dev,kaggle]'

# 2. Pull the starter kit (requires Kaggle CLI + accepted rules)
python scripts/download.py
# or:
kaggle competitions download -c orbit-wars -p data/raw && unzip -o data/raw/orbit-wars.zip -d data/raw

# 3. Sanity-check the heuristic baseline
python scripts/selfplay.py --p0 heuristic --p1 sniper --episodes 5

# 4. Train PPO baseline
python scripts/train.py --config configs/ppo_base.yaml

# 5. Watch tensorboard
tensorboard --logdir runs

# 6. Evaluate the trained checkpoint
python scripts/evaluate.py --ckpt checkpoints/ppo_base/final.pt --games 50

# 7. Run an ablation sweep
python scripts/ablate.py --matrix ablations/headline.yaml
```

## Submitting

Use the helper script for real submissions. It builds the tarball, validates the extracted bundle locally, loads Kaggle credentials from the environment or `.env`, then uploads through the Kaggle CLI.

```bash
uv run --extra kaggle python scripts/submit.py \
  --ckpt checkpoints/ppo_sniper/latest.pt \
  --out submission_ppo_sniper_latest.tar.gz \
  --message "ppo_sniper latest checkpoint"
```

For a build + validation pass without upload, add `--dry-run`. To inspect current server status:

```bash
set -a; . ./.env; set +a
uv run --extra kaggle kaggle competitions submissions -c orbit-wars
```

See [`submission/README.md`](./submission/README.md) for runtime constraints (no internet, 1 second per turn, validation episode).
