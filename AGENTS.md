# AGENTS.md

You are an expert machine learning researcher. Your goal is to maximize elo results using reinforcement learning on a transformer model.

## What this repo is

Code for the **Orbit Wars** Kaggle simulation competition. We submit an agent (a Python callable) that plays a real-time strategy game on a 100×100 continuous board — capture planets, manage fleets, dodge the sun. Submissions are ranked by Glicko-style skill rating from games against other submitted bots.

- Competition page: https://www.kaggle.com/competitions/orbit-wars
- Sponsor: Google LLC.
- Tagline: *"Conquer the void."* — successor to the 2010 Planet Wars challenge.
- Prize pool: $50,000 split as 10 × $5,000 (1st through 10th place).
- **Start**: 2026-04-16. **Final submission**: 2026-06-23 23:59 UTC. **Final play period**: 2026-06-24 → ~2026-07-08 (continued matches until ratings converge).
- Format: simulation competition (Kaggle Environments). Submit `main.py` (or a tar.gz with `main.py` at the root).

## What we're optimizing

- **Per-game objective**: total ships at end (planets + fleets). Highest wins; if only one player has any ships left, the game ends early.
- **Submission scoring**: each submission gets a Gaussian rating N(μ, σ²), initialized at μ₀ = 600. Plays games against similar-rated bots; μ rises with wins, falls with losses, σ shrinks over time. Only the latest 2 submissions per team count toward final scoring.
- **Local proxy**: win-rate and mean ship-margin against a fixed pool of opponents (random, sniper, heuristic). The Kaggle ladder is the only thing that decides prizes, but it's noisy and slow — local proxies are how we iterate.

## Hard rules / restrictions (confirmed against live page)

1. **`main.py` at the bundle root** with an `agent(obs)` function (or a callable equivalent). Bundles are tar.gz; the runner imports `main` and calls `main.agent`.
2. **No ingress or egress.** During an episode, the submission may not pull in or send out *any* information beyond the observation/action protocol. Pre-pack model weights inside the bundle.
3. **Time budget**: `actTimeout = 1` second per turn (default). The simulator gives no extra credit for fast inference, but a turn that exceeds the budget consumes the per-game overage budget; running out → loss.
4. **Submission limit**: 5 per day per team. Up to 2 final submissions count.
5. **Team size**: max 5.
6. **External data**: allowed if freely & publicly available, or under a "reasonable" cost threshold per the General Rules. Pre-trained models are OK (subject to license).
7. **Code license** if you win: CC-BY 4.0 on the submission; competition data is Apache 2.0.
8. **Validation episode**: every submission first plays itself; if it crashes, the submission is marked Error. Pull logs with `kaggle competitions logs <EPISODE_ID> 0`.

## Repo conventions

- **Configs are the source of truth.** A run is `python scripts/train.py --config configs/<name>.yaml`. Don't bury hyperparameters in code.
- **Ablations are configs that override a base config.** See `ablations/`. Run them via `scripts/ablate.py`; one tensorboard subdir per cell.
- **Tensorboard, always.** Every training run writes to `runs/<config_name>/<timestamp>/`. Scalars: policy loss, value loss, entropy, KL, SPO penalty, win-rate per opponent, mean margin. Histograms: predicted-target distributions, fraction-of-garrison.
- **Don't commit data, checkpoints, runs, or `submission*.tar.gz`** — `.gitignore` covers them.
- **The default model is in `src/owars/policies/model.py`** — a small set-transformer over planet/fleet tokens with target-attention and a Beta-distributed fraction head. See `STRATEGY.md` for the why.
- **The runtime entry point is `submission/main.py`.** It vendors `src/owars/` and lazy-loads weights — the bundle runs offline by design.
- Don't concern yourself with backwards compatiQbility with old weights and architectures

## Game spec (confirmed)

Board: 100×100 continuous. Sun at `(50, 50)` with radius 10 (fleets crossing it are destroyed). Planets are placed with 4-fold mirror symmetry. Map has 20–40 planets in 5–10 symmetric groups; ≥3 groups static, ≥1 group orbiting.

- **Planet** = `[id, owner, x, y, radius, ships, production]`. `owner = -1` is neutral. `radius = 1 + ln(production)`. Production ∈ [1, 5]; each turn an owned planet generates `production` ships.
- **Orbiting planets**: those whose `orbital_radius + planet_radius < 50` rotate at a per-game-randomized angular velocity ∈ [0.025, 0.05] rad/turn.
- **Fleet** = `[id, owner, x, y, angle, from_planet_id, ships]`. Fleet speed: `1 + (max_speed − 1) · (log(ships) / log(1000))^1.5`. Default max_speed 6.0; ~1000 ships hit the cap.
- **Comets**: temporary planets that spawn in groups of 4 at steps 50/150/250/350/450 on highly elliptical paths. Production 1 ship/turn, radius 1.0.
- **Action format**: list of `[from_planet_id, angle_radians, num_ships]`. You can only launch from your own planets, can't launch more than the planet has, can launch multiple per turn. Empty list = no-op.
- **Combat**: when fleets hit a planet, group by owner, biggest attacker fights second biggest (winner = ship-difference). Surviving attacker fights the garrison; if surplus exceeds garrison, ownership flips.
- **Game ends**: at step 500 or when only one player (or zero) has any planets/fleets left.

The full canonical reference is in `data/raw/README.md` (the official "How to Play" doc).

## The standard loop

1. Pick or write a config in `configs/`.
2. `python scripts/train.py --config configs/<x>.yaml` → checkpoint + tensorboard.
3. `python scripts/evaluate.py --ckpt <path> --games 100` → win-rate vs each fixed baseline.
4. If it's better than the previous best, build a submission with `scripts/bundle.py` and submit via the Kaggle CLI.
5. Record what changed in `ablations/<sweep>.results.md` if it was part of a sweep.

## Things to be paranoid about

- **Symmetry tests.** A bot that wins 60% playing seat 0 and loses 60% playing seat 1 has *latent advantage from initial conditions*, not a strategy. Always evaluate across random seat assignments.
- **Self-play strategy collapse.** Pure self-play converges to whatever the gradient is shaped like that day. Mix in heuristic/sniper/random opponents (see `OpponentsCfg`) to keep the policy honest.
- **Snapshot diversity.** A frozen snapshot from update 50 plays a *meaningfully different* policy than the current one; without snapshots, opponent diversity narrows over time. `OpponentsCfg.snapshot_every` controls this — too rare and the pool stays stale, too frequent and it grows unbounded.
- **Reward shaping vs win-rate divergence.** Heavy `capture_bonus` makes the policy farm planets but ignore endgame defense. Always sanity-check final win-rate against a *different* shaping than what the model was trained on.
- **Sun collisions.** Easy to overlook — a fleet sent toward an attacking position can clip the sun and disappear. The geometry helpers (`line_circle_intersects`) exist precisely so we can pre-screen launch angles in `agents/heuristic.py` and in the policy's sampler.
- **Time budget.** The runner kills slow turns. Inference must comfortably fit in <100ms; the bundle ships `LearnedAgent` in `deterministic=True` mode for this reason.
- **Submission validation episode.** Always run `python submission/main.py` locally before uploading. A bundle that errors out in validation just wastes a daily submission slot.

## Quick links

- `STRATEGY.md` — modeling plan, architecture rationale, ablation plan
- `configs/baseline_heuristic.yaml` — sanity baseline (no learning)
- `configs/ppo_base.yaml` — first PPO config (the "main" 2-player model)
- `configs/ppo_4p.yaml` — 4-player FFA variant
- `ablations/` — `headline.yaml` (heuristic vs PPO variants), `policy_v0.yaml` (architecture), `opponents_v0.yaml` (pool composition), `reward_v0.yaml` (shaping), `ppo_v0.yaml` (PPO knobs)
- `scripts/train.py` / `scripts/evaluate.py` / `scripts/ablate.py` / `scripts/bundle.py` / `scripts/selfplay.py`
