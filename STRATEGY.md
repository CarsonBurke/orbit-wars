# Strategy & Architecture Notes

This is the modeling plan for Orbit Wars. It's a living doc — update it as we learn what works.

## Reference baselines (planned head-to-head, 2-player, 100 games per matchup)

| Matchup | win-rate (P0) |
|---|---|
| `random` vs `random` | ~0.50 |
| `sniper` vs `random` | should be ≥0.95 |
| `heuristic` vs `random` | ≥0.95 |
| `heuristic` vs `sniper` | ≥0.65 (production-aware target picking should beat nearest-only) |
| **`ppo_base` vs `heuristic`** | **target: ≥0.65 to be worth submitting** |

These numbers are aspirational until we run them — they're the expectations against which the headline ablation will report. Fill in the realized numbers in `ablations/headline.results.md` once the sweep completes.

**The bar is `heuristic`, not `random`.** A random opponent collapses too easily — a policy that beats random 99% of the time can still lose to anything that picks targets sensibly. The heuristic is the load-bearing baseline.

## TL;DR

1. **Baselines**: `random`, `sniper` (nearest-planet starter), `heuristic` (production-weighted target picker with one-step orbit prediction). Free strategy floor.
2. **Main model**: a small set-transformer over `(planet, fleet)` tokens, with target-attention and a Beta-distributed fraction-of-garrison head; PPO trained against an opponent pool that mixes self, frozen snapshots, and the heuristic baselines.
3. **Architecture**: token-set encoder (no rasterization); shared transformer backbone for actor and critic. Per-owned-planet decode picks a target (or no-op) and a send-fraction. Orbit *parameters* (radius, angular velocity, current direction-of-motion) baked into planet features so the model can project to any horizon; owner one-hot is seat-relative (`self / neutral / enemy_0..2`) for FFA symmetry. Fleets carry source-planet provenance.
4. **Critic**: VAPO/VC-PPO-style. ±1 terminal reward, γ=1, **value-pretraining** against a frozen behavior policy (heuristic) before PPO turns on, **decoupled GAE** (λ_critic=1 → MC return, λ_policy=0.95 for variance reduction), token-level (per-owned-planet-step) policy loss.
5. **Validation is local self-play vs fixed baselines.** Win-rate per opponent + mean ship-margin. Kaggle ladder is the only thing that ranks for prizes, but it's noisy and slow.
6. **Ablate aggressively, in a matrix.** One config knob at a time, each cell is its own tensorboard run.

## Why a set-transformer here

The simulator gives us:
- A small, *typed*, variable-size set of entities each turn: 20–40 planets + a handful of fleets. Not a grid, not a sequence.
- A 100×100 *continuous* coordinate space. Rasterizing to a CNN throws away precision exactly where the action space wants it (angles).
- Strong relational structure: "send from this planet, towards that planet, accounting for what those other fleets are doing".

A vanilla MLP can't handle variable counts without a fixed-padding hack. A graph net would work, but for ≤300 tokens (planets + fleets) the simplicity of full self-attention is hard to beat. So:

- **Tokens**: each planet and each fleet becomes one token, with a per-type linear projection into `dim`.
- **Self-attention**: every token attends to every other token. Owner-relative one-hot lets the model treat enemies symmetrically.
- **Per-planet target attention**: each *owned* planet's encoded representation is the query; every planet's representation is a key. Logits over all planets + a no-op slot. This is the natural inductive bias for "pick a target".
- **Fraction-of-garrison head**: a small Beta(α, β) over `[0, 1]`. Continuous, bounded, and `α, β > 1` lets us decode a deterministic mode at inference time.

## Architecture (in `src/owars/policies/model.py`)

```
       planet tokens ──┐
                       ├─► linear ─► [N × Transformer block] ─► token reps
       fleet tokens  ──┘                                              │
                                                                      ├─► policy:
                                                                      │     for each owned planet,
                                                                      │     attend over all planets
                                                                      │     to score targets;
                                                                      │     plus a Beta head for
                                                                      │     fraction-of-garrison.
                                                                      └─► value: pool over real
                                                                            tokens, MLP scalar.
```

Key choices:

- **Per-token features encoded once** (`policies/features.py`):
  - **Planet token** (19 dims): `(x, y)` normalized, distance-to-sun, radius, log-ships, production, **direction-of-motion `(cos, sin)`** (orbit tangent for orbiters, path-step direction for comets, zero for static), motion speed, orbital radius, normalized angular velocity, `is_orbiting` and `is_comet` bits, owner one-hot, planet-token marker.
  - **Fleet token** (15 dims): `(x, y)`, heading `(cos, sin)`, log-ships, **source-planet `(x, y)` + has-source bit** (provenance — "this fleet originated there"), fleet's own speed, owner one-hot, planet-token marker (=0).
- **Orbit parameters, not predicted positions.** We hand the model `(orbital_radius, angular_velocity, direction-of-motion)` and let it project forward as needed. Cheaper, cleaner, and lets the model trade off horizon vs. confidence implicitly.
- **Seat-relative owner one-hot**: `[self, neutral, enemy_0, enemy_1, enemy_2]`. Enemy slots are stable under `(owner - player) mod 4`, so a 4-player FFA sees three canonical enemy slots. No "ally" slot — the competition is FFA / 1v1.
- **Padded fixed caps** (`MAX_PLANETS=64`, `MAX_FLEETS=384`) plus boolean masks. 64 covers 40 base planets + ~8 comet overlap with comfortable headroom; 384 covers heavy 4-player endgame. Batching is a stack-of-tensors with masks, no per-step `pack_padded`.
- **Action factorization** (`policies/sampling.py`): `(target_planet | no-op) × Beta(fraction)`. The target is a discrete categorical (cheap, no exploration headache); the fraction is continuous so we don't have to bucket "send 12.5% of garrison". Angle is *derived* from the target bearing, not predicted — adding a continuous angle head almost always degenerates to "aim at center of mass".

## Position sizing → action sizing

Hull Tactical's "Kelly under a vol cap" pattern doesn't apply here directly — the simulator scores by total ships, not Sharpe — but the *spirit* (ration the resource you have, against the variance of the outcome) does:

- **Reserve a garrison** (in `HeuristicAgent`) so a planet can't be one-shot the turn we attack from it.
- **Aim at predicted position** for orbiting targets (1-step prediction baked into features; the policy can learn to over/under-shoot if it wants to lead more).
- **Don't drain to zero** — the Beta head's mode is at `(α-1)/(α+β-2)`, which the `fraction_concentration` knob biases away from 0/1 endpoints.

## Self-play loop & opponent pool

The single biggest knob in this competition. Pure self-play converges fast but narrow. Pure-baseline play (no self) is too easy and the policy stops improving once it dominates the heuristics.

The default `OpponentsCfg.pool = ["self", "heuristic", "sniper", "random"]` is the equal-weight balanced mix. `opponents_v0.yaml` ablates this:

- `pool_self_only`: pure self-play. Fast convergence, narrow strategies. Expect a clean self-play win-rate (~0.5 by construction) but *worse* generalization to baselines than the mixed pool.
- `pool_self_heavy`: 3 self-play slots + 1 heuristic. Should land closest to the balanced mix in self-play strength but beat it slightly in pool-vs-baselines.
- `pool_baselines`: no self-play. Easy to dominate; the policy plateaus once it routinely beats heuristic.
- `pool_with_frozen` + `snapshot_every: 10`: heaviest opponent diversity. Most expensive (frozen snapshots are still neural nets).

`OpponentPool.sample()` picks uniformly by default. If we ever see strategy collapse during a run, the next move is *prioritized fictitious self-play* (PFSP): weight opponent-selection toward those we *barely* beat. That's a ~30-line change in `league.py`.

## Reward & critic — VAPO/VC-PPO

The default reward is **pure ±1 terminal** (win/loss/draw, no shaping). Per-turn reward = 0. This is the regime VAPO (arxiv 2504.05118) and its prerequisite VC-PPO (arxiv 2503.01491) were designed for, and the recipe for not-falling-on-our-faces is:

1. **γ = 1.0**: episodes are bounded ≤500 steps; there's no infinite-horizon variance issue and no reason to discount the only signal we have. `1/(1−γ) ≈ ∞` is fine here.
2. **Value pretraining (highest-leverage knob)**. Before PPO turns on, freeze a behavior policy (default: `heuristic`) and run `pretrain_updates` rounds of *critic-only* MSE: target = trajectory outcome (win → +1 for every state, loss → −1). With γ=1 this is the Monte-Carlo return; with terminal-only reward it collapses to "every state in this trajectory has target = the eventual game outcome". Tracked metric: **explained variance** of the value head; we want to see it climb to >0.1 before unfreezing the actor. (VC-PPO §3.2; VAPO §4.1, ablation cost: 60 → 11 if removed.)
3. **Decoupled GAE**. During PPO, compute two separate rollouts of GAE per batch: critic target uses **λ_critic = 1.0** (Monte-Carlo, unbiased — avoids biasing V toward 0 during cold start), actor advantage uses **λ_policy = 0.95** (variance-reduced). Provably non-biasing for the policy gradient (VC-PPO §3.3, eqs. 7–8). (VAPO ablation: 60 → 33 if removed.)
4. **Token-level (per-owned-planet-step) policy loss**. `loss = sum-over-(sample, planet) / count-of-active-planets` instead of mean-of-means. Stops long episodes from being down-weighted. Small effect for our bounded episodes, but free. (VAPO §4.2, eq. 7.)
5. **Length-adaptive λ_policy** is implemented as a flag (`lambda_policy_alpha`) but defaulted off (`α=0`). Episodes are bounded ≤500 with low length variance, so `λ = 1 − 1/(α·l)` lands close to a fixed `0.95` and the gain is small (VAPO ablation: 60 → 45, but in a regime with much higher length variance than ours).

`RewardCfg` retains shaping fields (`capture_bonus`, `loss_penalty`, `sun_loss_penalty`, `margin_scale`) all defaulting to 0. The `reward_v0.yaml` ablation flips them on as a fallback if pure terminal fails to learn — but the strong prior is that with value pretraining + decoupled GAE, it won't.

**The critic shares the transformer backbone with the policy.** Faster, fewer params, value-loss gradient helps the encoder. The trade-off (value-loss spikes destabilize the policy) is what value-pretraining fixes — by the time PPO starts, the value head's gradient is already well-conditioned.

## Validation protocol

- **Self-play train win-rate**: tensorboard `train/win_rate`. Centers around 0.5 by construction; useful as a "is the model still learning *something*" signal, not for absolute strength.
- **Held-out evaluation**: `python scripts/evaluate.py --ckpt <path> --games 100` plays N games against each fixed baseline. Report win-rate, draw-rate, mean margin, std margin.
- **Seat-symmetry sanity**: same checkpoint plays N games as P0 and N games as P1 against the same opponent. Win-rates should be within a few % of each other; otherwise our action format or encoder has a hidden seat-bias.
- **Across game configs**: 2-player vs 4-player. A model trained 2v2 may be terrible at 4-player FFA — different opponent dynamics, longer games, more comets. Track them separately.

## Ablation plan (first matrix → `ablations/headline.yaml`)

Each row is one knob; everything else stays at the base config.

| Knob | Values | Where |
|---|---|---|
| `model.depth` | 2, 3, 4, 6 | `policy_v0.yaml` |
| `model.dim` | 64, 96, 128, 192 | `policy_v0.yaml` |
| `model.n_heads` | 2, 4, 8 | `policy_v0.yaml` |
| `model.dropout` | 0.0, 0.1, 0.2 | `policy_v0.yaml` |
| `opponents.pool` | self-only / balanced / baselines / heuristic-only / + frozen | `opponents_v0.yaml` |
| `reward.*` | terminal-only / cap+loss / heavy / light / margin-only | `reward_v0.yaml` |
| `optim.lr` | 1e-4, 3e-4, 1e-3 | `ppo_v0.yaml` |
| `ppo.spo_eps_high` | 0.20, 0.28, 0.40 | `ppo_v0.yaml` |
| `ppo.gamma` | 0.99, 0.995, 0.999 | `ppo_v0.yaml` |

Read the matrix; then expand: pick the best 1–2 settings per row and run a 2-D combo of the top knobs.

## What we explicitly are *not* doing yet

- **No rasterization / CNN encoder.** The token-set encoder is the right inductive bias here; rasterizing throws away precision and adds parameters.
- **No PFSP / AlphaStar-style league play.** Plain pool sampling is good enough until we observe strategy collapse. PFSP is a ~30-line change in `league.py` when we need it.
- **No planning at inference time** (e.g., MCTS over candidate moves). Each turn is 1 second on Kaggle's runner; a single forward pass on a small set-transformer fits comfortably. Adding tree search trades inference speed for marginal strength.
- **No vendored simulator.** Each PPO update pays the kaggle-environments stepping cost. If the bottleneck shows up (likely once we run >10k games per update), the natural move is to write a numpy reimplementation matching the official rules — we already have the geometry/physics primitives in `src/owars/game/`.

## Confirmed (was: open questions)

1. **Action format**: list of `[from_planet_id, angle_in_radians, num_ships]`. Empty list = no-op. Confirmed against starter `main.py`.
2. **Observation contains `initial_planets` and `angular_velocity`** for predicting orbiting planet positions. We use `angular_velocity` directly in `features.py` — orbit params, not predicted positions.
3. **Comets follow normal planet rules** — captured/produced from like a planet, just with a fixed lifespan along a precomputed path; `comet_planet_ids` flags them in the obs. We extract the next-step direction vector from `obs.comets[].paths` per comet group.
4. **Validation episode** runs the agent against itself first; an error there marks the submission Error.

Still to characterize in `notebooks/`:
- Distribution of orbital vs static planet counts across maps (the spec says ≥3 static, ≥1 orbiting — what's the typical split?).
- Comet-spawn timing impact on win probability (do comet groups disproportionately advantage the leader?).
- Empirical move count per turn the policy emits at convergence.
