# Learned Fleet→Planet Destination Attention (Option 3)

*How the inbound-fleet representation and learning capacity change with
`destination_learned_fleet_attention: true`, and confirmation that it does the
destination-scoped attention we intended.*

Status: implemented behind a config flag (default **off**), independently
reviewed, smoke-verified, and currently training as
`configs/ppo_sniper_pmpo_fleetattn.yaml` (flag **on**) vs the
`ppo_sniper_pmpo` baseline.

---

## TL;DR

- **Before:** each planet's inbound-fleet situation was compressed into a
  **fixed, hand-designed 13-number summary** (`_destination_fleet_stats` /
  the native-Rust `planet_inbound_feats`). The learned cross-attention module
  existed in the code but was **bypassed** in every compiled training/rollout
  path — its output was hard-zeroed.
- **After:** each planet **attends over the individual fleets inbound to it**
  (per-fleet tokens), producing a learned, content-weighted, `dim`-dimensional
  context vector — **in addition to** the 13 hand stats, which are retained.
- **Is it the destination attention we wanted?** **Yes.** Each fleet attends to
  *exactly its one destination planet* (block-sparse mask, not dense
  all-pairs). Planets query; their own inbound fleets are the keys/values.
- **Cost:** ~18% lower steady-state SPS (12008→9882 steps/s, ae95's
  measurement); the live run will give the definitive number. Capacity strictly
  increases; quality is the open empirical question the run answers.

---

## 1. The representation before

Code: `_destination_fleet_stats` (`src/owars/policies/model.py:1050`) and the
summary branch of `DestinationFleetConditioner.forward`
(`model.py:1173-1187`).

For every planet, the model pooled **all** fleets whose destination is that
planet into a **frozen 13-dimensional vector** (`_DEST_FLEET_STATS_DIM = 13`):

| # | Stat |
|---|------|
| 0 | total inbound count (norm) |
| 1 | self-owned inbound count |
| 2 | enemy inbound count |
| 3–5 | log total / self / enemy inbound **ship mass** |
| 6–8 | max per-fleet log-ship-mass: any / self / enemy |
| 9 | count-weighted mean speed |
| 10 | max speed |
| 11 | mean ETA (over fleets with known ETA) |
| 12 | known-ETA count |

This summary then drove a **FiLM** modulation of the planet token:

```python
# summary branch — fleet_ctx is ZERO:
fleet_ctx  = zeros(b, p, dim)                       # model.py:1186
fleet_stats = planet_inbound_feats   # the 13-dim Rust summary
gamma, beta = mod(cat([fleet_ctx, fleet_stats]))    # model.py:1243
conditioned = justnorm(h_p * (1 + gamma) + beta)    # model.py:1247
```

Two properties matter:

1. **The aggregation function is fixed, not learned.** Counts, sums, means, and
   maxes are hard-coded. The network cannot change *how* fleets are pooled —
   only how it reacts to the 13 resulting scalars.
2. **`fleet_ctx` was identically zero.** Because the compiled rollout/update
   always supplied `planet_inbound_feats`, the conditioner took this branch and
   the `DestinationFleetCrossAttention` module never ran in training. The first
   `dim` input columns of `mod` were fed zeros — so the *entire* learned signal
   about inbound fleets was those 13 hand-crafted numbers.

Anything the 13 aggregates discard was **invisible** to the policy: the joint
structure across individual fleets (e.g. "one 900-ship enemy fleet arriving in
3 turns" vs "nine 100-ship fleets spread over 40 turns" can look similar in
sums/means), per-fleet identity, or any nonlinear interaction the designer
didn't pre-compute.

---

## 2. The representation after

Code: `DestinationFleetCrossAttention` (`model.py:834-1047`) and the learned
branch of the conditioner (`model.py:1188-1248`), gated on
`self.learned_fleet_attention` (`model.py:1155`, set from
`OrbitPolicyConfig.destination_learned_fleet_attention`).

Now each **fleet is a token**. The conditioner runs real attention:

```python
# learned branch:
h_f         = justnorm(fleet_token_embeddings)       # model.py:1208
fleet_ctx   = self.cross_attn(h_p, h_f, ...)         # model.py:1209  (LEARNED, dim-wide)
fleet_stats = _destination_fleet_stats(...)          # model.py:1217  (13 hand stats RETAINED)
gamma, beta = mod(cat([fleet_ctx, fleet_stats]))     # model.py:1243
conditioned = justnorm(h_p * (1 + gamma) + beta)
```

Inside `cross_attn`:

- **Planet = query, fleets = keys/values** (`c_q` on planets, `c_k`/`c_v` on
  fleet tokens; `model.py:963-965`), all learnable, orthogonally initialized,
  weight-normalized, with nGPT hypersphere-QK normalization + per-channel `sqk`
  scale (`model.py:972-973`).
- The per-fleet **token** is the model's learned embedding of that fleet's raw
  features (position, owner one-hots, ship mass, speed, ETA, …) — far richer
  than a contribution to 13 scalars.
- The output `fleet_ctx` is a **learned, softmax-weighted combination** of the
  per-fleet value vectors of *that planet's* inbound fleets, projected by a
  learnable `out_proj`.

### What this buys in capacity

| | Before (summary) | After (learned attention) |
|---|---|---|
| Inbound-fleet signal | 13 fixed scalars | learned `dim`-wide context **+** the same 13 scalars |
| Pooling | hard-coded sum/mean/max | learned content-based softmax weighting |
| Per-fleet detail | averaged away | preserved as tokens; planet learns which to attend to |
| Strictly more expressive? | — | **Yes**: it can recover a fixed pooling as a special case and go beyond |

Crucially, the learned path **keeps** the 13 stats (`model.py:1217`). The
attention gives a content-weighted *average*; combat/timing also care about
*additive* quantities (total mass, counts) that a softmax average blurs (see the
docstring at `model.py:1056-1060`). So this is **learned attention *on top of*
the prior summary**, not a blind replacement — the policy gets both the
additive combat signal and a learnable, per-fleet-aware context.

Because `mod` is **zero-initialized** (`model.py:1157-1158`), both paths start
at `gamma = beta = 0` (no conditioning), so the flag-on model begins training
neutral and *learns* to use the new context.

---

## 3. Is it doing the destination attention we wanted?

**Yes — destination-scoped, not dense.** This was the explicit design goal
("attention against destination," not all-pairs planet×fleet).

The mask (`build_destination_block_mask`, `model.py:110-139`):

```python
def mask_mod(b, h, q_idx, kv_idx):
    return (dest[b, kv_idx] == q_idx) & valid[b, kv_idx]
```

Each fleet (key `kv_idx`) is visible to **exactly one** planet query
(`q_idx == its destination`), and only if it's a valid in-flight fleet. So:

- A planet attends over **only the fleets inbound to it** — never the whole
  fleet set. The score matrix is **block-sparse**; `flex_attention` skips the
  fully-masked blocks rather than materializing dense `[B,H,P,F]` scores.
- Planets with no inbound fleet get a zeroed context (`keep` mask,
  `model.py:927-934`), matching the reference's empty-softmax → 0 behavior, and
  are left unconditioned (`has_inbound`, `model.py:1248`).

Implementation notes that make it correct and fast on this box (RTX 5090,
sm_120):

- **bf16-exact** vs the hand-rolled CPU scatter-softmax reference
  (`model.py:994-1047`): forward max |diff| ~0.016, well within bf16 rounding
  for a ≤P-way softmax; verified by `test_policy.py`'s CUDA equivalence test.
- GQA flex-decoding is broken on sm_120, so the single KV head is **MHA-expanded**
  and called with `enable_gqa=False` (`model.py:911-924`).
- The `BlockMask` is built **outside** every compiled / CUDA-graph-captured
  region and threaded in, rebuilt each step/minibatch because destinations move
  (`vec_rollout.py:1467-1489`, `ppo.py` compute/update paths). Building it
  *inside* a captured region silently freezes a stale mask — this was a real
  bug found in review and fixed across all six model-invoking kernels (see §5).

### One honest scope limit

Fleets participate **only at conditioning time**, before the planet-only trunk.
The conditioner returns planets plus a **zero-width** fleet block
(`model.py:1249-1250`), so fleets do **not** become trunk tokens, do not attend
to each other, and are not refined across trunk layers. Planets attend to their
inbound fleets **once**. This is intentional (keeps the trunk planet-only and
O(planets)), but it means the added capacity is "one learned cross-attention
read of inbound fleets," not full fleet-in-the-trunk modeling.

Also: in the compiled rollout the fleet set is bucketed to a fixed
`compile_fleet_width` (64 in the live config). That's the per-step cap on how
many inbound fleet tokens are attended; typical inbound counts sit well under
it, but it is a cap.

---

## 4. Cost

- **SPS:** ~18% slower steady-state (12008 → 9882 steps/s; ae95's measurement on
  the flag-on vs flag-off configs). The dominant added cost is the per-step
  host-side `BlockMask` build on the host-bound rollout critical path, plus the
  flex kernel itself. The live run will report the definitive figure.
- **Why we accept it:** the rollout is host-bound enough to absorb the extra GPU
  work, and the capacity gain is real. Whether it converts to **elo** is exactly
  what the running A/B answers; the bet is killable fast if the summary turns
  out to have already captured the signal.

---

## 5. Correctness fix applied before launch (context)

Option 3 originally wired the destination `BlockMask` into only the two main
PPO kernels. Four other compiled, CUDA-graph-captured kernels still called the
model **without** the mask — most importantly `_OldPolicyDistKernel`, which runs
**every PMPO update** to recompute the frozen rollout policy's distribution for
the reverse-KL trust region. With the flag on, those built the mask *inside* the
captured region → silent stale-mask reuse → corrupted PMPO KL reference.

Fixed by threading the mask (built outside, rebuilt per minibatch) through all
four kernels + their wrappers, with `learned_fleet_attn` added to each
`shape_key`. Verified: flag-on PMPO smoke → all 92 scalars finite, reverse-KL
small and stable across 4 distinct rollouts (the stale-mask tell — it would
blow up otherwise); flag-off baseline smoke clean; full `test_ppo_update` /
`test_policy` suites pass; independent diff review APPROVED.

---

## 6. Where to look

| Thing | Location |
|---|---|
| Destination mask builder | `model.py:110-139` |
| Cross-attention (flex CUDA + scatter ref) | `model.py:834-1047` |
| Hand-crafted 13-stat summary | `model.py:1050-1125` |
| Conditioner (summary vs learned branch) | `model.py:1160-1251` |
| Flag | `OrbitPolicyConfig.destination_learned_fleet_attention` |
| Rollout mask threading | `vec_rollout.py:1467-1540` |
| Update/compute mask threading | `ppo.py` (`_staged_destination_block_mask`) |
| Live A/B config | `configs/ppo_sniper_pmpo_fleetattn.yaml` |
| Baseline | `configs/ppo_sniper_pmpo.yaml` |
