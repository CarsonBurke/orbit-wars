# Fleet Destination Conditioning Spec

## Objective

Replace global fleet latent mixing with deterministic destination-conditioned
planet conditioning.

The policy should not learn fleet destination geometry from scratch. We will
compute, for each fleet, the exact planet it will physically hit under the
simulator rules, then condition only that destination planet token on the
fleet token(s) assigned to it.

Hard constraints:

- No ETA buckets.
- No board grid.
- No global fleet Perceiver latents in the main encoder.
- No hand-written scalar fleet aggregates as the primary model input.
  Destination planets receive learned conditioning from their assigned raw
  fleet token set.
- No use of `Fleet.target_id`, `eta`, `target_x`, or `target_y` as physics
  truth. They are metadata only.
- Destination assignment is by current planet row index, not by normalized
  feature columns.
- Include every current fleet from every player. Do not filter fleets to the
  acting player. Fleet owner/player identity is part of the raw fleet token
  feature vector and must remain available to the conditioner.

## Current State

`src/owars/policies/features.py` encodes up to `MAX_FLEETS = 384` raw fleet
rows, but active configs use `encoder_backend: fleet_latent`. In
`OrbitPolicy._embed_tokens`, raw fleet embeddings are compressed through
`FleetLatentTokenizer`, then the main encoder sees:

```text
[actor, critic, planet tokens, fleet latent tokens]
```

The existing Rust helper `inferred_fleet_target` in
`rust/owars_env_py/src/lib.rs` is heuristic support for sniper pressure. It is
not an exact simulator oracle: it ignores board/sun precedence, pads radius,
and does not fully reproduce moving-planet/comet sweep semantics.

## Exact Destination Semantics

The destination oracle must match `rust/owars_env/src/core.rs`, not the launch
solver or sniper helper.

For each simulator turn:

1. `step += 1`.
2. Remove expired comets.
3. Spawn comets.
4. Process new launches.
5. Produce ships.
6. Move fleets against current planet positions.
7. Move planets and comets, then sweep surviving fleets with moving planets.
8. Resolve combat.
9. Check terminal state.

For destination inference of existing fleets, future player actions are
unknown and should not be assumed. The initial oracle should simulate only
current fleets and passive planet/comet motion until each fleet hits a planet,
exits the board, is destroyed by the sun, or reaches the inference horizon.

### Fleet Movement

Fleet speed:

```text
speed = min(ship_speed,
            1 + (ship_speed - 1) * max(ln(max(ships, 1)) / ln(1000), 0)^1.5)
```

For each live fleet in vector order:

1. Save old `(x, y)`.
2. Move by one speed step along `angle`.
3. If final point is outside `[0, 100] x [0, 100]`, remove the fleet. This
   takes precedence over sun and planet collision in the same turn.
4. If the segment from old to new has distance strictly `< SUN_RADIUS` from
   `(50, 50)`, destroy the fleet. This takes precedence over planet collision.
5. Check planets in current `self.planets` vector order, using positions
   snapshotted before planet motion for this turn.
6. The first planet with center-to-segment distance strictly `< planet.radius`
   is the fleet destination for this turn.

Tie behavior is simulator order, not nearest along the segment. Strict
inequality matters: exact tangency is not a hit.

### Moving Planet Sweep

After fleet movement, orbiting planets and comets move. A surviving fleet can
be swept by a moving planet if the distance from the fleet's post-move point to
the planet's movement segment is strictly `< planet.radius`.

Sweep checks moving planets in movement collection order. First sweep wins.

Orbiting non-comet planets use initial map position:

```text
orbiting iff orbital_radius + planet.radius < 50
angle_after_step = initial_angle + angular_velocity * (step - 1)
```

Comets use explicit path tables:

- New comet planets start at `(-99, -99)` with `path_index = -1`.
- On movement, `path_index += 1`, then position becomes `path[path_index]`.
- First appearance from `-99` to `path[0]` does not sweep fleets.
- Future unspawned comet paths are not exactly known from bare Kaggle
  observations unless simulator state/RNG is available. The oracle should be
  exact for active comets and may mark later unknown once unobserved future
  comet spawns matter.

## Destination Oracle API

Add an exact Rust oracle first. Proposed public Python binding surface:

```python
dest_idx, eta, status = core.infer_fleet_destinations(rows)
```

Where:

```text
rows: list[(env_idx, player)] or env_idx-only batch
dest_idx: int64[B, MAX_FLEETS]       # planet row index, -1 if no destination
eta: float32[B, MAX_FLEETS]          # turns until hit, 0 for no destination
status: int8[B, MAX_FLEETS]          # hit / board_exit / sun / horizon / unknown / pad
```

For policy conditioning, only `dest_idx` is required. `eta/status` exist for
tests, debugging, and later diagnostics.

The `player` argument is used only to build seat-relative fleet features, not
to filter the oracle. `dest_idx[:, f]` corresponds to `fleet_feats[:, f, :]`
for the same all-fleet ordering. The model must learn friendly/hostile meaning
from the fleet token's owner/player feature columns.

Destination index must be the row index in the current policy planet tensor,
not planet id. This matches action logits `[B, P, P]` and avoids reverse
lookup inside the model.

Extend `EncodedObs`:

```python
fleet_target_planet_idx: torch.Tensor  # [F] or [B, F], long, -1 for none/pad
```

Every feature path must fill it:

- Rust `policy_batch` / `policy_batch_no_context`
- Python raw observation encoder
- parsed observation encoder
- NumPy vector env encoder

For fallback Python encoders, implement the same exact reference algorithm
first, even if slower. Rust is the rollout hot path; Python parity is the
submission and test safety path.

## Efficient Exact Algorithm

The correctness gate is simulator parity. Optimization cannot change results.

### Event-Key Oracle

Implement one primitive and optimize around it:

```text
first_event(snapshot, fleet, max_turn) -> Event
```

Where events are compared by:

```text
EventKey = (turn, phase, order)
phase:
  Board          = 0
  Sun            = 1
  PreMovePlanet  = 2
  MovingSweep    = 3
```

`order` is planet vector order for pre-move hits and moving-list order for
sweeps. This is critical. The simulator does not choose the geometrically
nearest intersection when a segment overlaps multiple planets; it chooses the
first planet in vector order for that phase.

Destination inference is then:

```text
dest_idx = event.planet_row_idx if event.phase in {PreMovePlanet, MovingSweep}
status   = event.phase or Horizon/Pad
eta      = event.turn
```

### Static Closed Forms

Do not use an avoidable `O(K * P * H)` implementation. The production oracle
should compute static fatal events analytically, then use those events as
per-fleet cutoffs for exact dynamic candidate queries.

For a fleet ray:

```text
p(s) = start + s * dir
segment turn n covers distance interval [(n - 1) * speed, n * speed]
```

Board exit:

- Compute first ray distance where the endpoint is outside the inclusive
  `[0, 100] x [0, 100]` square.
- Exact boundary landing survives; board destruction happens only once the
  final point is outside.
- Candidate key: `(n_board, Board, 0)`.

Sun/static disk:

```text
b  = dot(center - start, dir)
d2 = |center - start|^2 - b^2
```

- No hit if `d2 >= radius^2`; strict simulator collision means tangency is not
  a hit.
- Otherwise the open interval is `(b - h, b + h)` where
  `h = sqrt(radius^2 - d2)`.
- Earliest simulator turn is the first `n` where:

```text
n * speed > enter && (n - 1) * speed < exit
```

- Sun candidate key: `(n_sun, Sun, 0)`.
- Static planet candidate key: `(n_planet, PreMovePlanet, planet_vector_idx)`.

Static planet events must be ordered by `(turn, phase, planet_vector_idx)`, not
by physical entry distance along the segment.

### 100x Moving Solver

The current bounded moving scan is not the target implementation. It is exact,
but it is too slow in moving-heavy states because it evaluates:

```text
O(K * H * P_moving)
```

where `K` is fleet count, `H` is future turns, and `P_moving` is moving
planet/comet count.

The production solver must replace this with a per-env scene cache and
conservative spatial-temporal candidate generation:

```text
O(H * P_moving) scene build
+ O(K * (grid_cells_crossed + exact_candidate_count)) queries
```

Measured baseline before this rewrite:

```text
40 planets / 384 fleets
static planets:       ~5 ms per env row
all planets moving:  ~56 ms per env row
```

Target after this rewrite:

```text
40 planets / 384 fleets, moving-heavy: <= 0.5-1.0 ms per unique env state
```

This is an exactness target, not an approximation target. The spatial index is
only a broadphase. Every returned candidate must still be verified with the
same simulator predicate and `EventKey` ordering.

### Oracle Scene Cache

Build one immutable `OracleSceneCache` per unique env state:

```rust
struct OracleSceneCache {
    horizon: usize,
    unknown_comet_event: Option<OracleEvent>,
    static_planets: Vec<StaticDisk>,
    pre_grid: SpatialTurnGrid,
    sweep_grid: SpatialTurnGrid,
}

struct StaticDisk {
    key_order: usize,       // planet vector index
    dest_idx: i64,          // current policy planet row
    center: (f64, f64),
    radius: f64,
}

enum DynamicKind {
    PreMoveDisk,
    MovingSweepChord,
}

struct DynamicPrimitive {
    key: EventKey,
    dest_idx: i64,
    radius: f64,
    a: (f64, f64),          // disk center, or sweep chord start
    b: (f64, f64),          // same as a for disk, or sweep chord end
    kind: DynamicKind,
}
```

Cache invalidation must be explicit on:

- `reset`
- `reset_subset`
- `load_observation`
- `step_subset_fast`
- any future direct state mutation API

Do not key only on `step` or `planets.len()`. Correct cache identity depends
on planet order/ids/radii/positions, `initial_planets`, `angular_velocity`,
fleet rows, comet ids, comet path indices, and comet paths.

### Static Events

For each fleet, initialize `best_event` from closed-form events:

- board exit
- sun disk
- static planet disks
- future unknown comet spawn boundary, if public observation state cannot know
  future comet paths

Static planet events keep `(turn, PreMovePlanet, planet_vector_idx)`. Do not
choose nearest physical entry distance.

Use this `best_event` as a per-fleet cutoff. A dynamic candidate only needs to
be tested if its `EventKey` can beat the current best key.

### Dynamic Primitive Build

For each future turn `n`, precompute moving simulator primitives once:

1. Orbiting non-comet planets:
   - `before` is the initial position rotated by
     `angular_velocity * (game.step + n - 2)`.
   - `after` is the initial position rotated by
     `angular_velocity * (game.step + n - 1)`.
   - Add `PreMoveDisk` with key `(n, PreMovePlanet, planet_vector_idx)`.
   - If `before != after`, add `MovingSweepChord` with key
     `(n, MovingSweep, moving_order)`.

2. Active comets:
   - Pre-move position uses `path_index + n - 1`.
   - After position uses `path_index + n`.
   - Add `PreMoveDisk` only when the pre-move path point exists.
   - Add `MovingSweepChord` only when the after point exists, `before != after`,
     and `before.x >= 0.0`, matching the simulator's no-sweep-from-`-99` rule.
   - Sweep order is all non-comet moving planets in planet-vector order, then
     comets in `game.comets` group order and `group.planet_ids` order.

Do not approximate orbiting movement with arcs. The simulator sweeps the chord
from `before` to `after`.

For each future turn `n`, this still preserves:

- Pre-move hit: fleet segment `[n - 1, n]` against moving planet position at
  `n - 1`.
- Moving sweep: fleet endpoint at `n` against planet movement chord from
  `n - 1` to `n`.

Orbiting positions must be derived from `initial_planets`, not accumulated
current position drift:

```text
theta(t) = initial_theta + angular_velocity * ((game.step - 1) + t)
```

Comets use exact `CometGroup.paths` and current `path_index`:

- Pre-move point exists while `path_index + n - 1 < len(path)`.
- Spawned `path_index = -1` has pre-move `(-99, -99)` on first turn.
- No sweep from `-99` to `path[0]`.
- Last path point can be hit pre-move before expiry.

For internal simulator state, future comet spawns are exactly knowable by
cloning/simulating the game because the RNG state is present. For bare Kaggle
observations reloaded without RNG state, future unobserved comet spawns are
not exact; if no earlier board/sun/planet event wins, return `unknown` at the
first unobserved spawn boundary instead of pretending a later event is exact.

### Spatial-Turn Grid

Use two conservative uniform grids over board coordinates:

- `pre_grid` for `PreMoveDisk`
- `sweep_grid` for `MovingSweepChord`

Recommended first cell size: `4.0` or `5.0`.

Insert each primitive into every cell touched by its inflated AABB:

```text
PreMoveDisk:
  [center.x - radius, center.x + radius]
  [center.y - radius, center.y + radius]

MovingSweepChord:
  [min(a.x, b.x) - radius, max(a.x, b.x) + radius]
  [min(a.y, b.y) - radius, max(a.y, b.y) + radius]
```

Each cell stores primitive indices sorted by `EventKey`. The grid must be a
conservative broadphase: false positives are allowed, false negatives are not.

Fleet query:

1. Compute static `best_event`.
2. Traverse the fleet ray `F(t) = start + step_delta * t` through grid cells
   using 2D DDA for `t in [0, best_event.turn]`.
3. For each crossed cell interval `[ta, tb]`:
   - Query `pre_grid` for primitives whose turn segment `[n - 1, n]`
     intersects `[ta, tb]`.
   - Query `sweep_grid` for primitives whose integer turn `n` falls inside
     `[ceil(ta), floor(tb)]`.
4. Verify candidates exactly:

```text
PreMoveDisk:
  point_to_segment_distance(center, F(n - 1), F(n)) < radius

MovingSweepChord:
  point_to_segment_distance(F(n), chord_start, chord_end) < radius
```

5. Return the minimum verified `EventKey`.

The final predicate should duplicate the simulator's distance comparison
semantics. If the simulator computes `sqrt(distance_sq) < radius`, the oracle
must either call the same function or intentionally match its floating-point
behavior. Squared-distance shortcuts are not acceptable near exact tangencies
unless proven equivalent by tests.

### Caching And Vectorization

Expected practical complexity:

```text
O(H * P_moving) + O(K * (P_static + grid_cells_crossed + exact_candidates))
```

Safe exact upgrades:

- Cache per-state structure-of-arrays for planet id, vector index, position,
  radius squared, and motion kind.
- Cache orbit/comet positions and sweep chords once per env step.
- Cache `fleet_speed[ships]`.
- Deduplicate requested rows by `env_idx`. Destination oracle output is
  player-independent; player only changes seat-relative fleet feature columns.
- Cache full oracle output by env state and invalidate explicitly on mutation.
- SIMD/vectorize static disk interval checks across all static planets.
- SIMD/vectorize exact candidate verification over precomputed SoA arrays.
- Conservative broadphase is allowed only if followed by exact distance checks.

Do not introduce approximate angle bins or raster maps into the oracle. Exact
interval maps are allowed later only if they are proven bit-for-bit equivalent
against `first_event`.

## Model Integration

Add a new backend:

```yaml
encoder_backend: destination_conditioned
```

This backend consumes raw fleet tokens and `fleet_target_planet_idx`; it does
not append raw fleet tokens or fleet latents to the main transformer. Raw fleet
tokens are grouped by exact destination planet and used only as token-aligned
conditioning for the corresponding planet token.

Tensor flow:

```text
planet_feats             [B, P, planet_features]
fleet_feats              [B, F, fleet_features]
planet_mask              [B, P]
fleet_mask               [B, F]
fleet_target_planet_idx  [B, F]     # -1 for no destination

h_p = planet_embed(planet_feats)    [B, P, D]
h_f = fleet_embed(fleet_feats)      [B, F, D]
h_p = justnorm(h_p)
h_f = justnorm(h_f)
```

`fleet_feats` includes all fleets from all players. Owner/player identity must
be encoded in the fleet feature dimension, preferably using the existing
seat-relative owner one-hot columns. Destination grouping never erases owner:
friendly reinforcements, hostile arrivals, and third-party pressure all remain
separate information inside `p_i_fleets`.

Conceptually, each planet owns a variable-length fleet token set:

```text
p_i_fleets = fleet_tokens[b, f] where fleet_target_planet_idx[b, f] == i
p_i_fleets: [fleet_count_i, D]
```

The batched representation remains dense:

```text
dest_mask[b, p, f] =
    planet_mask[b, p]
    & fleet_mask[b, f]
    & (fleet_target_planet_idx[b, f] == p)
```

Destination-conditioned fleet context is computed by masked attention from
each planet token into only its assigned fleet token set:

```text
fleet_ctx = DestinationFleetCrossAttention(
    queries=h_p,          # [B, P, D]
    keys_values=h_f,      # [B, F, D]
    mask=dest_mask,       # [B, P, F]
)                         # [B, P, D]
```

This is one learned fleet summary per destination planet. It is not a global
latent bottleneck, grid, or bucketed handcrafted map.

Empty destination groups must produce exact zero context and exact no-op
conditioning.

### AdaLN-Zero Conditioning

Use the `../le-wm/module.py` AdaLN-zero pattern, adapted to this model's nGPT
hypersphere residual stream. In `le-wm`, each block maps a token-aligned
condition `c` to:

```text
shift_msa, scale_msa, gate_msa,
shift_mlp, scale_mlp, gate_mlp = adaLN(c).chunk(6, -1)
```

and applies modulation inside the transformer block, not as a single
preprocessing step.

For Orbit Wars, the condition is:

```text
c_full = [zero_actor, zero_critic, fleet_ctx_by_planet]
c_full: [B, 2 + P, D]
```

Actor and critic tokens receive zero condition. Each planet token receives the
context derived from exactly the fleet tokens assigned to that planet.

Each destination-conditioned transformer block should therefore accept
`x_full` and `c_full`:

```text
shift_attn, scale_attn, gate_attn,
shift_mlp,  scale_mlp,  gate_mlp = adaLN(c_full).chunk(6, -1)

attn_in = rms_norm(x_full) * (1 + scale_attn) + shift_attn
mlp_in  = rms_norm(x_full) * (1 + scale_mlp)  + shift_mlp
```

Then preserve nGPT residual semantics:

```text
x = eigen_residual(x, gate_attn * self_attention(attn_in), attn_alpha)
x = eigen_residual(x, gate_mlp  * mlp(mlp_in),             mlp_alpha)
```

Alternative acceptable form, if simpler for the first implementation:

```text
gamma, beta = mod_mlp(fleet_ctx).chunk(2, -1)
conditioned = rms_norm(h_p) * (1 + gamma) + beta
h_p = justnorm(conditioned)
h_p = where(has_dest_group[..., None], h_p, original_h_p)
```

But the target architecture is per-block AdaLN-zero conditioning, not one
pre-trunk modulation.

Zero-initialize the final modulation projection so the new path is identity at
initialization:

```text
shift_* = 0
scale_* = 0
gate_* = 0
```

The main encoder input becomes:

```text
[actor, critic, conditioned_planets]
```

Planet RoPE still applies to the planet token slice. There are no fleet tokens
in the main transformer for this backend.

## nGPT / Optimizer Requirements

The model uses a unit-L2 hypersphere residual stream. The destination
conditioner must preserve that invariant:

- Apply `justnorm` after affine conditioning, or inject via
  `eigen_residual`.
- Cross-attention Q/K should use the same `justnorm` + `sqk` pattern as
  existing attention.
- AdaLN modulation follows `le-wm` semantics: `x * (1 + scale) + shift`, with
  final projection zero-initialized.
- Gates must be zero at initialization so the conditioned block is an exact
  no-op relative to the unconditioned block's added condition path.
- Destination attention matrices are trunk parameters and should be included
  in Muon normalization/splitting.
- `sqk_*` and any conditioner eigen/gating scalars belong in the control
  group.
- Final AdaLN modulation projection is zero-init and should not be hypersphere
  normalized.

## Test Plan

### Oracle Unit Tests

Add Rust tests for exact destination inference:

- Static direct hit.
- Static miss.
- Strict tangent: exactly radius does not hit.
- Board exit before planet hit in same turn.
- Board exit before sun in same turn.
- Sun before planet in same turn.
- Multiple planets hit by same segment: earlier planet vector entry wins.
- Bogus `target_id/eta` metadata does not affect physical destination.
- Orbiting planet pre-move hit.
- Orbiting planet post-move sweep hit.
- Moving sweep tie uses simulator moving order.
- Orbit off-by-one for `step -> step + 1`.
- Comet first appearance has no sweep.
- Comet `path[0] -> path[1]` sweep works.
- Comet expiry before launch when `path_index >= len`.
- Terminal game state stops future inference.

### Oracle Property Tests

For randomized states:

1. Clone a game.
2. Run destination oracle for every current fleet.
3. Run a no-action simulator rollout until each fleet disappears or hits.
4. Compare destination planet id, ETA, and destruction status.

This property test is the release gate for replacing the current heuristic
destination inference.

Add a differential parity harness for the optimized solver:

1. Run the optimized oracle for every current fleet.
2. Independently roll out a destination-only reference that mirrors
   `core.rs` step order without player actions.
3. Compare `status`, `eta`, and destination planet id/row for every fleet.
4. If the optimized oracle returns `unknown`, assert that the reference first
   requires an unobserved future comet spawn before any known fatal event.

The reference may be slower. It exists only to prove the optimized candidate
index has no false negatives and preserves event ordering.

### Feature Parity Tests

Extend existing Rust/Python feature parity tests:

- `fleet_target_planet_idx` from Rust `policy_batch` matches Python raw
  encoder.
- Shuffled planet ids still map destinations to row index.
- Padded fleets have `-1`.
- Missing/unknown destination has `-1`.
- Own fleet metadata does not override the exact physical oracle.

### Model Tests

- Backend shape test: `[B, 64, D]` planet path and existing action/value shapes.
- Empty fleet groups are exact no-op at conditioner boundary.
- Padded fleet rows do not affect logits.
- A fleet assigned to planet `i` changes only planet `i` before the main trunk.
- Per-block AdaLN-zero parity: with zero condition or zero-init modulation,
  conditioned blocks match unconditioned blocks to numerical tolerance.
- Valid planet tokens remain finite and unit norm after conditioning.
- Optimizer split includes new destination-attention params in intended groups.
- Existing `sample_actions` legality tests still pass.

### Performance Tests

Add deterministic microbenchmarks that fail loudly when the optimized path
regresses. They should run outside normal unit-test CI by default, but must be
easy to invoke before training with the destination backend.

Benchmark scenarios:

- `40` static planets, `384` fleets, no comets.
- `40` orbiting planets, `384` fleets, no comets.
- mixed static/orbiting/active-comet state, `384` fleets.
- duplicate policy rows for the same env: `(env, player0)`, `(env, player1)`.

Acceptance targets on the development machine used for current measurements:

```text
static 40p/384f:           <= 0.25 ms per unique env state
moving-heavy 40p/384f:     <= 1.00 ms per unique env state
duplicate player rows:     <= 1.10x one-row wall time
```

Hard failure criteria:

- Any optimized result differs from the reference oracle.
- Any exact collision is missed by the spatial broadphase.
- Any destination depends on requested player row instead of fleet owner
  feature columns.
- Any padded fleet has non-pad destination.

## Implementation Milestones

1. Replace the bounded moving scan with `OracleSceneCache` and static SoA.
2. Add row deduplication and per-env oracle output cache with explicit
   invalidation.
3. Add `SpatialTurnGrid` for pre-move disks and moving-sweep chords.
4. Add exact candidate verification and differential simulator parity tests.
5. Hit the performance acceptance targets above.
6. Add `fleet_target_planet_idx` to `EncodedObs` and all encoders.
7. Restore Rust/Python feature parity tests.
8. Add `destination_conditioned` model backend with zero-init conditioner.
9. Add model/optimizer tests.
10. Add config ablation against current `fleet_latent` backend.

No training run should use the new backend until milestones 1-9 pass.

## Implementation Notes (2026-06-09, milestones 1-5 + Python oracle)

The oracle landed in `rust/owars_env/src/oracle.rs` with the binding in
`rust/owars_env_py/src/lib.rs` and the Python exact twin in
`src/owars/game/destination_oracle.py`. Deviations from the proposals above,
all within the exactness rules:

### Annulus bands instead of `SpatialTurnGrid`

The moving-planet broadphase does not build a per-turn spatial grid. Orbiting
planets move on circles of fixed radius `R` around the sun, so for each
(fleet, orbit) pair the turns whose fleet segment can touch the orbit's
annulus `[R - r - margin, R + r + margin]` come from two ray/circle
quadratics: at most two distance intervals, mapped to turn intervals in
`O(1)` per pair. Comets get a per-slot bounding circle plus a per-turn
chord-AABB test over their remaining path window. Complexity is

```text
O(H * P_orbit) exact orbit-position table build (only up to the
               max surviving static cutoff turn, not the full horizon)
+ O(K * (P + exact_candidate_count)) queries
```

which replaces the proposed `O(grid_cells_crossed)` term with a closed-form
candidate set and avoids grid tuning entirely. Everything else from the spec
holds: candidates are ε-inflated conservative windows (`BAND_MARGIN`,
`DISK_MARGIN`, `PREFILTER_MARGIN` in `oracle.rs`), and every candidate turn is
re-verified with `core::point_to_segment_distance` against accumulated fleet
positions and full `EventKey` ordering before it can win.

### Exactness strategy

- Fleet positions are tracked by per-turn accumulation (`pos += delta`),
  matching `move_fleets` rounding; closed forms never decide a hit.
- The orbit table stores index 0 as the *stored* current planet position
  (authoritative for loaded observations) and computes index `k` with the
  simulator expression `theta0 + angular_velocity * f64::from(step + k - 1)`;
  turn 1 sweeps are checked explicitly against stored positions.
- Comet-expiry removals never reorder surviving planets, so current vector
  indices (pre-move order) and fixed sweep ranks (orbiters by vector order,
  then comets in group-major order) stay valid tie-break keys for all future
  turns; sweep order is precomputed once per scene.
- `infer_fleet_destinations_reference` is a literal simulator-mirror rollout
  kept as the differential gate; `bench_oracle` asserts optimized == reference
  on every scenario before timing.

### Binding and caching

`fleet_destination_oracle(rows)` dedups rows by env index and stores one
`Arc<Vec<FleetDestination>>` per env; every state mutation (`reset`,
`reset_subset`, `load_observation`, `step_subset_fast`) clears the slot
explicitly. `fleet_destination_oracle_reference` bypasses the cache and runs
the rollout for parity tests.

### Python twin

`owars.game.destination_oracle.infer_fleet_destinations` mirrors the
reference rollout over a parsed `Observation`, vectorized over fleets with
NumPy but using the simulator's exact elementwise expressions (including the
sqrt-based segment predicate and strict `<`). `tests/test_destination_oracle.py`
pins bit-exact equality against both Rust bindings on randomized states with
statics, orbiters, and comets.

### Measured results (dev machine, 2026-06-09)

```text
cargo run --release --bin bench_oracle   (rust/owars_env)
static 40p/384f:        ~0.06 ms   (reference ~0.48 ms)
orbiting 40p/384f:      ~0.18 ms   (reference ~0.59 ms)
mixed + active comets:  ~0.14 ms   (reference ~0.48 ms)

PYTHONPATH=src python scripts/benchmark_oracle.py   (through the binding)
cold (cache invalidated) matches the above + ~0 binding overhead
duplicate rows: ~1.0x   warm (cached): ~0.8 us   python twin: 7-33 ms
```

Both benchmarks are gates: they exit non-zero on any threshold miss or
optimized/reference divergence.

### Review-driven fixes (landed with the initial implementation)

An adversarial parity audit caught and fixed, with regression tests:

- the closed-form sun/static scans could report a hit at `horizon + 1` (the
  sentinel turn) — now clamped to the horizon (`disk_hit_one_past_horizon_is_horizon`);
- fleets loaded already off-board but heading inward were not flagged as
  board-removed at turn 1 (`off_board_fleet_is_removed_at_turn_one`);
- comet groups loaded with `path_index < -1` did not mirror the simulator's
  `max(0)` clamp (dwell on `path[0]`) in the optimized pass
  (`comet_negative_path_index_dwells_on_first_point`, plus the
  `differential_edge_states_match_reference` fuzz over hostile loaded states);
- the Python twin now mirrors the binding's empty-`initial_planets` fallback,
  survives ±inf angles/angular velocities where libm raises but Rust yields
  NaN, and resolves duplicate planet ids exactly like the simulator
  (first-instance moves, last-row-by-id sweep destinations).
