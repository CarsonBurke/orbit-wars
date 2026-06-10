//! Exact fleet destination oracle.
//!
//! Computes, for every in-flight fleet, the planet it will physically hit
//! under passive simulator dynamics (no future launches), matching
//! `core.rs` bit-for-bit:
//!
//! - fleet positions are accumulated per turn (`x += cos(angle) * speed`),
//!   never reconstructed as `start + n * delta`;
//! - every collision decision uses `core::point_to_segment_distance` with
//!   the simulator's strict `<` comparison;
//! - event precedence is `EventKey = (turn, phase, order)` with phases
//!   Board < Sun < PreMovePlanet < MovingSweep, and `order` equal to the
//!   planet-vector index for pre-move hits and the moving-list rank for
//!   sweeps (survivor removals never reorder survivors, so current-state
//!   ranks are valid tie-break keys for future turns);
//! - orbiting positions are recomputed from `initial_planets` with the
//!   simulator's own `atan2 + angular_velocity * (step - 1)` expression;
//! - comets follow their explicit path tables, never sweep out of the
//!   `(-99, -99)` spawn point, and expire exactly like the simulator;
//! - future *unspawned* comets are unknowable without RNG state, so the
//!   first turn at which one could matter is reported as `STATUS_UNKNOWN`.
//!
//! Two implementations share these semantics:
//!
//! - [`infer_fleet_destinations_reference`]: a destination-only rollout that
//!   mirrors the simulator step order literally. `O(K * H * P)`; the parity
//!   gate, not the hot path.
//! - [`infer_fleet_destinations`]: the production solver. Static fatal
//!   events (board exit, sun, static planets) come from closed-form ray
//!   casts used as a conservative broadphase around exact per-turn
//!   verification. Moving planets ride circles around the sun, so per
//!   (fleet, planet) the candidate turns are the at-most-two ray/annulus
//!   crossing intervals — `O(1)` per pair plus a few exact checks — instead
//!   of an `O(H * P_moving)` per-turn scan. Comets are checked over their
//!   bounded path windows behind a bounding-circle reject. Expected cost is
//!   `O(H * P_orbit)` scene build plus `O(K * (P + verified candidates))`.

use std::collections::{HashMap, HashSet};

use rayon::prelude::*;

use crate::core::{
    BOARD_SIZE, CENTER, COMET_SPAWN_STEPS, CometGroup, Fleet, Game, Planet,
    ROTATION_RADIUS_LIMIT, SUN_RADIUS, fleet_step_speed, point_to_segment_distance,
};

pub const STATUS_NONE: i64 = 0;
pub const STATUS_PLANET: i64 = 1;
pub const STATUS_BOARD: i64 = 2;
pub const STATUS_SUN: i64 = 3;
pub const STATUS_HORIZON: i64 = 4;
pub const STATUS_UNKNOWN: i64 = 5;

pub const PHASE_BOARD: u8 = 0;
pub const PHASE_SUN: u8 = 1;
pub const PHASE_PRE_MOVE_PLANET: u8 = 2;
pub const PHASE_MOVING_SWEEP: u8 = 3;

/// Hard cap on lookahead turns; the episode bound is usually tighter.
pub const HORIZON_CAP: i32 = 600;

/// Closed forms are broadphase only: windows are inflated by margins that
/// dominate accumulated-position drift (~1e-11 over 600 turns) and quadratic
/// root error, then every candidate turn is verified with the simulator
/// predicate. Margins only add candidates; they can never change a result.
const DISK_MARGIN: f64 = 1e-6;
const BAND_MARGIN: f64 = 1e-3;
const PREFILTER_MARGIN: f64 = 1e-6;
const BOARD_EXIT_MARGIN: f64 = 1e-9;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FleetDestination {
    pub dest_idx: i64,
    pub eta: f64,
    pub status: i64,
}

impl FleetDestination {
    fn none() -> Self {
        Self {
            dest_idx: -1,
            eta: 0.0,
            status: STATUS_NONE,
        }
    }

    fn horizon(turns: u32) -> Self {
        Self {
            dest_idx: -1,
            eta: f64::from(turns),
            status: STATUS_HORIZON,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct EventKey {
    turn: u32,
    phase: u8,
    order: u32,
}

#[derive(Clone, Copy, Debug)]
struct OracleEvent {
    key: EventKey,
    dest_idx: i64,
    status: i64,
}

impl OracleEvent {
    fn sentinel(horizon: u32) -> Self {
        Self {
            key: EventKey {
                turn: horizon + 1,
                phase: u8::MAX,
                order: u32::MAX,
            },
            dest_idx: -1,
            status: STATUS_HORIZON,
        }
    }

    fn destination(self, horizon: u32) -> FleetDestination {
        if self.status == STATUS_HORIZON {
            return FleetDestination::horizon(horizon);
        }
        FleetDestination {
            dest_idx: self.dest_idx,
            eta: f64::from(self.key.turn),
            status: self.status,
        }
    }
}

/// Lookahead in future turns. Step `episode_steps - 1` is the last one that
/// still moves fleets (`check_done` fires at the end of it).
pub fn inference_horizon(game: &Game) -> u32 {
    HORIZON_CAP.min((game.episode_steps - 1 - game.step).max(0)) as u32
}

/// First future turn whose outcome can depend on a not-yet-spawned comet.
/// A comet spawned at relative turn `s` sits at `(-99, -99)` through turn
/// `s` and becomes hittable pre-move at `s + 1`. From that key on, board,
/// sun, and lower-order pre-move hits still resolve exactly; everything
/// else is `STATUS_UNKNOWN`.
fn unknown_comet_turn(game: &Game, horizon: u32) -> Option<u32> {
    let turn = COMET_SPAWN_STEPS.iter().copied().find_map(|spawn_step| {
        let spawn_turn = spawn_step - game.step;
        (spawn_turn >= 1).then_some(spawn_turn as u32 + 1)
    })?;
    (turn <= horizon).then_some(turn)
}

fn dist_sq(a: (f64, f64), b: (f64, f64)) -> f64 {
    let dx = a.0 - b.0;
    let dy = a.1 - b.1;
    dx * dx + dy * dy
}

fn outside_board(p: (f64, f64)) -> bool {
    p.0 < 0.0 || p.0 > BOARD_SIZE || p.1 < 0.0 || p.1 > BOARD_SIZE
}

// ---------------------------------------------------------------------------
// Shared scene classification
// ---------------------------------------------------------------------------

/// Simulator-consistent orbit parameters: `None` for planets that never
/// rotate (no initial entry, or orbital_radius + radius >= limit).
fn orbit_params(planet: &Planet, initial: Option<&Planet>) -> Option<(f64, f64, f64)> {
    let initial = initial?;
    let dx = initial.x - CENTER;
    let dy = initial.y - CENTER;
    let orbital_radius = (dx * dx + dy * dy).sqrt();
    if orbital_radius + planet.radius < ROTATION_RADIUS_LIMIT {
        Some((orbital_radius, dy.atan2(dx), planet.radius))
    } else {
        None
    }
}

fn orbit_position(orbital_radius: f64, theta0: f64, angular_velocity: f64, phase_step: i32) -> (f64, f64) {
    let angle = theta0 + angular_velocity * f64::from(phase_step);
    (
        CENTER + orbital_radius * angle.cos(),
        CENTER + orbital_radius * angle.sin(),
    )
}

fn comet_id_set(game: &Game) -> HashSet<i32> {
    game.comets
        .iter()
        .flat_map(|group| group.planet_ids.iter().copied())
        .collect()
}

fn initial_by_id(game: &Game) -> HashMap<i32, Planet> {
    game.initial_planets.iter().map(|p| (p.id, *p)).collect()
}

// ---------------------------------------------------------------------------
// Reference oracle: destination-only rollout mirroring core.rs
// ---------------------------------------------------------------------------

struct RefFleet {
    slot: usize,
    pos: (f64, f64),
    delta: (f64, f64),
}

struct RefPlanet {
    id: i32,
    pos: (f64, f64),
    radius: f64,
    row: usize,
}

/// Destination-only rollout that mirrors the simulator step order without
/// player actions. Used as the differential parity gate for the optimized
/// solver; intentionally has no broadphase or closed forms.
pub fn infer_fleet_destinations_reference(game: &Game, max_fleets: usize) -> Vec<FleetDestination> {
    let n_fleets = game.fleets.len().min(max_fleets);
    let mut out = vec![FleetDestination::none(); n_fleets];
    if n_fleets == 0 {
        return out;
    }
    if game.done {
        return out;
    }
    let horizon = inference_horizon(game);
    if horizon == 0 {
        out.fill(FleetDestination::horizon(0));
        return out;
    }
    let unknown_turn = unknown_comet_turn(game, horizon);

    let initials = initial_by_id(game);
    let row_by_id: HashMap<i32, usize> = game
        .planets
        .iter()
        .enumerate()
        .map(|(row, p)| (p.id, row))
        .collect();
    let mut planets: Vec<RefPlanet> = game
        .planets
        .iter()
        .enumerate()
        .map(|(row, p)| RefPlanet {
            id: p.id,
            pos: (p.x, p.y),
            radius: p.radius,
            row,
        })
        .collect();
    let mut comets: Vec<CometGroup> = game.comets.clone();
    let mut fleets: Vec<RefFleet> = Vec::with_capacity(n_fleets);
    for (slot, fleet) in game.fleets.iter().take(max_fleets).enumerate() {
        let speed = fleet_step_speed(fleet.ships, game.ship_speed);
        let delta = (fleet.angle.cos() * speed, fleet.angle.sin() * speed);
        if !(speed > 0.0)
            || !delta.0.is_finite()
            || !delta.1.is_finite()
            || !fleet.x.is_finite()
            || !fleet.y.is_finite()
        {
            continue;
        }
        out[slot] = FleetDestination::horizon(horizon);
        fleets.push(RefFleet {
            slot,
            pos: (fleet.x, fleet.y),
            delta,
        });
    }

    for turn in 1..=horizon {
        if fleets.is_empty() {
            break;
        }
        // Mirror remove_expired_comets_before_launch (uses pre-increment index).
        let mut expired: Vec<i32> = Vec::new();
        for group in &comets {
            let path_idx = group.path_index.max(0) as usize;
            for (i, pid) in group.planet_ids.iter().enumerate() {
                if group
                    .paths
                    .get(i)
                    .is_some_and(|path| path_idx >= path.len())
                {
                    expired.push(*pid);
                }
            }
        }
        remove_ref_comets(&mut planets, &mut comets, &expired);

        // Mirror move_fleets: board, then sun, then planets in vector order,
        // against positions snapshotted before this turn's planet motion.
        fleets.retain_mut(|fleet| {
            let old = fleet.pos;
            let new = (old.0 + fleet.delta.0, old.1 + fleet.delta.1);
            fleet.pos = new;
            if outside_board(new) {
                out[fleet.slot] = FleetDestination {
                    dest_idx: -1,
                    eta: f64::from(turn),
                    status: STATUS_BOARD,
                };
                return false;
            }
            if point_to_segment_distance((CENTER, CENTER), old, new) < SUN_RADIUS {
                out[fleet.slot] = FleetDestination {
                    dest_idx: -1,
                    eta: f64::from(turn),
                    status: STATUS_SUN,
                };
                return false;
            }
            for planet in &planets {
                if point_to_segment_distance(planet.pos, old, new) < planet.radius {
                    out[fleet.slot] = FleetDestination {
                        dest_idx: planet.row as i64,
                        eta: f64::from(turn),
                        status: STATUS_PLANET,
                    };
                    return false;
                }
            }
            true
        });

        // From the first unobserved comet spawn boundary on, only board /
        // sun / lower-order pre-move outcomes (handled above) are exact.
        if unknown_turn == Some(turn) {
            for fleet in &fleets {
                out[fleet.slot] = FleetDestination {
                    dest_idx: -1,
                    eta: f64::from(turn),
                    status: STATUS_UNKNOWN,
                };
            }
            return out;
        }

        // Mirror move_planets_and_sweep + move_comets.
        let comet_ids: HashSet<i32> = comets
            .iter()
            .flat_map(|group| group.planet_ids.iter().copied())
            .collect();
        let mut moving: Vec<(i32, f64, (f64, f64), (f64, f64))> = Vec::new();
        for planet in &mut planets {
            if comet_ids.contains(&planet.id) {
                continue;
            }
            let Some(initial) = initials.get(&planet.id) else {
                continue;
            };
            let dx = initial.x - CENTER;
            let dy = initial.y - CENTER;
            let orbital_radius = (dx * dx + dy * dy).sqrt();
            let old = planet.pos;
            if orbital_radius + planet.radius < ROTATION_RADIUS_LIMIT {
                let angle =
                    dy.atan2(dx) + game.angular_velocity * f64::from(game.step + turn as i32 - 1);
                planet.pos = (
                    CENTER + orbital_radius * angle.cos(),
                    CENTER + orbital_radius * angle.sin(),
                );
            }
            if old != planet.pos {
                moving.push((planet.id, planet.radius, old, planet.pos));
            }
        }
        let mut expired: Vec<i32> = Vec::new();
        for group in &mut comets {
            group.path_index += 1;
            let path_idx = group.path_index.max(0) as usize;
            for (i, pid) in group.planet_ids.iter().copied().enumerate() {
                let Some(path) = group.paths.get(i) else {
                    expired.push(pid);
                    continue;
                };
                let Some(planet) = planets.iter_mut().find(|p| p.id == pid) else {
                    continue;
                };
                if path_idx >= path.len() {
                    expired.push(pid);
                    continue;
                }
                let old = planet.pos;
                planet.pos = (path[path_idx].x, path[path_idx].y);
                if old.0 >= 0.0 && old != planet.pos {
                    moving.push((pid, planet.radius, old, planet.pos));
                }
            }
        }
        remove_ref_comets(&mut planets, &mut comets, &expired);

        if !moving.is_empty() {
            fleets.retain(|fleet| {
                for (pid, radius, old, new) in &moving {
                    if point_to_segment_distance(fleet.pos, *old, *new) < *radius {
                        out[fleet.slot] = FleetDestination {
                            dest_idx: row_by_id[pid] as i64,
                            eta: f64::from(turn),
                            status: STATUS_PLANET,
                        };
                        return false;
                    }
                }
                true
            });
        }
    }
    out
}

fn remove_ref_comets(planets: &mut Vec<RefPlanet>, comets: &mut Vec<CometGroup>, pids: &[i32]) {
    if pids.is_empty() {
        return;
    }
    let expired: HashSet<i32> = pids.iter().copied().collect();
    planets.retain(|planet| !expired.contains(&planet.id));
    for group in comets.iter_mut() {
        let mut new_ids = Vec::with_capacity(group.planet_ids.len());
        let mut new_paths = Vec::with_capacity(group.paths.len());
        for (idx, pid) in group.planet_ids.iter().copied().enumerate() {
            if expired.contains(&pid) {
                continue;
            }
            new_ids.push(pid);
            if let Some(path) = group.paths.get(idx) {
                new_paths.push(path.clone());
            }
        }
        group.planet_ids = new_ids;
        group.paths = new_paths;
    }
    comets.retain(|group| !group.planet_ids.is_empty());
}

// ---------------------------------------------------------------------------
// Optimized oracle
// ---------------------------------------------------------------------------

/// Accumulated fleet positions, extended lazily. `pos[n]` is the position
/// after `n` future turns, produced by the simulator's repeated addition.
struct Track {
    pos: Vec<(f64, f64)>,
    delta: (f64, f64),
}

impl Track {
    fn new(start: (f64, f64), delta: (f64, f64)) -> Self {
        let mut pos = Vec::with_capacity(32);
        pos.push(start);
        Self { pos, delta }
    }

    #[inline]
    fn at(&mut self, n: u32) -> (f64, f64) {
        let n = n as usize;
        while self.pos.len() <= n {
            let last = self.pos[self.pos.len() - 1];
            self.pos
                .push((last.0 + self.delta.0, last.1 + self.delta.1));
        }
        self.pos[n]
    }
}

struct StaticDisk {
    row: u32,
    center: (f64, f64),
    radius: f64,
}

struct OrbitDisk {
    row: u32,
    radius: f64,
    orbital_radius: f64,
    theta0: f64,
    current: (f64, f64),
    /// Rank in the future moving list (vector order among orbit-class
    /// planets; orbit-class planets precede all comets).
    sweep_order: u32,
    /// Chord deviation bound from the orbit circle (sagitta).
    sagitta_ub: f64,
}

struct CometSlot<'g> {
    row: u32,
    radius: f64,
    path: &'g [crate::core::Point],
    has_path: bool,
    path_index: i64,
    current: (f64, f64),
    sweep_order: u32,
    bound_center: (f64, f64),
    bound_radius: f64,
    max_step_len: f64,
}

struct Scene<'g> {
    game: &'g Game,
    horizon: u32,
    unknown_turn: Option<u32>,
    planets_len: u32,
    statics: Vec<StaticDisk>,
    orbits: Vec<OrbitDisk>,
    comets: Vec<CometSlot<'g>>,
    /// `orbit_table[oi][k]` = exact post-move position of orbit `oi` after
    /// future turn `k` (`k = 0` is the stored current position). Pre-move
    /// position at turn `n` is `orbit_table[oi][n - 1]`.
    orbit_table: Vec<Vec<(f64, f64)>>,
}

impl<'g> Scene<'g> {
    fn build(game: &'g Game, horizon: u32) -> Self {
        let initials = initial_by_id(game);
        let comet_ids = comet_id_set(game);
        let mut statics = Vec::new();
        let mut orbits = Vec::new();
        for (row, planet) in game.planets.iter().enumerate() {
            if comet_ids.contains(&planet.id) {
                continue;
            }
            match orbit_params(planet, initials.get(&planet.id)) {
                Some((orbital_radius, theta0, radius)) => {
                    let half = game.angular_velocity / 2.0;
                    let sagitta_ub = orbital_radius * (1.0 - half.cos()).abs() + BAND_MARGIN;
                    let sweep_order = orbits.len() as u32;
                    orbits.push(OrbitDisk {
                        row: row as u32,
                        radius,
                        orbital_radius,
                        theta0,
                        current: (planet.x, planet.y),
                        sweep_order,
                        sagitta_ub,
                    });
                }
                None => statics.push(StaticDisk {
                    row: row as u32,
                    center: (planet.x, planet.y),
                    radius: planet.radius,
                }),
            }
        }
        // Sweep ranks: every orbit-class planet precedes every comet in the
        // simulator's moving list; comets follow group-major order. Skipped
        // non-movers never reorder the survivors, so fixed ranks are valid
        // tie-break keys.
        let orbit_count = orbits.len() as u32;
        let mut comets = Vec::new();
        for group in &game.comets {
            for (slot_idx, pid) in group.planet_ids.iter().enumerate() {
                let Some(row) = game.planets.iter().position(|p| p.id == *pid) else {
                    continue;
                };
                let planet = &game.planets[row];
                let has_path = group.paths.get(slot_idx).is_some();
                let path: &[crate::core::Point] =
                    group.paths.get(slot_idx).map(Vec::as_slice).unwrap_or(&[]);
                let mut min_x = planet.x;
                let mut max_x = planet.x;
                let mut min_y = planet.y;
                let mut max_y = planet.y;
                let mut max_step_len: f64 = 0.0;
                let start = group.path_index.max(0) as usize;
                for (i, point) in path.iter().enumerate().skip(start.min(path.len())) {
                    min_x = min_x.min(point.x);
                    max_x = max_x.max(point.x);
                    min_y = min_y.min(point.y);
                    max_y = max_y.max(point.y);
                    if i + 1 < path.len() {
                        let next = path[i + 1];
                        max_step_len =
                            max_step_len.max(dist_sq((point.x, point.y), (next.x, next.y)).sqrt());
                    }
                }
                let bound_center = ((min_x + max_x) / 2.0, (min_y + max_y) / 2.0);
                let bound_radius =
                    dist_sq(bound_center, (max_x, max_y)).sqrt() + BAND_MARGIN;
                let sweep_order = orbit_count + comets.len() as u32;
                comets.push(CometSlot {
                    row: row as u32,
                    radius: planet.radius,
                    path,
                    has_path,
                    path_index: i64::from(group.path_index),
                    current: (planet.x, planet.y),
                    sweep_order,
                    bound_center,
                    bound_radius,
                    max_step_len,
                });
            }
        }
        Self {
            game,
            horizon,
            unknown_turn: unknown_comet_turn(game, horizon),
            planets_len: game.planets.len() as u32,
            statics,
            orbits,
            comets,
            orbit_table: Vec::new(),
        }
    }

    /// Exact orbit positions through turn `max_turn`, with the simulator's
    /// own expression tree; `[0]` is the stored current position.
    fn build_orbit_table(&mut self, max_turn: u32) {
        let av = self.game.angular_velocity;
        let step = self.game.step;
        self.orbit_table = self
            .orbits
            .iter()
            .map(|orbit| {
                let mut table = Vec::with_capacity(max_turn as usize + 1);
                table.push(orbit.current);
                for k in 1..=max_turn {
                    table.push(orbit_position(
                        orbit.orbital_radius,
                        orbit.theta0,
                        av,
                        step + k as i32 - 1,
                    ));
                }
                table
            })
            .collect();
    }
}

struct FleetState {
    slot: usize,
    track: Track,
    speed: f64,
    best: OracleEvent,
    resolved_static: bool,
}

/// Production destination oracle. Exact: every result equals
/// [`infer_fleet_destinations_reference`] bit-for-bit.
pub fn infer_fleet_destinations(game: &Game, max_fleets: usize) -> Vec<FleetDestination> {
    let n_fleets = game.fleets.len().min(max_fleets);
    let mut out = vec![FleetDestination::none(); n_fleets];
    if n_fleets == 0 {
        return out;
    }
    if game.done {
        return out;
    }
    let horizon = inference_horizon(game);
    if horizon == 0 {
        out.fill(FleetDestination::horizon(0));
        return out;
    }
    let mut scene = Scene::build(game, horizon);

    let mut states: Vec<FleetState> = game.fleets[..n_fleets]
        .par_iter()
        .enumerate()
        .with_min_len(32)
        .map(|(slot, fleet)| static_pass(&scene, slot, fleet))
        .collect();

    let needs_dynamic = !scene.orbits.is_empty() || !scene.comets.is_empty();
    if needs_dynamic {
        let table_turns = states
            .iter()
            .map(|s| s.best.key.turn.min(scene.horizon))
            .max()
            .unwrap_or(0);
        if !scene.orbits.is_empty() {
            scene.build_orbit_table(table_turns);
        }
        states
            .par_iter_mut()
            .with_min_len(16)
            .for_each(|state| dynamic_pass(&scene, state));
    }

    for state in states {
        out[state.slot] = state.best.destination(horizon);
    }
    out
}

fn static_pass(scene: &Scene<'_>, slot: usize, fleet: &Fleet) -> FleetState {
    let speed = fleet_step_speed(fleet.ships, scene.game.ship_speed);
    let delta = (fleet.angle.cos() * speed, fleet.angle.sin() * speed);
    if !(speed > 0.0)
        || !delta.0.is_finite()
        || !delta.1.is_finite()
        || !fleet.x.is_finite()
        || !fleet.y.is_finite()
    {
        // Unrepresentable fleet: report "no destination" (matches reference).
        return FleetState {
            slot,
            track: Track::new((0.0, 0.0), (0.0, 0.0)),
            speed: 0.0,
            best: OracleEvent {
                key: EventKey {
                    turn: 0,
                    phase: 0,
                    order: 0,
                },
                dest_idx: -1,
                status: STATUS_NONE,
            },
            resolved_static: true,
        };
    }
    let mut track = Track::new((fleet.x, fleet.y), delta);
    let mut best = OracleEvent::sentinel(scene.horizon);

    if let Some(turn) = first_board_exit(&mut track, scene.horizon) {
        best = OracleEvent {
            key: EventKey {
                turn,
                phase: PHASE_BOARD,
                order: 0,
            },
            dest_idx: -1,
            status: STATUS_BOARD,
        };
    }
    // `best` may still be the sentinel (turn = horizon + 1); disk scans must
    // never report a hit on a turn the simulator does not execute.
    if let Some(turn) = first_disk_hit(
        &mut track,
        (CENTER, CENTER),
        SUN_RADIUS,
        best.key.turn.min(scene.horizon),
    ) {
        let key = EventKey {
            turn,
            phase: PHASE_SUN,
            order: 0,
        };
        if key < best.key {
            best = OracleEvent {
                key,
                dest_idx: -1,
                status: STATUS_SUN,
            };
        }
    }
    let start = track.pos[0];
    for disk in &scene.statics {
        let reach = speed * f64::from(best.key.turn) + disk.radius + DISK_MARGIN;
        if dist_sq(start, disk.center) > reach * reach {
            continue;
        }
        if let Some(turn) = first_disk_hit(
            &mut track,
            disk.center,
            disk.radius,
            best.key.turn.min(scene.horizon),
        ) {
            let key = EventKey {
                turn,
                phase: PHASE_PRE_MOVE_PLANET,
                order: disk.row,
            };
            if key < best.key {
                best = OracleEvent {
                    key,
                    dest_idx: i64::from(disk.row),
                    status: STATUS_PLANET,
                };
            }
        }
    }
    if let Some(turn) = scene.unknown_turn {
        let key = EventKey {
            turn,
            phase: PHASE_PRE_MOVE_PLANET,
            order: scene.planets_len,
        };
        if key < best.key {
            best = OracleEvent {
                key,
                dest_idx: -1,
                status: STATUS_UNKNOWN,
            };
        }
    }
    FleetState {
        slot,
        track,
        speed,
        best,
        resolved_static: false,
    }
}

/// First turn whose post-move point is outside the inclusive board, by
/// exact accumulated-position scan from a conservative closed-form start.
fn first_board_exit(track: &mut Track, max_turn: u32) -> Option<u32> {
    let start = track.pos[0];
    let x_bound = axis_exit_scan_start(start.0, track.delta.0);
    let y_bound = axis_exit_scan_start(start.1, track.delta.1);
    let scan_from = match (x_bound, y_bound) {
        (Some(x), Some(y)) => x.min(y),
        (Some(x), None) => x,
        (None, Some(y)) => y,
        (None, None) => return None,
    };
    let scan_from = scan_from.min(f64::from(max_turn) + 1.0) as u32;
    for n in scan_from.max(1)..=max_turn {
        if outside_board(track.at(n)) {
            return Some(n);
        }
    }
    None
}

/// Conservative (never late) first turn at which this axis can leave the
/// board; `None` if it never can.
fn axis_exit_scan_start(pos: f64, delta: f64) -> Option<f64> {
    // Already outside on this axis: turn 1 is a candidate no matter which
    // way the fleet moves (it may stay outside for several turns).
    if pos < 0.0 || pos > BOARD_SIZE {
        return Some(1.0);
    }
    if delta > 0.0 {
        Some(((BOARD_SIZE - BOARD_EXIT_MARGIN - pos) / delta).floor().max(0.0) + 1.0)
    } else if delta < 0.0 {
        Some(((pos - BOARD_EXIT_MARGIN) / -delta).floor().max(0.0) + 1.0)
    } else {
        None
    }
}

/// First turn whose movement segment passes strictly inside the disk.
/// Closed-form ray cast on an inflated radius bounds the candidate window;
/// each candidate turn is verified with the simulator predicate.
fn first_disk_hit(
    track: &mut Track,
    center: (f64, f64),
    radius: f64,
    max_turn: u32,
) -> Option<u32> {
    if max_turn == 0 {
        return None;
    }
    let start = track.pos[0];
    let r_inf = radius + DISK_MARGIN;
    let px = start.0 - center.0;
    let py = start.1 - center.1;
    let vx = track.delta.0;
    let vy = track.delta.1;
    let a = vx * vx + vy * vy;
    let c = px * px + py * py - r_inf * r_inf;
    let (lo, hi);
    if c < 0.0 {
        lo = 0.0;
        let b = 2.0 * (px * vx + py * vy);
        let disc = b * b - 4.0 * a * c;
        hi = (-b + disc.sqrt()) / (2.0 * a);
    } else {
        let b = 2.0 * (px * vx + py * vy);
        let disc = b * b - 4.0 * a * c;
        if disc <= 0.0 {
            return None;
        }
        let sq = disc.sqrt();
        let exit = (-b + sq) / (2.0 * a);
        if exit <= 0.0 {
            return None;
        }
        lo = ((-b - sq) / (2.0 * a)).max(0.0);
        hi = exit;
    }
    let mut n = (lo.floor() + 1.0).max(1.0) as u32;
    while n <= max_turn && f64::from(n - 1) <= hi {
        let s = track.at(n - 1);
        let e = track.at(n);
        if point_to_segment_distance(center, s, e) < radius {
            return Some(n);
        }
        n += 1;
    }
    None
}

fn dynamic_pass(scene: &Scene<'_>, state: &mut FleetState) {
    if state.resolved_static {
        return;
    }
    let speed = state.speed;

    // Turn 1 explicitly, against stored current positions: loaded states may
    // place a planet off its recomputed orbit, so the band broadphase only
    // covers turns >= 2.
    if state.best.key.turn >= 1 {
        for (oi, orbit) in scene.orbits.iter().enumerate() {
            let table = &scene.orbit_table[oi];
            let p0 = state.track.at(0);
            let p1 = state.track.at(1);
            try_pre_move(state, 1, orbit.row, orbit.current, orbit.radius, p0, p1, speed);
            if table.len() > 1 {
                try_sweep(
                    state,
                    1,
                    orbit.row,
                    orbit.sweep_order,
                    orbit.current,
                    table[1],
                    orbit.radius,
                    p1,
                );
            }
        }
    }

    // Turns >= 2 for orbit-class planets: candidate turns are the ray's
    // annulus-band crossings around each orbit circle.
    for (oi, orbit) in scene.orbits.iter().enumerate() {
        let table = &scene.orbit_table[oi];
        let band = orbit.radius + orbit.sagitta_ub + BAND_MARGIN;
        let intervals = annulus_intervals(
            state.track.pos[0],
            state.track.delta,
            orbit.orbital_radius,
            band,
        );
        for &(lo, hi) in intervals.iter().flatten() {
            let cutoff = state.best.key.turn;
            if cutoff < 2 {
                break;
            }
            let n_end_f = (hi + 1.0).min(f64::from(cutoff)).min(f64::from(scene.horizon));
            if n_end_f < 2.0 {
                continue;
            }
            let n_start = lo.max(0.0) as u32;
            let n_start = n_start.max(2);
            let n_end = n_end_f as u32;
            for n in n_start..=n_end {
                if n > state.best.key.turn {
                    break;
                }
                let before = table[(n - 1) as usize];
                let after = table[n as usize];
                let p_prev = state.track.at(n - 1);
                let p_now = state.track.at(n);
                try_pre_move(state, n, orbit.row, before, orbit.radius, p_prev, p_now, speed);
                try_sweep(
                    state,
                    n,
                    orbit.row,
                    orbit.sweep_order,
                    before,
                    after,
                    orbit.radius,
                    p_now,
                );
            }
        }
    }

    // Comets: bounded path windows behind a bounding-circle reject.
    for slot in &scene.comets {
        let cutoff = state.best.key.turn;
        if cutoff == 0 {
            break;
        }
        let len = slot.path.len() as i64;
        let pi = slot.path_index;
        // Pre-move position index pi + n - 1 must stay within the path. A
        // slot with a *missing* path entry survives the expiry pass and is
        // only removed (without sweeping) during turn 1's comet move, so it
        // stays hittable pre-move at turn 1; an *empty* path is expired
        // before turn 1's launches and is never hittable.
        let n_alive = if !slot.has_path {
            1
        } else if len == 0 {
            0
        } else {
            len - pi
        };
        let n_hi = i64::from(cutoff)
            .min(n_alive)
            .min(i64::from(scene.horizon));
        if n_hi < 1 {
            continue;
        }
        // Bounding-circle reject over the fleet's reachable segment.
        let reach_end = (
            state.track.pos[0].0 + state.track.delta.0 * n_hi as f64,
            state.track.pos[0].1 + state.track.delta.1 * n_hi as f64,
        );
        let reject_radius = slot.bound_radius
            + slot.radius
            + speed
            + slot.max_step_len
            + 1.0;
        if point_to_segment_distance(slot.bound_center, state.track.pos[0], reach_end)
            > reject_radius
        {
            continue;
        }
        for n in 1..=n_hi as u32 {
            if n > state.best.key.turn {
                break;
            }
            // The simulator clamps the path index (`path_index.max(0)`), so a
            // group loaded with path_index < -1 dwells on path[0] until the
            // index catches up.
            let pre_idx = (pi + i64::from(n) - 1).max(0);
            let pre_pos = if n == 1 {
                slot.current
            } else {
                let p = slot.path[pre_idx as usize];
                (p.x, p.y)
            };
            let p_prev = state.track.at(n - 1);
            let p_now = state.track.at(n);
            try_pre_move(state, n, slot.row, pre_pos, slot.radius, p_prev, p_now, speed);
            let new_idx = (pi + i64::from(n)).max(0);
            // Comets never sweep out of the off-board spawn point.
            if new_idx < len && pre_pos.0 >= 0.0 {
                let after = slot.path[new_idx as usize];
                try_sweep(
                    state,
                    n,
                    slot.row,
                    slot.sweep_order,
                    pre_pos,
                    (after.x, after.y),
                    slot.radius,
                    p_now,
                );
            }
        }
    }
}

/// Ray/annulus crossing intervals in turn units: the parts of
/// `[0, inf)` where `| |F(t) - sun| - orbital_radius | <= band`.
/// Conservative by construction; at most two intervals.
fn annulus_intervals(
    start: (f64, f64),
    delta: (f64, f64),
    orbital_radius: f64,
    band: f64,
) -> [Option<(f64, f64)>; 2] {
    let px = start.0 - CENTER;
    let py = start.1 - CENTER;
    let a = delta.0 * delta.0 + delta.1 * delta.1;
    let b = 2.0 * (px * delta.0 + py * delta.1);
    let c0 = px * px + py * py;
    let outer = orbital_radius + band;
    let c_out = c0 - outer * outer;
    let disc_out = b * b - 4.0 * a * c_out;
    if disc_out <= 0.0 {
        return [None, None];
    }
    let sq_out = disc_out.sqrt();
    let o1 = (-b - sq_out) / (2.0 * a);
    let o2 = (-b + sq_out) / (2.0 * a);
    if o2 < 0.0 {
        return [None, None];
    }
    let inner = orbital_radius - band;
    if inner > 0.0 {
        let c_in = c0 - inner * inner;
        let disc_in = b * b - 4.0 * a * c_in;
        if disc_in > 0.0 {
            let sq_in = disc_in.sqrt();
            let i1 = (-b - sq_in) / (2.0 * a);
            let i2 = (-b + sq_in) / (2.0 * a);
            let first = (i1 >= o1).then_some((o1, i1.min(o2)));
            let second = (i2 <= o2).then_some((i2.max(o1), o2));
            return [first, second];
        }
    }
    [Some((o1, o2)), None]
}

#[allow(clippy::too_many_arguments)]
#[inline]
fn try_pre_move(
    state: &mut FleetState,
    turn: u32,
    row: u32,
    center: (f64, f64),
    radius: f64,
    p_prev: (f64, f64),
    p_now: (f64, f64),
    speed: f64,
) {
    let key = EventKey {
        turn,
        phase: PHASE_PRE_MOVE_PLANET,
        order: row,
    };
    if key >= state.best.key {
        return;
    }
    let pf = radius + speed + PREFILTER_MARGIN;
    if dist_sq(p_prev, center) > pf * pf {
        return;
    }
    if point_to_segment_distance(center, p_prev, p_now) < radius {
        state.best = OracleEvent {
            key,
            dest_idx: i64::from(row),
            status: STATUS_PLANET,
        };
    }
}

/// Moving-sweep check. The simulator's "no sweep from `(-99, -99)`" rule is
/// comet-specific and enforced at the comet call site, not here.
#[allow(clippy::too_many_arguments)]
#[inline]
fn try_sweep(
    state: &mut FleetState,
    turn: u32,
    row: u32,
    sweep_order: u32,
    before: (f64, f64),
    after: (f64, f64),
    radius: f64,
    p_now: (f64, f64),
) {
    if before == after {
        return;
    }
    let key = EventKey {
        turn,
        phase: PHASE_MOVING_SWEEP,
        order: sweep_order,
    };
    if key >= state.best.key {
        return;
    }
    // Inflated chord AABB containment: necessary for any point within
    // `radius` of the chord, and immune to chord-length surprises.
    let pf = radius + PREFILTER_MARGIN;
    if p_now.0 < before.0.min(after.0) - pf
        || p_now.0 > before.0.max(after.0) + pf
        || p_now.1 < before.1.min(after.1) - pf
        || p_now.1 > before.1.max(after.1) + pf
    {
        return;
    }
    if point_to_segment_distance(p_now, before, after) < radius {
        state.best = OracleEvent {
            key,
            dest_idx: i64::from(row),
            status: STATUS_PLANET,
        };
    }
}

#[cfg(test)]
mod tests;
