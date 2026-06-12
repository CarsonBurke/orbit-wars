//! Destination-oracle microbenchmark for the spec acceptance scenarios:
//!
//! ```text
//! static 40p/default fleets:        <= linearly scaled 384-fleet gate
//! moving-heavy 40p/default fleets:  <= linearly scaled 384-fleet gate
//! mixed w/ active comets:           <= linearly scaled 384-fleet gate
//! ```
//!
//! Run with `cargo run --release --bin bench_oracle`. Exits non-zero when a
//! threshold is exceeded, so it can gate training runs without living in
//! the default test suite.

use std::time::Instant;

use owars_env::core::{CometGroup, Fleet, Game, GameConfig, GameState, Planet, Point};
use owars_env::oracle;

const NUM_FLEETS: usize = 512;
const BASELINE_NUM_FLEETS: f64 = 384.0;
const STATIC_LIMIT_US_384: f64 = 250.0;
const MOVING_LIMIT_US_384: f64 = 500.0;

struct Lcg(u64);

impl Lcg {
    fn next_f64(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 33) as f64) / f64::from(1u32 << 31)
    }

    fn range(&mut self, lo: f64, hi: f64) -> f64 {
        lo + (hi - lo) * self.next_f64()
    }
}

fn fleets(rng: &mut Lcg, count: usize) -> Vec<Fleet> {
    (0..count)
        .map(|i| Fleet {
            id: i as i32,
            owner: (i % 2) as i32,
            x: rng.range(1.0, 99.0),
            y: rng.range(1.0, 99.0),
            angle: rng.range(-std::f64::consts::PI, std::f64::consts::PI),
            from_planet_id: -1,
            ships: 1 + (rng.range(0.0, 1.0).powi(2) * 800.0) as i32,
            target_id: -1,
            eta: 0.0,
            target_x: 0.0,
            target_y: 0.0,
        })
        .collect()
}

fn static_planets(count: usize) -> Vec<Planet> {
    // Corner clusters spreading toward the board edge, away from the sun:
    // orbital_radius + radius >= 50 for every member, so none ever rotate.
    let corners = [(12.0, 12.0), (88.0, 12.0), (12.0, 88.0), (88.0, 88.0)];
    (0..count)
        .map(|i| {
            let (cx, cy) = corners[i % 4];
            let k = (i / 4) as f64;
            let dx = if cx < 50.0 { -3.5 } else { 3.5 };
            let dy = if cy < 50.0 { -3.5 } else { 3.5 };
            Planet {
                id: i as i32,
                owner: if i == 0 { 0 } else if i == 1 { 1 } else { -1 },
                x: cx + dx * (k % 3.0),
                y: cy + dy * (k / 3.0).floor(),
                radius: 1.0 + (1.0 + (i % 5) as f64).ln() * 0.55,
                ships: 30,
                production: 1 + (i % 5) as i32,
            }
        })
        .collect()
}

fn orbiting_planets(count: usize) -> Vec<Planet> {
    (0..count)
        .map(|i| {
            let radius = 1.0 + (1.0 + (i % 5) as f64).ln() * 0.55;
            let orb_r = 13.0 + (i as f64) * (46.0 - 13.0 - radius.ceil()) / count as f64;
            let angle = i as f64 * 2.399_963; // golden angle spread
            Planet {
                id: i as i32,
                owner: if i == 0 { 0 } else if i == 1 { 1 } else { -1 },
                x: 50.0 + orb_r * angle.cos(),
                y: 50.0 + orb_r * angle.sin(),
                radius,
                ships: 30,
                production: 1 + (i % 5) as i32,
            }
        })
        .collect()
}

fn comet_group(next_id: i32, path_index: i32) -> (Vec<Planet>, CometGroup) {
    let base: Vec<Point> = (0..30)
        .map(|i| Point::new(2.0 + i as f64 * 3.4, 8.0 + i as f64 * 2.9))
        .collect();
    let paths: Vec<Vec<Point>> = vec![
        base.clone(),
        base.iter().map(|p| Point::new(100.0 - p.x, p.y)).collect(),
        base.iter().map(|p| Point::new(p.x, 100.0 - p.y)).collect(),
        base.iter()
            .map(|p| Point::new(100.0 - p.x, 100.0 - p.y))
            .collect(),
    ];
    let planet_ids: Vec<i32> = (0..4).map(|i| next_id + i).collect();
    let planets = planet_ids
        .iter()
        .zip(&paths)
        .map(|(pid, path)| {
            let pos = if path_index >= 0 {
                path[path_index as usize]
            } else {
                Point::new(-99.0, -99.0)
            };
            Planet {
                id: *pid,
                owner: -1,
                x: pos.x,
                y: pos.y,
                radius: 1.0,
                ships: 5,
                production: 1,
            }
        })
        .collect();
    (
        planets,
        CometGroup {
            planet_ids,
            paths,
            path_index,
        },
    )
}

fn build_game(planets: Vec<Planet>, comets: Vec<CometGroup>, fleet_seed: u64, step: i32) -> Game {
    let mut rng = Lcg(fleet_seed);
    let fleets = fleets(&mut rng, NUM_FLEETS);
    let initial = planets.clone();
    Game::from_state(
        GameConfig::new(2, 2000, 6.0),
        GameState::new(step, 0.04, planets, initial, fleets, comets, 10_000),
    )
}

fn bench(name: &str, game: &Game, threshold_us: f64) -> bool {
    let fleet_limit = game.fleets.len();
    let reference = oracle::infer_fleet_destinations_reference(game, fleet_limit);
    let fast = oracle::infer_fleet_destinations(game, fleet_limit);
    assert_eq!(fast, reference, "{name}: optimized diverged from reference");

    for _ in 0..20 {
        std::hint::black_box(oracle::infer_fleet_destinations(game, fleet_limit));
    }
    let iters = 200;
    let start = Instant::now();
    for _ in 0..iters {
        std::hint::black_box(oracle::infer_fleet_destinations(game, fleet_limit));
    }
    let per_call_us = start.elapsed().as_secs_f64() * 1e6 / iters as f64;

    let ref_start = Instant::now();
    for _ in 0..5 {
        std::hint::black_box(oracle::infer_fleet_destinations_reference(game, fleet_limit));
    }
    let ref_us = ref_start.elapsed().as_secs_f64() * 1e6 / 5.0;

    let ok = per_call_us <= threshold_us;
    println!(
        "{name:<28} optimized {per_call_us:9.1} us   reference {ref_us:10.1} us   limit {threshold_us:7.1} us   {}",
        if ok { "OK" } else { "FAIL" }
    );
    ok
}

fn main() {
    // step 460 with a long synthetic episode: no future comet spawns, so no
    // unknown-boundary cutoff shortens the lookahead (worst case for us).
    let static_game = build_game(static_planets(40), vec![], 1, 460);
    let orbit_game = build_game(orbiting_planets(40), vec![], 2, 460);
    let (mut mixed_planets, group) = {
        let mut planets = static_planets(20);
        planets.extend(orbiting_planets(16).into_iter().map(|mut p| {
            p.id += 100;
            p
        }));
        let (comet_planets, group) = comet_group(500, 6);
        planets.extend(comet_planets);
        (planets, group)
    };
    // Keep ids unique and stable.
    for (i, p) in mixed_planets.iter_mut().enumerate() {
        p.id = i as i32;
    }
    let group = CometGroup {
        planet_ids: (mixed_planets.len() - 4..mixed_planets.len())
            .map(|i| i as i32)
            .collect(),
        ..group
    };
    let mixed_game = build_game(mixed_planets, vec![group], 3, 460);
    let scale = NUM_FLEETS as f64 / BASELINE_NUM_FLEETS;
    let static_limit = STATIC_LIMIT_US_384 * scale;
    let moving_limit = MOVING_LIMIT_US_384 * scale;

    let mut ok = true;
    ok &= bench(
        &format!("static 40p / {NUM_FLEETS}f"),
        &static_game,
        static_limit,
    );
    ok &= bench(
        &format!("orbiting 40p / {NUM_FLEETS}f"),
        &orbit_game,
        moving_limit,
    );
    ok &= bench(
        &format!("mixed + active comets / {NUM_FLEETS}f"),
        &mixed_game,
        moving_limit,
    );

    if !ok {
        std::process::exit(1);
    }
}
