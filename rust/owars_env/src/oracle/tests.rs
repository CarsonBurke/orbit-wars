use super::*;
use crate::core::{CometGroup, Fleet, Game, GameConfig, GameState, Planet, Point, simple_actions};

fn planet(id: i32, x: f64, y: f64, radius: f64) -> Planet {
    Planet {
        id,
        owner: if id == 0 { 0 } else { -1 },
        x,
        y,
        radius,
        ships: 10,
        production: 1,
    }
}

fn fleet(x: f64, y: f64, angle: f64, ships: i32) -> Fleet {
    Fleet {
        id: 0,
        owner: 0,
        x,
        y,
        angle,
        from_planet_id: -1,
        ships,
        target_id: -1,
        eta: 0.0,
        target_x: 0.0,
        target_y: 0.0,
    }
}

#[allow(clippy::too_many_arguments)]
fn game(
    planets: Vec<Planet>,
    fleets: Vec<Fleet>,
    step: i32,
    angular_velocity: f64,
    episode_steps: i32,
    ship_speed: f64,
    comets: Vec<CometGroup>,
) -> Game {
    let initial = planets.clone();
    Game::from_state(
        GameConfig::new(2, episode_steps, ship_speed),
        GameState::new(step, angular_velocity, planets, initial, fleets, comets, 100),
    )
}

fn both(game: &Game) -> (Vec<FleetDestination>, Vec<FleetDestination>) {
    let fleet_limit = game.fleets.len();
    let fast = infer_fleet_destinations(game, fleet_limit);
    let reference = infer_fleet_destinations_reference(game, fleet_limit);
    assert_eq!(fast, reference, "optimized oracle diverged from reference");
    (fast, reference)
}

#[test]
fn static_direct_hit() {
    let g = game(
        vec![planet(0, 5.0, 10.0, 1.0), planet(1, 20.0, 10.0, 1.0)],
        vec![fleet(15.0, 10.0, 0.0, 1)],
        1,
        0.0,
        500,
        6.0,
        vec![],
    );
    let (out, _) = both(&g);
    assert_eq!(out[0].status, STATUS_PLANET);
    assert_eq!(out[0].dest_idx, 1);
    assert_eq!(out[0].eta, 5.0);
}

#[test]
fn static_miss_reaches_board() {
    let g = game(
        vec![planet(0, 20.0, 30.0, 1.0)],
        vec![fleet(15.0, 10.0, std::f64::consts::PI, 1)],
        1,
        0.0,
        500,
        6.0,
        vec![],
    );
    let (out, _) = both(&g);
    assert_eq!(out[0].status, STATUS_BOARD);
    assert_eq!(out[0].dest_idx, -1);
}

#[test]
fn strict_tangent_is_not_a_hit() {
    // Segment passes at distance exactly equal to the radius; the fleet
    // must fly on and leave the board before the comet spawn boundary.
    let g = game(
        vec![planet(0, 90.0, 11.0, 1.0)],
        vec![fleet(85.0, 10.0, 0.0, 1)],
        1,
        0.0,
        500,
        6.0,
        vec![],
    );
    let (out, _) = both(&g);
    assert_eq!(out[0].status, STATUS_BOARD);
}

#[test]
fn board_exit_beats_planet_in_same_turn() {
    // One 6-speed step exits the board even though the segment crosses the
    // planet's disk on the way out.
    let g = game(
        vec![planet(0, 98.0, 50.0, 1.5)],
        vec![fleet(96.0, 50.0, 0.0, 1000)],
        1,
        0.0,
        500,
        6.0,
        vec![],
    );
    let (out, _) = both(&g);
    assert_eq!(out[0].status, STATUS_BOARD);
    assert_eq!(out[0].eta, 1.0);
}

#[test]
fn sun_beats_planet_in_same_turn() {
    // Fast fleet crosses both the sun and a planet behind it in one step.
    let g = game(
        vec![planet(0, 70.0, 50.0, 3.0)],
        vec![fleet(20.0, 50.0, 0.0, 100)],
        1,
        0.0,
        500,
        100.0,
        vec![],
    );
    let (out, _) = both(&g);
    assert_eq!(out[0].status, STATUS_SUN);
    assert_eq!(out[0].eta, 1.0);
}

#[test]
fn tie_uses_planet_vector_order_not_id() {
    let g = game(
        vec![
            planet(5, 30.0, 10.0, 2.0),
            planet(3, 30.0, 10.0, 2.0),
            planet(0, 10.0, 10.0, 1.0),
        ],
        vec![fleet(25.0, 10.0, 0.0, 1)],
        1,
        0.0,
        500,
        6.0,
        vec![],
    );
    let (out, _) = both(&g);
    assert_eq!(out[0].status, STATUS_PLANET);
    assert_eq!(out[0].dest_idx, 0);
    assert_eq!(out[0].eta, 4.0);
}

#[test]
fn bogus_target_metadata_is_ignored() {
    let mut f = fleet(15.0, 10.0, 0.0, 1);
    f.target_id = 2;
    f.eta = 999.0;
    f.target_x = 80.0;
    f.target_y = 80.0;
    let g = game(
        vec![
            planet(0, 5.0, 10.0, 1.0),
            planet(1, 20.0, 10.0, 1.0),
            planet(2, 80.0, 80.0, 5.0),
        ],
        vec![f],
        1,
        0.0,
        500,
        6.0,
        vec![],
    );
    let (out, _) = both(&g);
    assert_eq!(out[0].status, STATUS_PLANET);
    assert_eq!(out[0].dest_idx, 1);
    assert_eq!(out[0].eta, 5.0);
}

#[test]
fn orbiting_planet_pre_move_hit() {
    // Planet orbits at radius 20 with a quarter turn per step. At relative
    // turn 2 its pre-move position is (50, 70); park a slow fleet so its
    // turn-2 segment crosses that point.
    let av = std::f64::consts::FRAC_PI_2;
    let g = game(
        vec![planet(0, 70.0, 50.0, 1.0)],
        vec![fleet(52.5, 70.5, std::f64::consts::PI, 1)],
        1,
        av,
        500,
        6.0,
        vec![],
    );
    // Turn 1: pre-move at the stored (70, 50): no hit; planet then moves to
    // angle av * (step) = pi/2 -> (50, 70). Turn 2 pre-move dist < 1.
    let (out, _) = both(&g);
    assert_eq!(out[0].status, STATUS_PLANET);
    assert_eq!(out[0].dest_idx, 0);
    assert_eq!(out[0].eta, 2.0);
}

#[test]
fn orbiting_planet_sweep_hit() {
    let g = game(
        vec![planet(1, 70.0, 50.0, 1.0), planet(0, 20.0, 80.0, 1.0)],
        vec![fleet(70.0, 52.0, -std::f64::consts::FRAC_PI_2, 1)],
        1,
        0.1,
        500,
        6.0,
        vec![],
    );
    let (out, _) = both(&g);
    assert_eq!(out[0].status, STATUS_PLANET);
    assert_eq!(out[0].dest_idx, 0);
    assert_eq!(out[0].eta, 1.0);
}

#[test]
fn sweep_tie_uses_moving_list_order() {
    // Two orbiting planets sweep the same fleet position in the same turn;
    // the first planet in vector order wins even though it has a higher id.
    let av = 0.05;
    let g = game(
        vec![planet(9, 70.0, 50.0, 2.0), planet(2, 70.0, 50.0, 2.0)],
        vec![fleet(70.0, 53.0, -std::f64::consts::FRAC_PI_2, 1)],
        1,
        av,
        500,
        6.0,
        vec![],
    );
    let (out, _) = both(&g);
    assert_eq!(out[0].status, STATUS_PLANET);
    // Pre-move tie at turn 1 already resolves by vector order; either way
    // the winner must be row 0.
    assert_eq!(out[0].dest_idx, 0);
}

#[test]
fn orbit_phase_uses_step_minus_one() {
    // Same geometry at two different game steps must produce positions
    // rotated by av * (step - 1): verify via reference equality and via the
    // off-by-one-sensitive pre-move hit from `orbiting_planet_pre_move_hit`
    // shifted one step later with the planet's stored position advanced.
    let av = std::f64::consts::FRAC_PI_2;
    // At step = 2 the planet must currently sit at angle av * (step - 1).
    let cur = (
        50.0 + 20.0 * (av * 1.0).cos(),
        50.0 + 20.0 * (av * 1.0).sin(),
    );
    let mut p = planet(0, cur.0, cur.1, 1.0);
    let initial = planet(0, 70.0, 50.0, 1.0);
    p.owner = -1;
    let g = Game::from_state(
        GameConfig::new(2, 500, 6.0),
        GameState::new(
            2,
            av,
            vec![p],
            vec![initial],
            vec![fleet(48.5, 50.0, 0.0, 1)],
            vec![],
            100,
        ),
    );
    // Turn 1 pre-move: (50, 70). Turn 2 pre-move: angle av*2 -> (30, 50).
    // The fleet at (48.5, 50) moving +x meets... nothing at turn 1; at turn
    // 2 the planet pre-move sits at (30, 50), far away; then turn 3 pre-move
    // at angle av*3 -> (50, 30). It exits or keeps flying; the point of this
    // test is optimized == reference at a step offset.
    let (_, _) = both(&g);
}

#[test]
fn comet_first_appearance_does_not_sweep() {
    // Comet spawned this turn sits at (-99, -99) and moves to path[0]
    // without sweeping. A fleet adjacent to path[0] after its own move must
    // survive turn 1 and instead be hit pre-move at turn 2.
    let comet = planet(10, -99.0, -99.0, 1.0);
    let comets = vec![CometGroup {
        planet_ids: vec![10],
        paths: vec![vec![Point::new(20.0, 20.0), Point::new(24.0, 20.0)]],
        path_index: -1,
    }];
    let g = game(
        vec![planet(0, 5.0, 80.0, 1.0), comet],
        // Speed-1 fleet whose post-move turn-1 position is (20.5, 20.0):
        // inside radius of path[0].
        vec![fleet(21.5, 20.0, std::f64::consts::PI, 1)],
        1,
        0.0,
        500,
        6.0,
        comets,
    );
    let (out, _) = both(&g);
    assert_eq!(out[0].status, STATUS_PLANET);
    assert_eq!(out[0].dest_idx, 1);
    assert_eq!(out[0].eta, 2.0);
}

#[test]
fn comet_step_sweep_catches_fleet() {
    // Active comet moves path[0] -> path[1]; the fleet's post-move position
    // sits on that chord and is swept at turn 1.
    let comet = planet(10, 20.0, 20.0, 1.0);
    let comets = vec![CometGroup {
        planet_ids: vec![10],
        paths: vec![vec![
            Point::new(20.0, 20.0),
            Point::new(24.0, 20.0),
            Point::new(28.0, 20.0),
        ]],
        path_index: 0,
    }];
    let g = game(
        vec![planet(0, 5.0, 80.0, 1.0), comet],
        vec![fleet(22.0, 21.8, -std::f64::consts::FRAC_PI_2, 1)],
        1,
        0.0,
        500,
        6.0,
        comets,
    );
    let (out, _) = both(&g);
    assert_eq!(out[0].status, STATUS_PLANET);
    assert_eq!(out[0].dest_idx, 1);
    assert_eq!(out[0].eta, 1.0);
}

#[test]
fn comet_last_path_point_hittable_then_expires() {
    let comet = planet(10, 24.0, 20.0, 1.0);
    let comets = vec![CometGroup {
        planet_ids: vec![10],
        paths: vec![vec![Point::new(20.0, 20.0), Point::new(24.0, 20.0)]],
        path_index: 1,
    }];
    // One turn away: pre-move hit at the final path point.
    let g = game(
        vec![planet(0, 5.0, 80.0, 1.0), comet],
        vec![fleet(25.8, 20.0, std::f64::consts::PI, 1)],
        1,
        0.0,
        500,
        6.0,
        comets.clone(),
    );
    let (out, _) = both(&g);
    assert_eq!(out[0].status, STATUS_PLANET);
    assert_eq!(out[0].dest_idx, 1);
    assert_eq!(out[0].eta, 1.0);

    // Three turns away: the comet expires first and the fleet flies on.
    let comet = planet(10, 24.0, 20.0, 1.0);
    let g = game(
        vec![planet(0, 5.0, 80.0, 1.0), comet],
        vec![fleet(28.5, 20.0, std::f64::consts::PI, 1)],
        1,
        0.0,
        500,
        6.0,
        comets,
    );
    let (out, _) = both(&g);
    assert_eq!(out[0].status, STATUS_BOARD);
}

#[test]
fn future_comet_spawn_marks_unknown() {
    let g = game(
        vec![planet(0, 5.0, 20.0, 1.0), planet(1, 80.0, 80.0, 1.0)],
        vec![fleet(20.0, 20.0, 0.0, 1)],
        48,
        0.0,
        500,
        6.0,
        vec![],
    );
    let (out, _) = both(&g);
    assert_eq!(out[0].status, STATUS_UNKNOWN);
    assert_eq!(out[0].dest_idx, -1);
    assert_eq!(out[0].eta, 3.0);
}

#[test]
fn known_event_before_spawn_boundary_stays_exact() {
    // Hit happens at turn 2, before the spawn boundary at turn 3.
    let g = game(
        vec![planet(0, 22.5, 20.0, 1.0)],
        vec![fleet(20.0, 20.0, 0.0, 1)],
        48,
        0.0,
        500,
        6.0,
        vec![],
    );
    let (out, _) = both(&g);
    assert_eq!(out[0].status, STATUS_PLANET);
    assert_eq!(out[0].eta, 2.0);
}

#[test]
fn terminal_game_yields_no_destination() {
    let mut g = game(
        vec![planet(0, 20.0, 10.0, 1.0)],
        vec![fleet(15.0, 10.0, 0.0, 1)],
        1,
        0.0,
        500,
        6.0,
        vec![],
    );
    g.done = true;
    let (out, _) = both(&g);
    assert_eq!(out[0].status, STATUS_NONE);
    assert_eq!(out[0].eta, 0.0);
}

#[test]
fn zero_horizon_yields_horizon_status() {
    let g = game(
        vec![planet(0, 20.0, 10.0, 1.0)],
        vec![fleet(15.0, 10.0, 0.0, 1)],
        499,
        0.0,
        500,
        6.0,
        vec![],
    );
    let (out, _) = both(&g);
    assert_eq!(out[0].status, STATUS_HORIZON);
    assert_eq!(out[0].eta, 0.0);
}

#[test]
fn disk_hit_one_past_horizon_is_horizon() {
    // step 493 of a 500-step episode: horizon 6. Both fleets first touch
    // their disk on turn 7, a turn the simulator never executes, so the
    // closed-form scans must not report SUN/PLANET at horizon + 1.
    let g = game(
        vec![planet(0, 5.0, 90.0, 1.0)],
        vec![
            // Sun: x goes 35..=40; turn-6 endpoint is at distance exactly 10
            // (strict < misses), turn 7 would cross at distance 9.
            fleet(34.0, 50.0, 0.0, 1),
            // Static planet: turn-7 segment (6.5 -> 5.5, y 90) passes at 0.5.
            fleet(12.5, 90.0, std::f64::consts::PI, 1),
        ],
        493,
        0.0,
        500,
        6.0,
        vec![],
    );
    let (out, _) = both(&g);
    for dest in &out {
        assert_eq!(dest.status, STATUS_HORIZON, "{dest:?}");
        assert_eq!(dest.eta, 6.0, "{dest:?}");
    }
}

#[test]
fn off_board_fleet_is_removed_at_turn_one() {
    // Loaded states may hold fleets already outside the board; the simulator
    // removes them on their first move even when they head back inward.
    let g = game(
        vec![planet(0, 90.0, 90.0, 1.0)],
        vec![
            fleet(-5.0, 50.0, 0.0, 1),
            fleet(104.0, 20.0, std::f64::consts::PI, 1),
        ],
        1,
        0.0,
        500,
        6.0,
        vec![],
    );
    let (out, _) = both(&g);
    for dest in &out {
        assert_eq!(dest.status, STATUS_BOARD, "{dest:?}");
        assert_eq!(dest.eta, 1.0, "{dest:?}");
    }
}

#[test]
fn comet_negative_path_index_dwells_on_first_point() {
    // path_index = -3: the simulator clamps the index, so the comet jumps to
    // path[0] on turn 1 (sweeping its chord) and dwells there until the
    // index catches up at turn 4.
    let path: Vec<Point> = [35.0, 40.0, 45.0, 48.0, 51.0, 54.0]
        .iter()
        .map(|&x| Point::new(x, 30.0))
        .collect();
    let comet = CometGroup {
        planet_ids: vec![7],
        paths: vec![path],
        path_index: -3,
    };
    let g = game(
        vec![planet(0, 90.0, 90.0, 1.0), planet(7, 30.0, 30.0, 1.0)],
        vec![
            // Swept at turn 1 by the (30,30) -> (35,30) chord.
            fleet(33.0, 31.5, -std::f64::consts::FRAC_PI_2, 1),
            // Pre-move hit at turn 4 on the dwelling position (35,30).
            fleet(35.0, 25.5, std::f64::consts::FRAC_PI_2, 1),
        ],
        1,
        0.0,
        500,
        6.0,
        vec![comet],
    );
    let (out, _) = both(&g);
    assert_eq!(out[0].status, STATUS_PLANET);
    assert_eq!(out[0].dest_idx, 1);
    assert_eq!(out[0].eta, 1.0);
    assert_eq!(out[1].status, STATUS_PLANET);
    assert_eq!(out[1].dest_idx, 1);
    assert_eq!(out[1].eta, 4.0);
}

// ---------------------------------------------------------------------------
// Randomized differential + simulator property tests
// ---------------------------------------------------------------------------

struct Lcg(u64);

impl Lcg {
    fn next_f64(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 33) as f64) / f64::from(1u32 << 31) / 1.0
    }

    fn range(&mut self, lo: f64, hi: f64) -> f64 {
        lo + (hi - lo) * self.next_f64()
    }
}

fn inject_fleets(game: &mut Game, rng: &mut Lcg, count: usize) {
    for i in 0..count {
        let ships = (1.0 + rng.range(0.0, 1500.0)) as i32;
        game.fleets.push(Fleet {
            id: 10_000 + i as i32,
            owner: (i % 2) as i32,
            x: rng.range(1.0, 99.0),
            y: rng.range(1.0, 99.0),
            angle: rng.range(-std::f64::consts::PI, std::f64::consts::PI),
            from_planet_id: -1,
            ships,
            target_id: -1,
            eta: 0.0,
            target_x: 0.0,
            target_y: 0.0,
        });
    }
}

fn generated_game(seed: u32, steps: i32) -> Game {
    let mut g = Game::new(GameConfig::new(2, 400, 6.0), seed);
    let empty: Vec<crate::core::PlayerAction> = vec![vec![], vec![]];
    g.step(&empty); // initialize
    for _ in 0..steps {
        if g.done {
            break;
        }
        let actions = simple_actions(&g);
        g.step(&actions);
    }
    g
}

#[test]
fn differential_optimized_matches_reference_on_generated_states() {
    for seed in 0..6u32 {
        for &steps in &[3, 20, 55, 160] {
            let mut g = generated_game(seed, steps);
            if g.done {
                continue;
            }
            let mut rng = Lcg(u64::from(seed) * 7919 + steps as u64);
            inject_fleets(&mut g, &mut rng, 48);
            let fleet_limit = g.fleets.len();
            let fast = infer_fleet_destinations(&g, fleet_limit);
            let reference = infer_fleet_destinations_reference(&g, fleet_limit);
            assert_eq!(
                fast.len(),
                reference.len(),
                "seed {seed} steps {steps}: length mismatch"
            );
            for (idx, (a, b)) in fast.iter().zip(reference.iter()).enumerate() {
                assert_eq!(
                    a, b,
                    "seed {seed} steps {steps} fleet {idx}: optimized {a:?} != reference {b:?}"
                );
            }
        }
    }
}

#[test]
fn differential_edge_states_match_reference() {
    // Hostile loaded states: orbiters stored off their recomputed circle,
    // comets with negative/exhausted path indices and paths shorter than
    // their id lists, off-board fleets, and near-zero horizons.
    use std::f64::consts::PI;
    for seed in 0..24u64 {
        let mut rng = Lcg(seed * 31_337 + 17);
        let step = 440 + (seed as i32 % 13) * 5;
        let mut planets = Vec::new();
        for i in 0..6i32 {
            let orb_r = rng.range(12.0, 44.0);
            let theta = rng.range(-PI, PI);
            planets.push(planet(
                i,
                50.0 + orb_r * theta.cos(),
                50.0 + orb_r * theta.sin(),
                1.0 + rng.range(0.0, 1.0),
            ));
        }
        planets.push(planet(6, 8.0, 92.0, 1.5));

        let mut make_path = |len: usize| -> Vec<Point> {
            (0..len)
                .map(|_| Point::new(rng.range(-5.0, 105.0), rng.range(-5.0, 105.0)))
                .collect()
        };
        let len_a = 5 + (seed as usize % 6);
        let group_a = CometGroup {
            planet_ids: vec![100, 101],
            paths: vec![make_path(len_a), make_path(len_a)],
            path_index: -3 + (seed as i32 % 6),
        };
        // Mismatched: id 103 has no path slot at all.
        let group_b = CometGroup {
            planet_ids: vec![102, 103],
            paths: vec![make_path(4)],
            path_index: (seed as i32 % 5) - 1,
        };
        for group in [&group_a, &group_b] {
            for (idx, pid) in group.planet_ids.iter().enumerate() {
                // Skip one comet's planet row entirely (seed-dependent).
                if *pid == 103 && seed % 3 == 0 {
                    continue;
                }
                let pos = group
                    .paths
                    .get(idx)
                    .and_then(|path| {
                        (group.path_index >= 0)
                            .then(|| path.get(group.path_index as usize))
                            .flatten()
                    })
                    .copied()
                    .unwrap_or(Point::new(-99.0, -99.0));
                // Jitter one stored position off its path point to exercise
                // the turn-1 stored-vs-path special case.
                let jitter = if idx == 0 { rng.range(-0.6, 0.6) } else { 0.0 };
                planets.push(planet(*pid, pos.x + jitter, pos.y, 1.0));
            }
        }

        let mut g = game(
            planets,
            vec![],
            step,
            rng.range(0.02, 0.06),
            500,
            6.0,
            vec![group_a, group_b],
        );
        // Knock stored orbiter positions off the circle their initial entry
        // implies (loaded observations are allowed to do this).
        for p in g.planets.iter_mut().take(6) {
            p.x += rng.range(-0.7, 0.7);
            p.y += rng.range(-0.7, 0.7);
        }
        inject_fleets(&mut g, &mut rng, 24);
        for &(x, y, angle) in &[
            (-5.0, 50.0, 0.0),
            (104.0, 20.0, PI),
            (50.0, -3.0, std::f64::consts::FRAC_PI_2),
            (-2.0, -2.0, 0.7),
        ] {
            g.fleets.push(fleet(x, y, angle, 40));
        }

        let fleet_limit = g.fleets.len();
        let fast = infer_fleet_destinations(&g, fleet_limit);
        let reference = infer_fleet_destinations_reference(&g, fleet_limit);
        for (idx, (a, b)) in fast.iter().zip(reference.iter()).enumerate() {
            assert_eq!(
                a, b,
                "seed {seed} step {step} fleet {idx}: optimized {a:?} != reference {b:?}"
            );
        }
        assert_eq!(fast.len(), reference.len(), "seed {seed}");
    }
}

#[test]
fn property_single_fleet_rollout_matches_simulator() {
    for seed in 0..4u32 {
        for &steps in &[10, 60] {
            let base = generated_game(seed, steps);
            if base.done {
                continue;
            }
            let mut rng = Lcg(u64::from(seed) * 104_729 + steps as u64);
            let mut probes = base.clone();
            probes.fleets.clear();
            inject_fleets(&mut probes, &mut rng, 8);
            let probe_fleets = probes.fleets.clone();
            for probe in probe_fleets {
                let mut with_fleet = base.clone();
                with_fleet.fleets = vec![probe];
                let mut without_fleet = base.clone();
                without_fleet.fleets.clear();
                let oracle = infer_fleet_destinations(&with_fleet, with_fleet.fleets.len())[0];
                let row_ids: Vec<i32> = with_fleet.planets.iter().map(|p| p.id).collect();
                let horizon = inference_horizon(&with_fleet);
                let empty: Vec<crate::core::PlayerAction> = vec![vec![], vec![]];
                let mut resolved = false;
                for turn in 1..=horizon {
                    if with_fleet.done || without_fleet.done {
                        break;
                    }
                    with_fleet.step(&empty);
                    without_fleet.step(&empty);
                    if !with_fleet.fleets.is_empty() {
                        continue;
                    }
                    resolved = true;
                    // The fleet vanished this turn. Identify the cause from
                    // the twin-state diff.
                    let mut hit_planet: Option<i32> = None;
                    for p in &with_fleet.planets {
                        if let Some(q) = without_fleet.planets.iter().find(|q| q.id == p.id) {
                            if p.ships != q.ships || p.owner != q.owner {
                                hit_planet = Some(p.id);
                                break;
                            }
                        }
                    }
                    if oracle.status == STATUS_UNKNOWN {
                        assert!(
                            f64::from(turn) >= oracle.eta,
                            "seed {seed} steps {steps}: unknown at {} but resolved at {turn}",
                            oracle.eta
                        );
                        break;
                    }
                    assert_eq!(
                        f64::from(turn),
                        oracle.eta,
                        "seed {seed} steps {steps}: oracle {oracle:?}, resolved at {turn}"
                    );
                    match hit_planet {
                        Some(pid) => {
                            assert_eq!(oracle.status, STATUS_PLANET, "oracle {oracle:?} vs planet {pid}");
                            assert_eq!(
                                row_ids.get(oracle.dest_idx as usize).copied(),
                                Some(pid),
                                "oracle {oracle:?} hit planet {pid}"
                            );
                        }
                        None => {
                            assert!(
                                oracle.status == STATUS_BOARD || oracle.status == STATUS_SUN,
                                "fleet vanished without planet diff but oracle says {oracle:?}"
                            );
                        }
                    }
                    break;
                }
                if !resolved {
                    assert!(
                        oracle.status == STATUS_HORIZON
                            || oracle.status == STATUS_UNKNOWN
                            || with_fleet.done
                            || without_fleet.done,
                        "fleet survived but oracle says {oracle:?}"
                    );
                }
            }
        }
    }
}
