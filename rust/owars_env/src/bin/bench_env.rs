use std::env;
use std::hint::black_box;
use std::time::Instant;

use owars_env::core::{fixture_game, simple_actions, Game, GameConfig};

fn main() {
    let args: Vec<String> = env::args().collect();
    let num_envs = parse_arg(&args, "--num-envs", 128usize);
    let episode_steps = parse_arg(&args, "--episode-steps", 500i32);
    let workload = parse_string_arg(&args, "--workload", "simple");
    let scenario = parse_string_arg(&args, "--scenario", "fixture");
    let mut envs = (0..num_envs)
        .map(|idx| match scenario.as_str() {
            "fixture" => fixture_game(2, episode_steps),
            "generated" => Game::new(GameConfig::new(2, episode_steps, 6.0), idx as u32),
            other => panic!("unknown scenario: {other}"),
        })
        .collect::<Vec<_>>();

    let start = Instant::now();
    let mut env_steps = 0usize;
    loop {
        let mut any_alive = false;
        for game in &mut envs {
            if game.done {
                continue;
            }
            any_alive = true;
            let actions = match workload.as_str() {
                "noop" => black_box(vec![Vec::new(), Vec::new()]),
                "simple" => black_box(simple_actions(black_box(game))),
                other => panic!("unknown workload: {other}"),
            };
            black_box(game.step(black_box(&actions)));
            env_steps += 1;
        }
        if !any_alive {
            break;
        }
    }
    let elapsed = start.elapsed().as_secs_f64();
    let first = &envs[0];
    println!(
        "rust_core: envs={num_envs} workload={workload} steps={env_steps} wall={elapsed:.6}s wall_sps={:.1} final_step={} final_planets={} final_fleets={} final_next_fleet_id={}",
        env_steps as f64 / elapsed,
        first.step,
        first.planets.len(),
        first.fleets.len(),
        first.next_fleet_id,
    );
}

fn parse_arg<T>(args: &[String], flag: &str, default: T) -> T
where
    T: std::str::FromStr,
{
    args.windows(2)
        .find_map(|pair| {
            if pair[0] == flag {
                pair[1].parse::<T>().ok()
            } else {
                None
            }
        })
        .unwrap_or(default)
}

fn parse_string_arg(args: &[String], flag: &str, default: &str) -> String {
    args.windows(2)
        .find_map(|pair| {
            if pair[0] == flag {
                Some(pair[1].clone())
            } else {
                None
            }
        })
        .unwrap_or_else(|| default.to_string())
}
