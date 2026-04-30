use std::env;

use owars_env::core::{
    CometGroup, Game, GameConfig, GameState, Planet, Point, fixture_game, simple_actions,
};

fn main() {
    let args: Vec<String> = env::args().collect();
    let steps = parse_arg(&args, "--steps", 40usize);
    let workload = parse_string_arg(&args, "--workload", "simple");
    let scenario = parse_string_arg(&args, "--scenario", "fixture");
    let seed = parse_arg(&args, "--seed", 0u32);
    let mut game = match scenario.as_str() {
        "fixture" => fixture_game(2, 120),
        "comet" => comet_game(),
        "generated" => Game::new(GameConfig::new(2, 500, 6.0), seed),
        other => panic!("unknown scenario: {other}"),
    };
    for _ in 0..steps {
        if game.done {
            break;
        }
        let actions = match workload.as_str() {
            "noop" => vec![Vec::new(), Vec::new()],
            "simple" => simple_actions(&game),
            other => panic!("unknown workload: {other}"),
        };
        game.step(&actions);
    }
    print_game_json(&game);
}

fn print_game_json(game: &owars_env::Game) {
    print!(
        "{{\"step\":{},\"done\":{},\"next_fleet_id\":{},\"planets\":[",
        game.step, game.done, game.next_fleet_id
    );
    for (idx, p) in game.planets.iter().enumerate() {
        if idx > 0 {
            print!(",");
        }
        print!(
            "[{},{},{:.17},{:.17},{:.17},{},{}]",
            p.id, p.owner, p.x, p.y, p.radius, p.ships, p.production
        );
    }
    print!("],\"fleets\":[");
    for (idx, f) in game.fleets.iter().enumerate() {
        if idx > 0 {
            print!(",");
        }
        print!(
            "[{},{},{:.17},{:.17},{:.17},{},{}]",
            f.id, f.owner, f.x, f.y, f.angle, f.from_planet_id, f.ships
        );
    }
    println!("]}}");
}

fn comet_game() -> Game {
    let planets = vec![
        Planet {
            id: 0,
            owner: 0,
            x: 10.0,
            y: 80.0,
            radius: 1.0,
            ships: 20,
            production: 1,
        },
        Planet {
            id: 1,
            owner: 1,
            x: 80.0,
            y: 10.0,
            radius: 1.0,
            ships: 20,
            production: 1,
        },
        Planet {
            id: 10,
            owner: -1,
            x: -99.0,
            y: -99.0,
            radius: 1.0,
            ships: 5,
            production: 1,
        },
    ];
    let comets = vec![CometGroup {
        planet_ids: vec![10],
        paths: vec![vec![Point::new(10.0, 10.0), Point::new(14.0, 10.0)]],
        path_index: -1,
    }];
    Game::from_state(
        GameConfig::new(2, 120, 6.0),
        GameState::new(1, 0.0, planets.clone(), planets, vec![], comets, 0),
    )
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
