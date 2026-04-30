use owars_env::core::{fixture_game, simple_actions};

#[test]
fn fixture_rollout_is_deterministic_across_envs() {
    let mut envs = (0..32).map(|_| fixture_game(2, 120)).collect::<Vec<_>>();
    let mut steps = 0usize;
    loop {
        let mut alive = false;
        for game in &mut envs {
            if game.done {
                continue;
            }
            alive = true;
            let actions = simple_actions(game);
            game.step(&actions);
            steps += 1;
        }
        if !alive {
            break;
        }
    }

    assert_eq!(steps, 32 * 118);
    let first_planets = envs[0].planets.clone();
    let first_fleets = envs[0].fleets.clone();
    let first_rewards = envs[0].rewards();
    for game in &envs[1..] {
        assert_eq!(game.planets, first_planets);
        assert_eq!(game.fleets, first_fleets);
        assert_eq!(game.rewards(), first_rewards);
    }
}
