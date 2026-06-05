import math

from owars.agents import HeuristicAgent, random_agent, sniper_agent
from owars.agents.learned import _FleetTargetTracker
from owars.policies.sampling import _lead_solution


def _obs(my_planet, enemy_planet, my_ships=50, enemy_ships=10):
    return {
        "player": 0,
        "step": 0,
        "planets": [
            [0, 0, my_planet[0], my_planet[1], 1.0, my_ships, 3],
            [1, 1, enemy_planet[0], enemy_planet[1], 1.0, enemy_ships, 2],
        ],
        "fleets": [],
        "angular_velocity": 0.0,
        "initial_planets": [],
        "comet_planet_ids": [],
        "comets": [],
        "remainingOverageTime": 60.0,
    }


def _check_move_format(moves, owned_ids: set[int]):
    assert isinstance(moves, list)
    for m in moves:
        assert isinstance(m, list) and len(m) == 3
        from_id, angle, ships = m
        assert from_id in owned_ids
        assert isinstance(ships, int) and ships >= 1
        assert -math.pi - 1e-9 <= float(angle) <= math.pi + 1e-9 + 0.01


def test_random_agent_returns_valid_move_format():
    moves = random_agent(_obs((10, 10), (90, 90)))
    _check_move_format(moves, owned_ids={0})


def test_sniper_attacks_when_strong_enough():
    moves = sniper_agent(_obs((10, 10), (30, 10), my_ships=50, enemy_ships=10))
    assert len(moves) == 1
    from_id, angle, ships = moves[0]
    assert from_id == 0
    assert ships == 11  # garrison + 1
    assert abs(angle) < 1e-6


def test_sniper_leads_orbiting_targets():
    obs = {
        "player": 1,
        "step": 39,
        "planets": [
            [7, 1, 40.94067950847622, 3.0626606347240966, 2.6094379124341005, 23, 5],
            [15, -1, 50.91883994305307, 15.10845967653814, 1.0, 18, 1],
        ],
        "fleets": [],
        "angular_velocity": 0.04,
        "initial_planets": [],
        "comet_planet_ids": [],
        "comets": [],
        "remainingOverageTime": 60.0,
    }

    moves = sniper_agent(obs)
    assert len(moves) == 1
    naive = math.atan2(
        15.10845967653814 - 3.0626606347240966,
        50.91883994305307 - 40.94067950847622,
    )
    assert abs(moves[0][1] - naive) > 0.1


def test_sniper_treats_comet_target_as_non_orbiting():
    source_x, source_y = 30.230528337169282, 37.113714576315715
    target_x, target_y = 47.52883413524248, 97.05854540070332
    obs = {
        "player": 0,
        "step": 50,
        "planets": [
            [13, 0, source_x, source_y, 1.0, 20, 1],
            [29, -1, target_x, target_y, 1.0, 7, 1],
        ],
        "fleets": [],
        "angular_velocity": 0.04,
        "initial_planets": [],
        "comet_planet_ids": [29],
        "comets": [
            {
                "planet_ids": [29],
                "paths": [[[47.52883413524248, 97.05854540070332], [47.0, 95.0]]],
                "path_index": 0,
            }
        ],
        "remainingOverageTime": 60.0,
    }
    static_solution = _lead_solution(
        source_x,
        source_y,
        1.0,
        target_x,
        target_y,
        1.0,
        obs["angular_velocity"],
        8,
        target_is_comet=True,
    )
    orbiting_solution = _lead_solution(
        source_x,
        source_y,
        1.0,
        target_x,
        target_y,
        1.0,
        obs["angular_velocity"],
        8,
        target_is_comet=False,
    )

    moves = sniper_agent(obs)

    assert len(moves) == 1
    assert moves[0][0] == 13
    assert static_solution is not None
    assert orbiting_solution is not None
    assert abs(moves[0][1] - static_solution.angle) < 1e-9
    assert abs(moves[0][1] - orbiting_solution.angle) > 0.1


def test_sniper_skips_route_swept_moving_source_shot():
    obs = {
        "player": 0,
        "step": 0,
        "planets": [
            [0, 0, 80.0, 50.0, 1.0, 50, 3],
            [1, -1, 56.68747071561468, 74.08895463542983, 1.0, 10, 2],
        ],
        "fleets": [],
        "angular_velocity": 0.05,
        "initial_planets": [],
        "comet_planet_ids": [],
        "comets": [],
        "remainingOverageTime": 60.0,
    }

    assert sniper_agent(obs) == []


def test_sniper_holds_when_weak():
    moves = sniper_agent(_obs((10, 10), (90, 90), my_ships=5, enemy_ships=20))
    assert moves == []


def test_heuristic_keeps_reserve():
    a = HeuristicAgent(reserve_ships=10)
    moves = a(_obs((10, 10), (90, 90), my_ships=15, enemy_ships=2))
    # 15 - 10 reserve = 5 budget; need 4 ships, so this is allowed and we send 4.
    assert len(moves) == 1
    assert moves[0][2] <= 5


def test_fleet_target_tracker_resets_and_expires_eta():
    tracker = _FleetTargetTracker()
    obs0 = _obs((10, 10), (90, 90))
    tracker.annotate(obs0)
    tracker.record(obs0, [[0, 0.5, 10, 1, 2.0, 90.0, 90.0]])

    obs1 = _obs((10, 10), (90, 90))
    obs1["step"] = 1
    obs1["fleets"] = [[0, 0, 11.0, 11.0, 0.5, 0, 10]]
    annotated = tracker.annotate(obs1)
    assert annotated["fleet_targets"] == {"0": [1, 1.0, 90.0, 90.0]}

    obs2 = dict(obs1)
    obs2["step"] = 2
    assert tracker.annotate(obs2)["fleet_targets"] == {}

    tracker.by_fleet_id[0] = [1, 5.0, 90.0, 90.0]
    reset_obs = _obs((10, 10), (90, 90))
    reset_obs["fleets"] = [[0, 0, 11.0, 11.0, 0.5, 0, 10]]
    assert tracker.annotate(reset_obs)["fleet_targets"] == {}
