import math

from owars.agents import HeuristicAgent, random_agent, sniper_agent
from owars.agents.learned import _FleetTargetTracker


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
    moves = sniper_agent(_obs((10, 10), (90, 90), my_ships=50, enemy_ships=10))
    assert len(moves) == 1
    from_id, angle, ships = moves[0]
    assert from_id == 0
    assert ships == 11  # garrison + 1
    # Aimed at +x +y direction (target is to the lower-right).
    assert angle > 0


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
