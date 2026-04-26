from owars.game import parse_observation
from owars.game.types import Fleet, Planet


def _toy_obs(player: int = 0) -> dict:
    return {
        "player": player,
        "step": 5,
        "planets": [
            [0, player, 10.0, 10.0, 1.0, 50, 3],   # owned
            [1, 1 - player, 90.0, 90.0, 1.0, 30, 2],  # enemy
            [2, -1, 50.0, 90.0, 1.0, 10, 1],       # neutral
        ],
        "fleets": [
            [0, player, 30.0, 30.0, 0.5, 0, 20],
            [1, 1 - player, 70.0, 70.0, -0.5, 1, 10],
        ],
        "angular_velocity": 0.04,
        "initial_planets": [],
        "comet_planet_ids": [],
        "comets": [],
        "remainingOverageTime": 60.0,
    }


def test_parse_typed_lists():
    o = parse_observation(_toy_obs())
    assert all(isinstance(p, Planet) for p in o.planets)
    assert all(isinstance(f, Fleet) for f in o.fleets)
    assert o.player == 0
    assert o.step == 5
    assert o.angular_velocity == 0.04


def test_partition_by_owner():
    o = parse_observation(_toy_obs())
    assert {p.id for p in o.my_planets()} == {0}
    assert {p.id for p in o.enemy_planets()} == {1}
    assert {p.id for p in o.neutral_planets()} == {2}
    assert {f.id for f in o.my_fleets()} == {0}
    assert {f.id for f in o.enemy_fleets()} == {1}
