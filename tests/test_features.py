"""Encoding-level tests: owner one-hot, orbit params, fleet provenance."""

from __future__ import annotations

import math

from owars.game import parse_observation
from owars.policies.features import (
    FLEET_FEAT_DIM,
    PLANET_FEAT_DIM,
    _owner_onehot,
    _planet_motion,
    encode_observation,
)
from owars.game.types import CENTER, Planet


def test_owner_onehot_self_neutral():
    assert _owner_onehot(-1, 0, 4) == (0.0, 1.0, 0.0, 0.0, 0.0)
    assert _owner_onehot(0, 0, 4) == (1.0, 0.0, 0.0, 0.0, 0.0)
    assert _owner_onehot(2, 2, 4) == (1.0, 0.0, 0.0, 0.0, 0.0)


def test_owner_onehot_ffa_stable_seat_relative():
    # 4-player FFA: each seat sees its 3 enemies in the same canonical order.
    for player in range(4):
        slots: list[tuple[int, int]] = []  # (other_id, enemy_slot_idx)
        for other in range(4):
            if other == player:
                continue
            vec = _owner_onehot(other, player, 4)
            assert vec[0] == 0.0 and vec[1] == 0.0  # not self / neutral
            slot = vec[2:].index(1.0)
            slots.append((other, slot))
        # Each enemy ends up in a unique slot.
        slot_ids = sorted(s for _, s in slots)
        assert slot_ids == [0, 1, 2]
        for other, slot in slots:
            assert ((other - player) % 4) - 1 == slot


def test_owner_onehot_2p_symmetric_enemy_0():
    # Both seats in a 2-player game must see the opponent in enemy_0,
    # otherwise the model has to learn each seat's view independently.
    for player in (0, 1):
        opp = 1 - player
        vec = _owner_onehot(opp, player, 2)
        assert vec == (0.0, 0.0, 1.0, 0.0, 0.0), f"seat {player} saw {vec}"


def test_planet_motion_static_planet():
    # A planet placed beyond the rotation limit: heading and orbit params zero.
    cx, cy = CENTER
    p = Planet(id=0, owner=0, x=cx + 60.0, y=cy, radius=1.0, ships=10, production=2)
    cos_h, sin_h, sp, orb_r, om, is_orb, is_com = _planet_motion(p, 0.04, {})
    assert cos_h == 0.0 and sin_h == 0.0
    assert sp == 0.0 and om == 0.0
    assert is_orb == 0.0 and is_com == 0.0
    assert orb_r > 0.0  # we still report the (large) orbital radius


def test_planet_motion_orbiter_tangent_perpendicular_to_radial():
    cx, cy = CENTER
    # Orbiter on +x axis. Radial direction is (+1, 0); CCW tangent is (0, +1).
    p = Planet(id=1, owner=0, x=cx + 15.0, y=cy, radius=1.0, ships=10, production=2)
    cos_h, sin_h, sp, orb_r, om, is_orb, is_com = _planet_motion(p, 0.04, {})
    assert is_orb == 1.0
    # |dot(heading, radial)| should be ~0 (perpendicular).
    assert abs(cos_h) < 1e-6
    assert abs(sin_h - 1.0) < 1e-6  # +ω → +y tangent
    assert sp > 0.0
    assert om > 0.0


def test_planet_motion_comet_uses_path_step():
    p = Planet(id=7, owner=-1, x=10.0, y=10.0, radius=1.0, ships=4, production=1)
    cos_h, sin_h, sp, orb_r, om, is_orb, is_com = _planet_motion(p, 0.04, {7: (3.0, 4.0)})
    assert is_com == 1.0 and is_orb == 0.0
    assert math.isclose(cos_h, 0.6, abs_tol=1e-6)
    assert math.isclose(sin_h, 0.8, abs_tol=1e-6)
    assert sp > 0.0


def _toy_obs_with_fleet():
    return {
        "player": 0,
        "step": 0,
        "planets": [
            [0, 0, 10.0, 10.0, 1.0, 50, 3],
            [1, 1, 90.0, 90.0, 1.0, 30, 2],
        ],
        "fleets": [[42, 0, 30.0, 30.0, 0.5, 0, 20]],
        "angular_velocity": 0.04,
        "initial_planets": [],
        "comet_planet_ids": [],
        "comets": [],
        "remainingOverageTime": 60.0,
    }


def test_fleet_features_include_provenance():
    o = parse_observation(_toy_obs_with_fleet())
    feats = encode_observation(o)
    assert feats.fleet_feats.shape[1] == FLEET_FEAT_DIM
    assert feats.planet_feats.shape[1] == PLANET_FEAT_DIM
    fleet_row = feats.fleet_feats[0].tolist()
    # `from_planet_id=0`, planet 0 is at (10, 10): normalized = (10-50)/100=-0.4.
    from_nx = fleet_row[5]
    from_ny = fleet_row[6]
    has_from = fleet_row[7]
    assert math.isclose(from_nx, -0.4, abs_tol=1e-6)
    assert math.isclose(from_ny, -0.4, abs_tol=1e-6)
    assert has_from == 1.0


def test_fleet_features_handle_missing_source():
    obs = _toy_obs_with_fleet()
    obs["fleets"] = [[42, 0, 30.0, 30.0, 0.5, 999, 20]]  # nonexistent source
    o = parse_observation(obs)
    feats = encode_observation(o)
    fleet_row = feats.fleet_feats[0].tolist()
    assert fleet_row[5] == 0.0 and fleet_row[6] == 0.0
    assert fleet_row[7] == 0.0  # has_from bit
