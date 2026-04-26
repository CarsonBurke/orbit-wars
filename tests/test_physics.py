from owars.game.physics import fleet_speed, travel_steps


def test_fleet_speed_one_ship_is_one():
    assert fleet_speed(1) == 1.0


def test_fleet_speed_monotonic_with_size():
    speeds = [fleet_speed(n) for n in (1, 2, 10, 100, 500, 1000, 5000)]
    assert all(a <= b + 1e-9 for a, b in zip(speeds, speeds[1:], strict=False))


def test_fleet_speed_caps_near_max():
    # ~1000 ships should hit the cap (default 6.0).
    assert abs(fleet_speed(1000) - 6.0) < 1e-6
    # Larger fleets exceed the modeled cap (formula doesn't clamp), but
    # the simulator does — that's the runtime concern, not ours here.


def test_travel_steps_small_distance():
    # 5 ships, distance 5 — at speed >= 1 this is 5 turns or fewer.
    s = travel_steps(distance=5.0, ships=5)
    assert 1 <= s <= 5


def test_travel_steps_large_fleet_arrives_faster():
    s_small = travel_steps(distance=50.0, ships=10)
    s_large = travel_steps(distance=50.0, ships=1000)
    assert s_large < s_small
