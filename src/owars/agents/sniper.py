"""Nearest-planet sniper baseline.

This started as the competition starter bot, but our local training baseline
leads orbiting targets so it is a useful fixed opponent instead of mostly
teaching the learner to exploit missed shots.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from ..game import distance, fleet_speed, parse_observation
from ..game.geometry import line_circle_intersects
from ..game.types import CENTER, ROTATION_RADIUS_LIMIT, Fleet, Planet
from ..policies.sampling import (
    _lead_solution,
    _route_blockers_from_rows,
    _route_clear_to_solution,
)


def sniper_agent(obs: Any) -> list[list]:
    o = parse_observation(obs)
    targets = o.enemy_planets() + o.neutral_planets()
    if not targets:
        return []
    blockers = _route_blockers_from_rows(o.planets, o.angular_velocity, o.comet_planet_ids)

    moves: list[list] = []
    for mine in o.my_planets():
        for target in sorted(targets, key=lambda t: distance(mine.x, mine.y, t.x, t.y)):
            ships_needed = target.ships + 1
            if mine.ships < ships_needed:
                continue
            solution = _lead_solution(
                mine.x,
                mine.y,
                mine.radius,
                target.x,
                target.y,
                target.radius,
                o.angular_velocity,
                ships_needed,
                target_is_comet=int(target.id) in o.comet_planet_ids,
            )
            if solution is None:
                continue
            if not _route_clear_to_solution(
                mine.id,
                target.id,
                mine.x,
                mine.y,
                mine.radius,
                solution,
                ships_needed,
                blockers,
                o.angular_velocity,
            ):
                continue
            moves.append([mine.id, solution.angle, ships_needed])
            break
    return moves


@dataclass(frozen=True)
class _SniperProfile:
    reserve_base: int
    reserve_production: float
    send_buffer: int
    enemy_growth: bool
    enemy_value: float
    neutral_value: float
    production_weight: float
    ship_cost_weight: float
    time_cost_weight: float
    duplicate_penalty: float
    allow_partial: bool = False
    partial_min_fraction: float = 0.5
    partial_score_scale: float = 0.45
    net_defense_reserve: bool = False
    defense_horizon: float = 35.0
    contested_extra_buffer: int = 0
    contested_window: float = 2.0
    reinforce_owned: bool = False
    defense_arrival_slack: float = 1.0
    defense_score_weight: float = 7.5
    chronological_forecast: bool = False
    comet_max_eta: float | None = None
    counter_recapture: bool = False
    recapture_min_gap: float = 0.5
    recapture_max_gap: float = 8.0
    recapture_score_weight: float = 6.0
    recapture_gap_cost: float = 0.25
    source_order: str = "ships"


_SNIPER_V2 = _SniperProfile(
    reserve_base=6,
    reserve_production=1.2,
    send_buffer=2,
    enemy_growth=True,
    enemy_value=2.4,
    neutral_value=1.35,
    production_weight=5.0,
    ship_cost_weight=0.82,
    time_cost_weight=0.55,
    duplicate_penalty=0.45,
)

_SNIPER_V3 = _SniperProfile(
    reserve_base=3,
    reserve_production=0.7,
    send_buffer=1,
    enemy_growth=True,
    enemy_value=3.1,
    neutral_value=1.15,
    production_weight=5.8,
    ship_cost_weight=0.72,
    time_cost_weight=0.42,
    duplicate_penalty=0.30,
)

_SNIPER_V4 = _SniperProfile(
    reserve_base=2,
    reserve_production=0.5,
    send_buffer=1,
    enemy_growth=True,
    enemy_value=3.4,
    neutral_value=1.05,
    production_weight=6.2,
    ship_cost_weight=0.68,
    time_cost_weight=0.35,
    duplicate_penalty=0.22,
)

_SNIPER_V5 = _SniperProfile(
    reserve_base=2,
    reserve_production=0.45,
    send_buffer=1,
    enemy_growth=True,
    enemy_value=3.7,
    neutral_value=1.0,
    production_weight=6.6,
    ship_cost_weight=0.64,
    time_cost_weight=0.34,
    duplicate_penalty=0.16,
    allow_partial=True,
    partial_min_fraction=0.38,
    partial_score_scale=0.62,
)

_SNIPER_V6 = _SniperProfile(
    reserve_base=1,
    reserve_production=0.35,
    send_buffer=1,
    enemy_growth=True,
    enemy_value=3.55,
    neutral_value=1.0,
    production_weight=6.5,
    ship_cost_weight=0.66,
    time_cost_weight=0.34,
    duplicate_penalty=0.20,
    net_defense_reserve=True,
    defense_horizon=42.0,
)

_SNIPER_V7 = _SniperProfile(
    reserve_base=2,
    reserve_production=0.5,
    send_buffer=1,
    enemy_growth=True,
    enemy_value=3.5,
    neutral_value=1.0,
    production_weight=6.4,
    ship_cost_weight=0.67,
    time_cost_weight=0.34,
    duplicate_penalty=0.20,
    contested_extra_buffer=2,
    contested_window=2.0,
)

_SNIPER_V8 = _SniperProfile(
    reserve_base=1,
    reserve_production=0.35,
    send_buffer=1,
    enemy_growth=True,
    enemy_value=3.55,
    neutral_value=1.0,
    production_weight=6.5,
    ship_cost_weight=0.66,
    time_cost_weight=0.34,
    duplicate_penalty=0.20,
    net_defense_reserve=True,
    defense_horizon=42.0,
    reinforce_owned=True,
    defense_arrival_slack=1.0,
    defense_score_weight=9.0,
)

_SNIPER_V9 = _SniperProfile(
    reserve_base=1,
    reserve_production=0.35,
    send_buffer=1,
    enemy_growth=True,
    enemy_value=3.55,
    neutral_value=1.0,
    production_weight=6.5,
    ship_cost_weight=0.66,
    time_cost_weight=0.34,
    duplicate_penalty=0.20,
    net_defense_reserve=True,
    defense_horizon=42.0,
    reinforce_owned=True,
    defense_arrival_slack=1.0,
    defense_score_weight=9.0,
    chronological_forecast=True,
)

_SNIPER_V10 = _SniperProfile(
    reserve_base=1,
    reserve_production=0.35,
    send_buffer=1,
    enemy_growth=True,
    enemy_value=3.55,
    neutral_value=1.0,
    production_weight=6.5,
    ship_cost_weight=0.66,
    time_cost_weight=0.34,
    duplicate_penalty=0.20,
    net_defense_reserve=True,
    defense_horizon=42.0,
    reinforce_owned=True,
    defense_arrival_slack=1.0,
    defense_score_weight=9.0,
    comet_max_eta=8.0,
)

_SNIPER_V11 = _SniperProfile(
    reserve_base=1,
    reserve_production=0.35,
    send_buffer=1,
    enemy_growth=True,
    enemy_value=3.55,
    neutral_value=1.0,
    production_weight=6.5,
    ship_cost_weight=0.66,
    time_cost_weight=0.34,
    duplicate_penalty=0.20,
    net_defense_reserve=True,
    defense_horizon=42.0,
    reinforce_owned=True,
    defense_arrival_slack=1.0,
    defense_score_weight=9.0,
    comet_max_eta=8.0,
    counter_recapture=True,
    recapture_min_gap=0.5,
    recapture_max_gap=8.0,
    recapture_score_weight=6.0,
    recapture_gap_cost=0.25,
)


def sniper_v2_agent(obs: Any) -> list[list]:
    """Production-aware value sniper.

    Keeps a small reserve, scores targets by production and owner, accounts for
    enemy production before arrival, and avoids overcommitting multiple sources
    to the same target in one turn.
    """

    return _scored_sniper(obs, _SNIPER_V2)


def sniper_v3_agent(obs: Any) -> list[list]:
    """More aggressive scored sniper variant for benchmark sweeps."""

    return _scored_sniper(obs, _SNIPER_V3)


def sniper_v4_agent(obs: Any) -> list[list]:
    """High-tempo scored sniper variant for benchmark sweeps."""

    return _scored_sniper(obs, _SNIPER_V4)


def sniper_v5_agent(obs: Any) -> list[list]:
    """Coordinated high-tempo sniper variant for benchmark sweeps."""

    return _scored_sniper(obs, _SNIPER_V5)


def sniper_v6_agent(obs: Any) -> list[list]:
    """Net-defense high-tempo sniper variant for benchmark sweeps."""

    return _scored_sniper(obs, _SNIPER_V6)


def sniper_v7_agent(obs: Any) -> list[list]:
    """Contested-arrival high-tempo sniper variant for benchmark sweeps."""

    return _scored_sniper(obs, _SNIPER_V7)


def sniper_v8_agent(obs: Any) -> list[list]:
    """Emergency-defense high-tempo sniper variant for benchmark sweeps."""

    return _scored_sniper(obs, _SNIPER_V8)


def sniper_v9_agent(obs: Any) -> list[list]:
    """Chronological-forecast sniper variant for benchmark sweeps."""

    return _scored_sniper(obs, _SNIPER_V9)


def sniper_v10_agent(obs: Any) -> list[list]:
    """Comet-sane emergency-defense sniper variant for benchmark sweeps."""

    return _scored_sniper(obs, _SNIPER_V10)


def sniper_v11_agent(obs: Any) -> list[list]:
    """Comet-sane counter-recapture sniper variant for benchmark sweeps."""

    return _scored_sniper(obs, _SNIPER_V11)


def _scored_sniper(obs: Any, profile: _SniperProfile) -> list[list]:
    o = parse_observation(obs)
    targets = o.enemy_planets() + o.neutral_planets()
    my_planets = o.my_planets()
    if not targets and not my_planets:
        return []

    blockers = _route_blockers_from_rows(o.planets, o.angular_velocity, o.comet_planet_ids)
    if profile.source_order == "ships":
        my_planets = sorted(my_planets, key=lambda p: (p.ships, p.production), reverse=True)
    planned_by_target: dict[int, list[tuple[float, int]]] = {}
    pressure = _fleet_pressure(o)
    moves: list[list] = []
    for mine in my_planets:
        move = _best_scored_move(o, mine, targets, blockers, planned_by_target, pressure, profile)
        if move is None:
            continue
        target_id, eta, action = move
        planned_by_target.setdefault(target_id, []).append((eta, int(action[2])))
        moves.append(action)
    return moves


def _best_scored_move(
    o,
    mine: Planet,
    targets: list[Planet],
    blockers,
    planned_by_target: dict[int, list[tuple[float, int]]],
    pressure: dict[int, list[tuple[float, int, int]]],
    profile: _SniperProfile,
) -> tuple[int, float, list] | None:
    reserve = int(math.ceil(profile.reserve_base + profile.reserve_production * mine.production))
    if profile.net_defense_reserve:
        reserve += _defensive_reserve(
            pressure,
            int(mine.id),
            int(mine.owner),
            int(mine.production),
            profile.defense_horizon,
        )
    else:
        reserve += _enemy_pressure_by(pressure, int(mine.id), int(mine.owner), profile.defense_horizon)
    budget = mine.ships - reserve
    if budget <= 1:
        return None

    best: tuple[float, int, float, list] | None = None
    if profile.reinforce_owned:
        for target in o.my_planets():
            if int(target.id) == int(mine.id):
                continue
            planned = planned_by_target.get(int(target.id), [])
            candidate = _defense_candidate_action(
                o,
                mine,
                target,
                blockers,
                budget,
                planned,
                pressure,
                profile,
            )
            if candidate is not None:
                score, eta, action = candidate
                if best is None or score > best[0]:
                    best = (score, int(target.id), eta, action)
            if profile.counter_recapture:
                candidate = _recapture_candidate_action(
                    o,
                    mine,
                    target,
                    blockers,
                    budget,
                    planned,
                    pressure,
                    profile,
                )
                if candidate is not None:
                    score, eta, action = candidate
                    if best is None or score > best[0]:
                        best = (score, int(target.id), eta, action)

    for target in targets:
        planned = planned_by_target.get(int(target.id), [])
        candidate = _candidate_action(o, mine, target, blockers, budget, planned, pressure, profile)
        if candidate is None:
            continue
        score, eta, action = candidate
        if best is None or score > best[0]:
            best = (score, int(target.id), eta, action)
    if best is None:
        return None
    return best[1], best[2], best[3]


def _candidate_action(
    o,
    mine: Planet,
    target: Planet,
    blockers,
    budget: int,
    planned: list[tuple[float, int]],
    pressure: dict[int, list[tuple[float, int, int]]],
    profile: _SniperProfile,
) -> tuple[float, float, list] | None:
    ships_needed = target.ships + profile.send_buffer - _planned_by(planned, 0.0)
    if ships_needed <= 0:
        return None
    partial_required = ships_needed
    partial = False
    solution = None
    for _ in range(4):
        if ships_needed > budget:
            if not _partial_allowed(ships_needed, budget, profile):
                return None
            partial_required = ships_needed
            ships_needed = budget
            partial = True
        solution = _lead_solution(
            mine.x,
            mine.y,
            mine.radius,
            target.x,
            target.y,
            target.radius,
            o.angular_velocity,
            ships_needed,
            target_is_comet=int(target.id) in o.comet_planet_ids,
        )
        if solution is None:
            return None
        if (
            profile.comet_max_eta is not None
            and int(target.id) in o.comet_planet_ids
            and solution.time > profile.comet_max_eta
        ):
            return None
        revised = _ships_needed_at_eta(o, target, solution.time, planned, pressure, profile)
        if revised <= 0:
            return None
        if partial and revised >= ships_needed:
            partial_required = revised
            break
        if revised == ships_needed:
            break
        ships_needed = revised
    if solution is None or ships_needed > budget:
        return None

    if not _route_clear_to_solution(
        mine.id,
        target.id,
        mine.x,
        mine.y,
        mine.radius,
        solution,
        ships_needed,
        blockers,
        o.angular_velocity,
    ):
        return None

    owner_value = profile.neutral_value if target.owner == -1 else profile.enemy_value
    production_value = owner_value * (1.0 + profile.production_weight * target.production)
    distance_bonus = 1.0 / (1.0 + 0.02 * distance(mine.x, mine.y, target.x, target.y))
    already_planned = _planned_by(planned, solution.time)
    duplicate_scale = 1.0 / (1.0 + profile.duplicate_penalty * max(0, already_planned))
    cost = (
        profile.ship_cost_weight * max(1, ships_needed)
        + profile.time_cost_weight * max(1.0, solution.time)
    )
    score = production_value * distance_bonus * duplicate_scale / max(1.0, cost)
    if partial:
        score *= profile.partial_score_scale * (ships_needed / max(1.0, float(partial_required)))
    return score, solution.time, [mine.id, solution.angle, int(ships_needed)]


def _defense_candidate_action(
    o,
    mine: Planet,
    target: Planet,
    blockers,
    budget: int,
    planned: list[tuple[float, int]],
    pressure: dict[int, list[tuple[float, int, int]]],
    profile: _SniperProfile,
) -> tuple[float, float, list] | None:
    threat = _defense_need(target, planned, pressure, profile.defense_horizon)
    if threat is None:
        return None
    threat_eta, ships_needed = threat
    ships_needed = min(ships_needed, budget)
    if ships_needed <= 0:
        return None
    solution = _lead_solution(
        mine.x,
        mine.y,
        mine.radius,
        target.x,
        target.y,
        target.radius,
        o.angular_velocity,
        ships_needed,
        target_is_comet=int(target.id) in o.comet_planet_ids,
    )
    if solution is None or solution.time > threat_eta + profile.defense_arrival_slack:
        return None
    if not _route_clear_to_solution(
        mine.id,
        target.id,
        mine.x,
        mine.y,
        mine.radius,
        solution,
        ships_needed,
        blockers,
        o.angular_velocity,
    ):
        return None
    urgency = 1.0 + max(0.0, profile.defense_horizon - threat_eta) / profile.defense_horizon
    value = profile.defense_score_weight * urgency * (2.0 + target.production)
    cost = ships_needed + 0.35 * max(1.0, solution.time)
    return value / max(1.0, cost), solution.time, [mine.id, solution.angle, int(ships_needed)]


def _defense_need(
    target: Planet,
    planned: list[tuple[float, int]],
    pressure: dict[int, list[tuple[float, int, int]]],
    horizon: float,
) -> tuple[float, int] | None:
    best: tuple[float, int] | None = None
    for arrival, fleet_owner, _ships in sorted(pressure.get(int(target.id), [])):
        if fleet_owner == target.owner or arrival > horizon:
            continue
        hostile = _enemy_pressure_by(pressure, int(target.id), int(target.owner), arrival)
        friendly = _pressure_by(pressure, int(target.id), int(target.owner), arrival)
        planned_friendly = _planned_by(planned, arrival)
        produced = int(math.floor(max(0.0, arrival) * target.production))
        deficit = hostile + 1 - target.ships - produced - friendly - planned_friendly
        if deficit > 0 and (best is None or arrival < best[0]):
            best = (arrival, deficit)
    return best


def _recapture_candidate_action(
    o,
    mine: Planet,
    target: Planet,
    blockers,
    budget: int,
    planned: list[tuple[float, int]],
    pressure: dict[int, list[tuple[float, int, int]]],
    profile: _SniperProfile,
) -> tuple[float, float, list] | None:
    capture = _project_hostile_capture(target, planned, pressure, profile.defense_horizon)
    if capture is None:
        return None
    capture_eta, captor, surplus = capture
    ships_needed = surplus + profile.send_buffer
    solution = None
    for _ in range(4):
        if ships_needed > budget:
            return None
        solution = _lead_solution(
            mine.x,
            mine.y,
            mine.radius,
            target.x,
            target.y,
            target.radius,
            o.angular_velocity,
            ships_needed,
            target_is_comet=int(target.id) in o.comet_planet_ids,
        )
        if solution is None:
            return None
        gap = solution.time - capture_eta
        if gap < profile.recapture_min_gap or gap > profile.recapture_max_gap:
            return None
        later_enemy = _pressure_between(pressure, int(target.id), captor, capture_eta, solution.time)
        planned_recapture = _planned_between(planned, capture_eta, solution.time)
        revised = max(
            1,
            surplus
            + int(math.floor(max(0.0, gap) * target.production))
            + later_enemy
            + profile.send_buffer
            - planned_recapture,
        )
        if revised == ships_needed:
            break
        ships_needed = revised
    if solution is None or ships_needed > budget:
        return None
    if not _route_clear_to_solution(
        mine.id,
        target.id,
        mine.x,
        mine.y,
        mine.radius,
        solution,
        ships_needed,
        blockers,
        o.angular_velocity,
    ):
        return None
    gap = solution.time - capture_eta
    urgency = 1.0 + max(0.0, profile.defense_horizon - capture_eta) / profile.defense_horizon
    gap_penalty = 1.0 / (1.0 + profile.recapture_gap_cost * gap)
    value = profile.recapture_score_weight * urgency * (2.0 + target.production) * gap_penalty
    cost = ships_needed + 0.35 * max(1.0, solution.time)
    return value / max(1.0, cost), solution.time, [mine.id, solution.angle, int(ships_needed)]


def _project_hostile_capture(
    target: Planet,
    planned: list[tuple[float, int]],
    pressure: dict[int, list[tuple[float, int, int]]],
    horizon: float,
) -> tuple[float, int, int] | None:
    for arrival, fleet_owner, _ships in sorted(pressure.get(int(target.id), [])):
        if fleet_owner == target.owner or arrival > horizon:
            continue
        hostile = _enemy_pressure_by(pressure, int(target.id), int(target.owner), arrival)
        friendly = _pressure_by(pressure, int(target.id), int(target.owner), arrival)
        planned_friendly = _planned_by(planned, arrival)
        produced = int(math.floor(max(0.0, arrival) * target.production))
        surplus = hostile - target.ships - produced - friendly - planned_friendly
        if surplus > 0:
            return arrival, fleet_owner, surplus
    return None


def _pressure_between(
    pressure: dict[int, list[tuple[float, int, int]]],
    target_id: int,
    owner: int,
    start: float,
    end: float,
) -> int:
    return sum(
        ships
        for arrival, fleet_owner, ships in pressure.get(target_id, [])
        if fleet_owner == owner and start < arrival <= end + 1.0
    )


def _planned_between(planned: list[tuple[float, int]], start: float, end: float) -> int:
    return sum(ships for arrival, ships in planned if start < arrival <= end + 1.0)


def _partial_allowed(ships_needed: int, budget: int, profile: _SniperProfile) -> bool:
    if not profile.allow_partial or budget <= 1:
        return False
    return budget >= max(2, int(math.ceil(ships_needed * profile.partial_min_fraction)))


def _ships_needed_at_eta(
    o,
    target: Planet,
    eta: float,
    planned: list[tuple[float, int]],
    pressure: dict[int, list[tuple[float, int, int]]],
    profile: _SniperProfile,
) -> int:
    if profile.chronological_forecast:
        owner, ships = _forecast_target_state(o.player, target, eta, planned, pressure)
        if owner == o.player:
            return 0
        return max(0, ships + profile.send_buffer)

    friendly = _planned_by(planned, eta) + _pressure_by(pressure, int(target.id), o.player, eta)
    contested_eta = eta + profile.contested_window
    if target.owner != -1 and target.owner != o.player:
        hostile = _pressure_by(pressure, int(target.id), int(target.owner), eta)
        contested_hostile = _pressure_by(pressure, int(target.id), int(target.owner), contested_eta)
    else:
        hostile = _non_player_pressure_by(pressure, int(target.id), o.player, eta)
        contested_hostile = _non_player_pressure_by(pressure, int(target.id), o.player, contested_eta)
    growth = 0
    if profile.enemy_growth and target.owner not in (o.player, -1):
        growth = int(math.ceil(target.production * max(1.0, eta)))
    contested_buffer = profile.contested_extra_buffer if contested_hostile > friendly else 0
    return max(0, target.ships + growth + hostile + profile.send_buffer + contested_buffer - friendly)


def _forecast_target_state(
    player: int,
    target: Planet,
    eta: float,
    planned: list[tuple[float, int]],
    pressure: dict[int, list[tuple[float, int, int]]],
) -> tuple[int, int]:
    target_id = int(target.id)
    horizon = max(0, int(math.ceil(eta)))
    events: dict[int, dict[int, int]] = {}
    for arrival, owner, ships in pressure.get(target_id, []):
        turn = int(math.ceil(arrival))
        if 0 <= turn <= horizon:
            events.setdefault(turn, {})[owner] = events.setdefault(turn, {}).get(owner, 0) + ships
    for arrival, ships in planned:
        turn = int(math.ceil(arrival))
        if 0 <= turn <= horizon:
            events.setdefault(turn, {})[player] = events.setdefault(turn, {}).get(player, 0) + ships

    owner = int(target.owner)
    garrison = int(target.ships)
    prev_turn = 0
    for turn in sorted(events):
        if owner != -1:
            garrison += max(0, turn - prev_turn) * int(target.production)
        survivor_owner, survivor_ships = _resolve_arrivals(events[turn])
        if survivor_ships > 0:
            if survivor_owner == owner:
                garrison += survivor_ships
            else:
                garrison -= survivor_ships
                if garrison < 0:
                    owner = survivor_owner
                    garrison = -garrison
        prev_turn = turn
    if owner != -1:
        garrison += max(0, horizon - prev_turn) * int(target.production)
    return owner, max(0, garrison)


def _resolve_arrivals(arrivals: dict[int, int]) -> tuple[int, int]:
    rows = sorted(arrivals.items(), key=lambda item: (-item[1], item[0]))
    if not rows:
        return -1, 0
    if len(rows) == 1:
        return rows[0]
    diff = rows[0][1] - rows[1][1]
    if diff <= 0:
        return -1, 0
    return rows[0][0], diff


def _planned_by(planned: list[tuple[float, int]], eta: float) -> int:
    return sum(ships for arrival, ships in planned if arrival <= eta + 1.0)


def _pressure_by(
    pressure: dict[int, list[tuple[float, int, int]]],
    target_id: int,
    owner: int,
    eta: float,
) -> int:
    return sum(
        ships
        for arrival, fleet_owner, ships in pressure.get(target_id, [])
        if fleet_owner == owner and arrival <= eta + 1.0
    )


def _non_player_pressure_by(
    pressure: dict[int, list[tuple[float, int, int]]],
    target_id: int,
    player: int,
    eta: float,
) -> int:
    return sum(
        ships
        for arrival, fleet_owner, ships in pressure.get(target_id, [])
        if fleet_owner != player and arrival <= eta + 1.0
    )


def _enemy_pressure_by(
    pressure: dict[int, list[tuple[float, int, int]]],
    target_id: int,
    owner: int,
    eta: float,
) -> int:
    return sum(
        ships
        for arrival, fleet_owner, ships in pressure.get(target_id, [])
        if fleet_owner != owner and arrival <= eta + 1.0
    )


def _defensive_reserve(
    pressure: dict[int, list[tuple[float, int, int]]],
    target_id: int,
    owner: int,
    production: int,
    eta: float,
) -> int:
    needed = 0
    for arrival, fleet_owner, _ships in pressure.get(target_id, []):
        if fleet_owner == owner or arrival > eta:
            continue
        hostile = _enemy_pressure_by(pressure, target_id, owner, arrival)
        friendly = _pressure_by(pressure, target_id, owner, arrival)
        produced = int(math.floor(max(0.0, arrival) * production))
        needed = max(needed, hostile - friendly - produced + 1)
    return max(0, needed)


def _fleet_pressure(o) -> dict[int, list[tuple[float, int, int]]]:
    pressure: dict[int, list[tuple[float, int, int]]] = {int(p.id): [] for p in o.planets}
    for fleet in o.fleets:
        inferred = _inferred_fleet_target(
            fleet,
            o.planets,
            o.angular_velocity,
            o.comet_planet_ids,
        )
        if inferred is None:
            continue
        target, eta = inferred
        pressure[int(target.id)].append((eta, int(fleet.owner), int(fleet.ships)))
    return pressure


def _inferred_fleet_target(
    fleet: Fleet,
    planets: list[Planet],
    angular_velocity: float,
    comet_planet_ids: set[int],
) -> tuple[Planet, float] | None:
    speed = fleet_speed(fleet.ships)
    dx = math.cos(fleet.angle)
    dy = math.sin(fleet.angle)
    max_turns = min(180, max(1, int(math.ceil(150.0 / max(1e-6, speed))) + 2))
    best: tuple[float, Planet] | None = None
    for planet in planets:
        for turn in range(1, max_turns + 1):
            old_x = fleet.x + dx * speed * (turn - 1)
            old_y = fleet.y + dy * speed * (turn - 1)
            new_x = fleet.x + dx * speed * turn
            new_y = fleet.y + dy * speed * turn
            px, py = _planet_position_at(
                planet,
                angular_velocity,
                turn - 1,
                comet_planet_ids,
            )
            if not line_circle_intersects(
                old_x,
                old_y,
                new_x,
                new_y,
                px,
                py,
                planet.radius + 0.05,
            ):
                continue
            eta = float(turn)
            if best is None or eta < best[0]:
                best = (eta, planet)
            break
    return None if best is None else (best[1], best[0])


def _planet_position_at(
    planet: Planet,
    angular_velocity: float,
    steps: int,
    comet_planet_ids: set[int],
) -> tuple[float, float]:
    if int(planet.id) in comet_planet_ids:
        return planet.x, planet.y
    dx = planet.x - CENTER[0]
    dy = planet.y - CENTER[1]
    orbital_radius = math.hypot(dx, dy)
    if orbital_radius + planet.radius >= ROTATION_RADIUS_LIMIT or abs(angular_velocity) <= 1e-12:
        return planet.x, planet.y
    theta = math.atan2(dy, dx) + angular_velocity * steps
    return (
        CENTER[0] + orbital_radius * math.cos(theta),
        CENTER[1] + orbital_radius * math.sin(theta),
    )
