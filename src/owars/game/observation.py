"""Typed wrapper around the raw obs dict.

The Kaggle gateway hands the agent a dict (or namedtuple, depending on
context). `parse_observation` normalizes it into something we can rely on
without scattering `obs.get("...")` across every agent.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .types import Fleet, Planet


@dataclass
class Observation:
    player: int
    step: int
    planets: list[Planet]
    fleets: list[Fleet]
    angular_velocity: float
    initial_planets: list[Planet]
    comet_planet_ids: set[int]
    comets: list[dict[str, Any]] = field(default_factory=list)
    remaining_overage_time: float = 0.0
    # Total episode length. Defaults to the competition's fixed 500 when the obs
    # does not carry it; the destination oracle uses it to size its lookahead so
    # Python forecasts match the simulator exactly at any episode length.
    episode_steps: int = 500
    raw: dict[str, Any] | None = None

    def my_planets(self) -> list[Planet]:
        return [p for p in self.planets if p.owner == self.player]

    def enemy_planets(self) -> list[Planet]:
        return [p for p in self.planets if p.owner not in (self.player, -1)]

    def neutral_planets(self) -> list[Planet]:
        return [p for p in self.planets if p.owner == -1]

    def my_fleets(self) -> list[Fleet]:
        return [f for f in self.fleets if f.owner == self.player]

    def enemy_fleets(self) -> list[Fleet]:
        return [f for f in self.fleets if f.owner != self.player]


def _get(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def parse_observation(obs: Any) -> Observation:
    raw_planets = _get(obs, "planets", []) or []
    raw_fleets = _get(obs, "fleets", []) or []
    raw_initial = _get(obs, "initial_planets", []) or []
    return Observation(
        player=int(_get(obs, "player", 0) or 0),
        step=int(_get(obs, "step", 0) or 0),
        planets=[Planet(*p) for p in raw_planets],
        fleets=[Fleet(*f) for f in raw_fleets],
        angular_velocity=float(_get(obs, "angular_velocity", 0.0) or 0.0),
        initial_planets=[Planet(*p) for p in raw_initial],
        comet_planet_ids=set(_get(obs, "comet_planet_ids", []) or []),
        comets=list(_get(obs, "comets", []) or []),
        remaining_overage_time=float(_get(obs, "remainingOverageTime", 0.0) or 0.0),
        episode_steps=int(_get(obs, "episode_steps", 500) or 500),
        raw=obs if isinstance(obs, dict) else None,
    )
