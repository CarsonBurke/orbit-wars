"""NumPy implementation of the Orbit Wars simulator.

This module mirrors the public `kaggle_environments` Orbit Wars interpreter
closely enough for rollout generation while exposing a Gym-friendly API.
The hot path stores planets and fleets in dense arrays; observations are
materialized as plain Kaggle-style dictionaries at the boundary.
"""

from __future__ import annotations

import math
import random
from collections import defaultdict
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import numpy as np

BOARD_SIZE = 100.0
CENTER = 50.0
SUN_RADIUS = 10.0
ROTATION_RADIUS_LIMIT = 50.0
COMET_RADIUS = 1.0
COMET_PRODUCTION = 1
PLANET_CLEARANCE = 7
MIN_PLANET_GROUPS = 5
MAX_PLANET_GROUPS = 10
MIN_STATIC_GROUPS = 3
COMET_SPAWN_STEPS = (50, 150, 250, 350, 450)
COMET_T = np.linspace(0.3 * math.pi, 1.7 * math.pi, 5000, dtype=np.float64)
COMET_COS_T = np.cos(COMET_T)
COMET_SIN_T = np.sin(COMET_T)

P_ID = 0
P_OWNER = 1
P_X = 2
P_Y = 3
P_RADIUS = 4
P_SHIPS = 5
P_PROD = 6

F_ID = 0
F_OWNER = 1
F_X = 2
F_Y = 3
F_ANGLE = 4
F_FROM = 5
F_SHIPS = 6


@dataclass(slots=True)
class NumpyOrbitWarsConfig:
    num_players: int = 2
    episode_steps: int = 500
    ship_speed: float = 6.0
    comet_speed: float = 4.0
    random_seed: int | None = None


def _distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def _point_to_segment_distance(
    p: tuple[float, float], v: tuple[float, float], w: tuple[float, float]
) -> float:
    l2 = (v[0] - w[0]) ** 2 + (v[1] - w[1]) ** 2
    if l2 == 0.0:
        return _distance(p, v)
    t = max(
        0.0,
        min(1.0, ((p[0] - v[0]) * (w[0] - v[0]) + (p[1] - v[1]) * (w[1] - v[1])) / l2),
    )
    proj = (v[0] + t * (w[0] - v[0]), v[1] + t * (w[1] - v[1]))
    return _distance(p, proj)


def _points_to_segments_distance(
    points: np.ndarray, starts: np.ndarray, ends: np.ndarray
) -> np.ndarray:
    """Distance from each point to its matching segment.

    `points`, `starts`, and `ends` are `[N, 2]`.
    """
    seg = ends - starts
    l2 = np.einsum("ij,ij->i", seg, seg)
    out = np.empty(points.shape[0], dtype=np.float64)
    zero = l2 == 0.0
    if np.any(zero):
        out[zero] = np.linalg.norm(points[zero] - starts[zero], axis=1)
    nz = ~zero
    if np.any(nz):
        t = np.einsum("ij,ij->i", points[nz] - starts[nz], seg[nz]) / l2[nz]
        t = np.clip(t, 0.0, 1.0)
        proj = starts[nz] + t[:, None] * seg[nz]
        out[nz] = np.linalg.norm(points[nz] - proj, axis=1)
    return out


def _all_point_segment_distances(
    points: np.ndarray, starts: np.ndarray, ends: np.ndarray
) -> np.ndarray:
    """Distance matrix `[num_segments, num_points]`."""
    seg = ends - starts
    l2 = np.einsum("ij,ij->i", seg, seg)
    diff = points[None, :, :] - starts[:, None, :]
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.einsum("fpd,fd->fp", diff, seg) / l2[:, None]
    t = np.clip(np.nan_to_num(t, nan=0.0, posinf=0.0, neginf=0.0), 0.0, 1.0)
    proj = starts[:, None, :] + t[:, :, None] * seg[:, None, :]
    return np.linalg.norm(points[None, :, :] - proj, axis=2)


def _as_array(rows: list[list[float]] | np.ndarray, width: int) -> np.ndarray:
    if isinstance(rows, np.ndarray):
        arr = rows.astype(np.float64, copy=True)
    elif rows:
        arr = np.asarray(rows, dtype=np.float64)
    else:
        arr = np.empty((0, width), dtype=np.float64)
    return arr.reshape((-1, width))


def _planet_row_to_list(row: np.ndarray) -> list[Any]:
    return [
        int(row[P_ID]),
        int(row[P_OWNER]),
        float(row[P_X]),
        float(row[P_Y]),
        float(row[P_RADIUS]),
        int(row[P_SHIPS]),
        int(row[P_PROD]),
    ]


def _fleet_row_to_list(row: np.ndarray) -> list[Any]:
    return [
        int(row[F_ID]),
        int(row[F_OWNER]),
        float(row[F_X]),
        float(row[F_Y]),
        float(row[F_ANGLE]),
        int(row[F_FROM]),
        int(row[F_SHIPS]),
    ]


class NumpyOrbitWarsEnv:
    """Single Orbit Wars environment with Kaggle and Gym-style APIs."""

    def __init__(
        self,
        num_players: int = 2,
        episode_steps: int = 500,
        ship_speed: float = 6.0,
        comet_speed: float = 4.0,
        random_seed: int | None = None,
    ) -> None:
        self.cfg = NumpyOrbitWarsConfig(
            num_players=num_players,
            episode_steps=episode_steps,
            ship_speed=ship_speed,
            comet_speed=comet_speed,
            random_seed=random_seed,
        )
        self.rng = random.Random(random_seed)
        self.steps: list[list[SimpleNamespace]] = []
        self.done = False
        self._initialized = False
        self._step = 0
        self.angular_velocity = 0.0
        self.planets = np.empty((0, 7), dtype=np.float64)
        self.initial_planets = np.empty((0, 7), dtype=np.float64)
        self.fleets = np.empty((0, 7), dtype=np.float64)
        self.next_fleet_id = 0
        self.comets: list[dict[str, Any]] = []
        self.comet_planet_ids: list[int] = []

    @classmethod
    def from_observation(
        cls,
        obs: Any,
        *,
        num_players: int = 2,
        episode_steps: int = 500,
        ship_speed: float = 6.0,
        comet_speed: float = 4.0,
    ) -> "NumpyOrbitWarsEnv":
        """Create an env from a visible observation.

        This is deterministic for future non-random transitions. It cannot
        recover Kaggle's hidden Python RNG state, so parity with an official
        env loaded this way is only expected until the next comet spawn.
        """
        env = cls(num_players, episode_steps, ship_speed, comet_speed)
        env.load_observation(obs)
        env.steps = [env._state([None] * num_players)]
        return env

    def reset(
        self, *, seed: int | None = None, num_agents: int | None = None
    ) -> list[dict[str, Any]]:
        if seed is not None:
            self.cfg.random_seed = seed
        self.rng = random.Random(self.cfg.random_seed)
        if num_agents is not None:
            self.cfg.num_players = num_agents
        self.done = False
        self._initialized = False
        self._step = 0
        self.angular_velocity = 0.0
        self.planets = np.empty((0, 7), dtype=np.float64)
        self.initial_planets = np.empty((0, 7), dtype=np.float64)
        self.fleets = np.empty((0, 7), dtype=np.float64)
        self.next_fleet_id = 0
        self.comets = []
        self.comet_planet_ids = []
        self.steps = [self._state([None] * self.cfg.num_players)]
        return self.steps[-1]

    def reset_gym(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        options = options or {}
        return self.reset(seed=seed, num_agents=options.get("num_agents")), {}

    def load_observation(self, obs: Any) -> None:
        get = obs.get if isinstance(obs, dict) else lambda k, d=None: getattr(obs, k, d)
        self._step = int(get("step", 0) or 0)
        self.angular_velocity = float(get("angular_velocity", 0.0) or 0.0)
        self.planets = _as_array([list(p) for p in (get("planets", []) or [])], 7)
        self.initial_planets = _as_array(
            [list(p) for p in (get("initial_planets", []) or [])], 7
        )
        self.fleets = _as_array([list(f) for f in (get("fleets", []) or [])], 7)
        self.next_fleet_id = int(get("next_fleet_id", 0) or 0)
        self.comet_planet_ids = [int(pid) for pid in (get("comet_planet_ids", []) or [])]
        self.comets = []
        for group in get("comets", []) or []:
            self.comets.append(
                {
                    "planet_ids": [int(pid) for pid in group["planet_ids"]],
                    "paths": [np.asarray(path, dtype=np.float64) for path in group["paths"]],
                    "path_index": int(group["path_index"]),
                }
            )
        self._initialized = len(self.planets) > 0
        self.done = False

    def step(self, actions: list[Any]) -> list[dict[str, Any]]:
        if self.done:
            return self.steps[-1]
        actions = actions if isinstance(actions, list) else []
        if len(actions) < self.cfg.num_players:
            actions = [*actions, *([None] * (self.cfg.num_players - len(actions)))]
        self._step += 1
        if not self._initialized:
            self._initialize()
            state = self._state(actions)
            self.steps.append(state)
            return state

        self._remove_expired_comets_before_launch()
        self._spawn_comets()
        planet_idx_by_id = {
            int(pid): idx for idx, pid in enumerate(self.planets[:, P_ID].astype(np.int64))
        }
        launch_rows: list[list[float]] = []
        for player_id in range(self.cfg.num_players):
            self._process_moves(
                player_id, actions[player_id], planet_idx_by_id, launch_rows
            )
        if launch_rows:
            launch_arr = np.asarray(launch_rows, dtype=np.float64)
            self.fleets = (
                launch_arr if len(self.fleets) == 0 else np.vstack([self.fleets, launch_arr])
            )
        self._produce()
        combat_lists = self._move_fleets()
        self._move_planets_and_sweep(combat_lists)
        self._resolve_combat(combat_lists)
        self._check_done()
        state = self._state(actions)
        self.steps.append(state)
        return state

    def step_gym(
        self, actions: list[Any]
    ) -> tuple[list[dict[str, Any]], list[float], bool, bool, dict[str, Any]]:
        state = self.step(actions)
        rewards = [float(s["reward"] or 0.0) for s in state]
        return state, rewards, self.done, False, {}

    def _state(self, actions: list[Any]) -> list[dict[str, Any]]:
        rewards = self._current_rewards()
        statuses = ["DONE" if self.done else "ACTIVE"] * self.cfg.num_players
        observation_base = self._observation_base()
        state = []
        for player in range(self.cfg.num_players):
            state.append(
                {
                    "action": actions[player] if player < len(actions) else None,
                    "reward": rewards[player],
                    "info": {},
                    "observation": self._observation(player, observation_base),
                    "status": statuses[player],
                }
            )
        return state

    def final_state(self) -> list[SimpleNamespace] | None:
        if not self.done:
            return None
        return [
            SimpleNamespace(
                reward=s["reward"],
                status=s["status"],
                action=s["action"],
                observation=s["observation"],
            )
            for s in self.steps[-1]
        ]

    def _observation_base(self) -> dict[str, Any]:
        return {
            "remainingOverageTime": 60,
            "step": self._step,
            "planets": [_planet_row_to_list(row) for row in self.planets],
            "fleets": [_fleet_row_to_list(row) for row in self.fleets],
            "angular_velocity": self.angular_velocity,
            "initial_planets": [_planet_row_to_list(row) for row in self.initial_planets],
            "next_fleet_id": self.next_fleet_id,
            "comets": self._comets_to_lists(),
            "comet_planet_ids": list(self.comet_planet_ids),
        }

    def _observation(self, player: int, base: dict[str, Any]) -> dict[str, Any]:
        obs = dict(base)
        obs["player"] = player
        return obs

    def _comets_to_lists(self) -> list[dict[str, Any]]:
        return [
            {
                "planet_ids": list(group["planet_ids"]),
                "paths": [path.tolist() for path in group["paths"]],
                "path_index": int(group["path_index"]),
            }
            for group in self.comets
        ]

    def _initialize(self) -> None:
        self.angular_velocity = self.rng.uniform(0.025, 0.05)
        planets = self._generate_planets()
        initial_planets = [p.copy() for p in planets]
        num_groups = len(planets) // 4
        if num_groups > 0:
            home_group = self.rng.randint(0, num_groups - 1)
            base = home_group * 4
            if self.cfg.num_players == 4:
                q1 = planets[base]
                orb_r = _distance((q1[P_X], q1[P_Y]), (CENTER, CENTER))
                if orb_r + q1[P_RADIUS] < ROTATION_RADIUS_LIMIT:
                    for group_id in range(num_groups):
                        gb = group_id * 4
                        gp = planets[gb]
                        g_orb = _distance((gp[P_X], gp[P_Y]), (CENTER, CENTER))
                        if (
                            g_orb + gp[P_RADIUS] < ROTATION_RADIUS_LIMIT
                            and abs((gp[P_X] - CENTER) - (gp[P_Y] - CENTER)) < 0.01
                        ):
                            base = gb
                            break
            if self.cfg.num_players == 2:
                planets[base][P_OWNER] = 0
                planets[base][P_SHIPS] = 10
                planets[base + 3][P_OWNER] = 1
                planets[base + 3][P_SHIPS] = 10
            elif self.cfg.num_players == 4:
                for j in range(4):
                    planets[base + j][P_OWNER] = j
                    planets[base + j][P_SHIPS] = 10
        self.planets = _as_array(planets, 7)
        self.initial_planets = _as_array(initial_planets, 7)
        self.fleets = np.empty((0, 7), dtype=np.float64)
        self.next_fleet_id = 0
        self.comets = []
        self.comet_planet_ids = []
        self._initialized = True

    def _generate_planets(self) -> list[list[float]]:
        planets: list[list[float]] = []
        num_q1 = self.rng.randint(MIN_PLANET_GROUPS, MAX_PLANET_GROUPS)
        id_counter = 0
        static_groups = 0
        for _ in range(5000):
            if static_groups >= MIN_STATIC_GROUPS:
                break
            prod = self.rng.randint(1, 5)
            r = 1 + math.log(prod)
            angle = self.rng.uniform(0, math.pi / 2)
            min_orbital = ROTATION_RADIUS_LIMIT - r
            max_orbital = (BOARD_SIZE - CENTER - r) / max(math.cos(angle), math.sin(angle))
            if min_orbital > max_orbital:
                continue
            orbital_r = self.rng.uniform(min_orbital, max_orbital)
            x = CENTER + orbital_r * math.cos(angle)
            y = CENTER + orbital_r * math.sin(angle)
            if x + r > BOARD_SIZE or x - r < 0 or y + r > BOARD_SIZE or y - r < 0:
                continue
            if (BOARD_SIZE - x) - r < 0 or (BOARD_SIZE - y) - r < 0:
                continue
            if (x - CENTER) < r + 5 or (y - CENTER) < r + 5:
                continue
            ships = min(self.rng.randint(5, 99), self.rng.randint(5, 99))
            temp = self._symmetric_group(id_counter, x, y, r, ships, prod)
            if self._valid_against_existing(temp, planets):
                planets.extend(temp)
                id_counter += 4
                static_groups += 1

        for _ in range(1000):
            prod = self.rng.randint(1, 5)
            r = 1 + math.log(prod)
            min_orbital = SUN_RADIUS + r + 10
            max_orbital = ROTATION_RADIUS_LIMIT - r
            if min_orbital >= max_orbital:
                continue
            orbital_r = self.rng.uniform(min_orbital, max_orbital)
            x = CENTER + orbital_r * math.cos(math.pi / 4)
            y = CENTER + orbital_r * math.sin(math.pi / 4)
            ships = min(self.rng.randint(5, 99), self.rng.randint(5, 99))
            temp = self._symmetric_group(id_counter, x, y, r, ships, prod)
            if self._valid_orbit_group(temp, planets, allow_same_mode=True):
                planets.extend(temp)
                id_counter += 4
                break

        attempts = 0
        has_orbiting = False
        while len(planets) < num_q1 * 4 or (not has_orbiting and attempts < 5000):
            attempts += 1
            if attempts >= 5000:
                break
            prod = self.rng.randint(1, 5)
            r = 1 + math.log(prod)
            x = self.rng.uniform(CENTER + 15, BOARD_SIZE - r - 5)
            y = self.rng.uniform(CENTER + 15, BOARD_SIZE - r - 5)
            orbital_radius = _distance((x, y), (CENTER, CENTER))
            if orbital_radius < SUN_RADIUS + r + 10:
                continue
            if orbital_radius + r >= ROTATION_RADIUS_LIMIT:
                if x + r > BOARD_SIZE or x - r < 0 or y + r > BOARD_SIZE or y - r < 0:
                    continue
            ships = self.rng.randint(5, 30)
            temp = self._symmetric_group(id_counter, x, y, r, ships, prod)
            if self._valid_orbit_group(temp, planets, allow_same_mode=False):
                if orbital_radius + r < ROTATION_RADIUS_LIMIT:
                    has_orbiting = True
                planets.extend(temp)
                id_counter += 4
        return planets

    @staticmethod
    def _symmetric_group(
        id_counter: int, x: float, y: float, r: float, ships: int, prod: int
    ) -> list[list[float]]:
        return [
            [id_counter, -1, x, y, r, ships, prod],
            [id_counter + 1, -1, BOARD_SIZE - x, y, r, ships, prod],
            [id_counter + 2, -1, x, BOARD_SIZE - y, r, ships, prod],
            [id_counter + 3, -1, BOARD_SIZE - x, BOARD_SIZE - y, r, ships, prod],
        ]

    @staticmethod
    def _valid_against_existing(
        temp: list[list[float]], planets: list[list[float]]
    ) -> bool:
        for tp in temp:
            for p in planets:
                if _distance((p[P_X], p[P_Y]), (tp[P_X], tp[P_Y])) < (
                    p[P_RADIUS] + tp[P_RADIUS] + PLANET_CLEARANCE
                ):
                    return False
        return True

    @staticmethod
    def _valid_orbit_group(
        temp: list[list[float]], planets: list[list[float]], *, allow_same_mode: bool
    ) -> bool:
        for tp in temp:
            tp_orb = _distance((tp[P_X], tp[P_Y]), (CENTER, CENTER))
            tp_rot = tp_orb + tp[P_RADIUS] < ROTATION_RADIUS_LIMIT
            for p in planets:
                p_orb = _distance((p[P_X], p[P_Y]), (CENTER, CENTER))
                p_rot = p_orb + p[P_RADIUS] < ROTATION_RADIUS_LIMIT
                if _distance((p[P_X], p[P_Y]), (tp[P_X], tp[P_Y])) < (
                    p[P_RADIUS] + tp[P_RADIUS] + PLANET_CLEARANCE
                ):
                    return False
                if (allow_same_mode or tp_rot != p_rot) and p_orb + p[P_RADIUS] >= ROTATION_RADIUS_LIMIT:
                    if abs(tp_orb - p_orb) < tp[P_RADIUS] + p[P_RADIUS] + PLANET_CLEARANCE:
                        return False
                elif not allow_same_mode and tp_rot != p_rot:
                    if abs(tp_orb - p_orb) < tp[P_RADIUS] + p[P_RADIUS] + PLANET_CLEARANCE:
                        return False
        return True

    def _remove_expired_comets_before_launch(self) -> None:
        expired: list[int] = []
        for group in self.comets:
            idx = int(group["path_index"])
            for i, pid in enumerate(group["planet_ids"]):
                if idx >= len(group["paths"][i]):
                    expired.append(pid)
        if expired:
            self._remove_comet_planets(expired)

    def _spawn_comets(self) -> None:
        step = self._step - 1
        if (step + 1) not in COMET_SPAWN_STEPS:
            return
        paths = self._generate_comet_paths(step + 1)
        if not paths:
            return
        next_id = int(np.max(self.planets[:, P_ID])) + 1 if len(self.planets) else 0
        comet_ships = min(
            self.rng.randint(1, 99),
            self.rng.randint(1, 99),
            self.rng.randint(1, 99),
            self.rng.randint(1, 99),
        )
        group = {"planet_ids": [], "paths": paths, "path_index": -1}
        rows = []
        for i, _path in enumerate(paths):
            pid = next_id + i
            group["planet_ids"].append(pid)
            self.comet_planet_ids.append(pid)
            rows.append([pid, -1, -99, -99, COMET_RADIUS, comet_ships, COMET_PRODUCTION])
        self.comets.append(group)
        self.planets = np.vstack([self.planets, np.asarray(rows, dtype=np.float64)])
        self.initial_planets = np.vstack(
            [self.initial_planets, np.asarray(rows, dtype=np.float64)]
        )

    def _generate_comet_paths(self, spawn_step: int) -> list[np.ndarray] | None:
        comet_ids = set(self.comet_planet_ids)
        for _ in range(300):
            e = self.rng.uniform(0.75, 0.93)
            a = self.rng.uniform(60, 150)
            if a * (1 - e) < SUN_RADIUS + COMET_RADIUS:
                continue
            b = a * math.sqrt(1 - e**2)
            c_val = a * e
            phi = self.rng.uniform(math.pi / 6, math.pi / 3)
            ex = c_val + a * COMET_COS_T
            ey = b * COMET_SIN_T
            cos_phi = math.cos(phi)
            sin_phi = math.sin(phi)
            dense = np.column_stack(
                (
                    CENTER + ex * cos_phi - ey * sin_phi,
                    CENTER + ex * sin_phi + ey * cos_phi,
                )
            )
            segment_lengths = np.linalg.norm(np.diff(dense, axis=0), axis=1)
            cumulative = np.cumsum(segment_lengths)
            targets = np.arange(
                self.cfg.comet_speed,
                cumulative[-1] + self.cfg.comet_speed,
                self.cfg.comet_speed,
                dtype=np.float64,
            )
            sample_idx = np.searchsorted(cumulative, targets, side="left") + 1
            sample_idx = sample_idx[sample_idx < len(dense)]
            path = np.vstack([dense[0], dense[sample_idx]])
            on_board = np.nonzero(
                (path[:, 0] >= 0)
                & (path[:, 0] <= BOARD_SIZE)
                & (path[:, 1] >= 0)
                & (path[:, 1] <= BOARD_SIZE)
            )[0]
            if len(on_board) == 0:
                continue
            visible_arr = path[int(on_board[0]) : int(on_board[-1]) + 1]
            if not (5 <= len(visible_arr) <= 40):
                continue
            paths = [
                visible_arr.copy(),
                np.column_stack((BOARD_SIZE - visible_arr[:, 0], visible_arr[:, 1])),
                np.column_stack((visible_arr[:, 0], BOARD_SIZE - visible_arr[:, 1])),
                BOARD_SIZE - visible_arr,
            ]
            if self._comet_paths_valid(visible_arr, paths, comet_ids, spawn_step):
                return paths
        return None

    def _comet_paths_valid(
        self,
        visible: np.ndarray,
        paths: list[np.ndarray],
        comet_ids: set[int],
        spawn_step: int,
    ) -> bool:
        if np.any(np.hypot(visible[:, 0] - CENTER, visible[:, 1] - CENTER) < SUN_RADIUS + COMET_RADIUS):
            return False

        planets = self.initial_planets
        if len(planets) == 0:
            return True
        if comet_ids:
            planets = planets[
                ~np.isin(planets[:, P_ID].astype(np.int64), list(comet_ids))
            ]
            if len(planets) == 0:
                return True

        sym_pts = np.stack(paths, axis=1)  # [path_step, symmetry, xy]
        dx = planets[:, P_X] - CENTER
        dy = planets[:, P_Y] - CENTER
        orbital_r = np.hypot(dx, dy)
        rotating = orbital_r + planets[:, P_RADIUS] < ROTATION_RADIUS_LIMIT

        static = planets[~rotating]
        if len(static):
            delta = sym_pts[:, :, None, :] - static[None, None, :, [P_X, P_Y]]
            dists = np.linalg.norm(delta, axis=3)
            thresholds = static[:, P_RADIUS] + COMET_RADIUS + 0.5
            if np.any(dists < thresholds[None, None, :]):
                return False

        orbiting = planets[rotating]
        if len(orbiting):
            odx = orbiting[:, P_X] - CENTER
            ody = orbiting[:, P_Y] - CENTER
            orbit_r = np.hypot(odx, ody)
            init_angle = np.arctan2(ody, odx)
            game_steps = np.arange(
                spawn_step - 1,
                spawn_step - 1 + len(visible),
                dtype=np.float64,
            )
            angles = init_angle[None, :] + self.angular_velocity * game_steps[:, None]
            orbit_pos = np.empty((len(visible), len(orbiting), 2), dtype=np.float64)
            orbit_pos[:, :, 0] = CENTER + orbit_r[None, :] * np.cos(angles)
            orbit_pos[:, :, 1] = CENTER + orbit_r[None, :] * np.sin(angles)
            delta = sym_pts[:, :, None, :] - orbit_pos[:, None, :, :]
            dists = np.linalg.norm(delta, axis=3)
            thresholds = orbiting[:, P_RADIUS] + COMET_RADIUS
            if np.any(dists < thresholds[None, None, :]):
                return False
        return True

    def _process_moves(
        self,
        player_id: int,
        action: Any,
        planet_idx_by_id: dict[int, int],
        launch_rows: list[list[float]],
    ) -> None:
        if not action or not isinstance(action, list):
            return
        for move in action:
            try:
                if len(move) != 3:
                    continue
                from_id, angle, ships = move
                ships = int(ships)
                angle = float(angle)
            except (TypeError, ValueError):
                continue
            idx = planet_idx_by_id.get(int(from_id))
            if idx is None:
                continue
            from_planet = self.planets[idx]
            if int(from_planet[P_OWNER]) != player_id:
                continue
            if from_planet[P_SHIPS] >= ships and ships > 0:
                self.planets[idx, P_SHIPS] -= ships
                start_x = from_planet[P_X] + math.cos(angle) * (from_planet[P_RADIUS] + 0.1)
                start_y = from_planet[P_Y] + math.sin(angle) * (from_planet[P_RADIUS] + 0.1)
                launch_rows.append(
                    [
                        self.next_fleet_id,
                        player_id,
                        start_x,
                        start_y,
                        angle,
                        int(from_id),
                        ships,
                    ]
                )
                self.next_fleet_id += 1

    def _produce(self) -> None:
        owned = self.planets[:, P_OWNER] != -1
        self.planets[owned, P_SHIPS] += self.planets[owned, P_PROD]

    def _move_fleets(self) -> dict[int, list[np.ndarray]]:
        combat_lists: defaultdict[int, list[np.ndarray]] = defaultdict(list)
        if len(self.fleets) == 0:
            return combat_lists
        old = self.fleets[:, [F_X, F_Y]].copy()
        ships = self.fleets[:, F_SHIPS]
        speeds = 1.0 + (self.cfg.ship_speed - 1.0) * (np.log(ships) / math.log(1000.0)) ** 1.5
        speeds = np.minimum(speeds, self.cfg.ship_speed)
        self.fleets[:, F_X] += np.cos(self.fleets[:, F_ANGLE]) * speeds
        self.fleets[:, F_Y] += np.sin(self.fleets[:, F_ANGLE]) * speeds
        new = self.fleets[:, [F_X, F_Y]].copy()
        remove = np.zeros(len(self.fleets), dtype=bool)
        remove |= (
            (self.fleets[:, F_X] < 0)
            | (self.fleets[:, F_X] > BOARD_SIZE)
            | (self.fleets[:, F_Y] < 0)
            | (self.fleets[:, F_Y] > BOARD_SIZE)
        )
        active = ~remove
        if np.any(active):
            center = np.repeat(np.asarray([[CENTER, CENTER]], dtype=np.float64), int(np.sum(active)), axis=0)
            sun_dist = _points_to_segments_distance(center, old[active], new[active])
            active_idx = np.nonzero(active)[0]
            remove[active_idx[sun_dist < SUN_RADIUS]] = True
        active = ~remove
        if np.any(active) and len(self.planets):
            active_idx = np.nonzero(active)[0]
            dists = _all_point_segment_distances(
                self.planets[:, [P_X, P_Y]], old[active], new[active]
            )
            hits = dists < self.planets[:, P_RADIUS][None, :]
            has_hit = hits.any(axis=1)
            first_hit = np.argmax(hits, axis=1)
            for local_idx, planet_idx in zip(np.nonzero(has_hit)[0], first_hit[has_hit], strict=False):
                fleet_idx = int(active_idx[local_idx])
                pid = int(self.planets[int(planet_idx), P_ID])
                combat_lists[pid].append(self.fleets[fleet_idx].copy())
                remove[fleet_idx] = True
        self._pending_remove = remove
        return combat_lists

    def _move_planets_and_sweep(self, combat_lists: dict[int, list[np.ndarray]]) -> None:
        comet_set = set(self.comet_planet_ids)
        initial_by_id = {int(p[P_ID]): p.copy() for p in self.initial_planets}
        remove = getattr(self, "_pending_remove", np.zeros(len(self.fleets), dtype=bool))
        track_sweep = len(self.fleets) > 0 and np.any(~remove)
        moving: list[tuple[int, float, float, float, float, float]] = []

        for idx in range(len(self.planets)):
            planet = self.planets[idx]
            if int(planet[P_ID]) in comet_set:
                continue
            initial = initial_by_id.get(int(planet[P_ID]))
            if initial is None:
                continue
            dx = initial[P_X] - CENTER
            dy = initial[P_Y] - CENTER
            r = math.sqrt(dx**2 + dy**2)
            old_pos = (float(planet[P_X]), float(planet[P_Y]))
            if r + planet[P_RADIUS] < ROTATION_RADIUS_LIMIT:
                angle = math.atan2(dy, dx) + self.angular_velocity * (self._step - 1)
                self.planets[idx, P_X] = CENTER + r * math.cos(angle)
                self.planets[idx, P_Y] = CENTER + r * math.sin(angle)
            new_x = float(self.planets[idx, P_X])
            new_y = float(self.planets[idx, P_Y])
            if track_sweep and old_pos != (new_x, new_y):
                moving.append((int(planet[P_ID]), float(planet[P_RADIUS]), old_pos[0], old_pos[1], new_x, new_y))

        expired: list[int] = []
        for group in self.comets:
            group["path_index"] += 1
            idx = int(group["path_index"])
            for i, pid in enumerate(list(group["planet_ids"])):
                matches = np.nonzero(self.planets[:, P_ID].astype(np.int64) == int(pid))[0]
                if len(matches) == 0:
                    continue
                pidx = int(matches[0])
                path = group["paths"][i]
                if idx >= len(path):
                    expired.append(int(pid))
                else:
                    old_pos = (float(self.planets[pidx, P_X]), float(self.planets[pidx, P_Y]))
                    self.planets[pidx, P_X] = path[idx, 0]
                    self.planets[pidx, P_Y] = path[idx, 1]
                    if track_sweep and old_pos[0] >= 0:
                        moving.append(
                            (
                                int(pid),
                                float(self.planets[pidx, P_RADIUS]),
                                old_pos[0],
                                old_pos[1],
                                float(path[idx, 0]),
                                float(path[idx, 1]),
                            )
                        )
        self._sweep_moving_planets(moving, combat_lists, remove)
        if expired:
            self._remove_comet_planets(expired)
        if len(self.fleets):
            self.fleets = self.fleets[~remove].copy()
        self._pending_remove = np.zeros(len(self.fleets), dtype=bool)

    def _sweep_moving_planets(
        self,
        moving: list[tuple[int, float, float, float, float, float]],
        combat_lists: dict[int, list[np.ndarray]],
        remove: np.ndarray,
    ) -> None:
        if not moving or len(self.fleets) == 0:
            return
        candidates = np.nonzero(~remove)[0]
        if len(candidates) == 0:
            return
        arr = np.asarray(moving, dtype=np.float64)
        pids = arr[:, 0].astype(np.int64)
        radii = arr[:, 1]
        starts = arr[:, 2:4]
        ends = arr[:, 4:6]
        pts = self.fleets[candidates][:, [F_X, F_Y]]
        dists = _all_point_segment_distances(pts, starts, ends)
        hits = dists < radii[:, None]
        hit_fleets = np.nonzero(hits.any(axis=0))[0]
        if len(hit_fleets) == 0:
            return
        first_planet = np.argmax(hits[:, hit_fleets], axis=0)
        for local_fleet, moving_idx in zip(hit_fleets, first_planet, strict=False):
            fleet_idx = int(candidates[int(local_fleet)])
            combat_lists[int(pids[int(moving_idx)])].append(self.fleets[fleet_idx].copy())
            remove[fleet_idx] = True

    def _remove_comet_planets(self, pids: list[int]) -> None:
        expired = set(int(pid) for pid in pids)
        if len(self.planets):
            self.planets = self.planets[
                ~np.isin(self.planets[:, P_ID].astype(np.int64), list(expired))
            ].copy()
        if len(self.initial_planets):
            self.initial_planets = self.initial_planets[
                ~np.isin(self.initial_planets[:, P_ID].astype(np.int64), list(expired))
            ].copy()
        self.comet_planet_ids = [pid for pid in self.comet_planet_ids if pid not in expired]
        for group in self.comets:
            keep = [pid not in expired for pid in group["planet_ids"]]
            group["planet_ids"] = [pid for pid, ok in zip(group["planet_ids"], keep, strict=False) if ok]
            group["paths"] = [path for path, ok in zip(group["paths"], keep, strict=False) if ok]
        self.comets = [group for group in self.comets if group["planet_ids"]]

    def _resolve_combat(self, combat_lists: dict[int, list[np.ndarray]]) -> None:
        planet_idx_by_id = {
            int(pid): idx for idx, pid in enumerate(self.planets[:, P_ID].astype(np.int64))
        }
        for pid, planet_fleets in combat_lists.items():
            if not planet_fleets:
                continue
            pidx = planet_idx_by_id.get(int(pid))
            if pidx is None:
                continue
            player_ships: dict[int, int] = {}
            for fleet in planet_fleets:
                owner = int(fleet[F_OWNER])
                player_ships[owner] = player_ships.get(owner, 0) + int(fleet[F_SHIPS])
            sorted_players = sorted(player_ships.items(), key=lambda item: item[1], reverse=True)
            top_player, top_ships = sorted_players[0]
            if len(sorted_players) > 1:
                survivor_ships = top_ships - sorted_players[1][1]
                if sorted_players[0][1] == sorted_players[1][1]:
                    survivor_ships = 0
                survivor_owner = top_player if survivor_ships > 0 else -1
            else:
                survivor_owner = top_player
                survivor_ships = top_ships
            if survivor_ships > 0:
                if int(self.planets[pidx, P_OWNER]) == survivor_owner:
                    self.planets[pidx, P_SHIPS] += survivor_ships
                else:
                    self.planets[pidx, P_SHIPS] -= survivor_ships
                    if self.planets[pidx, P_SHIPS] < 0:
                        self.planets[pidx, P_OWNER] = survivor_owner
                        self.planets[pidx, P_SHIPS] = abs(self.planets[pidx, P_SHIPS])

    def _check_done(self) -> None:
        terminated = (self._step - 1) >= self.cfg.episode_steps - 2
        alive = set(int(o) for o in self.planets[:, P_OWNER] if int(o) != -1)
        alive.update(int(o) for o in self.fleets[:, F_OWNER])
        if len(alive) <= 1:
            terminated = True
        self.done = terminated

    def _current_rewards(self) -> list[int]:
        if not self.done:
            return [0] * self.cfg.num_players
        scores = [0] * self.cfg.num_players
        for p in self.planets:
            owner = int(p[P_OWNER])
            if owner != -1:
                scores[owner] += int(p[P_SHIPS])
        for f in self.fleets:
            scores[int(f[F_OWNER])] += int(f[F_SHIPS])
        max_score = max(scores)
        return [1 if score == max_score and max_score > 0 else -1 for score in scores]


class NumpyVecEnv:
    """Adaptive in-process vector env with the same subset protocol as `VecEnv`.

    Map generation and comet path generation remain delegated to
    `NumpyOrbitWarsEnv` because they are per-game stochastic rejection
    samplers. Idle turns are updated in padded batch arrays; turns with
    launches or live fleets use the scalar NumPy env as the correctness oracle.
    """

    def __init__(
        self,
        num_envs: int,
        num_players: int,
        episode_steps: int,
        ship_speed: float,
        comet_speed: float = 4.0,
        random_seed: int | None = None,
        replay_env_idx: int | None = None,
    ) -> None:
        self.num_envs = num_envs
        self.replay_env_idx = replay_env_idx
        self.last_replay_html: str | None = None
        self.num_players = num_players
        self.episode_steps = episode_steps
        self.ship_speed = ship_speed
        self.comet_speed = comet_speed
        self.envs = [
            NumpyOrbitWarsEnv(
                num_players=num_players,
                episode_steps=episode_steps,
                ship_speed=ship_speed,
                comet_speed=comet_speed,
                random_seed=None if random_seed is None else random_seed + i,
            )
            for i in range(num_envs)
        ]
        self.planet_cap = 96
        self.fleet_cap = 256
        self.planets = np.empty((num_envs, self.planet_cap, 7), dtype=np.float64)
        self.initial_planets = np.empty((num_envs, self.planet_cap, 7), dtype=np.float64)
        self.planet_mask = np.zeros((num_envs, self.planet_cap), dtype=bool)
        self.initial_planet_mask = np.zeros((num_envs, self.planet_cap), dtype=bool)
        self.fleets = np.empty((num_envs, self.fleet_cap, 7), dtype=np.float64)
        self.fleet_mask = np.zeros((num_envs, self.fleet_cap), dtype=bool)
        self.done = np.zeros(num_envs, dtype=bool)
        self.initialized = np.zeros(num_envs, dtype=bool)
        self.step_count = np.zeros(num_envs, dtype=np.int64)
        self.angular_velocity = np.zeros(num_envs, dtype=np.float64)
        self.next_fleet_id = np.zeros(num_envs, dtype=np.int64)
        self.last_states: list[list[dict[str, Any]]] = []

    def reset(self) -> list[list[dict[str, Any]]]:
        self.last_replay_html = None
        states = [env.reset() for env in self.envs]
        self.planet_mask.fill(False)
        self.initial_planet_mask.fill(False)
        self.fleet_mask.fill(False)
        self.done.fill(False)
        self.initialized.fill(False)
        self.step_count.fill(0)
        self.angular_velocity.fill(0.0)
        self.next_fleet_id.fill(0)
        for i, env in enumerate(self.envs):
            self._store_env(i, env)
        self.last_states = states
        return states

    def step_subset(
        self, indices: list[int], actions: list[Any]
    ) -> dict[int, tuple[list[dict[str, Any]], bool, list[SimpleNamespace] | None]]:
        assert len(indices) == len(actions), (len(indices), len(actions))
        out: dict[int, tuple[list[dict[str, Any]], bool, list[SimpleNamespace] | None]] = {}
        active: list[int] = []
        active_actions: dict[int, list[Any]] = {}
        for idx, action in zip(indices, actions, strict=True):
            if self.done[idx]:
                state = self.last_states[idx]
                out[idx] = (state, True, self._final_state_from_state(state))
                continue
            normalized = self._normalize_actions(action)
            if not self.initialized[idx]:
                env = self.envs[idx]
                env.step(normalized)
                self._store_env(idx, env)
                state = self._state(idx, normalized)
                self.last_states[idx] = state
                out[idx] = (state, bool(self.done[idx]), None)
                continue
            if self._needs_scalar_step(idx, normalized):
                env = self._write_env(idx)
                state = env.step(normalized)
                self._store_env(idx, env)
                self.last_states[idx] = state
                out[idx] = (
                    state,
                    bool(self.done[idx]),
                    env.final_state() if self.done[idx] else None,
                )
                continue
            self.step_count[idx] += 1
            active.append(idx)
            active_actions[idx] = normalized

        if active:
            self._prepare_comets(active)
            self._produce_batch(active)
            self._move_planets_batch(active)
            self._check_done_batch(active)
            for idx in active:
                state = self._state(idx, active_actions[idx])
                self.last_states[idx] = state
                out[idx] = (
                    state,
                    bool(self.done[idx]),
                    self._final_state_from_state(state) if self.done[idx] else None,
                )
        return out

    def _needs_scalar_step(self, idx: int, actions: list[Any]) -> bool:
        if self.fleet_mask[idx].any():
            return True
        return any(action and isinstance(action, list) for action in actions)

    def set_recording(self, enabled: bool) -> None:
        if not enabled:
            self.last_replay_html = None

    def close(self) -> None:
        return

    def __enter__(self) -> "NumpyVecEnv":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    # ----- storage ---------------------------------------------------------

    def _ensure_planet_capacity(self, needed: int) -> None:
        if needed <= self.planet_cap:
            return
        new_cap = max(needed, self.planet_cap * 2)
        planets = np.empty((self.num_envs, new_cap, 7), dtype=np.float64)
        initial = np.empty((self.num_envs, new_cap, 7), dtype=np.float64)
        pm = np.zeros((self.num_envs, new_cap), dtype=bool)
        im = np.zeros((self.num_envs, new_cap), dtype=bool)
        planets[:, : self.planet_cap] = self.planets
        initial[:, : self.planet_cap] = self.initial_planets
        pm[:, : self.planet_cap] = self.planet_mask
        im[:, : self.planet_cap] = self.initial_planet_mask
        self.planet_cap = new_cap
        self.planets = planets
        self.initial_planets = initial
        self.planet_mask = pm
        self.initial_planet_mask = im

    def _ensure_fleet_capacity(self, needed: int) -> None:
        if needed <= self.fleet_cap:
            return
        new_cap = max(needed, self.fleet_cap * 2)
        fleets = np.empty((self.num_envs, new_cap, 7), dtype=np.float64)
        fm = np.zeros((self.num_envs, new_cap), dtype=bool)
        fleets[:, : self.fleet_cap] = self.fleets
        fm[:, : self.fleet_cap] = self.fleet_mask
        self.fleet_cap = new_cap
        self.fleets = fleets
        self.fleet_mask = fm

    def _store_env(self, idx: int, env: NumpyOrbitWarsEnv) -> None:
        self._ensure_planet_capacity(max(len(env.planets), len(env.initial_planets)))
        self._ensure_fleet_capacity(max(1, len(env.fleets)))
        self.planet_mask[idx].fill(False)
        self.initial_planet_mask[idx].fill(False)
        self.fleet_mask[idx].fill(False)
        if len(env.planets):
            self.planets[idx, : len(env.planets)] = env.planets
            self.planet_mask[idx, : len(env.planets)] = True
        if len(env.initial_planets):
            self.initial_planets[idx, : len(env.initial_planets)] = env.initial_planets
            self.initial_planet_mask[idx, : len(env.initial_planets)] = True
        if len(env.fleets):
            self.fleets[idx, : len(env.fleets)] = env.fleets
            self.fleet_mask[idx, : len(env.fleets)] = True
        self.done[idx] = env.done
        self.initialized[idx] = env._initialized
        self.step_count[idx] = env._step
        self.angular_velocity[idx] = env.angular_velocity
        self.next_fleet_id[idx] = env.next_fleet_id

    def _write_env(self, idx: int) -> NumpyOrbitWarsEnv:
        env = self.envs[idx]
        env.planets = self.planets[idx, self.planet_mask[idx]].copy()
        env.initial_planets = self.initial_planets[
            idx, self.initial_planet_mask[idx]
        ].copy()
        env.fleets = self.fleets[idx, self.fleet_mask[idx]].copy()
        env.done = bool(self.done[idx])
        env._initialized = bool(self.initialized[idx])
        env._step = int(self.step_count[idx])
        env.angular_velocity = float(self.angular_velocity[idx])
        env.next_fleet_id = int(self.next_fleet_id[idx])
        return env

    def _normalize_actions(self, action: Any) -> list[Any]:
        actions = action if isinstance(action, list) else []
        if len(actions) < self.num_players:
            actions = [*actions, *([None] * (self.num_players - len(actions)))]
        return actions

    # ----- public state materialization ------------------------------------

    def _state(self, idx: int, actions: list[Any]) -> list[dict[str, Any]]:
        rewards = self._current_rewards(idx)
        status = "DONE" if self.done[idx] else "ACTIVE"
        base = self._observation_base(idx)
        return [
            {
                "action": actions[player] if player < len(actions) else None,
                "reward": rewards[player],
                "info": {},
                "observation": self._observation(idx, player, base),
                "status": status,
            }
            for player in range(self.num_players)
        ]

    def _observation_base(self, idx: int) -> dict[str, Any]:
        env = self.envs[idx]
        return {
            "remainingOverageTime": 60,
            "step": int(self.step_count[idx]),
            "planets": [
                _planet_row_to_list(row)
                for row in self.planets[idx, self.planet_mask[idx]]
            ],
            "fleets": [
                _fleet_row_to_list(row)
                for row in self.fleets[idx, self.fleet_mask[idx]]
            ],
            "angular_velocity": float(self.angular_velocity[idx]),
            "initial_planets": [
                _planet_row_to_list(row)
                for row in self.initial_planets[idx, self.initial_planet_mask[idx]]
            ],
            "next_fleet_id": int(self.next_fleet_id[idx]),
            "comets": [
                {
                    "planet_ids": list(group["planet_ids"]),
                    "paths": [path.tolist() for path in group["paths"]],
                    "path_index": int(group["path_index"]),
                }
                for group in env.comets
            ],
            "comet_planet_ids": list(env.comet_planet_ids),
        }

    def _observation(self, idx: int, player: int, base: dict[str, Any]) -> dict[str, Any]:
        obs = dict(base)
        obs["player"] = player
        return obs

    @staticmethod
    def _final_state_from_state(state: list[dict[str, Any]]) -> list[SimpleNamespace]:
        return [
            SimpleNamespace(
                reward=s["reward"],
                status=s["status"],
                action=s["action"],
                observation=s["observation"],
            )
            for s in state
        ]

    # ----- batched simulation ----------------------------------------------

    def _prepare_comets(self, active: list[int]) -> None:
        for idx in active:
            env = self.envs[idx]
            spawn_step = int(self.step_count[idx])
            expired: list[int] = []
            for group in env.comets:
                path_idx = int(group["path_index"])
                for i, pid in enumerate(group["planet_ids"]):
                    if path_idx >= len(group["paths"][i]):
                        expired.append(int(pid))
            if expired:
                self._remove_comet_planets_batch(idx, expired)
            if spawn_step in COMET_SPAWN_STEPS:
                self._write_env(idx)
                env._spawn_comets()
                self._store_env(idx, env)

    def _produce_batch(self, active: list[int]) -> None:
        env_idx = np.asarray(active, dtype=np.int64)
        owned = self.planet_mask[env_idx] & (self.planets[env_idx, :, P_OWNER] != -1)
        self.planets[env_idx, :, P_SHIPS] += owned * self.planets[env_idx, :, P_PROD]

    def _move_planets_batch(self, active: list[int]) -> None:
        for idx in active:
            comet_ids = set(self.envs[idx].comet_planet_ids)
            for pslot in np.nonzero(self.planet_mask[idx])[0]:
                pid = int(self.planets[idx, pslot, P_ID])
                if pid in comet_ids:
                    continue
                if not self.initial_planet_mask[idx, pslot]:
                    continue
                init = self.initial_planets[idx, pslot]
                if int(init[P_ID]) != pid:
                    matches = np.nonzero(
                        self.initial_planet_mask[idx]
                        & (
                            self.initial_planets[idx, :, P_ID].astype(np.int64)
                            == pid
                        )
                    )[0]
                    if len(matches) == 0:
                        continue
                    init = self.initial_planets[idx, int(matches[0])]
                dx = init[P_X] - CENTER
                dy = init[P_Y] - CENTER
                radius = math.sqrt(dx**2 + dy**2)
                if radius + self.planets[idx, pslot, P_RADIUS] < ROTATION_RADIUS_LIMIT:
                    angle = math.atan2(dy, dx) + self.angular_velocity[idx] * (
                        self.step_count[idx] - 1
                    )
                    self.planets[idx, pslot, P_X] = CENTER + radius * math.cos(angle)
                    self.planets[idx, pslot, P_Y] = CENTER + radius * math.sin(angle)
            self._move_comets_for_env(idx)

    def _move_comets_for_env(self, idx: int) -> None:
        env = self.envs[idx]
        expired: list[int] = []
        ids = np.zeros(self.planet_cap, dtype=np.int64)
        ids[self.planet_mask[idx]] = self.planets[
            idx, self.planet_mask[idx], P_ID
        ].astype(np.int64)
        for group in env.comets:
            group["path_index"] += 1
            path_idx = int(group["path_index"])
            for i, pid in enumerate(list(group["planet_ids"])):
                matches = np.nonzero(self.planet_mask[idx] & (ids == int(pid)))[0]
                if len(matches) == 0:
                    continue
                pslot = int(matches[0])
                path = group["paths"][i]
                if path_idx >= len(path):
                    expired.append(int(pid))
                    continue
                self.planets[idx, pslot, P_X] = path[path_idx, 0]
                self.planets[idx, pslot, P_Y] = path[path_idx, 1]
        if expired:
            self._remove_comet_planets_batch(idx, expired)

    def _remove_comet_planets_batch(self, idx: int, pids: list[int]) -> None:
        expired = set(int(pid) for pid in pids)
        ids = np.zeros(self.planet_cap, dtype=np.int64)
        ids[self.planet_mask[idx]] = self.planets[
            idx, self.planet_mask[idx], P_ID
        ].astype(np.int64)
        initial_ids = np.zeros(self.planet_cap, dtype=np.int64)
        initial_ids[self.initial_planet_mask[idx]] = self.initial_planets[
            idx, self.initial_planet_mask[idx], P_ID
        ].astype(np.int64)
        self.planet_mask[idx] &= ~np.isin(ids, list(expired))
        self.initial_planet_mask[idx] &= ~np.isin(initial_ids, list(expired))
        env = self.envs[idx]
        env.comet_planet_ids = [
            pid for pid in env.comet_planet_ids if pid not in expired
        ]
        for group in env.comets:
            keep = [pid not in expired for pid in group["planet_ids"]]
            group["planet_ids"] = [
                pid for pid, ok in zip(group["planet_ids"], keep, strict=False) if ok
            ]
            group["paths"] = [
                path for path, ok in zip(group["paths"], keep, strict=False) if ok
            ]
        env.comets = [group for group in env.comets if group["planet_ids"]]

    def _check_done_batch(self, active: list[int]) -> None:
        for idx in active:
            terminated = (self.step_count[idx] - 1) >= self.episode_steps - 2
            alive = set(
                int(o)
                for o in self.planets[idx, self.planet_mask[idx], P_OWNER]
                if int(o) != -1
            )
            alive.update(
                int(o) for o in self.fleets[idx, self.fleet_mask[idx], F_OWNER]
            )
            if len(alive) <= 1:
                terminated = True
            self.done[idx] = terminated

    def _current_rewards(self, idx: int) -> list[int]:
        if not self.done[idx]:
            return [0] * self.num_players
        scores = [0] * self.num_players
        for p in self.planets[idx, self.planet_mask[idx]]:
            owner = int(p[P_OWNER])
            if owner != -1:
                scores[owner] += int(p[P_SHIPS])
        for f in self.fleets[idx, self.fleet_mask[idx]]:
            scores[int(f[F_OWNER])] += int(f[F_SHIPS])
        max_score = max(scores)
        return [1 if score == max_score and max_score > 0 else -1 for score in scores]
