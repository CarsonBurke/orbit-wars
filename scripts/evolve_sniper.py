"""Evolve native sniper heuristic profiles with batched Rust rollouts."""

from __future__ import annotations

import argparse
import concurrent.futures as futures
import json
import math
import random
import time
from dataclasses import dataclass
from typing import Any

from owars.training.rust_env import RustVecEnv

BASE_PROFILE: dict[str, Any] = {
    "reserve_base": 0,
    "reserve_production": 0.35,
    "send_buffer": 1,
    "enemy_growth": True,
    "enemy_value": 3.55,
    "neutral_value": 1.45,
    "production_weight": 4.7243564847164325,
    "ship_cost_weight": 0.66,
    "time_cost_weight": 0.34358831285363617,
    "duplicate_penalty": 0.20,
    "allow_partial": False,
    "partial_min_fraction": 0.5,
    "partial_score_scale": 0.45,
    "net_defense_reserve": True,
    "defense_horizon": 41.30069766848027,
    "contested_extra_buffer": 0,
    "contested_window": 2.0,
    "reinforce_owned": True,
    "defense_arrival_slack": 1.0,
    "defense_score_weight": 13.0,
    "chronological_forecast": True,
    "comet_max_eta": 5.067770406031305,
    "counter_recapture": True,
    "recapture_min_gap": 0.5,
    "recapture_max_gap": 14.0,
    "recapture_score_weight": 8.52469036246334,
    "recapture_gap_cost": 0.24472159101961058,
    "aggressive_sources": True,
    "speed_bid": False,
    "speed_bid_max_factor": 1.2770669321193258,
    "speed_bid_tempo_weight": 0.47561154430408675,
    "global_assignment": False,
    "strict_defense": False,
    "shadow_capture": False,
}

GENES: tuple[tuple[str, float, float, str], ...] = (
    ("reserve_base", 0, 3, "int"),
    ("reserve_production", 0.15, 0.80, "float"),
    ("send_buffer", 1, 3, "int"),
    ("enemy_value", 2.6, 4.8, "float"),
    ("neutral_value", 0.65, 1.45, "float"),
    ("production_weight", 4.5, 9.0, "float"),
    ("ship_cost_weight", 0.42, 0.95, "float"),
    ("time_cost_weight", 0.18, 0.62, "float"),
    ("duplicate_penalty", 0.04, 0.42, "float"),
    ("defense_horizon", 28.0, 60.0, "float"),
    ("defense_score_weight", 5.5, 13.0, "float"),
    ("recapture_max_gap", 4.0, 14.0, "float"),
    ("recapture_score_weight", 3.5, 9.5, "float"),
    ("recapture_gap_cost", 0.08, 0.65, "float"),
    ("comet_max_eta", 5.0, 14.0, "float"),
    ("speed_bid_max_factor", 1.15, 1.8, "float"),
    ("speed_bid_tempo_weight", 0.1, 0.8, "float"),
)

BOOL_GENES: tuple[str, ...] = (
    "chronological_forecast",
    "counter_recapture",
    "strict_defense",
    "shadow_capture",
    "speed_bid",
    "global_assignment",
)


@dataclass(frozen=True)
class EvalResult:
    profile: dict[str, Any]
    fitness: float
    score_rate: float
    mean_margin: float
    games: int
    bot_seconds_per_step: float


def _native_actions_by_name(vec: RustVecEnv, name: str, rows: list[tuple[int, int]]) -> list:
    return [] if not rows else vec.builtin_actions(name, rows, native_actions=True)


def _native_actions_by_profile(
    vec: RustVecEnv, profile: dict[str, Any], rows: list[tuple[int, int]]
) -> list:
    return [] if not rows else vec.sniper_profile_actions(profile, rows, native_actions=True)


def play_profile_matchup(
    profile: dict[str, Any],
    opponent: str,
    *,
    games: int,
    num_envs: int,
    episode_steps: int,
    ship_speed: float,
    seed: int,
) -> tuple[float, float, int, float, int]:
    envs = min(max(1, num_envs), max(1, games))
    scores: list[float] = []
    margins: list[float] = []
    decision_seconds = 0.0
    decision_steps = 0
    with RustVecEnv(
        num_envs=envs,
        num_players=2,
        episode_steps=episode_steps,
        ship_speed=ship_speed,
        random_seed=seed,
    ) as vec:
        while len(scores) < games:
            vec.reset()
            profile_seats = [(len(scores) + env_idx) % 2 for env_idx in range(envs)]
            dones = [False] * envs
            finals = [None] * envs
            while not all(dones):
                profile_rows: list[tuple[int, int]] = []
                opponent_rows: list[tuple[int, int]] = []
                for env_idx, done in enumerate(dones):
                    if done:
                        continue
                    seat = profile_seats[env_idx]
                    profile_rows.append((env_idx, seat))
                    opponent_rows.append((env_idx, 1 - seat))

                actions_per_env = [[None, None] for _ in range(envs)]
                started = time.perf_counter()
                profile_actions = _native_actions_by_profile(vec, profile, profile_rows)
                decision_seconds += time.perf_counter() - started
                decision_steps += len(profile_rows)
                for (env_idx, seat), acts in zip(profile_rows, profile_actions, strict=True):
                    actions_per_env[env_idx][seat] = acts

                opponent_actions = _native_actions_by_name(vec, opponent, opponent_rows)
                for (env_idx, seat), acts in zip(opponent_rows, opponent_actions, strict=True):
                    actions_per_env[env_idx][seat] = acts

                active = [idx for idx, done in enumerate(dones) if not done]
                stepped = vec.step_subset_fast(active, [actions_per_env[idx] for idx in active])
                for env_idx, (_state, done, final) in stepped.items():
                    if done:
                        dones[env_idx] = True
                        finals[env_idx] = final

            for env_idx, final in enumerate(finals):
                if len(scores) >= games:
                    break
                if final is None:
                    raise RuntimeError("finished environment did not return final scores")
                seat = profile_seats[env_idx]
                final_scores = [float(item.score) for item in final]
                own = final_scores[seat]
                opp = final_scores[1 - seat]
                scores.append(1.0 if own > opp else 0.5 if own == opp else 0.0)
                margins.append(own - opp)
    score_rate = sum(scores) / max(1, len(scores))
    mean_margin = sum(margins) / max(1, len(margins))
    return score_rate, mean_margin, len(scores), decision_seconds, decision_steps


def evaluate_profile(
    item: tuple[int, dict[str, Any], list[str], int, int, int, float, int]
) -> EvalResult:
    idx, profile, opponents, games, num_envs, episode_steps, ship_speed, seed = item
    total_score = 0.0
    total_margin = 0.0
    total_games = 0
    total_decision_seconds = 0.0
    total_decision_steps = 0
    for opp_idx, opponent in enumerate(opponents):
        score, margin, played, seconds, steps = play_profile_matchup(
            profile,
            opponent,
            games=games,
            num_envs=num_envs,
            episode_steps=episode_steps,
            ship_speed=ship_speed,
            seed=seed + opp_idx * 10_000,
        )
        total_score += score * played
        total_margin += margin * played
        total_games += played
        total_decision_seconds += seconds
        total_decision_steps += steps
    score_rate = total_score / max(1, total_games)
    mean_margin = total_margin / max(1, total_games)
    fitness = score_rate + 0.05 * math.tanh(mean_margin / 1200.0)
    profile = dict(profile)
    profile["_id"] = idx
    return EvalResult(
        profile=profile,
        fitness=fitness,
        score_rate=score_rate,
        mean_margin=mean_margin,
        games=total_games,
        bot_seconds_per_step=total_decision_seconds / max(1, total_decision_steps),
    )


def random_profile(rng: random.Random) -> dict[str, Any]:
    profile = dict(BASE_PROFILE)
    for name, low, high, kind in GENES:
        if kind == "int":
            profile[name] = rng.randint(int(low), int(high))
        else:
            profile[name] = rng.uniform(low, high)
    for name in BOOL_GENES:
        if name in {"chronological_forecast", "counter_recapture"}:
            profile[name] = rng.random() < 0.75
        elif name == "global_assignment":
            profile[name] = rng.random() < 0.15
        else:
            profile[name] = rng.random() < 0.35
    return profile


def mutate(profile: dict[str, Any], rng: random.Random, rate: float, scale: float) -> dict[str, Any]:
    child = dict(profile)
    child.pop("_id", None)
    for name, low, high, kind in GENES:
        if rng.random() > rate:
            continue
        if kind == "int":
            child[name] = max(int(low), min(int(high), int(child[name]) + rng.choice([-1, 1])))
        else:
            width = high - low
            child[name] = max(low, min(high, float(child[name]) + rng.gauss(0.0, scale * width)))
    for name in BOOL_GENES:
        if rng.random() < rate * 0.35:
            child[name] = not bool(child[name])
    return child


def crossover(a: dict[str, Any], b: dict[str, Any], rng: random.Random) -> dict[str, Any]:
    child = dict(BASE_PROFILE)
    for key in child:
        child[key] = a.get(key, child[key]) if rng.random() < 0.5 else b.get(key, child[key])
    return child


def evolve(args: argparse.Namespace) -> None:
    rng = random.Random(args.seed)
    opponents = args.opponents
    population = [dict(BASE_PROFILE)]
    population.extend(random_profile(rng) for _ in range(args.population - 1))
    for generation in range(args.generations):
        eval_seed = args.seed + generation * 1_000_000
        jobs = [
            (
                idx,
                profile,
                opponents,
                args.games,
                args.num_envs,
                args.episode_steps,
                args.ship_speed,
                eval_seed,
            )
            for idx, profile in enumerate(population)
        ]
        started = time.perf_counter()
        if args.workers == 1:
            results = [evaluate_profile(job) for job in jobs]
        else:
            with futures.ProcessPoolExecutor(max_workers=args.workers) as pool:
                results = list(pool.map(evaluate_profile, jobs))
        results.sort(key=lambda row: row.fitness, reverse=True)
        best = results[0]
        print(
            "generation,best_fitness,best_score_rate,best_mean_margin,"
            "best_bot_seconds_per_step,games,wall_seconds"
        )
        print(
            f"{generation},{best.fitness:.6f},{best.score_rate:.4f},"
            f"{best.mean_margin:.2f},{best.bot_seconds_per_step:.9f},"
            f"{best.games},{time.perf_counter() - started:.3f}"
        )
        print("best_profile_json")
        print(json.dumps(best.profile, sort_keys=True))

        elites = [row.profile for row in results[: args.elites]]
        next_population = [dict(profile) for profile in elites]
        while len(next_population) < args.population:
            parent_a = rng.choice(elites)
            parent_b = rng.choice(elites)
            child = crossover(parent_a, parent_b, rng)
            child = mutate(child, rng, args.mutation_rate, args.mutation_scale)
            next_population.append(child)
        population = next_population


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--population", type=int, default=32)
    parser.add_argument("--generations", type=int, default=12)
    parser.add_argument("--elites", type=int, default=6)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--games", type=int, default=24, help="Games per opponent per genome.")
    parser.add_argument("--num-envs", type=int, default=24)
    parser.add_argument("--episode-steps", type=int, default=500)
    parser.add_argument("--ship-speed", type=float, default=6.0)
    parser.add_argument("--mutation-rate", type=float, default=0.35)
    parser.add_argument("--mutation-scale", type=float, default=0.12)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--opponents",
        nargs="+",
        default=[
            "sniper_v10",
            "sniper_v11",
            "sniper_v14",
            "sniper_v15",
            "sniper_v17",
        ],
    )
    args = parser.parse_args()
    args.population = max(2, args.population)
    args.elites = max(1, min(args.elites, args.population))
    evolve(args)


if __name__ == "__main__":
    main()
