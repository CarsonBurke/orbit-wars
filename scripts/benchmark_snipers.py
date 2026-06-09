"""Benchmark builtin sniper variants head-to-head in the Rust env."""

from __future__ import annotations

import argparse
import itertools
import time
from dataclasses import dataclass

from owars.training.elo import EloTracker
from owars.training.rust_env import RustVecEnv


@dataclass(frozen=True)
class GameResult:
    bot: str
    opponent: str
    score: float
    margin: float


@dataclass
class MatchupResult:
    bot: str
    opponent: str
    outcomes: list[GameResult]
    wall_seconds: float
    decision_seconds: dict[str, float]
    decision_steps: dict[str, int]

    @property
    def games(self) -> int:
        return len(self.outcomes)

    @property
    def wins(self) -> int:
        return sum(1 for outcome in self.outcomes if outcome.score == 1.0)

    @property
    def draws(self) -> int:
        return sum(1 for outcome in self.outcomes if outcome.score == 0.5)

    @property
    def win_rate(self) -> float:
        return self.wins / max(1, self.games)

    @property
    def draw_rate(self) -> float:
        return self.draws / max(1, self.games)

    @property
    def mean_margin(self) -> float:
        return sum(outcome.margin for outcome in self.outcomes) / max(1, self.games)

    @property
    def games_per_second(self) -> float:
        return self.games / max(1e-9, self.wall_seconds)

    def seconds_per_step(self, bot: str) -> float:
        return self.decision_seconds.get(bot, 0.0) / max(1, self.decision_steps.get(bot, 0))


def _native_actions_by_name(vec: RustVecEnv, name: str, rows: list[tuple[int, int]]) -> list:
    if not rows:
        return []
    return vec.builtin_actions(name, rows, native_actions=True)


def play_matchup(
    bot: str,
    opponent: str,
    *,
    games: int,
    num_envs: int,
    episode_steps: int,
    ship_speed: float,
    seed: int,
) -> MatchupResult:
    envs = min(max(1, num_envs), max(1, games))
    outcomes: list[GameResult] = []
    decision_seconds = {bot: 0.0, opponent: 0.0}
    decision_steps = {bot: 0, opponent: 0}
    started = time.perf_counter()
    with RustVecEnv(
        num_envs=envs,
        num_players=2,
        episode_steps=episode_steps,
        ship_speed=ship_speed,
        random_seed=seed,
    ) as vec:
        while len(outcomes) < games:
            vec.reset()
            learner_seats = [(len(outcomes) + env_idx) % 2 for env_idx in range(envs)]
            dones = [False] * envs
            finals = [None] * envs
            while not all(dones):
                bot_rows: list[tuple[int, int]] = []
                opp_rows: list[tuple[int, int]] = []
                for env_idx, done in enumerate(dones):
                    if done:
                        continue
                    seat = learner_seats[env_idx]
                    bot_rows.append((env_idx, seat))
                    opp_rows.append((env_idx, 1 - seat))

                actions_per_env = [[None, None] for _ in range(envs)]
                bot_started = time.perf_counter()
                bot_actions = _native_actions_by_name(vec, bot, bot_rows)
                decision_seconds[bot] += time.perf_counter() - bot_started
                decision_steps[bot] += len(bot_rows)
                for (env_idx, seat), acts in zip(
                    bot_rows, bot_actions, strict=True
                ):
                    actions_per_env[env_idx][seat] = acts
                opp_started = time.perf_counter()
                opp_actions = _native_actions_by_name(vec, opponent, opp_rows)
                decision_seconds[opponent] += time.perf_counter() - opp_started
                decision_steps[opponent] += len(opp_rows)
                for (env_idx, seat), acts in zip(
                    opp_rows,
                    opp_actions,
                    strict=True,
                ):
                    actions_per_env[env_idx][seat] = acts

                active = [idx for idx, done in enumerate(dones) if not done]
                stepped = vec.step_subset_fast(active, [actions_per_env[idx] for idx in active])
                for env_idx, (_state, done, final) in stepped.items():
                    if done:
                        dones[env_idx] = True
                        finals[env_idx] = final

            for env_idx, final in enumerate(finals):
                if len(outcomes) >= games:
                    break
                if final is None:
                    raise RuntimeError("finished environment did not return final scores")
                seat = learner_seats[env_idx]
                scores = [float(item.score) for item in final]
                bot_score = scores[seat]
                opp_score = scores[1 - seat]
                if bot_score > opp_score:
                    score = 1.0
                elif bot_score == opp_score:
                    score = 0.5
                else:
                    score = 0.0
                outcomes.append(
                    GameResult(
                        bot=bot,
                        opponent=opponent,
                        score=score,
                        margin=bot_score - opp_score,
                    )
                )
    return MatchupResult(
        bot=bot,
        opponent=opponent,
        outcomes=outcomes,
        wall_seconds=time.perf_counter() - started,
        decision_seconds=decision_seconds,
        decision_steps=decision_steps,
    )


def _elo_table(results: list[MatchupResult], bots: list[str]) -> list[tuple[str, float, int]]:
    elo = EloTracker(initial_rating=1500.0, k_factor=24.0)
    for bot in bots:
        elo.ensure(bot)
    for result in results:
        for outcome in result.outcomes:
            elo.update_pair(outcome.bot, outcome.opponent, outcome.score)
    return sorted(
        ((bot, elo.get(bot), elo.games_played.get(bot, 0)) for bot in bots),
        key=lambda row: row[1],
        reverse=True,
    )


def _bot_step_time_table(results: list[MatchupResult], bots: list[str]) -> dict[str, float]:
    seconds_per_step: dict[str, float] = {}
    for bot in bots:
        seconds = sum(row.decision_seconds.get(bot, 0.0) for row in results)
        steps = sum(row.decision_steps.get(bot, 0) for row in results)
        seconds_per_step[bot] = seconds / max(1, steps)
    return seconds_per_step


def _per_opponent_elo_rows(
    results: list[MatchupResult], bots: list[str]
) -> list[tuple[str, str, float, int]]:
    rows: list[tuple[str, str, float, int]] = []
    for opponent in bots:
        candidates = [bot for bot in bots if bot != opponent]
        if not candidates:
            continue
        elo = EloTracker(initial_rating=1500.0, k_factor=24.0)
        for bot in candidates:
            elo.ensure(bot)
        for result in results:
            if result.opponent == opponent and result.bot != opponent:
                for outcome in result.outcomes:
                    elo.update_pair(outcome.bot, opponent, outcome.score)
            elif result.bot == opponent and result.opponent != opponent:
                for outcome in result.outcomes:
                    elo.update_pair(outcome.opponent, opponent, 1.0 - outcome.score)
        for bot in candidates:
            rows.append((bot, opponent, elo.get(bot), elo.games_played.get(bot, 0)))
    return sorted(rows, key=lambda row: (row[1], -row[2], row[0]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bots",
        nargs="+",
        default=[
            "sniper_v2",
            "sniper_v3",
            "sniper_v4",
            "sniper_v6",
            "sniper_v7",
            "sniper_v8",
            "sniper_v9",
            "sniper_v10",
            "sniper_v11",
        ],
        help="Native builtin bots to benchmark.",
    )
    parser.add_argument("--games", type=int, default=200)
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--episode-steps", type=int, default=500)
    parser.add_argument("--ship-speed", type=float, default=6.0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rows: list[MatchupResult] = []
    started = time.perf_counter()
    for bot, opponent in itertools.permutations(args.bots, 2):
        rows.append(
            play_matchup(
                bot,
                opponent,
                games=args.games,
                num_envs=args.num_envs,
                episode_steps=args.episode_steps,
                ship_speed=args.ship_speed,
                seed=args.seed,
            )
        )

    total_wall_seconds = time.perf_counter() - started
    total_games = sum(row.games for row in rows)

    print(
        "bot,opponent,games,win_rate,draw_rate,mean_margin,wins,draws,"
        "wall_seconds,games_per_sec,bot_seconds_per_step,opponent_seconds_per_step"
    )
    for row in rows:
        print(
            f"{row.bot},{row.opponent},{row.games},"
            f"{row.win_rate:.4f},{row.draw_rate:.4f},{row.mean_margin:.2f},"
            f"{row.wins},{row.draws},{row.wall_seconds:.3f},{row.games_per_second:.1f},"
            f"{row.seconds_per_step(row.bot):.9f},{row.seconds_per_step(row.opponent):.9f}"
        )

    print()
    print("overall_elo")
    print("bot,elo,games,bot_seconds_per_step")
    bot_step_times = _bot_step_time_table(rows, args.bots)
    for bot, rating, played in _elo_table(rows, args.bots):
        print(f"{bot},{rating:.1f},{played},{bot_step_times[bot]:.9f}")

    print()
    print("per_opponent_elo")
    print("bot,opponent,elo,games")
    for bot, opponent, rating, played in _per_opponent_elo_rows(rows, args.bots):
        print(f"{bot},{opponent},{rating:.1f},{played}")

    print()
    print("wall_clock")
    print("total_games,total_wall_seconds,games_per_sec")
    print(f"{total_games},{total_wall_seconds:.3f},{total_games / max(1e-9, total_wall_seconds):.1f}")


if __name__ == "__main__":
    main()
