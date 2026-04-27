"""Multi-process orbit-wars envs over Pipes.

One worker process per env, each owning its own `kaggle_environments`
instance. Main process drives the workers in lockstep:

    vec = VecEnv(num_envs=16, num_players=2, episode_steps=500, ship_speed=6.0)
    states = vec.reset()
    while not all(dones):
        # main: compute actions for the alive envs (batched policy forward)
        results = vec.step_subset(active_indices, actions_per_active_env)
        ...
    vec.close()

Why subprocess workers and not threads: kaggle_environments steps in pure
Python and is GIL-bound. With 16 threads we'd serialize on the GIL and
get no speedup; with 16 subprocesses the env steps actually run in
parallel on separate cores.

Why not vendor a numpy reimplementation: that's a bigger change tracked
separately in STRATEGY.md. Subprocess parallelism is the cheap win.
"""

from __future__ import annotations

import multiprocessing as mp
from typing import Any

_RESET = "reset"
_STEP = "step"
_CLOSE = "close"


def _env_worker(
    remote: Any,
    num_players: int,
    episode_steps: int,
    ship_speed: float,
) -> None:
    """Subprocess entrypoint. Lazy-imports kaggle_environments so the main
    process can spawn workers even on machines that lack the dep at
    import-time (the import only fires inside the worker)."""
    from kaggle_environments import make  # type: ignore[import-not-found]

    env = make(
        "orbit_wars",
        configuration={
            "episodeSteps": episode_steps,
            "shipSpeed": ship_speed,
        },
    )
    try:
        while True:
            try:
                cmd, payload = remote.recv()
            except EOFError:
                return
            if cmd == _CLOSE:
                return
            if cmd == _RESET:
                state = env.reset(num_agents=num_players)
                remote.send(("ok", state, False, None))
            elif cmd == _STEP:
                state = env.step(payload)
                done = bool(env.done)
                final = env.steps[-1] if done else None
                remote.send(("ok", state, done, final))
            else:
                remote.send(("err", f"unknown cmd: {cmd!r}", True, None))
    except Exception as e:  # bubble worker errors back to main
        try:
            remote.send(("err", repr(e), True, None))
        except Exception:
            pass


class VecEnv:
    """Pool of subprocess-backed orbit-wars envs.

    `step_subset` exists because envs finish on different env-steps —
    once `done`, an env shouldn't be stepped again, but we want the rest
    to keep going. Stepping in lockstep with all-or-nothing was simpler
    but burned ~2× the env time when episode lengths varied.
    """

    def __init__(
        self,
        num_envs: int,
        num_players: int,
        episode_steps: int,
        ship_speed: float,
        start_method: str = "spawn",
    ):
        self.num_envs = num_envs
        ctx = mp.get_context(start_method)
        pipe_pairs = [ctx.Pipe(duplex=True) for _ in range(num_envs)]
        self._remotes = [main for main, _ in pipe_pairs]
        work_remotes = [worker for _, worker in pipe_pairs]
        self._workers: list[mp.process.BaseProcess] = []
        for wr in work_remotes:
            p = ctx.Process(
                target=_env_worker,
                args=(wr, num_players, episode_steps, ship_speed),
                daemon=True,
            )
            p.start()
            self._workers.append(p)
        # Close child ends in the parent so EOF is raised cleanly on shutdown.
        for wr in work_remotes:
            wr.close()
        self._closed = False

    # ----- env protocol -----------------------------------------------------

    def reset(self) -> list[Any]:
        for r in self._remotes:
            r.send((_RESET, None))
        out = []
        for i, r in enumerate(self._remotes):
            tag, state, _done, _final = r.recv()
            if tag != "ok":
                raise RuntimeError(f"worker {i} reset failed: {state}")
            out.append(state)
        return out

    def step_subset(
        self, indices: list[int], actions: list[Any]
    ) -> dict[int, tuple[Any, bool, Any]]:
        """Step only `indices` (typically the not-yet-done envs).

        Returns `{env_idx: (state, done, final)}`. `final` is the kaggle
        env's last `steps` entry (per-seat reward + status) when done, else
        None.
        """
        assert len(indices) == len(actions), (len(indices), len(actions))
        for i, a in zip(indices, actions):
            self._remotes[i].send((_STEP, a))
        results: dict[int, tuple[Any, bool, Any]] = {}
        for i in indices:
            tag, state, done, final = self._remotes[i].recv()
            if tag != "ok":
                raise RuntimeError(f"worker {i} step failed: {state}")
            results[i] = (state, done, final)
        return results

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for r in self._remotes:
            try:
                r.send((_CLOSE, None))
            except Exception:
                pass
        for p in self._workers:
            p.join(timeout=2.0)
            if p.is_alive():
                p.terminate()
                p.join(timeout=1.0)
        for r in self._remotes:
            try:
                r.close()
            except Exception:
                pass

    def __enter__(self) -> "VecEnv":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def __del__(self) -> None:  # best-effort cleanup
        try:
            self.close()
        except Exception:
            pass
