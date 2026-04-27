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
_SET_RECORDING = "set_recording"


def _env_worker(
    remote: Any,
    num_players: int,
    episode_steps: int,
    ship_speed: float,
    record_replay: bool,
) -> None:
    """Subprocess entrypoint. Lazy-imports kaggle_environments so the main
    process can spawn workers even on machines that lack the dep at
    import-time (the import only fires inside the worker).

    Non-recording workers null out interior entries of `env.steps`/`env.logs`
    after every step to free the per-step state dicts. **`len(env.steps)` is
    preserved** because kaggle's `core.py:602` writes the next observation's
    `step` field as `len(self.steps)`, and the orbit_wars interpreter keys
    planet rotation, comet spawns, and episode termination off that field —
    truncating the list freezes time at step 1 and silently breaks training
    dynamics. The recording worker keeps full history so it can render the
    finished game to HTML on done.
    """
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
            if cmd == _SET_RECORDING:
                record_replay = bool(payload)
                remote.send(("ok", None, False, None, None))
                continue
            if cmd == _RESET:
                state = env.reset(num_agents=num_players)
                remote.send(("ok", state, False, None, None))
            elif cmd == _STEP:
                state = env.step(payload)
                done = bool(env.done)
                final = env.steps[-1] if done else None
                replay_html: str | None = None
                if done and record_replay:
                    replay_html = env.render(mode="html")
                if not record_replay and len(env.steps) >= 2:
                    # Free the just-superseded state dict but leave the
                    # list length alone (see core.py:602 dependency above).
                    env.steps[-2] = None
                    if len(env.logs) >= 2:
                        env.logs[-2] = None
                remote.send(("ok", state, done, final, replay_html))
            else:
                remote.send(("err", f"unknown cmd: {cmd!r}", True, None, None))
    except Exception as e:  # bubble worker errors back to main
        try:
            remote.send(("err", repr(e), True, None, None))
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
        replay_env_idx: int | None = None,
    ):
        self.num_envs = num_envs
        self.replay_env_idx = replay_env_idx
        # Latest finished game's rendered HTML from the recording worker.
        # Caller reads after a rollout, then resets to None on the next reset().
        self.last_replay_html: str | None = None
        ctx = mp.get_context(start_method)
        pipe_pairs = [ctx.Pipe(duplex=True) for _ in range(num_envs)]
        self._remotes = [main for main, _ in pipe_pairs]
        work_remotes = [worker for _, worker in pipe_pairs]
        self._workers: list[mp.process.BaseProcess] = []
        for i, wr in enumerate(work_remotes):
            p = ctx.Process(
                target=_env_worker,
                args=(
                    wr,
                    num_players,
                    episode_steps,
                    ship_speed,
                    i == replay_env_idx,
                ),
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
        self.last_replay_html = None
        for r in self._remotes:
            r.send((_RESET, None))
        out = []
        for i, r in enumerate(self._remotes):
            tag, state, _done, _final, _html = r.recv()
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
        None. If `replay_env_idx` is set and that env finishes during this
        call, the rendered HTML is stashed on `self.last_replay_html`.
        """
        assert len(indices) == len(actions), (len(indices), len(actions))
        for i, a in zip(indices, actions):
            self._remotes[i].send((_STEP, a))
        results: dict[int, tuple[Any, bool, Any]] = {}
        for i in indices:
            tag, state, done, final, html = self._remotes[i].recv()
            if tag != "ok":
                raise RuntimeError(f"worker {i} step failed: {state}")
            if html is not None:
                self.last_replay_html = html
            results[i] = (state, done, final)
        return results

    def set_recording(self, enabled: bool) -> None:
        """Toggle replay recording on the worker at `replay_env_idx`.

        No-op if `replay_env_idx` is None. Lets callers skip the
        render+pipe-transfer cost when no consumer wants the HTML
        (e.g., during value pretraining).
        """
        if self.replay_env_idx is None:
            return
        r = self._remotes[self.replay_env_idx]
        r.send((_SET_RECORDING, enabled))
        tag, _state, _done, _final, _html = r.recv()
        if tag != "ok":
            raise RuntimeError(f"worker {self.replay_env_idx} set_recording failed")
        if not enabled:
            self.last_replay_html = None

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
