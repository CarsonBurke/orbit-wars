"""Multi-process shards of the NumPy Orbit Wars vector env.

This backend keeps the public VecEnv protocol used by rollout code, but
splits a fixed number of envs across several worker processes. It is meant
for training runs where `NumpyVecEnv`'s in-process vectorization is correct
but leaves most CPU cores idle.
"""

from __future__ import annotations

import multiprocessing as mp
import os
from typing import Any

from .numpy_env import NumpyVecEnv

_RESET = "reset"
_STEP = "step"
_CLOSE = "close"
_SET_RECORDING = "set_recording"


def _balanced_shards(num_envs: int, num_workers: int) -> list[tuple[int, int]]:
    num_workers = max(1, min(num_envs, num_workers))
    base, extra = divmod(num_envs, num_workers)
    shards: list[tuple[int, int]] = []
    start = 0
    for worker_idx in range(num_workers):
        size = base + (1 if worker_idx < extra else 0)
        shards.append((start, size))
        start += size
    return shards


def _default_num_workers(num_envs: int) -> int:
    cores = os.cpu_count() or num_envs
    # Leave a few cores for the main process, dataloading/BLAS, and the GPU
    # driver. Users can set rollout.num_workers=num_envs for one worker per env.
    return max(1, min(num_envs, cores - 4 if cores > 8 else cores))


def _numpy_shard_worker(
    remote: Any,
    shard_size: int,
    num_players: int,
    episode_steps: int,
    ship_speed: float,
    comet_speed: float,
    random_seed: int | None,
) -> None:
    vec = NumpyVecEnv(
        num_envs=shard_size,
        num_players=num_players,
        episode_steps=episode_steps,
        ship_speed=ship_speed,
        comet_speed=comet_speed,
        random_seed=random_seed,
        replay_env_idx=None,
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
                vec.set_recording(bool(payload))
                remote.send(("ok", None))
            elif cmd == _RESET:
                remote.send(("ok", vec.reset()))
            elif cmd == _STEP:
                indices, actions = payload
                remote.send(("ok", vec.step_subset(indices, actions)))
            else:
                remote.send(("err", f"unknown cmd: {cmd!r}"))
    except Exception as exc:
        try:
            remote.send(("err", repr(exc)))
        except Exception:
            pass
    finally:
        vec.close()


class ShardedNumpyVecEnv:
    """Subprocess-backed shards of `NumpyVecEnv`.

    `num_envs` is still the total number of games per rollout. `num_workers`
    controls how many CPU processes those envs are split across. This is
    intentionally separate from the official `VecEnv`, which already runs
    one Kaggle env per worker process.
    """

    fast_rollout = False

    def __init__(
        self,
        num_envs: int,
        num_players: int,
        episode_steps: int,
        ship_speed: float,
        comet_speed: float = 4.0,
        random_seed: int | None = None,
        replay_env_idx: int | None = None,
        num_workers: int = 0,
        start_method: str = "spawn",
    ) -> None:
        del replay_env_idx  # NumPy backend does not render Kaggle replays.
        self.num_envs = num_envs
        self.replay_env_idx: int | None = None
        self.last_replay_html: str | None = None
        if num_envs < 1:
            raise ValueError("num_envs must be >= 1")
        if num_workers <= 0:
            num_workers = _default_num_workers(num_envs)
        self.shards = _balanced_shards(num_envs, num_workers)

        ctx = mp.get_context(start_method)
        pipe_pairs = [ctx.Pipe(duplex=True) for _ in self.shards]
        self._remotes = [main for main, _worker in pipe_pairs]
        worker_remotes = [worker for _main, worker in pipe_pairs]
        self._workers: list[mp.process.BaseProcess] = []
        self._env_to_shard: list[tuple[int, int]] = [(-1, -1)] * num_envs
        for shard_idx, (start, size) in enumerate(self.shards):
            for local_idx in range(size):
                self._env_to_shard[start + local_idx] = (shard_idx, local_idx)
            shard_seed = None if random_seed is None else random_seed + start
            proc = ctx.Process(
                target=_numpy_shard_worker,
                args=(
                    worker_remotes[shard_idx],
                    size,
                    num_players,
                    episode_steps,
                    ship_speed,
                    comet_speed,
                    shard_seed,
                ),
                daemon=True,
            )
            proc.start()
            self._workers.append(proc)
        for worker_remote in worker_remotes:
            worker_remote.close()
        self._closed = False

    def reset(self) -> list[Any]:
        self.last_replay_html = None
        for remote in self._remotes:
            remote.send((_RESET, None))
        states: list[Any] = [None] * self.num_envs
        for shard_idx, remote in enumerate(self._remotes):
            tag, payload = remote.recv()
            if tag != "ok":
                raise RuntimeError(f"numpy shard {shard_idx} reset failed: {payload}")
            start, _size = self.shards[shard_idx]
            for local_idx, state in enumerate(payload):
                states[start + local_idx] = state
        return states

    def step_subset(
        self, indices: list[int], actions: list[Any]
    ) -> dict[int, tuple[Any, bool, Any]]:
        assert len(indices) == len(actions), (len(indices), len(actions))
        grouped_indices: list[list[int]] = [[] for _ in self.shards]
        grouped_actions: list[list[Any]] = [[] for _ in self.shards]
        for env_idx, action in zip(indices, actions, strict=True):
            shard_idx, local_idx = self._env_to_shard[env_idx]
            grouped_indices[shard_idx].append(local_idx)
            grouped_actions[shard_idx].append(action)

        active_shards: list[int] = []
        for shard_idx, local_indices in enumerate(grouped_indices):
            if not local_indices:
                continue
            self._remotes[shard_idx].send(
                (_STEP, (local_indices, grouped_actions[shard_idx]))
            )
            active_shards.append(shard_idx)

        results: dict[int, tuple[Any, bool, Any]] = {}
        for shard_idx in active_shards:
            tag, payload = self._remotes[shard_idx].recv()
            if tag != "ok":
                raise RuntimeError(f"numpy shard {shard_idx} step failed: {payload}")
            start, _size = self.shards[shard_idx]
            for local_idx, result in payload.items():
                results[start + int(local_idx)] = result
        return results

    def set_recording(self, enabled: bool) -> None:
        for shard_idx, remote in enumerate(self._remotes):
            remote.send((_SET_RECORDING, enabled))
            tag, payload = remote.recv()
            if tag != "ok":
                raise RuntimeError(
                    f"numpy shard {shard_idx} set_recording failed: {payload}"
                )
        if not enabled:
            self.last_replay_html = None

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for remote in self._remotes:
            try:
                remote.send((_CLOSE, None))
            except Exception:
                pass
        for proc in self._workers:
            proc.join(timeout=2.0)
            if proc.is_alive():
                proc.terminate()
                proc.join(timeout=1.0)
        for remote in self._remotes:
            try:
                remote.close()
            except Exception:
                pass

    def __enter__(self) -> "ShardedNumpyVecEnv":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
