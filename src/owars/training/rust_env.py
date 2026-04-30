"""In-process Rust Orbit Wars vector env.

The Rust extension owns all simulator state and exposes the same fast rollout
surface as `NumpyVecEnv`: direct policy feature batches, compact action-context
data, and fast stepping without materializing per-seat observations.
"""

from __future__ import annotations

import importlib
import importlib.util
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch

from ..policies.features import EncodedObs
from ..policies.sampling import ActionContext


def _load_native() -> Any:
    try:
        return importlib.import_module("_owars_env")
    except ModuleNotFoundError:
        root = Path(__file__).resolve().parents[3]
        candidates = [
            root / "rust" / "owars_env_py" / "target" / "release" / "lib_owars_env.so",
            root / "rust" / "owars_env_py" / "target" / "debug" / "lib_owars_env.so",
        ]
        for path in candidates:
            if not path.exists():
                continue
            spec = importlib.util.spec_from_file_location("_owars_env", path)
            if spec is None or spec.loader is None:
                continue
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module
        manifest = root / "rust" / "owars_env_py" / "Cargo.toml"
        if manifest.exists():
            subprocess.run(
                ["cargo", "build", "--release"],
                cwd=manifest.parent,
                check=True,
                stdout=subprocess.DEVNULL,
            )
            path = root / "rust" / "owars_env_py" / "target" / "release" / "lib_owars_env.so"
            if path.exists():
                spec = importlib.util.spec_from_file_location("_owars_env", path)
                if spec is not None and spec.loader is not None:
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                    return module
        raise


_native = _load_native()


def _tensor_from_numpy(
    array: np.ndarray,
    device: str | torch.device,
    *,
    pin_memory: bool,
) -> torch.Tensor:
    tensor = torch.from_numpy(array)
    target = torch.device(device)
    if pin_memory and target.type == "cuda":
        return tensor.pin_memory().to(target, non_blocking=True)
    return tensor.to(target)


class RustVecEnv:
    fast_rollout = True
    supports_replay = False

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
        del comet_speed, replay_env_idx
        self.num_envs = int(num_envs)
        self.replay_env_idx: int | None = None
        self.last_replay_html: str | None = None
        self._num_players = int(num_players)
        self._core = _native.RustCoreVecEnv(
            int(num_envs),
            int(num_players),
            int(episode_steps),
            float(ship_speed),
            int(random_seed or 0),
        )

    def reset(self) -> list[Any]:
        self.last_replay_html = None
        self._core.reset()
        return [None] * self.num_envs

    def step_subset_fast(
        self, indices: list[int], actions: list[Any]
    ) -> dict[int, tuple[Any, bool, Any]]:
        raw = self._core.step_subset_fast(indices, actions)
        out: dict[int, tuple[Any, bool, Any]] = {}
        for idx, (_state, done, rewards) in raw.items():
            final = None
            if rewards is not None:
                final = [
                    SimpleNamespace(
                        reward=float(reward),
                        status="DONE",
                        action=None,
                        observation=None,
                    )
                    for reward in rewards
                ]
            out[int(idx)] = (None, bool(done), final)
        return out

    def step_subset(
        self, indices: list[int], actions: list[Any]
    ) -> dict[int, tuple[Any, bool, Any]]:
        stepped = self.step_subset_fast(indices, actions)
        out: dict[int, tuple[Any, bool, Any]] = {}
        for idx, (_state, done, final) in stepped.items():
            state = None if not done else self._state(idx)
            out[idx] = (state, done, final)
        return out

    def observation(self, idx: int, player: int) -> dict[str, Any]:
        return self._core.observation(int(idx), int(player))

    def observations(self, rows: list[tuple[int, int]]) -> list[dict[str, Any]]:
        return self._core.observations([(int(idx), int(player)) for idx, player in rows])

    def policy_batch(
        self,
        rows: list[tuple[int, int]],
        *,
        device: str = "cpu",
        pin_memory: bool = False,
    ) -> tuple[EncodedObs, list[ActionContext]]:
        data = self._core.policy_batch([(int(idx), int(player)) for idx, player in rows])
        contexts = [
            ActionContext(
                planets=ctx[0],
                angular_velocity=float(ctx[1]),
                comet_planet_ids=tuple(int(pid) for pid in ctx[2].tolist()),
            )
            for ctx in data["contexts"]
        ]
        return (
            EncodedObs(
                planet_feats=_tensor_from_numpy(
                    data["planet_feats"], device, pin_memory=pin_memory
                ),
                planet_mask=_tensor_from_numpy(
                    data["planet_mask"], device, pin_memory=pin_memory
                ),
                planet_owned_mask=_tensor_from_numpy(
                    data["planet_owned_mask"], device, pin_memory=pin_memory
                ),
                planet_ids=_tensor_from_numpy(
                    data["planet_ids"], device, pin_memory=pin_memory
                ),
                planet_garrison=_tensor_from_numpy(
                    data["planet_garrison"], device, pin_memory=pin_memory
                ),
                fleet_feats=_tensor_from_numpy(
                    data["fleet_feats"], device, pin_memory=pin_memory
                ),
                fleet_mask=_tensor_from_numpy(
                    data["fleet_mask"], device, pin_memory=pin_memory
                ),
            ),
            contexts,
        )

    def _state(self, idx: int) -> list[dict[str, Any]]:
        return [
            {
                "action": None,
                "reward": 0.0,
                "info": {},
                "observation": self.observation(idx, player),
                "status": "DONE",
            }
            for player in range(self._num_players)
        ]

    def set_recording(self, enabled: bool) -> None:
        if not enabled:
            self.last_replay_html = None

    def close(self) -> None:
        return

    def __enter__(self) -> RustVecEnv:
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()
