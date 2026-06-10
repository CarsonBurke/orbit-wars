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
from ..policies.sampling import (
    ActionContext,
    _apply_target_legal_mask,
    _batch_record_from_materialized_launch,
    _categorical_support_launch_fraction,
    _ensure_deterministic_launch_if_idle,
    _mask_impossible_launches,
    _policy_fraction_params,
    _sample_categorical_action,
    _sample_fraction,
    _sample_launch_fraction,
    _sample_target,
)


def _native_has_required_api(module: Any) -> bool:
    core = getattr(module, "RustCoreVecEnv", None)
    return core is not None and hasattr(core, "builtin_actions")


def _load_native_path(path: Path) -> Any | None:
    if not path.exists():
        return None
    spec = importlib.util.spec_from_file_location("_owars_env", path)
    if spec is None or spec.loader is None:
        return None
    try:
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    except ImportError:
        # The extension is tied to the Python ABI. A checked-in or
        # previously-built .so from another venv can exist but fail to load.
        return None
    if not _native_has_required_api(module):
        return None
    return module


def _source_is_newer(artifact: Path, sources: list[Path]) -> bool:
    if not artifact.exists():
        return True
    artifact_mtime = artifact.stat().st_mtime
    return any(path.exists() and path.stat().st_mtime > artifact_mtime for path in sources)


def _load_native() -> Any:
    root = Path(__file__).resolve().parents[3]
    crate = root / "rust" / "owars_env_py"
    release = crate / "target" / "release" / "lib_owars_env.so"
    debug = crate / "target" / "debug" / "lib_owars_env.so"
    sources = [
        crate / "Cargo.toml",
        crate / "src" / "lib.rs",
        root / "rust" / "owars_env" / "Cargo.toml",
        root / "rust" / "owars_env" / "src" / "core.rs",
        root / "rust" / "owars_env" / "src" / "lib.rs",
    ]
    if (crate / "Cargo.toml").exists() and _source_is_newer(release, sources):
        subprocess.run(
            ["cargo", "build", "--release"],
            cwd=crate,
            check=True,
            stdout=subprocess.DEVNULL,
        )
    for path in (release, debug):
        module = _load_native_path(path)
        if module is not None:
            return module
    module = importlib.import_module("_owars_env")
    if not _native_has_required_api(module):
        raise ImportError("_owars_env is missing required RustCoreVecEnv.builtin_actions API")
    return module


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


def _numpy_from_tensor(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy()


class RustVecEnv:
    fast_rollout = True
    supports_replay = False
    native_builtin_opponents = frozenset(
        {
            "sniper",
            "sniper_v2",
            "sniper_v3",
            "sniper_v4",
            "sniper_v5",
            "sniper_v6",
            "sniper_v7",
            "sniper_v8",
            "sniper_v9",
            "sniper_v10",
            "sniper_v11",
            "sniper_v12",
            "sniper_v13",
            "sniper_v14",
            "sniper_v15",
            "sniper_v16",
            "sniper_v17",
        }
    )

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
        self.episode_steps = int(episode_steps)
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

    def reset_subset(self, indices: list[int]) -> dict[int, Any]:
        self._core.reset_subset([int(idx) for idx in indices])
        return {int(idx): None for idx in indices}

    def step_subset_fast(
        self, indices: list[int], actions: list[Any]
    ) -> dict[int, tuple[Any, bool, Any]]:
        raw = self._core.step_subset_fast(indices, actions)
        out: dict[int, tuple[Any, bool, Any]] = {}
        for idx, (_state, done, final_result) in raw.items():
            final = None
            if final_result is not None:
                rewards, scores = final_result
                final = [
                    SimpleNamespace(
                        reward=float(reward),
                        score=float(score),
                        status="DONE",
                        action=None,
                        observation=None,
                    )
                    for reward, score in zip(rewards, scores, strict=True)
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

    def reward_potentials(
        self,
        rows: list[tuple[int, int]],
        *,
        production_weight: float,
    ) -> np.ndarray:
        return self._core.reward_potentials(
            [(int(idx), int(player)) for idx, player in rows],
            float(production_weight),
        )

    def production_margins(self, rows: list[tuple[int, int]]) -> np.ndarray:
        return self._core.production_margins(
            [(int(idx), int(player)) for idx, player in rows]
        )

    def sample_batch_with_records(
        self,
        out: Any,
        rows: list[tuple[int, int]],
        *,
        deterministic: bool = False,
        record_rows: list[int] | None = None,
        record_source_mask: Any | None = None,
        deterministic_fallback: bool = False,
        native_actions: bool = False,
    ) -> tuple[list[list[list]], Any]:
        row_pairs = [(int(idx), int(player)) for idx, player in rows]
        if record_rows is None:
            record_rows = list(range(len(rows)))
        launch_logits = out.launch_logits
        launch_log_std = out.launch_log_std
        action_logit_softcap = out.action_logit_softcap
        target_logits = out.target_logits
        fraction_param1, fraction_param2, fraction_dist = _policy_fraction_params(out)
        if action_logit_softcap is None:
            launch, frac = _sample_launch_fraction(
                launch_logits,
                fraction_param1,
                fraction_param2,
                deterministic,
                fraction_dist,
                launch_log_std=launch_log_std,
                launch_prob_floor=out.launch_prob_floor,
            )
        else:
            frac = _sample_fraction(
                fraction_param1,
                fraction_param2,
                deterministic,
                fraction_dist,
            )
            support_launch, _support_frac = _categorical_support_launch_fraction(
                fraction_param1,
                fraction_param2,
                out.planet_owned_mask,
                out.planet_mask,
                fraction_dist,
            )
        mask_launch = launch if action_logit_softcap is None else support_launch
        mask_frac = frac
        active_fields_masker = getattr(
            self._core,
            "legal_target_mask_from_state_active_fields",
            None,
        )
        active_masker = getattr(self._core, "legal_target_mask_from_state_active", None)
        has_active_fields_masker = callable(active_fields_masker)
        if has_active_fields_masker:
            active_fields_np = _numpy_from_tensor(
                torch.stack((mask_frac.float(), mask_launch.float()), dim=-1)
            )
            if deterministic_fallback:
                active_fields_np[:, :, 1] = 1.0
            elif record_rows and action_logit_softcap is None:
                active_fields_np[record_rows, :, 1] = 1.0
            frac_np = active_fields_np[:, :, 0]
            target_legal_mask_np = active_fields_masker(row_pairs, active_fields_np)
        else:
            frac_np = _numpy_from_tensor(mask_frac.float())
        if not has_active_fields_masker and callable(active_masker):
            active_source = mask_launch.to(dtype=torch.bool)
            if deterministic_fallback:
                active_source = torch.ones_like(active_source, dtype=torch.bool)
            elif record_rows and action_logit_softcap is None:
                active_source = active_source.clone()
                record_idx = torch.as_tensor(
                    record_rows,
                    device=active_source.device,
                    dtype=torch.long,
                )
                active_source.index_fill_(0, record_idx, True)
            target_legal_mask_np = active_masker(
                row_pairs,
                frac_np,
                _numpy_from_tensor(active_source),
            )
        elif not has_active_fields_masker:
            target_legal_mask_np = self._core.legal_target_mask_from_state(
                row_pairs,
                frac_np,
            )
        target_legal_mask = torch.as_tensor(
            target_legal_mask_np, device=target_logits.device, dtype=torch.bool
        )
        target_logits = _apply_target_legal_mask(
            target_logits,
            target_legal_mask,
            mask_launch,
            out.planet_owned_mask,
            out.planet_mask,
        )
        if action_logit_softcap is None:
            launch_logits, launch = _mask_impossible_launches(
                launch_logits,
                launch,
                target_legal_mask,
                out.planet_owned_mask,
                out.planet_mask,
            )
            launch = _ensure_deterministic_launch_if_idle(
                launch_logits,
                launch,
                target_legal_mask,
                out.planet_owned_mask,
                out.planet_mask,
                deterministic_fallback,
            )
            target_idx = _sample_target(target_logits, deterministic)
        else:
            launch, target_idx = _sample_categorical_action(
                launch_logits,
                target_logits,
                action_logit_softcap,
                out.planet_owned_mask,
                out.planet_mask,
                deterministic,
            )
        materialize_fields = getattr(
            self._core,
            "materialize_masked_action_fields_from_state",
            None,
        )
        if callable(materialize_fields):
            action_fields = torch.stack(
                (launch.float(), target_idx.float(), frac.float()),
                dim=-1,
            )
            materialized = materialize_fields(
                row_pairs,
                _numpy_from_tensor(action_fields),
                bool(native_actions),
            )
        else:
            materializer = getattr(
                self._core,
                "materialize_masked_actions_from_state",
                self._core.materialize_actions_from_state,
            )
            materialized = materializer(
                row_pairs,
                _numpy_from_tensor(launch.float()),
                _numpy_from_tensor(target_idx.to(torch.int64)),
                _numpy_from_tensor(frac.float()),
                bool(native_actions),
            )
        actions_list = materialized["actions"]
        materialized_rows = materialized["materialized"][record_rows]
        records = _batch_record_from_materialized_launch(
            launch,
            target_idx,
            frac,
            launch_logits,
            launch_log_std,
            action_logit_softcap,
            target_logits,
            fraction_param1,
            fraction_param2,
            fraction_dist,
            materialized_rows,
            record_rows,
            launch_prob_floor=out.launch_prob_floor,
        )
        record_target_legal = np.ascontiguousarray(target_legal_mask_np[record_rows])
        if len(record_rows) > 0:
            if record_source_mask is None:
                source_mask_np = _numpy_from_tensor(
                    (out.planet_owned_mask & out.planet_mask).to(dtype=torch.bool)
                )[record_rows]
            else:
                source_mask_np = np.asarray(record_source_mask, dtype=bool)
                if source_mask_np.shape[0] == len(rows):
                    source_mask_np = source_mask_np[record_rows]
                elif source_mask_np.shape[0] != len(record_rows):
                    raise ValueError(
                        "record_source_mask must have one row per batch row "
                        "or one row per record row"
                    )
            record_target_legal = np.where(
                source_mask_np[:, :, None],
                record_target_legal,
                True,
            )
        records.target_legal_mask = torch.as_tensor(record_target_legal, dtype=torch.bool)
        return actions_list, records

    def sample_batch_actions(
        self,
        out: Any,
        rows: list[tuple[int, int]],
        *,
        deterministic: bool = True,
        native_actions: bool = False,
    ) -> list[list[list]]:
        actions, _records = self.sample_batch_with_records(
            out,
            rows,
            deterministic=deterministic,
            record_rows=[],
            deterministic_fallback=deterministic,
            native_actions=native_actions,
        )
        return actions

    def builtin_actions(
        self,
        name: str,
        rows: list[tuple[int, int]],
        *,
        native_actions: bool = False,
    ) -> list[Any]:
        return self._core.builtin_actions(
            str(name),
            [(int(idx), int(player)) for idx, player in rows],
            bool(native_actions),
        )

    def sniper_profile_actions(
        self,
        profile: dict[str, Any],
        rows: list[tuple[int, int]],
        *,
        native_actions: bool = False,
    ) -> list[Any]:
        return self._core.sniper_profile_actions(
            dict(profile),
            [(int(idx), int(player)) for idx, player in rows],
            bool(native_actions),
        )

    def policy_batch(
        self,
        rows: list[tuple[int, int]],
        *,
        device: str = "cpu",
        pin_memory: bool = False,
    ) -> tuple[EncodedObs, list[ActionContext]]:
        return self._policy_batch_from_core(
            self._core.policy_batch([(int(idx), int(player)) for idx, player in rows]),
            device=device,
            pin_memory=pin_memory,
            include_contexts=True,
        )

    def policy_batch_no_context(
        self,
        rows: list[tuple[int, int]],
        *,
        device: str = "cpu",
        pin_memory: bool = False,
    ) -> tuple[EncodedObs, list[ActionContext]]:
        return self._policy_batch_from_core(
            self._core.policy_batch_no_context(
                [(int(idx), int(player)) for idx, player in rows]
            ),
            device=device,
            pin_memory=pin_memory,
            include_contexts=False,
        )

    def _policy_batch_from_core(
        self,
        data: Any,
        *,
        device: str,
        pin_memory: bool,
        include_contexts: bool,
    ) -> tuple[EncodedObs, list[ActionContext]]:
        contexts = (
            [
                ActionContext(
                    planets=ctx[0],
                    angular_velocity=float(ctx[1]),
                    comet_planet_ids=tuple(int(pid) for pid in ctx[2].tolist()),
                )
                for ctx in data["contexts"]
            ]
            if include_contexts
            else []
        )
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
