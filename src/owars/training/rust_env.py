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
from time import perf_counter
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
from torch import nn

from ..policies.features import GLOBAL_FEAT_DIM, EncodedObs
from ..policies.sampling import (
    ActionContext,
    SampleBatchRecord,
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


class _CategoricalActionFields(nn.Module):
    def __init__(self, action_logit_softcap: float, deterministic: bool) -> None:
        super().__init__()
        self.action_logit_softcap = float(action_logit_softcap)
        self.deterministic = bool(deterministic)

    def forward(
        self,
        launch_logits: torch.Tensor,
        target_logits: torch.Tensor,
        target_legal_mask: torch.Tensor,
        owned: torch.Tensor,
        planet_mask: torch.Tensor,
        fraction: torch.Tensor,
    ) -> torch.Tensor:
        support_launch = (owned & planet_mask).to(
            dtype=fraction.dtype,
            device=fraction.device,
        )
        masked_target_logits = _apply_target_legal_mask(
            target_logits,
            target_legal_mask,
            support_launch,
            owned,
            planet_mask,
        )
        launch, target_idx = _sample_categorical_action(
            launch_logits,
            masked_target_logits,
            self.action_logit_softcap,
            owned,
            planet_mask,
            self.deterministic,
        )
        return torch.stack(
            (launch.float(), target_idx.float(), fraction.float()),
            dim=-1,
        )


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
    return tensor.detach().contiguous().cpu().numpy()


def _bool_tensor_from_numpy(array: np.ndarray, device: torch.device) -> torch.Tensor:
    tensor = torch.from_numpy(np.ascontiguousarray(array))
    if device.type == "cuda":
        tensor = tensor.pin_memory().to(device, non_blocking=True)
    else:
        tensor = tensor.to(device)
    return tensor.to(dtype=torch.bool)


def _long_tensor_from_numpy(array: np.ndarray, device: torch.device) -> torch.Tensor:
    tensor = torch.from_numpy(np.ascontiguousarray(array, dtype=np.int64))
    if device.type == "cuda":
        tensor = tensor.pin_memory().to(device, non_blocking=True)
    else:
        tensor = tensor.to(device)
    return tensor.to(dtype=torch.long)


def _cpu_float_tensor(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.device.type == "cpu" and tensor.dtype == torch.float32:
        return tensor.detach().contiguous()
    return tensor.detach().to(device="cpu", dtype=torch.float32).contiguous()


def _cpu_bool_tensor(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.device.type == "cpu" and tensor.dtype == torch.bool:
        return tensor.detach().contiguous()
    return tensor.detach().to(device="cpu", dtype=torch.bool).contiguous()


def _dense_legal_mask_from_compact(
    compact: dict[str, np.ndarray],
    *,
    batch: int,
    planets: int,
    device: torch.device,
) -> torch.Tensor:
    target_legal_mask = torch.zeros(
        (batch, planets, planets),
        dtype=torch.bool,
        device=device,
    )
    row_idx_np = np.asarray(compact["row_idx"], dtype=np.int64)
    if row_idx_np.size == 0:
        return target_legal_mask
    row_idx = _long_tensor_from_numpy(row_idx_np, device)
    source_idx = _long_tensor_from_numpy(compact["source_idx"], device)
    mask = _bool_tensor_from_numpy(compact["mask"], device)
    target_legal_mask[row_idx, source_idx] = mask
    return target_legal_mask


def _record_legal_mask_from_compact(
    compact: dict[str, np.ndarray],
    *,
    record_rows: list[int],
    planets: int,
) -> np.ndarray:
    record_target_legal = np.zeros((len(record_rows), planets, planets), dtype=bool)
    if not record_rows:
        return record_target_legal
    record_pos = {int(row): idx for idx, row in enumerate(record_rows)}
    row_idx = np.asarray(compact["row_idx"], dtype=np.int64)
    source_idx = np.asarray(compact["source_idx"], dtype=np.int64)
    masks = np.asarray(compact["mask"], dtype=bool)
    for compact_idx, row in enumerate(row_idx):
        pos = record_pos.get(int(row))
        if pos is None:
            continue
        record_target_legal[pos, int(source_idx[compact_idx])] = masks[compact_idx]
    return record_target_legal


def _sample_categorical_action_from_compact(
    launch_logits: torch.Tensor,
    target_logits: torch.Tensor,
    compact: dict[str, np.ndarray],
    action_logit_softcap: float,
    fraction: torch.Tensor,
    deterministic: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    launch = torch.zeros_like(fraction)
    target_idx = torch.zeros(
        fraction.shape,
        dtype=torch.long,
        device=target_logits.device,
    )
    row_idx_np = np.asarray(compact["row_idx"], dtype=np.int64)
    if row_idx_np.size == 0:
        return launch, target_idx
    row_idx = _long_tensor_from_numpy(row_idx_np, target_logits.device)
    source_idx = _long_tensor_from_numpy(compact["source_idx"], target_logits.device)
    legal_mask = _bool_tensor_from_numpy(compact["mask"], target_logits.device)
    source_target_logits = (
        target_logits[row_idx, source_idx]
        .float()
        .masked_fill(
            ~legal_mask,
            float("-inf"),
        )
    )
    source_launch_logits = launch_logits[row_idx, source_idx].unsqueeze(1)
    source_mask = torch.ones(
        (row_idx.numel(), 1),
        dtype=torch.bool,
        device=target_logits.device,
    )
    source_launch, source_target_idx = _sample_categorical_action(
        source_launch_logits,
        source_target_logits.unsqueeze(1),
        action_logit_softcap,
        source_mask,
        source_mask,
        deterministic,
    )
    launch[row_idx, source_idx] = source_launch.squeeze(1).to(dtype=launch.dtype)
    target_idx[row_idx, source_idx] = source_target_idx.squeeze(1)
    return launch, target_idx


def _as_numpy_view(tensor: torch.Tensor) -> np.ndarray:
    if tensor.device.type != "cpu":
        raise ValueError("_as_numpy_view requires a CPU tensor")
    return tensor.contiguous().numpy()


def _next_power_of_two(n: int) -> int:
    n = max(1, int(n))
    return 1 << (n - 1).bit_length()


def _pad_first_dim(
    tensor: torch.Tensor,
    rows: int,
    *,
    fill: int | float | bool = 0,
) -> torch.Tensor:
    current = int(tensor.shape[0])
    if current >= rows:
        return tensor
    out = tensor.new_full((rows, *tensor.shape[1:]), fill)
    out[:current] = tensor
    return out


def _action_field_graph_rows(rows: int) -> int:
    return max(64, _next_power_of_two(rows))


def _add_timing(timings: dict[str, float] | None, key: str, seconds: float) -> None:
    if timings is not None:
        timings[key] = timings.get(key, 0.0) + float(seconds)


def _timing_start(timings: dict[str, float] | None) -> float:
    return perf_counter() if timings is not None else 0.0


def _add_elapsed(
    timings: dict[str, float] | None,
    key: str,
    start: float,
) -> None:
    if timings is not None:
        _add_timing(timings, key, perf_counter() - start)


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
        self._numpy_stage: dict[tuple[str, torch.dtype], torch.Tensor] = {}
        self._action_field_kernel_cache: dict[
            tuple[float, bool, tuple[int, ...], str], nn.Module
        ] = {}

    def _numpy_from_tensor_staged(self, name: str, tensor: torch.Tensor) -> np.ndarray:
        src = tensor.detach().contiguous()
        if src.device.type != "cuda":
            return src.cpu().numpy()
        key = (name, src.dtype)
        buf = self._numpy_stage.get(key)
        capacity_shape = (_next_power_of_two(src.shape[0]), *tuple(src.shape[1:]))
        needs_buffer = (
            buf is None
            or buf.ndim != src.ndim
            or buf.shape[0] < src.shape[0]
            or tuple(buf.shape[1:]) != tuple(src.shape[1:])
        )
        if needs_buffer:
            buf = torch.empty(
                capacity_shape,
                dtype=src.dtype,
                device="cpu",
                pin_memory=True,
            )
            self._numpy_stage[key] = buf
        view = buf[tuple(slice(0, dim) for dim in src.shape)]
        view.copy_(src, non_blocking=True)
        torch.cuda.current_stream(src.device).synchronize()
        return view.numpy()

    def _categorical_action_field_kernel(
        self,
        action_logit_softcap: float,
        deterministic: bool,
        shape_key: tuple[int, ...],
        device: torch.device,
        compile_mode: str,
    ) -> nn.Module:
        key = (
            float(action_logit_softcap),
            bool(deterministic),
            shape_key,
            str(compile_mode),
        )
        cached = self._action_field_kernel_cache.get(key)
        if cached is not None:
            return cached
        kernel: nn.Module = _CategoricalActionFields(
            action_logit_softcap,
            deterministic,
        )
        if device.type == "cuda":
            kernel = torch.compile(
                kernel,
                dynamic=False,
                fullgraph=True,
                mode=compile_mode,
            )
        self._action_field_kernel_cache[key] = kernel
        return kernel

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
        return self._core.production_margins([(int(idx), int(player)) for idx, player in rows])

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
        compute_log_prob: bool = True,
        timings: dict[str, float] | None = None,
    ) -> tuple[list[list[list]], Any]:
        row_pairs = [(int(idx), int(player)) for idx, player in rows]
        if record_rows is None:
            record_rows = list(range(len(rows)))
        launch_logits = out.launch_logits
        launch_log_std = out.launch_log_std
        action_logit_softcap = out.action_logit_softcap
        target_logits = out.target_logits
        fraction_param1, fraction_param2, fraction_dist = _policy_fraction_params(out)
        active_fields_masker = getattr(
            self._core,
            "legal_target_mask_from_state_active_fields",
            None,
        )
        compact_active_fields_masker = getattr(
            self._core,
            "compact_legal_target_mask_from_state_active_fields",
            None,
        )
        active_masker = getattr(self._core, "legal_target_mask_from_state_active", None)
        has_active_fields_masker = callable(active_fields_masker)
        use_compiled_action_fields = (
            deterministic
            and deterministic_fallback
            and not record_rows
            and action_logit_softcap is not None
            and target_logits.device.type == "cuda"
            and not compute_log_prob
        )
        native_categorical_beta_sampler = getattr(
            self._core,
            "categorical_beta_actions_from_state",
            None,
        )
        compact_native_categorical_beta_sampler = getattr(
            self._core,
            "categorical_beta_actions_from_state_compact_sources",
            None,
        )
        use_native_categorical_beta_action = (
            callable(native_categorical_beta_sampler)
            and action_logit_softcap is not None
            and fraction_dist == "beta"
            and not compute_log_prob
            # CUDA rollout should not copy logits to Rust just to sample actions;
            # keep sampling on GPU and only cross the CPU boundary for legal masks
            # and final action materialization.
            and target_logits.device.type == "cpu"
        )
        use_compact_native_categorical_beta_action = (
            use_native_categorical_beta_action
            and callable(compact_native_categorical_beta_sampler)
            and (not deterministic or not record_rows)
        )
        if use_native_categorical_beta_action:
            phase_t0 = _timing_start(timings)
            if use_compact_native_categorical_beta_action:
                source_mask = (out.planet_owned_mask & out.planet_mask).to(
                    device=target_logits.device,
                    dtype=torch.bool,
                )
                source_indices = torch.nonzero(source_mask, as_tuple=False)
                source_rows = source_indices[:, 0]
                source_cols = source_indices[:, 1]
                compact_launch_logits = launch_logits[source_rows, source_cols]
                compact_target_logits = target_logits[source_rows, source_cols]
                compact_fraction_param1 = fraction_param1[source_rows, source_cols]
                compact_fraction_param2 = fraction_param2[source_rows, source_cols]
                native_source_rows = np.ascontiguousarray(
                    source_rows.detach().cpu().numpy(),
                    dtype=np.int64,
                )
                native_source_cols = np.ascontiguousarray(
                    source_cols.detach().cpu().numpy(),
                    dtype=np.int64,
                )
                native_launch_logits = _as_numpy_view(_cpu_float_tensor(compact_launch_logits))
                native_target_logits = _as_numpy_view(_cpu_float_tensor(compact_target_logits))
                native_fraction_param1 = _as_numpy_view(_cpu_float_tensor(compact_fraction_param1))
                native_fraction_param2 = _as_numpy_view(_cpu_float_tensor(compact_fraction_param2))
            else:
                native_launch_logits = _as_numpy_view(_cpu_float_tensor(launch_logits))
                native_target_logits = _as_numpy_view(_cpu_float_tensor(target_logits))
                native_fraction_param1 = _as_numpy_view(_cpu_float_tensor(fraction_param1))
                native_fraction_param2 = _as_numpy_view(_cpu_float_tensor(fraction_param2))
            _add_elapsed(timings, "action_cpu_d2h_s", phase_t0)

            phase_t0 = _timing_start(timings)
            if use_compact_native_categorical_beta_action:
                native_result = compact_native_categorical_beta_sampler(
                    row_pairs,
                    native_source_rows,
                    native_source_cols,
                    native_launch_logits,
                    native_target_logits,
                    native_fraction_param1,
                    native_fraction_param2,
                    int(target_logits.shape[1]),
                    float(action_logit_softcap),
                    bool(deterministic),
                    [int(row) for row in record_rows],
                    bool(native_actions),
                )
            else:
                native_result = native_categorical_beta_sampler(
                    row_pairs,
                    native_launch_logits,
                    native_target_logits,
                    native_fraction_param1,
                    native_fraction_param2,
                    float(action_logit_softcap),
                    bool(deterministic),
                    [int(row) for row in record_rows],
                    bool(native_actions),
                )
            _add_elapsed(timings, "native_action_s", phase_t0)

            actions_list = native_result["actions"]
            if not record_rows:
                planets = int(target_logits.shape[1])
                empty = target_logits.new_empty(0, planets)
                empty_idx = torch.empty(
                    0,
                    planets,
                    dtype=torch.long,
                    device=target_logits.device,
                )
                return actions_list, SampleBatchRecord(
                    launch=empty,
                    raw_launch=empty,
                    target_idx=empty_idx,
                    fraction=empty,
                    log_prob=empty,
                    target_legal_mask=torch.empty(0, dtype=torch.bool),
                )

            phase_t0 = _timing_start(timings)
            launch = torch.as_tensor(native_result["launch"], dtype=torch.float32)
            raw_launch = torch.as_tensor(native_result["raw_launch"], dtype=torch.float32)
            target_idx = torch.as_tensor(native_result["target_idx"], dtype=torch.long)
            fraction = torch.as_tensor(native_result["fraction"], dtype=torch.float32)
            target_legal_mask = torch.as_tensor(
                native_result["target_legal_mask"],
                dtype=torch.bool,
            )
            records = SampleBatchRecord(
                launch=launch,
                raw_launch=raw_launch,
                target_idx=target_idx,
                fraction=fraction,
                log_prob=launch.new_zeros(launch.shape),
                target_legal_mask=target_legal_mask,
            )
            _add_elapsed(timings, "record_build_s", phase_t0)
            return actions_list, records

        use_cpu_categorical_action = (
            callable(compact_active_fields_masker)
            and action_logit_softcap is not None
            and not compute_log_prob
            # Same rule as the native Beta sampler above: CUDA logits stay on GPU.
            and target_logits.device.type == "cpu"
        )
        target_legal_compact: dict[str, np.ndarray] | None = None
        target_legal_mask_np: np.ndarray | None = None
        cpu_source_mask_np: np.ndarray | None = None
        if use_cpu_categorical_action:
            phase_t0 = _timing_start(timings)
            cpu_launch_logits = _cpu_float_tensor(launch_logits)
            cpu_target_logits = _cpu_float_tensor(target_logits)
            cpu_fraction_param1 = _cpu_float_tensor(fraction_param1)
            cpu_fraction_param2 = _cpu_float_tensor(fraction_param2)
            cpu_owned_mask = _cpu_bool_tensor(out.planet_owned_mask)
            cpu_planet_mask = _cpu_bool_tensor(out.planet_mask)
            _add_elapsed(timings, "action_cpu_d2h_s", phase_t0)

            phase_t0 = _timing_start(timings)
            frac = _sample_fraction(
                cpu_fraction_param1,
                cpu_fraction_param2,
                deterministic,
                fraction_dist,
            )
            support_launch, _support_frac = _categorical_support_launch_fraction(
                cpu_fraction_param1,
                cpu_fraction_param2,
                cpu_owned_mask,
                cpu_planet_mask,
                fraction_dist,
            )
            cpu_source_mask_np = _as_numpy_view(cpu_owned_mask & cpu_planet_mask)
            mask_launch = support_launch
            if deterministic_fallback:
                mask_launch = torch.ones_like(mask_launch)
            mask_frac = frac
            _add_elapsed(timings, "sample_fraction_s", phase_t0)

            phase_t0 = _timing_start(timings)
            active_fields_np = _as_numpy_view(
                torch.stack((mask_frac.float(), mask_launch.float()), dim=-1)
            )
            frac_np = active_fields_np[:, :, 0]
            target_legal_compact = compact_active_fields_masker(
                row_pairs,
                active_fields_np,
            )
            _add_elapsed(timings, "legal_rust_s", phase_t0)

            phase_t0 = _timing_start(timings)
            launch, target_idx = _sample_categorical_action_from_compact(
                cpu_launch_logits,
                cpu_target_logits,
                target_legal_compact,
                float(action_logit_softcap),
                frac,
                deterministic,
            )
            action_fields = torch.stack(
                (launch.float(), target_idx.float(), frac.float()),
                dim=-1,
            )
            _add_elapsed(timings, "action_select_s", phase_t0)
        else:
            phase_t0 = _timing_start(timings)
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
                if use_compiled_action_fields:
                    support_launch = torch.ones_like(frac)
                else:
                    support_launch, _support_frac = _categorical_support_launch_fraction(
                        fraction_param1,
                        fraction_param2,
                        out.planet_owned_mask,
                        out.planet_mask,
                        fraction_dist,
                    )
            mask_launch = launch if action_logit_softcap is None else support_launch
            mask_frac = frac
            _add_elapsed(timings, "sample_fraction_s", phase_t0)
        if not use_cpu_categorical_action:
            phase_t0 = _timing_start(timings)
            if has_active_fields_masker:
                active_fields_np = self._numpy_from_tensor_staged(
                    "active_fields",
                    torch.stack((mask_frac.float(), mask_launch.float()), dim=-1),
                )
                _add_elapsed(timings, "legal_active_d2h_s", phase_t0)
                if deterministic_fallback:
                    active_fields_np[:, :, 1] = 1.0
                elif record_rows and action_logit_softcap is None:
                    active_fields_np[record_rows, :, 1] = 1.0
                frac_np = active_fields_np[:, :, 0]
                phase_t0 = _timing_start(timings)
                use_compact_masker = (
                    callable(compact_active_fields_masker)
                    and action_logit_softcap is not None
                    and target_logits.device.type == "cuda"
                )
                if use_compact_masker:
                    target_legal_compact = compact_active_fields_masker(
                        row_pairs,
                        active_fields_np,
                    )
                else:
                    target_legal_mask_np = active_fields_masker(row_pairs, active_fields_np)
                _add_elapsed(timings, "legal_rust_s", phase_t0)
            else:
                frac_np = self._numpy_from_tensor_staged("frac_mask", mask_frac.float())
                _add_elapsed(timings, "legal_active_d2h_s", phase_t0)
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
                phase_t0 = _timing_start(timings)
                active_source_np = self._numpy_from_tensor_staged("active_source", active_source)
                _add_elapsed(timings, "legal_active_d2h_s", phase_t0)
                phase_t0 = _timing_start(timings)
                target_legal_mask_np = active_masker(
                    row_pairs,
                    frac_np,
                    active_source_np,
                )
                _add_elapsed(timings, "legal_rust_s", phase_t0)
            elif not has_active_fields_masker:
                phase_t0 = _timing_start(timings)
                target_legal_mask_np = self._core.legal_target_mask_from_state(
                    row_pairs,
                    frac_np,
                )
                _add_elapsed(timings, "legal_rust_s", phase_t0)
        use_compact_categorical = (
            target_legal_compact is not None
            and action_logit_softcap is not None
            and not use_cpu_categorical_action
        )
        target_legal_mask: torch.Tensor | None = None
        phase_t0 = _timing_start(timings)
        if (
            target_legal_compact is not None
            and not use_compact_categorical
            and not use_cpu_categorical_action
        ):
            target_legal_mask = _dense_legal_mask_from_compact(
                target_legal_compact,
                batch=int(target_logits.shape[0]),
                planets=int(target_logits.shape[1]),
                device=target_logits.device,
            )
        elif not use_compact_categorical and not use_cpu_categorical_action:
            if target_legal_mask_np is None:
                raise RuntimeError("Rust legal mask path did not produce a mask")
            target_legal_mask = _bool_tensor_from_numpy(
                target_legal_mask_np,
                target_logits.device,
            )
        _add_elapsed(timings, "legal_mask_h2d_s", phase_t0)
        if not use_cpu_categorical_action:
            phase_t0 = _timing_start(timings)
        if use_cpu_categorical_action:
            pass
        elif use_compiled_action_fields and target_legal_mask is not None:
            real_rows = int(target_logits.shape[0])
            graph_rows = _action_field_graph_rows(real_rows)
            kernel_launch_logits = _pad_first_dim(launch_logits, graph_rows)
            kernel_target_logits = _pad_first_dim(target_logits, graph_rows)
            kernel_target_legal_mask = _pad_first_dim(
                target_legal_mask,
                graph_rows,
                fill=False,
            )
            kernel_owned_mask = _pad_first_dim(
                out.planet_owned_mask,
                graph_rows,
                fill=False,
            )
            kernel_planet_mask = _pad_first_dim(
                out.planet_mask,
                graph_rows,
                fill=False,
            )
            kernel_frac = _pad_first_dim(frac, graph_rows)
            kernel = self._categorical_action_field_kernel(
                float(action_logit_softcap),
                bool(deterministic),
                tuple(int(dim) for dim in kernel_target_logits.shape),
                target_logits.device,
                "reduce-overhead",
            )
            mark = getattr(torch.compiler, "cudagraph_mark_step_begin", None)
            if callable(mark):
                mark()
            action_fields = kernel(
                kernel_launch_logits,
                kernel_target_logits,
                kernel_target_legal_mask,
                kernel_owned_mask,
                kernel_planet_mask,
                kernel_frac,
            )[:real_rows]
            launch = action_fields[..., 0]
            target_idx = action_fields[..., 1].to(dtype=torch.long)
        elif use_compact_categorical:
            if target_legal_compact is None or action_logit_softcap is None:
                raise RuntimeError("compact categorical action path missing inputs")
            launch, target_idx = _sample_categorical_action_from_compact(
                launch_logits,
                target_logits,
                target_legal_compact,
                float(action_logit_softcap),
                frac,
                deterministic,
            )
            action_fields = torch.stack(
                (launch.float(), target_idx.float(), frac.float()),
                dim=-1,
            )
            if compute_log_prob:
                target_legal_mask = _dense_legal_mask_from_compact(
                    target_legal_compact,
                    batch=int(target_logits.shape[0]),
                    planets=int(target_logits.shape[1]),
                    device=target_logits.device,
                )
                target_logits = _apply_target_legal_mask(
                    target_logits,
                    target_legal_mask,
                    mask_launch,
                    out.planet_owned_mask,
                    out.planet_mask,
                )
        else:
            if target_legal_mask is None:
                raise RuntimeError("dense target legal mask is required for action selection")
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
            action_fields = torch.stack(
                (launch.float(), target_idx.float(), frac.float()),
                dim=-1,
            )
        if not use_cpu_categorical_action:
            _add_elapsed(timings, "action_select_s", phase_t0)
        materialize_fields = getattr(
            self._core,
            "materialize_masked_action_fields_from_state",
            None,
        )
        phase_t0 = _timing_start(timings)
        if callable(materialize_fields):
            materialized = materialize_fields(
                row_pairs,
                self._numpy_from_tensor_staged("action_fields", action_fields),
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
                self._numpy_from_tensor_staged("launch", launch.float()),
                self._numpy_from_tensor_staged("target_idx", target_idx.to(torch.int64)),
                self._numpy_from_tensor_staged("frac", frac.float()),
                bool(native_actions),
            )
        _add_elapsed(timings, "action_materialize_s", phase_t0)
        actions_list = materialized["actions"]
        if not record_rows:
            planets = int(target_logits.shape[1])
            empty = target_logits.new_empty(0, planets)
            empty_idx = torch.empty(
                0,
                planets,
                dtype=torch.long,
                device=target_logits.device,
            )
            return actions_list, SampleBatchRecord(
                launch=empty,
                raw_launch=empty,
                target_idx=empty_idx,
                fraction=empty,
                log_prob=empty,
                target_legal_mask=torch.empty(0, dtype=torch.bool),
            )
        phase_t0 = _timing_start(timings)
        materialized_rows = materialized["materialized"][record_rows]
        if compute_log_prob:
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
        else:
            row_idx = torch.as_tensor(record_rows, device=launch.device, dtype=torch.long)
            raw_launch = launch.index_select(0, row_idx)
            actual_launch = torch.as_tensor(
                materialized_rows,
                device=launch.device,
                dtype=launch.dtype,
            )
            record_launch = actual_launch if action_logit_softcap is None else raw_launch
            records = SampleBatchRecord(
                launch=record_launch.detach().cpu(),
                raw_launch=raw_launch.detach().cpu(),
                target_idx=target_idx.index_select(0, row_idx).detach().cpu(),
                fraction=frac.index_select(0, row_idx).detach().cpu(),
                log_prob=record_launch.new_zeros(record_launch.shape).detach().cpu(),
                target_legal_mask=torch.empty(0, dtype=torch.bool),
            )
        if target_legal_compact is not None:
            record_target_legal = _record_legal_mask_from_compact(
                target_legal_compact,
                record_rows=record_rows,
                planets=int(target_logits.shape[1]),
            )
        else:
            if target_legal_mask_np is None:
                raise RuntimeError("Rust legal mask path did not produce a record mask")
            record_target_legal = np.ascontiguousarray(target_legal_mask_np[record_rows])
        if len(record_rows) > 0:
            if record_source_mask is None:
                if cpu_source_mask_np is not None:
                    source_mask_np = cpu_source_mask_np[record_rows]
                else:
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
        _add_elapsed(timings, "record_build_s", phase_t0)
        return actions_list, records

    def sample_batch_actions(
        self,
        out: Any,
        rows: list[tuple[int, int]],
        *,
        deterministic: bool = True,
        native_actions: bool = False,
        timings: dict[str, float] | None = None,
    ) -> list[list[list]]:
        actions, _records = self.sample_batch_with_records(
            out,
            rows,
            deterministic=deterministic,
            record_rows=[],
            deterministic_fallback=deterministic,
            native_actions=native_actions,
            compute_log_prob=False,
            timings=timings,
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
        include_fleet_targets: bool = False,
    ) -> tuple[EncodedObs, list[ActionContext]]:
        return self._policy_batch_from_core(
            self._core.policy_batch(
                [(int(idx), int(player)) for idx, player in rows],
                bool(include_fleet_targets),
            ),
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
        include_fleet_targets: bool = False,
    ) -> tuple[EncodedObs, list[ActionContext]]:
        return self._policy_batch_from_core(
            self._core.policy_batch_no_context(
                [(int(idx), int(player)) for idx, player in rows],
                bool(include_fleet_targets),
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
        fleet_targets = data.get("fleet_target_planet_idx")
        planet_inbound = data.get("planet_inbound_feats")
        return (
            EncodedObs(
                planet_feats=_tensor_from_numpy(
                    data["planet_feats"], device, pin_memory=pin_memory
                ),
                planet_mask=_tensor_from_numpy(data["planet_mask"], device, pin_memory=pin_memory),
                planet_owned_mask=_tensor_from_numpy(
                    data["planet_owned_mask"], device, pin_memory=pin_memory
                ),
                planet_ids=_tensor_from_numpy(data["planet_ids"], device, pin_memory=pin_memory),
                planet_garrison=_tensor_from_numpy(
                    data["planet_garrison"], device, pin_memory=pin_memory
                ),
                fleet_feats=_tensor_from_numpy(data["fleet_feats"], device, pin_memory=pin_memory),
                fleet_mask=_tensor_from_numpy(data["fleet_mask"], device, pin_memory=pin_memory),
                fleet_target_planet_idx=None
                if fleet_targets is None
                else _tensor_from_numpy(fleet_targets, device, pin_memory=pin_memory),
                planet_inbound_feats=None
                if planet_inbound is None
                else _tensor_from_numpy(planet_inbound, device, pin_memory=pin_memory),
                global_feats=_tensor_from_numpy(
                    data.get(
                        "global_feats",
                        np.zeros(
                            (int(data["planet_feats"].shape[0]), GLOBAL_FEAT_DIM),
                            dtype=np.float32,
                        ),
                    ),
                    device,
                    pin_memory=pin_memory,
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
