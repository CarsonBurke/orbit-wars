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

from ..policies.features import (
    FLEET_FEAT_DIM,
    GLOBAL_FEAT_DIM,
    MAX_PLANETS,
    PLANET_FEAT_DIM,
    PLANET_INBOUND_FEAT_DIM,
    EncodedObs,
)
from ..policies.sampling import (
    SAMPLE_EPS,
    ActionContext,
    SampleBatchRecord,
    _apply_target_legal_mask,
    _batch_record_from_materialized_launch,
    _categorical_action_logits,
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
    return core is not None and all(
        hasattr(core, name)
        for name in (
            "builtin_actions",
            "enqueue_builtin_actions",
            "step_subset_flat_actions",
            "step_subset_pending_actions",
            "categorical_beta_actions_from_state_compact_sources",
            "enqueue_categorical_beta_actions_from_state_compact_sources",
        )
    )


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


def _rust_crate_sources(crate: Path) -> list[Path]:
    source_files = [
        path
        for path in sorted((crate / "src").rglob("*.rs"))
        if path.relative_to(crate / "src").parts[:1] != ("bin",)
        and path.name != "tests.rs"
    ]
    return [
        *(
            path
            for path in (crate / "Cargo.toml", crate / "Cargo.lock", crate / "build.rs")
            if path.exists()
        ),
        *source_files,
    ]


def _load_native() -> Any:
    root = Path(__file__).resolve().parents[3]
    crate = root / "rust" / "owars_env_py"
    core_crate = root / "rust" / "owars_env"
    release = crate / "target" / "release" / "lib_owars_env.so"
    debug = crate / "target" / "debug" / "lib_owars_env.so"
    sources = [*_rust_crate_sources(crate), *_rust_crate_sources(core_crate)]
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
        raise ImportError("_owars_env is missing required RustCoreVecEnv fast rollout API")
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
    if pin_memory and torch.cuda.is_available():
        tensor = tensor.pin_memory()
        if target.type == "cuda":
            return tensor.to(target, non_blocking=True)
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
    row_idx = np.asarray(compact["row_idx"], dtype=np.int64)
    if row_idx.size == 0:
        return record_target_legal
    source_idx = np.asarray(compact["source_idx"], dtype=np.int64)
    masks = np.asarray(compact["mask"], dtype=bool)
    record_rows_np = np.asarray(record_rows, dtype=np.int64)
    max_row = int(max(row_idx.max(initial=0), record_rows_np.max(initial=0)))
    row_to_record = np.full(max_row + 1, -1, dtype=np.int64)
    row_to_record[record_rows_np] = np.arange(len(record_rows_np), dtype=np.int64)
    valid = (row_idx >= 0) & (row_idx <= max_row)
    record_pos = np.full(row_idx.shape, -1, dtype=np.int64)
    record_pos[valid] = row_to_record[row_idx[valid]]
    keep = record_pos >= 0
    if keep.any():
        record_target_legal[record_pos[keep], source_idx[keep]] = masks[keep]
    return record_target_legal


def _dense_record_legal_mask_from_compact_native_result(
    native_result: Any,
    *,
    source_mask: np.ndarray,
    planets: int,
) -> np.ndarray:
    source_mask = np.asarray(source_mask, dtype=bool)
    records = int(source_mask.shape[0])
    out = np.ones((records, planets, planets), dtype=bool)
    owned_rows, owned_cols = np.nonzero(source_mask)
    if owned_rows.size:
        out[owned_rows, owned_cols] = False
    row_idx = np.asarray(native_result["target_legal_row_idx"], dtype=np.int64)
    if row_idx.size == 0:
        return out
    source_idx = np.asarray(native_result["target_legal_source_idx"], dtype=np.int64)
    masks = np.asarray(native_result["target_legal_source_mask"], dtype=bool)
    out[row_idx, source_idx] = masks
    return out


def _pad_native_categorical_beta_records(
    native_result: Any,
    *,
    planets: int,
) -> Any:
    raw_launch = np.asarray(native_result.get("raw_launch"))
    if raw_launch.ndim != 2:
        return native_result
    width = int(raw_launch.shape[1])
    if width == planets:
        return native_result
    if width > planets:
        raise ValueError("native categorical records are wider than policy logits")

    def pad_array(name: str, *, fill: float | int | bool = 0) -> np.ndarray | None:
        if name not in native_result:
            return None
        arr = np.asarray(native_result[name])
        if arr.ndim != 2 or int(arr.shape[1]) != width:
            return arr
        padded = np.full((int(arr.shape[0]), planets), fill, dtype=arr.dtype)
        padded[:, :width] = arr
        return np.ascontiguousarray(padded)

    out = dict(native_result)
    for name in ("launch", "raw_launch"):
        padded = pad_array(name)
        if padded is not None:
            out[name] = padded
    padded_target_idx = pad_array("target_idx")
    if padded_target_idx is not None:
        out["target_idx"] = padded_target_idx
    padded_fraction = pad_array("fraction", fill=0.5)
    if padded_fraction is not None:
        out["fraction"] = padded_fraction
    padded_log_prob = pad_array("log_prob")
    if padded_log_prob is not None:
        out["log_prob"] = padded_log_prob
    padded_legal = pad_array("target_legal_source_mask", fill=False)
    if padded_legal is not None:
        out["target_legal_source_mask"] = padded_legal
    return out


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
    source_target_logits = target_logits[row_idx, source_idx].float().masked_fill(
        ~legal_mask,
        float("-inf"),
    )
    action_logits = _categorical_action_logits(
        launch_logits[row_idx, source_idx],
        source_target_logits,
        action_logit_softcap,
    )
    if deterministic:
        action_idx = action_logits.argmax(dim=-1)
    else:
        uniform = torch.rand_like(action_logits).clamp_(SAMPLE_EPS, 1.0 - SAMPLE_EPS)
        gumbel = -torch.log(-torch.log(uniform))
        action_idx = (action_logits + gumbel).argmax(dim=-1)
    source_launch = (action_idx > 0).to(dtype=launch.dtype)
    source_target_idx = (action_idx - 1).clamp_min(0)
    launch[row_idx, source_idx] = source_launch
    target_idx[row_idx, source_idx] = source_target_idx
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
        strict_target_legality: bool = True,
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
            bool(strict_target_legality),
        )
        self._numpy_stage: dict[tuple[str, torch.dtype], torch.Tensor] = {}
        self._cuda_stage: dict[tuple[str, torch.dtype, tuple[int, ...]], torch.Tensor] = {}
        self._feature_stage: dict[tuple[str, torch.dtype], torch.Tensor] = {}
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

    def _float_numpy_from_tensor_staged(self, name: str, tensor: torch.Tensor) -> np.ndarray:
        # Native samplers consume these NumPy views synchronously and never retain
        # them, so reusing the pinned staging buffers across calls is safe.
        if tensor.device.type != "cuda":
            return _as_numpy_view(_cpu_float_tensor(tensor))
        return self._numpy_from_tensor_staged(name, tensor.detach().to(dtype=torch.float32))

    def _compact_categorical_beta_numpy_fields(
        self,
        launch_logits: torch.Tensor,
        target_logits: torch.Tensor,
        fraction_param1: torch.Tensor,
        fraction_param2: torch.Tensor,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if target_logits.dim() != 2:
            raise ValueError("compact target_logits must have shape [sources, planets]")
        source_count = int(target_logits.shape[0])
        planets = int(target_logits.shape[1])
        if launch_logits.shape != (source_count,):
            raise ValueError("compact launch_logits must have shape [sources]")
        if fraction_param1.shape != (source_count,) or fraction_param2.shape != (source_count,):
            raise ValueError("compact fraction params must have shape [sources]")
        if target_logits.device.type != "cuda":
            return (
                _as_numpy_view(_cpu_float_tensor(launch_logits)),
                _as_numpy_view(_cpu_float_tensor(target_logits)),
                _as_numpy_view(_cpu_float_tensor(fraction_param1)),
                _as_numpy_view(_cpu_float_tensor(fraction_param2)),
            )
        flat_len = source_count * (planets + 3)
        stage_key = ("categorical_beta_packed", torch.float32, (planets,))
        flat_stage = self._cuda_stage.get(stage_key)
        if (
            flat_stage is None
            or flat_stage.device != target_logits.device
            or flat_stage.shape[0] < flat_len
        ):
            flat_stage = torch.empty(
                _next_power_of_two(flat_len),
                device=target_logits.device,
                dtype=torch.float32,
            )
            self._cuda_stage[stage_key] = flat_stage
        flat = flat_stage[:flat_len]
        launch_pos = 0
        alpha_pos = source_count
        beta_pos = source_count * 2
        target_pos = source_count * 3
        torch.cat(
            (
                launch_logits.detach().float(),
                fraction_param1.detach().float(),
                fraction_param2.detach().float(),
                target_logits.detach().float().reshape(-1),
            ),
            out=flat,
        )
        packed = self._numpy_from_tensor_staged("categorical_beta_packed", flat)
        return (
            packed[launch_pos : launch_pos + source_count],
            packed[target_pos:].reshape(source_count, planets),
            packed[alpha_pos : alpha_pos + source_count],
            packed[beta_pos : beta_pos + source_count],
        )

    def _tensor_from_numpy_staged(
        self,
        name: str,
        array: np.ndarray,
        device: str | torch.device,
        *,
        pin_memory: bool,
    ) -> torch.Tensor:
        # Rollout-only fast path: returned CPU tensors are views into reusable
        # staging buffers and must be consumed before the next staged feature
        # batch call. Public policy_batch callers keep reuse_pinned_buffers=False.
        target = torch.device(device)
        if not (pin_memory and torch.cuda.is_available()):
            return _tensor_from_numpy(array, target, pin_memory=False)
        src = torch.from_numpy(np.ascontiguousarray(array))
        key = (name, src.dtype)
        buf = self._feature_stage.get(key)
        trailing_shape = tuple(src.shape[1:])
        needs_buffer = (
            buf is None
            or buf.ndim != src.ndim
            or buf.shape[0] < src.shape[0]
            or tuple(buf.shape[1:]) != trailing_shape
        )
        if needs_buffer:
            capacity_shape = (_next_power_of_two(src.shape[0]), *trailing_shape)
            buf = torch.empty(
                capacity_shape,
                dtype=src.dtype,
                device="cpu",
                pin_memory=True,
            )
            self._feature_stage[key] = buf
        view = buf[tuple(slice(0, dim) for dim in src.shape)]
        view.copy_(src, non_blocking=False)
        if target.type == "cuda":
            return view.to(target, non_blocking=True)
        return view

    def _feature_stage_view(
        self,
        name: str,
        shape: tuple[int, ...],
        dtype: torch.dtype,
    ) -> torch.Tensor:
        key = (name, dtype)
        buf = self._feature_stage.get(key)
        trailing_shape = tuple(shape[1:])
        needs_buffer = (
            buf is None
            or buf.ndim != len(shape)
            or buf.shape[0] < shape[0]
            or tuple(buf.shape[1:]) != trailing_shape
        )
        if needs_buffer:
            capacity_shape = (_next_power_of_two(shape[0]), *trailing_shape)
            buf = torch.empty(
                capacity_shape,
                dtype=dtype,
                device="cpu",
                pin_memory=True,
            )
            self._feature_stage[key] = buf
        return buf[tuple(slice(0, dim) for dim in shape)]

    def _policy_batch_no_context_destination_direct(
        self,
        rows: list[tuple[int, int]],
        *,
        device: str,
    ) -> tuple[EncodedObs, list[ActionContext]]:
        batch = len(rows)
        planet_feats = self._feature_stage_view(
            "planet_feats",
            (batch, MAX_PLANETS, PLANET_FEAT_DIM),
            torch.float32,
        )
        planet_mask = self._feature_stage_view(
            "planet_mask",
            (batch, MAX_PLANETS),
            torch.bool,
        )
        planet_owned_mask = self._feature_stage_view(
            "planet_owned_mask",
            (batch, MAX_PLANETS),
            torch.bool,
        )
        planet_ids = self._feature_stage_view(
            "planet_ids",
            (batch, MAX_PLANETS),
            torch.long,
        )
        planet_garrison = self._feature_stage_view(
            "planet_garrison",
            (batch, MAX_PLANETS),
            torch.float32,
        )
        fleet_feats = self._feature_stage_view(
            "fleet_feats",
            (batch, 0, FLEET_FEAT_DIM),
            torch.float32,
        )
        fleet_mask = self._feature_stage_view("fleet_mask", (batch, 0), torch.bool)
        fleet_target_planet_idx = self._feature_stage_view(
            "fleet_target_planet_idx",
            (batch, 0),
            torch.long,
        )
        planet_inbound_feats = self._feature_stage_view(
            "planet_inbound_feats",
            (batch, MAX_PLANETS, PLANET_INBOUND_FEAT_DIM),
            torch.float32,
        )
        global_feats = self._feature_stage_view(
            "global_feats",
            (batch, GLOBAL_FEAT_DIM),
            torch.float32,
        )
        meta = self._core.policy_batch_no_context_destination_into(
            [(int(idx), int(player)) for idx, player in rows],
            global_feats.numpy(),
            planet_feats.numpy(),
            planet_mask.numpy(),
            planet_owned_mask.numpy(),
            planet_ids.numpy(),
            planet_garrison.numpy(),
            fleet_feats.numpy(),
            fleet_mask.numpy(),
            fleet_target_planet_idx.numpy(),
            planet_inbound_feats.numpy(),
        )
        target = torch.device(device)

        def move(t: torch.Tensor) -> torch.Tensor:
            if target.type == "cuda":
                return t.to(target, non_blocking=True)
            if target.type == "cpu":
                return t
            return t.to(target)

        return (
            EncodedObs(
                planet_feats=move(planet_feats),
                planet_mask=move(planet_mask),
                planet_owned_mask=move(planet_owned_mask),
                planet_ids=move(planet_ids),
                planet_garrison=move(planet_garrison),
                fleet_feats=move(fleet_feats),
                fleet_mask=move(fleet_mask),
                fleet_target_planet_idx=move(fleet_target_planet_idx),
                planet_inbound_feats=move(planet_inbound_feats),
                global_feats=move(global_feats),
                compact_source_rows=np.asarray(
                    meta["compact_source_rows"],
                    dtype=np.int64,
                ),
                compact_source_cols=np.asarray(
                    meta["compact_source_cols"],
                    dtype=np.int64,
                ),
                compact_target_planets=int(
                    meta.get("compact_target_planets", MAX_PLANETS)
                ),
            ),
            [],
        )

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
        flat_step = getattr(self._core, "step_subset_flat_actions", None)
        if callable(flat_step):
            env_rows: list[int] = []
            player_rows: list[int] = []
            flat_actions: list[Any] = []
            for env_idx, env_actions in zip(indices, actions, strict=True):
                if not isinstance(env_actions, list):
                    continue
                for player, player_actions in enumerate(
                    env_actions[: self._num_players]
                ):
                    if player_actions is None:
                        continue
                    env_rows.append(int(env_idx))
                    player_rows.append(int(player))
                    flat_actions.append(player_actions)
            raw = flat_step(
                [int(idx) for idx in indices],
                env_rows,
                player_rows,
                flat_actions,
            )
        else:
            raw = self._core.step_subset_fast(indices, actions)
        return self._convert_step_results(raw)

    def step_subset_flat_actions(
        self,
        indices: list[int],
        env_rows: list[int],
        player_rows: list[int],
        actions: list[Any],
    ) -> dict[int, tuple[Any, bool, Any]]:
        raw = self._core.step_subset_flat_actions(
            [int(idx) for idx in indices],
            [int(idx) for idx in env_rows],
            [int(player) for player in player_rows],
            actions,
        )
        return self._convert_step_results(raw)

    def step_subset_pending_actions(
        self,
        indices: list[int],
        env_rows: list[int],
        player_rows: list[int],
        actions: list[Any],
    ) -> dict[int, tuple[Any, bool, Any]]:
        raw = self._core.step_subset_pending_actions(
            [int(idx) for idx in indices],
            [int(idx) for idx in env_rows],
            [int(player) for player in player_rows],
            actions,
        )
        return self._convert_step_results(raw)

    @staticmethod
    def _convert_step_results(raw: Any) -> dict[int, tuple[Any, bool, Any]]:
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
        enqueue_actions: bool = False,
        compute_log_prob: bool = True,
        compact_legal_records: bool = False,
        compact_source_rows: Any | None = None,
        compact_source_cols: Any | None = None,
        compact_target_planets: int | None = None,
        timings: dict[str, float] | None = None,
    ) -> tuple[list[list[list]] | None, Any]:
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
        enqueue_compact_native_categorical_beta_sampler = getattr(
            self._core,
            "enqueue_categorical_beta_actions_from_state_compact_sources",
            None,
        )
        can_native_categorical_beta_action = (
            callable(native_categorical_beta_sampler)
            and action_logit_softcap is not None
            and fraction_dist == "beta"
            and not compute_log_prob
        )
        use_compact_native_categorical_beta_action = (
            can_native_categorical_beta_action
            and callable(compact_native_categorical_beta_sampler)
            and (
                (
                    target_logits.device.type == "cpu"
                    and (enqueue_actions or not deterministic)
                )
                or target_logits.device.type == "cuda"
            )
        )
        use_native_categorical_beta_action = can_native_categorical_beta_action and (
            target_logits.device.type == "cpu" or use_compact_native_categorical_beta_action
        )
        if use_native_categorical_beta_action:
            phase_t0 = _timing_start(timings)
            native_launch_logits = None
            native_target_logits = None
            native_fraction_param1 = None
            native_fraction_param2 = None
            if use_compact_native_categorical_beta_action:
                if compact_source_rows is not None or compact_source_cols is not None:
                    if compact_source_rows is None or compact_source_cols is None:
                        raise ValueError(
                            "compact_source_rows and compact_source_cols must be provided together"
                        )
                    native_source_rows = np.ascontiguousarray(
                        compact_source_rows,
                        dtype=np.int64,
                    )
                    native_source_cols = np.ascontiguousarray(
                        compact_source_cols,
                        dtype=np.int64,
                    )
                    if native_source_rows.shape != native_source_cols.shape:
                        raise ValueError(
                            "compact source row/col arrays must have matching shapes"
                        )
                    source_rows = torch.as_tensor(
                        native_source_rows,
                        device=target_logits.device,
                        dtype=torch.long,
                    )
                    source_cols = torch.as_tensor(
                        native_source_cols,
                        device=target_logits.device,
                        dtype=torch.long,
                    )
                else:
                    source_mask = (out.planet_owned_mask & out.planet_mask).to(
                        device=target_logits.device,
                        dtype=torch.bool,
                    )
                    source_indices = torch.nonzero(source_mask, as_tuple=False)
                    source_rows = source_indices[:, 0]
                    source_cols = source_indices[:, 1]
                    native_source_rows = np.ascontiguousarray(
                        source_rows.detach().cpu().numpy(),
                        dtype=np.int64,
                    )
                    native_source_cols = np.ascontiguousarray(
                        source_cols.detach().cpu().numpy(),
                        dtype=np.int64,
                    )
                compact_launch_logits = launch_logits[source_rows, source_cols]
                target_planets = int(target_logits.shape[1])
                if compact_target_planets is not None:
                    target_planets = max(
                        1,
                        min(target_planets, int(compact_target_planets)),
                    )
                compact_target_logits = target_logits[
                    source_rows,
                    source_cols,
                    :target_planets,
                ]
                compact_fraction_param1 = fraction_param1[source_rows, source_cols]
                compact_fraction_param2 = fraction_param2[source_rows, source_cols]
                (
                    native_launch_logits,
                    native_target_logits,
                    native_fraction_param1,
                    native_fraction_param2,
                ) = self._compact_categorical_beta_numpy_fields(
                    compact_launch_logits,
                    compact_target_logits,
                    compact_fraction_param1,
                    compact_fraction_param2,
                )
            else:
                native_launch_logits = self._float_numpy_from_tensor_staged(
                    "categorical_beta_launch",
                    launch_logits,
                )
                native_target_logits = self._float_numpy_from_tensor_staged(
                    "categorical_beta_target",
                    target_logits,
                )
                native_fraction_param1 = self._float_numpy_from_tensor_staged(
                    "categorical_beta_fraction_param1",
                    fraction_param1,
                )
                native_fraction_param2 = self._float_numpy_from_tensor_staged(
                    "categorical_beta_fraction_param2",
                    fraction_param2,
                )
            _add_elapsed(timings, "action_cpu_d2h_s", phase_t0)

            phase_t0 = _timing_start(timings)
            enqueued_actions = False
            if use_compact_native_categorical_beta_action:
                if enqueue_actions and callable(enqueue_compact_native_categorical_beta_sampler):
                    assert native_launch_logits is not None
                    assert native_target_logits is not None
                    assert native_fraction_param1 is not None
                    assert native_fraction_param2 is not None
                    native_result = enqueue_compact_native_categorical_beta_sampler(
                        row_pairs,
                        native_source_rows,
                        native_source_cols,
                        native_launch_logits,
                        native_target_logits,
                        native_fraction_param1,
                        native_fraction_param2,
                        int(compact_target_logits.shape[1]),
                        float(action_logit_softcap),
                        bool(deterministic),
                        [int(row) for row in record_rows],
                    )
                    enqueued_actions = True
                else:
                    assert native_launch_logits is not None
                    assert native_target_logits is not None
                    assert native_fraction_param1 is not None
                    assert native_fraction_param2 is not None
                    native_result = compact_native_categorical_beta_sampler(
                        row_pairs,
                        native_source_rows,
                        native_source_cols,
                        native_launch_logits,
                        native_target_logits,
                        native_fraction_param1,
                        native_fraction_param2,
                        int(compact_target_logits.shape[1]),
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

            actions_list = None if enqueued_actions else native_result["actions"]
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
                    target_legal_mask=None,
                )

            phase_t0 = _timing_start(timings)
            native_result = _pad_native_categorical_beta_records(
                native_result,
                planets=int(target_logits.shape[1]),
            )
            launch = torch.as_tensor(native_result["launch"], dtype=torch.float32)
            raw_launch = torch.as_tensor(native_result["raw_launch"], dtype=torch.float32)
            target_idx = torch.as_tensor(native_result["target_idx"], dtype=torch.long)
            fraction = torch.as_tensor(native_result["fraction"], dtype=torch.float32)
            if "target_legal_source_mask" in native_result:
                if record_source_mask is None:
                    source_mask_np = _numpy_from_tensor(
                        (out.planet_owned_mask & out.planet_mask).to(dtype=torch.bool)
                    )
                    source_mask_np = source_mask_np[record_rows]
                else:
                    source_mask_np = np.asarray(record_source_mask, dtype=bool)
                    if source_mask_np.shape[0] == len(rows):
                        source_mask_np = source_mask_np[record_rows]
                    elif source_mask_np.shape[0] != len(record_rows):
                        raise ValueError(
                            "record_source_mask must have one row per batch row "
                            "or one row per record row"
                        )
                if compact_legal_records:
                    target_legal_mask = None
                else:
                    target_legal_mask = torch.as_tensor(
                        _dense_record_legal_mask_from_compact_native_result(
                            native_result,
                            source_mask=source_mask_np,
                            planets=int(target_logits.shape[1]),
                        ),
                        dtype=torch.bool,
                    )
            else:
                target_legal_mask = torch.as_tensor(
                    native_result["target_legal_mask"],
                    dtype=torch.bool,
                )
            old_log_prob_computed = False
            log_prob = launch.new_zeros(launch.shape)
            if compute_log_prob and "log_prob" in native_result:
                log_prob = torch.as_tensor(native_result["log_prob"], dtype=torch.float32)
                old_log_prob_computed = True
            records = SampleBatchRecord(
                launch=launch,
                raw_launch=raw_launch,
                target_idx=target_idx,
                fraction=fraction,
                log_prob=log_prob,
                target_legal_mask=target_legal_mask,
            )
            records.old_log_prob_computed = old_log_prob_computed
            if compact_legal_records and "target_legal_source_mask" in native_result:
                records.target_legal_row_idx = torch.as_tensor(
                    native_result["target_legal_row_idx"],
                    dtype=torch.long,
                )
                records.target_legal_source_idx = torch.as_tensor(
                    native_result["target_legal_source_idx"],
                    dtype=torch.long,
                )
                records.target_legal_source_mask = torch.as_tensor(
                    native_result["target_legal_source_mask"],
                    dtype=torch.bool,
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
            support_launch = (cpu_owned_mask & cpu_planet_mask).to(
                dtype=frac.dtype,
                device=frac.device,
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
                    support_launch = (out.planet_owned_mask & out.planet_mask).to(
                        dtype=frac.dtype,
                        device=frac.device,
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
        action_fields_np: np.ndarray | None = None
        if callable(materialize_fields):
            action_fields_np = self._numpy_from_tensor_staged("action_fields", action_fields)
            materialized = materialize_fields(
                row_pairs,
                action_fields_np,
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
                target_legal_mask=None,
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
        elif action_fields_np is not None:
            record_rows_np = np.asarray(record_rows, dtype=np.int64)
            raw_launch_np = np.ascontiguousarray(
                action_fields_np[record_rows_np, :, 0],
                dtype=np.float32,
            )
            target_idx_np = np.ascontiguousarray(
                action_fields_np[record_rows_np, :, 1],
                dtype=np.int64,
            )
            fraction_np = np.ascontiguousarray(
                action_fields_np[record_rows_np, :, 2],
                dtype=np.float32,
            )
            raw_launch = torch.from_numpy(raw_launch_np)
            actual_launch = torch.as_tensor(materialized_rows, dtype=torch.float32)
            record_launch = actual_launch if action_logit_softcap is None else raw_launch
            records = SampleBatchRecord(
                launch=record_launch,
                raw_launch=raw_launch,
                target_idx=torch.from_numpy(target_idx_np),
                fraction=torch.from_numpy(fraction_np),
                log_prob=torch.zeros_like(record_launch),
                target_legal_mask=None,
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
                target_legal_mask=None,
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
        enqueue_actions: bool = False,
        compact_source_rows: Any | None = None,
        compact_source_cols: Any | None = None,
        compact_target_planets: int | None = None,
        timings: dict[str, float] | None = None,
    ) -> list[list[list]] | None:
        actions, _records = self.sample_batch_with_records(
            out,
            rows,
            deterministic=deterministic,
            record_rows=[],
            deterministic_fallback=deterministic,
            native_actions=native_actions,
            enqueue_actions=enqueue_actions,
            compute_log_prob=False,
            compact_legal_records=False,
            compact_source_rows=compact_source_rows,
            compact_source_cols=compact_source_cols,
            compact_target_planets=compact_target_planets,
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

    def enqueue_builtin_actions(self, name: str, rows: list[tuple[int, int]]) -> None:
        self._core.enqueue_builtin_actions(
            str(name),
            [(int(idx), int(player)) for idx, player in rows],
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
        reuse_pinned_buffers: bool = False,
    ) -> tuple[EncodedObs, list[ActionContext]]:
        return self._policy_batch_from_core(
            self._core.policy_batch(
                [(int(idx), int(player)) for idx, player in rows],
                bool(include_fleet_targets),
            ),
            device=device,
            pin_memory=pin_memory,
            include_contexts=True,
            reuse_pinned_buffers=reuse_pinned_buffers,
        )

    def policy_batch_no_context(
        self,
        rows: list[tuple[int, int]],
        *,
        device: str = "cpu",
        pin_memory: bool = False,
        include_fleet_targets: bool = False,
        reuse_pinned_buffers: bool = False,
    ) -> tuple[EncodedObs, list[ActionContext]]:
        direct_destination = getattr(
            self._core,
            "policy_batch_no_context_destination_into",
            None,
        )
        if (
            reuse_pinned_buffers
            and pin_memory
            and include_fleet_targets
            and callable(direct_destination)
        ):
            return self._policy_batch_no_context_destination_direct(
                [(int(idx), int(player)) for idx, player in rows],
                device=device,
            )
        return self._policy_batch_from_core(
            self._core.policy_batch_no_context(
                [(int(idx), int(player)) for idx, player in rows],
                bool(include_fleet_targets),
            ),
            device=device,
            pin_memory=pin_memory,
            include_contexts=False,
            reuse_pinned_buffers=reuse_pinned_buffers,
        )

    def _policy_batch_from_core(
        self,
        data: Any,
        *,
        device: str,
        pin_memory: bool,
        include_contexts: bool,
        reuse_pinned_buffers: bool,
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
        compact_source_rows = data.get("compact_source_rows")
        compact_source_cols = data.get("compact_source_cols")
        compact_target_planets = data.get("compact_target_planets")

        def feature_tensor(name: str, array: np.ndarray) -> torch.Tensor:
            if reuse_pinned_buffers:
                return self._tensor_from_numpy_staged(
                    name,
                    array,
                    device,
                    pin_memory=pin_memory,
                )
            return _tensor_from_numpy(array, device, pin_memory=pin_memory)

        return (
            EncodedObs(
                planet_feats=feature_tensor("planet_feats", data["planet_feats"]),
                planet_mask=feature_tensor("planet_mask", data["planet_mask"]),
                planet_owned_mask=feature_tensor(
                    "planet_owned_mask",
                    data["planet_owned_mask"],
                ),
                planet_ids=feature_tensor("planet_ids", data["planet_ids"]),
                planet_garrison=feature_tensor(
                    "planet_garrison",
                    data["planet_garrison"],
                ),
                fleet_feats=feature_tensor("fleet_feats", data["fleet_feats"]),
                fleet_mask=feature_tensor("fleet_mask", data["fleet_mask"]),
                fleet_target_planet_idx=None
                if fleet_targets is None
                else feature_tensor(
                    "fleet_target_planet_idx",
                    fleet_targets,
                ),
                planet_inbound_feats=None
                if planet_inbound is None
                else feature_tensor(
                    "planet_inbound_feats",
                    planet_inbound,
                ),
                global_feats=feature_tensor(
                    "global_feats",
                    data.get(
                        "global_feats",
                        np.zeros(
                            (int(data["planet_feats"].shape[0]), GLOBAL_FEAT_DIM),
                            dtype=np.float32,
                        ),
                    ),
                ),
                compact_source_rows=None
                if compact_source_rows is None
                else np.asarray(compact_source_rows, dtype=np.int64),
                compact_source_cols=None
                if compact_source_cols is None
                else np.asarray(compact_source_cols, dtype=np.int64),
                compact_target_planets=None
                if compact_target_planets is None
                else int(compact_target_planets),
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
