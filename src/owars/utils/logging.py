"""Tensorboard logging wrapper (same shape as hull-tactical's TBLogger)."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
from torch.utils.tensorboard import SummaryWriter


class TBLogger:
    """Thin wrapper around `SummaryWriter` with a `runs/<name>/<ts>` layout."""

    def __init__(self, run_name: str, root: str | Path = "runs", subdir: str | None = None):
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        path = Path(root) / run_name
        if subdir is not None:
            path = path / subdir
        path = path / ts
        path.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.writer = SummaryWriter(log_dir=str(path))

    def scalar(self, tag: str, value: float, step: int) -> None:
        if value is None or not np.isfinite(value):
            return
        self.writer.add_scalar(tag, float(value), step)

    def scalars(self, prefix: str, mapping: dict[str, float], step: int) -> None:
        for k, v in mapping.items():
            self.scalar(f"{prefix}/{k}", v, step)

    def histogram(self, tag: str, values, step: int) -> None:
        try:
            self.writer.add_histogram(tag, values, step)
        except (ValueError, RuntimeError):
            pass

    def hparams(self, hparams: dict, metrics: dict) -> None:
        flat = {k: (v if isinstance(v, (int, float, str, bool)) else str(v)) for k, v in hparams.items()}
        self.writer.add_hparams(flat, metrics)

    def close(self) -> None:
        self.writer.close()
