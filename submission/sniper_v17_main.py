"""Kaggle entry point for the tuned sniper_v17 heuristic."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any


def _bundle_root() -> Path:
    candidates: list[Path] = []
    file_name = globals().get("__file__")
    if file_name:
        candidates.append(Path(str(file_name)).resolve().parent)
    candidates.append(Path.cwd().resolve())
    candidates.extend(Path(p).resolve() for p in reversed(sys.path) if p)
    for candidate in candidates:
        if (candidate / "owars").exists():
            return candidate
    return candidates[0]


_HERE = _bundle_root()
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from owars.agents.sniper import sniper_v17_agent  # noqa: E402


def agent(obs: Any, *_args: Any) -> list[list]:
    return sniper_v17_agent(obs)
