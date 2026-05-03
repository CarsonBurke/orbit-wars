"""Kaggle Orbit Wars submission entry point.

Layout (built by `scripts/bundle.py`):
    submission.tar.gz/
      main.py              <-- this file
      weights/policy.pt    <-- trained checkpoint
      owars/...            <-- vendored package

Kaggle's runner imports `main.agent` from this file with no internet
access. Build the agent at import time so Torch/model loading happens before
the first timed action call.
"""

from __future__ import annotations

import os
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
        if (candidate / "owars").exists() or (candidate / "weights").exists():
            return candidate
    return candidates[0]


# Make the vendored package importable regardless of how the runner executes
# the raw Python. Kaggle's validator execs this file without defining
# `__file__`, but it appends the bundle directory to `sys.path` first.
_HERE = _bundle_root()
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
# Local repo shake-down runs use src/ instead of a vendored owars/ package.
_REPO_SRC = _HERE.parent / "src"
if _REPO_SRC.exists() and str(_REPO_SRC) not in sys.path:
    sys.path.insert(0, str(_REPO_SRC))

def _heuristic_agent():
    from owars.agents.heuristic import HeuristicAgent

    return HeuristicAgent()


def _build_agent():
    weights = _HERE / "weights" / "policy.pt"
    if weights.exists():
        from owars.agents.learned import LearnedAgent

        return LearnedAgent(weights, deterministic=True)
    # Fallback so the bundle still plays a game even without weights —
    # useful for the validation episode and for shake-down runs.
    return _heuristic_agent()


_AGENT = _build_agent()


def agent(obs: Any, *_args: Any) -> list[list]:
    return _AGENT(obs)


# Minimal self-test you can run locally with:
#   python submission/main.py
if __name__ == "__main__":
    os.environ.setdefault("PYTHONHASHSEED", "0")
    from kaggle_environments import make  # type: ignore[import-not-found]

    env = make("orbit_wars", debug=True)
    env.run([agent, "random"])
    final = env.steps[-1]
    print({i: float(s.reward or 0.0) for i, s in enumerate(final)})
