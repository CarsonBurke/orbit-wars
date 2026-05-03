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

# Make the vendored package importable regardless of how the runner cwd's.
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

def _build_agent():
    weights = _HERE / "weights" / "policy.pt"
    if weights.exists():
        from owars.agents.learned import LearnedAgent

        return LearnedAgent(weights, deterministic=True)
    # Fallback so the bundle still plays a game even without weights —
    # useful for the validation episode and for shake-down runs.
    from owars.agents.heuristic import HeuristicAgent

    return HeuristicAgent()


_AGENT = _build_agent()


def agent(obs):
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
