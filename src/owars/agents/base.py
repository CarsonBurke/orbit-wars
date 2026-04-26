"""Agent protocol.

A Kaggle Orbit Wars agent is just a callable: `agent(obs) -> list[list]`. We
wrap that with a class so learned agents can carry state (model weights,
running buffers) without polluting module globals — the framework happily
calls a class instance via `__call__`.
"""

from __future__ import annotations

from typing import Any, Protocol


class Agent(Protocol):
    def __call__(self, obs: Any) -> list[list]: ...
