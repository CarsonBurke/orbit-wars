from .base import Agent
from .heuristic import HeuristicAgent, heuristic_agent
from .random_agent import random_agent
from .sniper import sniper_agent

__all__ = [
    "Agent",
    "HeuristicAgent",
    "LearnedAgent",
    "heuristic_agent",
    "random_agent",
    "sniper_agent",
]


def __getattr__(name: str):
    if name == "LearnedAgent":
        from .learned import LearnedAgent

        return LearnedAgent
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
