from .base import Agent
from .heuristic import HeuristicAgent, heuristic_agent
from .random_agent import random_agent
from .sniper import sniper_agent

__all__ = [
    "Agent",
    "HeuristicAgent",
    "LearnedAgent",
    "SACAgent",
    "heuristic_agent",
    "random_agent",
    "sniper_agent",
]


def __getattr__(name: str):
    if name == "LearnedAgent":
        from .learned import LearnedAgent

        return LearnedAgent
    if name == "SACAgent":
        from .sac_agent import SACAgent

        return SACAgent
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
