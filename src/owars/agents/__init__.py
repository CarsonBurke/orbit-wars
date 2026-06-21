from .base import Agent
from .heuristic import HeuristicAgent, heuristic_agent
from .random_agent import random_agent
from .sniper import (
    sniper_agent,
    sniper_v2_agent,
    sniper_v3_agent,
    sniper_v4_agent,
    sniper_v5_agent,
    sniper_v6_agent,
    sniper_v7_agent,
    sniper_v8_agent,
    sniper_v9_agent,
    sniper_v10_agent,
    sniper_v11_agent,
    sniper_v12_agent,
    sniper_v13_agent,
    sniper_v14_agent,
    sniper_v15_agent,
    sniper_v16_agent,
    sniper_v17_agent,
    sniper_v18_agent,
)

__all__ = [
    "Agent",
    "HeuristicAgent",
    "LearnedAgent",
    "SACAgent",
    "heuristic_agent",
    "random_agent",
    "sniper_agent",
    "sniper_v2_agent",
    "sniper_v3_agent",
    "sniper_v4_agent",
    "sniper_v5_agent",
    "sniper_v6_agent",
    "sniper_v7_agent",
    "sniper_v8_agent",
    "sniper_v9_agent",
    "sniper_v10_agent",
    "sniper_v11_agent",
    "sniper_v12_agent",
    "sniper_v13_agent",
    "sniper_v14_agent",
    "sniper_v15_agent",
    "sniper_v16_agent",
    "sniper_v17_agent",
    "sniper_v18_agent",
]


def __getattr__(name: str):
    if name == "LearnedAgent":
        from .learned import LearnedAgent

        return LearnedAgent
    if name == "SACAgent":
        from .sac_agent import SACAgent

        return SACAgent
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
