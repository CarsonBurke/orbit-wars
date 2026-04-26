"""Random baseline.

For each owned planet, with some probability send a random fraction of its
ships in a random direction. Useful as a smoke-test opponent and as a
sanity floor for a trained policy ("if it doesn't beat random, something is
broken").
"""

from __future__ import annotations

import math
import random
from typing import Any

from ..game import parse_observation


def random_agent(obs: Any, launch_prob: float = 0.3, max_send_frac: float = 0.5) -> list[list]:
    o = parse_observation(obs)
    moves: list[list] = []
    for p in o.my_planets():
        if p.ships < 2 or random.random() > launch_prob:
            continue
        send = max(1, int(p.ships * random.uniform(0.1, max_send_frac)))
        send = min(send, p.ships)
        angle = random.uniform(-math.pi, math.pi)
        moves.append([p.id, angle, send])
    return moves
