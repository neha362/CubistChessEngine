"""
Scenario 6 — evolutionary comparison of evaluators with engine-vs-engine play.

- :mod:`evolution` — multi-round weight evolution against a reference signal.
- :mod:`arena` — quick head-to-head games.
- :mod:`engines` — per-turn analysis (optimal move + centipawn loss per line).
"""

from .engines import Berserker2Engine, ClassicalEngine, TurnReport
from .evolution import run_evolution

__all__ = [
    "Berserker2Engine",
    "ClassicalEngine",
    "TurnReport",
    "run_evolution",
]
