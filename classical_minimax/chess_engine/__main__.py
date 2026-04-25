"""Entry point: ``python -m chess_engine`` (with ``PYTHONPATH`` including ``classical_minimax``)."""

from __future__ import annotations

from chess_engine.eval import EvalAgent
from chess_engine.move_gen import MoveGenAgent
from chess_engine.search import SearchAgent
from chess_engine.uci import UCIEngine

if __name__ == "__main__":
    move_gen = MoveGenAgent()
    eval_agent = EvalAgent()
    search = SearchAgent(eval_agent.evaluate, move_gen)
    UCIEngine(search).run()
