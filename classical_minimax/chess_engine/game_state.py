"""
game_state.py — terminal-state helpers for the classical engine.

Recreated to fix the broken import chain in __init__.py / search.py.
The single function this module needs to expose is `terminal_score`,
which returns the leaf-node value when the search hits a terminal position.
"""

from __future__ import annotations

import chess

# Must match search.py's MATE_SCORE constant.
MATE_SCORE = 100_000


def terminal_score(board: chess.Board) -> int:
    """
    Return the negamax-perspective score for a terminal position.

    Convention: score is from the side-to-move's perspective.
      - If the side to move is checkmated, they have lost → return -MATE_SCORE.
      - Stalemate / insufficient material / 50-move / 3-fold rep → 0 (draw).
      - Caller should only invoke this when board.is_game_over() is True.
    """
    if board.is_checkmate():
        return -MATE_SCORE
    # Any other terminal condition is a draw.
    return 0


def is_terminal(board: chess.Board) -> bool:
    """Convenience: True if the position is checkmate, stalemate, or any draw rule."""
    return board.is_game_over(claim_draw=True)
