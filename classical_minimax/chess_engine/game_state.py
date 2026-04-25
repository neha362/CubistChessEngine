"""Shared terminal detection for search, MCTS, and other agents."""

from __future__ import annotations

import chess


def is_terminal(board: chess.Board) -> bool:
    return (
        board.is_checkmate()
        or board.is_stalemate()
        or board.is_insufficient_material()
        or board.can_claim_fifty_moves()
        or board.can_claim_threefold_repetition()
    )


def terminal_score(board: chess.Board) -> int:
    if board.is_checkmate():
        return -100000  # side to move is in checkmate = losing
    return 0  # all other terminals are draws
