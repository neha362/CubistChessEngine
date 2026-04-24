"""
Move generation agent — wraps python-chess legal move generation.

python-chess already encodes the rules of chess in ``chess.Board.legal_moves``.
This agent is a thin, documented façade so the rest of the engine never calls
``legal_moves`` directly.

Rules surfaced through legal generation (all delegated to python-chess):

- **Castling**: King moves two squares toward a rook that has not moved; the rook
  jumps to the square the king crossed. Illegal when in check, through a
  controlled square, or when king or rook has moved.

- **En passant**: After a double pawn push, an adjacent pawn may capture the
  skipped square ``as if`` the pawn had only moved one square. The capture is
  only legal on the immediate reply move.

- **Promotion**: Pawn reaching the last rank must promote to N/B/R/Q (python-chess
  encodes each as a distinct move).

- **Pins**: A pinned piece may not move off the pin line if that would expose
  the king to check. ``legal_moves`` only emits moves that leave the king safe.

- **Checks**: In check, only moves that remove the check are legal; ``legal_moves``
  is already filtered.

See: https://python-chess.readthedocs.io/en/latest/core.html#chess.Board.legal_moves
"""

from __future__ import annotations

import chess
from typing import List


class MoveGenAgent:
    """Generates all legal moves for a position using python-chess."""

    def generate_moves(self, board: chess.Board) -> List[chess.Move]:
        """Return every legal move in the given position (same order as python-chess)."""
        return list(board.legal_moves)


def perft(board: chess.Board, depth: int, move_gen: MoveGenAgent | None = None) -> int:
    """
    Count leaf nodes at ``depth`` full moves (plies) from ``board``.

    ``depth`` is in plies: ``depth=1`` counts legal moves from the current node.
    """
    mg = move_gen or MoveGenAgent()
    if depth <= 0:
        return 1
    total = 0
    for move in mg.generate_moves(board):
        board.push(move)
        total += perft(board, depth - 1, mg)
        board.pop()
    return total
