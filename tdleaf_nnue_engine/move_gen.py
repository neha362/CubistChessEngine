"""Move generation and lightweight move ordering heuristics."""

from __future__ import annotations

from typing import Iterable, Optional

import chess

PIECE_VALUES = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 20_000,
}


class MoveGenerator:
    """Generates legal moves and scores them for alpha-beta ordering."""

    def legal_moves(self, board: chess.Board) -> list[chess.Move]:
        return list(board.legal_moves)

    def ordered_moves(
        self,
        board: chess.Board,
        tt_move: Optional[chess.Move] = None,
        killer_moves: Optional[Iterable[chess.Move]] = None,
    ) -> list[chess.Move]:
        killers = set(killer_moves or [])
        scored: list[tuple[int, chess.Move]] = []
        for move in board.legal_moves:
            scored.append((self._score_move(board, move, tt_move, killers), move))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [move for _, move in scored]

    def _score_move(
        self,
        board: chess.Board,
        move: chess.Move,
        tt_move: Optional[chess.Move],
        killer_moves: set[chess.Move],
    ) -> int:
        if tt_move is not None and move == tt_move:
            return 1_000_000

        score = 0
        if board.is_capture(move):
            victim = board.piece_at(move.to_square)
            attacker = board.piece_at(move.from_square)
            victim_value = PIECE_VALUES.get(victim.piece_type, 0) if victim else 0
            attacker_value = PIECE_VALUES.get(attacker.piece_type, 0) if attacker else 1
            score += 100_000 + 10 * victim_value - attacker_value

        if move in killer_moves:
            score += 50_000

        if board.gives_check(move):
            score += 1_000

        if move.promotion is not None:
            score += 800 + PIECE_VALUES.get(move.promotion, 0)

        return score
