"""
Move Generation Agent.

Responsibilities:
  1. Produce all legal moves from a given board state. python-chess handles
     the hard parts (en passant, castling rights, pins, discovered checks)
     correctly — we use it as our trusted source of legality.
  2. Order moves for search. Good ordering produces more alpha-beta cutoffs.
     For the Berserker personality, ordering is BIASED: checks and captures
     come first, quiet moves come last. This isn't just an optimization — it
     reinforces the aggressive identity even at fixed depth.

Move ordering priority (highest first):
  1. Hint move from the transposition table (proven good in this position)
  2. Checks            — keep the king under pressure
  3. Captures, MVV-LVA — material trades, valuable victims first
  4. Promotions        — turning pawns into queens
  5. Quiet moves       — everything else
"""

from typing import Iterable, Optional
import chess

# Used for MVV-LVA capture ordering. Kept here (not imported from eval) because
# move-gen should not depend on eval — they're independent agents.
_PIECE_VALUES = {
    chess.PAWN: 100, chess.KNIGHT: 320, chess.BISHOP: 330,
    chess.ROOK: 500, chess.QUEEN: 900, chess.KING: 0,
}


class MoveGen:
    """Default move-gen agent. Emits legal moves and orders them aggressively."""

    name = "BerserkerMoveGen"

    def legal_moves(self, board: chess.Board) -> Iterable[chess.Move]:
        # python-chess's legal_moves already handles every edge case correctly:
        # pinned pieces can't move off the pin ray, king can't move into check,
        # castling rights/safety, en passant legality, etc.
        return list(board.legal_moves)

    def ordered_moves(
        self, board: chess.Board, hint_move: Optional[chess.Move] = None
    ) -> Iterable[chess.Move]:
        moves = list(board.legal_moves)
        moves.sort(key=lambda m: self._move_priority(board, m, hint_move), reverse=True)
        return moves

    def _move_priority(
        self, board: chess.Board, move: chess.Move, hint_move: Optional[chess.Move]
    ) -> int:
        """
        Compute a sort key. Higher = searched first.
        Buckets are spaced far apart so one criterion never overrules a higher one.
        """
        # 1. TT move always first — it's our best historical guess for this position.
        if hint_move is not None and move == hint_move:
            return 10_000_000

        score = 0

        # 2. Checks. board.gives_check is fast (doesn't push/pop). For the
        # Berserker personality we score checks very high — even non-capturing
        # checks beat captures-of-pawns in our search order. This is the bias
        # leaking out of eval and into move ordering.
        if board.gives_check(move):
            score += 1_000_000

        # 3. Captures (MVV-LVA). Multiply victim value by 10 so capturing a
        # queen with a queen still beats capturing a pawn with a queen.
        if board.is_capture(move):
            if board.is_en_passant(move):
                victim_val = _PIECE_VALUES[chess.PAWN]
            else:
                victim = board.piece_at(move.to_square)
                victim_val = _PIECE_VALUES[victim.piece_type] if victim else 0
            attacker = board.piece_at(move.from_square)
            attacker_val = _PIECE_VALUES[attacker.piece_type] if attacker else 0
            score += 100_000 + 10 * victim_val - attacker_val

        # 4. Promotions. Queen promotion is a huge swing.
        if move.promotion is not None:
            score += 50_000 + _PIECE_VALUES.get(move.promotion, 0)

        # 5. Quiet moves: sort by destination centrality as a tiebreaker.
        # Center squares (d4, e4, d5, e5) get small bonuses.
        file = chess.square_file(move.to_square)
        rank = chess.square_rank(move.to_square)
        center_distance = abs(3.5 - file) + abs(3.5 - rank)
        score += int(20 - 4 * center_distance)

        return score


# Convenience: a module-level instance teammates can import directly.
default = MoveGen()
