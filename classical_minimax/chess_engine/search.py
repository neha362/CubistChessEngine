"""Search agent — negamax, alpha-beta, iterative deepening, Zobrist transposition table."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable, Optional

import chess

from chess_engine.game_state import terminal_score
from chess_engine.move_gen import MoveGenAgent

MATE_SCORE = 100000

# Transposition flags
TT_EXACT = 0
TT_LOWER = 1
TT_UPPER = 2

_PIECE_TO_Z = (
    (chess.PAWN, 0),
    (chess.KNIGHT, 1),
    (chess.BISHOP, 2),
    (chess.ROOK, 3),
    (chess.QUEEN, 4),
    (chess.KING, 5),
)


def _piece_zobrist_index(p: chess.Piece) -> int:
    base = 0 if p.color == chess.WHITE else 6
    for pt, i in _PIECE_TO_Z:
        if p.piece_type == pt:
            return base + i
    return 0


class Zobrist:
    """Position hashing for transposition table keys."""

    def __init__(self, seed: int = 0xC055EC5) -> None:
        rng = random.Random(seed)
        self._piece = [[rng.getrandbits(64) for _ in range(12)] for _ in range(64)]
        self._side = rng.getrandbits(64)
        self._castle = [rng.getrandbits(64) for _ in range(4)]
        self._ep_file = [rng.getrandbits(64) for _ in range(8)]

    def hash_board(self, board: chess.Board) -> int:
        h = 0
        for sq in chess.SQUARES:
            p = board.piece_at(sq)
            if p:
                h ^= self._piece[sq][_piece_zobrist_index(p)]
        if board.turn == chess.BLACK:
            h ^= self._side
        if board.has_kingside_castling_rights(chess.WHITE):
            h ^= self._castle[0]
        if board.has_queenside_castling_rights(chess.WHITE):
            h ^= self._castle[1]
        if board.has_kingside_castling_rights(chess.BLACK):
            h ^= self._castle[2]
        if board.has_queenside_castling_rights(chess.BLACK):
            h ^= self._castle[3]
        ep = board.ep_square
        if ep is not None:
            h ^= self._ep_file[chess.square_file(ep)]
        return h


@dataclass
class TTEntry:
    depth: int
    score: int
    flag: int
    best_move: Optional[chess.Move]


class SearchAgent:
    """
    Negamax search with alpha-beta pruning, iterative deepening, and a TT.

    ``eval_fn`` must return centipawns from White's perspective; leaves convert
    to side-relative scores for negamax.
    """

    def __init__(
        self,
        eval_fn: Callable[[chess.Board], int],
        move_gen: Optional[MoveGenAgent] = None,
    ) -> None:
        self._eval_fn = eval_fn
        self._move_gen = move_gen or MoveGenAgent()
        self._zobrist = Zobrist()
        self._tt: dict[int, TTEntry] = {}

    def best_move(self, board: chess.Board, max_depth: int) -> chess.Move:
        legal = self._move_gen.generate_moves(board)
        if not legal:
            raise ValueError("no legal moves in position")
        if len(legal) == 1:
            return legal[0]
        if max_depth < 1:
            raise ValueError("max_depth must be >= 1")

        self._tt.clear()
        root_hint: Optional[chess.Move] = None
        best: chess.Move = legal[0]

        for depth in range(1, max_depth + 1):
            alpha, beta = -MATE_SCORE - 1, MATE_SCORE + 1
            score, move = self._root_negamax(board, depth, alpha, beta, root_hint)
            if move is not None:
                best = move
                root_hint = move
            _ = score  # score available for logging if needed

        return best

    def root_all_scores(
        self, board: chess.Board, max_depth: int
    ) -> tuple[list[tuple[chess.Move, int]], chess.Move, int]:
        """
        Score every legal move from the **side-to-move** perspective (``max_depth``-ply look-ahead
        to the quiescent leaf, same convention as :meth:`best_move`).

        **Expected centipawn loss** for a move is ``best_score - score`` (0 for a best line).
        """
        legal = self._move_gen.generate_moves(board)
        if not legal:
            raise ValueError("no legal moves in position")
        if max_depth < 1:
            raise ValueError("max_depth must be >= 1")
        self._tt.clear()
        per_move: list[tuple[chess.Move, int]] = []
        best_move: chess.Move = legal[0]
        best_score = -MATE_SCORE - 1
        for move in legal:
            board.push(move)
            try:
                score, _ = self._negamax(
                    board, max_depth - 1, -MATE_SCORE - 1, MATE_SCORE + 1, 1
                )
                score = -score
            finally:
                board.pop()
            per_move.append((move, score))
            if score > best_score:
                best_score = score
                best_move = move
        return per_move, best_move, best_score

    def _eval_leaf(self, board: chess.Board) -> int:
        v = self._eval_fn(board)
        return v if board.turn == chess.WHITE else -v

    def _terminal(self, board: chess.Board, _ply: int) -> int:
        return terminal_score(board)

    def _ordered_moves(self, board: chess.Board, prefer: Optional[chess.Move]) -> list[chess.Move]:
        moves = self._move_gen.generate_moves(board)
        if prefer is None:
            return moves
        for i, m in enumerate(moves):
            if m == prefer:
                if i == 0:
                    return moves
                out = [moves[i]] + moves[:i] + moves[i + 1 :]
                return out
        return moves

    def _tt_store(self, key: int, depth: int, score: int, flag: int, best: Optional[chess.Move]) -> None:
        old = self._tt.get(key)
        if old is not None and old.depth > depth:
            return
        self._tt[key] = TTEntry(depth=depth, score=score, flag=flag, best_move=best)

    def _negamax(
        self, board: chess.Board, depth: int, alpha: int, beta: int, ply: int
    ) -> tuple[int, Optional[chess.Move]]:
        key = self._zobrist.hash_board(board)
        ent = self._tt.get(key)
        tt_move = ent.best_move if ent else None

        if ent is not None and ent.depth >= depth:
            s = ent.score
            if ent.flag == TT_EXACT:
                return s, tt_move
            if ent.flag == TT_LOWER and s >= beta:
                return s, tt_move
            if ent.flag == TT_UPPER and s <= alpha:
                return s, tt_move

        moves = self._ordered_moves(board, tt_move)
        if not moves:
            return self._terminal(board, ply), None

        if depth == 0:
            return self._eval_leaf(board), None

        best_move: Optional[chess.Move] = None
        best_score = -MATE_SCORE - 1
        orig_alpha = alpha
        for move in moves:
            board.push(move)
            score, _ = self._negamax(board, depth - 1, -beta, -alpha, ply + 1)
            score = -score
            board.pop()
            if score > best_score:
                best_score = score
                best_move = move
            if score > alpha:
                alpha = score
            if alpha >= beta:
                break

        flag = TT_EXACT
        if best_score <= orig_alpha:
            flag = TT_UPPER
        elif best_score >= beta:
            flag = TT_LOWER

        self._tt_store(key, depth, best_score, flag, best_move)
        return best_score, best_move

    def _root_negamax(
        self,
        board: chess.Board,
        depth: int,
        alpha: int,
        beta: int,
        hint: Optional[chess.Move],
    ) -> tuple[int, Optional[chess.Move]]:
        moves = self._ordered_moves(board, hint)
        best_move: Optional[chess.Move] = None
        best_score = -MATE_SCORE - 1
        for move in moves:
            board.push(move)
            score, _ = self._negamax(board, depth - 1, -beta, -alpha, 1)
            score = -score
            board.pop()
            if score > best_score:
                best_score = score
                best_move = move
            if score > alpha:
                alpha = score
        return best_score, best_move
