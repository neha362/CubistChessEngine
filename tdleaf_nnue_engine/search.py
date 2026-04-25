"""Negamax alpha-beta search with TT and simple heuristics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import chess

from tdleaf_nnue_engine.eval import Evaluator
from tdleaf_nnue_engine.move_gen import MoveGenerator

MATE_SCORE = 100_000
TT_EXACT = 0
TT_LOWER = 1
TT_UPPER = 2
MAX_PLY = 64


@dataclass
class TTEntry:
    depth: int
    score: int
    flag: int
    best_move: Optional[chess.Move]


@dataclass
class SearchResult:
    best_move: chess.Move
    best_score: int
    leaf_scores: list[int]


class Searcher:
    """Practical search wrapper for move selection and TD-Leaf sampling."""

    def __init__(
        self,
        evaluator: Optional[Evaluator] = None,
        move_gen: Optional[MoveGenerator] = None,
    ) -> None:
        self.evaluator = evaluator or Evaluator()
        self.move_gen = move_gen or MoveGenerator()
        self.tt: dict[int, TTEntry] = {}
        self.killers: list[list[chess.Move]] = [[] for _ in range(MAX_PLY)]

    def best_move(self, board: chess.Board, depth: int = 3) -> chess.Move:
        return self.search(board, depth=depth).best_move

    def search(self, board: chess.Board, depth: int = 3) -> SearchResult:
        legal = self.move_gen.legal_moves(board)
        if not legal:
            raise ValueError("no legal moves in position")
        if len(legal) == 1:
            return SearchResult(best_move=legal[0], best_score=0, leaf_scores=[])
        self.tt.clear()
        self.killers = [[] for _ in range(MAX_PLY)]
        alpha = -MATE_SCORE - 1
        beta = MATE_SCORE + 1
        leaf_scores: list[int] = []
        best_score, best = self._root(board, depth, alpha, beta, leaf_scores)
        if best is None:
            best = legal[0]
        return SearchResult(best_move=best, best_score=best_score, leaf_scores=leaf_scores)

    def _root(
        self,
        board: chess.Board,
        depth: int,
        alpha: int,
        beta: int,
        leaf_scores: list[int],
    ) -> tuple[int, Optional[chess.Move]]:
        key = board._transposition_key()
        hint = self.tt.get(key)
        moves = self.move_gen.ordered_moves(board, tt_move=hint.best_move if hint else None)
        best_move = None
        best_score = -MATE_SCORE - 1
        for move in moves:
            board.push(move)
            score = -self._negamax(board, depth - 1, -beta, -alpha, 1, leaf_scores)
            board.pop()
            if score > best_score:
                best_score = score
                best_move = move
            if score > alpha:
                alpha = score
        return best_score, best_move

    def _negamax(
        self,
        board: chess.Board,
        depth: int,
        alpha: int,
        beta: int,
        ply: int,
        leaf_scores: list[int],
    ) -> int:
        if board.is_checkmate():
            return -MATE_SCORE + ply
        if board.is_stalemate() or board.is_insufficient_material() or board.can_claim_draw():
            return 0
        if depth <= 0:
            score = self._side_relative_eval(board)
            leaf_scores.append(score)
            return score

        key = board._transposition_key()
        entry = self.tt.get(key)
        tt_move = entry.best_move if entry else None
        if entry is not None and entry.depth >= depth:
            if entry.flag == TT_EXACT:
                return entry.score
            if entry.flag == TT_LOWER and entry.score >= beta:
                return entry.score
            if entry.flag == TT_UPPER and entry.score <= alpha:
                return entry.score

        orig_alpha = alpha
        best = -MATE_SCORE - 1
        best_move = None
        killers = self.killers[min(ply, MAX_PLY - 1)]
        moves = self.move_gen.ordered_moves(board, tt_move=tt_move, killer_moves=killers)

        for move in moves:
            board.push(move)
            score = -self._negamax(board, depth - 1, -beta, -alpha, ply + 1, leaf_scores)
            board.pop()

            if score > best:
                best = score
                best_move = move
            if score > alpha:
                alpha = score
            if alpha >= beta:
                if not board.is_capture(move):
                    self._store_killer(move, ply)
                break

        flag = TT_EXACT
        if best <= orig_alpha:
            flag = TT_UPPER
        elif best >= beta:
            flag = TT_LOWER
        self.tt[key] = TTEntry(depth=depth, score=best, flag=flag, best_move=best_move)
        return best

    def _store_killer(self, move: chess.Move, ply: int) -> None:
        bucket = self.killers[min(ply, MAX_PLY - 1)]
        if move in bucket:
            return
        bucket.insert(0, move)
        del bucket[2:]

    def _side_relative_eval(self, board: chess.Board) -> int:
        cp = self.evaluator.evaluate(board)
        return cp if board.turn == chess.WHITE else -cp
