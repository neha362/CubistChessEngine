"""
Search Agent — generic over MoveGen and Eval.

This is the strategic engine: given any move-gen and any evaluator, find the
best move. The search itself has NO personality — it just maximizes whatever
utility function the evaluator returns. That decoupling is what lets your
team mix and match: a Berserker eval + a normal search produces a Berserker
engine; same eval + a deeper search produces a smarter Berserker.

Implements:
  - Negamax with alpha-beta pruning
  - Iterative deepening (depth 1, 2, 3, ... until time)
  - Transposition table with EXACT/LOWER/UPPER bound flags
  - Quiescence search through captures (and checks, for Berserker — see note)
  - Mate-distance scoring (prefer Mate-in-2 over Mate-in-5)

Game-theory note: alpha-beta is provably equivalent to minimax — it returns
the same value, just visits fewer nodes. So the equilibrium found is purely
a function of the eval and depth, not of the search internals.

The ONE concession to personality is in quiescence: a normal q-search only
extends through captures, but for Berserker we ALSO extend through checks.
Without this, the engine miscounts attacks that resolve a few plies past the
horizon. This is the "give Berserker a chance to see its sacrifices through"
modification — toggleable via the `extend_checks` flag.
"""

import time
from typing import Optional, Tuple
import chess

MATE_SCORE = 100_000
INFINITY   = 1_000_000


class _TT:
    """Transposition table. Key is python-chess's Zobrist-stable position key."""
    EXACT, LOWER, UPPER = 0, 1, 2

    def __init__(self):
        self.table: dict = {}

    def store(self, key, depth, score, flag, move):
        # Replace-always policy. Sufficient for our scale; deeper engines use
        # bucket schemes (replace-by-depth, replace-by-age).
        self.table[key] = (depth, score, flag, move)

    def probe(self, key):
        return self.table.get(key)


class _Timeout(Exception):
    """Raised deep in search to unwind back to iterative-deepening control."""


class Search:
    """
    Implements the SearchAgent protocol. One instance per game (state lives
    in self.tt and is preserved across moves to benefit from prior analysis).
    """

    name = "AlphaBetaSearch"

    def __init__(self, extend_checks_in_qsearch: bool = True):
        self.tt = _TT()
        self.extend_checks = extend_checks_in_qsearch
        # Stats reset per search call.
        self.nodes = 0
        self.start_time = 0.0
        self.time_limit = 0.0

    # ------------------------------------------------------------------ public
    def find_best_move(
        self,
        board: chess.Board,
        move_gen,
        evaluator,
        time_limit: float = 3.0,
        max_depth: int = 64,
    ) -> Tuple[Optional[chess.Move], int]:
        """
        Iterative deepening main loop. Each completed depth gives us a better
        move; we return the deepest fully-completed result when time expires.
        """
        self.nodes = 0
        self.start_time = time.time()
        self.time_limit = time_limit

        best_move: Optional[chess.Move] = None
        best_score = 0

        for depth in range(1, max_depth + 1):
            try:
                score = self._negamax(
                    board, depth, -INFINITY, INFINITY, 0, move_gen, evaluator
                )
                # Pull the best move from the root TT entry.
                entry = self.tt.probe(board._transposition_key())
                if entry is not None and entry[3] is not None:
                    best_move = entry[3]
                    best_score = score
                self._emit_info(depth, best_score, best_move)
            except _Timeout:
                break
            # Found a mate? No need to search deeper.
            if abs(best_score) > MATE_SCORE - 1000:
                break

        return best_move, best_score

    # --------------------------------------------------------------- internals
    def _time_up(self) -> bool:
        # Check time every ~2048 nodes to avoid time.time() overhead per node.
        return (self.nodes & 2047) == 0 and (time.time() - self.start_time) >= self.time_limit

    def _negamax(self, board, depth, alpha, beta, ply, move_gen, evaluator):
        self.nodes += 1
        if self._time_up():
            raise _Timeout

        # --- Terminal / draw checks (before TT, so we never cache wrong values)
        if board.is_checkmate():
            # Side-to-move is mated. Subtract ply to prefer faster mates:
            # a mate-in-2 from root is worth more than a mate-in-5.
            return -MATE_SCORE + ply
        if (board.is_stalemate() or board.is_insufficient_material()
                or board.can_claim_threefold_repetition()):
            return 0

        alpha_orig = alpha
        key = board._transposition_key()

        # --- Transposition table probe ---
        # We disable TT cutoffs at the root (ply==0): we MUST search the root
        # to ensure the stored best_move is correct for THIS root position.
        # Otherwise we can return a move stored from a deeper transposition.
        tt_move = None
        entry = self.tt.probe(key)
        if entry is not None:
            tt_depth, tt_score, tt_flag, tt_move = entry
            if ply > 0 and tt_depth >= depth:
                if tt_flag == _TT.EXACT:
                    return tt_score
                if tt_flag == _TT.LOWER and tt_score > alpha:
                    alpha = tt_score
                elif tt_flag == _TT.UPPER and tt_score < beta:
                    beta = tt_score
                if alpha >= beta:
                    return tt_score

        # --- Leaf: drop into quiescence ---
        if depth <= 0:
            return self._quiesce(board, alpha, beta, move_gen, evaluator, qply=0)

        # --- Recursive search ---
        best_score = -INFINITY
        best_move = None

        for move in move_gen.ordered_moves(board, hint_move=tt_move):
            board.push(move)
            try:
                score = -self._negamax(
                    board, depth - 1, -beta, -alpha, ply + 1, move_gen, evaluator
                )
            finally:
                # Critical: pop even if a Timeout exception unwinds through us.
                # Without try/finally the board state corrupts on timeout.
                board.pop()

            if score > best_score:
                best_score = score
                best_move = move
            if score > alpha:
                alpha = score
            if alpha >= beta:
                break  # beta cutoff: opponent won't allow this line

        # --- Store result with the right bound flag ---
        if best_score <= alpha_orig:
            flag = _TT.UPPER          # we never raised alpha → upper bound
        elif best_score >= beta:
            flag = _TT.LOWER          # we caused a cutoff → lower bound
        else:
            flag = _TT.EXACT          # exact value found
        self.tt.store(key, depth, best_score, flag, best_move)

        return best_score

    def _quiesce(self, board, alpha, beta, move_gen, evaluator, qply):
        """
        Quiescence: only search 'noisy' moves until the position is quiet.
        Without this, the engine evaluates positions mid-trade ('horizon effect').

        For Berserker, 'noisy' = captures + (optionally) checks. Checks let the
        engine follow attacking sequences past the horizon, which is essential
        for sac-and-mate calculations.
        """
        self.nodes += 1
        if self._time_up():
            raise _Timeout

        # Hard depth cap on quiescence to prevent infinite check loops.
        if qply > 8:
            score = evaluator.evaluate(board)
            return score if board.turn == chess.WHITE else -score

        # Stand-pat: assume side-to-move can refuse to capture/check.
        # Negamax convention: score is from side-to-move's perspective.
        stand_pat = evaluator.evaluate(board)
        if board.turn == chess.BLACK:
            stand_pat = -stand_pat

        if stand_pat >= beta:
            return beta
        if stand_pat > alpha:
            alpha = stand_pat

        # Generate noisy moves only.
        noisy = []
        for m in board.legal_moves:
            if board.is_capture(m):
                noisy.append(m)
            elif self.extend_checks and qply < 4 and board.gives_check(m):
                # Limit check extensions to the first few qply to avoid runaway.
                noisy.append(m)

        # Re-use the move-gen's ordering by routing through ordered_moves...
        # except ordered_moves returns ALL moves. Simpler: order locally by
        # MVV-LVA-ish priority on this small subset.
        noisy.sort(
            key=lambda m: (
                board.is_capture(m) * 1000
                + (board.gives_check(m) * 500)
            ),
            reverse=True,
        )

        for move in noisy:
            board.push(move)
            try:
                score = -self._quiesce(board, -beta, -alpha, move_gen, evaluator, qply + 1)
            finally:
                board.pop()
            if score >= beta:
                return beta
            if score > alpha:
                alpha = score
        return alpha

    def _emit_info(self, depth, score, move):
        elapsed = max(time.time() - self.start_time, 1e-6)
        nps = int(self.nodes / elapsed)
        # UCI 'info' line — chess GUIs parse this to display search progress.
        print(
            f"info depth {depth} score cp {score} nodes {self.nodes} "
            f"nps {nps} time {int(elapsed*1000)} pv {move}",
            flush=True,
        )


default = Search()
