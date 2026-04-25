"""
Search Agent — Claude Oracle Chess Engine
Alpha-beta pruning with iterative deepening and a transposition table.
The eval function is stubbed out so this module can be tested independently.
Swap in eval_agent.py's Evaluator (or the Claude API oracle) at runtime.
"""

import math
import time
from typing import Callable, Optional
from move_gen_agent import BoardState, Move, MoveGenerator, parse_fen

# ─── Types ─────────────────────────────────────────────────────────────────────

# EvalFn takes a BoardState and returns a centipawn score (+ = white winning)
EvalFn = Callable[[BoardState], float]

# ─── Transposition table entry ─────────────────────────────────────────────────

TT_EXACT = 0   # exact score stored
TT_LOWER = 1   # alpha lower-bound (fail-high node)
TT_UPPER = 2   # beta upper-bound (fail-low node)

class TTEntry:
    __slots__ = ('depth', 'score', 'flag', 'best_move')
    def __init__(self, depth: int, score: float, flag: int, best_move: Optional[Move]):
        self.depth = depth
        self.score = score
        self.flag = flag
        self.best_move = best_move


# ─── Simple Zobrist-style hash (for TT keying) ─────────────────────────────────

import random
random.seed(42)

_PIECE_ORDER = ['P','N','B','R','Q','K','p','n','b','r','q','k']
_ZOBRIST_PIECE  = [[[random.getrandbits(64) for _ in range(12)] for _ in range(8)] for _ in range(8)]
_ZOBRIST_TURN   = random.getrandbits(64)
_ZOBRIST_CASTLE = [random.getrandbits(64) for _ in range(4)]  # K Q k q
_ZOBRIST_EP     = [random.getrandbits(64) for _ in range(8)]  # file 0-7

def zobrist_hash(state: BoardState) -> int:
    h = 0
    for r in range(8):
        for c in range(8):
            p = state.board[r][c]
            if p:
                idx = _PIECE_ORDER.index(p)
                h ^= _ZOBRIST_PIECE[r][c][idx]
    if state.turn == 'b':
        h ^= _ZOBRIST_TURN
    for i, ch in enumerate('KQkq'):
        if ch in state.castling:
            h ^= _ZOBRIST_CASTLE[i]
    if state.en_passant:
        h ^= _ZOBRIST_EP[state.en_passant[1]]
    return h


# ─── Move ordering ─────────────────────────────────────────────────────────────

_PIECE_VALUE = {'P':100,'N':320,'B':330,'R':500,'Q':900,'K':20000,
                'p':100,'n':320,'b':330,'r':500,'q':900,'k':20000}

def _mvv_lva(state: BoardState, move: Move) -> int:
    """Most Valuable Victim / Least Valuable Attacker heuristic for move ordering."""
    victim   = state.board[move.to_sq[0]][move.to_sq[1]]
    attacker = state.board[move.from_sq[0]][move.from_sq[1]]
    v = _PIECE_VALUE.get(victim.upper(), 0) * 10 - _PIECE_VALUE.get(attacker.upper(), 0)
    return v

def order_moves(state: BoardState, moves: list[Move], tt_move: Optional[Move]) -> list[Move]:
    """
    Move ordering priority:
      1. TT best move from previous iteration
      2. Captures (MVV-LVA)
      3. Promotions
      4. Quiet moves
    """
    def score(m: Move) -> int:
        if tt_move and m.uci() == tt_move.uci():
            return 100_000
        if m.promotion:
            return 50_000
        victim = state.board[m.to_sq[0]][m.to_sq[1]]
        if victim:
            return 10_000 + _mvv_lva(state, m)
        return 0

    return sorted(moves, key=score, reverse=True)


# ─── Search agent ──────────────────────────────────────────────────────────────

class SearchAgent:
    """
    Iterative deepening alpha-beta search.

    Usage:
        agent = SearchAgent(eval_fn=my_eval, max_depth=5, time_limit=5.0)
        best_move, score, pv = agent.search(state)

    The eval_fn can be:
        - eval_agent.Evaluator().evaluate  (fast, material + PST)
        - oracle_eval                       (calls Claude API — slow but creative)
        - lambda s: 0                       (stub for pure search testing)
    """

    def __init__(
        self,
        eval_fn:    EvalFn,
        max_depth:  int   = 6,
        time_limit: float = 10.0,   # seconds; 0 = unlimited
        tt_size:    int   = 1 << 20,  # ~1M entries
    ):
        self.eval_fn    = eval_fn
        self.max_depth  = max_depth
        self.time_limit = time_limit
        self.tt_size    = tt_size

        self._gen = MoveGenerator()
        self._tt: dict[int, TTEntry] = {}
        self._nodes = 0
        self._start_time = 0.0

    # ── Public API ─────────────────────────────────────────────────────────────

    def search(self, state: BoardState) -> tuple[Optional[Move], float, list[Move]]:
        """
        Run iterative deepening search.
        Returns (best_move, score_centipawns, principal_variation).
        Score is from white's perspective.
        """
        self._tt.clear()
        self._nodes = 0
        self._start_time = time.time()

        best_move: Optional[Move] = None
        best_score = -math.inf
        pv: list[Move] = []

        for depth in range(1, self.max_depth + 1):
            if self._time_up():
                break

            try:
                move, score, variation = self._root_search(state, depth)
            except _TimeOut:
                break

            if move is not None:
                best_move  = move
                best_score = score
                pv         = variation

            elapsed = time.time() - self._start_time
            print(f"  depth={depth:2d}  score={best_score:+.0f}cp  "
                  f"nodes={self._nodes:,}  time={elapsed:.2f}s  "
                  f"pv={' '.join(m.uci() for m in pv[:5])}")

        return best_move, best_score, pv

    # ── Root search ────────────────────────────────────────────────────────────

    def _root_search(
        self, state: BoardState, depth: int
    ) -> tuple[Optional[Move], float, list[Move]]:
        alpha = -math.inf
        beta  =  math.inf
        best_move: Optional[Move] = None
        best_score = -math.inf

        tt_key   = zobrist_hash(state)
        tt_entry = self._tt.get(tt_key)
        tt_move  = tt_entry.best_move if tt_entry else None

        moves = self._gen.legal_moves(state)
        if not moves:
            return None, 0.0, []

        moves = order_moves(state, moves, tt_move)

        for move in moves:
            if self._time_up():
                raise _TimeOut()

            child = self._gen.apply_move(state, move)
            child_pv: list = []
            score = -self._alpha_beta(child, depth - 1, -beta, -alpha, child_pv)
            # flip sign: alpha-beta is from mover's POV; we convert to white's POV
            if state.turn == 'b':
                score = -score

            if score > best_score:
                best_score = score
                best_move  = move
                alpha = max(alpha, score)

        pv = [best_move] if best_move else []
        return best_move, best_score, pv

    # ── Alpha-beta ─────────────────────────────────────────────────────────────

    def _alpha_beta(
        self,
        state: BoardState,
        depth: int,
        alpha: float,
        beta:  float,
        pv:    list,
    ) -> float:
        """
        Negamax alpha-beta. Returns score from the perspective of the side to move.
        `pv` is a mutable list used to record the current path (for the principal variation).
        """
        self._nodes += 1

        if self._time_up():
            raise _TimeOut()

        # TT lookup
        tt_key   = zobrist_hash(state)
        tt_entry = self._tt.get(tt_key)
        tt_move: Optional[Move] = None
        if tt_entry and tt_entry.depth >= depth:
            if tt_entry.flag == TT_EXACT:
                return tt_entry.score
            if tt_entry.flag == TT_LOWER:
                alpha = max(alpha, tt_entry.score)
            elif tt_entry.flag == TT_UPPER:
                beta  = min(beta, tt_entry.score)
            if alpha >= beta:
                return tt_entry.score
            tt_move = tt_entry.best_move

        # Terminal / leaf
        if self._gen.is_checkmate(state):
            return -100_000 + len(pv)   # prefer faster mates
        if self._gen.is_stalemate(state):
            return 0

        if depth == 0:
            return self._quiesce(state, alpha, beta)

        moves = self._gen.legal_moves(state)
        if not moves:
            return 0

        moves = order_moves(state, moves, tt_move)

        best_score = -math.inf
        best_move: Optional[Move] = None
        original_alpha = alpha

        for move in moves:
            child = self._gen.apply_move(state, move)
            child_pv: list = []
            score = -self._alpha_beta(child, depth - 1, -beta, -alpha, child_pv)

            if score > best_score:
                best_score = score
                best_move  = move
                pv.clear()
                pv.extend([move] + child_pv)

            alpha = max(alpha, score)
            if alpha >= beta:
                break   # beta cutoff

        # TT store
        flag = TT_EXACT
        if best_score <= original_alpha:
            flag = TT_UPPER
        elif best_score >= beta:
            flag = TT_LOWER
        if len(self._tt) < self.tt_size:
            self._tt[tt_key] = TTEntry(depth, best_score, flag, best_move)

        return best_score

    # ── Quiescence search ─────────────────────────────────────────────────────

    def _quiesce(self, state: BoardState, alpha: float, beta: float, depth: int = 0) -> float:
        """
        Search only captures to avoid the horizon effect.
        Falls back to static eval when no captures remain.
        """
        self._nodes += 1
        stand_pat = self._static_eval(state)

        if stand_pat >= beta:
            return beta
        alpha = max(alpha, stand_pat)

        if depth > 6:   # quiescence depth cap
            return alpha

        # Generate only capture moves
        captures = [
            m for m in self._gen.legal_moves(state)
            if state.board[m.to_sq[0]][m.to_sq[1]] or m.is_en_passant
        ]
        captures = order_moves(state, captures, None)

        for move in captures:
            child = self._gen.apply_move(state, move)
            score = -self._quiesce(child, -beta, -alpha, depth + 1)
            if score >= beta:
                return beta
            alpha = max(alpha, score)

        return alpha

    # ── Eval wrapper ──────────────────────────────────────────────────────────

    def _static_eval(self, state: BoardState) -> float:
        """
        Calls the injected eval function, then converts to the mover's perspective.
        The eval_fn always returns a score from white's perspective.
        """
        raw = self.eval_fn(state)
        return raw if state.turn == 'w' else -raw

    # ── Utilities ─────────────────────────────────────────────────────────────

    def _time_up(self) -> bool:
        if self.time_limit <= 0:
            return False
        return (time.time() - self._start_time) >= self.time_limit

    def stats(self) -> dict:
        return {
            'nodes': self._nodes,
            'tt_entries': len(self._tt),
            'elapsed': round(time.time() - self._start_time, 3),
        }


class _TimeOut(Exception):
    pass


# ─── Stub eval (for isolated search testing) ──────────────────────────────────

def stub_eval(state: BoardState) -> float:
    """
    Trivial material-only eval for testing the search in isolation.
    Swap out for eval_agent.Evaluator().evaluate or the Claude oracle.
    """
    values = {'P':100,'N':320,'B':330,'R':500,'Q':900,
              'p':-100,'n':-320,'b':-330,'r':-500,'q':-900}
    score = 0
    for row in state.board:
        for p in row:
            score += values.get(p, 0)
    return float(score)


# ─── Smoke tests ──────────────────────────────────────────────────────────────

if __name__ == '__main__':
    agent = SearchAgent(eval_fn=stub_eval, max_depth=5, time_limit=10.0)

    print("=== Test 1: Mate in 1 ===")
    # White to play, Qh5# is the only mate
    m1_fen = "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5Q2/PPPP1PPP/RNB1K1NR w KQkq - 4 4"
    state1 = parse_fen(m1_fen)
    move, score, pv = agent.search(state1)
    print(f"Best move: {move}  score: {score:+.0f}cp")
    print(f"PV: {' '.join(m.uci() for m in pv)}")

    print("\n=== Test 2: Starting position depth-4 search ===")
    state2 = parse_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
    agent2 = SearchAgent(eval_fn=stub_eval, max_depth=4, time_limit=15.0)
    move2, score2, pv2 = agent2.search(state2)
    print(f"Best move: {move2}  score: {score2:+.0f}cp")

    print("\n=== Test 3: Transposition table populated ===")
    print(f"TT entries after search: {len(agent2._tt):,}")
    print(f"Nodes searched: {agent2._nodes:,}")