"""
red_blue.py — Red vs Blue Zero-Sum Chess Engine Evaluator
==========================================================
Scenario 2: Red vs Blue (zero-sum game).

HOW IT WORKS
────────────
The attacker (Red) earns points by finding positions where engines fail.
The defender (Blue) earns points by validating correct responses.
Every point Red earns is subtracted from that engine's defender score.
The attacker has a win condition: find a failing position.
This mirrors alpha-beta search — the minimizing player hunts refutations.

FAILURE TYPES AND POINTS
─────────────────────────
  illegal_move  3 pts  — engine returned a move not in board.legal_moves
  crash         3 pts  — engine raised an exception or returned None
  timeout       2 pts  — engine exceeded TIME_LIMIT seconds
  blunder       2 pts  — engine played into mate-in-1 when not forced
  eval_wrong    1 pt   — engine eval outside expected centipawn range

OUTPUTS
───────
  Terminal table — ranked engine scores with reliability %
  Markdown file  — saved to scenarios/red_blue_results/tournament_TIMESTAMP.md

Usage:
  python scenarios/red_blue.py                              # full tournament
  python scenarios/red_blue.py --engines classical_minimax  # one engine
  python scenarios/red_blue.py --category tactical          # one category
  python scenarios/red_blue.py --head-to-head classical_minimax monte_carlo
  python scenarios/red_blue.py --test                       # self-test suite
  python scenarios/red_blue.py --simulate                   # quick gauntlet
"""

from __future__ import annotations

import chess
import time
import json
import os
import sys
import argparse
from dataclasses import dataclass
from datetime import datetime
from abc import ABC, abstractmethod

# ═══════════════════════════════════════════════════════════════════════════════
# IMPORTS AND CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

FAILURE_POINTS = {
    "illegal_move": 3,
    "crash":        3,
    "timeout":      2,
    "blunder":      2,
    "eval_wrong":   1,
}
TIME_LIMIT    = 5.0
DEFAULT_DEPTH = 4


def _repo_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ═══════════════════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class TestPosition:
    fen:                  str
    name:                 str
    category:             str           # opening|tactical|material|endgame|edge_case
    expected_best_moves:  list[str]     # UCI strings
    expected_eval_range:  tuple[int, int]
    difficulty:           int           # 1 easy → 5 hard


POSITIONS: list[TestPosition] = [
    # Opening (4)
    TestPosition(
        fen="rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        name="Starting position", category="opening",
        expected_best_moves=["e2e4", "d2d4", "g1f3", "c2c4"],
        expected_eval_range=(-50, 50), difficulty=1),
    TestPosition(
        fen="rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1",
        name="After 1.e4", category="opening",
        expected_best_moves=["e7e5", "c7c5", "e7e6"],
        expected_eval_range=(-30, 50), difficulty=1),
    TestPosition(
        fen="r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3",
        name="After 1.e4 e5 2.Nf3 Nc6", category="opening",
        expected_best_moves=["f1b5", "f1c4", "d2d4"],
        expected_eval_range=(-30, 50), difficulty=1),
    TestPosition(
        fen="rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq c6 0 2",
        name="Sicilian after 1.e4 c5", category="opening",
        expected_best_moves=["g1f3", "d2d4", "b1c3"],
        expected_eval_range=(-30, 60), difficulty=1),

    # Tactical (4)
    TestPosition(
        fen="r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/8/PPPP1PPP/RNBQK1NR w KQkq - 4 4",
        name="Scholar's mate threat", category="tactical",
        expected_best_moves=["d1h5"],
        expected_eval_range=(100, 900), difficulty=2),
    TestPosition(
        fen="6k1/5ppp/8/8/8/8/8/3R2K1 w - - 0 1",
        name="Back-rank mate in 1", category="tactical",
        expected_best_moves=["d1d8"],
        expected_eval_range=(9000, 100000), difficulty=2),
    TestPosition(
        fen="r1bq1rk1/pppp1ppp/2n2n2/2b1p3/2B1P3/2N2N2/PPPP1PPP/R1BQK2R w KQ - 6 6",
        name="Pin winning material", category="tactical",
        expected_best_moves=["d1e2", "d2d3"],
        expected_eval_range=(50, 400), difficulty=3),
    TestPosition(
        fen="r1bqkb1r/ppp2ppp/2n5/3np3/2B5/5N2/PPPP1PPP/RNBQK2R w KQkq - 0 6",
        name="Knight fork opportunity", category="tactical",
        expected_best_moves=["f3e5", "d1f3"],
        expected_eval_range=(50, 500), difficulty=3),

    # Material (4)
    TestPosition(
        fen="rnb1kbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        name="Queen up for white", category="material",
        expected_best_moves=["e2e4", "d2d4"],
        expected_eval_range=(800, 1100), difficulty=1),
    TestPosition(
        fen="4k3/8/8/8/8/8/8/RR2K3 w - - 0 1",
        name="Two rooks vs lone king", category="material",
        expected_best_moves=["b1b7", "a1a7", "b1b8"],
        expected_eval_range=(5000, 100000), difficulty=1),
    TestPosition(
        fen="4k3/8/8/3P4/8/8/8/4K3 w - - 0 1",
        name="Pawn majority endgame", category="material",
        expected_best_moves=["d5d6", "e1d2"],
        expected_eval_range=(100, 600), difficulty=2),
    TestPosition(
        fen="r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3",
        name="Rook vs minor piece imbalance", category="material",
        expected_best_moves=["f1b5", "d2d4"],
        expected_eval_range=(-50, 200), difficulty=2),

    # Endgame (4)
    TestPosition(
        fen="8/8/8/4k3/8/4K3/4P3/8 w - - 0 1",
        name="KP vs K winning", category="endgame",
        expected_best_moves=["e3d4", "e3f4", "e2e4"],
        expected_eval_range=(200, 800), difficulty=3),
    TestPosition(
        fen="8/8/8/8/8/4k3/8/4K2Q w - - 0 1",
        name="KQ vs K mate", category="endgame",
        expected_best_moves=["h1e4", "h1h8", "h1a1"],
        expected_eval_range=(9000, 100000), difficulty=1),
    TestPosition(
        fen="8/8/8/3k4/8/3K4/3R4/8 w - - 0 1",
        name="Rook endgame with passed pawn", category="endgame",
        expected_best_moves=["d2d5", "d3e4"],
        expected_eval_range=(300, 900), difficulty=3),
    TestPosition(
        fen="8/8/p7/8/8/P7/8/8 w - - 0 1",
        name="Zugzwang position", category="edge_case",
        expected_best_moves=["a3a4"],
        expected_eval_range=(-200, 200), difficulty=5),

    # Edge cases (4)
    TestPosition(
        fen="rnbqkbnr/ppp1pppp/8/3pP3/8/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 3",
        name="En passant available", category="edge_case",
        expected_best_moves=["e5d6"],
        expected_eval_range=(-50, 200), difficulty=2),
    TestPosition(
        fen="r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w KQkq - 0 1",
        name="Castling rights present", category="edge_case",
        expected_best_moves=["e1g1", "e1c1"],
        expected_eval_range=(-50, 50), difficulty=2),
    TestPosition(
        fen="rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 95 1",
        name="Fifty-move rule approaching", category="edge_case",
        expected_best_moves=["e2e4", "d2d4", "g1f3"],
        expected_eval_range=(-50, 50), difficulty=2),
    TestPosition(
        fen="6k1/5ppp/8/8/8/8/5PPP/6K1 w - - 0 1",
        name="Symmetric fortress", category="edge_case",
        expected_best_moves=["f2f3", "g2g3", "h2h3"],
        expected_eval_range=(-100, 100), difficulty=3),
]


@dataclass
class ProbeResult:
    engine_name:    str
    position_name:  str
    category:       str
    move_returned:  str | None
    eval_returned:  int | None
    is_legal:       bool
    is_blunder:     bool
    eval_error:     int          # 0 if in range, else abs diff from nearest bound
    time_taken:     float
    failure_reason: str | None   # key in FAILURE_POINTS or None
    attacker_points: int


# ═══════════════════════════════════════════════════════════════════════════════
# TIMING / BLUNDER HELPERS  (used by RedAgent)
# ═══════════════════════════════════════════════════════════════════════════════

_TIMEOUT = object()


def _timed_call(fn, limit):
    """Call fn(), return _TIMEOUT sentinel if it exceeds limit seconds."""
    import threading

    result = [_TIMEOUT]

    def target():
        try:
            result[0] = fn()
        except Exception:
            result[0] = None

    t = threading.Thread(target=target, daemon=True)
    t.start()
    t.join(timeout=limit)
    return result[0]


def _is_blunder(board: chess.Board, move: chess.Move) -> bool:
    """Return True if playing move leaves the engine in mate-in-1."""
    b2 = board.copy()
    b2.push(move)
    if b2.is_game_over():
        return False
    for opp_move in b2.legal_moves:
        b3 = b2.copy()
        b3.push(opp_move)
        if b3.is_checkmate():
            return True
    return False


# ═══════════════════════════════════════════════════════════════════════════════
# UNIFIED ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

class UnifiedEngine(ABC):
    name: str = "base"

    @abstractmethod
    def get_move(self, board: chess.Board,
                 depth: int = DEFAULT_DEPTH) -> chess.Move | None:
        ...

    @abstractmethod
    def get_eval(self, board: chess.Board) -> int | None:
        ...

    def shutdown(self) -> None:
        pass


# ═══════════════════════════════════════════════════════════════════════════════
# ENGINE ADAPTERS
# ═══════════════════════════════════════════════════════════════════════════════

try:
    _cm = os.path.join(_repo_root(), "classical_minimax")
    if _cm not in sys.path:
        sys.path.insert(0, _cm)
    from chess_engine.search import SearchAgent
    from chess_engine.eval import EvalAgent

    class ClassicalMinimaxAdapter(UnifiedEngine):
        name = "classical_minimax"

        def __init__(self):
            self._search = SearchAgent(EvalAgent().evaluate)
            self._eval = EvalAgent()

        def get_move(self, board, depth=DEFAULT_DEPTH):
            try:
                return self._search.best_move(board, depth)
            except Exception:
                return None

        def get_eval(self, board):
            try:
                return self._eval.evaluate(board)
            except Exception:
                return None
except ImportError:
    class ClassicalMinimaxAdapter(UnifiedEngine):
        name = "classical_minimax"

        def get_move(self, board, depth=DEFAULT_DEPTH):
            return None

        def get_eval(self, board):
            return None


try:
    _ca = os.path.join(_repo_root(), "claude_api")
    if _ca not in sys.path:
        sys.path.insert(0, _ca)
    from search_engine import SearchAgent as ClaudeSearchAgent
    from eval_engine import Evaluator as ClaudeEvaluator
    from move_engine import parse_fen

    class ClaudeApiAdapter(UnifiedEngine):
        name = "claude_api"

        def __init__(self):
            self._eval_obj = ClaudeEvaluator()
            self._search = ClaudeSearchAgent(
                eval_fn=self._eval_obj.evaluate,
                max_depth=DEFAULT_DEPTH,
                time_limit=TIME_LIMIT,
            )

        def get_move(self, board, depth=DEFAULT_DEPTH):
            try:
                self._search.max_depth = max(1, depth)
                state = parse_fen(board.fen())
                move, _score, _pv = self._search.search(state)
                if move is None:
                    return None
                return chess.Move.from_uci(move.uci())
            except Exception:
                return None

        def get_eval(self, board):
            try:
                state = parse_fen(board.fen())
                return int(self._eval_obj.evaluate(state))
            except Exception:
                return None
except ImportError:
    class ClaudeApiAdapter(UnifiedEngine):
        name = "claude_api"

        def get_move(self, board, depth=DEFAULT_DEPTH):
            return None

        def get_eval(self, board):
            return None


try:
    _mc = os.path.join(_repo_root(), "monte_carlo")
    if _mc not in sys.path:
        sys.path.insert(0, _mc)
    from movegen_agent import from_fen
    from mcts_agent import mcts_search

    def _mcts_material_cp(board: chess.Board) -> int:
        vals = {
            chess.PAWN: 100, chess.KNIGHT: 320, chess.BISHOP: 330,
            chess.ROOK: 500, chess.QUEEN: 900, chess.KING: 0,
        }
        s = 0
        for sq in chess.SQUARES:
            p = board.piece_at(sq)
            if p:
                v = vals[p.piece_type]
                s += v if p.color == chess.WHITE else -v
        return s

    class MonteCarloAdapter(UnifiedEngine):
        name = "monte_carlo"

        def get_move(self, board, depth=DEFAULT_DEPTH):
            try:
                state = from_fen(board.fen())
                if depth <= 1:
                    budget = 80
                elif depth <= 2:
                    budget = 200
                else:
                    budget = max(200, depth * 250)
                r = mcts_search(
                    state,
                    max_iter=budget,
                    movetime_ms=None,
                    verbose=False,
                )
                if not r.best_move:
                    return None
                uci = r.best_uci
                if uci == "0000":
                    return None
                return chess.Move.from_uci(uci)
            except Exception:
                return None

        def get_eval(self, board):
            try:
                return _mcts_material_cp(board)
            except Exception:
                return None
except ImportError:
    class MonteCarloAdapter(UnifiedEngine):
        name = "monte_carlo"

        def get_move(self, board, depth=DEFAULT_DEPTH):
            return None

        def get_eval(self, board):
            return None


class MockNNAdapter(UnifiedEngine):
    """Stub NN — mock_nn/ has no Python modules in this repo."""

    name = "mock_nn"

    def get_move(self, board, depth=DEFAULT_DEPTH):
        try:
            if board.legal_moves:
                return next(iter(board.legal_moves))
        except Exception:
            pass
        return None

    def get_eval(self, board):
        return 0


try:
    _b1 = os.path.join(_repo_root(), "berserker1")
    if _b1 not in sys.path:
        sys.path.insert(0, _b1)
    from movegen_agent import from_fen as b1_from_fen
    from berserker_search_agent import search as berserker1_search
    from berserker_eval_agent import evaluate as berserker1_evaluate

    class Berserker1Adapter(UnifiedEngine):
        name = "berserker_1"

        def get_move(self, board, depth=DEFAULT_DEPTH):
            try:
                state = b1_from_fen(board.fen())
                r = berserker1_search(state, max_depth=max(1, depth),
                                      movetime_ms=None, verbose=False)
                if not r.move:
                    return None
                return chess.Move.from_uci(r.uci)
            except Exception:
                return None

        def get_eval(self, board):
            try:
                return int(berserker1_evaluate(b1_from_fen(board.fen())))
            except Exception:
                return None
except ImportError:
    class Berserker1Adapter(UnifiedEngine):
        name = "berserker_1"

        def get_move(self, board, depth=DEFAULT_DEPTH):
            return None

        def get_eval(self, board):
            return None


try:
    from berserker_2.search import Search as Berserker2Search
    from berserker_2.move_gen import MoveGen as Berserker2MoveGen
    from berserker_2.eval import Evaluator as Berserker2Evaluator

    class Berserker2Adapter(UnifiedEngine):
        name = "berserker_2"

        def __init__(self):
            self._search = Berserker2Search()
            self._move_gen = Berserker2MoveGen()
            self._eval = Berserker2Evaluator()

        def get_move(self, board, depth=DEFAULT_DEPTH):
            try:
                b = board.copy()
                mv, _sc = self._search.find_best_move(
                    b, self._move_gen, self._eval,
                    time_limit=min(TIME_LIMIT, 3.0),
                    max_depth=max(1, depth),
                )
                return mv
            except Exception:
                return None

        def get_eval(self, board):
            try:
                return int(self._eval.evaluate(board))
            except Exception:
                return None
except ImportError:
    class Berserker2Adapter(UnifiedEngine):
        name = "berserker_2"

        def get_move(self, board, depth=DEFAULT_DEPTH):
            return None

        def get_eval(self, board):
            return None


# ═══════════════════════════════════════════════════════════════════════════════
# FACTORY
# ═══════════════════════════════════════════════════════════════════════════════

def build_engines(names: list[str] | None = None) -> list[UnifiedEngine]:
    """
    Instantiate all adapters. Skip any that fail. Log why.
    Filter by names if provided.
    """
    candidates = [
        ClassicalMinimaxAdapter,
        ClaudeApiAdapter,
        MonteCarloAdapter,
        MockNNAdapter,
        Berserker1Adapter,
        Berserker2Adapter,
    ]
    engines = []
    for cls in candidates:
        if names and cls.name not in names:
            continue
        try:
            e = cls()
            engines.append(e)
            print(f"  [OK]   {cls.name}")
        except Exception as ex:
            print(f"  [SKIP] {cls.name}: {ex}")
    return engines


# ═══════════════════════════════════════════════════════════════════════════════
# RED AGENT
# ═══════════════════════════════════════════════════════════════════════════════

class RedAgent:
    def __init__(self, time_limit: float = TIME_LIMIT, depth: int = DEFAULT_DEPTH):
        self.time_limit = time_limit
        self.depth = depth

    def probe(self, engine: UnifiedEngine, pos: TestPosition) -> ProbeResult:
        board = chess.Board(fen=pos.fen)
        move = None
        evl = None
        t0 = time.time()
        reason = None
        legal = False
        blunder = False
        eval_error = 0

        try:
            result = _timed_call(
                lambda: engine.get_move(board, self.depth), self.time_limit)
            if result is _TIMEOUT:
                reason = "timeout"
            elif result is None:
                reason = "crash"
            else:
                move = result
        except Exception:
            reason = "crash"

        elapsed = time.time() - t0

        if reason is None and move is not None:
            legal = move in board.legal_moves
            if not legal:
                reason = "illegal_move"

        if reason is None and legal and move is not None:
            blunder = _is_blunder(board, move)
            if blunder:
                reason = "blunder"

        if reason is None:
            try:
                evl = engine.get_eval(board)
                if evl is not None:
                    lo, hi = pos.expected_eval_range
                    if lo <= evl <= hi:
                        eval_error = 0
                    else:
                        eval_error = min(abs(evl - lo), abs(evl - hi))
                        if eval_error > 500:
                            reason = "eval_wrong"
                else:
                    eval_error = 0
            except Exception:
                eval_error = 0

        pts = FAILURE_POINTS.get(reason, 0)
        return ProbeResult(
            engine_name=engine.name,
            position_name=pos.name,
            category=pos.category,
            move_returned=move.uci() if move else None,
            eval_returned=evl,
            is_legal=legal,
            is_blunder=blunder,
            eval_error=eval_error,
            time_taken=round(elapsed, 3),
            failure_reason=reason,
            attacker_points=pts,
        )

    def run_gauntlet(self, engine: UnifiedEngine,
                     positions: list[TestPosition]) -> list[ProbeResult]:
        return [self.probe(engine, pos) for pos in positions]

    def find_worst(self, results: list[ProbeResult]) -> ProbeResult | None:
        if not results:
            return None
        return max(results, key=lambda r: (r.attacker_points, r.time_taken))


# ═══════════════════════════════════════════════════════════════════════════════
# BLUE AGENT
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class ConsensusResult:
    position_name:    str
    majority_move:    str | None
    agreeing_engines: list[str]
    outlier_engines:  list[str]
    outlier_flagged:  bool


class BlueAgent:
    def validate_move(self, board: chess.Board,
                      move: chess.Move | None) -> bool:
        if move is None:
            return False
        return move in board.legal_moves

    def cross_validate(self, engines: list[UnifiedEngine],
                       pos: TestPosition) -> ConsensusResult:
        board = chess.Board(fen=pos.fen)
        votes: dict[str, list[str]] = {}
        # Shallow depth — consensus is many calls per tournament; full depth
        # would make MCTS-style engines impractically slow here.
        consensus_depth = min(2, DEFAULT_DEPTH)
        for e in engines:
            try:
                m = e.get_move(board, consensus_depth)
                uci = m.uci() if m and m in board.legal_moves else "__invalid__"
            except Exception:
                uci = "__crash__"
            votes.setdefault(uci, []).append(e.name)

        majority_move = max(votes, key=lambda k: len(votes[k]))
        agreeing = votes[majority_move]
        outliers = [
            n for k, v in votes.items()
            if k != majority_move for n in v
        ]
        flagged = len(agreeing) >= 4 and len(outliers) > 0
        return ConsensusResult(
            position_name=pos.name,
            majority_move=(majority_move if majority_move not in
                           ("__invalid__", "__crash__") else None),
            agreeing_engines=agreeing,
            outlier_engines=outliers,
            outlier_flagged=flagged,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# SCORER
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class EngineScore:
    engine_name:              str
    total_positions:          int
    attacker_points_conceded: int
    defender_points:          int
    failures_by_category:     dict[str, int]
    failures_by_type:         dict[str, int]
    worst_category:           str
    reliability_pct:          float


class Scorer:
    def compute(self, engine_name: str,
                results: list[ProbeResult]) -> EngineScore:
        total = len(results)
        att = sum(r.attacker_points for r in results)
        by_cat: dict[str, int] = {}
        by_type: dict[str, int] = {}
        for r in results:
            if r.failure_reason:
                by_cat[r.category] = by_cat.get(r.category, 0) + 1
                by_type[r.failure_reason] = by_type.get(r.failure_reason, 0) + 1
        worst = max(by_cat, key=by_cat.get) if by_cat else "none"
        clean = sum(1 for r in results if r.failure_reason is None)
        return EngineScore(
            engine_name=engine_name,
            total_positions=total,
            attacker_points_conceded=att,
            defender_points=total - att,
            failures_by_category=by_cat,
            failures_by_type=by_type,
            worst_category=worst,
            reliability_pct=round(clean / total * 100, 1) if total else 0.0,
        )

    def rank(self, scores: list[EngineScore]) -> list[EngineScore]:
        return sorted(
            scores,
            key=lambda s: (s.defender_points, s.reliability_pct),
            reverse=True,
        )

    def head_to_head(self, a: str, b: str,
                     all_results: list[ProbeResult]) -> dict:
        ra = [r for r in all_results if r.engine_name == a]
        rb = [r for r in all_results if r.engine_name == b]
        by_pos_a = {r.position_name: r for r in ra}
        by_pos_b = {r.position_name: r for r in rb}
        shared = set(by_pos_a) & set(by_pos_b)
        wins_a = sum(
            1 for p in shared
            if by_pos_a[p].attacker_points < by_pos_b[p].attacker_points
        )
        wins_b = sum(
            1 for p in shared
            if by_pos_b[p].attacker_points < by_pos_a[p].attacker_points
        )
        return {
            "engine_a": a, "engine_b": b,
            "wins_a": wins_a, "wins_b": wins_b,
            "shared_positions": len(shared),
        }


# ═══════════════════════════════════════════════════════════════════════════════
# REPORT
# ═══════════════════════════════════════════════════════════════════════════════

class Report:
    def print_table(self, ranked: list[EngineScore]) -> None:
        """Print the same ╔══╗ box style used in auction_house.py summary()"""
        w = 64
        print("╔" + "═" * w + "╗")
        print(
            f"║  {'Engine':<22} {'Score':>7} {'Reliable':>9} "
            f"{'Illegal':>8} {'Blunders':>9} {'Worst cat.':<12}  ║"
        )
        print("╠" + "═" * w + "╣")
        total = ranked[0].total_positions if ranked else 0
        for s in ranked:
            ill = (s.failures_by_type.get("illegal_move", 0) +
                   s.failures_by_type.get("crash", 0))
            blu = s.failures_by_type.get("blunder", 0)
            print(
                f"║  {s.engine_name:<22} "
                f"{s.defender_points:>3}/{total:<3} "
                f"{s.reliability_pct:>8.1f}% "
                f"{ill:>8} "
                f"{blu:>9} "
                f"{s.worst_category:<12}  ║"
            )
        print("╚" + "═" * w + "╝")

    def write_markdown(self, ranked: list[EngineScore],
                       all_results: list[ProbeResult],
                       consensus: list[ConsensusResult]) -> str:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        folder = os.path.join(os.path.dirname(__file__), "red_blue_results")
        os.makedirs(folder, exist_ok=True)
        path = os.path.join(folder, f"tournament_{ts}.md")

        scorer = Scorer()
        lines = [
            f"# Red-Blue Tournament — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## Results",
            "",
            "| Engine | Score | Reliable | Illegal/Crash | Blunders | Worst category |",
            "|--------|-------|----------|---------------|----------|----------------|",
        ]
        total = ranked[0].total_positions if ranked else 0
        for s in ranked:
            ill = (s.failures_by_type.get("illegal_move", 0) +
                   s.failures_by_type.get("crash", 0))
            blu = s.failures_by_type.get("blunder", 0)
            lines.append(
                f"| {s.engine_name} | {s.defender_points}/{total} | "
                f"{s.reliability_pct}% | {ill} | {blu} | {s.worst_category} |"
            )

        lines += ["", "## Per-engine breakdown by category", ""]
        for s in ranked:
            lines.append(f"### {s.engine_name}")
            for cat, n in sorted(s.failures_by_category.items()):
                lines.append(f"- {cat}: {n} failure(s)")
            lines.append("")

        names = [s.engine_name for s in ranked]
        lines += ["## Head-to-head matrix", ""]
        header = "| |" + "|".join(names) + "|"
        lines.append(header)
        lines.append("|---|" + "---|" * len(names))
        for a in names:
            row = f"| {a} |"
            for b in names:
                if a == b:
                    row += " — |"
                else:
                    h2h = scorer.head_to_head(a, b, all_results)
                    row += f" {h2h['wins_a']}-{h2h['wins_b']} |"
            lines.append(row)

        pos_failures: dict[str, int] = {}
        for r in all_results:
            if r.failure_reason:
                pos_failures[r.position_name] = pos_failures.get(r.position_name, 0) + 1
        top3 = sorted(pos_failures, key=pos_failures.get, reverse=True)[:3]
        lines += ["", "## Top 3 killer positions (most engines failed)", ""]
        for p in top3:
            lines.append(f"- **{p}** — {pos_failures[p]} engine(s) failed")

        flagged = [c for c in consensus if c.outlier_flagged]
        lines += ["", "## Consensus analysis (positions with most disagreement)", ""]
        for c in flagged:
            lines.append(
                f"- **{c.position_name}**: majority={c.majority_move}, "
                f"outliers={c.outlier_engines}"
            )

        with open(path, "w") as f:
            f.write("\n".join(lines))
        return path


# ═══════════════════════════════════════════════════════════════════════════════
# TOURNAMENT
# ═══════════════════════════════════════════════════════════════════════════════

def run_tournament(engine_names: list[str] | None = None,
                   category: str | None = None,
                   depth: int = DEFAULT_DEPTH,
                   head_to_head_pair: tuple[str, str] | None = None) -> int:
    print("\nRed-Blue Chess Engine Tournament")
    print("=" * 62)
    print("Loading engines...")
    engines = build_engines(engine_names)
    if not engines:
        print("No engines loaded. Exiting.")
        return 1

    positions = [p for p in POSITIONS
                 if category is None or p.category == category]
    print(f"\nRunning {len(positions)} positions against "
          f"{len(engines)} engine(s)...\n")

    red = RedAgent(depth=depth)
    blue = BlueAgent()
    scorer = Scorer()
    report = Report()

    all_results: list[ProbeResult] = []
    all_scores: list[EngineScore] = []

    for engine in engines:
        print(f"  Probing {engine.name}...")
        results = red.run_gauntlet(engine, positions)
        all_results.extend(results)
        score = scorer.compute(engine.name, results)
        all_scores.append(score)
        worst = red.find_worst(results)
        if worst and worst.failure_reason:
            print(f"    Worst: {worst.position_name} "
                  f"({worst.failure_reason}, {worst.attacker_points}pts)")

    ranked = scorer.rank(all_scores)

    consensus: list[ConsensusResult] = []
    for pos in positions:
        c = blue.cross_validate(engines, pos)
        if c.outlier_flagged:
            consensus.append(c)

    print()
    report.print_table(ranked)
    md_path = report.write_markdown(ranked, all_results, consensus)
    print(f"\nReport saved to: {md_path}")

    if head_to_head_pair:
        a, b = head_to_head_pair
        h2h = scorer.head_to_head(a, b, all_results)
        print(f"\nHead-to-head: {a} vs {b}")
        print(f"  {a} wins: {h2h['wins_a']}")
        print(f"  {b} wins: {h2h['wins_b']}")
        print(f"  Shared positions: {h2h['shared_positions']}")

    had_serious = any(
        r.failure_reason in ("illegal_move", "crash")
        for r in all_results
    )
    return 1 if had_serious else 0


# ═══════════════════════════════════════════════════════════════════════════════
# SELF-TEST
# ═══════════════════════════════════════════════════════════════════════════════

def _run_tests() -> bool:
    print("red_blue.py — self-test")
    print("=" * 62)
    ok = True
    start = next(p for p in POSITIONS if p.name == "Starting position")
    board = chess.Board()

    class _GoodEngine(UnifiedEngine):
        name = "mock_good"

        def get_move(self, b, depth=DEFAULT_DEPTH):
            return chess.Move.from_uci("e2e4")

        def get_eval(self, b):
            return 0

    class _CrashEngine(UnifiedEngine):
        name = "mock_crash"

        def get_move(self, b, depth=DEFAULT_DEPTH):
            return None

        def get_eval(self, b):
            return None

    class _IllegalEngine(UnifiedEngine):
        name = "mock_illegal"

        def get_move(self, b, depth=DEFAULT_DEPTH):
            return chess.Move.from_uci("h7h8")

        def get_eval(self, b):
            return None

    class _SlowEngine(UnifiedEngine):
        name = "mock_slow"

        def get_move(self, b, depth=DEFAULT_DEPTH):
            time.sleep(10.0)
            return chess.Move.from_uci("e2e4")

        def get_eval(self, b):
            return 0

    red = RedAgent()
    # T1
    r1 = red.probe(_GoodEngine(), start)
    t1 = r1.attacker_points == 0
    ok &= t1
    print(f"  [{'PASS' if t1 else 'FAIL'}] T1  legal move → 0 attacker pts")

    # T2
    r2 = red.probe(_CrashEngine(), start)
    t2 = r2.attacker_points == 3 and r2.failure_reason == "crash"
    ok &= t2
    print(f"  [{'PASS' if t2 else 'FAIL'}] T2  None move → crash / 3 pts")

    # T3
    r3 = red.probe(_IllegalEngine(), start)
    t3 = r3.attacker_points == 3 and r3.failure_reason == "illegal_move"
    ok &= t3
    print(f"  [{'PASS' if t3 else 'FAIL'}] T3  illegal move → 3 pts")

    # T4
    red_slow = RedAgent(time_limit=0.15, depth=DEFAULT_DEPTH)
    r4 = red_slow.probe(_SlowEngine(), start)
    t4 = r4.attacker_points == 2 and r4.failure_reason == "timeout"
    ok &= t4
    print(f"  [{'PASS' if t4 else 'FAIL'}] T4  timeout → 2 pts")

    # T5
    t5 = not _is_blunder(board, chess.Move.from_uci("e2e4"))
    ok &= t5
    print(f"  [{'PASS' if t5 else 'FAIL'}] T5  no blunder from startpos e2e4")

    # T6
    t6 = _timed_call(lambda: time.sleep(1.0), 0.05) is _TIMEOUT
    ok &= t6
    print(f"  [{'PASS' if t6 else 'FAIL'}] T6  _timed_call timeout sentinel")

    # T7
    blue = BlueAgent()
    t7 = not blue.validate_move(board, None)
    ok &= t7
    print(f"  [{'PASS' if t7 else 'FAIL'}] T7  validate None → False")

    # T8
    t8 = blue.validate_move(board, chess.Move.from_uci("e2e4"))
    ok &= t8
    print(f"  [{'PASS' if t8 else 'FAIL'}] T8  validate e2e4 → True")

    # T9
    scorer = Scorer()
    fake = [
        ProbeResult("x", "p", "opening", "e2e4", 0, True, False, 0, 0.1, None, 0),
        ProbeResult("x", "q", "opening", None, None, False, False, 0, 0.1, "crash", 3),
    ]
    es = scorer.compute("x", fake)
    t9 = es.defender_points == es.total_positions - es.attacker_points_conceded
    ok &= t9
    print(f"  [{'PASS' if t9 else 'FAIL'}] T9  defender_pts = total − attacker_pts")

    print("=" * 62)
    print("All tests PASSED" if ok else "Some tests FAILED")
    return ok


# ═══════════════════════════════════════════════════════════════════════════════
# SIMULATION
# ═══════════════════════════════════════════════════════════════════════════════

def _simulate():
    """Quick gauntlet: all engines, opening positions only. Print table."""
    print("\nQuick simulate — opening positions only")
    print("─" * 62)
    engines = build_engines()
    positions = [p for p in POSITIONS if p.category == "opening"]
    red = RedAgent()
    scorer = Scorer()
    report = Report()
    all_scores = []
    for e in engines:
        results = red.run_gauntlet(e, positions)
        all_scores.append(scorer.compute(e.name, results))
    report.print_table(scorer.rank(all_scores))


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(description="Red-Blue Chess Engine Tournament")
    ap.add_argument(
        "--engines", nargs="+",
        choices=[
            "classical_minimax", "claude_api", "monte_carlo",
            "mock_nn", "berserker_1", "berserker_2",
        ],
        help="Engines to include (default: all)",
    )
    ap.add_argument(
        "--category",
        choices=["opening", "tactical", "material", "endgame", "edge_case"],
        help="Only test this position category",
    )
    ap.add_argument("--depth", type=int, default=DEFAULT_DEPTH)
    ap.add_argument("--head-to-head", nargs=2, metavar="ENGINE")
    ap.add_argument("--test", action="store_true")
    ap.add_argument("--simulate", action="store_true")
    args = ap.parse_args()

    if args.test:
        sys.exit(0 if _run_tests() else 1)
    if args.simulate:
        _simulate()
        return

    code = run_tournament(
        engine_names=args.engines,
        category=args.category,
        depth=args.depth,
        head_to_head_pair=tuple(args.head_to_head) if args.head_to_head else None,
    )
    sys.exit(code)


if __name__ == "__main__":
    main()
