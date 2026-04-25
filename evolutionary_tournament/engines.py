"""
Adapters over the python-chess engines: ``classical_minimax`` and ``berserker_2``.

Each provides :meth:`analyze` → optimal move, every legal line’s **expected
centipawn loss** (best_score − line_score) from the side to move’s perspective.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import chess

_ROOT = Path(__file__).resolve().parent.parent
_classical = str(_ROOT / "classical_minimax")
_b2 = str(_ROOT / "berserker_2")
if _classical not in sys.path:
    sys.path.insert(0, _classical)
if _b2 not in sys.path:
    sys.path.insert(0, _b2)

from chess_engine.eval import EvalAgent  # type: ignore
from chess_engine.move_gen import MoveGenAgent  # type: ignore
from chess_engine.search import SearchAgent  # type: ignore

from move_gen import MoveGen as B2MoveGen  # type: ignore
from eval import Evaluator as B2Evaluator  # type: ignore
from search import Search as B2Search  # type: ignore

from .tunable_classical import TunableWeights, make_search_eval_fn


@dataclass
class MoveLine:
    uci: str
    score_cp: int
    centipawn_loss: int


@dataclass
class TurnReport:
    engine_id: str
    best_uci: str
    best_score_cp: int
    lines: list[MoveLine] = field(default_factory=list)


def _report_from_map(engine_id: str, uci_to_cp: dict[str, int], best: str) -> TurnReport:
    bsc = uci_to_cp[best] if uci_to_cp and best in uci_to_cp else 0
    mxs = max(uci_to_cp.values(), default=0) if uci_to_cp else 0
    lines: list[MoveLine] = []
    for u, s in uci_to_cp.items():
        lines.append(MoveLine(uci=u, score_cp=s, centipawn_loss=mxs - s))
    lines.sort(key=lambda x: (-x.score_cp, x.uci))
    return TurnReport(
        engine_id=engine_id,
        best_uci=best,
        best_score_cp=bsc,
        lines=lines,
    )


def spearman_ordinal(xs: list[float], ys: list[float]) -> float:
    """ρ on paired ranks; empty / degenerate → 0."""
    n = len(xs)
    if n < 2 or n != len(ys):
        return 0.0
    rx = _ranks(xs)
    ry = _ranks(ys)
    mrx = sum(rx) / n
    mry = sum(ry) / n
    num = sum((rx[i] - mrx) * (ry[i] - mry) for i in range(n))
    denx = sum((r - mrx) ** 2 for r in rx) ** 0.5
    deny = sum((r - mry) ** 2 for r in ry) ** 0.5
    if denx < 1e-9 or deny < 1e-9:
        return 0.0
    return num / (denx * deny)


def _ranks(vs: list[float]) -> list[float]:
    n = len(vs)
    order = list(range(n))
    order.sort(key=lambda i: vs[i])
    r = [0.0] * n
    p = 0
    while p < n:
        q = p + 1
        v = vs[order[p]]
        while q < n and vs[order[q]] == v:
            q += 1
        mean_rank = (p + q - 1) / 2.0
        for k in range(p, q):
            r[order[k]] = mean_rank
        p = q
    return r


def rank_correlation_move_lists(ref: dict[str, int], ours: dict[str, int]) -> float:
    """Spearman on moves present in both maps."""
    common = [k for k in ref if k in ours]
    if len(common) < 3:
        return 0.0
    common.sort()
    a = [float(ours[k]) for k in common]
    b = [float(ref[k]) for k in common]
    return spearman_ordinal(b, a)


class ClassicalEngine:
    """Tunable static eval + classical search."""

    def __init__(self, depth: int = 4, weights: TunableWeights | None = None) -> None:
        self.id = "classical_minimax"
        self._depth = depth
        w = weights or TunableWeights()
        self._search = SearchAgent(make_search_eval_fn(w), MoveGenAgent())
        self.weights = w

    def pick_move(self, board: chess.Board) -> chess.Move:
        return self._search.best_move(board, self._depth)

    def analyze(self, board: chess.Board) -> TurnReport:
        per, bm, _bs = self._search.root_all_scores(board, self._depth)
        m = {mv.uci(): c for mv, c in per}
        b = bm.uci() if bm is not None else (max(m, key=m.get) if m else "")
        return _report_from_map(self.id, m, b)


class Berserker2Engine:
    def __init__(self, depth: int = 3) -> None:
        self.id = "berserker_2"
        self._depth = depth
        self._search = B2Search()
        self._move_gen = B2MoveGen()
        self._eval = B2Evaluator()

    def pick_move(self, board: chess.Board) -> chess.Move:
        mv, _ = self._search.find_best_move(
            board, self._move_gen, self._eval, time_limit=0.5, max_depth=self._depth, quiet=True
        )
        if mv is not None:
            return mv
        return next(iter(board.legal_moves))

    def analyze(self, board: chess.Board) -> TurnReport:
        per, bm, _bs = self._search.root_all_scores(
            board, self._move_gen, self._eval, self._depth, time_limit=90.0
        )
        m = {vv.uci(): c for vv, c in per}
        b = bm.uci() if bm is not None else (max(m, key=m.get) if m else "")
        return _report_from_map(self.id, m, b)


def reference_classical() -> SearchAgent:
    return SearchAgent(EvalAgent().evaluate, MoveGenAgent())


# ── Berserker1 and MCTS wrappers ──────────────────────────────────────────────

_b1 = str(_ROOT / "berserker1")
_mc = str(_ROOT / "monte_carlo")
if _b1 not in sys.path:
    sys.path.insert(0, _b1)
if _mc not in sys.path:
    sys.path.insert(0, _mc)


def _tuple_to_chess_move(move_tuple: tuple, board: chess.Board) -> chess.Move:
    """Inverse of ensemble_adapters chess_move_to_tuple: (row-inverted sq, sq, promo) → Move."""
    from_sq_idx, to_sq_idx, promotion = move_tuple
    from_file = from_sq_idx % 8
    from_rank = 7 - from_sq_idx // 8
    to_file = to_sq_idx % 8
    to_rank = 7 - to_sq_idx // 8
    from_square = chess.square(from_file, from_rank)
    to_square = chess.square(to_file, to_rank)
    promo = chess.Piece.from_symbol(promotion).piece_type if promotion else None
    return chess.Move(from_square, to_square, promotion=promo)


class Berserker1Engine:
    """Berserker1 negamax search (berserker1/berserker_search_agent.py)."""

    def __init__(self, max_depth: int = 3, movetime_ms: int = 800) -> None:
        self.id = "berserker_1"
        self._max_depth = max_depth
        self._movetime_ms = movetime_ms

    def pick_move(self, board: chess.Board) -> chess.Move:
        try:
            import berserker_search_agent as _b1s
            from movegen_agent import from_fen as _b1_from_fen
            state = _b1_from_fen(board.fen())
            result = _b1s.search(
                state,
                max_depth=self._max_depth,
                movetime_ms=self._movetime_ms,
                verbose=False,
            )
            if result.move is not None:
                mv = _tuple_to_chess_move(result.move, board)
                if mv in board.legal_moves:
                    return mv
        except Exception:
            pass
        return next(iter(board.legal_moves))


class MCTSEngine:
    """Monte Carlo Tree Search engine (monte_carlo/mcts_agent.py)."""

    def __init__(self, max_iter: int = 300, movetime_ms: int = 800) -> None:
        self.id = "mcts"
        self._max_iter = max_iter
        self._movetime_ms = movetime_ms

    def pick_move(self, board: chess.Board) -> chess.Move:
        try:
            import mcts_agent as _mcts
            from movegen_agent import from_fen as _mc_from_fen
            state = _mc_from_fen(board.fen())
            result = _mcts.mcts_search(
                state,
                max_iter=self._max_iter,
                movetime_ms=self._movetime_ms,
                verbose=False,
            )
            if result.best_move is not None:
                mv = _tuple_to_chess_move(result.best_move, board)
                if mv in board.legal_moves:
                    return mv
        except Exception:
            pass
        return next(iter(board.legal_moves))


# ── TD-Leaf NNUE wrapper ───────────────────────────────────────────────────────

_nnue = str(_ROOT / "tdleaf_nnue_engine")
if _nnue not in sys.path:
    sys.path.insert(0, _nnue)


class NNUEEngine:
    """TD-Leaf NNUE engine (tdleaf_nnue_engine/); falls back to material eval if no weights."""

    def __init__(self, depth: int = 3, weights_path: str | None = None) -> None:
        self.id = "nnue_tdleaf"
        self._depth = depth
        from tdleaf_nnue_engine.eval import Evaluator
        from tdleaf_nnue_engine.search import Searcher
        self._searcher = Searcher(evaluator=Evaluator(weights_path=weights_path))

    def pick_move(self, board: chess.Board) -> chess.Move:
        try:
            mv = self._searcher.best_move(board, depth=self._depth)
            if mv in board.legal_moves:
                return mv
        except Exception:
            pass
        return next(iter(board.legal_moves))
