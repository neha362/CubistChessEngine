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
