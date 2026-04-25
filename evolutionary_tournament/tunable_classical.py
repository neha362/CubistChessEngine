"""
Two-feature decomposition of the classical `EvalAgent` (material vs PST) so
evolution can re-weight evaluators toward a Stockfish-anchored signal.
"""

from __future__ import annotations

import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT / "classical_minimax") not in sys.path:
    sys.path.insert(0, str(_ROOT / "classical_minimax"))

import chess
from chess_engine import eval as _ce  # type: ignore  # after sys.path


@dataclass(frozen=True)
class TunableWeights:
    """w_mat * material + w_pst * PST; optional bias in centipawns."""

    w_mat: float = 1.0
    w_pst: float = 1.0
    w_bias: float = 0.0

    def with_noise(self, scale: float = 0.15, rng: random.Random | None = None) -> "TunableWeights":
        r = rng or random.Random()
        j = r.gauss(0, 1) * scale
        k = r.gauss(0, 1) * scale
        b = r.gauss(0, 5) * (scale * 0.3)
        return TunableWeights(
            w_mat=project(self.w_mat * (1.0 + j), 0.1, 3.0),
            w_pst=project(self.w_pst * (1.0 + k), 0.1, 3.0),
            w_bias=self.w_bias + b,
        )


def project(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _decomposed(board: chess.Board) -> tuple[int, int]:
    """(material_imbalance, pst_imbalance) from White's view in centipawns."""
    mat = 0
    pst = 0
    for sq in chess.SQUARES:
        p = board.piece_at(sq)
        if p is None:
            continue
        idx = _ce._pst_table_index(sq, p.color)  # type: ignore[attr-defined]
        m = _ce._MATERIAL[p.piece_type]  # type: ignore[attr-defined]
        pt = 0
        if p.piece_type == chess.PAWN:
            pt = _ce._PST_P[idx]  # type: ignore[attr-defined]
        elif p.piece_type == chess.KNIGHT:
            pt = _ce._PST_N[idx]  # type: ignore[attr-defined]
        elif p.piece_type == chess.BISHOP:
            pt = _ce._PST_B[idx]  # type: ignore[attr-defined]
        elif p.piece_type == chess.ROOK:
            pt = _ce._PST_R[idx] + _ce._open_file_bonus(board, sq, p.color)  # type: ignore
        if p.color == chess.WHITE:
            mat += m
            pst += pt
        else:
            mat -= m
            pst -= pt
    return mat, pst


def evaluate_tunable(board: chess.Board, w: TunableWeights) -> int:
    m, p = _decomposed(board)
    s = w.w_mat * m + w.w_pst * p + w.w_bias
    r = int(round(s))
    return max(-30_000, min(30_000, r))


def make_search_eval_fn(w: TunableWeights) -> Callable[[chess.Board], int]:
    return lambda b: evaluate_tunable(b, w)
