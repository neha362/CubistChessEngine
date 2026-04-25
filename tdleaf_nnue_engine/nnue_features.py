"""HalfKP-like sparse feature extraction for chess positions."""

from __future__ import annotations

from dataclasses import dataclass

import chess
import numpy as np

PIECE_TYPES = (
    chess.PAWN,
    chess.KNIGHT,
    chess.BISHOP,
    chess.ROOK,
    chess.QUEEN,
    chess.KING,
)
PIECE_TO_INDEX = {piece: i for i, piece in enumerate(PIECE_TYPES)}
N_PIECE_PLANES = len(PIECE_TYPES) * 2
FEATURE_SIZE = 64 * N_PIECE_PLANES + 1


@dataclass(frozen=True)
class FeatureSpec:
    size: int = FEATURE_SIZE


def extract_features(board: chess.Board) -> np.ndarray:
    """Build a simple dense feature vector compatible with NNUE-style MLP."""
    vec = np.zeros(FEATURE_SIZE, dtype=np.float32)
    for square, piece in board.piece_map().items():
        color_offset = 0 if piece.color == chess.WHITE else len(PIECE_TYPES)
        piece_offset = PIECE_TO_INDEX[piece.piece_type]
        idx = square * N_PIECE_PLANES + color_offset + piece_offset
        vec[idx] = 1.0
    vec[-1] = 1.0 if board.turn == chess.WHITE else -1.0
    return vec


def extract_sparse_indices(board: chess.Board) -> list[int]:
    """Sparse indices placeholder for future incremental updates."""
    dense = extract_features(board)
    return np.flatnonzero(dense).astype(np.int32).tolist()
