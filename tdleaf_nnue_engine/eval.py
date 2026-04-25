"""Runtime evaluator with exported NNUE weights and material fallback."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import chess
import numpy as np

from tdleaf_nnue_engine.nnue_features import FEATURE_SIZE, extract_features

MATERIAL = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 0,
}


class Evaluator:
    """Position evaluator returning centipawns from White perspective."""

    def __init__(self, weights_path: Optional[str] = None) -> None:
        self.weights_path = Path(weights_path) if weights_path else None
        self._weights = self._try_load_weights(self.weights_path)

    def evaluate(self, board: chess.Board) -> int:
        if board.is_checkmate():
            return -100_000 if board.turn == chess.WHITE else 100_000
        if board.is_stalemate() or board.is_insufficient_material():
            return 0
        if self._weights is None:
            return self._material_eval(board)
        return int(round(self._nnue_forward(board)))

    def _material_eval(self, board: chess.Board) -> int:
        score = 0
        for piece in board.piece_map().values():
            value = MATERIAL[piece.piece_type]
            score += value if piece.color == chess.WHITE else -value
        return score

    def _nnue_forward(self, board: chess.Board) -> float:
        features = np.asarray(extract_features(board), dtype=np.float32).reshape(-1)
        w1 = self._weights["fc1_weight"]
        b1 = self._weights["fc1_bias"]
        w2 = self._weights["fc2_weight"]
        b2 = self._weights["fc2_bias"]
        w3 = self._weights["out_weight"]
        b3 = self._weights["out_bias"]

        # Normalize intermediate activations as vectors to avoid shape drift.
        h1 = np.clip(np.maximum(features @ w1.T + b1, 0.0), 0.0, 1.0).reshape(-1)
        h2 = np.clip(np.maximum(h1 @ w2.T + b2, 0.0), 0.0, 1.0).reshape(-1)
        out = np.asarray(h2 @ w3.T + b3, dtype=np.float32).reshape(-1)
        if out.size != 1:
            raise ValueError(f"Expected scalar NNUE output, got shape {out.shape}")
        return float(out[0])

    def _try_load_weights(self, path: Optional[Path]) -> Optional[dict[str, np.ndarray]]:
        if path is None or not path.exists():
            return None
        data = np.load(path)
        required = {
            "fc1_weight",
            "fc1_bias",
            "fc2_weight",
            "fc2_bias",
            "out_weight",
            "out_bias",
            "input_dim",
        }
        if not required.issubset(data.files):
            return None
        input_dim = self._metadata_int(data["input_dim"])
        if input_dim is None or input_dim != FEATURE_SIZE:
            return None

        try:
            w1 = self._as_2d(data["fc1_weight"])
            b1 = self._as_1d(data["fc1_bias"])
            w2 = self._as_2d(data["fc2_weight"])
            b2 = self._as_1d(data["fc2_bias"])
            w3 = self._as_2d(data["out_weight"])
            b3 = self._as_1d(data["out_bias"])
        except (TypeError, ValueError):
            return None

        if w1.shape[1] != FEATURE_SIZE or b1.shape[0] != w1.shape[0]:
            return None
        if w2.shape[1] != w1.shape[0] or b2.shape[0] != w2.shape[0]:
            return None
        if w3.shape[1] != w2.shape[0] or b3.shape[0] != w3.shape[0]:
            return None
        if w3.shape[0] != 1:
            return None

        return {
            "fc1_weight": w1,
            "fc1_bias": b1,
            "fc2_weight": w2,
            "fc2_bias": b2,
            "out_weight": w3,
            "out_bias": b3,
        }

    @staticmethod
    def _as_1d(value: np.ndarray) -> np.ndarray:
        arr = np.asarray(value, dtype=np.float32).reshape(-1)
        if arr.size == 0:
            raise ValueError("Expected non-empty 1-D tensor")
        return arr

    @staticmethod
    def _as_2d(value: np.ndarray) -> np.ndarray:
        arr = np.asarray(value, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        elif arr.ndim > 2:
            arr = arr.reshape(arr.shape[0], -1)
        if arr.ndim != 2 or 0 in arr.shape:
            raise ValueError("Expected non-empty 2-D tensor")
        return arr

    @staticmethod
    def _metadata_int(value: np.ndarray | int | float) -> Optional[int]:
        """Parse integer metadata from scalar / 0-d / 1-d numeric fields."""
        arr = np.asarray(value)
        if arr.size != 1:
            return None
        scalar = arr.reshape(-1)[0]
        try:
            return int(scalar)
        except (TypeError, ValueError, OverflowError):
            return None
