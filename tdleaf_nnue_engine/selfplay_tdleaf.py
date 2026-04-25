"""Self-play data generation with TD-Leaf(lambda) style targets."""

from __future__ import annotations

import random
from dataclasses import dataclass

import chess
import numpy as np

from tdleaf_nnue_engine.nnue_features import extract_features
from tdleaf_nnue_engine.search import Searcher


@dataclass
class SelfPlayConfig:
    games: int = 2
    max_plies: int = 48
    search_depth: int = 2
    lambda_value: float = 0.7
    temperature: float = 0.15
    seed: int = 0


def generate_tdleaf_dataset(searcher: Searcher, cfg: SelfPlayConfig) -> tuple[np.ndarray, np.ndarray]:
    rng = random.Random(cfg.seed)
    xs: list[np.ndarray] = []
    ys: list[float] = []

    for _ in range(cfg.games):
        board = chess.Board()
        for _ply in range(cfg.max_plies):
            if board.is_game_over(claim_draw=True):
                break

            result = searcher.search(board, depth=cfg.search_depth)
            if not result.leaf_scores:
                break

            values = np.array(result.leaf_scores, dtype=np.float32)
            td_target = td_leaf_lambda_target(values, cfg.lambda_value)
            side_target = td_target if board.turn == chess.WHITE else -td_target

            xs.append(extract_features(board))
            ys.append(side_target)

            move = result.best_move
            if cfg.temperature > 0 and rng.random() < cfg.temperature:
                legal = list(board.legal_moves)
                move = rng.choice(legal)
            board.push(move)

    if not xs:
        return np.zeros((0, 0), dtype=np.float32), np.zeros((0,), dtype=np.float32)
    return np.stack(xs).astype(np.float32), np.array(ys, dtype=np.float32)


def td_leaf_lambda_target(leaf_values: np.ndarray, lambda_value: float) -> float:
    """
    Collapse searched leaf values into one bootstrap target.

    This v1 approximation discounts deeper leaves geometrically and normalizes.
    """
    if leaf_values.size == 0:
        return 0.0
    lam = float(np.clip(lambda_value, 0.0, 1.0))
    powers = np.power(lam, np.arange(leaf_values.size, dtype=np.float32))
    denom = float(np.sum(powers)) or 1.0
    return float(np.dot(leaf_values, powers) / denom)
