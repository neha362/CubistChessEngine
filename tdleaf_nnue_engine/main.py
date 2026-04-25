"""CLI demo: choose a best move from a position."""

from __future__ import annotations

import argparse

import chess

from tdleaf_nnue_engine.eval import Evaluator
from tdleaf_nnue_engine.search import Searcher


def main() -> None:
    parser = argparse.ArgumentParser(description="TD-Leaf NNUE engine demo CLI.")
    parser.add_argument("--fen", default=chess.STARTING_FEN, help="FEN to analyze.")
    parser.add_argument("--depth", type=int, default=3, help="Search depth in plies.")
    parser.add_argument(
        "--weights",
        default="tdleaf_nnue_engine/checkpoints/nnue_runtime.npz",
        help="Runtime weight file (.npz). Falls back to material eval if missing.",
    )
    args = parser.parse_args()

    board = chess.Board(args.fen)
    evaluator = Evaluator(weights_path=args.weights)
    searcher = Searcher(evaluator=evaluator)
    result = searcher.search(board, depth=args.depth)

    print("position:", board.fen())
    print("best_move:", result.best_move.uci())
    print("score_cp:", result.best_score)


if __name__ == "__main__":
    main()
