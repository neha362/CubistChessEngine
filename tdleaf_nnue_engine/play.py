"""Interactive CLI to play against the TD-Leaf NNUE engine."""

from __future__ import annotations

import argparse
from typing import Optional

import chess

from tdleaf_nnue_engine.eval import Evaluator
from tdleaf_nnue_engine.search import SearchResult, Searcher


HELP_TEXT = (
    "Enter a move in SAN (e.g. Nf3, O-O) or UCI (e.g. e2e4).\n"
    "Commands: help, board, fen, quit."
)


def _parse_side(value: str) -> chess.Color:
    side = value.strip().lower()
    if side in {"w", "white"}:
        return chess.WHITE
    if side in {"b", "black"}:
        return chess.BLACK
    raise argparse.ArgumentTypeError("side must be one of: white/w or black/b")


def _try_parse_move(board: chess.Board, text: str) -> Optional[chess.Move]:
    raw = text.strip()
    if not raw:
        return None
    try:
        move = board.parse_san(raw)
        if move in board.legal_moves:
            return move
    except ValueError:
        pass
    try:
        move = chess.Move.from_uci(raw)
    except ValueError:
        return None
    return move if move in board.legal_moves else None


def _render_board(board: chess.Board) -> str:
    files = "  a b c d e f g h"
    rows = []
    board_rows = board.__str__().splitlines()
    for rank in range(8, 0, -1):
        row = board_rows[8 - rank]
        rows.append(f"{rank} {row} {rank}")
    return "\n".join([files, *rows, files])


def _engine_move(searcher: Searcher, board: chess.Board, depth: int) -> SearchResult:
    result = searcher.search(board, depth=depth)
    board.push(result.best_move)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Play against the TD-Leaf NNUE engine.")
    parser.add_argument("--side", type=_parse_side, default=chess.WHITE, help="Your side: white|black.")
    parser.add_argument("--depth", type=int, default=3, help="Engine search depth in plies.")
    parser.add_argument(
        "--weights",
        default="tdleaf_nnue_engine/checkpoints/nnue_runtime.npz",
        help="Runtime weight file (.npz). Falls back to material eval if missing/invalid.",
    )
    args = parser.parse_args()

    if args.depth < 1:
        raise SystemExit("depth must be >= 1")

    board = chess.Board()
    evaluator = Evaluator(weights_path=args.weights)
    searcher = Searcher(evaluator=evaluator)
    human_side = args.side

    print("TD-Leaf NNUE Play Mode")
    print(f"You are playing as {'White' if human_side == chess.WHITE else 'Black'}")
    print(f"Engine depth: {args.depth}")
    print(HELP_TEXT)

    while True:
        print()
        print(_render_board(board))
        if board.is_game_over(claim_draw=True):
            print(f"Game over: {board.result(claim_draw=True)} ({board.outcome(claim_draw=True)})")
            return

        if board.turn != human_side:
            result = _engine_move(searcher, board, args.depth)
            print(f"Engine move: {result.best_move.uci()} (score_cp={result.best_score})")
            continue

        user_input = input("Your move> ").strip()
        cmd = user_input.lower()
        if cmd in {"q", "quit", "exit"}:
            print("Goodbye.")
            return
        if cmd in {"h", "help", "?"}:
            print(HELP_TEXT)
            continue
        if cmd == "board":
            continue
        if cmd == "fen":
            print(board.fen())
            continue

        move = _try_parse_move(board, user_input)
        if move is None:
            print("Invalid or illegal move. Type 'help' for examples.")
            continue
        board.push(move)


if __name__ == "__main__":
    main()
