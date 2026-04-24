"""Run a short self-play game using move generation, search, and evaluation agents."""

from __future__ import annotations

import argparse

import chess

from chess_engine.eval import EvalAgent
from chess_engine.move_gen import MoveGenAgent
from chess_engine.search import SearchAgent


def main() -> None:
    parser = argparse.ArgumentParser(description="Self-play demo for the modular chess engine.")
    parser.add_argument("--depth", type=int, default=3, help="Iterative deepening depth (plies).")
    args = parser.parse_args()

    board = chess.Board()
    move_gen = MoveGenAgent()
    eval_agent = EvalAgent()
    search = SearchAgent(eval_agent.evaluate, move_gen)

    print(f"Search depth (ID max): {args.depth}")
    for ply in range(20):
        if board.is_game_over():
            out = board.outcome()
            term = out.termination.name if out else "unknown"
            print(f"Game over: {board.result()} ({term})")
            break
        side = "White" if board.turn == chess.WHITE else "Black"
        move = search.best_move(board, args.depth)
        san = board.san(move)
        board.push(move)
        score = eval_agent.evaluate(board)
        print(f"{ply + 1:2d}. {side:5s} {san:6s}  eval (cp, White+): {score:+d}")


if __name__ == "__main__":
    main()
