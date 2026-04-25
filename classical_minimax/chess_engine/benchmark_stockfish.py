"""
Requires Stockfish installed:
  macOS:  brew install stockfish
  Linux:  sudo apt install stockfish
or set STOCKFISH_PATH env variable to binary location
"""

from __future__ import annotations

import os

import chess
import chess.engine

from chess_engine.eval import EvalAgent

STOCKFISH_PATH = os.environ.get("STOCKFISH_PATH", "stockfish")

TEST_POSITIONS = [
    ("Starting position", chess.Board()),
    (
        "After 1.e4",
        chess.Board(fen="rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1"),
    ),
    (
        "Queen up for white",
        chess.Board(fen="rnb1kbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"),
    ),
]


def compare_evals() -> None:
    eval_agent = EvalAgent()
    print(f"{'Position':<25} {'Our eval':>10} {'Stockfish':>12} {'Delta':>8}")
    print("-" * 58)
    try:
        engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)
    except OSError as exc:
        print(f"(Stockfish not available at {STOCKFISH_PATH!r}: {exc}; skipping engine rows.)")
        return
    try:
        for name, board in TEST_POSITIONS:
            our_score = eval_agent.evaluate(board)
            info = engine.analyse(board, chess.engine.Limit(depth=15))
            sf_score = info["score"].white().score(mate_score=100000)
            delta = our_score - sf_score if sf_score is not None else "N/A"
            print(f"{name:<25} {our_score:>10} {str(sf_score):>12} {str(delta):>8}")
    finally:
        engine.quit()


def play_vs_stockfish(our_engine_uci_path: str, elo: int = 1200) -> None:
    """Play our engine against Stockfish at reduced Elo."""
    our_engine = chess.engine.SimpleEngine.popen_uci(["python", our_engine_uci_path])
    sf_engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)
    sf_engine.configure({"UCI_LimitStrength": True, "UCI_Elo": elo})
    board = chess.Board()
    while not board.is_game_over():
        if board.turn == chess.WHITE:
            result = our_engine.play(board, chess.engine.Limit(depth=4))
        else:
            result = sf_engine.play(board, chess.engine.Limit(time=0.1))
        board.push(result.move)
        print(board.unicode(), "\n")
    print("Result:", board.result())
    our_engine.quit()
    sf_engine.quit()


if __name__ == "__main__":
    compare_evals()
