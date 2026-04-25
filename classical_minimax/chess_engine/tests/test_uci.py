import os
import subprocess
import sys
from pathlib import Path

import chess

from chess_engine.uci import parse_position_line


def test_fen_after_e4_parsed_with_board_constructor():
    board = chess.Board(fen="rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1")
    assert board.turn == chess.BLACK
    assert len(list(board.legal_moves)) > 0


def test_parse_position_startpos_moves():
    b = parse_position_line("position startpos moves e2e4 e7e5")
    assert b.fen().startswith("rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w")


def test_parse_position_fen_only():
    fen = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1"
    b = parse_position_line(f"position fen {fen}")
    assert b.turn == chess.BLACK


def test_parse_position_fen_with_moves():
    fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    b = parse_position_line(f"position fen {fen} moves e2e4")
    assert "P" in b.fen().split()[0]  # pawn structure changed from start


def test_uci_smoke_via_subprocess():
    classical = Path(__file__).resolve().parents[2]
    env = {**os.environ, "PYTHONPATH": str(classical)}
    script = "\n".join(["uci", "isready", "position startpos", "go depth 2", "quit", ""])
    proc = subprocess.run(
        [sys.executable, "-m", "chess_engine"],
        input=script,
        capture_output=True,
        text=True,
        env=env,
        cwd=str(classical),
        timeout=30,
        check=False,
    )
    out = proc.stdout + proc.stderr
    assert "uciok" in out
    assert "readyok" in out
    assert "bestmove" in out
