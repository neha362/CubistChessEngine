"""Universal Chess Interface (stdin/stdout) facade — engine-agnostic."""

from __future__ import annotations

import sys
from typing import Protocol

import chess


class SupportsBestMove(Protocol):
    def best_move(self, board: chess.Board, max_depth: int) -> chess.Move: ...


def parse_position_line(line: str) -> chess.Board:
    """
    Parse a UCI ``position ...`` command into a ``chess.Board``.

    All FEN paths use ``chess.Board(fen=...)`` only (no manual FEN parsing).
    """
    parts = line.strip().split()
    if len(parts) < 2 or parts[0] != "position":
        raise ValueError("expected line to start with 'position'")

    if parts[1] == "startpos":
        board = chess.Board()
        if len(parts) > 2 and parts[2] == "moves":
            for uci in parts[3:]:
                board.push(chess.Move.from_uci(uci))
        return board

    if parts[1] == "fen":
        try:
            moves_idx = parts.index("moves", 2)
        except ValueError:
            moves_idx = len(parts)
        fen = " ".join(parts[2:moves_idx])
        board = chess.Board(fen=fen)
        if moves_idx < len(parts) and parts[moves_idx] == "moves":
            for uci in parts[moves_idx + 1 :]:
                board.push(chess.Move.from_uci(uci))
        return board

    raise ValueError(f"unknown position subcommand: {parts[1]!r}")


class UCIEngine:
    """
    Minimal UCI loop. Inject any object implementing ``best_move(board, max_depth)``.
    """

    def __init__(self, search_agent: SupportsBestMove) -> None:
        self._search = search_agent
        self.board = chess.Board()

    def run(self) -> None:
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            cmd = parts[0]

            if cmd == "uci":
                print("id name ClassicalMinimax")
                print("id author CubistChessEngine")
                print("uciok", flush=True)
            elif cmd == "isready":
                print("readyok", flush=True)
            elif cmd == "ucinewgame":
                self.board = chess.Board()
            elif cmd == "position":
                self.board = parse_position_line(line)
            elif cmd == "go":
                depth = 4
                if len(parts) >= 3 and parts[1] == "depth":
                    depth = int(parts[2])
                elif len(parts) >= 3 and parts[1] == "movetime":
                    depth = 4
                if not list(self.board.legal_moves):
                    print("bestmove (none)", flush=True)
                else:
                    move = self._search.best_move(self.board, depth)
                    print(f"bestmove {move.uci()}", flush=True)
            elif cmd == "quit":
                break
