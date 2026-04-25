"""
ensemble_adapters/converters.py — FEN-pivot board converters.

The 5 engines disagree on data model:
  • berserker1, monte_carlo       → custom GameState (flat 64-element list)
  • berserker_2, classical_minimax → python-chess chess.Board
  • claude_api                    → custom BoardState (2D 8x8 list)

`engine_wrappers.py` already does these conversions inline. This module
exposes them as standalone utilities for code that needs to move a position
between formats outside the wrapper boundary (debugging, tests, scenario
glue, etc.). FEN is the pivot format — every conversion routes through it.
"""

from __future__ import annotations
from typing import Any

FILES = 'abcdefgh'

# ───────────────────────────────────────────────────────────────────
# GameState  (flat 64-element board)
# ───────────────────────────────────────────────────────────────────
_GS_TO_FEN = {
    'wP':'P','wN':'N','wB':'B','wR':'R','wQ':'Q','wK':'K',
    'bP':'p','bN':'n','bB':'b','bR':'r','bQ':'q','bK':'k',
}


def gamestate_to_fen(gs) -> str:
    """Compose a FEN from a GameState (flat board, row 0 = rank 8)."""
    rows = []
    for r in range(8):
        row, blank = '', 0
        for c in range(8):
            p = gs.board[r * 8 + c]
            if not p:
                blank += 1
                continue
            if blank:
                row += str(blank); blank = 0
            row += _GS_TO_FEN[p]
        if blank:
            row += str(blank)
        rows.append(row)
    placement = '/'.join(rows)

    cas = ''
    if gs.castling.get('wK'): cas += 'K'
    if gs.castling.get('wQ'): cas += 'Q'
    if gs.castling.get('bK'): cas += 'k'
    if gs.castling.get('bQ'): cas += 'q'
    cas = cas or '-'

    if gs.ep_square is None:
        ep = '-'
    else:
        ep = FILES[gs.ep_square % 8] + str(8 - gs.ep_square // 8)

    halfmove = getattr(gs, 'halfmove', 0)
    fullmove = getattr(gs, 'fullmove', 1)
    return f"{placement} {gs.turn} {cas} {ep} {halfmove} {fullmove}"


def fen_to_gamestate(fen: str):
    from movegen_agent import from_fen
    return from_fen(fen)


# ───────────────────────────────────────────────────────────────────
# chess.Board  (python-chess)
# ───────────────────────────────────────────────────────────────────
def board_to_fen(board) -> str:
    return board.fen()


def fen_to_board(fen: str):
    import chess
    return chess.Board(fen)


# ───────────────────────────────────────────────────────────────────
# BoardState  (claude_api — 2D 8x8 with single-char pieces)
# ───────────────────────────────────────────────────────────────────
def boardstate_to_fen(bs) -> str:
    """Compose a FEN from a claude_api BoardState (2D board[r][c])."""
    rows = []
    for r in range(8):
        row, blank = '', 0
        for c in range(8):
            p = bs.board[r][c]
            if not p:
                blank += 1
                continue
            if blank:
                row += str(blank); blank = 0
            row += p
        if blank:
            row += str(blank)
        rows.append(row)
    placement = '/'.join(rows)

    cas = bs.castling if (hasattr(bs, 'castling') and bs.castling) else '-'

    ep_attr = getattr(bs, 'en_passant', None)
    if ep_attr:
        r, c = ep_attr
        ep = FILES[c] + str(8 - r)
    else:
        ep = '-'

    halfmove = getattr(bs, 'halfmove', 0)
    fullmove = getattr(bs, 'fullmove', 1)
    return f"{placement} {bs.turn} {cas} {ep} {halfmove} {fullmove}"


def fen_to_boardstate(fen: str):
    try:
        from claude_api.move_engine import parse_fen
    except ImportError:
        from move_engine import parse_fen  # if cwd is claude_api/
    return parse_fen(fen)


# ───────────────────────────────────────────────────────────────────
# Cross-conversions (composed via FEN pivot)
# ───────────────────────────────────────────────────────────────────
def gamestate_to_board(gs):       return fen_to_board(gamestate_to_fen(gs))
def board_to_gamestate(b):        return fen_to_gamestate(b.fen())
def gamestate_to_boardstate(gs):  return fen_to_boardstate(gamestate_to_fen(gs))
def boardstate_to_gamestate(bs):  return fen_to_gamestate(boardstate_to_fen(bs))
def board_to_boardstate(b):       return fen_to_boardstate(b.fen())
def boardstate_to_board(bs):      return fen_to_board(boardstate_to_fen(bs))


def to_fen(source: Any) -> str:
    """Detect source representation, return FEN."""
    if isinstance(source, str):
        return source
    if hasattr(source, 'fen') and callable(getattr(source, 'fen')):
        return source.fen()                   # chess.Board
    if hasattr(source, 'board'):
        b = source.board
        if isinstance(b, list) and b and isinstance(b[0], list):
            return boardstate_to_fen(source)  # 2D BoardState
        if isinstance(b, list):
            return gamestate_to_fen(source)   # flat GameState
    raise TypeError(f"Cannot convert source of type {type(source).__name__}")
