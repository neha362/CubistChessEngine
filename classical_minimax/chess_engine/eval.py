"""Static evaluation — material and piece-square tables (white-positive centipawns)."""

from __future__ import annotations

import chess

# Material (centipawns)
PAWN = 100
KNIGHT = 320
BISHOP = 330
ROOK = 500
QUEEN = 900

_MATERIAL = {
    chess.PAWN: PAWN,
    chess.KNIGHT: KNIGHT,
    chess.BISHOP: BISHOP,
    chess.ROOK: ROOK,
    chess.QUEEN: QUEEN,
    chess.KING: 0,
}

def _pst_table_index(sq: int, color: chess.Color) -> int:
    """Map square to PST row-major index with rank 8 at row 0 (white's forward = decreasing row)."""
    f = chess.square_file(sq)
    r = chess.square_rank(sq)
    r_adj = r if color == chess.WHITE else 7 - r
    return (7 - r_adj) * 8 + f


def _pst_pawn_white() -> list[int]:
    """Advance bonus: reward pushing toward promotion, slight center tilt."""
    # rank 8 (promotion) row unused for pawns at start; index by rank from white POV
    # rows 0-7 in table = ranks 8..1 from white's view (a8-h8, ..., a1-h1)
    return [
        0, 0, 0, 0, 0, 0, 0, 0,  # rank 8 (no pawns normally)
        5, 5, 5, 5, 5, 5, 5, 5,  # 7th
        10, 10, 15, 20, 20, 15, 10, 10,
        5, 5, 10, 25, 25, 10, 5, 5,
        0, 0, 0, 20, 20, 0, 0, 0,
        5, -5, -10, 0, 0, -10, -5, 5,
        5, 10, 10, -20, -20, 10, 10, 5,
        0, 0, 0, 0, 0, 0, 0, 0,  # rank 1 home — symmetric row
    ]


def _pst_knight_white() -> list[int]:
    """Center and outpost bonus (mirrored for black)."""
    return [
        -50, -40, -30, -30, -30, -30, -40, -50,
        -40, -20, 0, 0, 0, 0, -20, -40,
        -30, 0, 10, 15, 15, 10, 0, -30,
        -30, 5, 15, 20, 20, 15, 5, -30,
        -30, 0, 15, 20, 20, 15, 0, -30,
        -30, 5, 10, 15, 15, 10, 5, -30,
        -40, -20, 0, 5, 5, 0, -20, -40,
        -50, -40, -30, -30, -30, -30, -40, -50,
    ]


def _pst_bishop_white() -> list[int]:
    """Long diagonal / central activity bonus."""
    return [
        -20, -10, -10, -10, -10, -10, -10, -20,
        -10, 5, 0, 0, 0, 0, 5, -10,
        -10, 10, 10, 10, 10, 10, 10, -10,
        -10, 0, 10, 15, 15, 10, 0, -10,
        -10, 5, 5, 10, 10, 5, 5, -10,
        -10, 0, 5, 10, 10, 5, 0, -10,
        -10, 0, 0, 0, 0, 0, 0, -10,
        -20, -10, -10, -10, -10, -10, -10, -20,
    ]


def _pst_rook_white() -> list[int]:
    """Open / semi-open file bonus (combined with file scan at runtime)."""
    return [
        0, 0, 0, 5, 5, 0, 0, 0,
        0, 0, 0, 5, 5, 0, 0, 0,
        0, 0, 0, 5, 5, 0, 0, 0,
        0, 0, 0, 5, 5, 0, 0, 0,
        0, 0, 0, 5, 5, 0, 0, 0,
        0, 0, 0, 5, 5, 0, 0, 0,
        0, 0, 0, 5, 5, 0, 0, 0,
        0, 0, 0, 5, 5, 0, 0, 0,
    ]


_PST_P = _pst_pawn_white()
_PST_N = _pst_knight_white()
_PST_B = _pst_bishop_white()
_PST_R = _pst_rook_white()


def _open_file_bonus(board: chess.Board, sq: int, color: chess.Color) -> int:
    """Extra centipawns if rook/queen on file with no friendly pawn."""
    f = chess.square_file(sq)
    for r in range(8):
        s = chess.square(f, r)
        p = board.piece_at(s)
        if p and p.piece_type == chess.PAWN and p.color == color:
            return 0
    return 15


class EvalAgent:
    """Heuristic evaluation from White's perspective (centipawns)."""

    def evaluate(self, board: chess.Board) -> int:
        score = 0
        for sq in chess.SQUARES:
            p = board.piece_at(sq)
            if p is None:
                continue
            idx = _pst_table_index(sq, p.color)
            mat = _MATERIAL[p.piece_type]
            pst = 0
            if p.piece_type == chess.PAWN:
                pst = _PST_P[idx]
            elif p.piece_type == chess.KNIGHT:
                pst = _PST_N[idx]
            elif p.piece_type == chess.BISHOP:
                pst = _PST_B[idx]
            elif p.piece_type == chess.ROOK:
                pst = _PST_R[idx] + _open_file_bonus(board, sq, p.color)
            elif p.piece_type == chess.QUEEN:
                pst = 0
            contrib = mat + pst
            score += contrib if p.color == chess.WHITE else -contrib
        return score
