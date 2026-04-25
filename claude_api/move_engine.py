"""
Move Generation Agent — Claude Oracle Chess Engine
Generates pseudo-legal and legal moves, handles en passant, castling, pins, checks.
Can be used standalone or called by the search agent.
"""

from dataclasses import dataclass, field
from typing import Optional
import copy

# ─── Board constants ───────────────────────────────────────────────────────────

PIECE_SYMBOLS = {
    'P': '♙', 'N': '♘', 'B': '♗', 'R': '♖', 'Q': '♕', 'K': '♔',
    'p': '♟', 'n': '♞', 'b': '♝', 'r': '♜', 'q': '♛', 'k': '♚',
}

# Directions: (row_delta, col_delta)
KNIGHT_MOVES  = [(-2,-1),(-2,1),(-1,-2),(-1,2),(1,-2),(1,2),(2,-1),(2,1)]
BISHOP_DIRS   = [(-1,-1),(-1,1),(1,-1),(1,1)]
ROOK_DIRS     = [(-1,0),(1,0),(0,-1),(0,1)]
QUEEN_DIRS    = BISHOP_DIRS + ROOK_DIRS
KING_DIRS     = QUEEN_DIRS


# ─── Data structures ───────────────────────────────────────────────────────────

@dataclass
class Move:
    from_sq: tuple[int,int]        # (row, col) 0-indexed, row 0 = rank 8
    to_sq:   tuple[int,int]
    promotion: Optional[str] = None   # 'q','r','b','n' if pawn promotes
    is_castling: bool = False
    is_en_passant: bool = False

    def uci(self) -> str:
        files = 'abcdefgh'
        fr, fc = self.from_sq
        tr, tc = self.to_sq
        s = files[fc] + str(8 - fr) + files[tc] + str(8 - tr)
        if self.promotion:
            s += self.promotion
        return s

    def __repr__(self):
        return self.uci()


@dataclass
class BoardState:
    """
    Minimal board state sufficient for move generation.
    board[r][c] is a piece char ('P','n', etc.) or '' for empty.
    """
    board: list[list[str]]
    turn: str = 'w'                      # 'w' or 'b'
    castling: str = 'KQkq'               # subset of 'KQkq'
    en_passant: Optional[tuple[int,int]] = None   # target square (row,col) or None
    halfmove_clock: int = 0
    fullmove_number: int = 1

    def copy(self) -> 'BoardState':
        return BoardState(
            board=[row[:] for row in self.board],
            turn=self.turn,
            castling=self.castling,
            en_passant=self.en_passant,
            fullmove_number=self.fullmove_number,
        )

    def piece_at(self, r: int, c: int) -> str:
        return self.board[r][c]

    def is_empty(self, r: int, c: int) -> bool:
        return self.board[r][c] == ''

    def is_enemy(self, r: int, c: int) -> bool:
        p = self.board[r][c]
        if not p:
            return False
        return p.islower() if self.turn == 'w' else p.isupper()

    def is_friendly(self, r: int, c: int) -> bool:
        p = self.board[r][c]
        if not p:
            return False
        return p.isupper() if self.turn == 'w' else p.islower()

    def fen(self) -> str:
        rows = []
        for row in self.board:
            empty = 0
            s = ''
            for cell in row:
                if cell == '':
                    empty += 1
                else:
                    if empty:
                        s += str(empty)
                        empty = 0
                    s += cell
            if empty:
                s += str(empty)
            rows.append(s)
        pos = '/'.join(rows)
        ep = '-'
        if self.en_passant:
            r, c = self.en_passant
            ep = 'abcdefgh'[c] + str(8 - r)
        castling = self.castling if self.castling else '-'
        return f"{pos} {self.turn} {castling} {ep} {self.halfmove_clock} {self.fullmove_number}"


# ─── FEN parser ────────────────────────────────────────────────────────────────

def parse_fen(fen: str) -> BoardState:
    parts = fen.strip().split()
    pos_str, turn, castling, ep_str = parts[0], parts[1], parts[2], parts[3]
    halfmove = int(parts[4]) if len(parts) > 4 else 0
    fullmove = int(parts[5]) if len(parts) > 5 else 1

    board = []
    for rank_str in pos_str.split('/'):
        row = []
        for ch in rank_str:
            if ch.isdigit():
                row.extend([''] * int(ch))
            else:
                row.append(ch)
        board.append(row)

    en_passant = None
    if ep_str != '-':
        col = ord(ep_str[0]) - ord('a')
        row = 8 - int(ep_str[1])
        en_passant = (row, col)

    return BoardState(
        board=board,
        turn=turn,
        castling=castling if castling != '-' else '',
        en_passant=en_passant,
        halfmove_clock=halfmove,
        fullmove_number=fullmove,
    )


# ─── Move generator ────────────────────────────────────────────────────────────

class MoveGenerator:
    """
    Generates all legal moves for the side to move.
    Call .legal_moves(state) for fully filtered moves,
    or .pseudo_legal_moves(state) for unfiltered (faster, used internally).
    """

    def legal_moves(self, state: BoardState) -> list[Move]:
        """Return only moves that don't leave own king in check."""
        moves = []
        for move in self.pseudo_legal_moves(state):
            next_state = self.apply_move(state, move)
            if not self._king_in_check(next_state, state.turn):
                moves.append(move)
        return moves

    def pseudo_legal_moves(self, state: BoardState) -> list[Move]:
        """All moves ignoring whether the king is left in check."""
        moves = []
        for r in range(8):
            for c in range(8):
                piece = state.piece_at(r, c)
                if not piece:
                    continue
                is_white_piece = piece.isupper()
                if state.turn == 'w' and not is_white_piece:
                    continue
                if state.turn == 'b' and is_white_piece:
                    continue
                moves.extend(self._piece_moves(state, r, c, piece.upper()))
        return moves

    # ── Per-piece move generation ──────────────────────────────────────────────

    def _piece_moves(self, state: BoardState, r: int, c: int, ptype: str) -> list[Move]:
        if ptype == 'P': return self._pawn_moves(state, r, c)
        if ptype == 'N': return self._knight_moves(state, r, c)
        if ptype == 'B': return self._slider_moves(state, r, c, BISHOP_DIRS)
        if ptype == 'R': return self._slider_moves(state, r, c, ROOK_DIRS)
        if ptype == 'Q': return self._slider_moves(state, r, c, QUEEN_DIRS)
        if ptype == 'K': return self._king_moves(state, r, c)
        return []

    def _pawn_moves(self, state: BoardState, r: int, c: int) -> list[Move]:
        moves = []
        direction = -1 if state.turn == 'w' else 1
        start_rank = 6 if state.turn == 'w' else 1
        promo_rank = 0 if state.turn == 'w' else 7

        # Single push
        nr = r + direction
        if 0 <= nr <= 7 and state.is_empty(nr, c):
            if nr == promo_rank:
                for p in ['q', 'r', 'b', 'n']:
                    moves.append(Move((r,c),(nr,c), promotion=p))
            else:
                moves.append(Move((r,c),(nr,c)))
            # Double push from start rank
            if r == start_rank:
                nr2 = r + 2 * direction
                if state.is_empty(nr2, c):
                    moves.append(Move((r,c),(nr2,c)))

        # Captures (diagonal)
        for dc in [-1, 1]:
            nc = c + dc
            if not (0 <= nc <= 7):
                continue
            nr = r + direction
            if 0 <= nr <= 7:
                # Normal capture
                if state.is_enemy(nr, nc):
                    if nr == promo_rank:
                        for p in ['q', 'r', 'b', 'n']:
                            moves.append(Move((r,c),(nr,nc), promotion=p))
                    else:
                        moves.append(Move((r,c),(nr,nc)))
                # En passant capture
                if state.en_passant == (nr, nc):
                    moves.append(Move((r,c),(nr,nc), is_en_passant=True))

        return moves

    def _knight_moves(self, state: BoardState, r: int, c: int) -> list[Move]:
        moves = []
        for dr, dc in KNIGHT_MOVES:
            nr, nc = r + dr, c + dc
            if 0 <= nr <= 7 and 0 <= nc <= 7:
                if not state.is_friendly(nr, nc):
                    moves.append(Move((r,c),(nr,nc)))
        return moves

    def _slider_moves(self, state: BoardState, r: int, c: int, dirs: list) -> list[Move]:
        moves = []
        for dr, dc in dirs:
            nr, nc = r + dr, c + dc
            while 0 <= nr <= 7 and 0 <= nc <= 7:
                if state.is_friendly(nr, nc):
                    break
                moves.append(Move((r,c),(nr,nc)))
                if state.is_enemy(nr, nc):
                    break
                nr += dr
                nc += dc
        return moves

    def _king_moves(self, state: BoardState, r: int, c: int) -> list[Move]:
        moves = []
        for dr, dc in KING_DIRS:
            nr, nc = r + dr, c + dc
            if 0 <= nr <= 7 and 0 <= nc <= 7:
                if not state.is_friendly(nr, nc):
                    moves.append(Move((r,c),(nr,nc)))
        moves.extend(self._castling_moves(state, r, c))
        return moves

    def _castling_moves(self, state: BoardState, r: int, c: int) -> list[Move]:
        """
        Generate castling moves with full legality checks:
        - King and rook haven't moved (via castling rights string)
        - Squares between them are empty
        - King does not pass through or land on an attacked square
        - King is not currently in check
        """
        moves = []
        if state.turn == 'w':
            if self._king_in_check(state, 'w'):
                return []
            # Kingside
            if 'K' in state.castling:
                if (state.is_empty(7,5) and state.is_empty(7,6)
                        and not self._square_attacked(state, 7, 5, 'b')
                        and not self._square_attacked(state, 7, 6, 'b')):
                    moves.append(Move((7,4),(7,6), is_castling=True))
            # Queenside
            if 'Q' in state.castling:
                if (state.is_empty(7,3) and state.is_empty(7,2) and state.is_empty(7,1)
                        and not self._square_attacked(state, 7, 3, 'b')
                        and not self._square_attacked(state, 7, 2, 'b')):
                    moves.append(Move((7,4),(7,2), is_castling=True))
        else:
            if self._king_in_check(state, 'b'):
                return []
            if 'k' in state.castling:
                if (state.is_empty(0,5) and state.is_empty(0,6)
                        and not self._square_attacked(state, 0, 5, 'w')
                        and not self._square_attacked(state, 0, 6, 'w')):
                    moves.append(Move((0,4),(0,6), is_castling=True))
            if 'q' in state.castling:
                if (state.is_empty(0,3) and state.is_empty(0,2) and state.is_empty(0,1)
                        and not self._square_attacked(state, 0, 3, 'w')
                        and not self._square_attacked(state, 0, 2, 'w')):
                    moves.append(Move((0,4),(0,2), is_castling=True))
        return moves

    # ── Apply move ────────────────────────────────────────────────────────────

    def apply_move(self, state: BoardState, move: Move) -> BoardState:
        """Return a new BoardState after applying the move."""
        ns = state.copy()
        fr, fc = move.from_sq
        tr, tc = move.to_sq
        piece = ns.board[fr][fc]

        # En passant
        ns.en_passant = None
        if move.is_en_passant:
            capture_row = fr   # the captured pawn is on the same rank as the moving pawn
            ns.board[capture_row][tc] = ''

        # Move the piece
        ns.board[tr][tc] = piece
        ns.board[fr][fc] = ''

        # Promotion
        if move.promotion:
            ns.board[tr][tc] = move.promotion.upper() if state.turn == 'w' else move.promotion.lower()

        # Castling: also move the rook
        if move.is_castling:
            if tc == 6:   # kingside
                rook_from_c, rook_to_c = 7, 5
            else:         # queenside
                rook_from_c, rook_to_c = 0, 3
            ns.board[tr][rook_to_c] = ns.board[tr][rook_from_c]
            ns.board[tr][rook_from_c] = ''

        # Set en passant square for double pawn push
        if piece.upper() == 'P' and abs(tr - fr) == 2:
            ep_row = (fr + tr) // 2
            ns.en_passant = (ep_row, fc)

        # Update castling rights
        castling = set(ns.castling)
        if piece == 'K': castling -= {'K', 'Q'}
        if piece == 'k': castling -= {'k', 'q'}
        if piece == 'R' and fr == 7:
            if fc == 7: castling.discard('K')
            if fc == 0: castling.discard('Q')
        if piece == 'r' and fr == 0:
            if fc == 7: castling.discard('k')
            if fc == 0: castling.discard('q')
        # Rook captured
        if tr == 7 and tc == 7: castling.discard('K')
        if tr == 7 and tc == 0: castling.discard('Q')
        if tr == 0 and tc == 7: castling.discard('k')
        if tr == 0 and tc == 0: castling.discard('q')
        ns.castling = ''.join(sorted(castling, key='KQkq'.index)) if castling else ''

        ns.turn = 'b' if state.turn == 'w' else 'w'
        return ns

    # ── Check / attack detection ──────────────────────────────────────────────

    def _king_in_check(self, state: BoardState, color: str) -> bool:
        """Is `color`'s king currently in check?"""
        king = 'K' if color == 'w' else 'k'
        king_pos = None
        for r in range(8):
            for c in range(8):
                if state.board[r][c] == king:
                    king_pos = (r, c)
                    break
        if king_pos is None:
            return False
        attacker = 'b' if color == 'w' else 'w'
        return self._square_attacked(state, king_pos[0], king_pos[1], attacker)

    def _square_attacked(self, state: BoardState, r: int, c: int, by: str) -> bool:
        """Is square (r,c) attacked by the given color?
        Uses direct per-piece checks (no castling) to avoid infinite recursion."""
        # Check knight attacks
        knight = 'N' if by == 'w' else 'n'
        for dr, dc in KNIGHT_MOVES:
            nr, nc = r + dr, c + dc
            if 0 <= nr <= 7 and 0 <= nc <= 7 and state.board[nr][nc] == knight:
                return True
        # Check pawn attacks
        pawn = 'P' if by == 'w' else 'p'
        pawn_dir = 1 if by == 'w' else -1  # direction pawns came FROM to attack (r,c)
        for dc in [-1, 1]:
            nr, nc = r + pawn_dir, c + dc
            if 0 <= nr <= 7 and 0 <= nc <= 7 and state.board[nr][nc] == pawn:
                return True
        # Check king attacks
        king = 'K' if by == 'w' else 'k'
        for dr, dc in KING_DIRS:
            nr, nc = r + dr, c + dc
            if 0 <= nr <= 7 and 0 <= nc <= 7 and state.board[nr][nc] == king:
                return True
        # Check slider attacks (bishop/queen diagonals)
        bishop = 'B' if by == 'w' else 'b'
        queen  = 'Q' if by == 'w' else 'q'
        for dr, dc in BISHOP_DIRS:
            nr, nc = r + dr, c + dc
            while 0 <= nr <= 7 and 0 <= nc <= 7:
                p = state.board[nr][nc]
                if p:
                    if p in (bishop, queen):
                        return True
                    break
                nr += dr; nc += dc
        # Check slider attacks (rook/queen straights)
        rook = 'R' if by == 'w' else 'r'
        for dr, dc in ROOK_DIRS:
            nr, nc = r + dr, c + dc
            while 0 <= nr <= 7 and 0 <= nc <= 7:
                p = state.board[nr][nc]
                if p:
                    if p in (rook, queen):
                        return True
                    break
                nr += dr; nc += dc
        return False

    # ── Helpers ───────────────────────────────────────────────────────────────

    def is_checkmate(self, state: BoardState) -> bool:
        return self._king_in_check(state, state.turn) and len(self.legal_moves(state)) == 0

    def is_stalemate(self, state: BoardState) -> bool:
        return not self._king_in_check(state, state.turn) and len(self.legal_moves(state)) == 0


# ─── Quick smoke tests ─────────────────────────────────────────────────────────

if __name__ == '__main__':
    gen = MoveGenerator()

    print("=== Test 1: Starting position move count ===")
    start_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    state = parse_fen(start_fen)
    moves = gen.legal_moves(state)
    print(f"Legal moves: {len(moves)}  (expected 20)")
    print(sorted([m.uci() for m in moves]))

    print("\n=== Test 2: En passant ===")
    # White pawn on e5, black just played d7-d5, so en passant on d6
    ep_fen = "rnbqkbnr/ppp1pppp/8/3pP3/8/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 3"
    state2 = parse_fen(ep_fen)
    ep_moves = [m for m in gen.legal_moves(state2) if m.is_en_passant]
    print(f"En passant moves found: {[m.uci() for m in ep_moves]}  (expected ['e5d6'])")

    print("\n=== Test 3: Castling ===")
    castle_fen = "r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w KQkq - 0 1"
    state3 = parse_fen(castle_fen)
    castle_moves = [m for m in gen.legal_moves(state3) if m.is_castling]
    print(f"Castling moves: {[m.uci() for m in castle_moves]}  (expected e1g1, e1c1)")

    print("\n=== Test 4: Checkmate detection ===")
    # Fool's mate position — black is checkmated
    fools_fen = "rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3"
    state4 = parse_fen(fools_fen)
    print(f"White in checkmate: {gen.is_checkmate(state4)}  (expected True)")

    print("\n=== Test 5: Pin (king must not move into check) ===")
    # White king on e1, white rook on e4, black rook on e8 — rook is pinned on e-file
    pin_fen = "4r3/8/8/8/4R3/8/8/4K3 w - - 0 1"
    state5 = parse_fen(pin_fen)
    rook_moves = [m for m in gen.legal_moves(state5)
                  if state5.board[m.from_sq[0]][m.from_sq[1]].upper() == 'R']
    print(f"Pinned rook can only move along pin ray (e-file):")
    print(sorted([m.uci() for m in rook_moves]))

    print("\nAll tests done.")