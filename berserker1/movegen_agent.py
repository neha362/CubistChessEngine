"""
movegen_agent.py — Move Generation Agent (MCTS Edition)
=========================================================
Same legal-move contract as before, but with additions optimised
for Monte Carlo simulation throughput:

  • fast_random_move(state)  — O(1) random legal move for rollouts
  • is_terminal(state)       — quick terminal check without full movegen
  • game_result(state)       — returns 1.0/0.5/0.0 for MCTS backprop

Public contract (consumed by mcts_agent.py and rollout_agent.py):
  all_legal_moves(state)    -> list[Move]
  fast_random_move(state)   -> Move | None
  make_move(state, move)    -> GameState
  is_in_check(state, color) -> bool
  is_terminal(state)        -> bool
  game_result(state)        -> float   (1.0=White wins, 0.0=Black, 0.5=draw)
  from_fen(fen)             -> GameState
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import random
import sys

FILES = 'abcdefgh'

def sq(r: int, c: int) -> int:  return r * 8 + c
def row(i: int) -> int:         return i // 8
def col(i: int) -> int:         return i % 8
def sq_name(i: int) -> str:     return FILES[col(i)] + str(8 - row(i))
def parse_sq(s: str) -> Optional[int]:
    if len(s) < 2 or s[0] not in FILES or s[1] not in '12345678':
        return None
    return sq(8 - int(s[1]), FILES.index(s[0]))

def color(p: Optional[str]) -> Optional[str]:  return p[0] if p else None
def ptype(p: Optional[str]) -> Optional[str]:  return p[1] if p else None

# ── GameState ────────────────────────────────────────────────────────────────
@dataclass
class GameState:
    board:      list
    turn:       str
    castling:   dict
    ep_square:  Optional[int]
    halfmove:   int = 0
    fullmove:   int = 1

    def copy(self) -> 'GameState':
        return GameState(self.board[:], self.turn, dict(self.castling),
                         self.ep_square, self.halfmove, self.fullmove)

    def opponent(self) -> str:
        return 'b' if self.turn == 'w' else 'w'

# ── FEN ──────────────────────────────────────────────────────────────────────
_FEN_MAP = {
    'P':'wP','N':'wN','B':'wB','R':'wR','Q':'wQ','K':'wK',
    'p':'bP','n':'bN','b':'bB','r':'bR','q':'bQ','k':'bK',
}
STARTPOS = 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1'

def from_fen(fen: str) -> GameState:
    parts = fen.split()
    board = [None] * 64
    r = 0
    for rank_str in parts[0].split('/'):
        c = 0
        for ch in rank_str:
            if ch.isdigit(): c += int(ch)
            else:            board[sq(r, c)] = _FEN_MAP[ch]; c += 1
        r += 1
    turn     = parts[1] if len(parts) > 1 else 'w'
    cas      = parts[2] if len(parts) > 2 else 'KQkq'
    castling = {'wK':'K' in cas,'wQ':'Q' in cas,'bK':'k' in cas,'bQ':'q' in cas}
    ep_s     = parts[3] if len(parts) > 3 else '-'
    ep       = parse_sq(ep_s) if ep_s != '-' else None
    hm       = int(parts[4]) if len(parts) > 4 else 0
    fm       = int(parts[5]) if len(parts) > 5 else 1
    return GameState(board, turn, castling, ep, hm, fm)

# ── Attack detection (reverse-ray) ───────────────────────────────────────────
def _is_square_attacked(board: list, square: int, by: str) -> bool:
    r, c_ = row(square), col(square)
    for dr, dc in [(-2,-1),(-2,1),(-1,-2),(-1,2),(1,-2),(1,2),(2,-1),(2,1)]:
        nr, nc = r+dr, c_+dc
        if 0 <= nr < 8 and 0 <= nc < 8:
            p = board[sq(nr,nc)]
            if p and color(p) == by and ptype(p) == 'N':
                return True
    for dirs, types in [([(-1,-1),(-1,1),(1,-1),(1,1)],{'B','Q'}),
                        ([(-1,0),(1,0),(0,-1),(0,1)],  {'R','Q'})]:
        for dr, dc in dirs:
            nr, nc = r+dr, c_+dc
            while 0 <= nr < 8 and 0 <= nc < 8:
                p = board[sq(nr,nc)]
                if p:
                    if color(p) == by and ptype(p) in types: return True
                    break
                nr += dr; nc += dc
    for dr, dc in [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]:
        nr, nc = r+dr, c_+dc
        if 0 <= nr < 8 and 0 <= nc < 8:
            p = board[sq(nr,nc)]
            if p and color(p) == by and ptype(p) == 'K': return True
    pawn_dir = 1 if by == 'w' else -1
    for dc in (-1, 1):
        nr, nc = r+pawn_dir, c_+dc
        if 0 <= nr < 8 and 0 <= nc < 8:
            p = board[sq(nr,nc)]
            if p and color(p) == by and ptype(p) == 'P': return True
    return False

def is_in_check(state: GameState, col_: Optional[str] = None) -> bool:
    c      = col_ or state.turn
    king   = next((i for i in range(64) if state.board[i] == c+'K'), -1)
    return king != -1 and _is_square_attacked(state.board, king,
                                              'b' if c == 'w' else 'w')

# ── Pseudo-legal move generation ─────────────────────────────────────────────
def _pseudo_moves(state: GameState) -> list:
    moves  = []
    board  = state.board
    turn   = state.turn
    opp    = state.opponent()

    for frm in range(64):
        p = board[frm]
        if not p or color(p) != turn: continue
        tp    = ptype(p)
        r, c_ = row(frm), col(frm)

        if tp == 'P':
            d         = -1 if turn == 'w' else 1
            start_row =  6 if turn == 'w' else 1
            promo_row =  0 if turn == 'w' else 7
            fwd = sq(r+d, c_)
            if 0 <= r+d < 8 and not board[fwd]:
                if row(fwd) == promo_row:
                    for pr in 'qrbn': moves.append((frm, fwd, pr))
                else:
                    moves.append((frm, fwd, ''))
                if r == start_row and not board[sq(r+2*d, c_)]:
                    moves.append((frm, sq(r+2*d, c_), ''))
            for dc in (-1, 1):
                nc = c_+dc
                if 0 <= nc < 8:
                    to = sq(r+d, nc)
                    if (board[to] and color(board[to]) == opp) or to == state.ep_square:
                        if row(to) == promo_row:
                            for pr in 'qrbn': moves.append((frm, to, pr))
                        else:
                            moves.append((frm, to, ''))

        elif tp == 'N':
            for dr, dc in [(-2,-1),(-2,1),(-1,-2),(-1,2),(1,-2),(1,2),(2,-1),(2,1)]:
                nr, nc = r+dr, c_+dc
                if 0 <= nr < 8 and 0 <= nc < 8:
                    to = sq(nr, nc)
                    if not board[to] or color(board[to]) == opp:
                        moves.append((frm, to, ''))

        elif tp in ('B','R','Q'):
            dirs = []
            if tp in ('B','Q'): dirs += [(-1,-1),(-1,1),(1,-1),(1,1)]
            if tp in ('R','Q'): dirs += [(-1,0),(1,0),(0,-1),(0,1)]
            for dr, dc in dirs:
                nr, nc = r+dr, c_+dc
                while 0 <= nr < 8 and 0 <= nc < 8:
                    to = sq(nr, nc)
                    if board[to]:
                        if color(board[to]) == opp: moves.append((frm, to, ''))
                        break
                    moves.append((frm, to, ''))
                    nr += dr; nc += dc

        elif tp == 'K':
            for dr, dc in [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]:
                nr, nc = r+dr, c_+dc
                if 0 <= nr < 8 and 0 <= nc < 8:
                    to = sq(nr, nc)
                    if not board[to] or color(board[to]) == opp:
                        moves.append((frm, to, ''))
            if turn == 'w' and frm == 60:
                if state.castling.get('wK') and not board[61] and not board[62]:
                    moves.append((60,62,''))
                if state.castling.get('wQ') and not board[59] and not board[58] and not board[57]:
                    moves.append((60,58,''))
            if turn == 'b' and frm == 4:
                if state.castling.get('bK') and not board[5] and not board[6]:
                    moves.append((4,6,''))
                if state.castling.get('bQ') and not board[3] and not board[2] and not board[1]:
                    moves.append((4,2,''))
    return moves

# ── Apply move ────────────────────────────────────────────────────────────────
def make_move(state: GameState, move: tuple) -> GameState:
    frm, to, promo = move
    ns    = state.copy()
    board = ns.board
    piece = board[frm]
    tp, t = ptype(piece), color(piece)

    board[to]    = piece
    board[frm]   = None
    ns.ep_square = None

    if tp == 'P':
        if to == state.ep_square:
            board[sq(row(frm), col(to))] = None
        if abs(row(to)-row(frm)) == 2:
            ns.ep_square = sq((row(frm)+row(to))//2, col(frm))
        if promo:
            board[to] = t + promo.upper()
        ns.halfmove = 0
    else:
        ns.halfmove = 0 if board[to] else ns.halfmove + 1

    if tp == 'K':
        if t == 'w': ns.castling['wK'] = ns.castling['wQ'] = False
        else:        ns.castling['bK'] = ns.castling['bQ'] = False
        if   col(to)-col(frm) ==  2: board[to-1]=board[to+1]; board[to+1]=None
        elif col(frm)-col(to) ==  2: board[to+1]=board[to-4]; board[to-4]=None

    for s in (frm, to):
        if s == 56: ns.castling['wQ'] = False
        if s == 63: ns.castling['wK'] = False
        if s ==  0: ns.castling['bQ'] = False
        if s ==  7: ns.castling['bK'] = False

    ns.turn = ns.opponent()
    if ns.turn == 'w': ns.fullmove += 1
    return ns

# ── Legal move generation ─────────────────────────────────────────────────────
def all_legal_moves(state: GameState) -> list:
    legal = []
    opp   = state.opponent()
    for move in _pseudo_moves(state):
        frm, to, _ = move
        p = state.board[frm]
        if ptype(p) == 'K' and abs(col(to)-col(frm)) == 2:
            step = 1 if col(to) > col(frm) else -1
            mid  = sq(row(frm), col(frm)+step)
            if (_is_square_attacked(state.board, frm, opp) or
                _is_square_attacked(state.board, mid, opp)):
                continue
        ns = make_move(state, move)
        if not is_in_check(ns, state.turn):
            legal.append(move)
    return legal

# ── Terminal helpers ──────────────────────────────────────────────────────────
def game_status(state: GameState) -> str:
    if state.halfmove >= 100 or state.fullmove > 200:
        return 'draw'
    moves = all_legal_moves(state)
    if not moves:
        return 'checkmate' if is_in_check(state) else 'stalemate'
    if set(p for p in state.board if p) <= {'wK','bK'}:
        return 'insufficient'
    return 'ongoing'

def is_terminal(state: GameState) -> bool:
    return game_status(state) != 'ongoing'

def game_result(state: GameState) -> float:
    """
    1.0 = White wins, 0.0 = Black wins, 0.5 = draw.
    Call only on terminal states.
    """
    status = game_status(state)
    if status == 'checkmate':
        return 0.0 if state.turn == 'w' else 1.0
    return 0.5

# ── Fast random move for rollouts ─────────────────────────────────────────────
def fast_random_move(state: GameState) -> Optional[tuple]:
    """
    Return a random legal move without building the full legal list.
    Shuffles pseudo-legal moves and returns the first legal one.
    This is the hot path inside rollout_agent — keep it lean.
    """
    pseudo = _pseudo_moves(state)
    random.shuffle(pseudo)
    opp = state.opponent()
    for move in pseudo:
        frm, to, _ = move
        p = state.board[frm]
        if ptype(p) == 'K' and abs(col(to)-col(frm)) == 2:
            step = 1 if col(to) > col(frm) else -1
            mid  = sq(row(frm), col(frm)+step)
            if (_is_square_attacked(state.board, frm, opp) or
                _is_square_attacked(state.board, mid, opp)):
                continue
        ns = make_move(state, move)
        if not is_in_check(ns, state.turn):
            return move
    return None

# ── Perft ────────────────────────────────────────────────────────────────────
def perft(state: GameState, depth: int) -> int:
    if depth == 0: return 1
    moves = all_legal_moves(state)
    if depth == 1: return len(moves)
    return sum(perft(make_move(state, m), depth-1) for m in moves)

def perft_divide(state: GameState, depth: int) -> dict:
    return {sq_name(f)+sq_name(t)+pr: perft(make_move(state,(f,t,pr)), depth-1)
            for f,t,pr in all_legal_moves(state)}

# ── Self-test ─────────────────────────────────────────────────────────────────
def _run_tests():
    state = from_fen(STARTPOS)
    print("movegen_agent self-test  (MCTS edition)")
    print("=" * 46)
    ok = True
    for d, exp in {1:20, 2:400, 3:8902}.items():
        got    = perft(state, d)
        passed = got == exp; ok &= passed
        print(f"  perft({d}) = {got:>8,}  expected {exp:>8,}  [{'PASS' if passed else 'FAIL'}]")

    ep_ok = 'e5d6' in [sq_name(f)+sq_name(t)
                        for f,t,_ in all_legal_moves(from_fen('8/8/8/3pP3/8/8/8/4K2k w - d6 0 1'))]
    ok &= ep_ok
    print(f"  en-passant e5xd6:              [{'PASS' if ep_ok else 'FAIL'}]")

    cas = [sq_name(f)+sq_name(t) for f,t,_ in all_legal_moves(from_fen('r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1'))]
    cas_ok = 'e1g1' in cas and 'e1c1' in cas; ok &= cas_ok
    print(f"  castling e1g1 / e1c1:          [{'PASS' if cas_ok else 'FAIL'}]")

    rnd_ok = fast_random_move(from_fen(STARTPOS)) is not None; ok &= rnd_ok
    print(f"  fast_random_move startpos:     [{'PASS' if rnd_ok else 'FAIL'}]")

    fool   = from_fen('rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3')
    gr_ok  = game_result(fool) == 0.0; ok &= gr_ok
    print(f"  game_result fool's mate=0.0:   [{'PASS' if gr_ok else 'FAIL'}]")

    print("=" * 46)
    print("All tests PASSED" if ok else "Some tests FAILED")
    return ok

if __name__ == '__main__':
    if '--test' in sys.argv:
        sys.exit(0 if _run_tests() else 1)
    fen   = input("FEN (blank=startpos): ").strip() or STARTPOS
    depth = int(input("Perft depth: ").strip())
    div   = perft_divide(from_fen(fen), depth)
    for lbl, cnt in sorted(div.items()): print(f"  {lbl}: {cnt:,}")
    print(f"Total: {sum(div.values()):,}")