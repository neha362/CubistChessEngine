"""
berserker_eval_agent.py — "The Berserker" Evaluation Agent
============================================================
Personality: AGGRESSIVE / CHAOTIC
  The Berserker does not play chess. It wages war.

  Evaluation philosophy (in priority order):
    1. ATTACK PROXIMITY  — pieces near the enemy king score massively more
    2. MOBILITY PRESSURE — the more squares we attack, the better
    3. MATERIAL SACRIFICE — willingly undervalues its own pieces to push forward
    4. PAWN STORMS       — advanced enemy-king-side pawns score a huge bonus
    5. KING SAFETY       — the Berserker's OWN king safety is penalised
                           (it doesn't care — it will die gloriously)
    6. STANDARD MATERIAL — present but heavily discounted vs positional aggression

  This engine will:
    • Sacrifice pieces for open files toward the enemy king
    • March pawns directly at the castled king even when behind on material
    • Never retreat a piece if it can stay close to the enemy king
    • Prefer checks and near-checks over winning material

Standalone test:
  python berserker_eval_agent.py --test
  python berserker_eval_agent.py        (FEN → score breakdown)

Contract consumed by berserker_search_agent.py:
  evaluate(state: GameState) -> int      centipawns, White's perspective
  explain(state: GameState)  -> str      human-readable breakdown
"""

from __future__ import annotations
import sys
from typing import Optional

try:
    from movegen_agent import (
        GameState, from_fen, all_legal_moves,
        sq, row, col, STARTPOS, _is_square_attacked
    )
    _MOVEGEN_OK = True
except ImportError:
    _MOVEGEN_OK = False
    # Minimal shims for standalone use
    def sq(r, c): return r * 8 + c
    def row(i):   return i // 8
    def col(i):   return i % 8
    STARTPOS = 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1'
    FILES = 'abcdefgh'

    from dataclasses import dataclass
    @dataclass
    class GameState:
        board: list; turn: str; castling: dict
        ep_square: object; halfmove: int = 0; fullmove: int = 1
        def opponent(self): return 'b' if self.turn == 'w' else 'w'

    def from_fen(fen):
        _FM = {'P':'wP','N':'wN','B':'wB','R':'wR','Q':'wQ','K':'wK',
               'p':'bP','n':'bN','b':'bB','r':'bR','q':'bQ','k':'bK'}
        parts = fen.split(); board = [None]*64; r=0
        for rs in parts[0].split('/'):
            c=0
            for ch in rs:
                if ch.isdigit(): c+=int(ch)
                else: board[sq(r,c)]=_FM[ch]; c+=1
            r+=1
        cas=parts[2] if len(parts)>2 else 'KQkq'
        return GameState(board, parts[1] if len(parts)>1 else 'w',
                         {'wK':'K'in cas,'wQ':'Q'in cas,'bK':'k'in cas,'bQ':'q'in cas},
                         None, int(parts[4]) if len(parts)>4 else 0)

    def all_legal_moves(s): return []
    def _is_square_attacked(b, sq_, by): return False

def _color(p): return p[0] if p else None
def _type(p):  return p[1] if p else None

# ─────────────────────────────────────────────────────────────────────────────
# TUNING KNOBS — these define The Berserker's personality
# Increase any weight to make it more extreme in that dimension
# ─────────────────────────────────────────────────────────────────────────────
W_MATERIAL        =  0.35   # ← deliberately LOW  (material barely matters)
W_ATTACK_PROX     =  3.50   # ← HIGH  (proximity to enemy king is everything)
W_MOBILITY        =  1.20   # ← moderate (more attacks = more chaos)
W_PAWN_STORM      =  2.80   # ← HIGH  (marching pawns at the enemy king)
W_OWN_KING_SAFETY = -0.15   # ← NEAR ZERO / slightly negative (ignores safety)
W_INITIATIVE      =  1.80   # ← HIGH  (checks, pins, threats score extra)
W_CENTRE_CONTROL  =  0.40   # ← LOW   (centre is just a path to the king)

# ─────────────────────────────────────────────────────────────────────────────
# Material values (intentionally compressed — the Berserker doesn't care much)
# ─────────────────────────────────────────────────────────────────────────────
_BASE_MAT = {'P': 80, 'N': 250, 'B': 260, 'R': 380, 'Q': 700, 'K': 0}

# ─────────────────────────────────────────────────────────────────────────────
# Aggressive PSTs  (white perspective, index 0=a8 … 63=h1)
# Pieces are rewarded for being DEEP in enemy territory
# ─────────────────────────────────────────────────────────────────────────────
_AGGR_PST = {
    # Pawns: sprint up the board toward the enemy king
    'P': [
         0,  0,  0,  0,  0,  0,  0,  0,
        90, 95,100,105,105,100, 95, 90,   # rank 7 — huge pawn storm bonus
        55, 60, 65, 75, 75, 65, 60, 55,   # rank 6
        30, 35, 40, 55, 55, 40, 35, 30,   # rank 5
        15, 18, 22, 38, 38, 22, 18, 15,   # rank 4
         5,  8, 10, 12, 12, 10,  8,  5,   # rank 3
         0,  0,  0, -5, -5,  0,  0,  0,   # rank 2 (starting square — no bonus)
         0,  0,  0,  0,  0,  0,  0,  0,
    ],
    # Knights: love outposts deep in enemy territory
    'N': [
        -20,-10,  0,  5,  5,  0,-10,-20,
        -10,  0, 15, 20, 20, 15,  0,-10,
          0, 15, 30, 40, 40, 30, 15,  0,
          5, 20, 40, 50, 50, 40, 20,  5,
          5, 20, 40, 50, 50, 40, 20,  5,
          0, 15, 30, 35, 35, 30, 15,  0,
        -10,  0, 15, 20, 20, 15,  0,-10,
        -30,-20,-10, -5, -5,-10,-20,-30,
    ],
    # Bishops: long diagonals pointing at the enemy king
    'B': [
        -10, -5, -5, -5, -5, -5, -5,-10,
         -5, 10, 10, 10, 10, 10, 10, -5,
         -5, 10, 20, 25, 25, 20, 10, -5,
         -5, 12, 25, 30, 30, 25, 12, -5,
         -5, 12, 25, 30, 30, 25, 12, -5,
         -5, 10, 20, 25, 25, 20, 10, -5,
         -5, 10, 10, 10, 10, 10, 10, -5,
        -10, -5, -5, -5, -5, -5, -5,-10,
    ],
    # Rooks: open files, 7th rank, pointed at the king
    'R': [
         20, 25, 25, 30, 30, 25, 25, 20,   # rank 8 — enemy back rank invasion
         40, 45, 45, 50, 50, 45, 45, 40,   # rank 7 — THE rook rank
         15, 20, 20, 25, 25, 20, 20, 15,
          5, 10, 10, 15, 15, 10, 10,  5,
          0,  5,  5, 10, 10,  5,  5,  0,
         -5,  0,  0,  5,  5,  0,  0, -5,
        -10, -5, -5,  0,  0, -5, -5,-10,
          0,  0,  5,  5,  5,  5,  0,  0,
    ],
    # Queens: charge to the enemy king's doorstep
    'Q': [
        -10,  0,  0,  5,  5,  0,  0,-10,
          0, 15, 15, 20, 20, 15, 15,  0,
          0, 15, 25, 30, 30, 25, 15,  0,
          5, 20, 30, 40, 40, 30, 20,  5,
          5, 20, 30, 40, 40, 30, 20,  5,
          0, 15, 25, 30, 30, 25, 15,  0,
          0, 15, 15, 20, 20, 15, 15,  0,
        -10,  0,  0,  5,  5,  0,  0,-10,
    ],
    # King: STAY BACK (but barely penalised — the Berserker ignores its own king)
    'K': [
        -10,-15,-20,-25,-25,-20,-15,-10,
        -10,-15,-20,-25,-25,-20,-15,-10,
        -10,-15,-20,-25,-25,-20,-15,-10,
        -10,-15,-20,-25,-25,-20,-15,-10,
        -10,-15,-20,-25,-25,-20,-15,-10,
         -5,-10,-15,-20,-20,-15,-10, -5,
          5,  0, -5,-10,-10, -5,  0,  5,
         10, 15,  5, -5, -5,  5, 15, 10,
    ],
}

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def _king_square(board: list, color: str) -> int:
    for i, p in enumerate(board):
        if p == color + 'K':
            return i
    return -1

def _chebyshev(i: int, j: int) -> int:
    """Maximum of rank-distance and file-distance (king-move distance)."""
    return max(abs(row(i) - row(j)), abs(col(i) - col(j)))

def _manhattan(i: int, j: int) -> int:
    return abs(row(i) - row(j)) + abs(col(i) - col(j))

def _pawn_storm_bonus(board: list, attacker: str, enemy_king_sq: int) -> int:
    """
    Reward pawns of `attacker` color that are storming toward the enemy king's
    file cluster (files within 2 of the king).
    The closer the pawn, the bigger the bonus.
    """
    king_col = col(enemy_king_sq)
    bonus    = 0
    for i, p in enumerate(board):
        if p != attacker + 'P':
            continue
        c_ = col(i)
        if abs(c_ - king_col) <= 2:          # on the storm files
            # How far has this pawn advanced?
            advance = (7 - row(i)) if attacker == 'w' else row(i)
            bonus  += advance * advance * 4   # quadratic bonus for advancement
    return bonus

def _attack_proximity_bonus(board: list, attacker: str, enemy_king_sq: int) -> int:
    """
    Every attacking piece scores a bonus inversely proportional to its
    Chebyshev distance from the enemy king.
    Distance 1 = 80 pts, distance 2 = 40, distance 3 = 20, etc.
    """
    bonus = 0
    for i, p in enumerate(board):
        if not p or p[0] != attacker or p[1] == 'K':
            continue
        dist  = _chebyshev(i, enemy_king_sq)
        if dist == 0: dist = 1
        bonus += 80 // dist
    return bonus

def _initiative_bonus(state: GameState, attacker: str) -> int:
    """
    Reward for:
      • Having checks available  (+60 per checking move)
      • Attacking squares adjacent to the enemy king  (+10 per attacked square)
    Only computed if movegen is available (returns 0 otherwise).
    """
    if not _MOVEGEN_OK:
        return 0

    board    = state.board
    opp      = 'b' if attacker == 'w' else 'w'
    king_sq  = _king_square(board, opp)
    if king_sq == -1:
        return 0

    bonus = 0
    # Attacked squares near enemy king
    adj_squares = [
        sq(row(king_sq)+dr, col(king_sq)+dc)
        for dr in (-1,0,1) for dc in (-1,0,1)
        if 0 <= row(king_sq)+dr < 8 and 0 <= col(king_sq)+dc < 8
        and (dr, dc) != (0, 0)
    ]
    for sq_ in adj_squares:
        if _is_square_attacked(board, sq_, attacker):
            bonus += 10

    # Checks: use a side-state with attacker to move
    from dataclasses import replace
    check_state = GameState(
        board     = board[:],
        turn      = attacker,
        castling  = dict(state.castling),
        ep_square = state.ep_square,
    )
    for frm, to, _ in all_legal_moves(check_state):
        from movegen_agent import make_move as _mm
        ns = _mm(check_state, (frm, to, ''))
        from movegen_agent import is_in_check as _chk
        if _chk(ns, opp):
            bonus += 60

    return bonus

def _mobility_score(state: GameState, attacker: str) -> int:
    """Count legal moves for attacker minus legal moves for defender."""
    if not _MOVEGEN_OK:
        return 0
    opp = 'b' if attacker == 'w' else 'w'
    s_att = GameState(state.board[:], attacker, dict(state.castling),
                      state.ep_square, state.halfmove, state.fullmove)
    s_def = GameState(state.board[:], opp,      dict(state.castling),
                      state.ep_square, state.halfmove, state.fullmove)
    return len(all_legal_moves(s_att)) - len(all_legal_moves(s_def))

# ─────────────────────────────────────────────────────────────────────────────
# Main evaluation
# ─────────────────────────────────────────────────────────────────────────────
def evaluate(state: GameState) -> int:
    """
    Returns centipawns from WHITE's perspective.
    Positive = White better, negative = Black better.
    The Berserker heavily rewards aggression over material safety.
    """
    board = state.board

    w_king = _king_square(board, 'w')
    b_king = _king_square(board, 'b')

    # ── 1. Material (discounted) + aggressive PST ─────────────────────────
    mat_pst = 0
    for i, p in enumerate(board):
        if not p: continue
        t, tp = p[0], p[1]
        sign  = 1 if t == 'w' else -1
        mat   = _BASE_MAT.get(tp, 0)
        idx   = i if t == 'w' else 63 - i
        pst   = _AGGR_PST.get(tp, [0]*64)[idx]
        mat_pst += sign * (mat + pst)

    # ── 2. Attack proximity to enemy king ────────────────────────────────
    w_prox = _attack_proximity_bonus(board, 'w', b_king) if b_king != -1 else 0
    b_prox = _attack_proximity_bonus(board, 'b', w_king) if w_king != -1 else 0
    prox_score = w_prox - b_prox

    # ── 3. Pawn storms ───────────────────────────────────────────────────
    w_storm = _pawn_storm_bonus(board, 'w', b_king) if b_king != -1 else 0
    b_storm = _pawn_storm_bonus(board, 'b', w_king) if w_king != -1 else 0
    storm_score = w_storm - b_storm

    # ── 4. Mobility ──────────────────────────────────────────────────────
    mob = _mobility_score(state, 'w') * 8   # 8 cp per extra legal move

    # ── 5. Initiative (checks / king-zone attacks) ───────────────────────
    #    Expensive — only called when it's this side's turn (saved for root)
    init_w = _initiative_bonus(state, 'w')
    init_b = _initiative_bonus(state, 'b')
    init_score = init_w - init_b

    # ── 6. Own king safety (nearly ignored by the Berserker) ────────────
    #    Standard king-zone attacker count, but weighted very low
    w_king_danger = 0
    b_king_danger = 0
    if w_king != -1:
        w_king_danger = sum(
            1 for i, p in enumerate(board)
            if p and p[0] == 'b' and _chebyshev(i, w_king) <= 2
        ) * 15
    if b_king != -1:
        b_king_danger = sum(
            1 for i, p in enumerate(board)
            if p and p[0] == 'w' and _chebyshev(i, b_king) <= 2
        ) * 15
    safety_score = b_king_danger - w_king_danger   # more attackers near opp king = good

    # ── Weighted total ───────────────────────────────────────────────────
    total = int(
        W_MATERIAL        * mat_pst    +
        W_ATTACK_PROX     * prox_score +
        W_PAWN_STORM      * storm_score +
        W_MOBILITY        * mob        +
        W_INITIATIVE      * init_score +
        W_OWN_KING_SAFETY * safety_score
    )
    return total


def explain(state: GameState) -> str:
    """Return a human-readable breakdown of the Berserker's evaluation."""
    board  = state.board
    w_king = _king_square(board, 'w')
    b_king = _king_square(board, 'b')

    mat_pst = sum(
        (1 if p[0]=='w' else -1) * (_BASE_MAT.get(p[1],0) +
        _AGGR_PST.get(p[1],[0]*64)[i if p[0]=='w' else 63-i])
        for i, p in enumerate(board) if p
    )
    w_prox  = _attack_proximity_bonus(board, 'w', b_king)  if b_king != -1 else 0
    b_prox  = _attack_proximity_bonus(board, 'b', w_king)  if w_king != -1 else 0
    w_storm = _pawn_storm_bonus(board, 'w', b_king)        if b_king != -1 else 0
    b_storm = _pawn_storm_bonus(board, 'b', w_king)        if w_king != -1 else 0
    mob     = _mobility_score(state, 'w') * 8

    total = evaluate(state)
    lines = [
        "┌─ THE BERSERKER EVALUATION ──────────────────┐",
       f"│  Material + aggressive PST  : {mat_pst:+7.0f} × {W_MATERIAL:.2f} = {W_MATERIAL*mat_pst:+8.1f}",
       f"│  Attack proximity (W-B)     : {w_prox-b_prox:+7d} × {W_ATTACK_PROX:.2f} = {W_ATTACK_PROX*(w_prox-b_prox):+8.1f}",
       f"│  Pawn storm (W-B)           : {w_storm-b_storm:+7d} × {W_PAWN_STORM:.2f} = {W_PAWN_STORM*(w_storm-b_storm):+8.1f}",
       f"│  Mobility (moves diff)      : {mob:+7d} × {W_MOBILITY:.2f} = {W_MOBILITY*mob:+8.1f}",
       f"│  ─────────────────────────────────────────── │",
       f"│  TOTAL                      : {total:+8d} cp             │",
        "└─────────────────────────────────────────────┘",
        "",
        "  Personality: The Berserker cares almost nothing about material.",
        "  It LIVES for proximity to the enemy king and advancing pawn storms.",
    ]
    return '\n'.join(lines)


# ── Self-test ─────────────────────────────────────────────────────────────────
def _run_tests():
    print("berserker_eval_agent self-test — The Berserker")
    print("=" * 56)
    ok = True

    tests = [
        (
            "start position near-zero",
            STARTPOS,
            lambda s: abs(evaluate(s)) < 200,
            "|score| < 200 cp"
        ),
        (
            "rook on 7th rank scores higher than home square",
            # White rook on e7 (aggressive) vs home position
            '4k3/4R3/8/8/8/8/8/4K3 w - - 0 1',
            lambda s: evaluate(s) > evaluate(from_fen('4k3/8/8/8/8/8/8/4KR2 w - - 0 1')),
            "rook on 7th rank > rook on 1st rank"
        ),
        (
            "advanced pawn storm valued highly",
            # White pawns on f6/g6/h6 charging the black king on g8
            '6k1/8/5PPP/8/8/8/8/4K3 w - - 0 1',
            lambda s: evaluate(s) > 200,
            "score > 200 cp (pawn storm bonus)"
        ),
        (
            "White piece near enemy king scores more than equivalent piece far away",
            # White queen on h6 (near black king g8) vs queen on a1
            '6k1/8/7Q/8/8/8/8/4K3 w - - 0 1',
            lambda s: evaluate(s) > evaluate(from_fen('6k1/8/8/8/8/8/8/Q3K3 w - - 0 1')),
            "queen near enemy king > queen on a1"
        ),
        (
            "material deficit forgiven if attack proximity compensates",
            # White is a rook down but has huge attacking presence near black king
            '6k1/5ppp/4NNB1/8/8/8/8/4K3 w - - 0 1',
            lambda s: evaluate(s) > -300,
            "score > -300 despite rook deficit"
        ),
    ]

    for desc, fen, cond, expected in tests:
        state  = from_fen(fen)
        passed = cond(state)
        score  = evaluate(state)
        ok    &= passed
        print(f"  [{'PASS' if passed else 'FAIL'}] {desc}")
        print(f"         score={score:+d} cp  ({expected})")

    print("=" * 56)
    print("All tests PASSED" if ok else "Some tests FAILED")
    return ok


if __name__ == '__main__':
    if '--test' in sys.argv:
        sys.exit(0 if _run_tests() else 1)
    while True:
        fen = input("\nFEN (blank to quit): ").strip()
        if not fen: break
        try:
            state = from_fen(fen)
            print()
            print(explain(state))
        except Exception as e:
            print(f"Error: {e}")