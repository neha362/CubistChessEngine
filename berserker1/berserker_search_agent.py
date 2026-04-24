"""
berserker_search_agent.py — "The Berserker" Search Agent
==========================================================
Personality: AGGRESSIVE / CHAOTIC

Search architecture:
  • Negamax with alpha-beta pruning
  • Iterative deepening (depth 1 → max_depth)
  • Transposition table (Zobrist-keyed, 512k entries)
  • Move ordering tuned for aggression:
      1. TT move
      2. Checks (the Berserker LOVES giving check)
      3. Captures ordered by MVV-LVA
      4. Promotions
      5. Moves toward the enemy king (proximity heuristic)
      6. Killer heuristic for quiet moves
  • NO quiescence search — the Berserker charges blindly into tactical chaos
    and lets the biased evaluator sort it out (this is the personality feature,
    not a bug — a calm engine would add quiescence)
  • UCI-compatible output (info lines + bestmove)

The Berserker will:
  ✓ Prefer checking sequences over winning material
  ✓ Choose the move that lands pieces closest to the enemy king
  ✓ Ignore horizon-effect losses — it doesn't look past its nose
  ✓ Sometimes find brilliancies; sometimes hang its queen
  ✓ Be terrifying to play against in blitz

Standalone:
  python berserker_search_agent.py --test
  python berserker_search_agent.py       (interactive UCI-style session)

Contract (UCI layer):
  search(state, max_depth, movetime_ms) -> BerserkerResult
"""

from __future__ import annotations
import math, time, random, sys
from dataclasses import dataclass, field
from typing import Optional

from movegen_agent import (
    GameState, from_fen, make_move, all_legal_moves,
    is_in_check, is_terminal, game_result, game_status,
    sq_name, STARTPOS, sq, row, col,
    _is_square_attacked
)
from berserker_eval_agent import evaluate as _berserker_eval, _king_square

AGENT_NAME   = "The Berserker"
AGENT_FLAVOR = "🩸 It does not play chess. It wages war."

# ── Piece values for MVV-LVA ──────────────────────────────────────────────────
_MVV = {'P':1,'N':3,'B':3,'R':5,'Q':9,'K':0}

# ── Zobrist hashing ───────────────────────────────────────────────────────────
random.seed(0xBE2523)
_ZP = {p: [random.getrandbits(64) for _ in range(64)]
       for p in ['wP','wN','wB','wR','wQ','wK','bP','bN','bB','bR','bQ','bK']}
_ZT  = random.getrandbits(64)
_ZC  = {k: random.getrandbits(64) for k in ['wK','wQ','bK','bQ']}
_ZEP = [random.getrandbits(64) for _ in range(8)]

def _zobrist(state: GameState) -> int:
    h = 0
    for i, p in enumerate(state.board):
        if p: h ^= _ZP[p][i]
    if state.turn == 'b': h ^= _ZT
    for k, v in state.castling.items():
        if v: h ^= _ZC[k]
    if state.ep_square is not None: h ^= _ZEP[state.ep_square % 8]
    return h

# ── Transposition table ───────────────────────────────────────────────────────
TT_EXACT, TT_LOWER, TT_UPPER = 0, 1, 2
_TT_SIZE = 1 << 19   # 512k slots

@dataclass
class _TTEntry:
    key: int; depth: int; flag: int; score: int; move: Optional[tuple]

_tt: dict[int, _TTEntry] = {}

def _tt_probe(key: int) -> Optional[_TTEntry]:
    e = _tt.get(key & (_TT_SIZE - 1))
    return e if (e and e.key == key) else None

def _tt_store(key: int, depth: int, flag: int, score: int, move: Optional[tuple]):
    idx = key & (_TT_SIZE - 1)
    e   = _tt.get(idx)
    if not e or e.key == key or depth >= e.depth:
        _tt[idx] = _TTEntry(key, depth, flag, score, move)

# ── Berserker move ordering ───────────────────────────────────────────────────
def _is_check_move(state: GameState, move: tuple) -> bool:
    """Does this move give check? (Berserker's top priority.)"""
    try:
        ns  = make_move(state, move)
        opp = 'b' if state.turn == 'w' else 'w'
        return is_in_check(ns, opp)
    except Exception:
        return False

def _proximity_to_enemy_king(state: GameState, move: tuple) -> int:
    """How close does this move land the piece to the enemy king? (lower=closer)"""
    board    = state.board
    opp      = 'b' if state.turn == 'w' else 'w'
    king_sq  = _king_square(board, opp)
    if king_sq == -1: return 99
    return max(abs(row(move[1]) - row(king_sq)), abs(col(move[1]) - col(king_sq)))

_killers: list[list] = [[] for _ in range(128)]

def _score_move(move: tuple, state: GameState,
                tt_move: Optional[tuple], ply: int) -> int:
    """
    Move ordering score — higher = search first.
    The Berserker's ordering is deliberately biased:
      checks >> captures >> proximity to enemy king >> killers >> quiet
    """
    frm, to, promo = move
    victim   = state.board[to]
    attacker = state.board[frm]

    if move == tt_move:
        return 10_000_000

    # Checks: the Berserker loves giving check more than winning material
    if _is_check_move(state, move):
        return 9_000_000

    # Captures: MVV-LVA
    if victim:
        return 5_000_000 + _MVV.get(victim[1],0)*10 - _MVV.get(attacker[1] if attacker else 'P', 0)

    # Promotion
    if promo == 'q':
        return 4_500_000

    # Proximity to enemy king — pieces closer to the enemy king go first
    dist = _proximity_to_enemy_king(state, move)
    prox_score = (7 - dist) * 50_000   # distance 0 → 350k, distance 7 → 0

    # Killers
    kl_bonus = 100_000 if move in _killers[ply] else 0

    return prox_score + kl_bonus

def _order_moves(moves: list, state: GameState,
                 tt_move: Optional[tuple], ply: int) -> list:
    return sorted(moves,
                  key=lambda m: _score_move(m, state, tt_move, ply),
                  reverse=True)

# ── Negamax ───────────────────────────────────────────────────────────────────
_nodes   = [0]
_stop    = [False]

def _negamax(state: GameState, depth: int, ply: int,
             alpha: int, beta: int) -> int:
    if _stop[0]: return 0
    _nodes[0] += 1

    key    = _zobrist(state)
    tt_ent = _tt_probe(key)
    tt_mv  = None

    if tt_ent:
        tt_mv = tt_ent.move
        if tt_ent.depth >= depth:
            s = tt_ent.score
            if   tt_ent.flag == TT_EXACT: return s
            elif tt_ent.flag == TT_LOWER: alpha = max(alpha, s)
            elif tt_ent.flag == TT_UPPER: beta  = min(beta,  s)
            if alpha >= beta: return s

    # Terminal
    status = game_status(state)
    if status == 'checkmate': return -(99_000 - ply)
    if status in ('stalemate','draw','insufficient'): return 0

    # Leaf — NO quiescence search (Berserker charges blindly)
    if depth == 0:
        score = _berserker_eval(state)
        # Negate for Black (negamax convention)
        return score if state.turn == 'w' else -score

    moves = all_legal_moves(state)
    if not moves: return 0

    moves     = _order_moves(moves, state, tt_mv, ply)
    best_move = None
    orig_alpha = alpha

    for move in moves:
        if _stop[0]: break
        ns    = make_move(state, move)
        score = -_negamax(ns, depth-1, ply+1, -beta, -alpha)

        if score > alpha:
            alpha     = score
            best_move = move
            if alpha >= beta:
                # Quiet move caused cutoff → killer
                if not state.board[move[1]] and not move[2]:
                    kl = _killers[ply]
                    if move not in kl:
                        kl.insert(0, move)
                        if len(kl) > 2: kl.pop()
                break

    if not _stop[0]:
        flag = (TT_EXACT if orig_alpha < alpha < beta
                else TT_LOWER if alpha >= beta
                else TT_UPPER)
        _tt_store(key, depth, flag, alpha, best_move)

    return alpha

# ── Result ────────────────────────────────────────────────────────────────────
@dataclass
class BerserkerResult:
    move:       Optional[tuple]
    score:      int
    depth:      int
    nodes:      int
    time_ms:    int
    pv:         list = field(default_factory=list)

    @property
    def uci(self) -> str:
        if not self.move: return '0000'
        f, t, pr = self.move
        return sq_name(f) + sq_name(t) + pr

    def info_str(self) -> str:
        nps = int(self.nodes / max(self.time_ms/1000, 1e-9))
        return (f"info depth {self.depth} score cp {self.score} "
                f"nodes {self.nodes} nps {nps} time {self.time_ms} "
                f"pv {self.uci}")

# ── Iterative deepening entry point ───────────────────────────────────────────
def search(state:       GameState,
           max_depth:   int           = 5,
           movetime_ms: Optional[int] = None,
           verbose:     bool          = True) -> BerserkerResult:
    """
    Iterative deepening search using The Berserker's biased evaluator.
    Respects movetime_ms budget if given.
    """
    global _killers
    _killers  = [[] for _ in range(128)]
    _tt.clear()
    _nodes[0] = 0
    _stop[0]  = False

    start  = time.time()
    result = BerserkerResult(None, 0, 0, 0, 0)
    moves  = all_legal_moves(state)
    if not moves:
        return result

    for d in range(1, max_depth + 1):
        elapsed = (time.time() - start) * 1000
        if movetime_ms and elapsed > movetime_ms * 0.85:
            break

        best_move  = None
        best_score = -math.inf
        alpha      = -math.inf
        beta       =  math.inf

        # Order root moves using same heuristic
        tt_ent  = _tt_probe(_zobrist(state))
        tt_mv   = tt_ent.move if tt_ent else None
        ordered = _order_moves(moves, state, tt_mv, 0)

        for move in ordered:
            if _stop[0]: break
            if movetime_ms and (time.time()-start)*1000 > movetime_ms*0.92:
                _stop[0] = True; break

            ns    = make_move(state, move)
            score = -_negamax(ns, d-1, 1, -beta, -alpha)

            if score > best_score:
                best_score = score
                best_move  = move
                alpha      = max(alpha, score)

        if not _stop[0] and best_move:
            elapsed_ms = int((time.time()-start)*1000)
            result = BerserkerResult(best_move, best_score, d,
                                     _nodes[0], elapsed_ms)
            if verbose:
                print(result.info_str(), flush=True)

    return result

# ── Self-test ─────────────────────────────────────────────────────────────────
def _run_tests():
    print(f"berserker_search_agent self-test — {AGENT_NAME}")
    print(f"  {AGENT_FLAVOR}")
    print("=" * 58)
    ok = True

    # 1. Returns a valid move from startpos
    state  = from_fen(STARTPOS)
    r      = search(state, max_depth=2, verbose=False)
    v1     = r.move is not None
    ok    &= v1
    print(f"  [{'PASS' if v1 else 'FAIL'}] startpos returns a move: {r.uci}")

    # 2. Prefers a check over a quiet move when available
    # White queen can give check on h5 or just move quietly
    check_pos = from_fen('rnbqkb1r/pppp1ppp/5n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4')
    rc = search(check_pos, max_depth=2, verbose=False)
    # Qxf7+ is a forcing check/attack — Berserker should prefer it
    print(f"  [INFO] check/attack position — Berserker plays: {rc.uci}  score={rc.score}")

    # 3. Finds mate in 1
    mate1  = from_fen('6k1/5ppp/8/8/8/8/5PPP/4R1K1 w - - 0 1')
    rm     = search(mate1, max_depth=2, verbose=False)
    m1_ok  = rm.uci == 'e1e8'
    ok    &= m1_ok
    print(f"  [{'PASS' if m1_ok else 'FAIL'}] mate-in-1 Re8#: {rm.uci}")

    # 4. Moves toward the enemy king (proximity ordering test)
    # White knight on d4 can jump to e6 (near black king g8) or b3 (far)
    prox_pos = from_fen('6k1/8/8/8/3N4/8/8/4K3 w - - 0 1')
    rp       = search(prox_pos, max_depth=2, verbose=False)
    print(f"  [INFO] proximity test — Berserker plays: {rp.uci}  (wants to be near g8)")

    # 5. Time-limited search completes within budget
    t0  = time.time()
    search(from_fen(STARTPOS), movetime_ms=400, verbose=False)
    dt  = (time.time()-t0)*1000
    t_ok = dt < 800
    ok  &= t_ok
    print(f"  [{'PASS' if t_ok else 'FAIL'}] movetime=400ms ran in {dt:.0f}ms")

    # 6. TT is populated after search
    search(from_fen(STARTPOS), max_depth=2, verbose=False)
    tt_ok = len(_tt) > 0; ok &= tt_ok
    print(f"  [{'PASS' if tt_ok else 'FAIL'}] TT populated: {len(_tt)} entries")

    print("=" * 58)
    print("All tests PASSED" if ok else "Some tests FAILED")
    return ok


# ── Interactive UCI-style shell ───────────────────────────────────────────────
def _uci_shell():
    print(f"\n  {AGENT_NAME}  —  {AGENT_FLAVOR}")
    print("  Commands: 'position <fen>'  |  'go depth <n>'  |  'go movetime <ms>'  |  'quit'\n")
    state = from_fen(STARTPOS)
    while True:
        try:
            line = input(">>> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not line: continue
        tokens = line.split()
        if tokens[0] == 'quit': break
        elif tokens[0] == 'position':
            fen = ' '.join(tokens[1:]) or STARTPOS
            state = from_fen(fen)
            print(f"  Position set.")
        elif tokens[0] == 'go':
            depth = 4; mt = None
            if 'depth' in tokens:
                depth = int(tokens[tokens.index('depth')+1])
            if 'movetime' in tokens:
                mt = int(tokens[tokens.index('movetime')+1])
            r = search(state, max_depth=depth, movetime_ms=mt, verbose=True)
            print(f"bestmove {r.uci}")
        else:
            print(f"  Unknown command: {line}")


if __name__ == '__main__':
    if '--test' in sys.argv:
        sys.exit(0 if _run_tests() else 1)
    _uci_shell()