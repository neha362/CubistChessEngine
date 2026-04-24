"""
Chess Engine — Universal Chess Interface (UCI) compliant
--------------------------------------------------------
Run directly:  python chess_engine_uci.py

Any UCI-compatible GUI (Arena, CuteChess, PyChess, Banksia, etc.)
can connect to it as an external engine.

UCI spec: https://www.shredderchess.com/chess-features/uci-universal-chess-interface.html

Protocol summary implemented here:
  GUI → Engine          Engine → GUI
  ──────────────────    ─────────────────────────────
  uci                   id name / id author / uciok
  isready               readyok
  ucinewgame            (resets state)
  position ...          (sets board from startpos / fen + moves)
  go [movetime N]       bestmove <move> [ponder <move>]
  quit                  (exits cleanly)
  stop                  (aborts search, outputs bestmove)
"""

import sys
import math
import time

# ── Piece-square tables (white's perspective; index 0 = a8, 63 = h1) ────────
PST = {
    'P': [
         0,  0,  0,  0,  0,  0,  0,  0,
        50, 50, 50, 50, 50, 50, 50, 50,
        10, 10, 20, 30, 30, 20, 10, 10,
         5,  5, 10, 25, 25, 10,  5,  5,
         0,  0,  0, 20, 20,  0,  0,  0,
         5, -5,-10,  0,  0,-10, -5,  5,
         5, 10, 10,-20,-20, 10, 10,  5,
         0,  0,  0,  0,  0,  0,  0,  0,
    ],
    'N': [
        -50,-40,-30,-30,-30,-30,-40,-50,
        -40,-20,  0,  0,  0,  0,-20,-40,
        -30,  0, 10, 15, 15, 10,  0,-30,
        -30,  5, 15, 20, 20, 15,  5,-30,
        -30,  0, 15, 20, 20, 15,  0,-30,
        -30,  5, 10, 15, 15, 10,  5,-30,
        -40,-20,  0,  5,  5,  0,-20,-40,
        -50,-40,-30,-30,-30,-30,-40,-50,
    ],
    'B': [
        -20,-10,-10,-10,-10,-10,-10,-20,
        -10,  0,  0,  0,  0,  0,  0,-10,
        -10,  0,  5, 10, 10,  5,  0,-10,
        -10,  5,  5, 10, 10,  5,  5,-10,
        -10,  0, 10, 10, 10, 10,  0,-10,
        -10, 10, 10, 10, 10, 10, 10,-10,
        -10,  5,  0,  0,  0,  0,  5,-10,
        -20,-10,-10,-10,-10,-10,-10,-20,
    ],
    'R': [
         0,  0,  0,  0,  0,  0,  0,  0,
         5, 10, 10, 10, 10, 10, 10,  5,
        -5,  0,  0,  0,  0,  0,  0, -5,
        -5,  0,  0,  0,  0,  0,  0, -5,
        -5,  0,  0,  0,  0,  0,  0, -5,
        -5,  0,  0,  0,  0,  0,  0, -5,
        -5,  0,  0,  0,  0,  0,  0, -5,
         0,  0,  0,  5,  5,  0,  0,  0,
    ],
    'Q': [
        -20,-10,-10, -5, -5,-10,-10,-20,
        -10,  0,  0,  0,  0,  0,  0,-10,
        -10,  0,  5,  5,  5,  5,  0,-10,
         -5,  0,  5,  5,  5,  5,  0, -5,
          0,  0,  5,  5,  5,  5,  0, -5,
        -10,  5,  5,  5,  5,  5,  0,-10,
        -10,  0,  5,  0,  0,  0,  0,-10,
        -20,-10,-10, -5, -5,-10,-10,-20,
    ],
    'K': [
        -30,-40,-40,-50,-50,-40,-40,-30,
        -30,-40,-40,-50,-50,-40,-40,-30,
        -30,-40,-40,-50,-50,-40,-40,-30,
        -30,-40,-40,-50,-50,-40,-40,-30,
        -20,-30,-30,-40,-40,-30,-30,-20,
        -10,-20,-20,-20,-20,-20,-20,-10,
         20, 20,  0,  0,  0,  0, 20, 20,
         20, 30, 10,  0,  0, 10, 30, 20,
    ],
}

PIECE_VAL = {'P': 100, 'N': 320, 'B': 330, 'R': 500, 'Q': 900, 'K': 20000}
FILES = 'abcdefgh'

# ── Square helpers ───────────────────────────────────────────────────────────
def sq(r, c):   return r * 8 + c
def row(i):     return i // 8
def col(i):     return i % 8
def pc(p):      return p[0] if p else None   # 'w' or 'b'
def pt(p):      return p[1] if p else None   # 'K','Q','R','B','N','P'

def score_move(board, frm, to, promo):
    p = board[frm]
    if not p:
        return 0
    attacker_type = pt(p)
    turn = pc(p)
    opp = 'b' if turn == 'w' else 'w'
    score = 0
    victim = board[to]
    if victim and pc(victim) == opp:
        score = PIECE_VAL[pt(victim)] * 10 - PIECE_VAL[attacker_type]
    elif attacker_type == 'P' and not victim and col(frm) != col(to):
        ep_sq = sq(row(frm), col(to))
        ep_v = board[ep_sq]
        if ep_v and pc(ep_v) == opp and pt(ep_v) == 'P':
            score = PIECE_VAL['P'] * 10 - PIECE_VAL[attacker_type]
    if promo:
        score += PIECE_VAL[promo.upper()]
    return score

def sq_name(i):
    """Index → algebraic  e.g. 0→'a8', 63→'h1'"""
    return FILES[col(i)] + str(8 - row(i))

def parse_sq(s):
    """Algebraic → index  e.g. 'e2'→52"""
    if len(s) < 2:
        return None
    f, r = s[0], s[1]
    if f not in FILES or r not in '12345678':
        return None
    return sq(8 - int(r), FILES.index(f))

# ── FEN parsing ──────────────────────────────────────────────────────────────
FEN_PIECE = {
    'P':'wP','N':'wN','B':'wB','R':'wR','Q':'wQ','K':'wK',
    'p':'bP','n':'bN','b':'bB','r':'bR','q':'bQ','k':'bK',
}

def parse_fen(fen):
    """
    Return (board, turn, castling, en_passant)
    board is a 64-element list, index 0 = a8.
    """
    parts = fen.split()
    board = [None] * 64
    r = 0
    for rank_str in parts[0].split('/'):
        c = 0
        for ch in rank_str:
            if ch.isdigit():
                c += int(ch)
            else:
                board[sq(r, c)] = FEN_PIECE[ch]
                c += 1
        r += 1

    turn = 'w' if parts[1] == 'w' else 'b'

    cas_str = parts[2] if len(parts) > 2 else '-'
    castling = {
        'wK': 'K' in cas_str,
        'wQ': 'Q' in cas_str,
        'bK': 'k' in cas_str,
        'bQ': 'q' in cas_str,
    }

    ep_str = parts[3] if len(parts) > 3 else '-'
    en_passant = parse_sq(ep_str) if ep_str != '-' else None

    return board, turn, castling, en_passant

STARTPOS_FEN = 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1'

# ── Move generation ──────────────────────────────────────────────────────────
def raw_moves(board, frm, turn, en_passant, castling):
    p   = board[frm]
    tp  = pt(p)
    opp = 'b' if turn == 'w' else 'w'
    r, c = row(frm), col(frm)
    moves = []

    def push(to):
        if 0 <= to < 64:
            moves.append(to)

    def slide(dr, dc):
        nr, nc = r + dr, c + dc
        while 0 <= nr < 8 and 0 <= nc < 8:
            s = sq(nr, nc)
            if board[s]:
                if pc(board[s]) == opp:
                    push(s)
                break
            push(s)
            nr += dr; nc += dc

    if tp == 'P':
        d = -1 if turn == 'w' else 1
        start_row = 6 if turn == 'w' else 1
        fwd = sq(r + d, c)
        if 0 <= r + d < 8 and not board[fwd]:
            push(fwd)
            if r == start_row and not board[sq(r + 2*d, c)]:
                push(sq(r + 2*d, c))
        for dc in (-1, 1):
            nc2 = c + dc
            if 0 <= nc2 < 8:
                ts = sq(r + d, nc2)
                if (board[ts] and pc(board[ts]) == opp) or en_passant == ts:
                    push(ts)

    elif tp == 'N':
        for dr, dc in [(-2,-1),(-2,1),(-1,-2),(-1,2),(1,-2),(1,2),(2,-1),(2,1)]:
            nr2, nc2 = r + dr, c + dc
            if 0 <= nr2 < 8 and 0 <= nc2 < 8:
                s = sq(nr2, nc2)
                if not board[s] or pc(board[s]) == opp:
                    push(s)

    elif tp == 'B':
        for dr, dc in [(-1,-1),(-1,1),(1,-1),(1,1)]:
            slide(dr, dc)

    elif tp == 'R':
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            slide(dr, dc)

    elif tp == 'Q':
        for dr, dc in [(-1,-1),(-1,1),(1,-1),(1,1),(-1,0),(1,0),(0,-1),(0,1)]:
            slide(dr, dc)

    elif tp == 'K':
        for dr, dc in [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]:
            nr2, nc2 = r + dr, c + dc
            if 0 <= nr2 < 8 and 0 <= nc2 < 8:
                s = sq(nr2, nc2)
                if not board[s] or pc(board[s]) == opp:
                    push(s)
        if turn == 'w' and r == 7 and c == 4:
            if castling.get('wK') and not board[61] and not board[62]:
                if not _sq_attacked(board,'w',60) and not _sq_attacked(board,'w',61) and not _sq_attacked(board,'w',62):
                    push(62)
            if castling.get('wQ') and not board[59] and not board[58] and not board[57]:
                if not _sq_attacked(board,'w',60) and not _sq_attacked(board,'w',59) and not _sq_attacked(board,'w',58):
                    push(58)
        if turn == 'b' and r == 0 and c == 4:
            if castling.get('bK') and not board[5] and not board[6]:
                if not _sq_attacked(board,'b',4) and not _sq_attacked(board,'b',5) and not _sq_attacked(board,'b',6):
                    push(6)
            if castling.get('bQ') and not board[3] and not board[2] and not board[1]:
                if not _sq_attacked(board,'b',4) and not _sq_attacked(board,'b',3) and not _sq_attacked(board,'b',2):
                    push(2)

    return moves


def _sq_attacked(board, color, square):
    opp = 'b' if color == 'w' else 'w'
    pseudo = _all_pseudo(board, opp, None, {'wK':False,'wQ':False,'bK':False,'bQ':False})
    return any(to == square for _, to in pseudo)


def _all_pseudo(board, turn, en_passant, castling):
    moves = []
    for frm in range(64):
        if board[frm] and pc(board[frm]) == turn:
            for to in raw_moves(board, frm, turn, en_passant, castling):
                moves.append((frm, to))
    return moves


def all_moves(board, turn, en_passant, castling):
    """Return fully legal (frm, to, promo) triples."""
    legal = []
    for frm in range(64):
        if not board[frm] or pc(board[frm]) != turn:
            continue
        for to in raw_moves(board, frm, turn, en_passant, castling):
            # check promotions
            promos = _promotion_pieces(board, frm, to, turn)
            for promo in promos:
                nb, _, _ = apply_move(board, frm, to, en_passant, castling, promo)
                if not in_check(nb, turn):
                    legal.append((frm, to, promo))
    return legal


def _promotion_pieces(board, frm, to, turn):
    """Return list of promotion suffixes; [''] for non-promotions."""
    p = board[frm]
    if pt(p) != 'P':
        return ['']
    if (turn == 'w' and row(to) == 0) or (turn == 'b' and row(to) == 7):
        return ['q', 'r', 'b', 'n']
    return ['']


def in_check(board, turn):
    king = next((i for i in range(64) if board[i] == turn + 'K'), -1)
    return _sq_attacked(board, turn, king)


def apply_move(board, frm, to, en_passant, castling, promo=''):
    nb     = board[:]
    piece  = nb[frm]
    tp, t  = pt(piece), pc(piece)
    nb[to] = piece
    nb[frm] = None
    nep    = None
    ncas   = dict(castling)

    if tp == 'P':
        if to == en_passant:
            nb[sq(row(frm), col(to))] = None
        if abs(row(to) - row(frm)) == 2:
            nep = sq((row(frm) + row(to)) // 2, col(frm))
        if (t == 'w' and row(to) == 0) or (t == 'b' and row(to) == 7):
            promo_type = promo.upper() if promo else 'Q'
            nb[to] = t + promo_type

    if tp == 'K':
        ncas['wK'] = ncas['wQ'] = False if t == 'w' else (ncas['wK'], ncas['wQ'])
        ncas['bK'] = ncas['bQ'] = False if t == 'b' else (ncas['bK'], ncas['bQ'])
        if t == 'w':
            ncas['wK'] = ncas['wQ'] = False
        else:
            ncas['bK'] = ncas['bQ'] = False
        if col(to) - col(frm) == 2:
            nb[to - 1] = nb[to + 1]; nb[to + 1] = None
        elif col(frm) - col(to) == 2:
            nb[to + 1] = nb[to - 4]; nb[to - 4] = None

    if frm == 56 or to == 56: ncas['wQ'] = False
    if frm == 63 or to == 63: ncas['wK'] = False
    if frm == 0  or to == 0:  ncas['bQ'] = False
    if frm == 7  or to == 7:  ncas['bK'] = False

    return nb, nep, ncas

# ── UCI move encoding / decoding ─────────────────────────────────────────────
def move_to_uci(frm, to, promo=''):
    """(52, 36, '') → 'e2e4'   (12, 4, 'q') → 'e7e8q'"""
    return sq_name(frm) + sq_name(to) + (promo or '')

def uci_to_move(uci_str):
    """'e2e4' → (52, 36, '')   'e7e8q' → (12, 4, 'q')"""
    frm   = parse_sq(uci_str[:2])
    to    = parse_sq(uci_str[2:4])
    promo = uci_str[4] if len(uci_str) > 4 else ''
    return frm, to, promo

# ── Evaluation ───────────────────────────────────────────────────────────────
def evaluate(board):
    score = 0
    for i, p in enumerate(board):
        if not p:
            continue
        t, tp = pc(p), pt(p)
        val      = PIECE_VAL[tp]
        pst_idx  = i if t == 'w' else 63 - i
        pst      = PST.get(tp, [0]*64)[pst_idx]
        score   += (val + pst) if t == 'w' else -(val + pst)
    return score

# ── Search ───────────────────────────────────────────────────────────────────
# Global flag that lets the GUI send "stop" mid-search
_stop = False

def minimax(board, depth, alpha, beta, maximising, en_passant, castling):
    global _stop
    if _stop:
        return 0

    turn  = 'w' if maximising else 'b'
    moves = all_moves(board, turn, en_passant, castling)
    moves.sort(key=lambda m: score_move(board, m[0], m[1], m[2]), reverse=True)

    if depth == 0 or not moves:
        if not moves:
            return (-99999 if maximising else 99999) if in_check(board, turn) else 0
        return evaluate(board)

    if maximising:
        value = -math.inf
        for frm, to, promo in moves:
            nb, nep, ncas = apply_move(board, frm, to, en_passant, castling, promo)
            value = max(value, minimax(nb, depth-1, alpha, beta, False, nep, ncas))
            alpha = max(alpha, value)
            if alpha >= beta:
                break
        return value
    else:
        value = math.inf
        for frm, to, promo in moves:
            nb, nep, ncas = apply_move(board, frm, to, en_passant, castling, promo)
            value = min(value, minimax(nb, depth-1, alpha, beta, True, nep, ncas))
            beta  = min(beta, value)
            if alpha >= beta:
                break
        return value


def search(board, turn, en_passant, castling, depth=3, movetime_ms=None):
    """
    Find best move for `turn`.
    Returns UCI move string, e.g. 'e2e4' or 'e7e8q'.
    Optionally respects movetime_ms budget.
    """
    global _stop
    _stop = False
    moves = all_moves(board, turn, en_passant, castling)
    moves.sort(key=lambda m: score_move(board, m[0], m[1], m[2]), reverse=True)
    if not moves:
        return None

    maximising = (turn == 'w')
    best_move  = moves[0]
    best_val   = -math.inf if maximising else math.inf
    start      = time.time()

    for frm, to, promo in moves:
        if _stop:
            break
        if movetime_ms and (time.time() - start) * 1000 > movetime_ms * 0.9:
            break
        nb, nep, ncas = apply_move(board, frm, to, en_passant, castling, promo)
        val = minimax(nb, depth-1, -math.inf, math.inf, not maximising, nep, ncas)
        if maximising and val > best_val:
            best_val  = val
            best_move = (frm, to, promo)
        elif not maximising and val < best_val:
            best_val  = val
            best_move = (frm, to, promo)

    elapsed_ms = int((time.time() - start) * 1000)
    send(f'info depth {depth} score cp {best_val} time {elapsed_ms} '
         f'pv {move_to_uci(*best_move)}')
    return move_to_uci(*best_move)

# ── UCI I/O helpers ──────────────────────────────────────────────────────────
def send(msg):
    """Write a line to stdout and flush immediately."""
    print(msg, flush=True)

def log(msg):
    """Optional stderr debug log — GUI never sees this."""
    print(f'# {msg}', file=sys.stderr, flush=True)

# ── Position command parser ──────────────────────────────────────────────────
def handle_position(tokens, state):
    """
    Handles:
      position startpos [moves e2e4 e7e5 ...]
      position fen <fen_string> [moves ...]
    Updates state in-place.
    """
    idx = 0
    if tokens[idx] == 'startpos':
        board, turn, castling, ep = parse_fen(STARTPOS_FEN)
        idx += 1
    elif tokens[idx] == 'fen':
        idx += 1
        fen_parts = []
        while idx < len(tokens) and tokens[idx] != 'moves':
            fen_parts.append(tokens[idx])
            idx += 1
        board, turn, castling, ep = parse_fen(' '.join(fen_parts))
    else:
        return  # unknown; ignore

    # Apply any moves listed after the position
    if idx < len(tokens) and tokens[idx] == 'moves':
        idx += 1
        while idx < len(tokens):
            uci = tokens[idx]
            frm, to, promo = uci_to_move(uci)
            board, ep, castling = apply_move(board, frm, to, ep, castling, promo)
            turn = 'b' if turn == 'w' else 'w'
            idx += 1

    state['board']      = board
    state['turn']       = turn
    state['castling']   = castling
    state['ep']         = ep

# ── Go command parser ────────────────────────────────────────────────────────
def handle_go(tokens, state):
    """
    Supported sub-commands:
      movetime <ms>   — think for at most N milliseconds
      depth <n>       — search to fixed depth (default 3)
      infinite        — search until 'stop' (uses depth 3)
      wtime/btime     — rough time management (uses 1/30 of remaining)
    """
    global _stop
    _stop = False

    params     = {}
    i = 0
    while i < len(tokens):
        key = tokens[i]
        if i + 1 < len(tokens) and tokens[i+1].lstrip('-').isdigit():
            params[key] = int(tokens[i+1])
            i += 2
        else:
            params[key] = True
            i += 1

    depth      = params.get('depth', 3)
    movetime   = params.get('movetime', None)

    # Basic time management
    turn = state['turn']
    if movetime is None:
        if turn == 'w' and 'wtime' in params:
            movetime = max(100, params['wtime'] // 30)
        elif turn == 'b' and 'btime' in params:
            movetime = max(100, params['btime'] // 30)

    best = search(
        state['board'], state['turn'],
        state['ep'], state['castling'],
        depth=depth, movetime_ms=movetime
    )
    send(f'bestmove {best if best else "0000"}')

# ── Main UCI loop ─────────────────────────────────────────────────────────────
def uci_loop():
    global _stop

    # Persistent game state
    state = {
        'board':    None,
        'turn':     'w',
        'castling': {'wK': True, 'wQ': True, 'bK': True, 'bQ': True},
        'ep':       None,
    }
    # Start with standard position
    state['board'], state['turn'], state['castling'], state['ep'] = parse_fen(STARTPOS_FEN)

    while True:
        try:
            line = input().strip()
        except EOFError:
            break

        if not line:
            continue

        tokens = line.split()
        cmd    = tokens[0]

        # ── uci ──────────────────────────────────────────────────────────────
        if cmd == 'uci':
            send('id name PythonChessEngine')
            send('id author Claude')
            # Expose search depth as a UCI option
            send('option name Depth type spin default 3 min 1 max 6')
            send('uciok')

        # ── debug ─────────────────────────────────────────────────────────────
        elif cmd == 'debug':
            pass  # not implemented

        # ── isready ───────────────────────────────────────────────────────────
        elif cmd == 'isready':
            send('readyok')

        # ── setoption ─────────────────────────────────────────────────────────
        elif cmd == 'setoption':
            # setoption name Depth value 4
            try:
                name_idx  = tokens.index('name')  + 1
                value_idx = tokens.index('value') + 1
                opt_name  = tokens[name_idx].lower()
                if opt_name == 'depth':
                    state['default_depth'] = int(tokens[value_idx])
            except (ValueError, IndexError):
                pass

        # ── ucinewgame ────────────────────────────────────────────────────────
        elif cmd == 'ucinewgame':
            state['board'], state['turn'], state['castling'], state['ep'] = \
                parse_fen(STARTPOS_FEN)

        # ── position ──────────────────────────────────────────────────────────
        elif cmd == 'position':
            handle_position(tokens[1:], state)

        # ── go ────────────────────────────────────────────────────────────────
        elif cmd == 'go':
            # Respect setoption Depth if set
            if 'depth' not in tokens and 'default_depth' in state:
                tokens = tokens + ['depth', str(state['default_depth'])]
            handle_go(tokens[1:], state)

        # ── stop ──────────────────────────────────────────────────────────────
        elif cmd == 'stop':
            _stop = True

        # ── quit ──────────────────────────────────────────────────────────────
        elif cmd == 'quit':
            break

        # ── ponderhit ─────────────────────────────────────────────────────────
        elif cmd == 'ponderhit':
            pass  # pondering not implemented

        else:
            log(f'Unknown command: {line}')


if __name__ == '__main__':
    uci_loop()
