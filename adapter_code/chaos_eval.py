from __future__ import annotations
import sys
from typing import Optional

try:
    from chaos_move_gen import GameState, from_fen, all_legal_moves, sq, row, col, STARTPOS, _is_square_attacked
    _MOVEGEN_OK = True
except ImportError:
    _MOVEGEN_OK = False
    def sq(r,c): return r*8+c
    def row(i): return i//8
    def col(i): return i%8
    STARTPOS='rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1'
    from dataclasses import dataclass
    @dataclass
    class GameState:
        board: list; turn: str; castling: dict; ep_square: object; halfmove: int=0; fullmove: int=1
        def opponent(self): return 'b' if self.turn=='w' else 'w'
    def from_fen(fen):
        _FM={'P':'wP','N':'wN','B':'wB','R':'wR','Q':'wQ','K':'wK','p':'bP','n':'bN','b':'bB','r':'bR','q':'bQ','k':'bK'}
        parts=fen.split(); board=[None]*64; r=0
        for rs in parts[0].split('/'):
            c=0
            for ch in rs:
                if ch.isdigit(): c+=int(ch)
                else: board[sq(r,c)]=_FM[ch]; c+=1
            r+=1
        cas=parts[2] if len(parts)>2 else 'KQkq'
        return GameState(board,parts[1] if len(parts)>1 else 'w',
                         {'wK':'K' in cas,'wQ':'Q' in cas,'bK':'k' in cas,'bQ':'q' in cas},None,
                         int(parts[4]) if len(parts)>4 else 0)
    def all_legal_moves(s): return []
    def _is_square_attacked(b,sq_,by): return False

def _color(p): return p[0] if p else None
def _type(p):  return p[1] if p else None

W_MATERIAL=0.35; W_ATTACK_PROX=3.50; W_MOBILITY=1.20; W_PAWN_STORM=2.80
W_OWN_KING_SAFETY=-0.15; W_INITIATIVE=1.80; W_CENTRE_CONTROL=0.40

_BASE_MAT={'P':80,'N':250,'B':260,'R':380,'Q':700,'K':0}

_AGGR_PST={
    'P':[0,0,0,0,0,0,0,0,90,95,100,105,105,100,95,90,55,60,65,75,75,65,60,55,30,35,40,55,55,40,35,30,15,18,22,38,38,22,18,15,5,8,10,12,12,10,8,5,0,0,0,-5,-5,0,0,0,0,0,0,0,0,0,0,0],
    'N':[-20,-10,0,5,5,0,-10,-20,-10,0,15,20,20,15,0,-10,0,15,30,40,40,30,15,0,5,20,40,50,50,40,20,5,5,20,40,50,50,40,20,5,0,15,30,35,35,30,15,0,-10,0,15,20,20,15,0,-10,-30,-20,-10,-5,-5,-10,-20,-30],
    'B':[-10,-5,-5,-5,-5,-5,-5,-10,-5,10,10,10,10,10,10,-5,-5,10,20,25,25,20,10,-5,-5,12,25,30,30,25,12,-5,-5,12,25,30,30,25,12,-5,-5,10,20,25,25,20,10,-5,-5,10,10,10,10,10,10,-5,-10,-5,-5,-5,-5,-5,-5,-10],
    'R':[20,25,25,30,30,25,25,20,40,45,45,50,50,45,45,40,15,20,20,25,25,20,20,15,5,10,10,15,15,10,10,5,0,5,5,10,10,5,5,0,-5,0,0,5,5,0,0,-5,-10,-5,-5,0,0,-5,-5,-10,0,0,5,5,5,5,0,0],
    'Q':[-10,0,0,5,5,0,0,-10,0,15,15,20,20,15,15,0,0,15,25,30,30,25,15,0,5,20,30,40,40,30,20,5,5,20,30,40,40,30,20,5,0,15,25,30,30,25,15,0,0,15,15,20,20,15,15,0,-10,0,0,5,5,0,0,-10],
    'K':[-10,-15,-20,-25,-25,-20,-15,-10,-10,-15,-20,-25,-25,-20,-15,-10,-10,-15,-20,-25,-25,-20,-15,-10,-10,-15,-20,-25,-25,-20,-15,-10,-10,-15,-20,-25,-25,-20,-15,-10,-5,-10,-15,-20,-20,-15,-10,-5,5,0,-5,-10,-10,-5,0,5,10,15,5,-5,-5,5,15,10],
}

def _king_square(board,color):
    for i,p in enumerate(board):
        if p==color+'K': return i
    return -1

def _chebyshev(i,j): return max(abs(row(i)-row(j)),abs(col(i)-col(j)))

def _pawn_storm_bonus(board,attacker,enemy_king_sq):
    king_col=col(enemy_king_sq); bonus=0
    for i,p in enumerate(board):
        if p!=attacker+'P': continue
        c_=col(i)
        if abs(c_-king_col)<=2:
            advance=(7-row(i)) if attacker=='w' else row(i)
            bonus+=advance*advance*4
    return bonus

def _attack_proximity_bonus(board,attacker,enemy_king_sq):
    bonus=0
    for i,p in enumerate(board):
        if not p or p[0]!=attacker or p[1]=='K': continue
        dist=_chebyshev(i,enemy_king_sq)
        if dist==0: dist=1
        bonus+=80//dist
    return bonus

def _mobility_score(state,attacker):
    if not _MOVEGEN_OK: return 0
    opp='b' if attacker=='w' else 'w'
    s_att=GameState(state.board[:],attacker,dict(state.castling),state.ep_square,state.halfmove,state.fullmove)
    s_def=GameState(state.board[:],opp,dict(state.castling),state.ep_square,state.halfmove,state.fullmove)
    return len(all_legal_moves(s_att))-len(all_legal_moves(s_def))

def evaluate(state):
    board=state.board
    w_king=_king_square(board,'w'); b_king=_king_square(board,'b')
    mat_pst=0
    for i,p in enumerate(board):
        if not p: continue
        t,tp=p[0],p[1]; sign=1 if t=='w' else -1
        mat=_BASE_MAT.get(tp,0); idx=i if t=='w' else 63-i
        pst=_AGGR_PST.get(tp,[0]*64)[idx]
        mat_pst+=sign*(mat+pst)
    w_prox=_attack_proximity_bonus(board,'w',b_king) if b_king!=-1 else 0
    b_prox=_attack_proximity_bonus(board,'b',w_king) if w_king!=-1 else 0
    prox_score=w_prox-b_prox
    w_storm=_pawn_storm_bonus(board,'w',b_king) if b_king!=-1 else 0
    b_storm=_pawn_storm_bonus(board,'b',w_king) if w_king!=-1 else 0
    storm_score=w_storm-b_storm
    mob=_mobility_score(state,'w')*8
    w_king_danger=0; b_king_danger=0
    if w_king!=-1: w_king_danger=sum(1 for i,p in enumerate(board) if p and p[0]=='b' and _chebyshev(i,w_king)<=2)*15
    if b_king!=-1: b_king_danger=sum(1 for i,p in enumerate(board) if p and p[0]=='w' and _chebyshev(i,b_king)<=2)*15
    safety_score=b_king_danger-w_king_danger
    total=int(W_MATERIAL*mat_pst+W_ATTACK_PROX*prox_score+W_PAWN_STORM*storm_score+W_MOBILITY*mob+W_OWN_KING_SAFETY*safety_score)
    return total

def explain(state): return f"Berserker eval: {evaluate(state):+d} cp"
