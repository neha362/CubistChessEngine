from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import random, sys

FILES = 'abcdefgh'
def sq(r,c): return r*8+c
def row(i):  return i//8
def col(i):  return i%8
def sq_name(i): return FILES[col(i)]+str(8-row(i))
def parse_sq(s):
    if len(s)<2 or s[0] not in FILES or s[1] not in '12345678': return None
    return sq(8-int(s[1]), FILES.index(s[0]))
def color(p): return p[0] if p else None
def ptype(p): return p[1] if p else None

@dataclass
class GameState:
    board: list; turn: str; castling: dict; ep_square: Optional[int]
    halfmove: int=0; fullmove: int=1
    def copy(self): return GameState(self.board[:],self.turn,dict(self.castling),self.ep_square,self.halfmove,self.fullmove)
    def opponent(self): return 'b' if self.turn=='w' else 'w'

_FEN_MAP={'P':'wP','N':'wN','B':'wB','R':'wR','Q':'wQ','K':'wK','p':'bP','n':'bN','b':'bB','r':'bR','q':'bQ','k':'bK'}
STARTPOS='rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1'

def from_fen(fen):
    parts=fen.split(); board=[None]*64; r=0
    for rank_str in parts[0].split('/'):
        c=0
        for ch in rank_str:
            if ch.isdigit(): c+=int(ch)
            else: board[sq(r,c)]=_FEN_MAP[ch]; c+=1
        r+=1
    turn=parts[1] if len(parts)>1 else 'w'
    cas=parts[2] if len(parts)>2 else 'KQkq'
    castling={'wK':'K' in cas,'wQ':'Q' in cas,'bK':'k' in cas,'bQ':'q' in cas}
    ep_s=parts[3] if len(parts)>3 else '-'
    ep=parse_sq(ep_s) if ep_s!='-' else None
    hm=int(parts[4]) if len(parts)>4 else 0
    fm=int(parts[5]) if len(parts)>5 else 1
    return GameState(board,turn,castling,ep,hm,fm)

def _is_square_attacked(board,square,by):
    r,c_=row(square),col(square)
    for dr,dc in[(-2,-1),(-2,1),(-1,-2),(-1,2),(1,-2),(1,2),(2,-1),(2,1)]:
        nr,nc=r+dr,c_+dc
        if 0<=nr<8 and 0<=nc<8:
            p=board[sq(nr,nc)]
            if p and color(p)==by and ptype(p)=='N': return True
    for dirs,types in[([(-1,-1),(-1,1),(1,-1),(1,1)],{'B','Q'}),([(-1,0),(1,0),(0,-1),(0,1)],{'R','Q'})]:
        for dr,dc in dirs:
            nr,nc=r+dr,c_+dc
            while 0<=nr<8 and 0<=nc<8:
                p=board[sq(nr,nc)]
                if p:
                    if color(p)==by and ptype(p) in types: return True
                    break
                nr+=dr; nc+=dc
    for dr,dc in[(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]:
        nr,nc=r+dr,c_+dc
        if 0<=nr<8 and 0<=nc<8:
            p=board[sq(nr,nc)]
            if p and color(p)==by and ptype(p)=='K': return True
    pawn_dir=1 if by=='w' else -1
    for dc in(-1,1):
        nr,nc=r+pawn_dir,c_+dc
        if 0<=nr<8 and 0<=nc<8:
            p=board[sq(nr,nc)]
            if p and color(p)==by and ptype(p)=='P': return True
    return False

def is_in_check(state,col_=None):
    c=col_ or state.turn
    king=next((i for i in range(64) if state.board[i]==c+'K'),-1)
    return king!=-1 and _is_square_attacked(state.board,king,'b' if c=='w' else 'w')

def _pseudo_moves(state):
    moves=[]; board=state.board; turn=state.turn; opp=state.opponent()
    for frm in range(64):
        p=board[frm]
        if not p or color(p)!=turn: continue
        tp=ptype(p); r,c_=row(frm),col(frm)
        if tp=='P':
            d=-1 if turn=='w' else 1; start_row=6 if turn=='w' else 1; promo_row=0 if turn=='w' else 7
            fwd=sq(r+d,c_)
            if 0<=r+d<8 and not board[fwd]:
                if row(fwd)==promo_row:
                    for pr in 'qrbn': moves.append((frm,fwd,pr))
                else:
                    moves.append((frm,fwd,''))
                if r==start_row and not board[sq(r+2*d,c_)]: moves.append((frm,sq(r+2*d,c_),''))
            for dc in(-1,1):
                nc=c_+dc
                if 0<=nc<8:
                    to=sq(r+d,nc)
                    if (board[to] and color(board[to])==opp) or to==state.ep_square:
                        if row(to)==promo_row:
                            for pr in 'qrbn': moves.append((frm,to,pr))
                        else: moves.append((frm,to,''))
        elif tp=='N':
            for dr,dc in[(-2,-1),(-2,1),(-1,-2),(-1,2),(1,-2),(1,2),(2,-1),(2,1)]:
                nr,nc=r+dr,c_+dc
                if 0<=nr<8 and 0<=nc<8:
                    to=sq(nr,nc)
                    if not board[to] or color(board[to])==opp: moves.append((frm,to,''))
        elif tp in('B','R','Q'):
            dirs=[]
            if tp in('B','Q'): dirs+=[(-1,-1),(-1,1),(1,-1),(1,1)]
            if tp in('R','Q'): dirs+=[(-1,0),(1,0),(0,-1),(0,1)]
            for dr,dc in dirs:
                nr,nc=r+dr,c_+dc
                while 0<=nr<8 and 0<=nc<8:
                    to=sq(nr,nc)
                    if board[to]:
                        if color(board[to])==opp: moves.append((frm,to,''))
                        break
                    moves.append((frm,to,'')); nr+=dr; nc+=dc
        elif tp=='K':
            for dr,dc in[(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]:
                nr,nc=r+dr,c_+dc
                if 0<=nr<8 and 0<=nc<8:
                    to=sq(nr,nc)
                    if not board[to] or color(board[to])==opp: moves.append((frm,to,''))
            if turn=='w' and frm==60:
                if state.castling.get('wK') and not board[61] and not board[62]: moves.append((60,62,''))
                if state.castling.get('wQ') and not board[59] and not board[58] and not board[57]: moves.append((60,58,''))
            if turn=='b' and frm==4:
                if state.castling.get('bK') and not board[5] and not board[6]: moves.append((4,6,''))
                if state.castling.get('bQ') and not board[3] and not board[2] and not board[1]: moves.append((4,2,''))
    return moves

def make_move(state,move):
    frm,to,promo=move; ns=state.copy(); board=ns.board; piece=board[frm]; tp,t=ptype(piece),color(piece)
    board[to]=piece; board[frm]=None; ns.ep_square=None
    if tp=='P':
        if to==state.ep_square: board[sq(row(frm),col(to))]=None
        if abs(row(to)-row(frm))==2: ns.ep_square=sq((row(frm)+row(to))//2,col(frm))
        if promo: board[to]=t+promo.upper()
        ns.halfmove=0
    else: ns.halfmove=0 if board[to] else ns.halfmove+1
    if tp=='K':
        if t=='w': ns.castling['wK']=ns.castling['wQ']=False
        else: ns.castling['bK']=ns.castling['bQ']=False
        if col(to)-col(frm)==2: board[to-1]=board[to+1]; board[to+1]=None
        elif col(frm)-col(to)==2: board[to+1]=board[to-4]; board[to-4]=None
    for s in(frm,to):
        if s==56: ns.castling['wQ']=False
        if s==63: ns.castling['wK']=False
        if s==0:  ns.castling['bQ']=False
        if s==7:  ns.castling['bK']=False
    ns.turn=ns.opponent()
    if ns.turn=='w': ns.fullmove+=1
    return ns

def all_legal_moves(state):
    legal=[]; opp=state.opponent()
    for move in _pseudo_moves(state):
        frm,to,_=move; p=state.board[frm]
        if ptype(p)=='K' and abs(col(to)-col(frm))==2:
            step=1 if col(to)>col(frm) else -1; mid=sq(row(frm),col(frm)+step)
            if _is_square_attacked(state.board,frm,opp) or _is_square_attacked(state.board,mid,opp): continue
        ns=make_move(state,move)
        if not is_in_check(ns,state.turn): legal.append(move)
    return legal

def game_status(state):
    if state.halfmove>=100 or state.fullmove>200: return 'draw'
    moves=all_legal_moves(state)
    if not moves: return 'checkmate' if is_in_check(state) else 'stalemate'
    if set(p for p in state.board if p)<=set(['wK','bK']): return 'insufficient'
    return 'ongoing'

def is_terminal(state): return game_status(state)!='ongoing'
def game_result(state):
    status=game_status(state)
    if status=='checkmate': return 0.0 if state.turn=='w' else 1.0
    return 0.5

def fast_random_move(state):
    pseudo=_pseudo_moves(state); random.shuffle(pseudo); opp=state.opponent()
    for move in pseudo:
        frm,to,_=move; p=state.board[frm]
        if ptype(p)=='K' and abs(col(to)-col(frm))==2:
            step=1 if col(to)>col(frm) else -1; mid=sq(row(frm),col(frm)+step)
            if _is_square_attacked(state.board,frm,opp) or _is_square_attacked(state.board,mid,opp): continue
        ns=make_move(state,move)
        if not is_in_check(ns,state.turn): return move
    return None
