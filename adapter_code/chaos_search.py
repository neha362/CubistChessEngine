from __future__ import annotations
import math, time, random, sys
from dataclasses import dataclass, field
from typing import Optional

from chaos_move_gen import GameState, from_fen, make_move, all_legal_moves, is_in_check, game_status, sq_name, STARTPOS, sq, row, col, _is_square_attacked
from chaos_eval import evaluate as _berserker_eval, _king_square

AGENT_NAME="The Berserker"
_MVV={'P':1,'N':3,'B':3,'R':5,'Q':9,'K':0}

random.seed(0xBE2523)
_ZP={p:[random.getrandbits(64) for _ in range(64)] for p in ['wP','wN','wB','wR','wQ','wK','bP','bN','bB','bR','bQ','bK']}
_ZT=random.getrandbits(64)
_ZC={k:random.getrandbits(64) for k in['wK','wQ','bK','bQ']}
_ZEP=[random.getrandbits(64) for _ in range(8)]

def _zobrist(state):
    h=0
    for i,p in enumerate(state.board):
        if p: h^=_ZP[p][i]
    if state.turn=='b': h^=_ZT
    for k,v in state.castling.items():
        if v: h^=_ZC[k]
    if state.ep_square is not None: h^=_ZEP[state.ep_square%8]
    return h

TT_EXACT,TT_LOWER,TT_UPPER=0,1,2
_TT_SIZE=1<<19

@dataclass
class _TTEntry:
    key:int; depth:int; flag:int; score:int; move:Optional[tuple]

_tt={}

def _tt_probe(key):
    e=_tt.get(key&(_TT_SIZE-1))
    return e if(e and e.key==key) else None

def _tt_store(key,depth,flag,score,move):
    idx=key&(_TT_SIZE-1); e=_tt.get(idx)
    if not e or e.key==key or depth>=e.depth: _tt[idx]=_TTEntry(key,depth,flag,score,move)

def _is_check_move(state,move):
    try:
        ns=make_move(state,move); opp='b' if state.turn=='w' else 'w'
        return is_in_check(ns,opp)
    except: return False

def _proximity_to_enemy_king(state,move):
    board=state.board; opp='b' if state.turn=='w' else 'w'
    king_sq=_king_square(board,opp)
    if king_sq==-1: return 99
    return max(abs(row(move[1])-row(king_sq)),abs(col(move[1])-col(king_sq)))

_killers=[[] for _ in range(128)]

def _score_move(move,state,tt_move,ply):
    frm,to,promo=move; victim=state.board[to]; attacker=state.board[frm]
    if move==tt_move: return 10_000_000
    if _is_check_move(state,move): return 9_000_000
    if victim: return 5_000_000+_MVV.get(victim[1],0)*10-_MVV.get(attacker[1] if attacker else 'P',0)
    if promo=='q': return 4_500_000
    dist=_proximity_to_enemy_king(state,move)
    prox_score=(7-dist)*50_000
    kl_bonus=100_000 if move in _killers[ply] else 0
    return prox_score+kl_bonus

def _order_moves(moves,state,tt_move,ply):
    return sorted(moves,key=lambda m:_score_move(m,state,tt_move,ply),reverse=True)

_nodes=[0]; _stop=[False]

def _negamax(state,depth,ply,alpha,beta):
    if _stop[0]: return 0
    _nodes[0]+=1
    key=_zobrist(state); tt_ent=_tt_probe(key); tt_mv=None
    if tt_ent:
        tt_mv=tt_ent.move
        if tt_ent.depth>=depth:
            s=tt_ent.score
            if tt_ent.flag==TT_EXACT: return s
            elif tt_ent.flag==TT_LOWER: alpha=max(alpha,s)
            elif tt_ent.flag==TT_UPPER: beta=min(beta,s)
            if alpha>=beta: return s
    status=game_status(state)
    if status=='checkmate': return -(99_000-ply)
    if status in('stalemate','draw','insufficient'): return 0
    if depth==0:
        score=_berserker_eval(state)
        return score if state.turn=='w' else -score
    moves=all_legal_moves(state)
    if not moves: return 0
    moves=_order_moves(moves,state,tt_mv,ply)
    best_move=None; orig_alpha=alpha
    for move in moves:
        if _stop[0]: break
        ns=make_move(state,move); score=-_negamax(ns,depth-1,ply+1,-beta,-alpha)
        if score>alpha:
            alpha=score; best_move=move
            if alpha>=beta:
                if not state.board[move[1]] and not move[2]:
                    kl=_killers[ply]
                    if move not in kl: kl.insert(0,move)
                    if len(kl)>2: kl.pop()
                break
    if not _stop[0]:
        flag=(TT_EXACT if orig_alpha<alpha<beta else TT_LOWER if alpha>=beta else TT_UPPER)
        _tt_store(key,depth,flag,alpha,best_move)
    return alpha

@dataclass
class BerserkerResult:
    move: Optional[tuple]; score: int; depth: int; nodes: int; time_ms: int
    pv: list=field(default_factory=list)
    @property
    def uci(self):
        if not self.move: return '0000'
        f,t,pr=self.move; return sq_name(f)+sq_name(t)+pr

def search(state,max_depth=5,movetime_ms=None,verbose=False):
    global _killers
    _killers=[[] for _ in range(128)]; _tt.clear(); _nodes[0]=0; _stop[0]=False
    start=time.time(); result=BerserkerResult(None,0,0,0,0)
    moves=all_legal_moves(state)
    if not moves: return result
    for d in range(1,max_depth+1):
        elapsed=(time.time()-start)*1000
        if movetime_ms and elapsed>movetime_ms*0.85: break
        best_move=None; best_score=-math.inf; alpha=-math.inf; beta=math.inf
        tt_ent=_tt_probe(_zobrist(state)); tt_mv=tt_ent.move if tt_ent else None
        ordered=_order_moves(moves,state,tt_mv,0)
        for move in ordered:
            if _stop[0]: break
            if movetime_ms and (time.time()-start)*1000>movetime_ms*0.92: _stop[0]=True; break
            ns=make_move(state,move); score=-_negamax(ns,d-1,1,-beta,-alpha)
            if score>best_score: best_score=score; best_move=move; alpha=max(alpha,score)
        if not _stop[0] and best_move:
            elapsed_ms=int((time.time()-start)*1000)
            result=BerserkerResult(best_move,best_score,d,_nodes[0],elapsed_ms)
    return result
