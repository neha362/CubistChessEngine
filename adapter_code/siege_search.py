import time
from typing import Optional, Tuple
import chess

MATE_SCORE=100_000; INFINITY=1_000_000

class _TT:
    EXACT,LOWER,UPPER=0,1,2
    def __init__(self): self.table={}
    def store(self,key,depth,score,flag,move): self.table[key]=(depth,score,flag,move)
    def probe(self,key): return self.table.get(key)

class _Timeout(Exception): pass

class Search:
    name="AlphaBetaSearch"
    def __init__(self,extend_checks_in_qsearch=True):
        self.tt=_TT(); self.extend_checks=extend_checks_in_qsearch
        self.nodes=0; self.start_time=0.0; self.time_limit=0.0

    def find_best_move(self,board,move_gen,evaluator,time_limit=3.0,max_depth=64):
        self.nodes=0; self.start_time=time.time(); self.time_limit=time_limit
        best_move=None; best_score=0
        for depth in range(1,max_depth+1):
            try:
                score=self._negamax(board,depth,-INFINITY,INFINITY,0,move_gen,evaluator)
                entry=self.tt.probe(board._transposition_key())
                if entry is not None and entry[3] is not None:
                    best_move=entry[3]; best_score=score
            except _Timeout: break
            if abs(best_score)>MATE_SCORE-1000: break
        return best_move,best_score

    def _time_up(self): return(self.nodes&2047)==0 and(time.time()-self.start_time)>=self.time_limit

    def _negamax(self,board,depth,alpha,beta,ply,move_gen,evaluator):
        self.nodes+=1
        if self._time_up(): raise _Timeout
        if board.is_checkmate(): return -MATE_SCORE+ply
        if board.is_stalemate() or board.is_insufficient_material() or board.can_claim_threefold_repetition(): return 0
        alpha_orig=alpha; key=board._transposition_key(); tt_move=None
        entry=self.tt.probe(key)
        if entry is not None:
            tt_depth,tt_score,tt_flag,tt_move=entry
            if ply>0 and tt_depth>=depth:
                if tt_flag==_TT.EXACT: return tt_score
                if tt_flag==_TT.LOWER and tt_score>alpha: alpha=tt_score
                elif tt_flag==_TT.UPPER and tt_score<beta: beta=tt_score
                if alpha>=beta: return tt_score
        if depth<=0: return self._quiesce(board,alpha,beta,move_gen,evaluator,qply=0)
        best_score=-INFINITY; best_move=None
        for move in move_gen.ordered_moves(board,hint_move=tt_move):
            board.push(move)
            try: score=-self._negamax(board,depth-1,-beta,-alpha,ply+1,move_gen,evaluator)
            finally: board.pop()
            if score>best_score: best_score=score; best_move=move
            if score>alpha: alpha=score
            if alpha>=beta: break
        if best_score<=alpha_orig: flag=_TT.UPPER
        elif best_score>=beta: flag=_TT.LOWER
        else: flag=_TT.EXACT
        self.tt.store(key,depth,best_score,flag,best_move)
        return best_score

    def _quiesce(self,board,alpha,beta,move_gen,evaluator,qply):
        self.nodes+=1
        if self._time_up(): raise _Timeout
        if qply>8:
            score=evaluator.evaluate(board)
            return score if board.turn==chess.WHITE else -score
        stand_pat=evaluator.evaluate(board)
        if board.turn==chess.BLACK: stand_pat=-stand_pat
        if stand_pat>=beta: return beta
        if stand_pat>alpha: alpha=stand_pat
        noisy=[]
        for m in board.legal_moves:
            if board.is_capture(m): noisy.append(m)
            elif self.extend_checks and qply<4 and board.gives_check(m): noisy.append(m)
        noisy.sort(key=lambda m:(board.is_capture(m)*1000+(board.gives_check(m)*500)),reverse=True)
        for move in noisy:
            board.push(move)
            try: score=-self._quiesce(board,-beta,-alpha,move_gen,evaluator,qply+1)
            finally: board.pop()
            if score>=beta: return beta
            if score>alpha: alpha=score
        return alpha

default=Search()
