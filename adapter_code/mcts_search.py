from __future__ import annotations
import math, time, random, sys
from dataclasses import dataclass, field
from typing import Optional

from mcts_move_gen import GameState,from_fen,make_move,all_legal_moves,is_terminal,game_result,sq_name,STARTPOS
from mcts_rollout import rollout

UCB1_C=math.sqrt(2)

class MCTSNode:
    __slots__=('state','move','parent','children','untried','visits','wins')
    def __init__(self,state,move=None,parent=None):
        self.state=state; self.move=move; self.parent=parent; self.children=[]
        self.untried=all_legal_moves(state); random.shuffle(self.untried)
        self.visits=0; self.wins=0.0
    def ucb1(self,parent_visits,c=UCB1_C):
        if self.visits==0: return float('inf')
        if self.parent and self.parent.state.turn=='b': exploitation=1.0-(self.wins/self.visits)
        else: exploitation=self.wins/self.visits
        return exploitation+c*math.sqrt(math.log(parent_visits)/self.visits)
    def is_fully_expanded(self): return len(self.untried)==0
    def is_terminal(self): return is_terminal(self.state)
    def best_child(self,c=UCB1_C): return max(self.children,key=lambda ch:ch.ucb1(self.visits,c))
    def most_visited_child(self): return max(self.children,key=lambda ch:ch.visits)
    def win_rate(self): return self.wins/self.visits if self.visits else 0.0

def _select(node):
    while not node.is_terminal() and node.is_fully_expanded(): node=node.best_child()
    return node

def _expand(node):
    if not node.untried: return node
    move=node.untried.pop(); ns=make_move(node.state,move); child=MCTSNode(state=ns,move=move,parent=node)
    node.children.append(child); return child

def _simulate(node):
    if node.is_terminal(): return game_result(node.state)
    return rollout(node.state)

def _backpropagate(node,result):
    current=node
    while current is not None:
        current.visits+=1; current.wins+=result; current=current.parent

@dataclass
class MCTSResult:
    best_move: Optional[tuple]; iterations: int; time_ms: int; root_visits: int
    win_rate: float; top_moves: list=field(default_factory=list)
    @property
    def best_uci(self):
        if not self.best_move: return '0000'
        f,t,pr=self.best_move; return sq_name(f)+sq_name(t)+pr

def mcts_search(state,max_iter=1000,movetime_ms=None,c=UCB1_C,verbose=False):
    if is_terminal(state): return MCTSResult(None,0,0,0,0.5)
    root=MCTSNode(state=state); start=time.time(); iters=0
    def _over_budget():
        if movetime_ms: return(time.time()-start)*1000>=movetime_ms
        return iters>=max_iter
    while not _over_budget():
        node=_select(root)
        if not node.is_terminal(): node=_expand(node)
        result=_simulate(node)
        _backpropagate(node,result)
        iters+=1
    if not root.children:
        moves=all_legal_moves(state); best_move=random.choice(moves) if moves else None
        return MCTSResult(best_move,iters,int((time.time()-start)*1000),root.visits,0.5)
    best_child=root.most_visited_child(); best_move=best_child.move
    wr=(best_child.wins/best_child.visits) if state.turn=='w' else 1.0-(best_child.wins/best_child.visits)
    ranked=sorted(root.children,key=lambda n:n.visits,reverse=True)
    top_moves=[]
    for ch in ranked[:5]:
        ch_wr=(ch.wins/ch.visits) if state.turn=='w' else 1.0-(ch.wins/ch.visits)
        top_moves.append((sq_name(ch.move[0])+sq_name(ch.move[1])+ch.move[2],ch.visits,ch_wr))
    return MCTSResult(best_move,iters,int((time.time()-start)*1000),root.visits,wr,top_moves)
