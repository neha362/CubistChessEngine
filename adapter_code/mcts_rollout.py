import random, time
from typing import Optional, Callable
from mcts_move_gen import GameState, from_fen, make_move, is_terminal, game_result, fast_random_move, all_legal_moves, STARTPOS

_MV={'P':1,'N':3,'B':3,'R':5,'Q':9,'K':0}

def _policy_random(state):    return fast_random_move(state)
def _policy_capture(state):
    moves=all_legal_moves(state)
    if not moves: return None
    captures=[m for m in moves if state.board[m[1]]]
    pool=captures if captures else moves
    return random.choice(pool)
def _policy_material(state):
    moves=all_legal_moves(state)
    if not moves: return None
    best_val=-1; best=[]
    for m in moves:
        captured=state.board[m[1]]; val=_MV.get(captured[1],0) if captured else 0
        if val>best_val: best_val=val; best=[m]
        elif val==best_val: best.append(m)
    return random.choice(best)

POLICY_RANDOM=_policy_random; POLICY_CAPTURE=_policy_capture; POLICY_MATERIAL=_policy_material
ACTIVE_POLICY=POLICY_RANDOM

def rollout(state,max_plies=200):
    current=state
    for _ in range(max_plies):
        if is_terminal(current): return game_result(current)
        move=ACTIVE_POLICY(current)
        if move is None: return game_result(current)
        current=make_move(current,move)
    return 0.5
