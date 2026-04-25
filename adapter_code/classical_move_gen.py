import chess
from typing import List

class MoveGenAgent:
    def generate_moves(self,board): return list(board.legal_moves)

def perft(board,depth,move_gen=None):
    mg=move_gen or MoveGenAgent()
    if depth<=0: return 1
    total=0
    for move in mg.generate_moves(board):
        board.push(move); total+=perft(board,depth-1,mg); board.pop()
    return total
