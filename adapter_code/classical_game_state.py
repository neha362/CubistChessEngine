import chess
def is_terminal(board):
    return(board.is_checkmate() or board.is_stalemate() or board.is_insufficient_material()
           or board.can_claim_fifty_moves() or board.can_claim_threefold_repetition())
def terminal_score(board):
    if board.is_checkmate(): return -100000
    return 0
