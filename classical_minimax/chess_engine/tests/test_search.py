import chess

from chess_engine.move_gen import MoveGenAgent
from chess_engine.search import SearchAgent


def test_search_runs_with_zero_eval_returns_legal_move():
    board = chess.Board()
    mg = MoveGenAgent()
    search = SearchAgent(lambda b: 0, mg)
    move = search.best_move(board, max_depth=2)
    assert move in board.legal_moves
