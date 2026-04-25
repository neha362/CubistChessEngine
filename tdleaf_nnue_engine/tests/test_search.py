import chess

from tdleaf_nnue_engine.eval import Evaluator
from tdleaf_nnue_engine.search import Searcher


def test_search_returns_legal_move():
    board = chess.Board()
    searcher = Searcher(evaluator=Evaluator())
    move = searcher.best_move(board, depth=2)
    assert move in board.legal_moves
