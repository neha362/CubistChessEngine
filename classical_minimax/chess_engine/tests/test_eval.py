import chess

from chess_engine.eval import EvalAgent


def test_starting_position_balanced():
    board = chess.Board()
    assert EvalAgent().evaluate(board) == 0


def test_white_queen_ahead():
    board = chess.Board()
    board.set_piece_at(chess.E4, chess.Piece(chess.QUEEN, chess.WHITE))
    assert EvalAgent().evaluate(board) > 800
