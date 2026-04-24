import chess

from chess_engine.move_gen import MoveGenAgent, perft


def test_starting_position_perft_depth_1():
    board = chess.Board()
    assert perft(board, depth=1) == 20


def test_move_gen_wraps_legal_moves():
    board = chess.Board()
    mg = MoveGenAgent()
    assert mg.generate_moves(board) == list(board.legal_moves)
