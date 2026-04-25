import chess

from chess_engine.game_state import is_terminal, terminal_score


def test_starting_position_not_terminal():
    board = chess.Board()
    assert is_terminal(board) is False


def test_scholars_mate_position_is_terminal():
    board = chess.Board()
    for uci in ("e2e4", "e7e5", "d1h5", "b8c6", "f1c4", "g8f6", "h5f7"):
        board.push(chess.Move.from_uci(uci))
    assert board.is_checkmate()
    assert is_terminal(board) is True


def test_terminal_score_checkmate_side_to_move_loses():
    board = chess.Board(fen="rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3")
    assert board.is_checkmate()
    assert terminal_score(board) == -100000


def test_terminal_score_stalemate_is_zero():
    board = chess.Board(fen="k7/8/1Q6/8/2K5/8/8/8 b - - 0 1")
    assert board.is_stalemate()
    assert terminal_score(board) == 0
