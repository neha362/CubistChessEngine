import chess

from tdleaf_nnue_engine.nnue_features import FEATURE_SIZE, extract_features


def test_feature_shape_and_stability():
    board = chess.Board()
    f1 = extract_features(board)
    f2 = extract_features(board)
    assert f1.shape == (FEATURE_SIZE,)
    assert f1.dtype.name == "float32"
    assert (f1 == f2).all()
