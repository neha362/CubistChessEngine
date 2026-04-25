"""Short games between two engines (optional cap on plies for hackathon speed)."""

from __future__ import annotations

import chess

from .engines import Berserker2Engine, ClassicalEngine
from .tunable_classical import TunableWeights


def play(
    white: object,
    black: object,
    max_plies: int = 60,
) -> tuple[chess.Board, str | None]:
    b = chess.Board()
    for pl in range(max_plies):
        if b.is_game_over():
            return b, b.outcome().result() if b.outcome() else None
        to_move = white if b.turn == chess.WHITE else black
        mv = to_move.pick_move(b)  # type: ignore[attr-defined]
        b.push(mv)
    return b, "1/2-1/2*"


def head_to_head(
    classical_weights: TunableWeights | None = None,
    depth_c: int = 3,
    depth_b2: int = 3,
    max_plies: int = 40,
) -> str:
    w = ClassicalEngine(depth=depth_c, weights=classical_weights)
    x = Berserker2Engine(depth=depth_b2)
    b, res = play(w, x, max_plies=max_plies)
    if res and res.endswith("*"):
        return "draw/timeout"
    r = b.result(claim_draw=True) if b.is_game_over() else "incomplete"
    return r
