"""
Run: ``python -m evolutionary_tournament`` from the ``cubist`` repo root.

``--demo`` runs a few plies of analysis for both engines on the startpos and
prints expected centipawn loss per move. ``--evolve`` runs 3–4 evolution rounds
then a short head-to-head game.
"""

from __future__ import annotations

import argparse
import textwrap

import chess

from . import ground_truth
from .arena import head_to_head
from .engines import Berserker2Engine, ClassicalEngine, rank_correlation_move_lists
from .evolution import run_evolution

_DEFAULT_FENS = [
    chess.Board().fen(),
    "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",
    "r3k2r/pp1n1ppp/2p1pn2/q7/2B5/2N2N2/PP2QPPP/2R1K2R w Kkq - 0 12",
    "2rq1rk1/1b2bppp/p2p1n2/1p1P4/3N4/1P2B3/P2Q1PPP/R3R1K1 w - - 0 18",
    "4k3/8/8/8/8/8/4P3/4K3 w - - 0 1",
    "8/5k2/8/8/8/8/5K2/8 w - - 0 1",
]


def _demo() -> None:
    b = chess.Board()
    c = ClassicalEngine(depth=3)
    b2 = Berserker2Engine(depth=2)
    with ground_truth.reference_engine() as eng:
        ref = ground_truth.root_move_scores_sm(b, 5, eng)
    print("=== startpos: classical_minimax (optimal + expected loss) ===")
    r1 = c.analyze(b)
    for L in r1.lines[:12]:
        print(f"  {L.uci:6s}  score={L.score_cp:+5d}  loss={L.centipawn_loss:3d} cp")
    print(f"best: {r1.best_uci}  (engine id: {r1.engine_id})")
    if ref:
        rho = rank_correlation_move_lists(ref, {x.uci: x.score_cp for x in r1.lines})
        print(f"Spearman rho vs reference: {rho:.3f}")
    print()
    print("=== berserker_2 ===")
    r2 = b2.analyze(b)
    for L in r2.lines[:12]:
        print(f"  {L.uci:6s}  score={L.score_cp:+5d}  loss={L.centipawn_loss:3d} cp")
    print(f"best: {r2.best_uci}")


def _evolve() -> None:
    stats = run_evolution(_DEFAULT_FENS, population=8, rounds=4)
    for s in stats:
        for L in s.log_lines:
            print(L)
    best = stats[-1].best_weights
    print()
    print("head-to-head (evolved classical as White vs berserker_2) ...")
    r = head_to_head(classical_weights=best, depth_c=3, depth_b2=2, max_plies=32)
    print("result:", r)


def main() -> None:
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent(__doc__ or ""),
    )
    p.add_argument("--demo", action="store_true", help="per-move loss on startpos only (default if no flag)")
    p.add_argument("--evolve", action="store_true", help="run evolution + short game")
    a = p.parse_args()
    if a.evolve:
        _evolve()
    else:
        _demo()


if __name__ == "__main__":
    main()
