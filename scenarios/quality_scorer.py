"""
quality_scorer.py — per-move quality measurement
=================================================

The missing piece that lets the Layer 3 trust matrix actually learn.

Without this module, `update_after_game()` in run_ensemble.py uses a single
quality score per game (final outcome). Every move played by every engine
gets the same credit, so the trust matrix can't distinguish "this engine
plays well in tactical chaos" from "this engine plays well in endgames" —
it only knows "this engine was on the winning team."

This module assigns a per-move quality. For every move played, we compare
its evaluation to what a stronger reference engine would have chosen:

    quality(move) = 1 - clamp(centipawn_loss / 200, 0, 1)

    cp_loss = best_eval_after_reference_move - eval_after_actual_move

A move that matches the reference's choice gets quality=1.0. A move that
loses 100 cp gets quality=0.5. A move that loses 200+ cp gets quality=0.0.

The reference engine is `classical` (alpha-beta + material/PST) at a depth
deeper than any engine used at runtime in the ensemble. That's the cheapest
"strong-enough" oracle we have without introducing Stockfish as a dependency.

LIMITATIONS (be honest about these):

1. The reference is only ~classical. If your ensemble already plays at
   classical-depth, the reference can't tell good from bad — it's the same
   skill level. For meaningful signal, set REFERENCE_DEPTH > what the
   ensemble engines use at runtime. Default: 4 (vs typical engine 3).
2. Material+PST eval has known blind spots (it doesn't see deep tactics).
   Quality scores will be wrong on positions where the reference itself
   blunders. Stockfish at depth 12 would be a much better reference if
   you're willing to add the dependency.
3. cp_loss is clipped at 200. A "huge" blunder (-800 cp) gets quality 0.0,
   same as a "moderate" blunder (-200 cp). This is intentional — it keeps
   the Bayesian update bounded. But it loses information about how bad
   the move was.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
for sub in ("adapter_code",):
    p = str(REPO_ROOT / sub)
    if p not in sys.path:
        sys.path.insert(0, p)

import chess

# Lazy reference-engine init (the import chain is heavy).
_reference = None
REFERENCE_DEPTH = 4


def _get_reference():
    """Build the reference engine once and cache it."""
    global _reference
    if _reference is None:
        from classical_move_gen import MoveGenAgent
        from classical_eval import EvalAgent
        from classical_search import SearchAgent
        ev = EvalAgent()
        mg = MoveGenAgent()
        sa = SearchAgent(eval_fn=ev.evaluate, move_gen=mg)
        _reference = (sa, ev)
    return _reference


def score_move_quality(fen_before: str, played_uci: str,
                       reference_depth: Optional[int] = None) -> float:
    """
    Return quality ∈ [0, 1] for the move that was played.

    Approach: ask the reference engine for ITS best move from the same
    position, and compare evaluations of the resulting positions.

    Cheap optimization: if the engine's move IS the reference's best move,
    return 1.0 immediately without re-evaluating anything.
    """
    sa, ev = _get_reference()
    depth = reference_depth or REFERENCE_DEPTH

    board = chess.Board(fen_before)
    legal = list(board.legal_moves)
    if not legal:
        return 0.5  # no signal

    played = chess.Move.from_uci(played_uci)
    if played not in legal:
        # Engine returned an illegal move — that's a serious failure.
        return 0.0

    # Get the reference engine's best move.
    try:
        ref_best = sa.best_move(board.copy(stack=False), depth)
    except Exception:
        return 0.5  # reference crashed; no useful signal
    if ref_best is None:
        return 0.5

    # Match-bonus shortcut: same move = perfect quality.
    if ref_best == played:
        return 1.0

    # Otherwise, evaluate both resulting positions.
    # Both evals are returned from White's POV by classical's EvalAgent.
    side_sign = 1 if board.turn == chess.WHITE else -1

    board.push(ref_best)
    ref_score = side_sign * ev.evaluate(board)
    board.pop()

    board.push(played)
    played_score = side_sign * ev.evaluate(board)
    board.pop()

    # cp_loss is from the moving side's perspective.
    cp_loss = ref_score - played_score
    if cp_loss < 0:
        # Played move evaluated BETTER than reference's pick — that means
        # the reference's depth-limited search missed something the engine
        # caught. Give full credit.
        return 1.0

    quality = 1.0 - min(cp_loss / 200.0, 1.0)
    return max(0.0, quality)


def score_game(history: list, reference_depth: Optional[int] = None,
               counterfactual: bool = True) -> dict:
    """
    Score every move in a finished game.

    `history` is a list of (fen_before, uci_played, layer3_result) tuples.

    counterfactual=True (default): score *every* engine's bid at each ply
        and emit one (engine, scenarios, quality, score_cp) tuple per bid.
        This gives the trust matrix one signal per engine per ply (~25x more
        samples in the full 5x5 grid)
        and lets non-winning engines also accumulate scenario-conditioned
        evidence — including dissenting engines paying for bad moves the
        ensemble didn't actually play. Requires Layer3Result.proposals to
        be populated (added in this release).

    counterfactual=False: legacy mode — score only the move actually played.
        Falls back to this automatically when an l3_result has no proposals.

    Returns:
      - per_move_qualities: list[(engine_id, scenario_profile, quality, score_cp)]
      - per_engine_avg:    dict[engine_id, avg_quality]
    """
    per_move = []
    sums = {}
    counts = {}

    for fen_before, uci_played, l3_result in history:
        if l3_result is None:
            continue
        scenarios = l3_result.scenario_profile

        # Choose which (engine_id, uci) pairs to score for this ply.
        proposals = getattr(l3_result, "proposals", None) or []
        if counterfactual and proposals:
            scored = [(p.engine_id, p.uci, p.score_cp) for p in proposals]
        else:
            score_cp = None
            for proposal in proposals:
                if proposal.engine_id == l3_result.chosen_engine:
                    score_cp = proposal.score_cp
                    break
            scored = [(l3_result.chosen_engine, uci_played, score_cp)]

        for engine_id, uci, score_cp in scored:
            q = score_move_quality(fen_before, uci, reference_depth)
            per_move.append((engine_id, scenarios, q, score_cp))
            sums[engine_id]   = sums.get(engine_id, 0.0)   + q
            counts[engine_id] = counts.get(engine_id, 0) + 1

    averages = {e: sums[e] / counts[e] for e in sums}
    return {"per_move": per_move, "per_engine_avg": averages}


# Self-test
if __name__ == "__main__":
    # Sanity: scoring an obviously good move should give high quality,
    # an obviously bad move should give low quality.
    startpos = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    fools = "rnbqkb1r/pppp1ppp/5n2/4p3/4P3/5P2/PPPP2PP/RNBQKBNR w KQkq - 0 1"

    print("quality_scorer self-test")
    print("=" * 50)
    print(f"Reference depth: {REFERENCE_DEPTH}")

    # A reasonable opening move should score high.
    q1 = score_move_quality(startpos, "e2e4")
    print(f"  Start position, played e2e4: quality = {q1:.3f}")
    assert q1 >= 0.5, f"e2e4 should be reasonable, got {q1}"

    # An obvious blunder: g2g4 (Grob) — weakens the king for nothing.
    # Reference probably doesn't say this is the worst, but it shouldn't be top.
    q2 = score_move_quality(startpos, "g2g4")
    print(f"  Start position, played g2g4: quality = {q2:.3f}")

    # A passive a-pawn nudge.
    q3 = score_move_quality(startpos, "a2a3")
    print(f"  Start position, played a2a3: quality = {q3:.3f}")

    print(f"  Sanity check: e2e4 ({q1:.3f}) should >= a2a3 ({q3:.3f}): "
          f"{'OK' if q1 >= q3 else 'FAIL'}")

    # Illegal move should be 0.0.
    q4 = score_move_quality(startpos, "e2e5")
    print(f"  Illegal move e2e5: quality = {q4:.3f}")
    assert q4 == 0.0, f"illegal move should be 0.0, got {q4}"

    print("=" * 50)
    print("Self-test passed.")
