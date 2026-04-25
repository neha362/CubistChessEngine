"""
run_ensemble.py — the Cubist Chess Engine
==========================================

This is the file that turns a 25-node Layer 1 proposal grid
(5 search approaches x 5 eval approaches) plus Layer 3 trust into one
chess engine.

Pipeline (matches the architecture diagram):

  INPUT (board state, as FEN)
    │
    ▼
  Each search/eval pair produces (move, score, confidence)
    │
    ▼
  LAYER 1: detect_scenarios(state) → 6 scenario activations
  LAYER 2: agreement_profile(proposals) → 3 consensus signals
  LAYER 3: Layer3Ensemble.evaluate() → softmax over engines weighted by
           Bayesian trust matrix, returns chosen move
    │
    ▼
  OUTPUT: best_move + diagnostics

Usage
─────
  # Single move from the start position:
  python scenarios/run_ensemble.py

  # Single move from a custom FEN, with full diagnostics:
  python scenarios/run_ensemble.py --fen "<FEN>" --explain

  # Self-play game between two ensembles (or ensemble vs single engine):
  python scenarios/run_ensemble.py --selfplay --moves 30

  # Run a position through the ensemble and print one explained move:
  python scenarios/run_ensemble.py --fen "..." --explain

Persistence
───────────
The trust matrix is loaded from / saved to scenarios/cubist_trust.json
between runs. Delete that file to reset learning.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
root = str(REPO_ROOT)
if root not in sys.path:
    sys.path.insert(0, root)
for sub in ("adapter_code", "scenarios"):
    p = str(REPO_ROOT / sub)
    if p not in sys.path:
        sys.path.insert(0, p)

from layer3_ensemble import (  # noqa: E402
    Layer3Ensemble, Layer3Result, EngineProposal, SCENARIO_NAMES,
)
from chaos_move_gen import (  # noqa: E402
    GameState, from_fen, STARTPOS, all_legal_moves, make_move, is_terminal,
    game_result, sq_name,
)
from ensemble_adapters.engine_wrappers import gather_proposals, ENGINE_REGISTRY, tuple_to_uci  # noqa: E402


TRUST_PATH = str(REPO_ROOT / "scenarios" / "cubist_trust.json")


def _configure_stdout() -> None:
    """
    Prefer UTF-8 console output when the current stream supports reconfigure().

    The ensemble CLI uses box-drawing characters in its diagnostics, which can
    fail on the default Windows code page.
    """
    stream = getattr(sys, "stdout", None)
    if stream is None or not hasattr(stream, "reconfigure"):
        return
    try:
        stream.reconfigure(encoding="utf-8")
    except Exception:
        pass


# ─────────────────────────────────────────────────────────────────────────────
# CubistEngine — the public interface for the assembled engine
# ─────────────────────────────────────────────────────────────────────────────

class CubistEngine:
    """
    The full Cubist engine: gather proposals, route through Layer 3, return
    a move. Maintains a persistent Bayesian trust matrix across calls.
    """

    def __init__(self, engines: list = None, persistence_path: str = TRUST_PATH,
                 consensus_threshold: Optional[int] = None):
        self.engine_names = engines or list(ENGINE_REGISTRY.keys())
        self.ensemble = Layer3Ensemble(
            engine_ids=self.engine_names,
            persistence_path=persistence_path,
            consensus_threshold=consensus_threshold,
        )

    def best_move(self, fen: str) -> tuple:
        """
        Returns (uci_string, Layer3Result). The Layer3Result carries all
        diagnostics — feed it to print_explanation() for human-readable output.
        """
        state = from_fen(fen)
        proposals = gather_proposals(fen, engines=self.engine_names)

        if not proposals:
            # No engine produced a proposal. Fall back to first legal move.
            legal = all_legal_moves(state)
            if not legal:
                return None, None
            return tuple_to_uci(legal[0]), None

        result = self.ensemble.evaluate(state, proposals)
        return result.best_move, result

    def update_after_game(self, scored_history: list, autosave: bool = True) -> dict:
        """
        Update the trust matrix using per-move quality scores.

        `scored_history` is a list of
        (engine_id, scenario_profile, quality, score_cp) tuples — one per
        scored proposal. Produced by
        quality_scorer.score_game(history).

        For each ply, the engine's trust cells get a Bayesian update on the
        active scenarios proportional to that ply's quality score. Engines
        that consistently score high on tactical positions earn high trust
        in tactical_chaos; engines that score high on quiet positions earn
        trust in endgame_structure; etc. This is what enables the ensemble
        to learn position-conditioned trust over time.

        Returns a per-engine summary {engine_id: average_quality}.
        """
        sums, counts = {}, {}
        for entry in scored_history:
            if len(entry) == 4:
                engine_id, scenario_profile, quality, score_cp = entry
            else:
                engine_id, scenario_profile, quality = entry
                score_cp = None
            self.ensemble.update(
                engine_id,
                scenario_profile,
                quality,
                score_cp=score_cp,
                autosave=False,
            )
            sums[engine_id] = sums.get(engine_id, 0.0) + quality
            counts[engine_id] = counts.get(engine_id, 0) + 1

        if autosave:
            self.ensemble._save()
        return {e: sums[e] / counts[e] for e in sums}


# ─────────────────────────────────────────────────────────────────────────────
# Pretty-printing
# ─────────────────────────────────────────────────────────────────────────────

def print_proposals(proposals: list) -> None:
    print(f"  Layer 1 proposals ({len(proposals)} active nodes):")
    print("  ┌───────────────────────────┬──────┬───────────┬────────────┬───────────┐")
    print("  │ engine_pair               │ move │ score_cp  │ confidence │ prior_wt  │")
    print("  ├───────────────────────────┼──────┼───────────┼────────────┼───────────┤")
    for p in proposals:
        print(f"  │ {p.engine_id:<25} │ {p.uci:<4} │ {p.score_cp:>+9} │ "
              f"{p.confidence:>10.3f} │ {p.prior_weight:>9.2f} │")
    print("  └───────────────────────────┴──────┴───────────┴────────────┴───────────┘")


def print_explanation(result: Layer3Result) -> None:
    """Show why Layer 3 picked what it picked."""
    print()
    print(f"  → CHOSEN MOVE: {result.best_move}  (via {result.chosen_engine})")
    print()

    # Scenarios
    print("  Layer 1 — scenario activations (board features):")
    for name in SCENARIO_NAMES:
        a = result.scenario_profile.activations[name]
        bar = "█" * int(a * 30)
        print(f"    {name:<22} {a:.3f}  {bar}")

    # Agreement
    ap = result.agreement_profile
    print()
    print(f"  Layer 2 — agreement: largest_group={ap.largest_group}  "
          f"ratio={ap.consensus_ratio:.2f}  threshold={ap.majority_threshold}  "
          f"all_agree={int(ap.all_agree)}  majority={int(ap.majority)}  "
          f"split={int(ap.split)}")

    # Trust + final probabilities
    print()
    print("  Layer 3 — engine trust × softmax voting:")
    print("    ┌───────────────────────────┬─────────┬────────────┬─────────────┐")
    print("    │ engine_pair               │  trust  │ confidence │ vote_weight │")
    print("    ├───────────────────────────┼─────────┼────────────┼─────────────┤")
    for engine_id in result.engine_trusts:
        t = result.engine_trusts[engine_id]
        c = result.engine_confidences.get(engine_id, 0.0)
        p = result.engine_probs.get(engine_id, 0.0)
        print(f"    │ {engine_id:<25} │ {t:>7.3f} │ {c:>10.3f} │ {p:>11.3f} │")
    print("    └───────────────────────────┴─────────┴────────────┴─────────────┘")

    if result.short_circuit:
        print("  (Consensus short-circuit: ≥4 engines agreed, no softmax needed)")

    print()
    print("  Move-weight distribution after voting:")
    for uci, w in sorted(result.move_weights.items(), key=lambda kv: -kv[1]):
        print(f"    {uci}  {w:.3f}")


# ─────────────────────────────────────────────────────────────────────────────
# Single-position demo
# ─────────────────────────────────────────────────────────────────────────────

def demo_single(fen: str, explain: bool, engines: list) -> int:
    print()
    print("=" * 64)
    print("  CUBIST ENSEMBLE — single position")
    print("=" * 64)
    print(f"  FEN: {fen}")
    print(f"  Active engines: {', '.join(engines)}")
    print()

    state = from_fen(fen)
    proposals = gather_proposals(fen, engines=engines)
    if not proposals:
        print("  No engine produced a proposal. (Position has no legal moves?)")
        return 1
    print_proposals(proposals)

    ensemble = Layer3Ensemble(engine_ids=engines, persistence_path=TRUST_PATH)
    result = ensemble.evaluate(state, proposals)

    if explain:
        print_explanation(result)
    else:
        print()
        print(f"  → {result.best_move}  (via {result.chosen_engine})")
    return 0


# ─────────────────────────────────────────────────────────────────────────────
# Self-play game (ensemble plays both sides)
# ─────────────────────────────────────────────────────────────────────────────

def selfplay(max_moves: int, engines: list, explain: bool) -> int:
    """Play a full game with the ensemble making moves for both sides."""
    print()
    print("=" * 64)
    print("  CUBIST ENSEMBLE — self-play")
    print("=" * 64)
    print(f"  Active engines: {', '.join(engines)}")
    print(f"  Max moves: {max_moves}")
    print()

    cubist = CubistEngine(engines=engines)
    state = from_fen(STARTPOS)
    # Each entry: (fen_before_move, uci_played, layer3_result)
    history = []
    move_log = []

    for ply in range(max_moves * 2):
        if is_terminal(state):
            break
        fen = _state_to_fen(state)
        uci, result = cubist.best_move(fen)
        if uci is None or result is None:
            break

        history.append((fen, uci, result))
        move_log.append(uci)

        if explain:
            print(f"\n── ply {ply+1} ({'White' if state.turn == 'w' else 'Black'} to move) ──")
            print(f"  FEN: {fen}")
            print_explanation(result)
        else:
            short = "CONS" if result.short_circuit else "soft"
            print(f"  {ply+1:>3}. {state.turn} {uci}   "
                  f"via {result.chosen_engine:<10} [{short}]")

        # Apply the chosen move.
        legal = all_legal_moves(state)
        chosen_tuple = next(
            (m for m in legal if _move_uci(m) == uci),
            None,
        )
        if chosen_tuple is None:
            print(f"  ! ensemble produced illegal/unknown move {uci}, stopping")
            break
        state = make_move(state, chosen_tuple)

    # Game over — score every move and apply trust updates.
    print()
    print("─" * 64)
    if is_terminal(state):
        outcome = game_result(state)
        if outcome == 1.0:
            print(f"  Result: White wins")
        elif outcome == 0.0:
            print(f"  Result: Black wins")
        else:
            print(f"  Result: Draw")
    else:
        print(f"  Result: stopped at move limit")

    # Per-move quality scoring against the reference engine. This is the
    # signal that lets the trust matrix learn meaningful per-engine
    # differences instead of just "everyone on the winning team is better".
    print()
    print("  Scoring moves against reference engine (classical at depth 4)...")
    from quality_scorer import score_game
    scored = score_game(history)
    averages = cubist.update_after_game(scored["per_move"])

    print()
    print("  Per-engine quality on this game:")
    for engine_id, avg in sorted(averages.items(), key=lambda kv: -kv[1]):
        n = sum(1 for e, _, _ in scored["per_move"] if e == engine_id)
        bar = "█" * int(avg * 30)
        print(f"    {engine_id:<12} avg quality {avg:.3f}  ({n} moves)  {bar}")

    print()
    print(f"  Trust matrix saved to {TRUST_PATH}")
    print()
    print("  Move log:", " ".join(move_log))
    return 0


# ─────────────────────────────────────────────────────────────────────────────
# Helpers — convert our GameState back to FEN, and our move-tuple to UCI
# ─────────────────────────────────────────────────────────────────────────────

def _state_to_fen(state: GameState) -> str:
    """Inverse of from_fen — needed because engine_wiring takes FEN strings."""
    rows = []
    for r in range(8):
        empty = 0
        row_str = ""
        for c in range(8):
            piece = state.board[r * 8 + c]
            if piece is None:
                empty += 1
            else:
                if empty:
                    row_str += str(empty)
                    empty = 0
                # piece is "wP", "bK", etc — convert to FEN char
                color, kind = piece[0], piece[1]
                row_str += kind.upper() if color == "w" else kind.lower()
        if empty:
            row_str += str(empty)
        rows.append(row_str)
    pieces = "/".join(rows)

    castling = ""
    if state.castling.get("wK"): castling += "K"
    if state.castling.get("wQ"): castling += "Q"
    if state.castling.get("bK"): castling += "k"
    if state.castling.get("bQ"): castling += "q"
    castling = castling or "-"

    ep = sq_name(state.ep_square) if state.ep_square is not None else "-"
    return f"{pieces} {state.turn} {castling} {ep} {state.halfmove} {state.fullmove}"


def _move_uci(move: tuple) -> str:
    frm, to, promo = move
    return tuple_to_uci((frm, to, promo))


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main() -> int:
    _configure_stdout()
    parser = argparse.ArgumentParser(
        description="Cubist ensemble chess engine (Layer 3 architecture)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--fen", default=STARTPOS,
                        help="FEN of the position to analyze (default: startpos)")
    parser.add_argument("--explain", action="store_true",
                        help="Print full Layer 1/2/3 diagnostics")
    parser.add_argument("--selfplay", action="store_true",
                        help="Play a self-play game instead of one move")
    parser.add_argument("--moves", type=int, default=20,
                        help="Self-play move limit (default 20)")
    parser.add_argument("--engines", nargs="+",
                        choices=list(ENGINE_REGISTRY.keys()),
                        default=None,
                        help="Subset of engines to use (default: all)")
    parser.add_argument("--reset", action="store_true",
                        help="Delete the persisted trust matrix before running")
    args = parser.parse_args()

    if args.reset and os.path.exists(TRUST_PATH):
        os.remove(TRUST_PATH)
        print(f"  Trust matrix reset ({TRUST_PATH} deleted)")

    engines = args.engines or list(ENGINE_REGISTRY.keys())
    # Drop oracle if no API key — keeps the demo runnable out of the box.
    if "oracle" in engines and not os.environ.get("ANTHROPIC_API_KEY"):
        print("  [info] ANTHROPIC_API_KEY not set; running without oracle")
        engines = [e for e in engines if e != "oracle"]

    if args.selfplay:
        return selfplay(args.moves, engines, args.explain)
    return demo_single(args.fen, args.explain, engines)


if __name__ == "__main__":
    raise SystemExit(main())
