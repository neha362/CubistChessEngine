"""
tournament_trust_bridge.py - train Layer 3 trust from tournament games
======================================================================

The tournament runner plays one engine (or combo) against another in a
head-to-head game. Layer 3's trust matrix, by contrast, expects per-position
quality updates keyed by engine_id and scenario profile.

This bridge replays a finished tournament game move by move, scores each move
against the existing reference engine, computes the active scenario profile for
that position, and applies one Bayesian trust update to the engine that made
that move.

This is the efficient "actual gameplay generates training data" design:
  - no 25-engine polling at every ply
  - every real move becomes one labeled trust update
  - the same persistence file used by the ensemble is updated in place
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
for sub in ("adapter_code", "scenarios"):
    path = str(REPO_ROOT / sub)
    if path not in sys.path:
        sys.path.insert(0, path)

from engine_adapter import apply_move
from layer3_ensemble import Layer3Ensemble, detect_scenarios
from move_gen_agent import parse_fen, MoveGenerator
from quality_scorer import REFERENCE_DEPTH, score_move_quality
from chaos_move_gen import from_fen as chaos_from_fen


DEFAULT_TRUST_PATH = str(REPO_ROOT / "scenarios" / "cubist_trust.json")
_GEN = MoveGenerator()
FAST_REFERENCE_DEPTH = 2


class TournamentTrustBridge:
    """Replay tournament games into the Layer 3 trust matrix."""

    def __init__(self, persistence_path: str = DEFAULT_TRUST_PATH):
        self.persistence_path = persistence_path
        self.ensemble = Layer3Ensemble(engine_ids=[], persistence_path=persistence_path)

    def reset(self) -> None:
        """Clear in-memory trust and delete the persisted trust file."""
        self.ensemble = Layer3Ensemble(engine_ids=[], persistence_path=self.persistence_path)
        self.ensemble.trust_matrix.cells.clear()
        self.ensemble.calibrators.clear()
        if self.persistence_path and os.path.exists(self.persistence_path):
            os.remove(self.persistence_path)

    def snapshot(self) -> dict[str, Any]:
        """Return a JSON-safe snapshot for the API/UI."""
        return {
            "path": self.persistence_path,
            "scenario_names": list(self._scenario_names()),
            "engines": sorted(self.ensemble.trust_matrix.cells),
            "matrix": self.ensemble.trust_matrix.snapshot(),
        }

    def train_from_game(
        self,
        start_fen: str,
        uci_moves: list[str],
        white_engine_id: str,
        black_engine_id: str,
        reference_depth: Optional[int] = None,
        autosave: bool = True,
        max_plies: Optional[int] = None,
    ) -> dict[str, Any]:
        """
        Replay a finished game and update trust for the engines that played it.

        Returns a compact summary suitable for the tournament API response.
        """
        state = parse_fen(start_fen)
        per_move: list[dict[str, Any]] = []
        sums: dict[str, float] = {}
        counts: dict[str, int] = {}

        capped_moves = uci_moves if max_plies is None else uci_moves[:max_plies]

        for ply, uci in enumerate(capped_moves):
            fen_before = state.fen()
            engine_id = white_engine_id if state.turn == "w" else black_engine_id
            legal = _GEN.legal_moves(state)
            move = next((candidate for candidate in legal if candidate.uci() == uci), None)
            if move is None:
                # Stop training if the recorded game contains an invalid move.
                break

            scenario_profile = detect_scenarios(chaos_from_fen(fen_before))
            quality = score_move_quality(fen_before, uci, reference_depth or REFERENCE_DEPTH)
            self.ensemble.update(engine_id, scenario_profile, quality, score_cp=None, autosave=False)

            per_move.append(
                {
                    "ply": ply + 1,
                    "engine_id": engine_id,
                    "uci": uci,
                    "quality": quality,
                }
            )
            sums[engine_id] = sums.get(engine_id, 0.0) + quality
            counts[engine_id] = counts.get(engine_id, 0) + 1

            state = apply_move(state, move)

        if autosave:
            self.ensemble._save()

        averages = {engine: (sums[engine] / counts[engine]) for engine in sums}
        return {
            "updated": bool(per_move),
            "moves_trained": len(per_move),
            "reference_depth": reference_depth or REFERENCE_DEPTH,
            "max_plies": max_plies,
            "per_engine_avg": averages,
            "per_move": per_move,
            "trust": self.snapshot(),
        }

    @staticmethod
    def _scenario_names() -> tuple[str, ...]:
        from layer3_ensemble import SCENARIO_NAMES

        return SCENARIO_NAMES


def load_persisted_trust(path: str = DEFAULT_TRUST_PATH) -> dict[str, Any]:
    """Read the raw persisted trust payload if it exists."""
    if not os.path.exists(path):
        return {"scenario_matrix": {}, "calibrators": {}}
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)
