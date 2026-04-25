from __future__ import annotations

import tempfile

from scenarios.tournament_trust_bridge import TournamentTrustBridge


STARTPOS = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"


def test_bridge_updates_both_engines_from_replayed_game():
    with tempfile.NamedTemporaryFile(suffix=".json") as handle:
        bridge = TournamentTrustBridge(persistence_path=handle.name)
        result = bridge.train_from_game(
            start_fen=STARTPOS,
            uci_moves=["e2e4", "e7e5", "g1f3", "b8c6"],
            white_engine_id="classical",
            black_engine_id="berserker_chaos",
            autosave=False,
        )

        assert result["updated"] is True
        assert result["moves_trained"] == 4
        assert set(result["per_engine_avg"]) == {"classical", "berserker_chaos"}
        snapshot = bridge.snapshot()
        assert "classical" in snapshot["engines"]
        assert "berserker_chaos" in snapshot["engines"]
