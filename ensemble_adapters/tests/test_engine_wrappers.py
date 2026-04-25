from __future__ import annotations

from ensemble_adapters.engine_wrappers import ENGINE_REGISTRY, clear_cache, gather_proposals


STARTPOS = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"


def test_engine_registry_exposes_25_search_eval_pairs():
    assert len(ENGINE_REGISTRY) == 25
    assert "classical_classical" in ENGINE_REGISTRY
    assert "chaos_siege" in ENGINE_REGISTRY
    assert "mcts_neural" in ENGINE_REGISTRY
    assert "neural_oracle" in ENGINE_REGISTRY


def test_gather_proposals_returns_wrapped_votes_for_available_pairs():
    proposals = gather_proposals(
        STARTPOS,
        engines=["chaos_chaos", "mcts_chaos"],
        parallel=False,
        cache=False,
    )
    engine_ids = {proposal.engine_id for proposal in proposals}
    assert engine_ids == {"chaos_chaos", "mcts_chaos"}
    for proposal in proposals:
        assert isinstance(proposal.score_cp, int)
        assert 0.0 <= proposal.confidence <= 1.0
        assert len(proposal.move) == 3
        assert proposal.search_id in {"chaos", "mcts"}
        assert proposal.eval_id == "chaos"


def test_gather_proposals_cache_returns_same_votes():
    clear_cache()
    first = gather_proposals(
        STARTPOS,
        engines=["chaos_chaos", "mcts_chaos"],
        parallel=False,
        cache=True,
    )
    second = gather_proposals(
        STARTPOS,
        engines=["chaos_chaos", "mcts_chaos"],
        parallel=False,
        cache=True,
    )
    assert [proposal.uci for proposal in first] == [proposal.uci for proposal in second]
