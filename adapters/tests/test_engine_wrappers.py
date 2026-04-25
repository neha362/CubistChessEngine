from __future__ import annotations

from adapters.engine_wrappers import clear_cache, gather_proposals


STARTPOS = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"


def test_gather_proposals_returns_wrapped_votes_for_working_engines():
    proposals = gather_proposals(
        STARTPOS,
        engines=["classical", "berserker", "mcts", "siege"],
        parallel=False,
        cache=False,
    )
    engine_ids = {proposal.engine_id for proposal in proposals}
    assert engine_ids == {"classical", "berserker", "mcts", "siege"}
    for proposal in proposals:
        assert isinstance(proposal.score_cp, int)
        assert 0.0 <= proposal.confidence <= 1.0
        assert len(proposal.move) == 3


def test_gather_proposals_cache_returns_same_votes():
    clear_cache()
    first = gather_proposals(
        STARTPOS,
        engines=["classical", "mcts"],
        parallel=False,
        cache=True,
    )
    second = gather_proposals(
        STARTPOS,
        engines=["classical", "mcts"],
        parallel=False,
        cache=True,
    )
    assert [proposal.uci for proposal in first] == [proposal.uci for proposal in second]
