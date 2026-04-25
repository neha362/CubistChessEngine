"""
engine_wrappers.py - stable wrapper boundary for the ensemble
=============================================================

This module gives every engine the same public contract:

    propose_<engine>(fen: str, **opts) -> EngineProposal | None

All wrappers accept FEN at the boundary, convert that FEN into the engine's
native board/state format internally, and return Layer 3's EngineProposal
wire format.
"""

from __future__ import annotations

import math
import os
import sys
from pathlib import Path
from typing import Callable, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
for sub in ("berserker1", "monte_carlo", "classical_minimax", "berserker_2", "scenarios"):
    path = str(REPO_ROOT / sub)
    if path not in sys.path:
        sys.path.insert(0, path)

import chess  # noqa: E402

from layer3_ensemble import EngineProposal  # noqa: E402
from movegen_agent import all_legal_moves, from_fen  # noqa: E402


FILES = "abcdefgh"


def uci_to_tuple(uci: str) -> tuple[int, int, str]:
    from_col = FILES.index(uci[0])
    from_row = 8 - int(uci[1])
    to_col = FILES.index(uci[2])
    to_row = 8 - int(uci[3])
    promotion = uci[4] if len(uci) >= 5 else ""
    return (from_row * 8 + from_col, to_row * 8 + to_col, promotion)


def tuple_to_uci(move: tuple[int, int, str]) -> str:
    from_sq, to_sq, promotion = move
    uci = (
        FILES[from_sq % 8]
        + str(8 - from_sq // 8)
        + FILES[to_sq % 8]
        + str(8 - to_sq // 8)
    )
    return uci + (promotion or "")


def chess_move_to_tuple(move: chess.Move) -> tuple[int, int, str]:
    from_file = chess.square_file(move.from_square)
    from_rank = chess.square_rank(move.from_square)
    to_file = chess.square_file(move.to_square)
    to_rank = chess.square_rank(move.to_square)
    from_sq = (7 - from_rank) * 8 + from_file
    to_sq = (7 - to_rank) * 8 + to_file
    promotion = chess.piece_symbol(move.promotion) if move.promotion is not None else ""
    return (from_sq, to_sq, promotion)


def ab_confidence(score_cp: int) -> float:
    return math.tanh(abs(score_cp) / 200.0)


def mcts_confidence(win_rate: float) -> float:
    return min(1.0, abs(win_rate - 0.5) * 2.0)


_classical_search = None
_siege_search = None
_proposal_cache: dict[tuple[str, tuple[str, ...]], list[EngineProposal]] = {}
_cache_hits = 0
_cache_misses = 0


def _get_classical():
    global _classical_search
    if _classical_search is None:
        from chess_engine.move_gen import MoveGenAgent
        from chess_engine.eval import EvalAgent
        from chess_engine.search import SearchAgent

        evaluator = EvalAgent()
        move_gen = MoveGenAgent()
        _classical_search = (SearchAgent(eval_fn=evaluator.evaluate, move_gen=move_gen), evaluator)
    return _classical_search


def _get_siege():
    global _siege_search
    if _siege_search is None:
        import importlib.util

        siege_dir = REPO_ROOT / "berserker_2"

        def _load(name: str):
            spec = importlib.util.spec_from_file_location(f"siege_{name}", siege_dir / f"{name}.py")
            module = importlib.util.module_from_spec(spec)
            assert spec.loader is not None
            spec.loader.exec_module(module)
            return module

        move_gen = _load("move_gen").MoveGen()
        evaluator = _load("eval").Evaluator()
        search = _load("search").Search(extend_checks_in_qsearch=True)
        _siege_search = (move_gen, evaluator, search)
    return _siege_search


def propose_classical(fen: str, max_depth: int = 3) -> Optional[EngineProposal]:
    search, evaluator = _get_classical()
    board = chess.Board(fen)
    move = search.best_move(board, max_depth)
    if move is None:
        return None
    score_cp = evaluator.evaluate(board)
    if board.turn == chess.BLACK:
        score_cp = -score_cp
    return EngineProposal(
        engine_id="classical",
        move=chess_move_to_tuple(move),
        score_cp=int(score_cp),
        confidence=ab_confidence(score_cp),
        prior_weight=1.0,
    )


def propose_berserker(fen: str, max_depth: int = 3, movetime_ms: int = 800) -> Optional[EngineProposal]:
    import berserker_search_agent as search_module

    state = from_fen(fen)
    if not all_legal_moves(state):
        return None
    result = search_module.search(state, max_depth=max_depth, movetime_ms=movetime_ms, verbose=False)
    if result.move is None:
        return None
    score_cp = result.score if state.turn == "w" else -result.score
    return EngineProposal(
        engine_id="berserker",
        move=result.move,
        score_cp=int(score_cp),
        confidence=ab_confidence(score_cp),
        prior_weight=1.0,
    )


def propose_mcts(fen: str, max_iter: int = 300, movetime_ms: int = 800) -> Optional[EngineProposal]:
    import mcts_agent as search_module

    state = from_fen(fen)
    if not all_legal_moves(state):
        return None
    result = search_module.mcts_search(state, max_iter=max_iter, movetime_ms=movetime_ms, verbose=False)
    if result.best_move is None:
        return None
    side_score_cp = int((result.win_rate - 0.5) * 800)
    score_cp = side_score_cp if state.turn == "w" else -side_score_cp
    return EngineProposal(
        engine_id="mcts",
        move=result.best_move,
        score_cp=score_cp,
        confidence=mcts_confidence(result.win_rate),
        prior_weight=1.0,
    )


def propose_siege(fen: str, time_limit: float = 0.6, max_depth: int = 3) -> Optional[EngineProposal]:
    move_gen, evaluator, search = _get_siege()
    board = chess.Board(fen)
    if not list(board.legal_moves):
        return None
    move, score_cp = search.find_best_move(board, move_gen, evaluator, time_limit=time_limit, max_depth=max_depth)
    if move is None:
        return None
    if board.turn == chess.BLACK:
        score_cp = -score_cp
    return EngineProposal(
        engine_id="siege",
        move=chess_move_to_tuple(move),
        score_cp=int(score_cp),
        confidence=ab_confidence(score_cp),
        prior_weight=1.0,
    )


ORACLE_SYSTEM_PROMPT = """\
You are a chess engine. Given a FEN and the list of legal moves in UCI form,
return ONLY a JSON object on a single line: {"move":"<uci>","reason":"..."}.
The move MUST be one of the legal moves verbatim. No markdown, no extra text."""


def propose_oracle(fen: str, timeout: float = 12.0) -> Optional[EngineProposal]:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return None

    import json
    import httpx

    board = chess.Board(fen)
    legal_moves = list(board.legal_moves)
    if not legal_moves:
        return None
    legal_ucis = [move.uci() for move in legal_moves]

    payload = {
        "model": "claude-sonnet-4-20250514",
        "max_tokens": 200,
        "system": ORACLE_SYSTEM_PROMPT,
        "messages": [{
            "role": "user",
            "content": (
                f"FEN: {fen}\n"
                f"Legal moves: {', '.join(legal_ucis)}\n"
                "Choose the best move."
            ),
        }],
    }

    try:
        with httpx.Client(timeout=timeout) as client:
            response = client.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": api_key,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
                json=payload,
            )
            response.raise_for_status()
            text = response.json()["content"][0]["text"]
    except Exception:
        return None

    try:
        last_brace = text.rfind("}")
        first_brace = text.rfind("{", 0, last_brace + 1)
        payload_obj = json.loads(text[first_brace:last_brace + 1])
        uci = payload_obj.get("move", "").strip()
    except Exception:
        uci = ""

    if uci not in legal_ucis:
        return None

    return EngineProposal(
        engine_id="oracle",
        move=uci_to_tuple(uci),
        score_cp=0,
        confidence=0.7,
        prior_weight=1.2,
    )


ENGINE_REGISTRY: dict[str, Callable[..., Optional[EngineProposal]]] = {
    "classical": propose_classical,
    "berserker": propose_berserker,
    "mcts": propose_mcts,
    "siege": propose_siege,
    "oracle": propose_oracle,
}


def cache_stats() -> dict[str, int]:
    return {
        "size": len(_proposal_cache),
        "hits": _cache_hits,
        "misses": _cache_misses,
    }


def clear_cache() -> None:
    global _cache_hits, _cache_misses
    _proposal_cache.clear()
    _cache_hits = 0
    _cache_misses = 0


def gather_proposals(
    fen: str,
    engines: Optional[list[str]] = None,
    parallel: bool = True,
    cache: bool = True,
) -> list[EngineProposal]:
    global _cache_hits, _cache_misses

    selected = engines or list(ENGINE_REGISTRY.keys())
    cache_key = (fen, tuple(selected)) if cache else None

    if cache_key is not None and cache_key in _proposal_cache:
        _cache_hits += 1
        return _proposal_cache[cache_key]
    if cache_key is not None:
        _cache_misses += 1

    if not parallel or len(selected) <= 1:
        proposals = []
        for name in selected:
            try:
                proposal = ENGINE_REGISTRY[name](fen)
            except Exception as exc:
                print(f"[{name}] crashed: {exc}")
                proposal = None
            if proposal is not None:
                proposals.append(proposal)
        if cache_key is not None:
            _proposal_cache[cache_key] = proposals
        return proposals

    import concurrent.futures

    proposals: list[EngineProposal] = []

    if "berserker" in selected:
        try:
            proposal = ENGINE_REGISTRY["berserker"](fen)
            if proposal is not None:
                proposals.append(proposal)
        except Exception as exc:
            print(f"[berserker] crashed: {exc}")

    parallel_names = [name for name in selected if name != "berserker"]
    if parallel_names:
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(parallel_names)) as executor:
            future_map = {executor.submit(ENGINE_REGISTRY[name], fen): name for name in parallel_names}
            for future in concurrent.futures.as_completed(future_map):
                name = future_map[future]
                try:
                    proposal = future.result()
                except Exception as exc:
                    print(f"[{name}] crashed: {exc}")
                    proposal = None
                if proposal is not None:
                    proposals.append(proposal)

    proposals.sort(key=lambda proposal: proposal.engine_id)
    if cache_key is not None:
        _proposal_cache[cache_key] = proposals
    return proposals
