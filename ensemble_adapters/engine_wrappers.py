"""
engine_wrappers.py - Layer 1 proposal generation for the ensemble
=================================================================

This module normalizes a 5 x 5 grid of search/eval pairings into one public
contract:

    propose_<search>_<eval>(fen: str, **opts) -> EngineProposal | None

The five search approaches are:
    classical, chaos, siege, mcts, neural

The five eval approaches are:
    classical, chaos, siege, neural, oracle

That yields 25 Layer 1 proposal nodes. Some pairings are "pure" diagonal
engines, some are cross-pairs, and some are bridge hybrids:

  - MCTS + eval uses short eval-guided rollouts
  - Neural search keeps its JS move selector, but its score/confidence is
    measured through the chosen eval pair so Layer 3 still sees 25 distinct
    nodes

Wrappers are intentionally defensive: if a dependency is missing (for example
python-chess, Node.js, or Anthropic credentials), the affected pair returns
None and the rest of the ensemble continues.
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Callable, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
for sub in ("adapter_code", "scenarios"):
    path = str(REPO_ROOT / sub)
    if path not in sys.path:
        sys.path.insert(0, path)

try:
    import chess  # type: ignore
except Exception:  # pragma: no cover - depends on local environment
    chess = None

from layer3_ensemble import EngineProposal
from chaos_move_gen import all_legal_moves, from_fen, make_move, sq_name


FILES = "abcdefgh"
SEARCH_APPROACHES = ("classical", "chaos", "siege", "mcts", "neural")
EVAL_APPROACHES = ("classical", "chaos", "siege", "neural", "oracle")
TREE_SEARCHABLE_EVALS = {"classical", "chaos", "siege"}


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


def chess_move_to_tuple(move) -> tuple[int, int, str]:
    from_file = chess.square_file(move.from_square)
    from_rank = chess.square_rank(move.from_square)
    to_file = chess.square_file(move.to_square)
    to_rank = chess.square_rank(move.to_square)
    from_sq = (7 - from_rank) * 8 + from_file
    to_sq = (7 - to_rank) * 8 + to_file
    promotion = chess.piece_symbol(move.promotion) if move.promotion is not None else ""
    return (from_sq, to_sq, promotion)


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(-20.0, min(20.0, x))))


def _score_prob(score_cp: float, scale: float = 400.0) -> float:
    return _sigmoid(score_cp / scale)


def score_confidence(score_cp: int, scale: float = 400.0) -> float:
    return abs(2.0 * _score_prob(score_cp, scale=scale) - 1.0)


def _state_to_fen(state) -> str:
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
                color, kind = piece[0], piece[1]
                row_str += kind.upper() if color == "w" else kind.lower()
        if empty:
            row_str += str(empty)
        rows.append(row_str)
    pieces = "/".join(rows)

    castling = ""
    if state.castling.get("wK"):
        castling += "K"
    if state.castling.get("wQ"):
        castling += "Q"
    if state.castling.get("bK"):
        castling += "k"
    if state.castling.get("bQ"):
        castling += "q"
    castling = castling or "-"
    ep = sq_name(state.ep_square) if state.ep_square is not None else "-"
    return f"{pieces} {state.turn} {castling} {ep} {state.halfmove} {state.fullmove}"


_proposal_cache: dict[tuple[str, tuple[str, ...]], list[EngineProposal]] = {}
_cache_hits = 0
_cache_misses = 0
_classical_components = {}
_siege_components = None


def _fen_eval_classical(fen: str) -> float:
    if chess is None:
        raise RuntimeError("python-chess unavailable")
    from classical_eval import EvalAgent

    evaluator = EvalAgent()
    return float(evaluator.evaluate(chess.Board(fen)))


def _fen_eval_chaos(fen: str) -> float:
    import chaos_eval

    return float(chaos_eval.evaluate(from_fen(fen)))


def _fen_eval_siege(fen: str) -> float:
    if chess is None:
        raise RuntimeError("python-chess unavailable")
    from siege_eval import Evaluator as SiegeEvaluator

    evaluator = SiegeEvaluator()
    return float(evaluator.evaluate(chess.Board(fen)))


def _fen_eval_neural(fen: str) -> float:
    bridge = REPO_ROOT / "adapter_code" / "_bridge.js"
    payload = json.dumps({"cmd": "eval", "fen": fen}) + "\n"
    result = subprocess.run(
        ["node", str(bridge)],
        input=payload,
        capture_output=True,
        text=True,
        cwd=str(bridge.parent),
        timeout=8.0,
    )
    if result.returncode != 0 or not result.stdout.strip():
        raise RuntimeError("neural eval bridge failed")
    resp = json.loads(result.stdout.strip().splitlines()[-1])
    return float(resp.get("score", 0.0))


def _fen_eval_oracle(fen: str, timeout: float = 12.0) -> float:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY not set")
    try:
        import httpx
    except Exception as exc:
        raise RuntimeError("httpx unavailable") from exc

    payload = {
        "model": "claude-sonnet-4-20250514",
        "max_tokens": 200,
        "system": (
            "You are a chess evaluator. Return ONLY JSON: "
            '{"score": <integer centipawns from white perspective>}.'
        ),
        "messages": [{"role": "user", "content": f"Evaluate this FEN: {fen}"}],
    }
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
    last_brace = text.rfind("}")
    first_brace = text.rfind("{", 0, last_brace + 1)
    data = json.loads(text[first_brace:last_brace + 1])
    return float(data.get("score", 0.0))


EVAL_BACKENDS: dict[str, Callable[[str], float]] = {
    "classical": _fen_eval_classical,
    "chaos": _fen_eval_chaos,
    "siege": _fen_eval_siege,
    "neural": _fen_eval_neural,
    "oracle": _fen_eval_oracle,
}


def _eval_from_fen(fen: str, eval_id: str) -> float:
    return EVAL_BACKENDS[eval_id](fen)


def _root_oriented_score(fen_before: str, move: tuple[int, int, str], eval_id: str) -> int:
    state = from_fen(fen_before)
    side = state.turn
    child = make_move(state, move)
    raw = _eval_from_fen(_state_to_fen(child), eval_id)
    oriented = raw if side == "w" else -raw
    return int(round(oriented))


def _guidance_eval(search_id: str, eval_id: str) -> str:
    if eval_id in TREE_SEARCHABLE_EVALS:
        return eval_id
    return {
        "classical": "classical",
        "chaos": "chaos",
        "siege": "siege",
        "mcts": eval_id,
        "neural": eval_id,
    }[search_id]


def _get_classical_search(eval_id: str):
    if eval_id in _classical_components:
        return _classical_components[eval_id]
    from classical_search import SearchAgent

    def eval_fn(state) -> int:
        return int(round(_eval_from_fen(state.fen(), eval_id)))

    search = SearchAgent(eval_fn=eval_fn, max_depth=3, time_limit=1.0)
    _classical_components[eval_id] = search
    return search


def _get_siege_components():
    global _siege_components
    if _siege_components is not None:
        return _siege_components
    if chess is None:
        raise RuntimeError("python-chess unavailable")
    from siege_move_gen import MoveGen as SiegeMoveGen
    from siege_search import Search as SiegeSearch

    _siege_components = (SiegeMoveGen(), SiegeSearch(extend_checks_in_qsearch=True))
    return _siege_components


def _search_classical(fen: str, eval_id: str, max_depth: int = 3):
    from move_gen_agent import parse_fen

    state = parse_fen(fen)
    search = _get_classical_search(_guidance_eval("classical", eval_id))
    search.max_depth = max_depth
    move, _score, _pv = search.search(state)
    return None if move is None else uci_to_tuple(move.uci())


def _search_chaos(fen: str, eval_id: str, max_depth: int = 3, movetime_ms: int = 800):
    import chaos_search as search_module

    state = from_fen(fen)
    if not all_legal_moves(state):
        return None

    original_eval = search_module._berserker_eval

    guidance_eval = _guidance_eval("chaos", eval_id)

    def bridged_eval(gs) -> float:
        return _eval_from_fen(_state_to_fen(gs), guidance_eval)

    search_module._berserker_eval = bridged_eval
    try:
        result = search_module.search(state, max_depth=max_depth, movetime_ms=movetime_ms, verbose=False)
    finally:
        search_module._berserker_eval = original_eval
    return result.move


def _search_siege(fen: str, eval_id: str, time_limit: float = 0.6, max_depth: int = 3):
    if chess is None:
        return None
    board = chess.Board(fen)
    if not list(board.legal_moves):
        return None

    move_gen, search = _get_siege_components()

    guidance_eval = _guidance_eval("siege", eval_id)

    class ProxyEval:
        def evaluate(self, board_obj) -> int:
            return int(round(_eval_from_fen(board_obj.fen(), guidance_eval)))

    move, _score = search.find_best_move(board, move_gen, ProxyEval(), time_limit=time_limit, max_depth=max_depth)
    return None if move is None else chess_move_to_tuple(move)


def _search_mcts(
    fen: str,
    eval_id: str,
    max_iter: int = 300,
    movetime_ms: int = 800,
    rollout_samples: int = 5,
    rollout_plies: int = 8,
):
    import mcts_search as search_module
    import mcts_move_gen as mcts_movegen

    state = search_module.from_fen(fen)
    if not all_legal_moves(state):
        return None

    def eval_rollout(state_inner):
        samples = []
        for _ in range(max(1, rollout_samples)):
            current = state_inner
            for _ply in range(max(1, rollout_plies)):
                if search_module.is_terminal(current):
                    samples.append(search_module.game_result(current))
                    break
                move = mcts_movegen.fast_random_move(current)
                if move is None:
                    samples.append(search_module.game_result(current))
                    break
                current = search_module.make_move(current, move)
            else:
                raw = _eval_from_fen(_state_to_fen(current), eval_id)
                samples.append(_score_prob(raw, scale=400.0))
        return sum(samples) / len(samples)

    result = search_module.mcts_search(
        state,
        max_iter=max_iter,
        movetime_ms=movetime_ms,
        verbose=False,
        leaf_value_fn=eval_rollout,
    )
    return result.best_move


def _search_neural(fen: str, _eval_id: str, depth: int = 3, time_ms: int = 3000):
    bridge = REPO_ROOT / "adapter_code" / "_bridge.js"
    payload = json.dumps({"cmd": "move", "fen": fen, "depth": depth, "timeLimitMs": time_ms}) + "\n"
    result = subprocess.run(
        ["node", str(bridge)],
        input=payload,
        capture_output=True,
        text=True,
        cwd=str(bridge.parent),
        timeout=time_ms / 1000 + 5.0,
    )
    if result.returncode != 0 or not result.stdout.strip():
        return None
    resp = json.loads(result.stdout.strip().splitlines()[-1])
    uci = resp.get("uci", "none")
    return None if uci == "none" else uci_to_tuple(uci)


SEARCH_BACKENDS: dict[str, Callable[..., Optional[tuple[int, int, str]]]] = {
    "classical": _search_classical,
    "chaos": _search_chaos,
    "siege": _search_siege,
    "mcts": _search_mcts,
    "neural": _search_neural,
}


def _build_proposal(fen: str, search_id: str, eval_id: str, **opts) -> Optional[EngineProposal]:
    move = SEARCH_BACKENDS[search_id](fen, eval_id, **opts)
    if move is None:
        return None
    try:
        score_cp = _root_oriented_score(fen, move, eval_id)
    except Exception:
        score_cp = 0
    return EngineProposal(
        engine_id=f"{search_id}_{eval_id}",
        move=move,
        score_cp=int(score_cp),
        confidence=score_confidence(score_cp),
        prior_weight=1.0,
        search_id=search_id,
        eval_id=eval_id,
    )


def _make_wrapper(search_id: str, eval_id: str):
    def wrapper(fen: str, **opts) -> Optional[EngineProposal]:
        return _build_proposal(fen, search_id, eval_id, **opts)

    wrapper.__name__ = f"propose_{search_id}_{eval_id}"
    return wrapper


ENGINE_REGISTRY: dict[str, Callable[..., Optional[EngineProposal]]] = {
    f"{search_id}_{eval_id}": _make_wrapper(search_id, eval_id)
    for search_id in SEARCH_APPROACHES
    for eval_id in EVAL_APPROACHES
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

    def run_one(name: str) -> Optional[EngineProposal]:
        try:
            return ENGINE_REGISTRY[name](fen)
        except Exception as exc:
            print(f"[{name}] skipped: {exc}")
            return None

    if not parallel or len(selected) <= 1:
        proposals = [proposal for proposal in (run_one(name) for name in selected) if proposal is not None]
        if cache_key is not None:
            _proposal_cache[cache_key] = proposals
        return proposals

    import concurrent.futures

    proposals: list[EngineProposal] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(selected), 8)) as executor:
        future_map = {executor.submit(run_one, name): name for name in selected}
        for future in concurrent.futures.as_completed(future_map):
            proposal = future.result()
            if proposal is not None:
                proposals.append(proposal)

    proposals.sort(key=lambda proposal: proposal.engine_id)
    if cache_key is not None:
        _proposal_cache[cache_key] = proposals
    return proposals
