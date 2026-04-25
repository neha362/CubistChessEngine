"""
engine_wiring.py — uniform `propose()` for every engine
========================================================

This module is the bridge between the 5 individual engines (which all have
different APIs and board representations) and Layer 3, which expects a
uniform `EngineProposal(engine_id, move=(frm,to,promo), score_cp, confidence)`.

For every engine variant we expose a `propose(fen, **opts) -> EngineProposal`
function. Each adapter is responsible for:
  1. Converting FEN → the engine's native board representation
  2. Calling the engine's native search
  3. Converting the result → an `EngineProposal`
  4. Computing a `confidence ∈ [0,1]`

Confidence calibration is engine-specific:
  - alpha-beta engines: tanh(|score_cp| / 200) — clear winning/losing scores
    register as confident, near-zero scores as uncertain.
  - MCTS: |win_rate - 0.5| * 2 — distance from 50/50 is the confidence.
  - oracle: a fixed 0.7 baseline (Claude doesn't expose self-uncertainty cleanly).

These are STARTING values, not tuned. Calibration is a future task: log
many positions, compare confidence to actual move quality, refit per engine.
"""

from __future__ import annotations

import math
import os
import sys
from pathlib import Path
from typing import Optional

# ── Path setup ────────────────────────────────────────────────────────────────
# We need to import from sibling directories. Root of the repo is one up.
REPO_ROOT = Path(__file__).resolve().parent.parent
for sub in ("berserker1", "monte_carlo", "classical_minimax", "berserker_2", "scenarios"):
    p = str(REPO_ROOT / sub)
    if p not in sys.path:
        sys.path.insert(0, p)

# Layer 3's EngineProposal is the wire format every adapter targets.
from layer3_ensemble import EngineProposal  # noqa: E402

# Common GameState (used by berserker1 + monte_carlo + scenarios + Layer 3).
# berserker1 and monte_carlo ship identical movegen_agent.py files, so
# importing from one is enough.
from movegen_agent import GameState, from_fen, all_legal_moves  # noqa: E402

# python-chess is used by berserker_2 + classical_minimax.
import chess  # noqa: E402


# ─────────────────────────────────────────────────────────────────────────────
# UCI ↔ tuple conversions
# ─────────────────────────────────────────────────────────────────────────────

_FILES = "abcdefgh"


def _uci_to_tuple(uci: str) -> tuple:
    """'e2e4' → (frm, to, promo). frm/to are 0-63 with row 0 = rank 8."""
    fc = _FILES.index(uci[0])
    fr = 8 - int(uci[1])
    tc = _FILES.index(uci[2])
    tr = 8 - int(uci[3])
    promo = uci[4] if len(uci) >= 5 else ""
    return (fr * 8 + fc, tr * 8 + tc, promo)


def _tuple_to_uci(move: tuple) -> str:
    """(frm, to, promo) → 'e2e4'."""
    frm, to, promo = move
    s = _FILES[frm % 8] + str(8 - frm // 8) + _FILES[to % 8] + str(8 - to // 8)
    return s + (promo or "")


def _chess_move_to_tuple(move: chess.Move) -> tuple:
    """python-chess Move → (frm, to, promo) in OUR coordinate system.

    Note the coordinate flip: python-chess uses square 0 = a1 (row from
    bottom), while our GameState uses index 0 = a8 (row from top, like FEN).
    """
    # python-chess: file = sq & 7, rank = sq >> 3 (0 = rank 1)
    # ours:         col = idx & 7, row = idx >> 3 (0 = rank 8)
    pc_from_file = chess.square_file(move.from_square)
    pc_from_rank = chess.square_rank(move.from_square)
    pc_to_file   = chess.square_file(move.to_square)
    pc_to_rank   = chess.square_rank(move.to_square)
    frm = (7 - pc_from_rank) * 8 + pc_from_file
    to  = (7 - pc_to_rank)   * 8 + pc_to_file
    promo = ""
    if move.promotion is not None:
        promo = chess.piece_symbol(move.promotion)  # 'q', 'r', 'b', 'n'
    return (frm, to, promo)


# ─────────────────────────────────────────────────────────────────────────────
# Confidence calibration helpers
# ─────────────────────────────────────────────────────────────────────────────

def _ab_confidence(score_cp: int) -> float:
    """
    Alpha-beta confidence: clearly winning/losing positions are confident,
    balanced positions are uncertain. tanh(|score|/200) gives:
      score ±50  → conf 0.24
      score ±200 → conf 0.76
      score ±400 → conf 0.96
    """
    return math.tanh(abs(score_cp) / 200.0)


def _mcts_confidence(win_rate: float) -> float:
    """MCTS confidence: distance from 50/50, scaled to [0,1]."""
    return min(1.0, abs(win_rate - 0.5) * 2.0)


# ─────────────────────────────────────────────────────────────────────────────
# Engine 1 — Classical minimax (python-chess board + material/PST eval)
# ─────────────────────────────────────────────────────────────────────────────

_classical_search = None


def _get_classical():
    """Lazy-init the classical search agent (slow imports, side effects)."""
    global _classical_search
    if _classical_search is None:
        from chess_engine.move_gen import MoveGenAgent
        from chess_engine.eval     import EvalAgent
        from chess_engine.search   import SearchAgent
        ev = EvalAgent()
        mg = MoveGenAgent()
        _classical_search = SearchAgent(eval_fn=ev.evaluate, move_gen=mg), ev
    return _classical_search


def propose_classical(fen: str, max_depth: int = 3) -> Optional[EngineProposal]:
    """Run the classical engine, return its proposal (or None if no legal moves)."""
    sa, ev = _get_classical()
    board = chess.Board(fen)
    move = sa.best_move(board, max_depth)
    if move is None:
        return None
    # SearchAgent.best_move only returns the move; rerun the eval for a score
    # estimate. (Production code would want to expose score from search itself.)
    score_cp = ev.evaluate(board)
    if board.turn == chess.BLACK:
        score_cp = -score_cp  # always report from White's POV for consistency
    return EngineProposal(
        engine_id="classical",
        move=_chess_move_to_tuple(move),
        score_cp=int(score_cp),
        confidence=_ab_confidence(score_cp),
        prior_weight=1.0,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Engine 2 — Berserker1 (custom GameState, aggressive negamax)
# ─────────────────────────────────────────────────────────────────────────────

def propose_berserker(fen: str, max_depth: int = 3, movetime_ms: int = 800) -> Optional[EngineProposal]:
    """Aggressive Berserker engine. Native GameState, no conversion needed."""
    import berserker_search_agent as bs  # imported lazily — heavy module
    state = from_fen(fen)
    if not all_legal_moves(state):
        return None
    result = bs.search(state, max_depth=max_depth, movetime_ms=movetime_ms, verbose=False)
    if result.move is None:
        return None
    # Berserker scores are from side-to-move's POV; rotate to White's POV.
    score_cp = result.score if state.turn == "w" else -result.score
    return EngineProposal(
        engine_id="berserker",
        move=result.move,
        score_cp=int(score_cp),
        confidence=_ab_confidence(score_cp),
        prior_weight=1.0,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Engine 3 — Monte Carlo Tree Search (custom GameState, UCB1 rollouts)
# ─────────────────────────────────────────────────────────────────────────────

def propose_mcts(fen: str, max_iter: int = 300, movetime_ms: int = 800) -> Optional[EngineProposal]:
    """MCTS — score is derived from win-rate, confidence from win-rate distance."""
    import mcts_agent as ms  # lazy import
    state = from_fen(fen)
    if not all_legal_moves(state):
        return None
    result = ms.mcts_search(state, max_iter=max_iter, movetime_ms=movetime_ms, verbose=False)
    if result.best_move is None:
        return None
    # Win rate is from side-to-move's POV. Convert to centipawns from White's POV.
    # We use a simple linear map: wr=1.0 → +400cp, wr=0.5 → 0, wr=0.0 → -400cp.
    # 400cp is a reasonable "this side is winning" threshold in our other engines.
    side_score_cp = int((result.win_rate - 0.5) * 800)
    score_cp = side_score_cp if state.turn == "w" else -side_score_cp
    return EngineProposal(
        engine_id="mcts",
        move=result.best_move,
        score_cp=score_cp,
        confidence=_mcts_confidence(result.win_rate),
        prior_weight=1.0,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Engine 4 — Berserker Siege (python-chess + king-zone attack eval)
# This is your `berserker_2` directory. Different aggressive flavor than
# berserker1 — useful as a 4th distinct voice.
# ─────────────────────────────────────────────────────────────────────────────

_siege_search = None


def _get_siege():
    global _siege_search
    if _siege_search is None:
        # berserker_2's modules use generic names that collide with other engines'
        # modules (move_gen, eval, search), so import them via the full path.
        import importlib.util as _ilu

        siege_dir = REPO_ROOT / "berserker_2"

        def _load(name: str):
            spec = _ilu.spec_from_file_location(f"siege_{name}", siege_dir / f"{name}.py")
            mod = _ilu.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return mod

        siege_mg = _load("move_gen").MoveGen()
        siege_ev = _load("eval").Evaluator()
        siege_se = _load("search").Search(extend_checks_in_qsearch=True)
        _siege_search = (siege_mg, siege_ev, siege_se)
    return _siege_search


def propose_siege(fen: str, time_limit: float = 0.6, max_depth: int = 3) -> Optional[EngineProposal]:
    """Berserker Siege — second aggressive voice, python-chess based."""
    mg, ev, se = _get_siege()
    board = chess.Board(fen)
    if not list(board.legal_moves):
        return None
    move, score_cp = se.find_best_move(board, mg, ev, time_limit=time_limit, max_depth=max_depth)
    if move is None:
        return None
    if board.turn == chess.BLACK:
        score_cp = -score_cp  # rotate to White's POV
    return EngineProposal(
        engine_id="siege",
        move=_chess_move_to_tuple(move),
        score_cp=int(score_cp),
        confidence=_ab_confidence(score_cp),
        prior_weight=1.0,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Engine 5 — Claude Oracle (API-based, optional — needs ANTHROPIC_API_KEY)
# Falls back gracefully if the key isn't set, so this still produces 4 voices.
# ─────────────────────────────────────────────────────────────────────────────

ORACLE_SYSTEM_PROMPT = """\
You are a chess engine. Given a FEN and the list of legal moves in UCI form,
return ONLY a JSON object on a single line: {"move":"<uci>","reason":"..."}.
The move MUST be one of the legal moves verbatim. No markdown, no extra text."""


def propose_oracle(fen: str, timeout: float = 12.0) -> Optional[EngineProposal]:
    """
    Call the Claude API for a move. If ANTHROPIC_API_KEY isn't set, return None
    so the ensemble degrades gracefully to 4 engines.
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return None

    import json
    import httpx

    board = chess.Board(fen)
    legal = list(board.legal_moves)
    if not legal:
        return None
    legal_ucis = [m.uci() for m in legal]

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
            r = client.post(
                "https://api.anthropic.com/v1/messages",
                headers={"x-api-key": api_key, "anthropic-version": "2023-06-01",
                         "content-type": "application/json"},
                json=payload,
            )
            r.raise_for_status()
            text = r.json()["content"][0]["text"]
    except Exception as e:
        # API failure: return None, ensemble continues with the other engines.
        print(f"  [oracle] API error ({e}), skipping this proposal")
        return None

    # Parse the last JSON object in Claude's response.
    try:
        last = text.rfind("}")
        first = text.rfind("{", 0, last + 1)
        obj = json.loads(text[first:last + 1])
        uci = obj.get("move", "").strip()
    except Exception:
        uci = ""

    if uci not in legal_ucis:
        # Claude returned something unparseable or illegal. Don't fake a vote.
        return None

    return EngineProposal(
        engine_id="oracle",
        move=_uci_to_tuple(uci),
        score_cp=0,        # Oracle doesn't expose a numeric score.
        confidence=0.7,    # Fixed baseline; calibrate empirically later.
        prior_weight=1.2,  # Slight prior boost — pretrained on millions of games.
    )


# ─────────────────────────────────────────────────────────────────────────────
# Registry of all engines.
# ─────────────────────────────────────────────────────────────────────────────

ENGINE_REGISTRY = {
    "classical": propose_classical,
    "berserker": propose_berserker,
    "mcts":      propose_mcts,
    "siege":     propose_siege,
    "oracle":    propose_oracle,
}


# ─────────────────────────────────────────────────────────────────────────────
# Position-keyed cache for repeated positions (transpositions, analysis).
# Each entry is the list of EngineProposals for one (fen, engines) pair.
# ─────────────────────────────────────────────────────────────────────────────

_PROPOSAL_CACHE: dict = {}
_CACHE_HITS = [0]
_CACHE_MISSES = [0]


def cache_stats() -> dict:
    return {
        "size": len(_PROPOSAL_CACHE),
        "hits": _CACHE_HITS[0],
        "misses": _CACHE_MISSES[0],
    }


def clear_cache() -> None:
    _PROPOSAL_CACHE.clear()
    _CACHE_HITS[0] = 0
    _CACHE_MISSES[0] = 0


def gather_proposals(fen: str, engines: Optional[list] = None,
                     parallel: bool = True, cache: bool = True) -> list:
    """
    Run every requested engine on the position and return their proposals.
    None entries (engine refused / errored) are filtered out.

    parallel=True (default): run engines concurrently in a thread pool.
        Best case speedup is ~Nx for N engines (when they're all CPU-bound
        of similar duration). The oracle's network I/O parallelizes for
        free under threads since httpx releases the GIL on the socket.

        Caveat: berserker1 uses module-level mutable state (_killers, _tt,
        _nodes, _stop in berserker_search_agent.py). Two threads calling
        berserker1.search() at once would corrupt that state. To be safe,
        we run berserker SEQUENTIALLY before kicking off the parallel pool
        for the rest. If you swap berserker1 for berserker_2 (recommended —
        see the head-to-head match), this hazard goes away.

    cache=True (default): cache proposals by FEN+engine-set. Saves the full
        cost of re-searching a transposition. Use clear_cache() if you
        change engine internals between calls.
    """
    selected = engines or list(ENGINE_REGISTRY.keys())
    cache_key = (fen, tuple(selected)) if cache else None

    if cache_key is not None and cache_key in _PROPOSAL_CACHE:
        _CACHE_HITS[0] += 1
        return _PROPOSAL_CACHE[cache_key]
    if cache_key is not None:
        _CACHE_MISSES[0] += 1

    if not parallel or len(selected) <= 1:
        # Sequential path — same behaviour as before.
        out = []
        for name in selected:
            try:
                p = ENGINE_REGISTRY[name](fen)
                if p is not None:
                    out.append(p)
            except Exception as e:
                print(f"  [{name}] crashed: {e}")
        if cache_key is not None:
            _PROPOSAL_CACHE[cache_key] = out
        return out

    # Parallel path. Threads (not processes) because the engines are mostly
    # bound by Python interpreter time and pickling them for multiprocessing
    # is impractical (closures, lazy imports, module state).
    import concurrent.futures as _cf

    out = []

    # Step 1: handle berserker first (sequential), since berserker1 uses
    # module-level globals that aren't thread-safe.
    if "berserker" in selected:
        try:
            p = ENGINE_REGISTRY["berserker"](fen)
            if p is not None:
                out.append(p)
        except Exception as e:
            print(f"  [berserker] crashed: {e}")

    # Step 2: parallel for the rest.
    parallel_engines = [n for n in selected if n != "berserker"]
    if parallel_engines:
        with _cf.ThreadPoolExecutor(max_workers=len(parallel_engines)) as ex:
            futures = {
                ex.submit(ENGINE_REGISTRY[n], fen): n
                for n in parallel_engines
            }
            for fut in _cf.as_completed(futures):
                name = futures[fut]
                try:
                    p = fut.result()
                    if p is not None:
                        out.append(p)
                except Exception as e:
                    print(f"  [{name}] crashed: {e}")

    # Sort to make output deterministic regardless of completion order.
    out.sort(key=lambda p: p.engine_id)

    if cache_key is not None:
        _PROPOSAL_CACHE[cache_key] = out
    return out
