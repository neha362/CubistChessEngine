"""
engine_adapter.py — Tournament Engine Adapter
==============================================
Six chess engines, all behind one interface.

Directory layout
────────────────
engines_v2/
├── engine_adapter.py          ← this file
├── move_gen_agent.py          ← shared board representation (used by root-level imports)
├── search_agent.py            ← shared alpha-beta search backbone
├── eval_agent.py              ← shared PST evaluator + Claude oracle eval
│
├── classical_minimax/         ENGINE 1 — Classical Alpha-Beta
│   └── chess_engine/
│       ├── classical_move_gen.py
│       ├── classical_eval.py
│       ├── classical_search.py
│       └── classical_game_state.py
│
├── berserker_chaos/           ENGINE 2 — Berserker Chaos (GameState negamax)
│   ├── chaos_move_gen.py
│   ├── chaos_eval.py
│   └── chaos_search.py
│
├── berserker_siege/           ENGINE 3 — Berserker Siege (python-chess alpha-beta)
│   ├── siege_move_gen.py
│   ├── siege_eval.py
│   └── siege_search.py
│
├── monte_carlo_mcts/          ENGINE 4 — Monte Carlo Tree Search
│   ├── mcts_move_gen.py
│   ├── mcts_rollout.py
│   └── mcts_search.py
│
├── neural_pattern_js/         ENGINE 5 — Neural Pattern Matcher (JavaScript)
│   ├── nn_move_gen.js
│   ├── nn_eval.js
│   ├── nn_search.js
│   └── _bridge.js
│
└── claude_oracle/             ENGINE 6 — Claude API Oracle
    ├── oracle_move_gen.py
    ├── oracle_search.py
    └── oracle_eval.py

Quick start
───────────
    from engine_adapter import build_engine, run_tournament

    engines = [
        build_engine('classical'),
        build_engine('berserker_chaos'),
        build_engine('berserker_siege'),
        build_engine('mcts'),
        build_engine('neural_nn'),
        build_engine('oracle'),        # requires ANTHROPIC_API_KEY
    ]
    run_tournament(engines, max_moves=120)

Engine names accepted by build_engine()
────────────────────────────────────────
  'classical'        Classical alpha-beta + material/PST (python-chess)
  'berserker_chaos'  Aggressive GameState negamax, king-proximity ordering
  'berserker_siege'  Aggressive python-chess alpha-beta, king-zone eval
  'mcts'             Monte Carlo Tree Search, UCB1 + random rollouts
  'neural_nn'        JS neural-inspired pattern matcher (subprocess bridge)
  'oracle'           Claude API, chain-of-thought move selection (no search tree)
  'oracle_direct'    Claude API, fast single-shot move selection
  'random'           Random legal move (baseline / sanity check)
  'stub'             Material-only alpha-beta (fast testing baseline)
"""

from __future__ import annotations

import asyncio
import json
import math
import random
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import httpx

# ── Shared base agents (all engines communicate via BoardState / FEN) ─────────
from move_gen_agent import BoardState, Move, MoveGenerator, parse_fen
from search_agent   import SearchAgent, stub_eval
from eval_agent     import Evaluator

ENGINES_DIR = Path(__file__).parent

# ─────────────────────────────────────────────────────────────────────────────
# Shared infrastructure
# ─────────────────────────────────────────────────────────────────────────────

_GEN = MoveGenerator()


def apply_move(state: BoardState, move: Move) -> BoardState:
    """Wrapper that correctly maintains halfmove clock and fullmove number."""
    piece   = state.board[move.from_sq[0]][move.from_sq[1]]
    capture = state.board[move.to_sq[0]][move.to_sq[1]] != '' or move.is_en_passant
    ns = _GEN.apply_move(state, move)
    ns.halfmove_clock  = 0 if (piece.upper() == 'P' or capture) else state.halfmove_clock + 1
    ns.fullmove_number = state.fullmove_number + (1 if state.turn == 'b' else 0)
    return ns


# ─────────────────────────────────────────────────────────────────────────────
# FEN converters — the universal bridge between board representations
# ─────────────────────────────────────────────────────────────────────────────

def _to_fen(state: BoardState) -> str:
    return state.fen()

def _fen_to_chess_board(fen: str):
    import chess
    return chess.Board(fen)

def _fen_to_chaos_gamestate(fen: str):
    """FEN → berserker_chaos GameState (1D list, wP/bK piece notation)."""
    chaos_path = str(ENGINES_DIR / 'berserker_chaos')
    if chaos_path not in sys.path:
        sys.path.insert(0, chaos_path)
    from chaos_move_gen import from_fen
    return from_fen(fen)

def _fen_to_mcts_gamestate(fen: str):
    """FEN → monte_carlo_mcts GameState (1D list, wP/bK piece notation)."""
    mcts_path = str(ENGINES_DIR / 'monte_carlo_mcts')
    if mcts_path not in sys.path:
        sys.path.insert(0, mcts_path)
    from mcts_move_gen import from_fen
    return from_fen(fen)

def _tuple_move_to_uci(move: tuple) -> str:
    """Convert (frm:int, to:int, promo:str) → UCI string like 'e2e4'."""
    FILES = 'abcdefgh'
    frm, to, promo = move
    uci = FILES[frm % 8] + str(8 - frm // 8) + FILES[to % 8] + str(8 - to // 8)
    if promo:
        uci += promo.lower()[0]
    return uci


# ─────────────────────────────────────────────────────────────────────────────
# Utility helpers shared by all adapters
# ─────────────────────────────────────────────────────────────────────────────

def _resolve_uci(state: BoardState, uci: str):
    """Find the Move object matching a UCI string in legal moves, or fallback."""
    legal = _GEN.legal_moves(state)
    move  = next((m for m in legal if m.uci() == uci), None)
    if move is None:
        return (legal[0], 0.0, []) if legal else (None, -math.inf, [])
    return move, 0.0, []

def _fallback(state: BoardState):
    legal = _GEN.legal_moves(state)
    return (legal[0], 0.0, []) if legal else (None, -math.inf, [])


# ─────────────────────────────────────────────────────────────────────────────
# EngineAdapter — the single uniform public interface
# ─────────────────────────────────────────────────────────────────────────────

class EngineAdapter:
    """
    All engines expose exactly one method:
        best_move(state: BoardState) -> Optional[str]
    Returns a UCI string (e.g. 'e2e4') or None if there are no legal moves.
    """

    def __init__(self, name: str, search_fn: Callable, description: str = ""):
        self.name        = name
        self.description = description
        self._search     = search_fn

    def best_move(self, state: BoardState) -> Optional[str]:
        move, score, _pv = self._search(state)
        if move is None or (isinstance(score, float) and math.isinf(score)):
            return None
        return move.uci() if isinstance(move, Move) else str(move)

    def __repr__(self):
        return f"EngineAdapter({self.name!r})"


# ═════════════════════════════════════════════════════════════════════════════
# ENGINE 1 — Classical Minimax
# classical_minimax/chess_engine/
# ═════════════════════════════════════════════════════════════════════════════
#
# The control-group engine. Pure alpha-beta minimax with iterative deepening,
# Zobrist transposition table, and a standard material + piece-square-table
# (PST) evaluator. Uses python-chess for move generation (most battle-tested
# legal-move generator available). No AI involved — every decision is
# deterministic given the same position and depth.
#
# Files:
#   classical_move_gen.py  — thin wrapper around python-chess board.legal_moves
#   classical_eval.py      — material values + 8 PST tables + open-file bonus
#   classical_search.py    — negamax alpha-beta, Zobrist TT, iterative deepening
#   classical_game_state.py — terminal detection helpers (checkmate/stalemate/draw)
#
# Bridge: BoardState → FEN → chess.Board → chess.Move → UCI string

def build_classical(max_depth: int = 4, time_limit: float = 10.0) -> EngineAdapter:
    cm_dir = str(ENGINES_DIR / 'classical_minimax')
    if cm_dir not in sys.path:
        sys.path.insert(0, cm_dir)
    from chess_engine.classical_eval     import EvalAgent    as CmEval
    from chess_engine.classical_move_gen import MoveGenAgent as CmMoveGen
    from chess_engine.classical_search   import SearchAgent  as CmSearch

    ev  = CmEval()
    mg  = CmMoveGen()
    sa  = CmSearch(eval_fn=ev.evaluate, move_gen=mg)

    def _search(state: BoardState):
        board = _fen_to_chess_board(_to_fen(state))
        try:
            m = sa.best_move(board, max_depth)
            return _resolve_uci(state, m.uci()) if m else _fallback(state)
        except Exception:
            return _fallback(state)

    return EngineAdapter(
        'classical', _search,
        f'Classical alpha-beta depth={max_depth}, material+PST eval (python-chess)'
    )


# ═════════════════════════════════════════════════════════════════════════════
# ENGINE 2 — Berserker Chaos
# berserker_chaos/
# ═════════════════════════════════════════════════════════════════════════════
#
# An aggressive engine using its own custom GameState representation (a flat
# 64-element list with pieces encoded as "wP", "bK", etc.). The evaluator
# deliberately discounts material (pieces are worth ~35% of normal) and
# instead rewards proximity to the enemy king, pawn storms, and checks.
# Move ordering is also biased — checking moves are searched before captures.
# No quiescence search: it charges blindly into tactical chaos.
#
# Files:
#   chaos_move_gen.py  — full legal move generation for the GameState format
#   chaos_eval.py      — aggressive PSTs, attack-proximity bonus, pawn-storm bonus
#   chaos_search.py    — negamax + alpha-beta + Zobrist TT + proximity move ordering
#
# Bridge: BoardState → FEN → GameState → (frm,to,promo) tuple → UCI string

def build_berserker_chaos(max_depth: int = 4, time_limit: float = 10.0) -> EngineAdapter:
    chaos_dir = str(ENGINES_DIR / 'berserker_chaos')
    if chaos_dir not in sys.path:
        sys.path.insert(0, chaos_dir)
    import chaos_search as cs

    movetime_ms = int(time_limit * 1000)

    def _search(state: BoardState):
        gs = _fen_to_chaos_gamestate(_to_fen(state))
        r  = cs.search(gs, max_depth=max_depth, movetime_ms=movetime_ms, verbose=False)
        if r.move is None:
            return _fallback(state)
        return _resolve_uci(state, r.uci)

    return EngineAdapter(
        'berserker_chaos', _search,
        f'Berserker Chaos: aggressive negamax depth={max_depth}, king-proximity ordering'
    )


# ═════════════════════════════════════════════════════════════════════════════
# ENGINE 3 — Berserker Siege
# berserker_siege/
# ═════════════════════════════════════════════════════════════════════════════
#
# A second aggressive engine, but built on python-chess instead of the custom
# GameState. The evaluator scores pieces attacking the enemy king zone with a
# super-linear coordination bonus (many attackers converging on one king score
# disproportionately more than isolated attackers). Includes quiescence search
# extended through checks — it can "see through" sacrifices a few plies past
# the horizon. More methodical than Chaos but still reckless.
#
# Files:
#   siege_move_gen.py  — python-chess wrapper with check-first move ordering
#   siege_eval.py      — king-zone attack bonus, pawn-storm bonus, attacker coordination
#   siege_search.py    — negamax alpha-beta + TT + quiescence (captures + checks)
#
# Bridge: BoardState → FEN → chess.Board → chess.Move → UCI string

def build_berserker_siege(max_depth: int = 4, time_limit: float = 10.0) -> EngineAdapter:
    siege_dir = str(ENGINES_DIR / 'berserker_siege')
    if siege_dir not in sys.path:
        sys.path.insert(0, siege_dir)
    from siege_eval     import Evaluator as SiegeEval
    from siege_move_gen import MoveGen   as SiegeMoveGen
    from siege_search   import Search    as SiegeSearch

    ev  = SiegeEval()
    mg  = SiegeMoveGen()
    sa  = SiegeSearch(extend_checks_in_qsearch=True)

    def _search(state: BoardState):
        board = _fen_to_chess_board(_to_fen(state))
        try:
            m, score = sa.find_best_move(board, mg, ev,
                                         time_limit=time_limit, max_depth=max_depth)
            if m is None:
                return _fallback(state)
            return _resolve_uci(state, m.uci())
        except Exception:
            return _fallback(state)

    return EngineAdapter(
        'berserker_siege', _search,
        f'Berserker Siege: python-chess alpha-beta depth={max_depth}, king-zone eval + quiescence'
    )


# ═════════════════════════════════════════════════════════════════════════════
# ENGINE 4 — Monte Carlo Tree Search
# monte_carlo_mcts/
# ═════════════════════════════════════════════════════════════════════════════
#
# No hand-crafted evaluation function at all. Instead, from each candidate
# position, simulate hundreds of random games to completion and pick the move
# that wins most often. Uses UCB1 (Upper Confidence Bound) to balance
# exploration vs exploitation across the tree. This is conceptually how
# AlphaGo worked at its core before neural networks replaced the random rollouts.
#
# Three rollout policies are available (swap via ACTIVE_POLICY):
#   POLICY_RANDOM    — pure random playout (fastest)
#   POLICY_CAPTURE   — prefer captures over quiet moves (slightly stronger)
#   POLICY_MATERIAL  — always take the most valuable piece available
#
# Files:
#   mcts_move_gen.py  — legal move generation + fast_random_move for rollouts
#   mcts_rollout.py   — rollout policies, runs random games to terminal state
#   mcts_search.py    — UCB1 tree, select/expand/simulate/backpropagate loop
#
# Bridge: BoardState → FEN → GameState → (frm,to,promo) tuple → UCI string

def build_mcts(iterations: int = 400, time_limit: float = 8.0) -> EngineAdapter:
    mcts_dir = str(ENGINES_DIR / 'monte_carlo_mcts')
    if mcts_dir not in sys.path:
        sys.path.insert(0, mcts_dir)
    import mcts_search as ms

    movetime_ms = int(time_limit * 1000)

    def _search(state: BoardState):
        gs = _fen_to_mcts_gamestate(_to_fen(state))
        r  = ms.mcts_search(gs, max_iter=iterations,
                            movetime_ms=movetime_ms, verbose=False)
        if r.best_move is None:
            return _fallback(state)
        return _resolve_uci(state, r.best_uci)

    return EngineAdapter(
        'mcts', _search,
        f'Monte Carlo Tree Search: UCB1, {iterations} iterations per move'
    )


# ═════════════════════════════════════════════════════════════════════════════
# ENGINE 5 — Neural Pattern Matcher (JavaScript)
# neural_pattern_js/
# ═════════════════════════════════════════════════════════════════════════════
#
# Written entirely in JavaScript and run as a subprocess. Represents the
# "neural-inspired" approach: instead of training a neural net, a table of
# ~40 hand-crafted positional rules is compiled (bishop pair, knight outpost,
# rook on open file, passed pawns, king safety, etc.) and each rule is a
# function that fires when the board matches a pattern. The total score is the
# weighted sum of all firing rules — mimicking what a shallow neural net's
# output layer does. Combined with alpha-beta search at depth 3-4.
#
# Files:
#   nn_move_gen.js  — JS legal move generation (perft-verified)
#   nn_eval.js      — JS pattern rule table + material/PST layer
#   nn_search.js    — JS iterative-deepening alpha-beta + MVV-LVA + history heuristic
#   _bridge.js      — stdin/stdout JSON protocol bridge (spawned by Python)
#
# Bridge: BoardState → FEN → JSON → Node.js → JSON → UCI string (subprocess)

class _NeuralNNRunner:
    """Manages the Node.js subprocess bridge for the JS engine."""

    def __init__(self):
        self._dir = ENGINES_DIR / 'neural_pattern_js'

    def get_move(self, fen: str, depth: int = 3, time_ms: int = 3000) -> Optional[str]:
        bridge = str(self._dir / '_bridge.js')
        payload = json.dumps({'cmd': 'move', 'fen': fen,
                              'depth': depth, 'timeLimitMs': time_ms}) + '\n'
        try:
            result = subprocess.run(
                ['node', bridge], input=payload,
                capture_output=True, text=True,
                cwd=str(self._dir), timeout=time_ms / 1000 + 5.0,
            )
            if result.returncode != 0 or not result.stdout.strip():
                return None
            resp = json.loads(result.stdout.strip().split('\n')[-1])
            uci  = resp.get('uci', 'none')
            return None if uci == 'none' else uci
        except Exception:
            return None


def build_neural_nn(depth: int = 3, time_limit: float = 5.0) -> EngineAdapter:
    runner  = _NeuralNNRunner()
    time_ms = int(time_limit * 1000)

    def _search(state: BoardState):
        uci = runner.get_move(_to_fen(state), depth=depth, time_ms=time_ms)
        if uci is None:
            return _fallback(state)
        return _resolve_uci(state, uci)

    return EngineAdapter(
        'neural_nn', _search,
        f'JS neural pattern matcher: {len([])} rules + alpha-beta depth={depth}'
    )


# ═════════════════════════════════════════════════════════════════════════════
# ENGINE 6 — Claude Oracle
# claude_oracle/
# ═════════════════════════════════════════════════════════════════════════════
#
# No search tree at all. Every move decision is a direct call to the Claude
# API. The board position is sent as a FEN string along with the list of legal
# moves, and Claude reasons about the position and returns its chosen move as
# a JSON object.
#
# Two modes:
#   'cot' (chain-of-thought) — Claude identifies key features, shortlists 3
#         candidate moves with reasoning, then commits. Slower (~600 tokens)
#         but the reasoning is visible and often tactically interesting.
#   'direct' — Single-shot, returns just the move and a one-line reason.
#         Faster (~120 tokens) but less considered.
#
# The oracle draws on pattern recognition from its training data (millions of
# chess games), not explicit search. It will occasionally produce brilliant
# intuitive sacrifices that shallow search would never find — and occasionally
# hang pieces. That unpredictability is the point.
#
# Files:
#   oracle_move_gen.py  — BoardState legal move generation (same as our base agent)
#   oracle_search.py    — alpha-beta search backbone (used for hybrid oracle+search)
#   oracle_eval.py      — material+PST evaluator AND oracle_eval() API call function
#
# Bridge: BoardState → FEN + legal UCIs → Claude API → JSON → UCI string

_ORACLE_COT_PROMPT = """\
You are a grandmaster chess player making a single move decision.
Think step-by-step:
  1. Identify the key tactical and positional features of the position.
  2. Shortlist 3 candidate moves and explain each briefly.
  3. Choose the best move and state why.
Then output ONLY valid JSON on the final line (no markdown):
{"move": "<uci>", "reason": "<one sentence>"}"""

_ORACLE_DIRECT_PROMPT = """\
You are a grandmaster chess player. Given a FEN and legal moves, choose the best move.
Return ONLY JSON (no markdown): {"move": "<uci>", "reason": "<one sentence>"}"""


async def _call_claude(payload: dict) -> dict:
    async with httpx.AsyncClient(timeout=45) as client:
        r = await client.post(
            "https://api.anthropic.com/v1/messages",
            headers={"Content-Type": "application/json"},
            json=payload,
        )
        r.raise_for_status()
        return r.json()


def _extract_uci_from_oracle(text: str, legal_ucis: list) -> str:
    """Parse Claude's response — finds the last JSON object and reads the 'move' field."""
    last  = text.rfind("}")
    first = text.rfind("{", 0, last + 1)
    if first != -1:
        try:
            obj = json.loads(text[first:last + 1])
            uci = obj.get("move", "")
            if uci in legal_ucis:
                print(f"    [oracle] {uci}  — {obj.get('reason', '')}")
                return uci
        except json.JSONDecodeError:
            pass
    # Fallback: scan for any legal UCI token in the response text
    for token in text.split():
        token = token.strip(".,;\"'()")
        if token in legal_ucis:
            print(f"    [oracle] fallback token match: {token}")
            return token
    print("    [oracle] could not parse response, using first legal move")
    return legal_ucis[0]


def _make_oracle_search_fn(mode: str = 'cot') -> Callable:
    system     = _ORACLE_COT_PROMPT if mode == 'cot' else _ORACLE_DIRECT_PROMPT
    max_tokens = 600 if mode == 'cot' else 120

    def _search(state: BoardState):
        moves = _GEN.legal_moves(state)
        if not moves:
            return None, -math.inf, []
        legal_ucis = [m.uci() for m in moves]
        payload = {
            "model": "claude-sonnet-4-20250514",
            "max_tokens": max_tokens,
            "system": system,
            "messages": [{
                "role": "user",
                "content": (
                    f"Position (FEN): {state.fen()}\n"
                    f"Legal moves: {', '.join(legal_ucis)}\n"
                    f"It is {'White' if state.turn == 'w' else 'Black'}'s turn.\n"
                    "Choose the best move."
                ),
            }],
        }
        try:
            data = asyncio.run(_call_claude(payload))
            text = data["content"][0]["text"]
            uci  = _extract_uci_from_oracle(text, legal_ucis)
            move = next(m for m in moves if m.uci() == uci)
            return move, 0.0, [move]
        except Exception as e:
            print(f"    [oracle] API error ({e}), falling back to first legal move")
            return _fallback(state)

    return _search


def build_oracle(mode: str = 'cot') -> EngineAdapter:
    label = 'oracle' if mode == 'cot' else 'oracle_direct'
    desc  = ('Claude API oracle: chain-of-thought reasoning, no search tree'
             if mode == 'cot' else
             'Claude API oracle: direct move selection, no search tree')
    return EngineAdapter(label, _make_oracle_search_fn(mode), desc)


# ─────────────────────────────────────────────────────────────────────────────
# build_engine() — main factory
# ─────────────────────────────────────────────────────────────────────────────

def build_engine(variant: str, max_depth: int = 4,
                 time_limit: float = 10.0, **kwargs) -> EngineAdapter:
    """
    Build any tournament engine by name.

    Parameters
    ──────────
    variant     : engine name string (see list below)
    max_depth   : search depth for tree-based engines (ignored by oracle/mcts)
    time_limit  : seconds per move
    **kwargs    : engine-specific options:
                    mcts        → iterations=400
                    neural_nn   → depth=3
                    oracle      → (no extra kwargs)

    Available variants
    ──────────────────
    'classical'         Engine 1: Classical alpha-beta + PST
    'berserker_chaos'   Engine 2: Aggressive GameState negamax
    'berserker_siege'   Engine 3: Aggressive python-chess alpha-beta + quiescence
    'mcts'              Engine 4: Monte Carlo Tree Search
    'neural_nn'         Engine 5: JS neural pattern matcher (Node.js subprocess)
    'oracle'            Engine 6: Claude API, chain-of-thought mode
    'oracle_direct'     Engine 6: Claude API, fast direct mode
    'random'            Baseline: random legal move
    'stub'              Baseline: material-only alpha-beta (for testing)
    """
    if variant == 'classical':
        return build_classical(max_depth, time_limit)

    if variant == 'berserker_chaos':
        return build_berserker_chaos(max_depth, time_limit)

    if variant == 'berserker_siege':
        return build_berserker_siege(max_depth, time_limit)

    if variant == 'mcts':
        return build_mcts(kwargs.get('iterations', 400), time_limit)

    if variant == 'neural_nn':
        return build_neural_nn(kwargs.get('depth', 3), time_limit)

    if variant == 'oracle':
        return build_oracle('cot')

    if variant == 'oracle_direct':
        return build_oracle('direct')

    if variant == 'random':
        def _rand(s):
            mv = _GEN.legal_moves(s)
            if not mv: return None, -math.inf, []
            m = random.choice(mv); return m, 0.0, [m]
        return EngineAdapter('random', _rand, 'Random legal move baseline')

    if variant == 'stub':
        agent = SearchAgent(eval_fn=stub_eval, max_depth=max_depth, time_limit=time_limit)
        return EngineAdapter('stub', agent.search, 'Material-only alpha-beta (testing baseline)')

    raise ValueError(
        f"Unknown variant {variant!r}. Options: "
        "classical, berserker_chaos, berserker_siege, mcts, neural_nn, "
        "oracle, oracle_direct, random, stub"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Game result / tournament data structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class GameResult:
    winner:    Optional[str]   # 'white', 'black', or None (draw)
    reason:    str             # 'checkmate', 'stalemate', '50-move', 'repetition', etc.
    moves:     list            # list of UCI strings
    final_fen: str

    def __str__(self):
        if self.winner:
            return f"{self.winner.upper()} wins by {self.reason} in {len(self.moves)} moves"
        return f"Draw by {self.reason} after {len(self.moves)} moves"


@dataclass
class TournamentResult:
    engines: list
    wins:    dict = field(default_factory=dict)
    losses:  dict = field(default_factory=dict)
    draws:   dict = field(default_factory=dict)
    games:   list = field(default_factory=list)

    def score_table(self):
        out = []
        for name in self.engines:
            w = self.wins.get(name, 0); d = self.draws.get(name, 0)
            t = w + self.losses.get(name, 0) + d
            out.append((name, (w + 0.5 * d) / t if t else 0.0))
        return sorted(out, key=lambda x: -x[1])

    def __str__(self):
        lines = ["", "=" * 65, "  TOURNAMENT RESULTS", "=" * 65]
        for rank, (name, score) in enumerate(self.score_table(), 1):
            w = self.wins.get(name, 0); d = self.draws.get(name, 0); l = self.losses.get(name, 0)
            lines.append(f"  {rank}. {name:<28s}  W:{w}  D:{d}  L:{l}  ({score:.0%})")
        lines.append("=" * 65)
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# GameRunner — plays a full game between two EngineAdapters
# ─────────────────────────────────────────────────────────────────────────────

class GameRunner:
    """
    Plays a complete game, handling all terminal conditions:
      - Checkmate
      - Stalemate
      - 50-move rule (halfmove clock reaches 100)
      - Threefold repetition
      - Max-moves cap (returns draw by 'max_moves')
      - Illegal move forfeit
    """

    def play(
        self,
        white:     EngineAdapter,
        black:     EngineAdapter,
        fen:       str  = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        max_moves: int  = 200,
        verbose:   bool = False,
    ) -> GameResult:

        state = parse_fen(fen)
        log:  list = []
        rep:  dict = {}

        for ply in range(max_moves * 2):
            # Repetition detection (ignore clocks for the key)
            key = ' '.join(state.fen().split()[:4])
            rep[key] = rep.get(key, 0) + 1
            if rep[key] >= 3:
                return GameResult(None, 'repetition', log, state.fen())

            # 50-move rule
            if state.halfmove_clock >= 100:
                return GameResult(None, '50-move', log, state.fen())

            engine = white if state.turn == 'w' else black
            uci    = engine.best_move(state)

            # No legal moves
            if uci is None:
                if _GEN.is_checkmate(state):
                    winner = 'black' if state.turn == 'w' else 'white'
                    return GameResult(winner, 'checkmate', log, state.fen())
                return GameResult(None, 'stalemate', log, state.fen())

            # Validate the move
            legal = _GEN.legal_moves(state)
            move  = next((m for m in legal if m.uci() == uci), None)
            if move is None:
                winner = 'black' if state.turn == 'w' else 'white'
                return GameResult(winner, f'illegal_move:{uci}', log, state.fen())

            log.append(uci)
            state = apply_move(state, move)

            if verbose:
                side = 'W' if ply % 2 == 0 else 'B'
                print(f"  {side}{ply // 2 + 1}: {uci}  "
                      f"hmc={state.halfmove_clock}  full={state.fullmove_number}")

        return GameResult(None, 'max_moves', log, state.fen())


# ─────────────────────────────────────────────────────────────────────────────
# run_tournament() — full round-robin
# ─────────────────────────────────────────────────────────────────────────────

def run_tournament(
    engines:   list,
    fen:       str  = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    max_moves: int  = 150,
    verbose:   bool = False,
) -> TournamentResult:
    """
    Every engine plays every other engine twice (once per colour).
    Prints a live result after each game and a final leaderboard.
    """
    names  = [e.name for e in engines]
    result = TournamentResult(engines=names)
    runner = GameRunner()
    total  = len(engines) * (len(engines) - 1)
    played = 0

    for i, eng_a in enumerate(engines):
        for j, eng_b in enumerate(engines):
            if i >= j:
                continue
            for white, black in [(eng_a, eng_b), (eng_b, eng_a)]:
                played += 1
                print(f"\n[{played}/{total}] {white.name} (W) vs {black.name} (B)")
                game = runner.play(white, black, fen=fen,
                                   max_moves=max_moves, verbose=verbose)
                result.games.append(game)
                print(f"  → {game}")

                if game.winner == 'white':
                    result.wins[white.name]   = result.wins.get(white.name, 0) + 1
                    result.losses[black.name] = result.losses.get(black.name, 0) + 1
                elif game.winner == 'black':
                    result.wins[black.name]   = result.wins.get(black.name, 0) + 1
                    result.losses[white.name] = result.losses.get(white.name, 0) + 1
                else:
                    result.draws[white.name] = result.draws.get(white.name, 0) + 1
                    result.draws[black.name] = result.draws.get(black.name, 0) + 1

    print(result)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Smoke test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    START = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    state = parse_fen(START)

    print("=" * 65)
    print("  Unified Engine Adapter — single-move smoke test")
    print("=" * 65)
    print(f"  {'Engine':<20}  {'Move':>6}  Description")
    print(f"  {'-'*20}  {'-'*6}  {'-'*35}")

    variants = [
        ('classical',       dict(max_depth=2, time_limit=5.0)),
        ('berserker_chaos', dict(max_depth=2, time_limit=5.0)),
        ('berserker_siege', dict(max_depth=2, time_limit=5.0)),
        ('mcts',            dict(time_limit=3.0, iterations=80)),
        ('neural_nn',       dict(time_limit=5.0, depth=2)),
        ('random',          {}),
        # ('oracle',        {}),  # uncomment when ANTHROPIC_API_KEY is set
    ]

    all_ok = True
    for variant, kwargs in variants:
        try:
            eng  = build_engine(variant, **kwargs)
            move = eng.best_move(state)
            print(f"  {eng.name:<20}  {move or '(none)':>6}  {eng.description[:45]}")
        except Exception as e:
            print(f"  {variant:<20}  {'FAIL':>6}  {e}")
            all_ok = False

    print()
    print(f"  {'All engines OK' if all_ok else 'Some engines failed'}")

    print("\n-- Two cross-family games --")
    runner = GameRunner()
    pairs = [
        ('classical',       'berserker_chaos',  dict(max_depth=2)),
        ('berserker_siege', 'mcts',             dict(max_depth=2, iterations=60)),
    ]
    for wn, bn, kw in pairs:
        w = build_engine(wn, time_limit=4.0, **kw)
        b = build_engine(bn, time_limit=4.0, **kw)
        r = runner.play(w, b, max_moves=30)
        print(f"  {wn} vs {bn}: {r}")


# ─────────────────────────────────────────────────────────────────────────────
# build_combo() — Mix-and-Match: any search + any eval
# ─────────────────────────────────────────────────────────────────────────────

def build_combo(
    search:     str,
    eval_fn:    str,
    max_depth:  int   = 4,
    time_limit: float = 10.0,
    **kwargs,
) -> EngineAdapter:
    """
    Build a custom engine by pairing any search strategy with any eval function.

    search options:
        'classical_search'   alpha-beta + iterative deepening + TT
        'chaos_search'       aggressive negamax, check-first ordering, no quiescence
        'siege_search'       alpha-beta + quiescence through captures and checks
        'mcts_search'        UCB1 Monte Carlo tree + random rollouts (ignores eval_fn)
        'nn_search'          JS alpha-beta via subprocess (ignores eval_fn, uses nn_eval)

    eval_fn options:
        'classical_eval'     material + piece-square tables + open-file bonus
        'chaos_eval'         attack proximity + pawn storm + discounted material
        'siege_eval'         king-zone attack bonus + super-linear coordination
        'nn_eval'            ~40 pattern rules via JS subprocess
        'oracle_eval'        Claude API position scorer (slow, needs ANTHROPIC_API_KEY)

    Examples:
        build_combo('siege_search', 'chaos_eval')         # siege discipline + chaos aggression
        build_combo('classical_search', 'oracle_eval')    # deep search + Claude scoring
        build_combo('chaos_search', 'siege_eval')         # chaos tree + king-zone awareness
        build_combo('mcts_search',  'classical_eval')     # MCTS with material rollout scoring
    """
    name = f"{search.replace('_search','')} + {eval_fn.replace('_eval','')}"

    # ── Resolve the eval function ──────────────────────────────────────────

    def _get_eval():
        if eval_fn == 'classical_eval':
            cm_dir = str(ENGINES_DIR / 'classical_minimax')
            if cm_dir not in sys.path: sys.path.insert(0, cm_dir)
            from chess_engine.classical_eval import EvalAgent
            import chess
            _ev = EvalAgent()
            # Wrap: BoardState → chess.Board → score
            def classical_eval_fn(state: BoardState) -> float:
                return float(_ev.evaluate(_fen_to_chess_board(_to_fen(state))))
            return classical_eval_fn

        if eval_fn == 'chaos_eval':
            chaos_dir = str(ENGINES_DIR / 'berserker_chaos')
            if chaos_dir not in sys.path: sys.path.insert(0, chaos_dir)
            from chaos_eval import evaluate as _ce
            # chaos eval takes GameState; wrap with FEN bridge
            def chaos_eval_fn(state: BoardState) -> float:
                gs = _fen_to_chaos_gamestate(_to_fen(state))
                return float(_ce(gs))
            return chaos_eval_fn

        if eval_fn == 'siege_eval':
            siege_dir = str(ENGINES_DIR / 'berserker_siege')
            if siege_dir not in sys.path: sys.path.insert(0, siege_dir)
            from siege_eval import Evaluator as _SE
            _ev = _SE()
            def siege_eval_fn(state: BoardState) -> float:
                return float(_ev.evaluate(_fen_to_chess_board(_to_fen(state))))
            return siege_eval_fn

        if eval_fn == 'nn_eval':
            # Call the JS eval via subprocess
            nn_dir = ENGINES_DIR / 'neural_pattern_js'
            def nn_eval_fn(state: BoardState) -> float:
                payload = json.dumps({'cmd': 'eval', 'fen': _to_fen(state)}) + '\n'
                try:
                    result = subprocess.run(
                        ['node', str(nn_dir / '_bridge.js')],
                        input=payload, capture_output=True, text=True,
                        cwd=str(nn_dir), timeout=10.0,
                    )
                    resp = json.loads(result.stdout.strip().split('\n')[-1])
                    return float(resp.get('score', 0))
                except Exception:
                    return 0.0
            return nn_eval_fn

        if eval_fn == 'oracle_eval':
            from eval_agent import oracle_eval as _oe
            return _oe

        # Fallback: our base PST evaluator
        return Evaluator().evaluate

    # ── Resolve the search strategy ────────────────────────────────────────

    if search == 'classical_search':
        resolved_eval = _get_eval()
        agent = SearchAgent(eval_fn=resolved_eval, max_depth=max_depth, time_limit=time_limit)
        return EngineAdapter(name, agent.search,
                             f'Classical α-β search + {eval_fn}')

    if search == 'chaos_search':
        # chaos_search uses its own internal eval call; we replace it by
        # monkey-patching the eval import in the chaos module at import time
        chaos_dir = str(ENGINES_DIR / 'berserker_chaos')
        if chaos_dir not in sys.path: sys.path.insert(0, chaos_dir)
        import chaos_search as _cs_mod
        resolved_eval = _get_eval()
        # Swap the eval function chaos_search calls internally
        import chaos_eval as _ce_mod
        _orig_evaluate = _ce_mod.evaluate

        def chaos_combo_search(state: BoardState):
            fen = _to_fen(state)
            gs  = _fen_to_chaos_gamestate(fen)
            # Temporarily swap chaos eval with our chosen eval (FEN-bridged)
            def _bridged(gs_inner):
                # Convert GameState back to BoardState for the eval
                inner_fen = _chaos_gs_to_fen(gs_inner)
                inner_bs  = parse_fen(inner_fen)
                return resolved_eval(inner_bs)
            _ce_mod.evaluate = _bridged
            try:
                r = _cs_mod.search(gs, max_depth=max_depth,
                                   movetime_ms=int(time_limit*1000), verbose=False)
            finally:
                _ce_mod.evaluate = _orig_evaluate  # always restore
            if r.move is None:
                return _fallback(state)
            return _resolve_uci(state, r.uci)

        return EngineAdapter(name, chaos_combo_search,
                             f'Chaos negamax + {eval_fn}')

    if search == 'siege_search':
        siege_dir = str(ENGINES_DIR / 'berserker_siege')
        if siege_dir not in sys.path: sys.path.insert(0, siege_dir)
        from siege_move_gen import MoveGen   as _SMG
        from siege_search   import Search    as _SS

        # Create a proxy Evaluator that wraps our chosen eval
        class _ProxyEval:
            def evaluate(self, chess_board) -> int:
                import chess
                fen = chess_board.fen()
                bs  = parse_fen(fen)
                raw = resolved_eval(bs)
                return int(raw)

        resolved_eval = _get_eval()
        mg = _SMG()
        sa = _SS(extend_checks_in_qsearch=True)
        proxy = _ProxyEval()

        def siege_combo_search(state: BoardState):
            board = _fen_to_chess_board(_to_fen(state))
            try:
                m, score = sa.find_best_move(board, mg, proxy,
                                             time_limit=time_limit, max_depth=max_depth)
                if m is None: return _fallback(state)
                return _resolve_uci(state, m.uci())
            except Exception:
                return _fallback(state)

        return EngineAdapter(name, siege_combo_search,
                             f'Siege α-β+Q + {eval_fn}')

    if search == 'mcts_search':
        # MCTS doesn't use a traditional eval — it uses rollouts.
        # When paired with an eval function, we use the eval as the rollout scorer:
        # instead of playing to terminal, we score the position after N random moves.
        mcts_dir = str(ENGINES_DIR / 'monte_carlo_mcts')
        if mcts_dir not in sys.path: sys.path.insert(0, mcts_dir)
        import mcts_search    as _ms_mod
        import mcts_rollout   as _mr_mod
        import mcts_move_gen  as _mmg_mod

        resolved_eval = _get_eval()
        iterations    = kwargs.get('iterations', 400)

        def mcts_combo_search(state: BoardState):
            gs = _fen_to_mcts_gamestate(_to_fen(state))

            # Replace the rollout policy with our eval-based scorer
            def eval_rollout(gs_inner, max_plies=8):
                """Short rollout then eval, instead of play-to-terminal."""
                current = gs_inner
                for _ in range(max_plies):
                    if _mr_mod.is_terminal(current):
                        return _mr_mod.game_result(current)
                    move = _mr_mod.fast_random_move(current)
                    if move is None: break
                    current = _mmg_mod.make_move(current, move)
                # Score with eval
                fen_str = _chaos_gs_to_fen(current)  # reuse FEN converter
                bs = parse_fen(fen_str)
                raw = resolved_eval(bs)
                # Normalise centipawns → [0,1]: 0=black winning, 1=white winning
                return min(1.0, max(0.0, (raw + 1500) / 3000))

            orig_rollout = _mr_mod.rollout
            _mr_mod.rollout = eval_rollout
            try:
                r = _ms_mod.mcts_search(gs, max_iter=iterations,
                                        movetime_ms=int(time_limit*1000), verbose=False)
            finally:
                _mr_mod.rollout = orig_rollout
            if r.best_move is None:
                return _fallback(state)
            return _resolve_uci(state, r.best_uci)

        return EngineAdapter(name, mcts_combo_search,
                             f'MCTS + {eval_fn} rollout scorer')

    if search == 'nn_search':
        # JS search always uses JS eval internally — we can't easily inject Python
        # eval into Node.js at runtime, so nn_search combos use the JS engine as-is
        # but are still included so the matrix is complete and comparable.
        runner  = _NeuralNNRunner()
        depth   = kwargs.get('depth', 3)
        time_ms = int(time_limit * 1000)

        def nn_combo_search(state: BoardState):
            uci = runner.get_move(_to_fen(state), depth=depth, time_ms=time_ms)
            if uci is None: return _fallback(state)
            return _resolve_uci(state, uci)

        return EngineAdapter(name, nn_combo_search,
                             f'Neural JS α-β (eval fixed to nn_eval in JS)')

    raise ValueError(
        f"Unknown search {search!r}. Options: "
        "classical_search, chaos_search, siege_search, mcts_search, nn_search"
    )


def _chaos_gs_to_fen(gs) -> str:
    """Convert a berserker_chaos / monte_carlo GameState back to FEN string."""
    FILES = 'abcdefgh'
    PIECE_TO_FEN = {
        'wP':'P','wN':'N','wB':'B','wR':'R','wQ':'Q','wK':'K',
        'bP':'p','bN':'n','bB':'b','bR':'r','bQ':'q','bK':'k',
    }
    rows = []
    for r in range(8):
        empty = 0
        row_str = ''
        for c in range(8):
            p = gs.board[r*8+c]
            if p is None:
                empty += 1
            else:
                if empty: row_str += str(empty); empty = 0
                row_str += PIECE_TO_FEN.get(p, '?')
        if empty: row_str += str(empty)
        rows.append(row_str)
    pos = '/'.join(rows)
    cas_parts = []
    if gs.castling.get('wK'): cas_parts.append('K')
    if gs.castling.get('wQ'): cas_parts.append('Q')
    if gs.castling.get('bK'): cas_parts.append('k')
    if gs.castling.get('bQ'): cas_parts.append('q')
    cas = ''.join(cas_parts) or '-'
    ep = '-'
    if gs.ep_square is not None:
        ep_r, ep_c = gs.ep_square // 8, gs.ep_square % 8
        ep = FILES[ep_c] + str(8 - ep_r)
    hm = getattr(gs, 'halfmove', 0)
    fm = getattr(gs, 'fullmove', 1)
    return f"{pos} {gs.turn} {cas} {ep} {hm} {fm}"
