"""
Eval Agent — Claude Oracle Chess Engine
Material counting + piece-square tables (PST).
Can be tested standalone by scoring known positions.
Designed to be drop-in compatible with search_agent.SearchAgent(eval_fn=...).

The Claude Oracle enhancement is at the bottom: oracle_eval() calls the
Claude API with the FEN and returns a centipawn score from Claude's reasoning.
"""

import os
import json
import asyncio
import httpx
from move_gen_agent import BoardState, parse_fen

# ─── Material values (centipawns) ──────────────────────────────────────────────

MATERIAL = {
    'P': 100, 'N': 320, 'B': 330, 'R': 500, 'Q': 900, 'K': 0,
    'p':-100, 'n':-320, 'b':-330, 'r':-500, 'q':-900, 'k': 0,
}

# ─── Piece-square tables (white's perspective, row 0 = rank 8) ────────────────
# Values in centipawns. Black uses the vertically mirrored table (row 7 - r).

PST: dict[str, list[list[int]]] = {
    'P': [
        [  0,  0,  0,  0,  0,  0,  0,  0],
        [ 50, 50, 50, 50, 50, 50, 50, 50],
        [ 10, 10, 20, 30, 30, 20, 10, 10],
        [  5,  5, 10, 25, 25, 10,  5,  5],
        [  0,  0,  0, 20, 20,  0,  0,  0],
        [  5, -5,-10,  0,  0,-10, -5,  5],
        [  5, 10, 10,-20,-20, 10, 10,  5],
        [  0,  0,  0,  0,  0,  0,  0,  0],
    ],
    'N': [
        [-50,-40,-30,-30,-30,-30,-40,-50],
        [-40,-20,  0,  0,  0,  0,-20,-40],
        [-30,  0, 10, 15, 15, 10,  0,-30],
        [-30,  5, 15, 20, 20, 15,  5,-30],
        [-30,  0, 15, 20, 20, 15,  0,-30],
        [-30,  5, 10, 15, 15, 10,  5,-30],
        [-40,-20,  0,  5,  5,  0,-20,-40],
        [-50,-40,-30,-30,-30,-30,-40,-50],
    ],
    'B': [
        [-20,-10,-10,-10,-10,-10,-10,-20],
        [-10,  0,  0,  0,  0,  0,  0,-10],
        [-10,  0,  5, 10, 10,  5,  0,-10],
        [-10,  5,  5, 10, 10,  5,  5,-10],
        [-10,  0, 10, 10, 10, 10,  0,-10],
        [-10, 10, 10, 10, 10, 10, 10,-10],
        [-10,  5,  0,  0,  0,  0,  5,-10],
        [-20,-10,-10,-10,-10,-10,-10,-20],
    ],
    'R': [
        [  0,  0,  0,  0,  0,  0,  0,  0],
        [  5, 10, 10, 10, 10, 10, 10,  5],
        [ -5,  0,  0,  0,  0,  0,  0, -5],
        [ -5,  0,  0,  0,  0,  0,  0, -5],
        [ -5,  0,  0,  0,  0,  0,  0, -5],
        [ -5,  0,  0,  0,  0,  0,  0, -5],
        [ -5,  0,  0,  0,  0,  0,  0, -5],
        [  0,  0,  0,  5,  5,  0,  0,  0],
    ],
    'Q': [
        [-20,-10,-10, -5, -5,-10,-10,-20],
        [-10,  0,  0,  0,  0,  0,  0,-10],
        [-10,  0,  5,  5,  5,  5,  0,-10],
        [ -5,  0,  5,  5,  5,  5,  0, -5],
        [  0,  0,  5,  5,  5,  5,  0, -5],
        [-10,  5,  5,  5,  5,  5,  0,-10],
        [-10,  0,  5,  0,  0,  0,  0,-10],
        [-20,-10,-10, -5, -5,-10,-10,-20],
    ],
    'K': [
        [-30,-40,-40,-50,-50,-40,-40,-30],
        [-30,-40,-40,-50,-50,-40,-40,-30],
        [-30,-40,-40,-50,-50,-40,-40,-30],
        [-30,-40,-40,-50,-50,-40,-40,-30],
        [-20,-30,-30,-40,-40,-30,-30,-20],
        [-10,-20,-20,-20,-20,-20,-20,-10],
        [ 20, 20,  0,  0,  0,  0, 20, 20],
        [ 20, 30, 10,  0,  0, 10, 30, 20],
    ],
    # Endgame king — more active
    'K_end': [
        [-50,-40,-30,-20,-20,-30,-40,-50],
        [-30,-20,-10,  0,  0,-10,-20,-30],
        [-30,-10, 20, 30, 30, 20,-10,-30],
        [-30,-10, 30, 40, 40, 30,-10,-30],
        [-30,-10, 30, 40, 40, 30,-10,-30],
        [-30,-10, 20, 30, 30, 20,-10,-30],
        [-30,-30,  0,  0,  0,  0,-30,-30],
        [-50,-30,-30,-30,-30,-30,-30,-50],
    ],
}


# ─── Evaluator class ───────────────────────────────────────────────────────────

class Evaluator:
    """
    Static position evaluator. Returns centipawns from white's perspective.
    Positive = white winning, negative = black winning.

    Compatible with search_agent.SearchAgent(eval_fn=Evaluator().evaluate).
    """

    def evaluate(self, state: BoardState) -> float:
        return float(self._material_and_pst(state) + self._mobility_bonus(state))

    # ── Material + PST ────────────────────────────────────────────────────────

    def _material_and_pst(self, state: BoardState) -> int:
        score = 0
        is_endgame = self._is_endgame(state)

        for r in range(8):
            for c in range(8):
                p = state.board[r][c]
                if not p:
                    continue

                # Material
                score += MATERIAL.get(p, 0)

                # PST (white uses the table as-is; black mirrors vertically)
                ptype = p.upper()
                if ptype == 'K' and is_endgame:
                    table = PST.get('K_end')
                else:
                    table = PST.get(ptype)

                if table:
                    if p.isupper():        # white piece
                        score += table[r][c]
                    else:                  # black piece (mirrored row)
                        score -= table[7 - r][c]

        return score

    # ── Mobility ──────────────────────────────────────────────────────────────

    def _mobility_bonus(self, state: BoardState) -> int:
        """Simple mobility: +4cp per legal move available."""
        from move_gen_agent import MoveGenerator
        gen = MoveGenerator()
        w_moves = len(gen.pseudo_legal_moves(state)) if state.turn == 'w' else 0
        # flip and count black's
        flipped = state.copy()
        flipped.turn = 'b' if state.turn == 'w' else 'w'
        b_moves = len(gen.pseudo_legal_moves(flipped)) if flipped.turn == 'b' else 0
        return 4 * (w_moves - b_moves)

    # ── Endgame detection ─────────────────────────────────────────────────────

    def _is_endgame(self, state: BoardState) -> bool:
        """Simple endgame: both sides have no queens, or each side has ≤ one minor piece."""
        queens = sum(1 for row in state.board for p in row if p in ('Q','q'))
        major  = sum(1 for row in state.board for p in row if p.upper() in ('Q','R'))
        return queens == 0 or major <= 2


# ─── Claude Oracle eval ───────────────────────────────────────────────────────

ORACLE_SYSTEM = """\
You are a grandmaster-level chess evaluator.
Given a chess position in FEN notation, evaluate it and return ONLY a JSON object with this shape:
{
  "score": <integer centipawns from white's perspective, positive = white winning>,
  "reasoning": "<one sentence explanation>"
}
No markdown, no extra text. JSON only."""

async def oracle_eval_async(state: BoardState) -> float:
    """
    Call the Claude API to evaluate a position.
    Returns a centipawn score from white's perspective.
    Falls back to material count on API error.
    """
    fen = state.fen()

    payload = {
        "model": "claude-sonnet-4-20250514",
        "max_tokens": 200,
        "system": ORACLE_SYSTEM,
        "messages": [{"role": "user", "content": f"Evaluate this position: {fen}"}],
    }

    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(
            "https://api.anthropic.com/v1/messages",
            headers={"Content-Type": "application/json"},
            json=payload,
        )
        resp.raise_for_status()
        data = resp.json()

    raw = data["content"][0]["text"].strip()
    parsed = json.loads(raw)
    score = float(parsed["score"])
    print(f"  [oracle] {fen[:40]}…  score={score:+.0f}cp  {parsed['reasoning']}")
    return score


def oracle_eval(state: BoardState) -> float:
    """
    Synchronous wrapper around oracle_eval_async.
    Use this as the eval_fn in SearchAgent when you want the Claude oracle.
    """
    try:
        return asyncio.run(oracle_eval_async(state))
    except Exception as e:
        print(f"  [oracle] API error ({e}), falling back to material eval")
        ev = Evaluator()
        return ev.evaluate(state)


# ─── Standalone tests ─────────────────────────────────────────────────────────

if __name__ == '__main__':
    ev = Evaluator()

    print("=== Test 1: Starting position — should be ~0 ===")
    state = parse_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
    print(f"Score: {ev.evaluate(state):+.0f}cp  (expected ~0)")

    print("\n=== Test 2: White up a queen — should be ~+900 ===")
    # Remove black's queen
    state2 = parse_fen("rnb1kbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
    print(f"Score: {ev.evaluate(state2):+.0f}cp  (expected ~+900)")

    print("\n=== Test 3: Black up a rook — should be ~-500 ===")
    state3 = parse_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/1NBQKBNR w Kkq - 0 1")
    print(f"Score: {ev.evaluate(state3):+.0f}cp  (expected ~-500)")

    print("\n=== Test 4: Centre pawns advanced (should reward white) ===")
    centre_fen = "rnbqkbnr/pppppppp/8/8/3PP3/8/PPP2PPP/RNBQKBNR b KQkq - 0 2"
    state4 = parse_fen(centre_fen)
    print(f"Score: {ev.evaluate(state4):+.0f}cp  (expect positive — white controls centre)")

    print("\n=== Test 5: Endgame detection ===")
    endgame_fen = "8/5k2/8/8/8/8/2K5/8 w - - 0 1"
    state5 = parse_fen(endgame_fen)
    print(f"Is endgame: {ev._is_endgame(state5)}  (expected True)")
    print(f"Score: {ev.evaluate(state5):+.0f}cp")

    print("\n=== Test 6 (optional — requires API): Claude oracle eval ===")
    run_oracle = input("Run oracle eval test? (y/N): ").strip().lower() == 'y'
    if run_oracle:
        fen_test = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1"
        state6 = parse_fen(fen_test)
        score6 = oracle_eval(state6)
        print(f"Oracle score: {score6:+.0f}cp")

    print("\nAll eval tests done.")