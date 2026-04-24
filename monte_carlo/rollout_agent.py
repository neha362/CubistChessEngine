"""
rollout_agent.py — Rollout Agent  (replaces eval_agent in MCTS)
================================================================
In classical alpha-beta, the eval agent scores a position statically
(material + PST).  In MCTS the eval is replaced by *simulation*:
play random moves until the game ends, then return the result.

This agent owns everything related to that simulation:

  rollout(state, max_plies)  -> float   (1.0/0.5/0.0 from White's view)

Pluggable policies
──────────────────
The default policy is purely random (fastest, weakest).
Two stronger policies are included and can be swapped in:

  POLICY_RANDOM    — uniform random legal move
  POLICY_CAPTURE   — prefer captures, else random  (slight improvement)
  POLICY_MATERIAL  — pick move with highest immediate material gain

Set active policy:
  rollout_agent.ACTIVE_POLICY = rollout_agent.POLICY_CAPTURE

Standalone test
───────────────
  python rollout_agent.py --test
  python rollout_agent.py          (interactive: FEN → N rollouts → stats)

Contract consumed by mcts_agent.py:
  rollout(state, max_plies=200) -> float
"""

from __future__ import annotations
import random
import sys
import time
from typing import Optional, Callable

# ── Import move generator ────────────────────────────────────────────────────
from movegen_agent import (
    GameState, from_fen, make_move, is_terminal, game_result,
    fast_random_move, all_legal_moves, STARTPOS, sq_name
)

# ── Rollout policies ──────────────────────────────────────────────────────────
# A policy is a callable:  (state: GameState) -> Optional[tuple]
# Returns a move tuple (frm, to, promo) or None if terminal.

def _policy_random(state: GameState) -> Optional[tuple]:
    """Uniform random legal move — fastest policy."""
    return fast_random_move(state)


def _policy_capture(state: GameState) -> Optional[tuple]:
    """
    Prefer any capturing move over quiet moves.
    Ties broken randomly.  Falls back to random if no captures exist.
    """
    moves    = all_legal_moves(state)
    if not moves:
        return None
    captures = [m for m in moves if state.board[m[1]]]
    pool     = captures if captures else moves
    return random.choice(pool)


_MV = {'P':1,'N':3,'B':3,'R':5,'Q':9,'K':0}

def _policy_material(state: GameState) -> Optional[tuple]:
    """
    Pick the move that captures the most material.
    Ties broken randomly.  Falls back to random if no captures exist.
    """
    moves = all_legal_moves(state)
    if not moves:
        return None
    best_val = -1
    best     = []
    for m in moves:
        captured = state.board[m[1]]
        val      = _MV.get(captured[1], 0) if captured else 0
        if   val > best_val: best_val = val; best = [m]
        elif val == best_val: best.append(m)
    return random.choice(best)


POLICY_RANDOM   = _policy_random
POLICY_CAPTURE  = _policy_capture
POLICY_MATERIAL = _policy_material

# Active policy used by rollout() — swap this out to change behaviour
ACTIVE_POLICY: Callable = POLICY_RANDOM


# ── Core rollout ──────────────────────────────────────────────────────────────
def rollout(state: GameState, max_plies: int = 200) -> float:
    """
    Simulate one game from `state` using ACTIVE_POLICY until terminal
    or `max_plies` is exhausted.

    Returns:
      1.0  — White wins
      0.0  — Black wins
      0.5  — draw / ply cap reached (treated as draw)

    This is the hot path — called thousands of times per second.
    Keep it as tight as possible.
    """
    current = state
    for _ in range(max_plies):
        if is_terminal(current):
            return game_result(current)
        move = ACTIVE_POLICY(current)
        if move is None:
            return game_result(current)
        current = make_move(current, move)
    # Ply cap: treat as draw
    return 0.5


def rollout_many(state: GameState, n: int, max_plies: int = 200) -> dict:
    """
    Run `n` rollouts from `state`.
    Returns dict with keys: wins_white, wins_black, draws, mean, sims_per_sec
    """
    wins_w = wins_b = draws = 0
    t0 = time.time()
    for _ in range(n):
        r = rollout(state, max_plies)
        if   r == 1.0: wins_w += 1
        elif r == 0.0: wins_b += 1
        else:          draws  += 1
    elapsed = time.time() - t0
    return {
        'wins_white':   wins_w,
        'wins_black':   wins_b,
        'draws':        draws,
        'mean':         (wins_w + 0.5 * draws) / n,
        'n':            n,
        'sims_per_sec': int(n / max(elapsed, 1e-6)),
    }


# ── Self-test ─────────────────────────────────────────────────────────────────
def _run_tests():
    print("rollout_agent self-test")
    print("=" * 50)
    ok = True

    # 1. Single rollout returns valid value
    state = from_fen(STARTPOS)
    r     = rollout(state)
    valid = r in (0.0, 0.5, 1.0)
    ok   &= valid
    print(f"  rollout returns valid value ({r}):   [{'PASS' if valid else 'FAIL'}]")

    # 2. Terminal position returns immediately
    fool = from_fen('rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3')
    r2   = rollout(fool)
    term_ok = r2 == 0.0   # Black wins
    ok      &= term_ok
    print(f"  terminal (fool's mate) = 0.0:        [{'PASS' if term_ok else 'FAIL'}]")

    # 3. Throughput benchmark
    stats = rollout_many(state, n=200)
    fast_ok = stats['sims_per_sec'] > 10   # should manage >10 rollouts/sec easily
    ok      &= fast_ok
    print(f"  200 rollouts in {200/max(stats['sims_per_sec'],1e-6):.1f}s  "
          f"({stats['sims_per_sec']} sims/s):  [{'PASS' if fast_ok else 'FAIL'}]")
    print(f"    W={stats['wins_white']}  B={stats['wins_black']}  D={stats['draws']}  "
          f"mean={stats['mean']:.3f}")

    # 4. Policy swap: capture policy should not crash
    global ACTIVE_POLICY
    prev_policy  = ACTIVE_POLICY
    ACTIVE_POLICY = POLICY_CAPTURE
    rc   = rollout(from_fen(STARTPOS))
    cap_ok = rc in (0.0, 0.5, 1.0); ok &= cap_ok
    print(f"  capture policy rollout valid ({rc}): [{'PASS' if cap_ok else 'FAIL'}]")
    ACTIVE_POLICY = prev_policy

    # 5. Material policy
    ACTIVE_POLICY = POLICY_MATERIAL
    rm   = rollout(from_fen(STARTPOS))
    mat_ok = rm in (0.0, 0.5, 1.0); ok &= mat_ok
    print(f"  material policy rollout valid ({rm}): [{'PASS' if mat_ok else 'FAIL'}]")
    ACTIVE_POLICY = prev_policy

    print("=" * 50)
    print("All tests PASSED" if ok else "Some tests FAILED")
    return ok


if __name__ == '__main__':
    if '--test' in sys.argv:
        sys.exit(0 if _run_tests() else 1)

    fen = input("FEN (blank=startpos): ").strip() or STARTPOS
    n   = int(input("Number of rollouts: ").strip() or '500')
    print(f"\nRunning {n} rollouts...")
    stats = rollout_many(from_fen(fen), n)
    print(f"  White wins: {stats['wins_white']:>5}  ({100*stats['wins_white']/n:.1f}%)")
    print(f"  Black wins: {stats['wins_black']:>5}  ({100*stats['wins_black']/n:.1f}%)")
    print(f"  Draws:      {stats['draws']:>5}  ({100*stats['draws']/n:.1f}%)")
    print(f"  Mean score (White): {stats['mean']:.4f}")
    print(f"  Throughput: {stats['sims_per_sec']:,} rollouts/sec")