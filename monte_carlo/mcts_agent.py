"""
mcts_agent.py — Monte Carlo Tree Search Agent  (replaces search_agent)
========================================================================
Implements the four MCTS phases for every iteration:

  1. SELECTION   — walk the tree using UCB1 until a node with
                   unexpanded children (or terminal) is reached
  2. EXPANSION   — add one new child node for an untried move
  3. SIMULATION  — run a rollout from the new node (via rollout_agent)
  4. BACKPROP    — update visit counts and win totals up to the root

Key features
────────────
  • UCB1 exploration formula with tunable constant C (default √2)
  • Configurable simulation budget: time-based or fixed iteration count
  • Move selection: most-visited child (robust), not highest win rate
  • UCI-style info lines during search
  • Fully standalone test suite

Public contract (consumed by UCI layer):
  mcts_search(state, budget) -> MCTSResult
  MCTSResult.best_move : (frm, to, promo)
  MCTSResult.best_uci  : str  e.g. 'e2e4'

Usage
─────
  python mcts_agent.py --test
  python mcts_agent.py           (interactive FEN → best move)
"""

from __future__ import annotations
import math
import time
import random
import sys
from dataclasses import dataclass, field
from typing import Optional

from movegen_agent import (
    GameState, from_fen, make_move, all_legal_moves,
    is_terminal, game_result, sq_name, STARTPOS
)
from rollout_agent import rollout

# ── UCB1 constant ─────────────────────────────────────────────────────────────
UCB1_C = math.sqrt(2)   # exploration weight — increase to explore more

# ── Tree node ─────────────────────────────────────────────────────────────────
class MCTSNode:
    """
    A single node in the MCTS tree.

    Attributes
    ──────────
    state       GameState at this node
    move        Move that led here from parent (None for root)
    parent      Parent node (None for root)
    children    Expanded child nodes
    untried     Moves not yet expanded into children
    visits      Number of times this node has been visited
    wins        Accumulated result from White's perspective
                (1.0 per White win, 0.5 per draw, 0.0 per Black win)
    """
    __slots__ = ('state','move','parent','children','untried','visits','wins')

    def __init__(self, state: GameState,
                 move: Optional[tuple] = None,
                 parent: Optional['MCTSNode'] = None):
        self.state    = state
        self.move     = move
        self.parent   = parent
        self.children: list['MCTSNode'] = []
        self.untried: list[tuple] = all_legal_moves(state)
        random.shuffle(self.untried)   # randomise expansion order
        self.visits: int   = 0
        self.wins:   float = 0.0

    # ── UCB1 score ────────────────────────────────────────────────────────────
    def ucb1(self, parent_visits: int, c: float = UCB1_C) -> float:
        """
        Upper Confidence Bound for Trees:
          UCB1 = exploitation + exploration
               = (wins/visits) + C * sqrt(ln(parent_visits) / visits)

        The exploitation term is always from the CURRENT node's perspective
        (i.e. the side that chose to move here), so we flip wins for Black.
        """
        if self.visits == 0:
            return float('inf')
        # Determine whose turn it was when this move was made
        # (the parent's turn, because the move transitions to this node)
        if self.parent and self.parent.state.turn == 'b':
            # Black moved here: black wants to minimise white's score
            exploitation = 1.0 - (self.wins / self.visits)
        else:
            exploitation = self.wins / self.visits
        exploration  = c * math.sqrt(math.log(parent_visits) / self.visits)
        return exploitation + exploration

    def is_fully_expanded(self) -> bool:
        return len(self.untried) == 0

    def is_terminal(self) -> bool:
        return is_terminal(self.state)

    def best_child(self, c: float = UCB1_C) -> 'MCTSNode':
        """Select child with highest UCB1 score."""
        return max(self.children,
                   key=lambda ch: ch.ucb1(self.visits, c))

    def most_visited_child(self) -> 'MCTSNode':
        """Final move selection: pick most-visited child (robust)."""
        return max(self.children, key=lambda ch: ch.visits)

    def win_rate(self) -> float:
        return self.wins / self.visits if self.visits else 0.0

    def __repr__(self) -> str:
        mv = (sq_name(self.move[0]) + sq_name(self.move[1]) + self.move[2]
              if self.move else 'root')
        return (f"MCTSNode(move={mv}, visits={self.visits}, "
                f"wins={self.wins:.1f}, wr={self.win_rate():.3f})")

# ── Four MCTS phases ──────────────────────────────────────────────────────────

def _select(node: MCTSNode) -> MCTSNode:
    """
    PHASE 1 — SELECTION
    Walk down the tree using UCB1 until we reach:
      (a) a node with untried moves, OR
      (b) a terminal node
    """
    while not node.is_terminal() and node.is_fully_expanded():
        node = node.best_child()
    return node


def _expand(node: MCTSNode) -> MCTSNode:
    """
    PHASE 2 — EXPANSION
    Pick one untried move, create a child node for it, return the child.
    If no untried moves (terminal), return the node itself.
    """
    if not node.untried:
        return node
    move    = node.untried.pop()
    ns      = make_move(node.state, move)
    child   = MCTSNode(state=ns, move=move, parent=node)
    node.children.append(child)
    return child


def _simulate(node: MCTSNode) -> float:
    """
    PHASE 3 — SIMULATION (rollout)
    Run a random playout from node's state to terminal.
    Returns result from White's perspective (1.0/0.5/0.0).
    Delegates entirely to rollout_agent.
    """
    if node.is_terminal():
        return game_result(node.state)
    return rollout(node.state)


def _backpropagate(node: MCTSNode, result: float) -> None:
    """
    PHASE 4 — BACKPROPAGATION
    Walk from `node` up to root, incrementing visits and adding result.
    `result` is always from White's perspective throughout the path.
    """
    current = node
    while current is not None:
        current.visits += 1
        current.wins   += result
        current = current.parent


# ── Search result ─────────────────────────────────────────────────────────────
@dataclass
class MCTSResult:
    best_move:   Optional[tuple]
    iterations:  int
    time_ms:     int
    root_visits: int
    win_rate:    float        # from side-to-move's perspective
    top_moves:   list = field(default_factory=list)   # [(uci, visits, wr), ...]

    @property
    def best_uci(self) -> str:
        if not self.best_move: return '0000'
        f, t, pr = self.best_move
        return sq_name(f) + sq_name(t) + pr

    def info_str(self) -> str:
        return (f"info nodes {self.root_visits} "
                f"time {self.time_ms} "
                f"score cp {int((self.win_rate - 0.5) * 200)} "
                f"pv {self.best_uci}")


# ── Main search entry point ───────────────────────────────────────────────────
def mcts_search(state:      GameState,
                max_iter:   int           = 1000,
                movetime_ms: Optional[int] = None,
                c:          float         = UCB1_C,
                verbose:    bool          = True) -> MCTSResult:
    """
    Run MCTS from `state`.

    Parameters
    ──────────
    state        : position to search from
    max_iter     : maximum number of simulations (ignored if movetime_ms set)
    movetime_ms  : think for this many milliseconds (overrides max_iter)
    c            : UCB1 exploration constant
    verbose      : print UCI info lines during search

    Returns MCTSResult with best_move, statistics, and top move ranking.
    """
    if is_terminal(state):
        return MCTSResult(None, 0, 0, 0, 0.5)

    root  = MCTSNode(state=state)
    start = time.time()
    iters = 0

    # Budget: time-based or iteration-based
    def _over_budget() -> bool:
        if movetime_ms:
            return (time.time() - start) * 1000 >= movetime_ms
        return iters >= max_iter

    while not _over_budget():
        # ── 1. Select ────────────────────────────────────────────────────────
        node = _select(root)

        # ── 2. Expand ────────────────────────────────────────────────────────
        if not node.is_terminal():
            node = _expand(node)

        # ── 3. Simulate ──────────────────────────────────────────────────────
        result = _simulate(node)

        # ── 4. Backpropagate ─────────────────────────────────────────────────
        _backpropagate(node, result)

        iters += 1

        # Periodic UCI info (every 100 iters)
        if verbose and iters % 100 == 0 and root.children:
            best = root.most_visited_child()
            wr   = (best.wins/best.visits) if state.turn == 'w' \
                   else 1.0-(best.wins/best.visits)
            elapsed = int((time.time()-start)*1000)
            print(f"info nodes {root.visits} time {elapsed} "
                  f"score cp {int((wr-0.5)*200)} "
                  f"pv {sq_name(best.move[0])+sq_name(best.move[1])+best.move[2]}",
                  flush=True)

    # ── Choose best move: most-visited child ─────────────────────────────────
    if not root.children:
        # Never expanded (root is near-terminal or no time): pick random legal
        moves = all_legal_moves(state)
        best_move = random.choice(moves) if moves else None
        return MCTSResult(best_move, iters, int((time.time()-start)*1000),
                          root.visits, 0.5)

    best_child = root.most_visited_child()
    best_move  = best_child.move

    # Win rate from side-to-move's perspective
    wr = (best_child.wins / best_child.visits) if state.turn == 'w' \
         else 1.0 - (best_child.wins / best_child.visits)

    # Top-5 moves for display
    ranked = sorted(root.children, key=lambda n: n.visits, reverse=True)
    top_moves = []
    for ch in ranked[:5]:
        ch_wr = (ch.wins/ch.visits) if state.turn == 'w' \
                else 1.0-(ch.wins/ch.visits)
        top_moves.append((sq_name(ch.move[0])+sq_name(ch.move[1])+ch.move[2],
                          ch.visits, ch_wr))

    elapsed_ms = int((time.time()-start)*1000)
    return MCTSResult(best_move, iters, elapsed_ms,
                      root.visits, wr, top_moves)


# ── Self-test ─────────────────────────────────────────────────────────────────
def _run_tests():
    print("mcts_agent self-test")
    print("=" * 52)
    ok = True

    # 1. Returns a valid move from startpos
    state  = from_fen(STARTPOS)
    result = mcts_search(state, max_iter=200, verbose=False)
    valid  = result.best_move is not None
    ok    &= valid
    print(f"  startpos returns move ({result.best_uci}): "
          f"[{'PASS' if valid else 'FAIL'}]")

    # 2. Win rate is in [0, 1]
    wr_ok = 0.0 <= result.win_rate <= 1.0; ok &= wr_ok
    print(f"  win rate in [0,1] ({result.win_rate:.3f}):     "
          f"[{'PASS' if wr_ok else 'FAIL'}]")

    # 3. Mate-in-1: should find Re8# with enough iterations
    # White rook on e1, Black king on g8 with f7/g7/h7 pawns blocking escape
    mate1 = from_fen('6k1/5ppp/8/8/8/8/5PPP/4R1K1 w - - 0 1')
    r1    = mcts_search(mate1, max_iter=500, verbose=False)
    mate_ok = r1.best_uci == 'e1e8'; ok &= mate_ok
    print(f"  mate-in-1 Re8# ({r1.best_uci}):         "
          f"[{'PASS' if mate_ok else 'FAIL (may need more iters)'}]")

    # 4. Terminal position handled gracefully
    fool   = from_fen('rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3')
    r_term = mcts_search(fool, max_iter=10, verbose=False)
    term_ok = r_term.best_move is None; ok &= term_ok
    print(f"  terminal pos returns None:        "
          f"[{'PASS' if term_ok else 'FAIL'}]")

    # 5. Time-based budget
    t0   = time.time()
    mcts_search(state, movetime_ms=300, verbose=False)
    dt   = (time.time()-t0)*1000
    tb_ok = dt < 600; ok &= tb_ok   # generous upper bound
    print(f"  movetime=300ms ran in {dt:.0f}ms:   "
          f"[{'PASS' if tb_ok else 'FAIL'}]")

    # 6. UCB1 formula correctness
    root  = MCTSNode(state=state)
    root.visits = 10
    child = MCTSNode(state=make_move(state, all_legal_moves(state)[0]),
                     parent=root)
    child.visits = 5; child.wins = 3.0
    expected_ucb = (3/5) + UCB1_C * math.sqrt(math.log(10)/5)
    actual_ucb   = child.ucb1(10)
    ucb_ok = abs(actual_ucb - expected_ucb) < 1e-9; ok &= ucb_ok
    print(f"  UCB1 formula ({actual_ucb:.6f} ≈ {expected_ucb:.6f}): "
          f"[{'PASS' if ucb_ok else 'FAIL'}]")

    print("=" * 52)
    print("All tests PASSED" if ok else "Some tests FAILED")
    return ok


if __name__ == '__main__':
    if '--test' in sys.argv:
        sys.exit(0 if _run_tests() else 1)

    fen  = input("FEN (blank=startpos): ").strip() or STARTPOS
    iters = int(input("Iterations (blank=1000): ").strip() or '1000')
    state = from_fen(fen)
    print(f"\nRunning {iters} MCTS iterations...\n")
    result = mcts_search(state, max_iter=iters, verbose=True)
    print(f"\nBest move : {result.best_uci}")
    print(f"Win rate  : {result.win_rate:.3f}  ({result.iterations} iters, "
          f"{result.time_ms}ms)")
    print("\nTop moves:")
    for uci, visits, wr in result.top_moves:
        bar = '█' * int(wr * 20)
        print(f"  {uci:6s}  visits={visits:>5}  wr={wr:.3f}  {bar}")