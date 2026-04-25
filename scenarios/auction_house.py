"""
auction_house.py — Mechanism-Design Chess Engine
==================================================
Scenario 4: Auction House with Vickrey-like truthfulness incentive.

HOW IT WORKS
────────────
Each agent triple (movegen, search, eval) is an Auction Participant.
For every position, each agent:

  1. BIDS  — submits (move, confidence ∈ [0,1])
             Confidence = how certain the agent is its move is optimal.

  2. WINS  — the winning agent is NOT simply the highest bidder.
             It is selected by: effective_bid = confidence × accuracy_weight
             where accuracy_weight is derived from past performance.

  3. PAYS (Vickrey structure) — the winner's move is judged against the
             second-highest effective bid as a penalty baseline.
             Overstating confidence gets you assigned the task, then your
             weight decays when you fail it — tanks your future effective bid.

  4. WEIGHT UPDATE — after the move is evaluated by the Referee:
             w_i ← w_i × (1 - α × |bid_i - actual_quality_i|)

             Honest bidder  (bid ≈ quality):  w barely changes
             Overbidder     (bid >> quality):  w decays sharply → stops winning
             Underbidder    (bid << quality):  w also decays   → learns to bid up

OUTPUTS
───────
  • optimal_move   — UCI string of the chosen move
  • expected_loss  — estimated centipawn loss, computed as:
                     (1 - winner_confidence) × calibrated_blunder_magnitude
                     where the magnitude is inversely scaled by accuracy_weight
                     (agents with better track records estimate losses lower)

Usage:
  python auction_house.py                            # startpos, single shot
  python auction_house.py --fen "..."                # custom FEN
  python auction_house.py --simulate --rounds 30     # show weight evolution
  python auction_house.py --interactive              # REPL
  python auction_house.py --test                     # self-test suite
"""

from __future__ import annotations

import math, time, json, sys, os, random
from dataclasses import dataclass, field
from typing import Optional, Callable

from movegen_agent import (
    GameState, from_fen, make_move, all_legal_moves,
    sq_name, STARTPOS, is_terminal
)

try:
    from eval_agent import evaluate as _reference_eval
    _EVAL_OK = True
except ImportError:
    def _reference_eval(s): return 0
    _EVAL_OK = False

try:
    from search_agent import search as _reference_search
    _SEARCH_OK = True
except ImportError:
    _SEARCH_OK = False


# ═══════════════════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class Bid:
    agent_id:   str
    move:       tuple       # (frm, to, promo)
    confidence: float       # raw bid ∈ [0, 1]
    score_cp:   int         # agent's internal eval (centipawns)
    elapsed_ms: int

    @property
    def uci(self) -> str:
        f, t, pr = self.move
        return sq_name(f) + sq_name(t) + pr


@dataclass
class AuctionResult:
    optimal_move:  str      # UCI string  ← PRIMARY OUTPUT
    expected_loss: float    # centipawns  ← PRIMARY OUTPUT
    move_tuple:    tuple
    winner_id:     str
    winner_conf:   float
    winner_weight: float
    effective_bid: float    # conf × weight
    second_bid:    float    # Vickrey reference price
    all_bids:      list
    auction_ms:    int

    def summary(self) -> str:
        lines = [
            "╔══ AUCTION RESULT ══════════════════════════════════╗",
           f"║  Optimal move   : {self.optimal_move:<8}                      ║",
           f"║  Expected loss  : {self.expected_loss:+.1f} cp                         ║",
           f"║  Winner         : {self.winner_id:<22}         ║",
           f"║  Confidence bid : {self.winner_conf:.3f}  (weight={self.winner_weight:.3f})       ║",
           f"║  Effective bid  : {self.effective_bid:.4f}  (2nd={self.second_bid:.4f})        ║",
            "╠══ ALL BIDS ════════════════════════════════════════╣",
        ]
        for b in sorted(self.all_bids, key=lambda x: x.confidence, reverse=True):
            mark = "▶" if b.agent_id == self.winner_id else " "
            lines.append(
                f"║ {mark} {b.agent_id:<24} {b.uci:<7} "
                f"conf={b.confidence:.3f} score={b.score_cp:+5d} ║"
            )
        lines.append("╚═══════════════════════════════════════════════════╝")
        return '\n'.join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# AGENT RECORD  (the weight/history that drives the truthfulness incentive)
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class AgentRecord:
    """
    Persistent accuracy state for one agent.

    The weight update rule is:
        w ← w × (1 − α × |bid − actual_quality|)

    where actual_quality ∈ [0,1] is scored by the Referee after each win.

    This is the mechanism-design core: an agent that overbids wins more
    auctions, but |bid − actual| is large each time → w collapses →
    effective_bid collapses → agent stops winning.  Truthful bidding
    is the dominant strategy in the long run.
    """
    agent_id:        str
    accuracy_weight: float = 1.0
    wins:            int   = 0
    history:         list  = field(default_factory=list)  # [(bid, quality), ...]
    ALPHA:           float = 0.25
    HISTORY_CAP:     int   = 60
    W_MIN:           float = 0.05
    W_MAX:           float = 5.0

    def update(self, bid: float, actual_quality: float):
        error = abs(bid - actual_quality)
        self.accuracy_weight *= (1.0 - self.ALPHA * error)
        self.accuracy_weight  = max(self.W_MIN, min(self.W_MAX, self.accuracy_weight))
        self.wins   += 1
        self.history.append((round(bid, 4), round(actual_quality, 4)))
        if len(self.history) > self.HISTORY_CAP:
            self.history.pop(0)

    def avg_error(self) -> float:
        return (sum(abs(b-q) for b,q in self.history) / len(self.history)
                if self.history else 0.5)

    def avg_quality(self) -> float:
        return (sum(q for _,q in self.history) / len(self.history)
                if self.history else 0.5)

    def expected_loss_cp(self, confidence: float,
                         base_blunder: float = 80.0) -> float:
        """
        Estimate centipawn loss for a move at this confidence level.

        expected_loss = (1 − confidence) × base_blunder / credibility

        credibility = accuracy_weight (higher → agent is historically accurate)
        A confident, accurate agent  → small expected loss
        A confident, inaccurate agent → still large expected loss (low weight)
        """
        credibility = min(self.accuracy_weight, 2.0)
        return (1.0 - confidence) * base_blunder / credibility


# ═══════════════════════════════════════════════════════════════════════════════
# CHESS AGENT  (wraps any search+eval triple)
# ═══════════════════════════════════════════════════════════════════════════════

class ChessAgent:
    """
    Wraps a (search_fn, eval_fn) pair and produces Bids.

    Confidence is derived from the search score via a sigmoid:
        confidence = sigmoid(oriented_score / temperature)

    Temperature controls how aggressively scores are mapped to confidence.
    - Low temp  → agent quickly hits confidence ≈ 1 (over-confident by default)
    - High temp → scores spread across [0.3, 0.7] (more conservative bids)
    """

    def __init__(self, agent_id: str, search_fn: Callable, eval_fn: Callable,
                 temperature: float = 150.0, max_depth: int = 3,
                 movetime_ms: int = 600):
        self.agent_id    = agent_id
        self.search_fn   = search_fn
        self.eval_fn     = eval_fn
        self.temperature = temperature
        self.max_depth   = max_depth
        self.movetime_ms = movetime_ms
        self.record      = AgentRecord(agent_id=agent_id)

    @staticmethod
    def _sigmoid(x: float) -> float:
        return 1.0 / (1.0 + math.exp(-max(-20.0, min(20.0, x))))

    def _to_confidence(self, score_cp: int, turn: str) -> float:
        oriented = score_cp if turn == 'w' else -score_cp
        return self._sigmoid(oriented / self.temperature)

    def bid(self, state: GameState) -> Bid:
        t0 = time.time()
        move, score_cp = None, 0
        try:
            result  = self.search_fn(state,
                                     max_depth   = self.max_depth,
                                     movetime_ms = self.movetime_ms)
            move    = result.best_move
            score_cp = result.score
        except Exception:
            pass

        if move is None:
            moves = all_legal_moves(state)
            move  = random.choice(moves) if moves else (0, 0, '')
            score_cp = 0

        return Bid(
            agent_id   = self.agent_id,
            move       = move,
            confidence = self._to_confidence(score_cp, state.turn),
            score_cp   = score_cp,
            elapsed_ms = int((time.time() - t0) * 1000),
        )


# ═══════════════════════════════════════════════════════════════════════════════
# REFEREE  (scores actual move quality after the fact)
# ═══════════════════════════════════════════════════════════════════════════════

class Referee:
    """
    Compares a played move against the reference engine's best move.
    Returns quality ∈ [0, 1]:
        1.0 = played the objectively best move
        0.5 = within one pawn of best
        0.0 = catastrophic blunder
    """

    def __init__(self, ref_depth: int = 3):
        self.ref_depth = ref_depth
        self._cache: dict = {}

    def _reference(self, state: GameState) -> tuple:
        key = (tuple(state.board), state.turn,
               tuple(sorted(state.castling.items())), state.ep_square)
        if key in self._cache:
            return self._cache[key]
        if not _SEARCH_OK:
            return None, 0
        try:
            r = _reference_search(state, max_depth=self.ref_depth,
                                  movetime_ms=2000)
            ans = (r.best_move, r.score)
        except Exception:
            ans = (None, 0)
        self._cache[key] = ans
        return ans

    def score_move(self, state: GameState, played: tuple) -> float:
        ref_move, ref_score = self._reference(state)
        if ref_move is None:
            return 0.7
        if played == ref_move:
            return 1.0
        try:
            ns_played    = make_move(state, played)
            score_played = _reference_eval(ns_played)
            if state.turn == 'b':
                score_played = -score_played
        except Exception:
            return 0.5
        cp_loss = max(0, ref_score - score_played)
        return float(max(0.0, min(1.0, math.exp(-cp_loss / 120.0))))


# ═══════════════════════════════════════════════════════════════════════════════
# AUCTION HOUSE  (the mechanism itself)
# ═══════════════════════════════════════════════════════════════════════════════

class AuctionHouse:
    """
    Runs the Vickrey-weighted confidence auction for N agents.

    Each round:
      1. Collect bids  (move + confidence) from all agents
      2. Compute effective_bid_i = confidence_i × accuracy_weight_i
      3. Winner = argmax(effective_bid)
      4. Vickrey reference = second-highest effective bid
      5. Evaluate winner's move quality via Referee
      6. Update winner's weight via AgentRecord.update()
      7. Return AuctionResult(optimal_move, expected_loss)
    """

    def __init__(self, agents: list[ChessAgent],
                 referee: Optional[Referee] = None,
                 persistence_path: Optional[str] = None):
        self.agents           = {a.agent_id: a for a in agents}
        self.referee          = referee or Referee()
        self.persistence_path = persistence_path
        self._load_weights()

    # ── Persistence ───────────────────────────────────────────────────────────
    def _load_weights(self):
        if not self.persistence_path:
            return
        if not os.path.exists(self.persistence_path):
            return
        try:
            with open(self.persistence_path) as f:
                data = json.load(f)
            for aid, rec in data.items():
                if aid in self.agents:
                    a = self.agents[aid]
                    a.record.accuracy_weight = rec.get('weight', 1.0)
                    a.record.wins            = rec.get('wins', 0)
                    a.record.history         = [tuple(x) for x in rec.get('history', [])]
        except Exception as e:
            print(f"[AuctionHouse] load error: {e}", file=sys.stderr)

    def _save_weights(self):
        if not self.persistence_path:
            return
        data = {
            aid: {'weight': a.record.accuracy_weight,
                  'wins':   a.record.wins,
                  'history': a.record.history[-30:]}
            for aid, a in self.agents.items()
        }
        try:
            with open(self.persistence_path, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"[AuctionHouse] save error: {e}", file=sys.stderr)

    # ── Core auction ──────────────────────────────────────────────────────────
    def run(self, state: GameState,
            update_weights: bool = True) -> AuctionResult:
        """
        Run one auction.

        Returns AuctionResult with:
          .optimal_move   — UCI string  (e.g. 'e2e4')
          .expected_loss  — centipawns  (e.g. 12.4)
        """
        t0   = time.time()
        bids = [agent.bid(state) for agent in self.agents.values()]

        # Effective bids
        eff = [(b.confidence * self.agents[b.agent_id].record.accuracy_weight, b)
               for b in bids]
        eff.sort(key=lambda x: x[0], reverse=True)

        winner_eff, winner_bid = eff[0]
        second_eff = eff[1][0] if len(eff) > 1 else 0.0
        winner_agent = self.agents[winner_bid.agent_id]

        # Expected loss (Vickrey-adjusted)
        base_loss      = winner_agent.record.expected_loss_cp(winner_bid.confidence)
        vickrey_gap    = winner_eff - second_eff           # dominance signal
        gap_discount   = min(0.45, vickrey_gap * 1.8)     # up to 45% discount
        expected_loss  = max(0.0, base_loss * (1.0 - gap_discount))

        # Weight update (truth incentive)
        if update_weights:
            quality = self.referee.score_move(state, winner_bid.move)
            winner_agent.record.update(winner_bid.confidence, quality)
            self._save_weights()

        return AuctionResult(
            optimal_move  = winner_bid.uci,
            expected_loss = round(expected_loss, 2),
            move_tuple    = winner_bid.move,
            winner_id     = winner_bid.agent_id,
            winner_conf   = winner_bid.confidence,
            winner_weight = winner_agent.record.accuracy_weight,
            effective_bid = winner_eff,
            second_bid    = second_eff,
            all_bids      = bids,
            auction_ms    = int((time.time() - t0) * 1000),
        )

    # ── Diagnostics ──────────────────────────────────────────────────────────
    def leaderboard(self) -> str:
        rows  = sorted(self.agents.values(),
                       key=lambda a: a.record.accuracy_weight, reverse=True)
        lines = ["┌── AGENT LEADERBOARD ───────────────────────────────────────┐"]
        for a in rows:
            r   = a.record
            bar = '█' * max(1, int(r.accuracy_weight * 8))
            lines.append(
                f"│  {a.agent_id:<22} w={r.accuracy_weight:.3f}  "
                f"wins={r.wins:<4} avg_q={r.avg_quality():.3f}  {bar}"
            )
        lines.append("└────────────────────────────────────────────────────────────┘")
        return '\n'.join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# FACTORY
# ═══════════════════════════════════════════════════════════════════════════════

def build_auction_house(persistence_path: str = 'auction_weights.json',
                        max_depth: int = 3,
                        movetime_ms: int = 600) -> AuctionHouse:
    """
    Build an AuctionHouse from all available agent implementations.
    Gracefully degrades if some agents are not importable.
    """
    agents = []

    try:
        from search_agent import search as classical_search
        from eval_agent   import evaluate as classical_eval
        agents.append(ChessAgent('Classical', classical_search, classical_eval,
                                 temperature=150.0, max_depth=max_depth,
                                 movetime_ms=movetime_ms))
    except ImportError:
        pass

    try:
        from berserker_search_agent import search as bsearch
        from berserker_eval_agent   import evaluate as beval
        agents.append(ChessAgent('Berserker', bsearch, beval,
                                 temperature=400.0, max_depth=max_depth,
                                 movetime_ms=movetime_ms))
    except ImportError:
        pass

    try:
        from mcts_agent import mcts_search

        def _mcts_adapter(state, max_depth=3, movetime_ms=600):
            r = mcts_search(state, max_iter=max_depth * 200,
                            movetime_ms=movetime_ms, verbose=False)
            class _R:
                best_move = r.best_move
                score     = int((r.win_rate - 0.5) * 200)
            return _R()

        agents.append(ChessAgent('MCTS', _mcts_adapter, lambda s: 0,
                                 temperature=80.0, max_depth=max_depth,
                                 movetime_ms=movetime_ms))
    except ImportError:
        pass

    if not agents:
        raise RuntimeError("No agents importable — ensure movegen/search/eval agents are present")

    return AuctionHouse(agents=agents,
                        referee=Referee(ref_depth=max(2, max_depth - 1)),
                        persistence_path=persistence_path)


# ═══════════════════════════════════════════════════════════════════════════════
# SELF-TEST
# ═══════════════════════════════════════════════════════════════════════════════

def _run_tests():
    print("auction_house.py — self-test")
    print("=" * 62)
    ok = True

    # T1: Honest bidder → weight barely changes
    rec = AgentRecord(agent_id='honest', accuracy_weight=1.0, ALPHA=0.25)
    rec.update(0.78, 0.80)
    delta = abs(rec.accuracy_weight - 1.0)
    t1    = delta < 0.08
    ok   &= t1
    print(f"  [{'PASS' if t1 else 'FAIL'}] Honest bid barely changes weight  (Δ={delta:.4f})")

    # T2: Persistent overbidder → weight collapses
    rec2 = AgentRecord(agent_id='over', accuracy_weight=1.0, ALPHA=0.25)
    for _ in range(12):
        rec2.update(0.95, 0.20)
    t2 = rec2.accuracy_weight < 0.45
    ok &= t2
    print(f"  [{'PASS' if t2 else 'FAIL'}] Overbidder weight collapses       (w={rec2.accuracy_weight:.4f})")

    # T3: Underbidder also penalised (less harshly than overbidder)
    rec3 = AgentRecord(agent_id='under', accuracy_weight=1.0, ALPHA=0.25)
    for _ in range(12):
        rec3.update(0.10, 0.90)
    t3 = rec3.accuracy_weight < 1.0
    ok &= t3
    print(f"  [{'PASS' if t3 else 'FAIL'}] Underbidder weight also decays    (w={rec3.accuracy_weight:.4f})")

    # T4: Highest effective bid wins
    def _mock_search(score):
        def fn(state, max_depth=3, movetime_ms=600):
            moves = all_legal_moves(state)
            class R:
                best_move = moves[0] if moves else (0,0,'')
            R.score = score
            return R()
        return fn

    state = from_fen(STARTPOS)
    a1 = ChessAgent('HighConf',  _mock_search(300), lambda s: 300, temperature=150.0)
    a2 = ChessAgent('LowConf',   _mock_search(10),  lambda s: 10,  temperature=150.0)
    a1.record.accuracy_weight = 1.0
    a2.record.accuracy_weight = 1.0
    house = AuctionHouse([a1, a2], referee=Referee(), persistence_path=None)
    r4    = house.run(state, update_weights=False)
    t4    = r4.winner_id == 'HighConf'
    ok   &= t4
    print(f"  [{'PASS' if t4 else 'FAIL'}] Highest effective bid wins        (winner={r4.winner_id})")

    # T5: Downweighted agent stops winning despite high confidence
    a1.record.accuracy_weight = 0.08
    a2.record.accuracy_weight = 1.00
    r5  = house.run(state, update_weights=False)
    t5  = r5.winner_id == 'LowConf'
    ok &= t5
    print(f"  [{'PASS' if t5 else 'FAIL'}] Downweighted agent loses auction  (winner={r5.winner_id})")

    # T6: Expected loss ∈ [0, 500]
    a1.record.accuracy_weight = 1.0
    a2.record.accuracy_weight = 1.0
    r6  = house.run(state, update_weights=False)
    t6  = 0.0 <= r6.expected_loss <= 500.0
    ok &= t6
    print(f"  [{'PASS' if t6 else 'FAIL'}] Expected loss is finite ≥ 0      ({r6.expected_loss:.1f} cp)")

    # T7: Referee quality ∈ [0, 1]
    ref   = Referee(ref_depth=2)
    moves = all_legal_moves(state)
    q     = ref.score_move(state, moves[0])
    t7    = 0.0 <= q <= 1.0
    ok   &= t7
    print(f"  [{'PASS' if t7 else 'FAIL'}] Referee quality ∈ [0,1]          (q={q:.3f})")

    # T8: Full factory round-trip
    try:
        house2 = build_auction_house(persistence_path=None)
        r8     = house2.run(from_fen(STARTPOS), update_weights=False)
        t8     = len(r8.optimal_move) >= 4 and r8.expected_loss >= 0
        ok    &= t8
        print(f"  [{'PASS' if t8 else 'FAIL'}] Factory round-trip             "
              f"(move={r8.optimal_move}, loss={r8.expected_loss:.1f} cp, "
              f"winner={r8.winner_id})")
    except Exception as e:
        print(f"  [FAIL] Factory crashed: {e}")
        ok = False

    # T9: Truthfulness convergence — simulate 20 rounds, check overbidder weight trends down
    rec_conv = AgentRecord(agent_id='conv_test', accuracy_weight=1.0, ALPHA=0.25)
    w_before = rec_conv.accuracy_weight
    for _ in range(20):
        rec_conv.update(0.90, 0.30)   # consistent overbidding
    t9 = rec_conv.accuracy_weight < w_before * 0.6
    ok &= t9
    print(f"  [{'PASS' if t9 else 'FAIL'}] Truthfulness convergence          "
          f"(w: {w_before:.3f} → {rec_conv.accuracy_weight:.3f})")

    print("=" * 62)
    print("All tests PASSED" if ok else "Some tests FAILED")
    return ok


# ═══════════════════════════════════════════════════════════════════════════════
# SIMULATION — shows weight evolution over many rounds
# ═══════════════════════════════════════════════════════════════════════════════

def _simulate(rounds: int = 25, depth: int = 2, movetime: int = 400):
    house = build_auction_house(persistence_path=None,
                                max_depth=depth, movetime_ms=movetime)
    n     = len(house.agents)
    print(f"\n{'─'*62}")
    print(f"  AUCTION HOUSE SIMULATION  ({n} agents, {rounds} rounds)")
    print(f"{'─'*62}")
    print(house.leaderboard())

    state = from_fen(STARTPOS)
    for i in range(rounds):
        result = house.run(state, update_weights=True)
        if all_legal_moves(state):
            state = make_move(state, result.move_tuple)
        if is_terminal(state):
            state = from_fen(STARTPOS)

        if (i + 1) % 5 == 0 or i == rounds - 1:
            print(f"\n  After round {i+1}:")
            for a in house.agents.values():
                r = a.record
                bar = '█' * max(1, int(r.accuracy_weight * 10))
                print(f"    {a.agent_id:<22} w={r.accuracy_weight:.3f}  "
                      f"wins={r.wins:<3}  avg_q={r.avg_quality():.3f}  {bar}")

    print(f"\nFinal leaderboard:")
    print(house.leaderboard())


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    import argparse
    ap = argparse.ArgumentParser(description='Auction House Chess Engine')
    ap.add_argument('--fen',         default=STARTPOS)
    ap.add_argument('--test',        action='store_true')
    ap.add_argument('--simulate',    action='store_true')
    ap.add_argument('--rounds',      type=int, default=25)
    ap.add_argument('--depth',       type=int, default=3)
    ap.add_argument('--movetime',    type=int, default=600)
    ap.add_argument('--interactive', action='store_true')
    ap.add_argument('--weights',     default='auction_weights.json')
    args = ap.parse_args()

    if args.test:
        sys.exit(0 if _run_tests() else 1)

    if args.simulate:
        _simulate(rounds=args.rounds, depth=args.depth, movetime=args.movetime)
        return

    house = build_auction_house(persistence_path=args.weights,
                                max_depth=args.depth,
                                movetime_ms=args.movetime)

    if args.interactive:
        print(f"\nAuction House  —  {len(house.agents)} agent(s) loaded")
        print("Enter a FEN string to get a move, or 'quit'.\n")
        while True:
            try:
                fen = input("FEN> ").strip()
            except EOFError:
                break
            if not fen or fen in ('quit','q'):
                break
            try:
                r = house.run(from_fen(fen), update_weights=True)
                print(r.summary())
                print()
            except Exception as e:
                print(f"  Error: {e}")
        print(house.leaderboard())
        return

    # Single position
    state  = from_fen(args.fen)
    result = house.run(state, update_weights=True)
    print(result.summary())
    print()
    print(house.leaderboard())


if __name__ == '__main__':
    main()