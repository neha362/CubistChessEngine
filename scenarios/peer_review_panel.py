"""
peer_review_panel.py - Condorcet Peer Review Chess Engine
=========================================================
Scenario 5: Peer review panel (Condorcet jury theorem).

The panel chooses ENGINE COMPONENTS by majority vote. Three independent reviewers inspect the position and vote on:

  1. move ordering policy
  2. transposition-table policy
  3. quiescence-search policy

The judge then synthesizes one engine from the winning components and searches
the position with that assembled design.

file structure:
  - dataclasses for panel records/results
  - reviewer wrappers
  - judge/panel orchestrator
  - factory
  - self-tests
  - simulation
  - CLI
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Callable, Optional

import chess

from cto_control_engine.eval import EvalAgent

MATE_SCORE = 30000
INFINITY = 1_000_000


# =============================================================================
# DATA STRUCTURES
# =============================================================================


@dataclass
class ComponentVote:
    reviewer_id: str
    component: str
    choice: str
    confidence: float
    rationale: str


@dataclass
class PanelDecision:
    move_ordering: str
    tt_policy: str
    quiescence: str
    vote_breakdown: dict[str, list[ComponentVote]]


@dataclass
class PanelResult:
    optimal_move: str
    move_tuple: chess.Move
    score_cp: int
    panel_ms: int
    decision: PanelDecision
    expected_loss: float

    def summary(self) -> str:
        lines = [
            "Peer Review Panel Result",
            f"  optimal_move  : {self.optimal_move}",
            f"  score_cp      : {self.score_cp:+d}",
            f"  expected_loss : {self.expected_loss:.1f} cp",
            f"  move_ordering : {self.decision.move_ordering}",
            f"  tt_policy     : {self.decision.tt_policy}",
            f"  quiescence    : {self.decision.quiescence}",
        ]
        return "\n".join(lines)


# =============================================================================
# REVIEWER RECORD
# =============================================================================


@dataclass
class ReviewerRecord:
    reviewer_id: str
    component_wins: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    history: list[dict] = field(default_factory=list)
    HISTORY_CAP: int = 80

    def update(self, component: str, choice: str, accepted: bool) -> None:
        if accepted:
            self.component_wins[component] += 1
        self.history.append(
            {"component": component, "choice": choice, "accepted": bool(accepted)}
        )
        if len(self.history) > self.HISTORY_CAP:
            self.history.pop(0)


# =============================================================================
# POSITION FEATURES
# =============================================================================


def _capture_count(board: chess.Board) -> int:
    return sum(1 for move in board.legal_moves if board.is_capture(move))


def _checking_count(board: chess.Board) -> int:
    return sum(1 for move in board.legal_moves if board.gives_check(move))


def _non_pawn_material(board: chess.Board, color: chess.Color) -> int:
    values = {
        chess.KNIGHT: 3,
        chess.BISHOP: 3,
        chess.ROOK: 5,
        chess.QUEEN: 9,
    }
    total = 0
    for piece_type, value in values.items():
        total += len(board.pieces(piece_type, color)) * value
    return total


def _phase(board: chess.Board) -> str:
    total = _non_pawn_material(board, chess.WHITE) + _non_pawn_material(board, chess.BLACK)
    if total <= 12:
        return "endgame"
    if total <= 24:
        return "middlegame"
    return "opening"


def _tactical_pressure(board: chess.Board) -> float:
    captures = _capture_count(board)
    checks = _checking_count(board)
    pressure = captures * 0.08 + checks * 0.18
    return max(0.0, min(1.0, pressure))


# =============================================================================
# REVIEWERS
# =============================================================================


class PanelReviewer:
    """Base reviewer with persistent record and simple interface."""

    def __init__(self, reviewer_id: str) -> None:
        self.reviewer_id = reviewer_id
        self.record = ReviewerRecord(reviewer_id=reviewer_id)

    def review(self, board: chess.Board) -> list[ComponentVote]:
        return [
            self.review_move_ordering(board),
            self.review_tt_policy(board),
            self.review_quiescence(board),
        ]

    def review_move_ordering(self, board: chess.Board) -> ComponentVote:
        raise NotImplementedError

    def review_tt_policy(self, board: chess.Board) -> ComponentVote:
        raise NotImplementedError

    def review_quiescence(self, board: chess.Board) -> ComponentVote:
        raise NotImplementedError


class ClassicalReviewer(PanelReviewer):
    """Prefers stable, low-drama search behavior."""

    def review_move_ordering(self, board: chess.Board) -> ComponentVote:
        pressure = _tactical_pressure(board)
        choice = "balanced" if pressure < 0.55 else "capture_first"
        return ComponentVote(
            self.reviewer_id,
            "move_ordering",
            choice,
            0.72,
            "Favor reliable cutoffs with modest tactical bias.",
        )

    def review_tt_policy(self, board: chess.Board) -> ComponentVote:
        return ComponentVote(
            self.reviewer_id,
            "tt_policy",
            "zobrist_depth",
            0.80,
            "Prefer bounded TT entries keyed by Zobrist hash.",
        )

    def review_quiescence(self, board: chess.Board) -> ComponentVote:
        return ComponentVote(
            self.reviewer_id,
            "quiescence",
            "captures_only",
            0.78,
            "Use orthodox quiescence through captures only.",
        )


class TacticalReviewer(PanelReviewer):
    """Pushes the panel toward sharper tactics when the board is volatile."""

    def review_move_ordering(self, board: chess.Board) -> ComponentVote:
        pressure = _tactical_pressure(board)
        choice = "aggressive" if pressure >= 0.35 else "balanced"
        return ComponentVote(
            self.reviewer_id,
            "move_ordering",
            choice,
            0.75,
            "Checks and forcing moves should rise to the front in sharp positions.",
        )

    def review_tt_policy(self, board: chess.Board) -> ComponentVote:
        choice = "zobrist_depth" if _phase(board) != "opening" else "fen_replace"
        return ComponentVote(
            self.reviewer_id,
            "tt_policy",
            choice,
            0.66,
            "Depth-aware TT in heavy search, simpler storage in early play.",
        )

    def review_quiescence(self, board: chess.Board) -> ComponentVote:
        pressure = _tactical_pressure(board)
        choice = "captures_checks" if pressure >= 0.25 else "captures_only"
        return ComponentVote(
            self.reviewer_id,
            "quiescence",
            choice,
            0.82,
            "Extend through checks when tactical momentum is real.",
        )


class StructuralReviewer(PanelReviewer):
    """Optimizes for robustness and maintainability."""

    def review_move_ordering(self, board: chess.Board) -> ComponentVote:
        phase = _phase(board)
        choice = "balanced" if phase != "opening" else "capture_first"
        return ComponentVote(
            self.reviewer_id,
            "move_ordering",
            choice,
            0.70,
            "Prefer predictable move ordering with good capture handling.",
        )

    def review_tt_policy(self, board: chess.Board) -> ComponentVote:
        return ComponentVote(
            self.reviewer_id,
            "tt_policy",
            "zobrist_depth",
            0.84,
            "Stable hashing plus bound flags is the safest reviewable choice.",
        )

    def review_quiescence(self, board: chess.Board) -> ComponentVote:
        phase = _phase(board)
        choice = "captures_only" if phase == "endgame" else "captures_checks"
        return ComponentVote(
            self.reviewer_id,
            "quiescence",
            choice,
            0.74,
            "Allow check extensions outside the driest endgames.",
        )


# =============================================================================
# SYNTHESIZED SEARCH ENGINE
# =============================================================================


@dataclass
class TTEntry:
    depth: int
    score: int
    flag: int
    best_move: Optional[chess.Move]


class Zobrist:
    """Small deterministic Zobrist hasher for TT keys."""

    def __init__(self, seed: int = 0xC055EC5) -> None:
        import random

        rng = random.Random(seed)
        self._piece = [[rng.getrandbits(64) for _ in range(12)] for _ in range(64)]
        self._side = rng.getrandbits(64)
        self._castle = [rng.getrandbits(64) for _ in range(4)]
        self._ep_file = [rng.getrandbits(64) for _ in range(8)]

    def hash_board(self, board: chess.Board) -> int:
        index_map = {
            (chess.WHITE, chess.PAWN): 0,
            (chess.WHITE, chess.KNIGHT): 1,
            (chess.WHITE, chess.BISHOP): 2,
            (chess.WHITE, chess.ROOK): 3,
            (chess.WHITE, chess.QUEEN): 4,
            (chess.WHITE, chess.KING): 5,
            (chess.BLACK, chess.PAWN): 6,
            (chess.BLACK, chess.KNIGHT): 7,
            (chess.BLACK, chess.BISHOP): 8,
            (chess.BLACK, chess.ROOK): 9,
            (chess.BLACK, chess.QUEEN): 10,
            (chess.BLACK, chess.KING): 11,
        }
        h = 0
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece is not None:
                h ^= self._piece[square][index_map[(piece.color, piece.piece_type)]]
        if board.turn == chess.BLACK:
            h ^= self._side
        if board.has_kingside_castling_rights(chess.WHITE):
            h ^= self._castle[0]
        if board.has_queenside_castling_rights(chess.WHITE):
            h ^= self._castle[1]
        if board.has_kingside_castling_rights(chess.BLACK):
            h ^= self._castle[2]
        if board.has_queenside_castling_rights(chess.BLACK):
            h ^= self._castle[3]
        if board.ep_square is not None:
            h ^= self._ep_file[chess.square_file(board.ep_square)]
        return h


class SynthesizedSearchEngine:
    """Search engine built from the panel's majority-voted components."""

    TT_EXACT = 0
    TT_LOWER = 1
    TT_UPPER = 2

    def __init__(
        self,
        evaluator: EvalAgent,
        move_ordering: str,
        tt_policy: str,
        quiescence: str,
    ) -> None:
        self.evaluator = evaluator
        self.move_ordering = move_ordering
        self.tt_policy = tt_policy
        self.quiescence = quiescence
        self._tt: dict[object, TTEntry] = {}
        self._zobrist = Zobrist()

    def best_move(self, board: chess.Board, max_depth: int) -> tuple[chess.Move, int]:
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            raise ValueError("no legal moves available")
        if len(legal_moves) == 1:
            return legal_moves[0], self.evaluator.evaluate(board)

        self._tt.clear()
        best_move = legal_moves[0]
        best_score = -MATE_SCORE - 1
        hint: Optional[chess.Move] = None

        for depth in range(1, max_depth + 1):
            score, move = self._root_search(board, depth, hint)
            if move is not None:
                best_move = move
                best_score = score
                hint = move

        return best_move, best_score

    def _root_search(
        self,
        board: chess.Board,
        depth: int,
        hint: Optional[chess.Move],
    ) -> tuple[int, Optional[chess.Move]]:
        alpha = -INFINITY
        beta = INFINITY
        best_move: Optional[chess.Move] = None
        best_score = -INFINITY

        for move in self._ordered_moves(board, hint):
            board.push(move)
            score = -self._negamax(board, depth - 1, -beta, -alpha, 1)
            board.pop()

            if score > best_score:
                best_score = score
                best_move = move
            if score > alpha:
                alpha = score

        return best_score, best_move

    def _negamax(self, board: chess.Board, depth: int, alpha: int, beta: int, ply: int) -> int:
        if board.is_checkmate():
            return -MATE_SCORE + ply
        if (
            board.is_stalemate()
            or board.is_insufficient_material()
            or board.can_claim_threefold_repetition()
        ):
            return 0

        key = self._tt_key(board)
        entry = self._tt.get(key)
        tt_move = None

        if entry is not None and entry.depth >= depth:
            tt_move = entry.best_move
            if entry.flag == self.TT_EXACT:
                return entry.score
            if entry.flag == self.TT_LOWER and entry.score >= beta:
                return entry.score
            if entry.flag == self.TT_UPPER and entry.score <= alpha:
                return entry.score

        if depth <= 0:
            return self._quiesce(board, alpha, beta, 0)

        best_score = -INFINITY
        best_move = None
        original_alpha = alpha

        for move in self._ordered_moves(board, tt_move):
            board.push(move)
            score = -self._negamax(board, depth - 1, -beta, -alpha, ply + 1)
            board.pop()
            if score > best_score:
                best_score = score
                best_move = move
            if score > alpha:
                alpha = score
            if alpha >= beta:
                break

        flag = self.TT_EXACT
        if best_score <= original_alpha:
            flag = self.TT_UPPER
        elif best_score >= beta:
            flag = self.TT_LOWER

        self._tt[key] = TTEntry(depth, best_score, flag, best_move)
        return best_score

    def _quiesce(self, board: chess.Board, alpha: int, beta: int, qply: int) -> int:
        stand_pat = self._leaf_eval(board)
        if self.quiescence == "none":
            return stand_pat

        if stand_pat >= beta:
            return beta
        if stand_pat > alpha:
            alpha = stand_pat

        if qply > 8:
            return alpha

        noisy = []
        for move in board.legal_moves:
            if board.is_capture(move):
                noisy.append(move)
            elif self.quiescence == "captures_checks" and qply < 4 and board.gives_check(move):
                noisy.append(move)

        noisy.sort(key=lambda m: self._move_priority(board, m, None), reverse=True)

        for move in noisy:
            board.push(move)
            score = -self._quiesce(board, -beta, -alpha, qply + 1)
            board.pop()
            if score >= beta:
                return beta
            if score > alpha:
                alpha = score

        return alpha

    def _leaf_eval(self, board: chess.Board) -> int:
        score = self.evaluator.evaluate(board)
        return score if board.turn == chess.WHITE else -score

    def _tt_key(self, board: chess.Board) -> object:
        if self.tt_policy == "fen_replace":
            return board.fen()
        return self._zobrist.hash_board(board)

    def _ordered_moves(
        self,
        board: chess.Board,
        preferred: Optional[chess.Move],
    ) -> list[chess.Move]:
        moves = list(board.legal_moves)
        return sorted(moves, key=lambda move: self._move_priority(board, move, preferred), reverse=True)

    def _move_priority(
        self,
        board: chess.Board,
        move: chess.Move,
        preferred: Optional[chess.Move],
    ) -> int:
        if preferred is not None and move == preferred:
            return 10_000_000

        victim = board.piece_at(move.to_square)
        attacker = board.piece_at(move.from_square)
        victim_value = 0 if victim is None else self._piece_value(victim.piece_type)
        attacker_value = 0 if attacker is None else self._piece_value(attacker.piece_type)
        center_distance = abs(3.5 - chess.square_file(move.to_square)) + abs(
            3.5 - chess.square_rank(move.to_square)
        )

        if self.move_ordering == "aggressive":
            score = 0
            if board.gives_check(move):
                score += 1_000_000
            if board.is_capture(move):
                score += 100_000 + 10 * victim_value - attacker_value
            if move.promotion is not None:
                score += 50_000 + self._piece_value(move.promotion)
            score += int(20 - 4 * center_distance)
            return score

        if self.move_ordering == "capture_first":
            score = 0
            if board.is_capture(move):
                score += 1_000_000 + 10 * victim_value - attacker_value
            if move.promotion is not None:
                score += 50_000 + self._piece_value(move.promotion)
            if board.gives_check(move):
                score += 25_000
            return score

        # Balanced policy mirrors the CTO control engine.
        score = 0
        if board.is_capture(move):
            score += 100_000 + 10 * victim_value - attacker_value
        if move.promotion is not None:
            score += 50_000 + self._piece_value(move.promotion)
        if board.gives_check(move):
            score += 25_000
        return score

    @staticmethod
    def _piece_value(piece_type: chess.PieceType) -> int:
        return {
            chess.PAWN: 100,
            chess.KNIGHT: 320,
            chess.BISHOP: 330,
            chess.ROOK: 500,
            chess.QUEEN: 900,
            chess.KING: 0,
        }[piece_type]


# =============================================================================
# REFEREE
# =============================================================================


class Referee:
    """
    Scores the panel's selected move against a simple best-of-one-ply baseline.

    This is intentionally lightweight: it gives the panel feedback without
    requiring a separate engine process.
    """

    def __init__(self, evaluator: Optional[EvalAgent] = None) -> None:
        self.evaluator = evaluator or EvalAgent()

    def score_move(self, board: chess.Board, move: chess.Move) -> float:
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return 0.0

        best_score = -INFINITY
        played_score = -INFINITY
        for candidate in legal_moves:
            board.push(candidate)
            score = self.evaluator.evaluate(board)
            board.pop()
            oriented = score if board.turn == chess.WHITE else -score
            if oriented > best_score:
                best_score = oriented
            if candidate == move:
                played_score = oriented

        cp_loss = max(0, best_score - played_score)
        return float(max(0.0, min(1.0, math.exp(-cp_loss / 120.0))))


# =============================================================================
# PEER REVIEW PANEL
# =============================================================================


class PeerReviewPanel:
    """
    Majority-votes search components, then synthesizes one engine from them.
    """

    def __init__(
        self,
        reviewers: list[PanelReviewer],
        evaluator: Optional[EvalAgent] = None,
        referee: Optional[Referee] = None,
        persistence_path: Optional[str] = None,
    ) -> None:
        self.reviewers = {reviewer.reviewer_id: reviewer for reviewer in reviewers}
        self.evaluator = evaluator or EvalAgent()
        self.referee = referee or Referee(self.evaluator)
        self.persistence_path = persistence_path
        self._load_records()

    def run(self, board: chess.Board, max_depth: int = 3, update_records: bool = True) -> PanelResult:
        t0 = time.time()
        votes = self._collect_votes(board)
        decision = self._majority_vote(votes)

        engine = SynthesizedSearchEngine(
            evaluator=self.evaluator,
            move_ordering=decision.move_ordering,
            tt_policy=decision.tt_policy,
            quiescence=decision.quiescence,
        )
        move, score = engine.best_move(board.copy(stack=False), max_depth)
        quality = self.referee.score_move(board.copy(stack=False), move)

        if update_records:
            self._update_records(votes, decision, quality)
            self._save_records()

        expected_loss = round((1.0 - quality) * 100.0, 2)
        return PanelResult(
            optimal_move=move.uci(),
            move_tuple=move,
            score_cp=score,
            panel_ms=int((time.time() - t0) * 1000),
            decision=decision,
            expected_loss=expected_loss,
        )

    def _collect_votes(self, board: chess.Board) -> dict[str, list[ComponentVote]]:
        votes: dict[str, list[ComponentVote]] = defaultdict(list)
        for reviewer in self.reviewers.values():
            for vote in reviewer.review(board.copy(stack=False)):
                votes[vote.component].append(vote)
        return votes

    def _majority_vote(self, votes: dict[str, list[ComponentVote]]) -> PanelDecision:
        winners: dict[str, str] = {}
        for component, component_votes in votes.items():
            counts = Counter(vote.choice for vote in component_votes)
            top_count = max(counts.values())
            leaders = [choice for choice, count in counts.items() if count == top_count]
            if len(leaders) == 1:
                winners[component] = leaders[0]
                continue

            # Tie-break by highest summed confidence among tied choices.
            confidence_by_choice = {
                choice: sum(vote.confidence for vote in component_votes if vote.choice == choice)
                for choice in leaders
            }
            winners[component] = max(confidence_by_choice, key=confidence_by_choice.get)

        return PanelDecision(
            move_ordering=winners["move_ordering"],
            tt_policy=winners["tt_policy"],
            quiescence=winners["quiescence"],
            vote_breakdown=votes,
        )

    def _update_records(
        self,
        votes: dict[str, list[ComponentVote]],
        decision: PanelDecision,
        quality: float,
    ) -> None:
        accepted_components = {
            "move_ordering": decision.move_ordering,
            "tt_policy": decision.tt_policy,
            "quiescence": decision.quiescence,
        }
        for component, choice in accepted_components.items():
            accepted_votes = {
                vote.reviewer_id
                for vote in votes[component]
                if vote.choice == choice
            }
            for reviewer in self.reviewers.values():
                reviewer.record.update(
                    component,
                    choice,
                    reviewer.reviewer_id in accepted_votes and quality >= 0.5,
                )

    def leaderboard(self) -> str:
        lines = ["Peer Review Panel Leaderboard"]
        for reviewer in self.reviewers.values():
            accepted = sum(1 for item in reviewer.record.history if item["accepted"])
            total = len(reviewer.record.history)
            rate = (accepted / total) if total else 0.0
            lines.append(
                f"  {reviewer.reviewer_id:<18} accepted={accepted:<3} total={total:<3} rate={rate:.3f}"
            )
        return "\n".join(lines)

    def _load_records(self) -> None:
        if not self.persistence_path or not os.path.exists(self.persistence_path):
            return
        try:
            with open(self.persistence_path, "r", encoding="utf-8") as handle:
                data = json.load(handle)
            for reviewer_id, payload in data.items():
                if reviewer_id not in self.reviewers:
                    continue
                record = self.reviewers[reviewer_id].record
                record.component_wins = defaultdict(int, payload.get("component_wins", {}))
                record.history = payload.get("history", [])
        except Exception as exc:
            print(f"[PeerReviewPanel] load error: {exc}", file=sys.stderr)

    def _save_records(self) -> None:
        if not self.persistence_path:
            return
        payload = {}
        for reviewer_id, reviewer in self.reviewers.items():
            payload[reviewer_id] = {
                "component_wins": dict(reviewer.record.component_wins),
                "history": reviewer.record.history[-30:],
            }
        try:
            with open(self.persistence_path, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2)
        except Exception as exc:
            print(f"[PeerReviewPanel] save error: {exc}", file=sys.stderr)


# =============================================================================
# FACTORY
# =============================================================================


def build_peer_review_panel(
    persistence_path: str = "peer_review_records.json",
) -> PeerReviewPanel:
    reviewers = [
        ClassicalReviewer("ClassicalReviewer"),
        TacticalReviewer("TacticalReviewer"),
        StructuralReviewer("StructuralReviewer"),
    ]
    return PeerReviewPanel(
        reviewers=reviewers,
        evaluator=EvalAgent(),
        referee=Referee(EvalAgent()),
        persistence_path=persistence_path,
    )


# =============================================================================
# SELF-TEST
# =============================================================================


def _run_tests() -> bool:
    print("peer_review_panel.py - self-test")
    print("=" * 60)
    ok = True

    panel = build_peer_review_panel(persistence_path=None)

    # T1: Factory builds three reviewers.
    t1 = len(panel.reviewers) == 3
    ok &= t1
    print(f"  [{'PASS' if t1 else 'FAIL'}] Factory loads 3 reviewers")

    # T2: Majority vote returns a component decision.
    board = chess.Board()
    votes = panel._collect_votes(board)
    decision = panel._majority_vote(votes)
    t2 = decision.move_ordering in {"balanced", "aggressive", "capture_first"}
    ok &= t2
    print(f"  [{'PASS' if t2 else 'FAIL'}] Majority picks move-ordering policy ({decision.move_ordering})")

    # T3: Synthesized engine returns a legal move.
    result = panel.run(board, max_depth=2, update_records=False)
    t3 = chess.Move.from_uci(result.optimal_move) in board.legal_moves
    ok &= t3
    print(f"  [{'PASS' if t3 else 'FAIL'}] Panel returns legal move ({result.optimal_move})")

    # T4: Tactical position should allow checks in quiescence by majority.
    tactical = chess.Board("r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5Q2/PPPP1PPP/RNB1K1NR w KQkq - 4 4")
    tactical_decision = panel._majority_vote(panel._collect_votes(tactical))
    t4 = tactical_decision.quiescence == "captures_checks"
    ok &= t4
    print(f"  [{'PASS' if t4 else 'FAIL'}] Tactical board selects sharper quiescence ({tactical_decision.quiescence})")

    # T5: Endgame should not force aggressive ordering.
    endgame = chess.Board("8/5k2/8/8/8/8/2K5/8 w - - 0 1")
    end_decision = panel._majority_vote(panel._collect_votes(endgame))
    t5 = end_decision.move_ordering != "aggressive"
    ok &= t5
    print(f"  [{'PASS' if t5 else 'FAIL'}] Quiet endgame avoids aggressive ordering ({end_decision.move_ordering})")

    print("=" * 60)
    print("All tests PASSED" if ok else "Some tests FAILED")
    return ok


# =============================================================================
# SIMULATION
# =============================================================================


def _simulate(rounds: int = 10, depth: int = 2) -> None:
    panel = build_peer_review_panel(persistence_path=None)
    board = chess.Board()

    print("Peer Review Panel Simulation")
    print("=" * 60)
    for index in range(rounds):
        result = panel.run(board, max_depth=depth, update_records=False)
        print(
            f"Round {index + 1:>2}: move={result.optimal_move:<5} "
            f"ordering={result.decision.move_ordering:<13} "
            f"tt={result.decision.tt_policy:<13} "
            f"q={result.decision.quiescence}"
        )
        board.push(result.move_tuple)
        if board.is_game_over():
            board = chess.Board()


# =============================================================================
# CLI
# =============================================================================


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Condorcet peer review chess engine")
    parser.add_argument("--fen", default=chess.STARTING_FEN)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--simulate", action="store_true")
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--records", default="peer_review_records.json")
    args = parser.parse_args()

    if args.test:
        sys.exit(0 if _run_tests() else 1)

    if args.simulate:
        _simulate(rounds=args.rounds, depth=args.depth)
        return

    panel = build_peer_review_panel(persistence_path=args.records)
    board = chess.Board(args.fen)
    result = panel.run(board, max_depth=args.depth, update_records=False)
    print(result.summary())
    print()
    print(panel.leaderboard())


if __name__ == "__main__":
    main()
