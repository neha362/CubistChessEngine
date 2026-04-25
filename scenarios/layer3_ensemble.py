"""
layer3_ensemble.py — Learned Ensemble / Layer 3
================================================

This module implements the position-conditioned trust layer that sits on top
of the first hidden-layer engine nodes.

Architecture
────────────
INPUT
  A list of engine proposals. Each proposal is an engine's vote:
    - move
    - score_cp
    - confidence
    - optional prior_weight (e.g. auction-house reliability)

LAYER 1
  Six scenario detectors derived from the board:
    tactical_chaos
    endgame_structure
    material_imbalance
    king_safety_crisis
    open_file_pressure
    pawn_storm_detected

LAYER 2
  Agreement signals:
    all_agree
    majority
    split

LAYER 3
  A Bayesian trust matrix W[engine][scenario].
  Each cell is a Beta(alpha, beta) belief about how reliable an engine is when
  that scenario is active.

OUTPUT
  Weighted vote over proposed moves, plus interpretable diagnostics.

Usage
─────
  python scenarios/layer3_ensemble.py --test
"""

from __future__ import annotations

import json
import math
import os
import sys
from dataclasses import dataclass, field
from typing import Optional

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for rel in ("adapter_code",):
    path = os.path.join(REPO_ROOT, rel)
    if path not in sys.path:
        sys.path.insert(0, path)

from chaos_move_gen import GameState, STARTPOS, all_legal_moves, from_fen, is_in_check, make_move, sq_name


SCENARIO_NAMES = (
    "tactical_chaos",
    "endgame_structure",
    "material_imbalance",
    "king_safety_crisis",
    "open_file_pressure",
    "pawn_storm_detected",
)


def row(i: int) -> int:
    return i // 8


def col(i: int) -> int:
    return i % 8


def clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def softmax(values: list[float]) -> list[float]:
    if not values:
        return []
    pivot = max(values)
    exps = [math.exp(v - pivot) for v in values]
    total = sum(exps)
    return [v / total for v in exps] if total else [1.0 / len(values)] * len(values)


@dataclass(frozen=True)
class EngineProposal:
    engine_id: str
    move: tuple
    score_cp: int
    confidence: float
    prior_weight: float = 1.0
    search_id: str = ""
    eval_id: str = ""

    @property
    def uci(self) -> str:
        frm, to, promo = self.move
        return sq_name(frm) + sq_name(to) + promo


@dataclass
class ScenarioProfile:
    activations: dict[str, float]

    def ordered(self) -> list[float]:
        return [self.activations[name] for name in SCENARIO_NAMES]

    def total_mass(self) -> float:
        return sum(self.ordered())


@dataclass
class AgreementProfile:
    all_agree: float
    majority: float
    split: float
    largest_group: int
    consensus_ratio: float
    majority_threshold: int


@dataclass
class PlattCalibrator:
    slope: float = 1.0
    bias: float = 0.0
    samples: list[tuple[float, float]] = field(default_factory=list)
    history_cap: int = 128

    @staticmethod
    def _sigmoid(x: float) -> float:
        return 1.0 / (1.0 + math.exp(-max(-20.0, min(20.0, x))))

    def predict(self, score_cp: int) -> float:
        x = max(-6.0, min(6.0, score_cp / 400.0))
        return self._sigmoid(self.slope * x + self.bias)

    def update(self, score_cp: int, quality: float, epochs: int = 24, lr: float = 0.15) -> None:
        x = max(-6.0, min(6.0, score_cp / 400.0))
        y = max(0.0, min(1.0, quality))
        self.samples.append((x, y))
        if len(self.samples) > self.history_cap:
            self.samples = self.samples[-self.history_cap:]

        if len(self.samples) < 3:
            return

        batch = self.samples[-64:]
        for _ in range(epochs):
            grad_slope = 0.0
            grad_bias = 0.0
            for x_i, y_i in batch:
                pred = self._sigmoid(self.slope * x_i + self.bias)
                err = pred - y_i
                grad_slope += err * x_i
                grad_bias += err
            n = float(len(batch))
            self.slope -= lr * grad_slope / n
            self.bias -= lr * grad_bias / n
            self.slope = max(0.05, min(6.0, self.slope))
            self.bias = max(-4.0, min(4.0, self.bias))


@dataclass
class TrustCell:
    alpha: float = 1.0
    beta: float = 1.0

    @property
    def mean(self) -> float:
        return self.alpha / (self.alpha + self.beta)

    def update(self, success: bool, strength: float) -> None:
        if strength <= 0.0:
            return
        if success:
            self.alpha += strength
        else:
            self.beta += strength


@dataclass
class Layer3Result:
    best_move: str
    chosen_engine: str
    move_weights: dict[str, float]
    engine_probs: dict[str, float]
    engine_trusts: dict[str, float]
    engine_confidences: dict[str, float]
    scenario_profile: ScenarioProfile
    agreement_profile: AgreementProfile
    short_circuit: bool
    # Carried forward so the quality scorer can do counterfactual updates
    # over every engine's bid, not just the chosen one. Defaulted to []
    # so external callers stay backward compatible.
    proposals: list = field(default_factory=list)


class ScenarioTrustMatrix:
    def __init__(self, engine_ids: list[str], success_threshold: float = 0.67):
        self.success_threshold = success_threshold
        self.cells: dict[str, dict[str, TrustCell]] = {
            engine_id: {name: TrustCell() for name in SCENARIO_NAMES}
            for engine_id in engine_ids
        }

    def ensure_engine(self, engine_id: str) -> None:
        self.cells.setdefault(engine_id, {name: TrustCell() for name in SCENARIO_NAMES})

    def trust_for(self, engine_id: str, scenario_profile: ScenarioProfile) -> float:
        self.ensure_engine(engine_id)
        mass = scenario_profile.total_mass()
        if mass <= 1e-9:
            return 1.0
        weighted = sum(
            self.cells[engine_id][name].mean * scenario_profile.activations[name]
            for name in SCENARIO_NAMES
        )
        return 0.5 + (weighted / mass)

    def update(self, engine_id: str, scenario_profile: ScenarioProfile, quality: float) -> None:
        self.ensure_engine(engine_id)
        success = quality >= self.success_threshold
        for name in SCENARIO_NAMES:
            self.cells[engine_id][name].update(success, scenario_profile.activations[name])

    def snapshot(self) -> dict[str, dict[str, dict[str, float]]]:
        return {
            engine_id: {
                scenario: {
                    "alpha": cell.alpha,
                    "beta": cell.beta,
                    "mean": cell.mean,
                }
                for scenario, cell in row.items()
            }
            for engine_id, row in self.cells.items()
        }

    @classmethod
    def from_snapshot(cls, snapshot: dict[str, dict[str, dict[str, float]]]) -> "ScenarioTrustMatrix":
        matrix = cls(engine_ids=list(snapshot))
        for engine_id, row in snapshot.items():
            matrix.ensure_engine(engine_id)
            for scenario, cell in row.items():
                if scenario not in matrix.cells[engine_id]:
                    continue
                matrix.cells[engine_id][scenario].alpha = float(cell.get("alpha", 1.0))
                matrix.cells[engine_id][scenario].beta = float(cell.get("beta", 1.0))
        return matrix


def _material_signature(state: GameState) -> dict[str, int]:
    values = {"P": 1, "N": 3, "B": 3, "R": 5, "Q": 9, "K": 0}
    out = {
        "w_total": 0,
        "b_total": 0,
        "w_nonpawn": 0,
        "b_nonpawn": 0,
        "w_queens": 0,
        "b_queens": 0,
    }
    for piece in state.board:
        if not piece:
            continue
        side, kind = piece[0], piece[1]
        out[f"{side}_total"] += values[kind]
        if kind != "P":
            out[f"{side}_nonpawn"] += values[kind]
        if kind == "Q":
            out[f"{side}_queens"] += 1
    return out


def _king_square(state: GameState, side: str) -> int:
    return next((i for i, p in enumerate(state.board) if p == side + "K"), -1)


def _chebyshev(a: int, b: int) -> int:
    return max(abs(row(a) - row(b)), abs(col(a) - col(b)))


def _file_counts(state: GameState) -> list[dict[str, int]]:
    counts = [{"wP": 0, "bP": 0, "wR": 0, "bR": 0, "wQ": 0, "bQ": 0} for _ in range(8)]
    for sq_idx, piece in enumerate(state.board):
        if piece and piece in counts[col(sq_idx)]:
            counts[col(sq_idx)][piece] += 1
    return counts


def _king_zone_pressure(state: GameState, defender: str) -> int:
    king_sq = _king_square(state, defender)
    if king_sq == -1:
        return 0
    attacker = "b" if defender == "w" else "w"
    pressure = 0
    for sq_idx, piece in enumerate(state.board):
        if not piece or piece[0] != attacker:
            continue
        dist = _chebyshev(sq_idx, king_sq)
        if dist <= 1:
            pressure += 3
        elif dist == 2:
            pressure += 1
    return pressure


def _pawn_storm_side(state: GameState, attacker: str, defender: str) -> float:
    king_sq = _king_square(state, defender)
    if king_sq == -1:
        return 0.0
    king_file = col(king_sq)
    total = 0.0
    for sq_idx, piece in enumerate(state.board):
        if piece != attacker + "P":
            continue
        if abs(col(sq_idx) - king_file) > 1:
            continue
        advance = (6 - row(sq_idx)) if attacker == "w" else (row(sq_idx) - 1)
        if advance <= 1:
            continue
        total += min(1.0, advance / 5.0)
    return total


def detect_scenarios(state: GameState) -> ScenarioProfile:
    legal_moves = all_legal_moves(state)
    material = _material_signature(state)

    capture_count = 0
    check_count = 0
    for frm, to, promo in legal_moves:
        moving_piece = state.board[frm]
        is_ep = moving_piece and moving_piece[1] == "P" and col(frm) != col(to) and state.ep_square == to
        if state.board[to] is not None or is_ep:
            capture_count += 1
        try:
            child = make_move(state, (frm, to, promo))
            if is_in_check(child, child.turn):
                check_count += 1
        except Exception:
            pass

    move_count = len(legal_moves)
    tactical = clamp01(
        0.45 * min(1.0, capture_count / 8.0) +
        0.25 * min(1.0, check_count / 5.0) +
        0.20 * min(1.0, move_count / 35.0) +
        0.10 * (1.0 if move_count <= 6 else 0.0)
    )

    total_nonpawn = material["w_nonpawn"] + material["b_nonpawn"]
    endgame = clamp01(
        0.65 * (1.0 - min(1.0, total_nonpawn / 24.0)) +
        0.35 * (1.0 if material["w_queens"] + material["b_queens"] == 0 else 0.0)
    )

    material_imbalance = clamp01(abs(material["w_total"] - material["b_total"]) / 8.0)
    king_safety = clamp01(max(_king_zone_pressure(state, "w"), _king_zone_pressure(state, "b")) / 10.0)

    open_file_hits = 0
    for file_info in _file_counts(state):
        if (file_info["wR"] + file_info["wQ"]) and file_info["wP"] == 0:
            open_file_hits += 1
        if (file_info["bR"] + file_info["bQ"]) and file_info["bP"] == 0:
            open_file_hits += 1
    open_file_pressure = clamp01(open_file_hits / 4.0)

    pawn_storm = clamp01(
        max(_pawn_storm_side(state, "w", "b"), _pawn_storm_side(state, "b", "w")) / 3.0
    )

    return ScenarioProfile(
        activations={
            "tactical_chaos": tactical,
            "endgame_structure": endgame,
            "material_imbalance": material_imbalance,
            "king_safety_crisis": king_safety,
            "open_file_pressure": open_file_pressure,
            "pawn_storm_detected": pawn_storm,
        }
    )


def agreement_profile(proposals: list[EngineProposal]) -> AgreementProfile:
    counts: dict[str, int] = {}
    for proposal in proposals:
        counts[proposal.uci] = counts.get(proposal.uci, 0) + 1
    largest = max(counts.values()) if counts else 0
    n = len(proposals)
    majority_threshold = max(1, math.ceil(n / 2.0))
    return AgreementProfile(
        all_agree=1.0 if n > 0 and largest == n else 0.0,
        majority=1.0 if largest >= majority_threshold else 0.0,
        split=1.0 if n > 0 and largest < majority_threshold else 0.0,
        largest_group=largest,
        consensus_ratio=(largest / n) if n else 0.0,
        majority_threshold=majority_threshold,
    )


class Layer3Ensemble:
    def __init__(
        self,
        engine_ids: list[str],
        persistence_path: Optional[str] = None,
        consensus_threshold: Optional[int] = None,
    ):
        self.consensus_threshold = consensus_threshold
        self.persistence_path = persistence_path
        self.trust_matrix = ScenarioTrustMatrix(engine_ids=engine_ids)
        self.calibrators = {engine_id: PlattCalibrator() for engine_id in engine_ids}
        self._load()

    def _load(self) -> None:
        if not self.persistence_path or not os.path.exists(self.persistence_path):
            return
        try:
            with open(self.persistence_path) as f:
                data = json.load(f)
            self.trust_matrix = ScenarioTrustMatrix.from_snapshot(data.get("scenario_matrix", {}))
            cal_data = data.get("calibration", {})
            for engine_id, payload in cal_data.items():
                calibrator = self.calibrators.setdefault(engine_id, PlattCalibrator())
                calibrator.slope = float(payload.get("slope", calibrator.slope))
                calibrator.bias = float(payload.get("bias", calibrator.bias))
                calibrator.samples = [
                    (float(x), float(y))
                    for x, y in payload.get("samples", [])[-calibrator.history_cap:]
                ]
        except Exception:
            pass

    def _save(self) -> None:
        if not self.persistence_path:
            return
        payload = {
            "scenario_matrix": self.trust_matrix.snapshot(),
            "calibration": {
                engine_id: {
                    "slope": calibrator.slope,
                    "bias": calibrator.bias,
                    "samples": calibrator.samples[-64:],
                }
                for engine_id, calibrator in self.calibrators.items()
            },
        }
        with open(self.persistence_path, "w") as f:
            json.dump(payload, f, indent=2)

    def _consensus_threshold(self, proposal_count: int) -> int:
        if self.consensus_threshold is not None:
            return self.consensus_threshold
        return max(4, math.ceil(proposal_count * 0.6))

    def evaluate(self, state: GameState, proposals: list[EngineProposal]) -> Layer3Result:
        if not proposals:
            raise ValueError("Layer3Ensemble requires at least one proposal")

        scenarios = detect_scenarios(state)
        agreement = agreement_profile(proposals)

        votes: dict[str, list[EngineProposal]] = {}
        for proposal in proposals:
            self.trust_matrix.ensure_engine(proposal.engine_id)
            self.calibrators.setdefault(proposal.engine_id, PlattCalibrator())
            votes.setdefault(proposal.uci, []).append(proposal)

        if agreement.largest_group >= self._consensus_threshold(len(proposals)):
            best_uci, group = max(votes.items(), key=lambda item: len(item[1]))
            chosen_engine = max(group, key=lambda p: p.confidence * p.prior_weight).engine_id
            prob = 1.0 / len(group)
            return Layer3Result(
                best_move=best_uci,
                chosen_engine=chosen_engine,
                move_weights={best_uci: 1.0},
                engine_probs={proposal.engine_id: (prob if proposal in group else 0.0) for proposal in proposals},
                engine_trusts={proposal.engine_id: self.trust_matrix.trust_for(proposal.engine_id, scenarios) for proposal in proposals},
                engine_confidences={
                    proposal.engine_id: 0.5 * proposal.confidence
                    + 0.5 * self.calibrators[proposal.engine_id].predict(proposal.score_cp)
                    for proposal in proposals
                },
                scenario_profile=scenarios,
                agreement_profile=agreement,
                short_circuit=True,
                proposals=list(proposals),
            )

        logits = []
        trusts = {}
        confidences = {}
        for proposal in proposals:
            trust = self.trust_matrix.trust_for(proposal.engine_id, scenarios)
            trusts[proposal.engine_id] = trust
            calibrated = self.calibrators[proposal.engine_id].predict(proposal.score_cp)
            combined_conf = 0.5 * proposal.confidence + 0.5 * calibrated
            confidences[proposal.engine_id] = combined_conf
            score_signal = 2.0 * calibrated - 1.0
            conf_signal = 2.0 * combined_conf - 1.0
            prior_signal = math.log(max(1e-6, proposal.prior_weight))
            logits.append(1.4 * trust + 0.8 * conf_signal + 0.5 * score_signal + 0.35 * prior_signal)

        probs = softmax(logits)
        engine_probs = {proposal.engine_id: prob for proposal, prob in zip(proposals, probs)}

        move_weights: dict[str, float] = {}
        for proposal, prob in zip(proposals, probs):
            move_weights[proposal.uci] = move_weights.get(proposal.uci, 0.0) + prob

        best_move = max(move_weights.items(), key=lambda item: item[1])[0]
        chosen_engine = max(
            (proposal for proposal in proposals if proposal.uci == best_move),
            key=lambda proposal: engine_probs[proposal.engine_id],
        ).engine_id

        return Layer3Result(
            best_move=best_move,
            chosen_engine=chosen_engine,
            move_weights=move_weights,
            engine_probs=engine_probs,
            engine_trusts=trusts,
            engine_confidences=confidences,
            scenario_profile=scenarios,
            agreement_profile=agreement,
            short_circuit=False,
            proposals=list(proposals),
        )

    def update(self, engine_id: str, scenario_profile: ScenarioProfile, quality: float,
               score_cp: Optional[int] = None, autosave: bool = True) -> None:
        self.trust_matrix.update(engine_id, scenario_profile, quality)
        if score_cp is not None:
            self.calibrators.setdefault(engine_id, PlattCalibrator()).update(score_cp, quality)
        if autosave:
            self._save()

    def update_from_result(self, result: Layer3Result, quality: float, autosave: bool = True) -> None:
        score_cp = None
        for proposal in result.proposals:
            if proposal.engine_id == result.chosen_engine:
                score_cp = proposal.score_cp
                break
        self.update(result.chosen_engine, result.scenario_profile, quality, score_cp=score_cp, autosave=autosave)

    def trust_report(self) -> str:
        lines = ["┌── LAYER 3 TRUST MATRIX ───────────────────────────────────┐"]
        for engine_id, row in self.trust_matrix.cells.items():
            lines.append(f"│  {engine_id:<22}                                   │")
            for scenario in SCENARIO_NAMES:
                cell = row[scenario]
                lines.append(
                    f"│    {scenario:<20} mean={cell.mean:.3f}  "
                    f"(a={cell.alpha:.1f}, b={cell.beta:.1f}) │"
                )
            calibrator = self.calibrators.get(engine_id, PlattCalibrator())
            lines.append(
                f"│    calibration           slope={calibrator.slope:.3f} "
                f"bias={calibrator.bias:.3f} │"
            )
        lines.append("└────────────────────────────────────────────────────────────┘")
        return "\n".join(lines)


def _run_tests() -> bool:
    print("layer3_ensemble.py — self-test")
    print("=" * 62)
    ok = True

    state = from_fen(STARTPOS)
    profile = detect_scenarios(state)
    t1 = all(0.0 <= profile.activations[name] <= 1.0 for name in SCENARIO_NAMES)
    ok &= t1
    print(f"  [{'PASS' if t1 else 'FAIL'}] Scenario activations bounded")

    ensemble = Layer3Ensemble(engine_ids=["classical_classical", "chaos_chaos", "mcts_neural", "neural_siege", "siege_classical"])

    proposals = [
        EngineProposal("classical_classical", (52, 36, ""), 40, 0.60, 1.0, "classical", "classical"),  # e2e4
        EngineProposal("chaos_chaos", (51, 35, ""), 25, 0.55, 1.0, "chaos", "chaos"),                  # d2d4
        EngineProposal("mcts_neural", (52, 36, ""), 10, 0.52, 1.0, "mcts", "neural"),                  # e2e4
        EngineProposal("neural_siege", (62, 45, ""), 55, 0.68, 1.0, "neural", "siege"),                # g1f3
        EngineProposal("siege_classical", (52, 36, ""), 15, 0.57, 1.0, "siege", "classical"),          # e2e4
    ]
    result = ensemble.evaluate(state, proposals)
    t2 = result.best_move in {"e2e4", "g1f3", "d2d4"}
    ok &= t2
    print(f"  [{'PASS' if t2 else 'FAIL'}] Ensemble returns a legal-looking move ({result.best_move})")

    before = ensemble.trust_matrix.cells["neural_siege"]["tactical_chaos"].mean
    chaos_only = ScenarioProfile({
        "tactical_chaos": 1.0,
        "endgame_structure": 0.0,
        "material_imbalance": 0.0,
        "king_safety_crisis": 0.0,
        "open_file_pressure": 0.0,
        "pawn_storm_detected": 0.0,
    })
    ensemble.update("neural_siege", chaos_only, quality=1.0, score_cp=90, autosave=False)
    after = ensemble.trust_matrix.cells["neural_siege"]["tactical_chaos"].mean
    untouched = ensemble.trust_matrix.cells["neural_siege"]["endgame_structure"].mean
    t3 = after > before and abs(untouched - 0.5) < 1e-9
    ok &= t3
    print(f"  [{'PASS' if t3 else 'FAIL'}] Bayesian updates stay localized")

    calibrated = ensemble.calibrators["neural_siege"].predict(90)
    t4 = 0.0 <= calibrated <= 1.0
    ok &= t4
    print(f"  [{'PASS' if t4 else 'FAIL'}] Platt calibration stays bounded ({calibrated:.3f})")

    consensus_props = [
        EngineProposal("A", (52, 36, ""), 10, 0.51),
        EngineProposal("B", (52, 36, ""), 20, 0.52),
        EngineProposal("C", (52, 36, ""), 30, 0.53),
        EngineProposal("D", (52, 36, ""), 40, 0.54),
        EngineProposal("E", (51, 35, ""), 80, 0.80),
    ]
    consensus = Layer3Ensemble(engine_ids=["A", "B", "C", "D", "E"], consensus_threshold=None)
    result2 = consensus.evaluate(state, consensus_props)
    t5 = result2.short_circuit and result2.best_move == "e2e4"
    ok &= t5
    print(f"  [{'PASS' if t5 else 'FAIL'}] Consensus short-circuit fires")

    twenty_five = [
        EngineProposal(f"engine_{i}", (52, 36, "") if i < 13 else (51, 35, ""), 20 + i, 0.55)
        for i in range(25)
    ]
    ap = agreement_profile(twenty_five)
    t6 = ap.majority == 1.0 and ap.majority_threshold == 13 and abs(ap.consensus_ratio - 13 / 25) < 1e-9
    ok &= t6
    print(f"  [{'PASS' if t6 else 'FAIL'}] Agreement thresholds scale to 25 nodes")

    print("=" * 62)
    print("All tests PASSED" if ok else "Some tests FAILED")
    return ok


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Layer 3 learned ensemble")
    parser.add_argument("--test", action="store_true")
    args = parser.parse_args()

    if args.test:
        raise SystemExit(0 if _run_tests() else 1)

    print("Run with --test to exercise the Layer 3 ensemble module.")


if __name__ == "__main__":
    main()
