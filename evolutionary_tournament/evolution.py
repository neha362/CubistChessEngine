"""
Evolutionary “tournament” over :class:`TunableWeights` — fitness rewards
**cooperation** with a reference (Stockfish when available, else the built-in
heuristic): per-position value correlation. Weak lineages are **mutated**;
strong parents seed the next round.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Callable

import chess
import chess.engine

from . import ground_truth
from .tunable_classical import TunableWeights, evaluate_tunable


@dataclass
class RoundStats:
    round: int
    best_fitness: float
    mean_fitness: float
    best_weights: TunableWeights
    log_lines: list[str] = field(default_factory=list)


def _pearson(a: list[float], b: list[float]) -> float:
    n = len(a)
    if n < 2 or n != len(b):
        return 0.0
    ma, mb = sum(a) / n, sum(b) / n
    ex = sum((x - ma) ** 2 for x in a) ** 0.5
    ey = sum((y - mb) ** 2 for y in b) ** 0.5
    if ex < 1e-9 or ey < 1e-9:
        return 0.0
    return sum((a[i] - ma) * (b[i] - mb) for i in range(n)) / (ex * ey)


def _ref_fn(engine: chess.engine.SimpleEngine | None) -> Callable[[chess.Board], int]:
    d = 10 if engine is not None else 4
    return lambda b: int(ground_truth.position_value_white(b, d, engine))  # noqa: B023


def _fitness(
    w: TunableWeights, fens: list[str], ref: Callable[[chess.Board], int]
) -> float:
    ours, refs = [], []
    for f in fens:
        b = chess.Board(fen=f)
        ours.append(float(evaluate_tunable(b, w)))
        refs.append(float(ref(b)))
    return _pearson(ours, refs)


def run_evolution(
    fens: list[str],
    population: int = 8,
    rounds: int = 4,
    rng: random.Random | None = None,
) -> list[RoundStats]:
    r = rng or random.Random(42)
    with ground_truth.reference_engine() as eng:
        ref = _ref_fn(eng)
        pop = [TunableWeights(1, 1, 0).with_noise(0.3, r) for _ in range(population)]
        out: list[RoundStats] = []
        for rnd in range(1, rounds + 1):
            fit = [(_fitness(p, fens, ref), p) for p in pop]
            fit.sort(key=lambda t: t[0], reverse=True)
            best, mean = fit[0][0], sum(f for f, _ in fit) / len(fit)
            lines: list[str] = [
                f"round {rnd} best_rho = {best:.3f} mean = {mean:.3f} "
                f"weights w_mat={fit[0][1].w_mat:.3f} w_pst={fit[0][1].w_pst:.3f} bias={fit[0][1].w_bias:.1f}"
            ]
            out.append(
                RoundStats(
                    round=rnd,
                    best_fitness=best,
                    mean_fitness=mean,
                    best_weights=fit[0][1],
                    log_lines=lines,
                )
            )
            # elitist + tournament mutation
            elite = [p for _, p in fit[: max(2, population // 4)]]
            next_p: list[TunableWeights] = list(elite)
            while len(next_p) < population:
                a, b = r.choices([p for _, p in fit[: population // 2]], k=2)
                child = TunableWeights(
                    w_mat=0.5 * (a.w_mat + b.w_mat),
                    w_pst=0.5 * (a.w_pst + b.w_pst),
                    w_bias=0.5 * (a.w_bias + b.w_bias),
                )
                if r.random() < 0.4:
                    child = child.with_noise(0.12, r)
                else:
                    child = TunableWeights(
                        w_mat=child.w_mat * r.uniform(0.9, 1.1),
                        w_pst=child.w_pst * r.uniform(0.9, 1.1),
                        w_bias=child.w_bias,
                    )
                next_p.append(child)
            pop = next_p
        return out
