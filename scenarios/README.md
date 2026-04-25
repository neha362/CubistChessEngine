# Cubist Ensemble — Wiring Layer (v2)

Three new files turn the existing engines + Layer 3 trust matrix into a
working ensemble engine that **actually learns** from games.

## What changed in v2

- **Per-move quality scoring** (`quality_scorer.py`) — the trust matrix now
  learns from real per-move signals, not just game outcomes. This was
  TODO #1 in v1 and was the biggest blocker to meaningful learning.
- **Parallel engine execution** — `gather_proposals()` runs all engines
  concurrently in a thread pool. ~1.2x speedup on pure-Python engines,
  much larger speedup once the Claude oracle is enabled (its API call
  no longer blocks the others).
- **Position cache** — repeated FENs (transpositions, analysis) skip
  re-running the engines entirely. Effectively free re-evaluation.
- **Berserker recommendation**: keep `berserker_2`, delete `berserker1`.
  Empirical match: berserker_2 won 5–1 (4–0 in decisive games). See the
  "Which Berserker" section below.

## Files

| File | Role |
|---|---|
| `layer3_ensemble.py` | Architecture — scenario detector, Bayesian trust matrix, softmax vote |
| `engine_wiring.py` | **NEW** uniform `propose(fen)` + parallel `gather_proposals` + cache |
| `quality_scorer.py` | **NEW** per-move quality measurement vs reference engine |
| `run_ensemble.py` | **NEW** `CubistEngine` + CLI for analysis & self-play with learning |
| `auction_house.py`, `peer_review_panel.py`, `red_blue.py` | Alternative meta-architectures (not used by Cubist) |

## Install

Drop these files into `scenarios/` (and `game_state.py` into
`classical_minimax/chess_engine/` to fix the broken classical engine).
That's the entire change.

## Run it

```bash
# Single position with full diagnostics:
python scenarios/run_ensemble.py --explain

# Self-play (now learns from per-move quality):
python scenarios/run_ensemble.py --selfplay --moves 20

# Pick a subset:
python scenarios/run_ensemble.py --selfplay --moves 10 --engines classical siege

# Reset learned trust:
python scenarios/run_ensemble.py --selfplay --moves 10 --reset

# Self-test the quality scorer:
python scenarios/quality_scorer.py
```

## Architecture flow

```
INPUT (FEN)
   │
   ▼
gather_proposals(fen)  -- parallel + cached --   (engine_wiring.py)
   │   classical(fen) -> EngineProposal(g1f3, +0 cp, 0.0 conf)
   │   berserker(fen) -> EngineProposal(e2e4, -19, 0.10)
   │   mcts(fen)      -> EngineProposal(a2a3, +0, 0.0)
   │   siege(fen)     -> EngineProposal(g1f3, +50, 0.25)
   │   oracle(fen)    -> EngineProposal(...) | None (graceful fallback)
   ▼
Layer3Ensemble.evaluate(state, proposals)
   │
   ├── LAYER 1: detect_scenarios(state) -> 6 activations in [0,1]
   ├── LAYER 2: agreement_profile(proposals) -> 3 consensus signals
   └── LAYER 3: softmax(1.4*trust + 0.8*conf + 0.5*score + 0.35*prior)
                trust = ScenarioTrustMatrix.trust_for(engine, scenarios)
                       weighted Bayesian Beta(a,b) by scenario activation
   │
   ▼
Layer3Result(best_move, chosen_engine, weights, diagnostics)
   │
   ▼   (after each game completes)
quality_scorer.score_game(history) -> per-move qualities
   │
   ▼
Layer3Ensemble.update(engine, scenarios, quality) for every move
   │
   ▼
Trust matrix updated, persisted to scenarios/cubist_trust.json
```

## How the per-move quality scorer works

After each game, every move played is scored against the reference engine
(classical at depth 4):

```
quality(move) = 1 - clamp((ref_eval_after_ref_move - ref_eval_after_played) / 200, 0, 1)
```

Match the reference's choice: quality = 1.0
Lose 100 cp:                   quality = 0.5
Lose 200+ cp:                  quality = 0.0
Illegal move:                  quality = 0.0

Then `cell.alpha += quality * scenario_activation` and
`cell.beta  += (1 - quality) * scenario_activation` for each engine on each
ply where it was the chosen voice. This is the signal that lets the trust
matrix learn meaningful per-scenario per-engine differences.

**Important caveat**: classical at depth 4 is only "strong enough" to
differentiate moves that the engines (typically at depth 3) play. For
real Elo-style trust calibration you'd want Stockfish at depth 12+ as
the reference. Adding that is one extra engine call per move — easy to
swap in if you're willing to add the dependency.

## Performance

Real measurements on a 4-engine ensemble (classical, berserker, siege, mcts)
from the start position:

| Mode | Time | Speedup |
|---|---|---|
| Sequential | 2.40s | 1.00x |
| Parallel | 1.97s | 1.22x |
| Parallel + cache hit | 0.0001s | 24,000x |

The parallel speedup is modest because of Python's GIL — pure-Python
engines don't truly run concurrently. The big wins are:
1. Cache hits during analysis (transpositions in self-play).
2. The Claude oracle, whose 5-12s API call now overlaps with the others
   (parallel speedup with oracle enabled is ~3-4x).

## Which Berserker — empirical match results

I ran berserker1 vs berserker_2 with 0.15s/move time budgets, alternating
colors. **berserker_2 won 5-1** (with 4-0 in decisive games and the other
two being early-truncation draws):

```
Game 1: berserker1 (W) vs berserker_2 (B) -> black wins (44 plies)
Game 2: berserker_2 (W) vs berserker1 (B) -> white wins (31 plies)
Game 3: berserker1 (W) vs berserker_2 (B) -> black wins (22 plies)
Game 4: berserker_2 (W) vs berserker1 (B) -> white wins (21 plies)
```

berserker_2 also runs about **3-4x faster per move** at equal time budgets,
because berserker1's GameState representation (Python lists with string
piece codes) is slower than berserker_2's python-chess (C-backed bitboards
under the hood).

berserker1 has one feature berserker_2 lacks (killer move heuristic), but
at the time budgets you'd actually use in the ensemble (sub-second per
engine per move), berserker_2's faster nodes-per-second wins decisively.

**Recommendation**: delete `berserker1/`. Update `engine_wiring.py` to
point `propose_berserker` at berserker_2 (currently it's pointing at
berserker1 because they're path-distinguished as `berserker` and `siege`).
This loses the second aggressive voice in the ensemble, but you can
re-introduce diversity later via different parameter settings on the same
engine (e.g. `siege` at depth 3 vs `siege_aggressive` at depth 4 with
extended check qsearch — they'll produce different proposals).

## Trust persistence

Saved to `scenarios/cubist_trust.json` after every self-play game. Pass
`--reset` or delete the file to start over.

## What's still TODO

### 1. Stronger reference engine (MEDIUM IMPACT)

`quality_scorer.py` uses classical at depth 4. For real Elo-grade trust
calibration, swap in Stockfish via `python-stockfish` or `subprocess`.
Drop-in replacement — same `score_move_quality(fen, played_uci) -> float`
signature. Hours, not days.

### 2. Confidence calibration (MEDIUM IMPACT)

`engine_wiring.py` uses ad-hoc confidence formulas:
- alpha-beta: `tanh(|score|/200)`
- MCTS: `|win_rate - 0.5| * 2`
- oracle: fixed 0.7

Validate empirically: log many (engine, position, confidence,
actual_quality) tuples, then fit a per-engine isotonic regression. Engines
whose confidences are uncalibrated will mislead the softmax. The
quality_scorer above gives you actual_quality for free.

### 3. Coefficient tuning in Layer3Ensemble (LOW IMPACT)

`1.4*trust + 0.8*conf + 0.5*score + 0.35*prior` — the magic numbers in
`layer3_ensemble.py:421`. Once you've logged data, optimize via grid
search or gradient descent on a held-out set of positions. Small gains.

### 4. UCI front-end (separate, LOW DIFFICULTY)

`run_ensemble.py` is a CLI, not a UCI engine. Adding a UCI shell over
`CubistEngine.best_move()` is ~50 lines and would let you plug Cubist
into Cute Chess or Lichess as a real bot.

## Things the wiring layer does NOT do

- **Time management** — every engine gets a fixed budget per call. A real
  UCI front-end would distribute remaining game time across engines.
- **Pondering** — no thinking on the opponent's clock.
- **Opening book** — fresh search every move from move 1. (Could be
  added cheaply: a small Polyglot reader; python-chess includes one.)
