# Cubist Ensemble — Wiring Layer 

Three new files turn the existing engines + Layer 3 trust matrix into a
working ensemble engine that **actually learns** from games.


- **Per-move quality scoring** (`quality_scorer.py`) — the trust matrix 
  learns from real per-move signals and game outcomes. 
- **Parallel engine execution** — `gather_proposals()` runs all engines
  concurrently in a thread pool. ~1.2x speedup on pure-Python engines,
  much larger speedup once the Claude oracle is enabled (its API call
  no longer blocks the others).
- **Position cache** — repeated FENs (transpositions, analysis) skip
  re-running the engines entirely. Effectively free re-evaluation.

## Files

| File | Role |
|---|---|
| `layer3_ensemble.py` | Architecture — scenario detector, Bayesian trust matrix, softmax vote |
| `engine_wiring.py` | uniform `propose(fen)` + parallel `gather_proposals` + cache |
| `quality_scorer.py` | per-move quality measurement vs reference engine |
| `run_ensemble.py` | `CubistEngine` + CLI for analysis & self-play with learning |
| `auction_house.py`, `peer_review_panel.py`, `red_blue.py` | Alternative meta-architectures (not used by Cubist) |


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

## Trust persistence

Saved to `scenarios/cubist_trust.json` after every self-play game. Pass
`--reset` or delete the file to start over.


## Things the wiring layer does NOT do

- **Time management** — every engine gets a fixed budget per call. A real
  UCI front-end would distribute remaining game time across engines.
- **Pondering** — no thinking on the opponent's clock.
- **Opening book** — fresh search every move from move 1. (Could be
  added cheaply: a small Polyglot reader; python-chess includes one.)
