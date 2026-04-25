# TDLeaf vs Sunfish benchmark

This folder contains a harness to benchmark `tdleaf_nnue_engine` against Sunfish.

## Baseline (100 games)

```bash
python -m tdleaf_nnue_engine.benchmarks.run_tdleaf_vs_sunfish --games 100 --seed 1234 --opening-plies 4 --tdleaf-depth 2 --sunfish-movetime-ms 35
```

## Baseline + adaptation + re-run

```bash
python -m tdleaf_nnue_engine.benchmarks.run_tdleaf_vs_sunfish --games 100 --adapt --adapted-games 100 --seed 1234 --opening-plies 4 --tdleaf-depth 2 --sunfish-movetime-ms 35
```

Artifacts are written to `tdleaf_nnue_engine/benchmarks/results/`.
