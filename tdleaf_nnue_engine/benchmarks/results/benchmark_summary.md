# TDLeaf vs Sunfish Benchmark Summary

## Baseline
- Games: 100
- W/D/L: 0/19/81
- Score rate: 0.095
- Win rate: 0.000
- Elo estimate: -391.6
- Weights: `tdleaf_nnue_engine\checkpoints\nnue_runtime.npz`

## Commands Used
- `python -m tdleaf_nnue_engine.benchmarks.run_tdleaf_vs_sunfish --games 100 --seed 1234 --opening-plies 4 --tdleaf-depth 2 --sunfish-movetime-ms 35`
