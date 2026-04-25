# TDLeaf vs Sunfish Benchmark Summary

## Baseline
- Games: 100
- W/D/L: 0/26/74
- Score rate: 0.130
- Win rate: 0.000
- Elo estimate: -330.2
- Weights: `C:\Users\kliu3\Downloads\cubist\tdleaf_nnue_engine\checkpoints\nnue_runtime_distilled_improved_30m_snapshot.npz`

## Commands Used
- `python -m tdleaf_nnue_engine.benchmarks.run_tdleaf_vs_sunfish --games 100 --seed 1234 --opening-plies 4 --tdleaf-depth 2 --sunfish-movetime-ms 35`
