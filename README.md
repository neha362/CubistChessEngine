# CubistChessEngine

## Evolutionary tournament (Scenario 6)

From the repo root, after installing `requirements.txt`:

```text
python -m evolutionary_tournament
python -m evolutionary_tournament --evolve
```

To point the reference at a local Stockfish binary, set (Windows example):

```text
set STOCKFISH_EXECUTABLE=C:\path\to\stockfish.exe
```

Then run the commands above. If unset, the harness falls back to a built-in evaluator.
