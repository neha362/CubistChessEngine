# Scenario Harness

This implements your self-improvement tournament loop across existing engine folders.

## What it does

- Starts every model from the same round-0 snapshot.
- Runs `N` improvement rounds.
- Forces round artifacts to include:
  - stated evaluation function
  - candidate rewrites
  - scored candidates
  - chosen rewrite
  - reason for rewrite
- After each round, benchmarks each model against:
  - original baseline (round 0 of `baseline_model`)
  - previous self (round `r-1`)
  - all other models at round `r`
  - fixed tactical FEN set
- Logs:
  - pass/fail tests
  - win rates
  - illegal move count
  - average move time
  - per-round decision metadata

## Run

From repo root:

```bash
python scenario/run_scenario.py --config scenario/config.example.json
```

## Config

Use `scenario/config.example.json` as template.

Per model:

- `engine_type` must be one of:
  - `classical_minimax`
  - `claude_api`
  - `mcts`
  - `mock_nn`
  - `aggressive_berserker`
- `source_dir` points to that engine folder.
- `improve_command` is optional. If provided, it runs once per round in the model snapshot dir.
- `test_command` is optional.

Environment variables provided to `improve_command`:

- `SCENARIO_MODEL`
- `SCENARIO_ROUND`
- `SCENARIO_PROMPT_PATH`
- `SCENARIO_DECISION_JSON_PATH`

If `improve_command` is omitted, the runner creates a placeholder decision JSON so the loop remains executable.

## Output

Each run creates `scenario_outputs/run_YYYYMMDD_HHMMSS` with:

- `summary.csv`
- `manifest.json`
- `finished.json`
- `snapshots/<model>/round_<n>/...`
- `decisions/<model>/round_<n>.json`
- `matches/<model>/round_<n>_vs_*.json`
- `tactics/<model>/round_<n>.json`
- `tests/<model>/round_<n>.json` (if emitted by test command)
- `logs/<model>/*.log`

