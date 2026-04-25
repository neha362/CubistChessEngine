# TD-Leaf NNUE Engine (Hackathon v1)

CPU-friendly chess engine prototype with:

- legal move generation + basic ordering heuristics
- negamax alpha-beta search with transposition table and killer moves
- runtime NNUE-style evaluator with material fallback
- self-play TD-Leaf(lambda) target generation and PyTorch training

## Quick Start

Install dependencies:

```bash
pip install -r requirements.txt
```

Run engine demo:

```bash
python -m tdleaf_nnue_engine.main --depth 3
```

Train small model:

```bash
python -m tdleaf_nnue_engine.train --games 2 --epochs 2
```

Train in a way that tends to actually improve (self-play + TD(0) + replay buffer):

```bash
python -m tdleaf_nnue_engine.improve --minutes 30 --depth 2
```

This writes:
- `tdleaf_nnue_engine/checkpoints/nnue_model_improved.pt`
- `tdleaf_nnue_engine/checkpoints/nnue_runtime_improved.npz`

Bootstrap from Sunfish first (move-only distillation):

```bash
python -m tdleaf_nnue_engine.distill_sunfish_move --samples 2000 --epochs 5 --sunfish-movetime-ms 35
```

This writes:
- `tdleaf_nnue_engine/checkpoints/nnue_model_distilled.pt`
- `tdleaf_nnue_engine/checkpoints/nnue_runtime_distilled.npz`

Then fine-tune with TD training:

```bash
python -m tdleaf_nnue_engine.improve --minutes 30 --depth 2 --init-checkpoint tdleaf_nnue_engine/checkpoints/nnue_model_distilled.pt --init-runtime tdleaf_nnue_engine/checkpoints/nnue_runtime_distilled.npz --checkpoint tdleaf_nnue_engine/checkpoints/nnue_model_distilled_improved.pt --runtime tdleaf_nnue_engine/checkpoints/nnue_runtime_distilled_improved.npz
```

Export runtime weights:

```bash
python -m tdleaf_nnue_engine.export
```

Run with exported weights:

```bash
python -m tdleaf_nnue_engine.main --weights tdleaf_nnue_engine/checkpoints/nnue_runtime.npz
```

Play against the engine (interactive CLI):

```bash
python -m tdleaf_nnue_engine.play --side white --depth 3 --weights tdleaf_nnue_engine/checkpoints/nnue_runtime.npz
```

Play as Black:

```bash
python -m tdleaf_nnue_engine.play --side black --depth 3
```

In-game commands:

- `help` show input examples and commands
- `board` reprint current board
- `fen` print current FEN
- `quit` exit the game

## Notes

- The NNUE runtime path supports future incremental updates; v1 recomputes features per node for simplicity.
- If runtime weights are missing, evaluation automatically falls back to material bootstrap.
