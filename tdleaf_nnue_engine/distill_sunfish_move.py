"""
Move-only distillation from Sunfish into tdleaf NNUE.

Idea:
  For random positions, ask Sunfish for best move.
  Train NNUE so that the post-move value of Sunfish's move ranks above
  a random legal alternative (pairwise ranking loss).

This produces a stronger initialization than pure self-play from scratch.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import chess
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from tdleaf_nnue_engine.benchmarks.run_tdleaf_vs_sunfish import (
    DEFAULT_SUNFISH_DIR,
    SunfishUCI,
    ensure_sunfish_checkout,
)
from tdleaf_nnue_engine.export import export_runtime_weights
from tdleaf_nnue_engine.nnue_features import FEATURE_SIZE, extract_features
from tdleaf_nnue_engine.nnue_model import NNUEModel


def _child_features(board: chess.Board, move: chess.Move) -> np.ndarray:
    b2 = board.copy(stack=False)
    b2.push(move)
    return extract_features(b2)


def collect_move_pairs(
    *,
    samples: int,
    seed: int,
    sunfish_movetime_ms: int,
    opening_plies: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
      x_best: [N, F] features after Sunfish-best move
      x_alt:  [N, F] features after random alternative move
      side:   [N] +1 if root side was White else -1 (for side-relative ranking)
    """
    rng = random.Random(seed)
    sunfish_repo_dir, sunfish_uci_entry = ensure_sunfish_checkout(DEFAULT_SUNFISH_DIR)

    best_rows: list[np.ndarray] = []
    alt_rows: list[np.ndarray] = []
    sides: list[float] = []

    with SunfishUCI(sunfish_uci_entry, sunfish_repo_dir=sunfish_repo_dir) as sunfish:
        attempts = 0
        while len(best_rows) < samples:
            attempts += 1
            if attempts > samples * 40:
                break

            board = chess.Board()
            moves_uci: list[str] = []
            for _ in range(opening_plies):
                if board.is_game_over(claim_draw=True):
                    break
                legal = list(board.legal_moves)
                if not legal:
                    break
                mv = rng.choice(legal)
                board.push(mv)
                moves_uci.append(mv.uci())

            if board.is_game_over(claim_draw=True):
                continue

            legal = list(board.legal_moves)
            if len(legal) < 2:
                continue

            side = 1.0 if board.turn == chess.WHITE else -1.0
            try:
                best_uci = sunfish.best_move(moves_uci, movetime_ms=sunfish_movetime_ms)
                best_mv = chess.Move.from_uci(best_uci)
            except Exception:
                continue
            if best_mv not in legal:
                continue

            alt_candidates = [m for m in legal if m != best_mv]
            if not alt_candidates:
                continue
            alt_mv = rng.choice(alt_candidates)

            best_rows.append(_child_features(board, best_mv))
            alt_rows.append(_child_features(board, alt_mv))
            sides.append(side)

            if len(best_rows) % 100 == 0:
                print(f"[distill] collected {len(best_rows)}/{samples} pairs", flush=True)

    if not best_rows:
        return (
            np.zeros((0, FEATURE_SIZE), dtype=np.float32),
            np.zeros((0, FEATURE_SIZE), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
        )

    return (
        np.stack(best_rows).astype(np.float32),
        np.stack(alt_rows).astype(np.float32),
        np.asarray(sides, dtype=np.float32),
    )


def train_distill(
    *,
    samples: int,
    epochs: int,
    batch_size: int,
    lr: float,
    seed: int,
    sunfish_movetime_ms: int,
    opening_plies: int,
    init_checkpoint: Path,
    out_checkpoint: Path,
    out_runtime: Path,
) -> None:
    torch.manual_seed(seed)

    x_best, x_alt, side = collect_move_pairs(
        samples=samples,
        seed=seed,
        sunfish_movetime_ms=sunfish_movetime_ms,
        opening_plies=opening_plies,
    )
    if x_best.shape[0] == 0:
        raise RuntimeError("No distillation samples collected from Sunfish.")

    model = NNUEModel(input_dim=FEATURE_SIZE)
    if init_checkpoint.exists():
        ck = torch.load(init_checkpoint, map_location="cpu")
        if "model_state_dict" in ck:
            model.load_state_dict(ck["model_state_dict"])
            print(f"[distill] loaded init checkpoint: {init_checkpoint}", flush=True)

    ds = TensorDataset(
        torch.from_numpy(x_best).float(),
        torch.from_numpy(x_alt).float(),
        torch.from_numpy(side).float(),
    )
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=False)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    model.train()
    for ep in range(1, epochs + 1):
        loss_sum = 0.0
        n = 0
        for xb, xa, s in dl:
            # Convert to side-relative values before comparing.
            vb = model(xb) * s
            va = model(xa) * s
            # Want vb > va; logistic ranking loss.
            diff = vb - va
            loss = F.softplus(-diff).mean()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            loss_sum += float(loss.detach().item()) * xb.shape[0]
            n += xb.shape[0]
        print(f"[distill] epoch {ep}/{epochs} loss={loss_sum/max(1,n):.4f}", flush=True)

    out_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "input_dim": FEATURE_SIZE,
            "hidden_dim": model.fc1.out_features,
        },
        out_checkpoint,
    )
    export_runtime_weights(str(out_checkpoint), str(out_runtime))
    print(f"[distill] wrote checkpoint: {out_checkpoint}", flush=True)
    print(f"[distill] wrote runtime: {out_runtime}", flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description="Move-only distillation from Sunfish.")
    p.add_argument("--samples", type=int, default=2000, help="Number of (best,alt) pairs.")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--sunfish-movetime-ms", type=int, default=35)
    p.add_argument("--opening-plies", type=int, default=4)
    p.add_argument("--init-checkpoint", default="tdleaf_nnue_engine/checkpoints/nnue_model.pt")
    p.add_argument("--out-checkpoint", default="tdleaf_nnue_engine/checkpoints/nnue_model_distilled.pt")
    p.add_argument("--out-runtime", default="tdleaf_nnue_engine/checkpoints/nnue_runtime_distilled.npz")
    args = p.parse_args()

    train_distill(
        samples=args.samples,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
        sunfish_movetime_ms=args.sunfish_movetime_ms,
        opening_plies=args.opening_plies,
        init_checkpoint=Path(args.init_checkpoint),
        out_checkpoint=Path(args.out_checkpoint),
        out_runtime=Path(args.out_runtime),
    )


if __name__ == "__main__":
    main()

