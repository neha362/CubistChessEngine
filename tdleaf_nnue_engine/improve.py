"""
Practical training loop that actually improves the TDLeaf NNUE engine.

This is a hackathon-friendly pipeline:
  - generate self-play (or self-play vs a slightly weaker opponent) using alpha-beta
  - create TD targets from successive searched values (TD(0), optional TD(lambda))
  - train a compact NNUEModel in PyTorch from a replay buffer
  - export updated runtime weights for the engine/search to use

Run:
  python -m tdleaf_nnue_engine.improve --minutes 30
"""

from __future__ import annotations

import argparse
import random
import time
from dataclasses import dataclass
from pathlib import Path

import chess
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from tdleaf_nnue_engine.eval import Evaluator
from tdleaf_nnue_engine.export import export_runtime_weights
from tdleaf_nnue_engine.nnue_features import FEATURE_SIZE, extract_features
from tdleaf_nnue_engine.nnue_model import NNUEModel
from tdleaf_nnue_engine.search import Searcher


def _outcome_white(board: chess.Board) -> float:
    """Return terminal reward in {-1,0,+1} from White's perspective."""
    out = board.outcome(claim_draw=True)
    if out is None:
        return 0.0
    res = out.result()
    if res == "1-0":
        return 1.0
    if res == "0-1":
        return -1.0
    return 0.0


def _searched_value_white(searcher: Searcher, board: chess.Board, depth: int) -> float:
    """Search value in *centipawns* from White's perspective."""
    r = searcher.search(board, depth=depth)
    v_side = float(r.best_score)  # side-to-move perspective
    return v_side if board.turn == chess.WHITE else -v_side


@dataclass
class ReplayBuffer:
    capacity: int = 200_000

    def __post_init__(self) -> None:
        self.x: list[np.ndarray] = []
        self.y: list[float] = []

    def add(self, feat: np.ndarray, target: float) -> None:
        if len(self.x) >= self.capacity:
            # Drop oldest (simple FIFO).
            self.x.pop(0)
            self.y.pop(0)
        self.x.append(feat.astype(np.float32, copy=False))
        self.y.append(float(target))

    def to_tensors(self) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.x:
            x = torch.zeros((1, FEATURE_SIZE), dtype=torch.float32)
            y = torch.zeros((1,), dtype=torch.float32)
            return x, y
        x = torch.from_numpy(np.stack(self.x, axis=0)).float()
        y = torch.tensor(self.y, dtype=torch.float32)
        return x, y


def generate_td_samples(
    *,
    weights_npz: Path,
    games: int,
    max_plies: int,
    depth: int,
    rng: random.Random,
    gamma: float = 0.99,
    value_scale_cp: float = 300.0,
) -> list[tuple[np.ndarray, float]]:
    """
    Generate (features, target) samples using TD(0) bootstrapping on searched values.

    Targets are scaled to roughly "pawns" units: (cp / value_scale_cp).
    """
    evaluator = Evaluator(weights_path=str(weights_npz))
    searcher = Searcher(evaluator=evaluator)

    samples: list[tuple[np.ndarray, float]] = []
    for _g in range(games):
        board = chess.Board()
        # Randomize a couple plies to diversify openings.
        for _ in range(2):
            if board.is_game_over(claim_draw=True):
                break
            mv = rng.choice(list(board.legal_moves))
            board.push(mv)

        prev_feat: np.ndarray | None = None
        prev_vw: float | None = None

        for _ply in range(max_plies):
            if board.is_game_over(claim_draw=True):
                break

            vw = _searched_value_white(searcher, board, depth)
            feat = extract_features(board)

            # Bootstrap the previous state toward the current state's searched value.
            if prev_feat is not None and prev_vw is not None:
                tgt_cp = gamma * vw
                samples.append((prev_feat, float(np.clip(tgt_cp / value_scale_cp, -100.0, 100.0))))

            prev_feat = feat
            prev_vw = vw

            mv = searcher.best_move(board, depth=depth)
            # Exploration noise: occasionally pick a random legal move.
            if rng.random() < 0.10:
                mv = rng.choice(list(board.legal_moves))
            board.push(mv)

        # Terminal update for the last pending state.
        if prev_feat is not None:
            z = _outcome_white(board)
            # Terminal reward scaled to cp-ish range so it matters.
            terminal_cp = z * 30_000.0
            samples.append((prev_feat, float(np.clip(terminal_cp / value_scale_cp, -100.0, 100.0))))

    return samples


def train_from_buffer(
    *,
    model: NNUEModel,
    buffer: ReplayBuffer,
    batch_size: int,
    epochs: int,
    lr: float,
) -> float:
    x, y = buffer.to_tensors()
    loader = DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=True, drop_last=False)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.SmoothL1Loss(beta=1.0)

    model.train()
    last_loss = 0.0
    for _ in range(epochs):
        for xb, yb in loader:
            pred = model(xb)
            loss = loss_fn(pred, yb)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
            last_loss = float(loss.detach().cpu().item())
    return last_loss


def main() -> None:
    p = argparse.ArgumentParser(description="Improve TDLeaf NNUE engine via self-play TD(0) training.")
    p.add_argument("--minutes", type=float, default=10.0, help="Total wall-clock training time.")
    p.add_argument("--depth", type=int, default=2, help="Search depth used for targets and play.")
    p.add_argument("--games-per-iter", type=int, default=8, help="Self-play games generated per iteration.")
    p.add_argument("--max-plies", type=int, default=80, help="Max plies per self-play game.")
    p.add_argument("--epochs", type=int, default=1, help="Training epochs per iteration (over replay buffer).")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--buffer-capacity", type=int, default=80_000)
    p.add_argument(
        "--checkpoint",
        default="tdleaf_nnue_engine/checkpoints/nnue_model_improved.pt",
        help="PyTorch checkpoint output path.",
    )
    p.add_argument(
        "--runtime",
        default="tdleaf_nnue_engine/checkpoints/nnue_runtime_improved.npz",
        help="Runtime weights (.npz) output path.",
    )
    p.add_argument(
        "--init-checkpoint",
        default="tdleaf_nnue_engine/checkpoints/nnue_model.pt",
        help="Initial model checkpoint to start from (if exists).",
    )
    p.add_argument(
        "--init-runtime",
        default="tdleaf_nnue_engine/checkpoints/nnue_runtime.npz",
        help="Initial runtime weights for generating targets.",
    )
    args = p.parse_args()

    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)

    model = NNUEModel(input_dim=FEATURE_SIZE)
    init_ckpt = Path(args.init_checkpoint)
    if init_ckpt.exists():
        ck = torch.load(init_ckpt, map_location="cpu")
        if "model_state_dict" in ck:
            model.load_state_dict(ck["model_state_dict"])
            print(f"[improve] loaded init checkpoint: {init_ckpt}", flush=True)

    buffer = ReplayBuffer(capacity=int(args.buffer_capacity))
    current_runtime = Path(args.init_runtime)
    if not current_runtime.exists():
        # Fall back to material-only evaluator runtime by exporting current (random) model.
        tmp_ckpt = Path(args.checkpoint)
        tmp_ckpt.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model_state_dict": model.state_dict(), "input_dim": FEATURE_SIZE}, tmp_ckpt)
        current_runtime = export_runtime_weights(str(tmp_ckpt), str(Path(args.runtime))).resolve()

    t_end = time.time() + float(args.minutes) * 60.0
    it = 0
    while time.time() < t_end:
        it += 1
        # 1) Generate new data using the *current* runtime weights.
        samples = generate_td_samples(
            weights_npz=current_runtime,
            games=int(args.games_per_iter),
            max_plies=int(args.max_plies),
            depth=int(args.depth),
            rng=rng,
        )
        for feat, tgt in samples:
            buffer.add(feat, tgt)

        # 2) Train from replay buffer.
        last_loss = train_from_buffer(
            model=model,
            buffer=buffer,
            batch_size=int(args.batch_size),
            epochs=int(args.epochs),
            lr=float(args.lr),
        )

        # 3) Save checkpoint + export runtime weights for the next data iteration.
        ckpt_path = Path(args.checkpoint)
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model_state_dict": model.state_dict(), "input_dim": FEATURE_SIZE}, ckpt_path)
        runtime_path = Path(args.runtime)
        export_runtime_weights(str(ckpt_path), str(runtime_path))
        current_runtime = runtime_path

        # Progress print.
        print(
            f"[improve] iter={it}  new_samples={len(samples)}  buffer={len(buffer.x)}  "
            f"loss={last_loss:.4f}  runtime={runtime_path}",
            flush=True,
        )

    print(f"[improve] done. final runtime weights: {current_runtime}", flush=True)


if __name__ == "__main__":
    main()

