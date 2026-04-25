"""Training entrypoint for TD-Leaf + NNUE-style model."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from tdleaf_nnue_engine.eval import Evaluator
from tdleaf_nnue_engine.nnue_features import FEATURE_SIZE
from tdleaf_nnue_engine.nnue_model import NNUEModel
from tdleaf_nnue_engine.search import Searcher
from tdleaf_nnue_engine.selfplay_tdleaf import SelfPlayConfig, generate_tdleaf_dataset


def train_model(
    output_checkpoint: str = "tdleaf_nnue_engine/checkpoints/nnue_model.pt",
    games: int = 2,
    max_plies: int = 36,
    depth: int = 2,
    epochs: int = 2,
    batch_size: int = 64,
    lr: float = 1e-3,
    seed: int = 0,
) -> Path:
    torch.manual_seed(seed)

    searcher = Searcher(evaluator=Evaluator())
    cfg = SelfPlayConfig(games=games, max_plies=max_plies, search_depth=depth, seed=seed)
    x_np, y_np = generate_tdleaf_dataset(searcher, cfg)
    if x_np.size == 0:
        # Fallback sample keeps pipeline runnable for tiny smoke configs.
        x_np = torch.zeros((1, FEATURE_SIZE), dtype=torch.float32).numpy()
        y_np = torch.zeros((1,), dtype=torch.float32).numpy()

    x = torch.from_numpy(x_np).float()
    y = torch.from_numpy(y_np).float()
    loader = DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=True)

    model = NNUEModel(input_dim=FEATURE_SIZE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    model.train()
    for _ in range(epochs):
        for xb, yb in loader:
            pred = model(xb)
            loss = criterion(pred, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()

    ckpt_path = Path(output_checkpoint)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "input_dim": FEATURE_SIZE,
            "hidden_dim": model.fc1.out_features,
        },
        ckpt_path,
    )
    return ckpt_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Train TD-Leaf NNUE model (CPU-friendly).")
    parser.add_argument("--output", default="tdleaf_nnue_engine/checkpoints/nnue_model.pt")
    parser.add_argument("--games", type=int, default=2)
    parser.add_argument("--max-plies", type=int, default=36)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    out = train_model(
        output_checkpoint=args.output,
        games=args.games,
        max_plies=args.max_plies,
        depth=args.depth,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
    )
    print(f"saved checkpoint: {out}")


if __name__ == "__main__":
    main()
