"""Export PyTorch checkpoint to lightweight NumPy runtime format."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch


def export_runtime_weights(
    checkpoint_path: str = "tdleaf_nnue_engine/checkpoints/nnue_model.pt",
    output_path: str = "tdleaf_nnue_engine/checkpoints/nnue_runtime.npz",
) -> Path:
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    state = ckpt["model_state_dict"]
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out,
        input_dim=np.array([ckpt["input_dim"]], dtype=np.int32),
        fc1_weight=state["fc1.weight"].cpu().numpy(),
        fc1_bias=state["fc1.bias"].cpu().numpy(),
        fc2_weight=state["fc2.weight"].cpu().numpy(),
        fc2_bias=state["fc2.bias"].cpu().numpy(),
        out_weight=state["out.weight"].cpu().numpy(),
        out_bias=state["out.bias"].cpu().numpy(),
    )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Export NNUE checkpoint to runtime npz.")
    parser.add_argument("--checkpoint", default="tdleaf_nnue_engine/checkpoints/nnue_model.pt")
    parser.add_argument("--output", default="tdleaf_nnue_engine/checkpoints/nnue_runtime.npz")
    args = parser.parse_args()
    out = export_runtime_weights(args.checkpoint, args.output)
    print(f"exported runtime weights: {out}")


if __name__ == "__main__":
    main()
