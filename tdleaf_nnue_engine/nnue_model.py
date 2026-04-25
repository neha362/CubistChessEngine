"""PyTorch model for NNUE-style position scoring."""

from __future__ import annotations

import torch
from torch import nn


class ClippedReLU(nn.Module):
    """ReLU clipped to a fixed ceiling, common in NNUE pipelines."""

    def __init__(self, cap: float = 1.0) -> None:
        super().__init__()
        self.cap = cap

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.clamp(torch.relu(x), max=self.cap)


class NNUEModel(nn.Module):
    """Compact fully-connected network suitable for CPU training/inference."""

    def __init__(self, input_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.act1 = ClippedReLU(1.0)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.act2 = ClippedReLU(1.0)
        self.out = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act1(self.fc1(x))
        x = self.act2(self.fc2(x))
        return self.out(x).squeeze(-1)
