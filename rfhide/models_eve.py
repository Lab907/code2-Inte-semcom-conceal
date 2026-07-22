"""CNN Eve classifiers for transmitter identity evaluation."""

from __future__ import annotations

import torch
from torch import nn


class EveCNN(nn.Module):
    """1D CNN classifier for Tx identity from IQ signals.

    Input shape is ``[B, 2, N]`` with real and imaginary channels. The
    ``extract_embedding`` method returns the pooled 128-dimensional feature
    vector used by the final classifier.
    """

    def __init__(self, num_classes: int = 6) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(2, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(128, num_classes)

    def extract_embedding(self, signals: torch.Tensor) -> torch.Tensor:
        """Return pooled CNN embeddings with shape ``[B, 128]``."""
        if signals.ndim != 3 or signals.shape[1] != 2:
            raise ValueError(f"Expected signals shape [B, 2, N], got {tuple(signals.shape)}")
        features = self.features(signals)
        return self.pool(features).squeeze(-1)

    def forward(self, signals: torch.Tensor) -> torch.Tensor:
        """Return logits with shape ``[B, num_classes]``."""
        return self.classifier(self.extract_embedding(signals))


EveModel = EveCNN
