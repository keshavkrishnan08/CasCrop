"""Binary waste prediction head."""

from __future__ import annotations

import torch
import torch.nn as nn


class WasteClassifier(nn.Module):
    """Binary waste prediction: 2-layer MLP -> sigmoid.

    Input:  h_graph  [batch, feature_dim]
    Output: logits   [batch, 1]   (raw, pre-sigmoid for BCEWithLogitsLoss)
    """

    def __init__(
        self,
        feature_dim: int = 64,
        hidden_dim: int = 32,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h: [batch, feature_dim] graph-enriched node embeddings.

        Returns:
            logits: [batch, 1] raw logits (apply sigmoid externally).
        """
        return self.head(h)  # (B, 1)

    def predict_proba(self, h: torch.Tensor) -> torch.Tensor:
        """Return waste probability in [0, 1]."""
        return torch.sigmoid(self.forward(h))  # (B, 1)
