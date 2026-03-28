"""Multi-class cause-of-loss prediction head."""

from __future__ import annotations

from enum import IntEnum

import torch
import torch.nn as nn


class CauseOfLoss(IntEnum):
    """Canonical cause-of-loss categories."""
    DROUGHT = 0
    EXCESS_MOISTURE = 1
    COLD = 2
    HEAT = 3
    PRICE = 4
    OTHER = 5


NUM_CAUSES = len(CauseOfLoss)


class CauseClassifier(nn.Module):
    """Multi-class cause-of-loss prediction: 2-layer MLP -> softmax.

    Six classes: DROUGHT, EXCESS_MOISTURE, COLD, HEAT, PRICE, OTHER.

    Input:  h_graph  [batch, feature_dim]
    Output: logits   [batch, 6]   (raw, pre-softmax for CrossEntropyLoss)
    """

    def __init__(
        self,
        feature_dim: int = 64,
        hidden_dim: int = 32,
        num_classes: int = NUM_CAUSES,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.head = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
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
            logits: [batch, num_classes] raw logits (apply softmax externally).
        """
        return self.head(h)  # (B, num_classes)

    def predict_proba(self, h: torch.Tensor) -> torch.Tensor:
        """Return class probabilities via softmax."""
        return torch.softmax(self.forward(h), dim=-1)  # (B, num_classes)

    def predict(self, h: torch.Tensor) -> torch.Tensor:
        """Return predicted class indices."""
        return self.forward(h).argmax(dim=-1)  # (B,)
