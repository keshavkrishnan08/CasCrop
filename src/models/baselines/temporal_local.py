"""Temporal Local-Only Baseline: GRU on local features, no graph.

Same temporal processing as TemporalCasCrop but without the graph.
Uses a GRU over monthly biophysical features to predict waste.
This is the temporal equivalent of Row 1 (local only).
"""

import torch
import torch.nn as nn
from typing import Any, Dict, Optional

from ..encoders.biophysical_encoder import BiophysicalEncoder
from ..heads.waste_classifier import WasteClassifier
from ..heads.cause_classifier import CauseClassifier


class TemporalLocalModel(nn.Module):
    """GRU on local features only — no graph, no economic features.

    Temporal baseline: shows the value of graph contagion modeling
    beyond just having temporal information.
    """

    def __init__(
        self,
        bio_input_dim: int = 17,
        econ_input_dim: int = 8,
        hist_dim: int = 3,
        latent_dim: int = 64,
        hidden_dim: int = 64,
        dropout: float = 0.3,
        **kwargs,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.bio_encoder = BiophysicalEncoder(
            input_dim=bio_input_dim, hidden_dim=latent_dim * 2,
            latent_dim=latent_dim, dropout=dropout,
        )
        # GRU on encoded bio features
        self.gru = nn.GRUCell(latent_dim + hist_dim, hidden_dim)
        self.waste_head = WasteClassifier(feature_dim=hidden_dim, hidden_dim=32, dropout=dropout)
        self.cause_head = CauseClassifier(feature_dim=hidden_dim, hidden_dim=32, dropout=dropout)

    def forward(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        if "x_bio_seq" in batch:
            return self._forward_temporal(batch)

        x_bio = batch["x_bio"]
        N = x_bio.size(0)
        z_bio = self.bio_encoder(x_bio)
        h = torch.cat([z_bio, batch["x_hist"]], dim=-1)
        state = self.gru(h, torch.zeros(N, self.hidden_dim, device=x_bio.device))
        return {
            "waste_logits": self.waste_head(state),
            "cause_logits": self.cause_head(state),
            "attention_weights": torch.zeros(1, 4),
            "disentangle_loss": torch.tensor(0.0, device=x_bio.device),
        }

    def _forward_temporal(self, batch):
        x_bio_seq = batch["x_bio_seq"]
        x_hist = batch["x_hist"]
        T, N, _ = x_bio_seq.shape
        device = x_bio_seq.device
        state = torch.zeros(N, self.hidden_dim, device=device)

        for t in range(T):
            z_bio = self.bio_encoder(x_bio_seq[t])
            h = torch.cat([z_bio, x_hist], dim=-1)
            state = self.gru(h, state)

        return {
            "waste_logits": self.waste_head(state),
            "cause_logits": self.cause_head(state),
            "attention_weights": torch.zeros(1, 4, device=device),
            "disentangle_loss": torch.tensor(0.0, device=device),
        }
