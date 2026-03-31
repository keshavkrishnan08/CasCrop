"""Temporal Symmetric ECMP Baseline: Same as TemporalCasCrop but symmetric.

Uses a single W_sym instead of separate W_pos/W_neg for shock embedding.
Isolates the value of asymmetric shock conditioning.
"""

import torch
import torch.nn as nn
from typing import Any, Dict, Optional

from ..encoders.biophysical_encoder import BiophysicalEncoder
from ..encoders.economic_encoder import EconomicEncoder
from ..encoders.disentanglement import DisentanglementModule
from ..graph.ecmp import ECMPLayer
from ..heads.waste_classifier import WasteClassifier
from ..heads.cause_classifier import CauseClassifier


class TemporalSymmetricModel(nn.Module):
    """Temporal cascade with SYMMETRIC shock conditioning.

    Same architecture as TemporalCasCrop but asymmetric=False.
    Tests whether the W_pos/W_neg split matters.
    """

    def __init__(
        self,
        bio_input_dim: int = 17,
        econ_input_dim: int = 8,
        hist_dim: int = 3,
        latent_dim: int = 64,
        hidden_dim: int = 64,
        num_heads: int = 4,
        shock_embed_dim: int = 32,
        dropout: float = 0.3,
        **kwargs,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.bio_encoder = BiophysicalEncoder(
            input_dim=bio_input_dim, hidden_dim=latent_dim * 2,
            latent_dim=latent_dim, dropout=dropout,
        )
        self.econ_encoder = EconomicEncoder(
            input_dim=econ_input_dim, hidden_dim=latent_dim,
            latent_dim=latent_dim, dropout=dropout,
        )
        self.disentangle = DisentanglementModule(
            latent_dim=latent_dim, hidden_dim=latent_dim, lambda_=1.0,
        )
        node_dim = latent_dim + latent_dim + hist_dim

        # SYMMETRIC ECMP — single W_sym instead of W_pos/W_neg
        self.ecmp = ECMPLayer(
            in_dim=node_dim, out_dim=hidden_dim, num_heads=num_heads,
            dropout=dropout, shock_embed_dim=shock_embed_dim,
            asymmetric=False,  # THE KEY DIFFERENCE
            concat_heads=False,
        )
        self.ecmp_norm = nn.LayerNorm(hidden_dim)
        self.gru = nn.GRUCell(hidden_dim, hidden_dim)
        self.waste_head = WasteClassifier(feature_dim=hidden_dim, hidden_dim=32, dropout=dropout)
        self.cause_head = CauseClassifier(feature_dim=hidden_dim, hidden_dim=32, dropout=dropout)

    def forward(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        if "x_bio_seq" in batch:
            return self._forward_temporal(batch)

        x_bio = batch["x_bio"]
        x_econ = batch["x_econ"]
        N = x_bio.size(0)
        z_bio = self.bio_encoder(x_bio)
        z_econ = self.econ_encoder(x_econ)
        dis_loss = self.disentangle(z_bio, z_econ)
        h = torch.cat([z_bio, z_econ, batch["x_hist"]], dim=-1)
        ecmp_out, attn = self.ecmp(h, batch["edge_index"], batch.get("edge_attr"), batch["price_shocks"])
        ecmp_out = self.ecmp_norm(ecmp_out)
        state = self.gru(ecmp_out, torch.zeros(N, self.hidden_dim, device=x_bio.device))

        return {
            "waste_logits": self.waste_head(state),
            "cause_logits": self.cause_head(state),
            "attention_weights": attn,
            "disentangle_loss": dis_loss,
        }

    def _forward_temporal(self, batch):
        x_bio_seq = batch["x_bio_seq"]
        x_econ_seq = batch["x_econ_seq"]
        x_hist = batch["x_hist"]
        edge_index = batch["edge_index"]
        edge_attr = batch.get("edge_attr")
        shocks_seq = batch["price_shocks_seq"]
        T, N, _ = x_bio_seq.shape
        device = x_bio_seq.device
        state = torch.zeros(N, self.hidden_dim, device=device)
        total_dis = torch.tensor(0.0, device=device)

        for t in range(T):
            z_bio = self.bio_encoder(x_bio_seq[t])
            z_econ = self.econ_encoder(x_econ_seq[t])
            total_dis = total_dis + self.disentangle(z_bio, z_econ)
            h = torch.cat([z_bio, z_econ, x_hist], dim=-1)
            ecmp_out, attn = self.ecmp(h, edge_index, edge_attr, shocks_seq[t])
            ecmp_out = self.ecmp_norm(ecmp_out)
            state = self.gru(ecmp_out, state)

        return {
            "waste_logits": self.waste_head(state),
            "cause_logits": self.cause_head(state),
            "attention_weights": attn,
            "disentangle_loss": total_dis / T,
        }
