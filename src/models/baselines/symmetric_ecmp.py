"""Row 4 ablation: symmetric ECMP (no positive/negative split).

Identical to CasCrop except phi(delta_p) = W_sym * delta_p instead of
the asymmetric W_pos * max(delta_p, 0) + W_neg * min(delta_p, 0).
Tests whether the asymmetric treatment of shocks matters.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from ..encoders.biophysical_encoder import BiophysicalEncoder
from ..encoders.economic_encoder import EconomicEncoder
from ..encoders.disentanglement import DisentanglementModule
from ..graph.ecmp import ECMPStack
from ..heads.waste_classifier import WasteClassifier
from ..heads.cause_classifier import CauseClassifier


class SymmetricECMPModel(nn.Module):
    """CasCrop variant with symmetric shock embedding.

    The only architectural difference from CasCrop:
        ECMPStack(..., asymmetric=False)

    This means phi(delta_p) = W_sym * delta_p — a single linear transform
    rather than the split positive/negative pathways.
    """

    def __init__(
        self,
        bio_input_dim: int = 30,
        econ_input_dim: int = 15,
        hist_dim: int = 8,
        latent_dim: int = 64,
        bio_hidden_dim: int = 128,
        econ_hidden_dim: int = 64,
        graph_hidden_dim: int = 64,
        num_heads: int = 4,
        dropout: float = 0.3,
        shock_embed_dim: int = 8,
        mu: float = 0.3,
        lambda_: float = 0.1,
        edge_feat_dim: int = 0,
    ) -> None:
        super().__init__()
        self.mu = mu
        self.lambda_ = lambda_
        self.latent_dim = latent_dim

        self.bio_encoder = BiophysicalEncoder(
            input_dim=bio_input_dim,
            hidden_dim=bio_hidden_dim,
            latent_dim=latent_dim,
            dropout=dropout,
        )
        self.econ_encoder = EconomicEncoder(
            input_dim=econ_input_dim,
            hidden_dim=econ_hidden_dim,
            latent_dim=latent_dim,
            dropout=dropout,
        )
        self.disentangle = DisentanglementModule(
            latent_dim=latent_dim,
            hidden_dim=latent_dim,
            lambda_=1.0,
        )

        graph_in_dim = latent_dim + latent_dim + hist_dim
        self.ecmp_stack = ECMPStack(
            in_dim=graph_in_dim,
            hidden_dim=graph_hidden_dim,
            out_dim=latent_dim,
            num_heads=num_heads,
            dropout=dropout,
            shock_embed_dim=shock_embed_dim,
            asymmetric=False,  # <-- THE KEY DIFFERENCE
            edge_feat_dim=edge_feat_dim,
        )

        self.waste_head = WasteClassifier(
            feature_dim=latent_dim, hidden_dim=32, dropout=dropout,
        )
        self.cause_head = CauseClassifier(
            feature_dim=latent_dim, hidden_dim=32, dropout=dropout,
        )

    # ------------------------------------------------------------------
    def forward(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """
        Expected batch keys:
            x_bio, x_econ, x_hist, edge_index, price_shocks.
            Optional: edge_attr.
        """
        x_bio = batch["x_bio"]
        x_econ = batch["x_econ"]
        x_hist = batch["x_hist"]
        edge_index = batch["edge_index"]
        edge_attr = batch.get("edge_attr")
        price_shocks = batch["price_shocks"]

        z_bio = self.bio_encoder(x_bio)
        z_econ = self.econ_encoder(x_econ)

        disentangle_loss = self.disentangle(z_bio, z_econ)

        h = torch.cat([z_bio, z_econ, x_hist], dim=-1)
        h_graph, attn = self.ecmp_stack(h, edge_index, edge_attr, price_shocks)

        waste_logits = self.waste_head(h_graph)
        cause_logits = self.cause_head(h_graph)

        return {
            "waste_logits": waste_logits,
            "cause_logits": cause_logits,
            "z_bio": z_bio,
            "z_econ": z_econ,
            "attention_weights": attn,
            "disentangle_loss": disentangle_loss,
        }

    # ------------------------------------------------------------------
    def compute_loss(
        self,
        outputs: Dict[str, torch.Tensor],
        waste_targets: torch.Tensor,
        cause_targets: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        waste_loss = nn.functional.binary_cross_entropy_with_logits(
            outputs["waste_logits"],
            waste_targets.view_as(outputs["waste_logits"]).float(),
        )
        cause_loss = torch.tensor(0.0, device=waste_loss.device)
        if cause_targets is not None:
            cause_loss = nn.functional.cross_entropy(
                outputs["cause_logits"], cause_targets.long(),
            )
        total_loss = (
            waste_loss
            + self.mu * cause_loss
            + self.lambda_ * outputs["disentangle_loss"]
        )
        return {
            "total_loss": total_loss,
            "waste_loss": waste_loss,
            "cause_loss": cause_loss,
            "disentangle_loss": outputs["disentangle_loss"],
        }

    # ------------------------------------------------------------------
    @torch.no_grad()
    def predict(self, batch: Dict[str, Any]) -> torch.Tensor:
        self.eval()
        return (torch.sigmoid(self.forward(batch)["waste_logits"]) >= 0.5).long()

    @torch.no_grad()
    def predict_proba(self, batch: Dict[str, Any]) -> torch.Tensor:
        self.eval()
        return torch.sigmoid(self.forward(batch)["waste_logits"])

    @torch.no_grad()
    def get_attention_weights(self, batch: Dict[str, Any]) -> torch.Tensor:
        self.eval()
        return self.forward(batch)["attention_weights"]
