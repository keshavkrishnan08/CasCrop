"""CasCrop: full cascading crop-waste prediction architecture.

Pipeline:
    1. BiophysicalEncoder(x_bio)  -> z_bio
    2. EconomicEncoder(x_econ)    -> z_econ
    3. DisentanglementModule(z_bio, z_econ) -> L_disentangle
    4. Node features h = [z_bio || z_econ || z_hist]
    5. ECMPStack(h, graph, price_shocks) -> h_graph
    6. WasteClassifier(h_graph)   -> waste logits
    7. CauseClassifier(h_graph)   -> cause logits

Total loss = L_waste + mu * L_cause + lambda_ * L_disentangle
    mu = 0.3,  lambda_ = 0.1
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from .encoders.biophysical_encoder import BiophysicalEncoder
from .encoders.economic_encoder import EconomicEncoder
from .encoders.disentanglement import DisentanglementModule
from .graph.ecmp import ECMPStack
from .heads.waste_classifier import WasteClassifier
from .heads.cause_classifier import CauseClassifier


class CasCrop(nn.Module):
    """Full CasCrop architecture with disentangled encoders and ECMP graph.

    Args:
        bio_input_dim:   biophysical feature count (~30).
        econ_input_dim:  economic feature count (~15).
        hist_dim:        historical feature dimension (e.g. lagged waste indicator).
        latent_dim:      encoder output / graph hidden dimension (64).
        bio_hidden_dim:  biophysical encoder hidden width (128).
        econ_hidden_dim: economic encoder hidden width (64).
        graph_hidden_dim: ECMP intermediate dimension (64).
        num_heads:       attention heads in ECMP (4).
        dropout:         global dropout rate (0.3).
        shock_embed_dim: price-shock embedding size (8).
        mu:              cause-loss weight (0.3).
        lambda_:         disentanglement-loss weight (0.1).
        edge_feat_dim:   edge feature dimension for ECMP (0 = none).
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
        self.hist_dim = hist_dim

        # --- Encoders ---
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

        # --- Disentanglement ---
        self.disentangle = DisentanglementModule(
            latent_dim=latent_dim,
            hidden_dim=latent_dim,
            lambda_=1.0,
        )

        # --- Graph ---
        graph_in_dim = latent_dim + latent_dim + hist_dim  # z_bio || z_econ || z_hist
        self.ecmp_stack = ECMPStack(
            in_dim=graph_in_dim,
            hidden_dim=graph_hidden_dim,
            out_dim=latent_dim,
            num_heads=num_heads,
            dropout=dropout,
            shock_embed_dim=shock_embed_dim,
            asymmetric=True,
            edge_feat_dim=edge_feat_dim,
        )

        # --- Prediction heads ---
        self.waste_head = WasteClassifier(
            feature_dim=latent_dim,
            hidden_dim=32,
            dropout=dropout,
        )
        self.cause_head = CauseClassifier(
            feature_dim=latent_dim,
            hidden_dim=32,
            dropout=dropout,
        )

    # ------------------------------------------------------------------
    def forward(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Full forward pass.

        Expected batch keys:
            x_bio:         [N, bio_input_dim]
            x_econ:        [N, econ_input_dim]
            x_hist:        [N, hist_dim]
            edge_index:    [2, E]
            edge_attr:     [E, edge_feat_dim] or None
            price_shocks:  [N, 1]

        Returns dict with:
            waste_logits:      [N, 1]
            cause_logits:      [N, 6]
            z_bio:             [N, latent_dim]
            z_econ:            [N, latent_dim]
            attention_weights: [E, num_heads]
            disentangle_loss:  scalar
        """
        x_bio = batch["x_bio"]              # (N, bio_input_dim)
        x_econ = batch["x_econ"]            # (N, econ_input_dim)
        x_hist = batch["x_hist"]            # (N, hist_dim)
        edge_index = batch["edge_index"]    # (2, E)
        edge_attr = batch.get("edge_attr")  # (E, F) or None
        price_shocks = batch["price_shocks"]  # (N, 1)

        # 1-2. Encode
        z_bio = self.bio_encoder(x_bio)     # (N, latent_dim)
        z_econ = self.econ_encoder(x_econ)  # (N, latent_dim)

        # 3. Disentanglement loss
        disentangle_loss = self.disentangle(z_bio, z_econ)  # scalar

        # 4. Concatenate node features
        h = torch.cat([z_bio, z_econ, x_hist], dim=-1)  # (N, 2*latent + hist)

        # 5. Graph message passing
        h_graph, attn_weights = self.ecmp_stack(
            h, edge_index, edge_attr, price_shocks,
        )  # (N, latent_dim), (E, H)

        # 6-7. Prediction heads
        waste_logits = self.waste_head(h_graph)   # (N, 1)
        cause_logits = self.cause_head(h_graph)   # (N, 6)

        return {
            "waste_logits": waste_logits,
            "cause_logits": cause_logits,
            "z_bio": z_bio,
            "z_econ": z_econ,
            "attention_weights": attn_weights,
            "disentangle_loss": disentangle_loss,
        }

    # ------------------------------------------------------------------
    def compute_loss(
        self,
        outputs: Dict[str, torch.Tensor],
        waste_targets: torch.Tensor,
        cause_targets: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Compute combined training loss.

        Args:
            outputs:        dict from forward().
            waste_targets:  [N, 1] or [N] binary labels.
            cause_targets:  [N] integer class labels (optional).

        Returns:
            dict with total_loss, waste_loss, cause_loss, disentangle_loss.
        """
        waste_loss = nn.functional.binary_cross_entropy_with_logits(
            outputs["waste_logits"],
            waste_targets.view_as(outputs["waste_logits"]).float(),
        )

        cause_loss = torch.tensor(0.0, device=waste_loss.device)
        if cause_targets is not None:
            cause_loss = nn.functional.cross_entropy(
                outputs["cause_logits"], cause_targets.long(),
            )

        disentangle_loss = outputs["disentangle_loss"]

        total_loss = waste_loss + self.mu * cause_loss + self.lambda_ * disentangle_loss

        return {
            "total_loss": total_loss,
            "waste_loss": waste_loss,
            "cause_loss": cause_loss,
            "disentangle_loss": disentangle_loss,
        }

    # ------------------------------------------------------------------
    @torch.no_grad()
    def predict(self, batch: Dict[str, Any]) -> torch.Tensor:
        """Return binary waste predictions (0 or 1)."""
        self.eval()
        outputs = self.forward(batch)
        probs = torch.sigmoid(outputs["waste_logits"])  # (N, 1)
        return (probs >= 0.5).long()

    @torch.no_grad()
    def predict_proba(self, batch: Dict[str, Any]) -> torch.Tensor:
        """Return waste probabilities in [0, 1]."""
        self.eval()
        outputs = self.forward(batch)
        return torch.sigmoid(outputs["waste_logits"])  # (N, 1)

    @torch.no_grad()
    def get_attention_weights(self, batch: Dict[str, Any]) -> torch.Tensor:
        """Return ECMP attention weight matrix for visualisation."""
        self.eval()
        outputs = self.forward(batch)
        return outputs["attention_weights"]  # (E, H)
