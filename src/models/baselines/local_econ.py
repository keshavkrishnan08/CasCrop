"""Row 2 baseline: MLP on biophysical + economic features, no graph.

Tests whether economic features improve prediction even without spatial
propagation through the graph.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from ..encoders.biophysical_encoder import BiophysicalEncoder
from ..encoders.economic_encoder import EconomicEncoder
from ..heads.waste_classifier import WasteClassifier
from ..heads.cause_classifier import CauseClassifier


class LocalEconModel(nn.Module):
    """Biophysical + economic MLP baseline (no graph).

    Pipeline:
        BiophysicalEncoder(x_bio)  -> z_bio
        EconomicEncoder(x_econ)    -> z_econ
        concat([z_bio, z_econ])    -> h
        WasteClassifier(h)         -> waste logits
        CauseClassifier(h)         -> cause logits
    """

    def __init__(
        self,
        bio_input_dim: int = 30,
        econ_input_dim: int = 15,
        bio_hidden_dim: int = 128,
        econ_hidden_dim: int = 64,
        latent_dim: int = 64,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
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

        # Projection from concatenated space to classifier-sized input
        combined_dim = 2 * latent_dim
        self.projector = nn.Sequential(
            nn.Linear(combined_dim, latent_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
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
            x_bio:  [N, bio_input_dim]
            x_econ: [N, econ_input_dim]

        Returns dict with waste_logits, cause_logits, z_bio, z_econ.
        """
        x_bio = batch["x_bio"]                          # (N, bio_input_dim)
        x_econ = batch["x_econ"]                         # (N, econ_input_dim)

        z_bio = self.bio_encoder(x_bio)                  # (N, latent_dim)
        z_econ = self.econ_encoder(x_econ)               # (N, latent_dim)

        h = torch.cat([z_bio, z_econ], dim=-1)           # (N, 2*latent_dim)
        h = self.projector(h)                            # (N, latent_dim)

        waste_logits = self.waste_head(h)                # (N, 1)
        cause_logits = self.cause_head(h)                # (N, 6)

        return {
            "waste_logits": waste_logits,
            "cause_logits": cause_logits,
            "z_bio": z_bio,
            "z_econ": z_econ,
        }

    # ------------------------------------------------------------------
    def compute_loss(
        self,
        outputs: Dict[str, torch.Tensor],
        waste_targets: torch.Tensor,
        cause_targets: Optional[torch.Tensor] = None,
        mu: float = 0.3,
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
        total_loss = waste_loss + mu * cause_loss
        return {
            "total_loss": total_loss,
            "waste_loss": waste_loss,
            "cause_loss": cause_loss,
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
    def get_attention_weights(self, batch: Dict[str, Any]) -> None:
        """No graph — returns None."""
        return None
