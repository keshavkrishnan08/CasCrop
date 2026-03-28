"""Economic encoder: commodity prices, costs, market signals -> z_econ."""

from __future__ import annotations

import torch
import torch.nn as nn


class EconomicEncoder(nn.Module):
    """Encodes commodity prices, input costs, and market signals into z_econ.

    Input:  x_econ  in R^(batch x d_econ)   where d_econ ~ 15
    Output: z_econ  in R^(batch x latent_dim) where latent_dim = 64

    Three-layer MLP mirroring BiophysicalEncoder but with a smaller hidden
    dimension (64) because the economic feature space is lower-dimensional.
    """

    def __init__(
        self,
        input_dim: int = 15,
        hidden_dim: int = 64,
        latent_dim: int = 64,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        self.encoder = nn.Sequential(
            # Layer 1: input_dim -> hidden_dim
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            # Layer 2: hidden_dim -> hidden_dim
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            # Layer 3: hidden_dim -> latent_dim (raw, no activation)
            nn.Linear(hidden_dim, latent_dim),
        )

        self._init_weights()

    # ------------------------------------------------------------------
    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # ------------------------------------------------------------------
    def forward(self, x_econ: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_econ: [batch, input_dim] economic feature vector.

        Returns:
            z_econ: [batch, latent_dim] latent embedding.
        """
        # x_econ shape: (B, input_dim)
        z_econ = self.encoder(x_econ)  # (B, latent_dim)
        return z_econ
