"""Biophysical encoder: satellite imagery, weather, soil, vegetation -> z_bio."""

from __future__ import annotations

import torch
import torch.nn as nn


class BiophysicalEncoder(nn.Module):
    """Encodes satellite imagery, weather, soil, and vegetation features into z_bio.

    Input:  x_bio  in R^(batch x d_bio)   where d_bio ~ 30
    Output: z_bio  in R^(batch x latent_dim) where latent_dim = 64

    Three-layer MLP with BatchNorm and ReLU.  The final layer produces raw
    latent embeddings (no activation) so downstream modules can impose their
    own constraints.
    """

    def __init__(
        self,
        input_dim: int = 30,
        hidden_dim: int = 128,
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
    def forward(self, x_bio: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_bio: [batch, input_dim] biophysical feature vector.

        Returns:
            z_bio: [batch, latent_dim] latent embedding.
        """
        # x_bio shape: (B, input_dim)
        z_bio = self.encoder(x_bio)  # (B, latent_dim)
        return z_bio
