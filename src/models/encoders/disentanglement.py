"""Disentanglement module: adversarial training to decorrelate z_bio and z_econ."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.autograd import Function


# ======================================================================
# Gradient Reversal Layer
# ======================================================================

class GradientReversalFunction(Function):
    """Reverses gradients during backward pass, leaves forward pass unchanged.

    Used for end-to-end adversarial training: the discriminator learns to
    predict z_econ from z_bio, while gradient reversal forces the encoder
    to *prevent* that prediction from succeeding.
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, lambda_: float) -> torch.Tensor:
        ctx.lambda_ = lambda_
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return -ctx.lambda_ * grad_output, None


class GradientReversalLayer(nn.Module):
    """Wraps GradientReversalFunction as a standard nn.Module."""

    def __init__(self, lambda_: float = 1.0) -> None:
        super().__init__()
        self.lambda_ = lambda_

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return GradientReversalFunction.apply(x, self.lambda_)

    def set_lambda(self, lambda_: float) -> None:
        """Adjust reversal strength (useful for curriculum scheduling)."""
        self.lambda_ = lambda_


# ======================================================================
# Disentanglement Module
# ======================================================================

class DisentanglementModule(nn.Module):
    """Forces z_bio and z_econ to capture independent information.

    A discriminator D tries to reconstruct z_econ from z_bio.  A gradient
    reversal layer sits between the biophysical encoder and D so that
    minimising D's reconstruction loss simultaneously *maximises* it from
    the encoder's perspective.

    Loss:
        L_disentangle = ||D(GRL(z_bio)) - z_econ||^2

    During training the encoder receives *reversed* gradients from this
    loss, pushing z_bio away from information already present in z_econ.
    """

    def __init__(
        self,
        latent_dim: int = 64,
        hidden_dim: int = 64,
        lambda_: float = 1.0,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.lambda_ = lambda_

        # Gradient reversal sits before the discriminator
        self.grl = GradientReversalLayer(lambda_=lambda_)

        # Discriminator: predict z_econ from z_bio
        self.discriminator = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, latent_dim),
        )

        self._init_weights()

    # ------------------------------------------------------------------
    def _init_weights(self) -> None:
        for m in self.discriminator.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # ------------------------------------------------------------------
    def forward(
        self, z_bio: torch.Tensor, z_econ: torch.Tensor
    ) -> torch.Tensor:
        """Compute disentanglement loss.

        Args:
            z_bio:  [batch, latent_dim] biophysical embeddings.
            z_econ: [batch, latent_dim] economic embeddings.

        Returns:
            disentangle_loss: scalar MSE between D(GRL(z_bio)) and z_econ.
        """
        # z_bio shape: (B, latent_dim)
        z_bio_rev = self.grl(z_bio)                   # (B, latent_dim)  — gradient reversed
        z_econ_pred = self.discriminator(z_bio_rev)    # (B, latent_dim)
        loss = torch.nn.functional.mse_loss(z_econ_pred, z_econ.detach())
        return loss  # scalar

    # ------------------------------------------------------------------
    @torch.no_grad()
    def get_discriminator_accuracy(
        self, z_bio: torch.Tensor, z_econ: torch.Tensor
    ) -> float:
        """Measure how well the discriminator can reconstruct z_econ.

        Returns the cosine similarity between predicted and true z_econ,
        averaged over the batch.  Values near 0 mean good disentanglement;
        values near 1 mean the discriminator still succeeds.
        """
        z_econ_pred = self.discriminator(z_bio)  # (B, latent_dim)
        cos = torch.nn.functional.cosine_similarity(z_econ_pred, z_econ, dim=-1)
        return cos.mean().item()

    # ------------------------------------------------------------------
    def set_lambda(self, lambda_: float) -> None:
        """Adjust gradient reversal strength."""
        self.lambda_ = lambda_
        self.grl.set_lambda(lambda_)
