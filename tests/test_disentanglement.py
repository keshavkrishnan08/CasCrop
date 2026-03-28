"""Tests for the adversarial disentanglement module.

Verifies gradient reversal, discriminator training, and
the overall disentanglement loss computation.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pytest
import torch


class TestGradientReversal:
    def test_forward_is_identity(self):
        from models.encoders.disentanglement import GradientReversalFunction

        x = torch.randn(16, 64, requires_grad=True)
        y = GradientReversalFunction.apply(x, 1.0)
        assert torch.allclose(x, y)

    def test_backward_reverses_gradient(self):
        from models.encoders.disentanglement import GradientReversalFunction

        x = torch.randn(16, 64, requires_grad=True)
        y = GradientReversalFunction.apply(x, 1.0)
        loss = y.sum()
        loss.backward()

        # Gradient should be -1 * ones (reversed from the +1 * ones forward grad)
        expected = -torch.ones_like(x)
        assert torch.allclose(x.grad, expected)


class TestDisentanglementModule:
    def test_loss_is_scalar(self):
        from models.encoders.disentanglement import DisentanglementModule

        module = DisentanglementModule(latent_dim=64, hidden_dim=128)
        z_bio = torch.randn(16, 64)
        z_econ = torch.randn(16, 64)

        loss = module(z_bio, z_econ)
        assert loss.ndim == 0
        assert loss.requires_grad

    def test_gradients_flow_to_encoder(self):
        from models.encoders.disentanglement import DisentanglementModule

        module = DisentanglementModule(latent_dim=64, hidden_dim=128)
        z_bio = torch.randn(16, 64, requires_grad=True)
        z_econ = torch.randn(16, 64, requires_grad=True)

        loss = module(z_bio, z_econ)
        loss.backward()

        # z_bio gets reversed gradients via the GRL
        assert z_bio.grad is not None
        # z_econ is intentionally detached in forward() to prevent
        # the discriminator loss from pushing z_econ around directly
        # (only z_bio should be affected via gradient reversal)

    def test_discriminator_accuracy_method(self):
        from models.encoders.disentanglement import DisentanglementModule

        module = DisentanglementModule(latent_dim=64, hidden_dim=128)
        z_bio = torch.randn(16, 64)
        z_econ = torch.randn(16, 64)

        acc = module.get_discriminator_accuracy(z_bio, z_econ)
        # Should return a float between 0 and some reasonable value
        assert isinstance(acc, float)
