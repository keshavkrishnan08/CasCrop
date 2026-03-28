"""Unit tests for all CasCrop model architectures.

Tests forward pass, output shapes, gradient flow, and compatibility
for CasCrop and all baseline models.
"""

import pytest
import torch
import numpy as np


def make_dummy_config():
    """Create a minimal config for testing."""
    return {
        "model": {
            "biophysical_input_dim": 30,
            "economic_input_dim": 15,
            "historical_input_dim": 10,
            "latent_dim": 64,
            "hidden_dim": 128,
            "ecmp_heads": 4,
            "ecmp_layers": 2,
            "shock_embed_dim": 8,
            "dropout": 0.1,
            "disentangle_lambda": 0.1,
            "cause_loss_weight": 0.3,
            "num_cause_classes": 6,
        },
        "training": {
            "learning_rate": 0.001,
        },
    }


def make_dummy_batch(batch_size=16, num_nodes=32, num_edges=128):
    """Create a dummy batch for testing."""
    return {
        "x_bio": torch.randn(num_nodes, 30),
        "x_econ": torch.randn(num_nodes, 15),
        "x_hist": torch.randn(num_nodes, 10),
        "y_waste": torch.randint(0, 2, (num_nodes, 1)).float(),
        "y_cause": torch.randint(0, 6, (num_nodes,)),
        "edge_index": torch.randint(0, num_nodes, (2, num_edges)),
        "edge_attr": torch.randn(num_edges, 3),
        "price_shock": torch.randn(num_nodes, 1),
    }


class TestBiophysicalEncoder:
    def test_forward_shape(self):
        from src.models.encoders.biophysical_encoder import BiophysicalEncoder

        encoder = BiophysicalEncoder(input_dim=30, latent_dim=64, hidden_dim=128)
        x = torch.randn(16, 30)
        z = encoder(x)
        assert z.shape == (16, 64)

    def test_gradient_flow(self):
        from src.models.encoders.biophysical_encoder import BiophysicalEncoder

        encoder = BiophysicalEncoder(input_dim=30, latent_dim=64)
        x = torch.randn(16, 30)
        z = encoder(x)
        loss = z.sum()
        loss.backward()
        for p in encoder.parameters():
            assert p.grad is not None


class TestEconomicEncoder:
    def test_forward_shape(self):
        from src.models.encoders.economic_encoder import EconomicEncoder

        encoder = EconomicEncoder(input_dim=15, latent_dim=64, hidden_dim=64)
        x = torch.randn(16, 15)
        z = encoder(x)
        assert z.shape == (16, 64)


class TestDisentanglement:
    def test_forward(self):
        from src.models.encoders.disentanglement import DisentanglementModule

        module = DisentanglementModule(latent_dim=64, hidden_dim=128)
        z_bio = torch.randn(16, 64)
        z_econ = torch.randn(16, 64)
        loss = module(z_bio, z_econ)
        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0  # scalar


class TestECMP:
    def test_ecmp_layer_forward(self):
        from src.models.graph.ecmp import ECMPLayer

        layer = ECMPLayer(
            in_dim=64, out_dim=32, num_heads=4,
            dropout=0.0, shock_embed_dim=8, asymmetric=True,
        )
        x = torch.randn(32, 64)
        edge_index = torch.randint(0, 32, (2, 128))
        edge_attr = torch.randn(128, 3)
        price_shocks = torch.randn(32, 1)

        x_out, attn_weights = layer(x, edge_index, edge_attr, price_shocks)
        assert x_out.shape[0] == 32
        assert attn_weights.shape[0] == 128

    def test_ecmp_asymmetric_vs_symmetric(self):
        from src.models.graph.ecmp import ECMPLayer

        asym = ECMPLayer(in_dim=64, out_dim=32, num_heads=4, asymmetric=True)
        sym = ECMPLayer(in_dim=64, out_dim=32, num_heads=4, asymmetric=False)

        # Asymmetric should have more parameters (separate W_pos, W_neg)
        asym_params = sum(p.numel() for p in asym.parameters())
        sym_params = sum(p.numel() for p in sym.parameters())
        assert asym_params > sym_params

    def test_ecmp_stack_forward(self):
        from src.models.graph.ecmp import ECMPStack

        stack = ECMPStack(in_dim=64, hidden_dim=32, out_dim=64, num_heads=4)
        x = torch.randn(32, 64)
        edge_index = torch.randint(0, 32, (2, 128))
        edge_attr = torch.randn(128, 3)
        price_shocks = torch.randn(32, 1)

        x_out, attn_list = stack(x, edge_index, edge_attr, price_shocks)
        assert x_out.shape == (32, 64)
        assert isinstance(attn_list, list)

    def test_ecmp_gradient_flow(self):
        from src.models.graph.ecmp import ECMPLayer

        layer = ECMPLayer(in_dim=64, out_dim=32, num_heads=4, asymmetric=True)
        x = torch.randn(32, 64, requires_grad=True)
        edge_index = torch.randint(0, 32, (2, 128))
        edge_attr = torch.randn(128, 3)
        price_shocks = torch.randn(32, 1, requires_grad=True)

        x_out, _ = layer(x, edge_index, edge_attr, price_shocks)
        loss = x_out.sum()
        loss.backward()
        assert x.grad is not None
        assert price_shocks.grad is not None


class TestCasCrop:
    def test_full_forward(self):
        from src.models.cascrop import CasCrop

        config = make_dummy_config()
        model = CasCrop(config)
        batch = make_dummy_batch()

        outputs = model(batch)
        assert "waste_logits" in outputs
        assert "cause_logits" in outputs
        assert "z_bio" in outputs
        assert "z_econ" in outputs
        assert "attention_weights" in outputs

    def test_predict(self):
        from src.models.cascrop import CasCrop

        config = make_dummy_config()
        model = CasCrop(config)
        batch = make_dummy_batch()

        probs = model.predict_proba(batch)
        assert probs.shape[0] == batch["x_bio"].shape[0]
        assert (probs >= 0).all() and (probs <= 1).all()


class TestBaselines:
    def test_local_only(self):
        from src.models.baselines.local_only import LocalOnlyModel

        config = make_dummy_config()
        model = LocalOnlyModel(config)
        batch = make_dummy_batch()
        outputs = model(batch)
        assert "waste_logits" in outputs

    def test_local_econ(self):
        from src.models.baselines.local_econ import LocalEconModel

        config = make_dummy_config()
        model = LocalEconModel(config)
        batch = make_dummy_batch()
        outputs = model(batch)
        assert "waste_logits" in outputs

    def test_geo_gat(self):
        from src.models.baselines.geo_gat import GeoGATModel

        config = make_dummy_config()
        model = GeoGATModel(config)
        batch = make_dummy_batch()
        outputs = model(batch)
        assert "waste_logits" in outputs
        assert "attention_weights" in outputs

    def test_symmetric_ecmp(self):
        from src.models.baselines.symmetric_ecmp import SymmetricECMPModel

        config = make_dummy_config()
        model = SymmetricECMPModel(config)
        batch = make_dummy_batch()
        outputs = model(batch)
        assert "waste_logits" in outputs
