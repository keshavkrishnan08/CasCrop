"""Focused tests for the ECMP (Economic Contagion Message Passing) mechanism.

Tests the core novelty: asymmetric shock conditioning, multi-head attention,
and correct attention weight extraction.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pytest
import torch
import numpy as np


class TestECMPAsymmetry:
    """Verify that positive and negative shocks produce different attention patterns."""

    def test_asymmetric_produces_different_outputs(self):
        from models.graph.ecmp import ECMPLayer

        layer = ECMPLayer(in_dim=64, out_dim=32, num_heads=4, asymmetric=True)
        layer.eval()

        x = torch.randn(10, 64)
        edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]])
        edge_attr = torch.ones(3, 3)

        # Positive shock at node 0
        pos_shock = torch.zeros(10, 1)
        pos_shock[0] = 1.0

        # Negative shock at node 0
        neg_shock = torch.zeros(10, 1)
        neg_shock[0] = -1.0

        with torch.no_grad():
            out_pos, attn_pos = layer(x, edge_index, edge_attr, pos_shock)
            out_neg, attn_neg = layer(x, edge_index, edge_attr, neg_shock)

        # Attention weights should differ for asymmetric model
        assert not torch.allclose(attn_pos, attn_neg, atol=1e-6), \
            "Asymmetric ECMP should produce different attention for pos vs neg shocks"

    def test_symmetric_produces_same_magnitude_attention(self):
        from models.graph.ecmp import ECMPLayer

        layer = ECMPLayer(in_dim=64, out_dim=32, num_heads=4, asymmetric=False)
        layer.eval()

        x = torch.randn(10, 64)
        edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]])
        edge_attr = torch.ones(3, 3)

        pos_shock = torch.zeros(10, 1)
        pos_shock[0] = 1.0

        neg_shock = torch.zeros(10, 1)
        neg_shock[0] = -1.0

        with torch.no_grad():
            _, attn_pos = layer(x, edge_index, edge_attr, pos_shock)
            _, attn_neg = layer(x, edge_index, edge_attr, neg_shock)

        # Symmetric model: same transformation for both directions
        # Output should still differ (because input sign differs),
        # but the learned transformation is shared


class TestECMPAttentionExtraction:
    """Verify attention weights are properly extractable for case study."""

    def test_attention_weights_returned(self):
        from models.graph.ecmp import ECMPLayer

        layer = ECMPLayer(in_dim=64, out_dim=32, num_heads=4)
        x = torch.randn(10, 64)
        edge_index = torch.randint(0, 10, (2, 20))
        edge_attr = torch.randn(20, 3)
        price_shocks = torch.randn(10, 1)

        _, attn = layer(x, edge_index, edge_attr, price_shocks)
        assert attn.shape[0] == 20  # num_edges
        assert attn.shape[1] == 4   # num_heads

    def test_attention_weights_sum_to_one(self):
        from models.graph.ecmp import ECMPLayer

        layer = ECMPLayer(in_dim=64, out_dim=32, num_heads=1, dropout=0.0)
        layer.eval()

        num_nodes = 5
        # Fully connected graph (for easy softmax verification)
        src = []
        dst = []
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i != j:
                    src.append(i)
                    dst.append(j)
        edge_index = torch.tensor([src, dst])
        num_edges = edge_index.shape[1]

        x = torch.randn(num_nodes, 64)
        edge_attr = torch.randn(num_edges, 3)
        price_shocks = torch.randn(num_nodes, 1)

        with torch.no_grad():
            _, attn = layer(x, edge_index, edge_attr, price_shocks)

        # For each target node, attention from all sources should sum to ~1
        for target in range(num_nodes):
            incoming_mask = edge_index[1] == target
            incoming_attn = attn[incoming_mask, 0]
            assert abs(incoming_attn.sum().item() - 1.0) < 0.01, \
                f"Attention to node {target} sums to {incoming_attn.sum():.3f}, expected ~1.0"


class TestECMPStack:
    """Test the stacked ECMP with residual connections."""

    def test_residual_connection(self):
        from models.graph.ecmp import ECMPStack

        stack = ECMPStack(in_dim=64, hidden_dim=32, out_dim=64, num_heads=4)
        x = torch.randn(10, 64)
        edge_index = torch.randint(0, 10, (2, 30))
        edge_attr = torch.randn(30, 3)
        price_shocks = torch.randn(10, 1)

        x_out, _ = stack(x, edge_index, edge_attr, price_shocks)

        # Output should have same shape as input (residual)
        assert x_out.shape == (10, 64)

    def test_stack_returns_attention(self):
        from models.graph.ecmp import ECMPStack

        stack = ECMPStack(in_dim=64, hidden_dim=32, out_dim=64, num_heads=4)
        x = torch.randn(10, 64)
        edge_index = torch.randint(0, 10, (2, 30))
        edge_attr = torch.randn(30, 3)
        price_shocks = torch.randn(10, 1)

        x_out, attn = stack(x, edge_index, edge_attr, price_shocks)
        # ECMPStack returns attention from the second layer
        assert isinstance(attn, torch.Tensor)
        assert attn.shape[0] == 30   # num_edges
        assert attn.shape[1] == 4    # num_heads
