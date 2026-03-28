"""Tests for graph construction and dynamic graph building."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pytest
import torch
import numpy as np


class TestDynamicGraphConstructor:
    def test_learnable_weights(self):
        from models.graph.graph_construction import DynamicGraphConstructor

        constructor = DynamicGraphConstructor()

        # Weights should be learnable parameters
        params = list(constructor.parameters())
        assert len(params) > 0

    def test_weights_sum_to_one(self):
        from models.graph.graph_construction import DynamicGraphConstructor

        constructor = DynamicGraphConstructor()

        # After softmax, alpha + beta + gamma should = 1
        weights = constructor.weights
        assert abs(weights.sum().item() - 1.0) < 1e-5

    def test_sparsification(self):
        from models.graph.graph_construction import DynamicGraphConstructor

        constructor = DynamicGraphConstructor(top_k=5)

        # Create dense adjacency
        n = 20
        geo = torch.rand(n, n)
        geo = (geo + geo.T) / 2
        geo.fill_diagonal_(0)

        commodity = torch.rand(n, n)
        commodity = (commodity + commodity.T) / 2
        commodity.fill_diagonal_(0)

        transport = torch.rand(n, n)
        transport = (transport + transport.T) / 2
        transport.fill_diagonal_(0)

        edge_index, edge_weight, weights = constructor(geo, commodity, transport)

        # Each node should have at most top_k edges
        for node in range(n):
            # edge_index[1] is dst (the aggregating node), so count outgoing
            # The constructor builds edges from topk_idx (src) to arange (dst)
            degree = (edge_index[1] == node).sum().item()
            assert degree <= 5, f"Node {node} has degree {degree}, expected <= 5"
