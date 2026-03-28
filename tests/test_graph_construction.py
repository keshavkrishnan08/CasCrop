"""Tests for graph construction and dynamic graph building."""

import pytest
import torch
import numpy as np


class TestDynamicGraphConstructor:
    def test_learnable_weights(self):
        from src.models.graph.graph_construction import DynamicGraphConstructor

        constructor = DynamicGraphConstructor()

        # Weights should be learnable parameters
        params = list(constructor.parameters())
        assert len(params) > 0

    def test_weights_sum_to_one(self):
        from src.models.graph.graph_construction import DynamicGraphConstructor

        constructor = DynamicGraphConstructor()

        # After softmax, alpha + beta + gamma should = 1
        weights = constructor.get_normalized_weights()
        assert abs(sum(weights.values()) - 1.0) < 1e-5

    def test_sparsification(self):
        from src.models.graph.graph_construction import DynamicGraphConstructor

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

        edge_index, edge_attr = constructor(geo, commodity, transport)

        # Each node should have at most top_k edges
        for node in range(n):
            degree = (edge_index[0] == node).sum().item()
            assert degree <= 5, f"Node {node} has degree {degree}, expected <= 5"
