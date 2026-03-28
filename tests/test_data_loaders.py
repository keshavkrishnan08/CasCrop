"""Tests for data loading and processing modules."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pytest
import numpy as np
import pandas as pd


class TestDataUtils:
    def test_cause_of_loss_mapping(self):
        from data.utils import CAUSE_OF_LOSS_MAPPING, map_cause_code

        # CAUSE_OF_LOSS_MAPPING maps category -> list of codes
        assert 2 in CAUSE_OF_LOSS_MAPPING["DROUGHT"]
        assert 3 in CAUSE_OF_LOSS_MAPPING["DROUGHT"]
        assert 10 in CAUSE_OF_LOSS_MAPPING["EXCESS_MOISTURE"]
        assert 15 in CAUSE_OF_LOSS_MAPPING["COLD"]
        assert 36 in CAUSE_OF_LOSS_MAPPING["HEAT"]
        assert 47 in CAUSE_OF_LOSS_MAPPING["PRICE"]

        # map_cause_code maps code -> category name
        assert map_cause_code(2) == "DROUGHT"
        assert map_cause_code(10) == "EXCESS_MOISTURE"
        assert map_cause_code(15) == "COLD"
        assert map_cause_code(36) == "HEAT"
        assert map_cause_code(47) == "PRICE"
        assert map_cause_code(999) == "OTHER"

    def test_commodity_codes(self):
        from data.utils import COMMODITY_CODES

        assert "CORN" in COMMODITY_CODES
        assert "SOYBEANS" in COMMODITY_CODES
        assert "WHEAT" in COMMODITY_CODES

    def test_validate_fips(self):
        from data.utils import validate_fips

        # Valid FIPS -- returns the same DataFrame
        df = pd.DataFrame({"fips": ["01001", "06037", "17031"]})
        result = validate_fips(df, "fips")
        assert len(result) == 3


class TestDataMatcher:
    def test_construct_target_labels(self):
        from data.matcher import DataMatcher

        config = {
            "data": {
                "waste_threshold": 10000,
                "commodities": ["CORN", "SOYBEANS", "WHEAT"],
                "train_years": list(range(2008, 2020)),
                "val_years": [2020, 2021],
                "test_years": [2022, 2023, 2024],
            }
        }
        matcher = DataMatcher(config)

        rma_df = pd.DataFrame({
            "fips": ["01001"] * 4,
            "commodity": ["CORN"] * 4,
            "year": [2020] * 4,
            "month": [6, 7, 8, 9],
            "indemnity_amount": [5000, 15000, 0, 50000],
            "cause_of_loss_code": [2, 47, 0, 15],
        })

        labels = matcher.construct_target_labels(rma_df, threshold=10000)
        assert "waste" in labels.columns
        assert "cause" in labels.columns

    def test_temporal_split_no_leakage(self):
        from data.matcher import DataMatcher

        config = {
            "data": {
                "waste_threshold": 10000,
                "commodities": ["CORN"],
                "train_years": [2018, 2019],
                "val_years": [2020],
                "test_years": [2021],
            }
        }
        matcher = DataMatcher(config)

        df = pd.DataFrame({
            "fips": ["01001"] * 48,
            "commodity": ["CORN"] * 48,
            "year": [2018]*12 + [2019]*12 + [2020]*12 + [2021]*12,
            "month": list(range(1, 13)) * 4,
        })

        splits = matcher.create_temporal_splits(df)

        train_years = set(df.iloc[splits["train"]]["year"])
        val_years = set(df.iloc[splits["val"]]["year"])
        test_years = set(df.iloc[splits["test"]]["year"])

        # No overlap between splits
        assert train_years & val_years == set()
        assert train_years & test_years == set()
        assert val_years & test_years == set()


class TestGraphBuilder:
    def test_dynamic_graph_sparsity(self):
        from data.graph_builder import GraphBuilder

        config = {"graph": {"sparsify_top_k": 3, "distance_sigma": "auto"}}
        builder = GraphBuilder(config)

        import scipy.sparse as sp

        n = 10
        dense = np.random.rand(n, n)
        dense = (dense + dense.T) / 2
        np.fill_diagonal(dense, 0)

        geo_adj = sp.csr_matrix(dense)
        commodity_adj = sp.csr_matrix(dense * 0.5)
        transport_adj = sp.csr_matrix(dense * 0.3)

        # Set up FIPS index manually
        builder._fips_list = [f"{i:05d}" for i in range(n)]
        builder._fips_to_idx = {f: i for i, f in enumerate(builder._fips_list)}

        edge_index, edge_attr = builder.build_dynamic_graph(
            geo_adj, commodity_adj, transport_adj, top_k=3
        )

        # Each node should have <= 3 outgoing edges
        for node in range(n):
            out_degree = (edge_index[0] == node).sum()
            assert out_degree <= 3
