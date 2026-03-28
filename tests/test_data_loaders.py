"""Tests for data loading and processing modules."""

import pytest
import numpy as np
import pandas as pd


class TestDataUtils:
    def test_cause_of_loss_mapping(self):
        from src.data.utils import CAUSE_OF_LOSS_MAPPING

        # Drought codes
        assert CAUSE_OF_LOSS_MAPPING[2] == "DROUGHT"
        assert CAUSE_OF_LOSS_MAPPING[3] == "DROUGHT"

        # Excess moisture codes
        assert CAUSE_OF_LOSS_MAPPING[10] == "EXCESS_MOISTURE"
        assert CAUSE_OF_LOSS_MAPPING[11] == "EXCESS_MOISTURE"
        assert CAUSE_OF_LOSS_MAPPING[14] == "EXCESS_MOISTURE"

        # Cold codes
        assert CAUSE_OF_LOSS_MAPPING[15] == "COLD"
        assert CAUSE_OF_LOSS_MAPPING[16] == "COLD"
        assert CAUSE_OF_LOSS_MAPPING[17] == "COLD"

        # Heat codes
        assert CAUSE_OF_LOSS_MAPPING[36] == "HEAT"
        assert CAUSE_OF_LOSS_MAPPING[40] == "HEAT"

        # Price codes
        assert CAUSE_OF_LOSS_MAPPING[47] == "PRICE"
        assert CAUSE_OF_LOSS_MAPPING[48] == "PRICE"

    def test_commodity_codes(self):
        from src.data.utils import COMMODITY_CODES

        assert "CORN" in COMMODITY_CODES
        assert "SOYBEANS" in COMMODITY_CODES
        assert "WHEAT" in COMMODITY_CODES

    def test_validate_fips(self):
        from src.data.utils import validate_fips

        # Valid FIPS
        df = pd.DataFrame({"fips": ["01001", "06037", "17031"]})
        assert validate_fips(df, "fips")

        # Invalid FIPS (wrong length)
        df_bad = pd.DataFrame({"fips": ["1001", "637", "17031"]})
        # Should not raise but may return False or log warning


class TestDataMatcher:
    def test_construct_target_labels(self):
        from src.data.matcher import DataMatcher

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
        assert "cause_category" in labels.columns

        # Only indemnity > 10000 should be waste=1
        assert labels["waste"].sum() == 2  # 15000 and 50000

    def test_temporal_split_no_leakage(self):
        from src.data.matcher import DataMatcher

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
    def test_geographic_adjacency_symmetric(self):
        from src.data.graph_builder import GraphBuilder

        config = {"graph": {"sparsify_top_k": 20, "distance_sigma": "auto"}}
        builder = GraphBuilder(config)

        fips_list = ["01001", "01003", "01005", "01007", "01009"]
        adj = builder.build_geographic_adjacency(fips_list)

        # Adjacency should be symmetric
        diff = abs(adj - adj.T)
        assert diff.sum() < 1e-10

    def test_dynamic_graph_sparsity(self):
        from src.data.graph_builder import GraphBuilder

        config = {"graph": {"sparsify_top_k": 3, "distance_sigma": "auto"}}
        builder = GraphBuilder(config)

        # Each node should have at most top_k neighbors
        # Test with small graph
        import scipy.sparse as sp

        n = 10
        dense = np.random.rand(n, n)
        dense = (dense + dense.T) / 2
        np.fill_diagonal(dense, 0)

        geo_adj = sp.csr_matrix(dense)
        commodity_adj = sp.csr_matrix(dense * 0.5)
        transport_adj = sp.csr_matrix(dense * 0.3)

        edge_index, edge_attr = builder.build_dynamic_graph(
            geo_adj, commodity_adj, transport_adj, top_k=3
        )

        # Each node should have <= 3 outgoing edges
        for node in range(n):
            out_degree = (edge_index[0] == node).sum()
            assert out_degree <= 3
