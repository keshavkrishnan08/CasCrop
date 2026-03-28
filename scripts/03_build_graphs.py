#!/usr/bin/env python3
"""Script 03: Construct county graphs for ECMP.

Builds geographic adjacency, commodity connectivity, and
transportation network matrices. Creates dynamic monthly
graph snapshots as PyTorch Geometric Data objects.

Estimated runtime: ~30 minutes.
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def load_config(config_path: str = "configs/default.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Build county graphs for CasCrop")
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    processed_dir = Path(config["paths"]["processed_data"])
    graph_dir = Path(config["paths"]["graphs"])
    graph_dir.mkdir(parents=True, exist_ok=True)

    import pandas as pd
    from scipy import sparse
    from src.data.graph_builder import GraphBuilder

    builder = GraphBuilder(config)

    # Load processed data for graph construction
    logger.info("Loading processed data")
    features_df = pd.read_parquet(processed_dir / "features.parquet")

    # Step 1: Geographic adjacency
    logger.info("Step 1: Building geographic adjacency matrix")
    fips_list = sorted(features_df["fips"].unique())
    geo_adj = builder.build_geographic_adjacency(fips_list)
    sparse.save_npz(graph_dir / "adjacency_geo.npz", geo_adj)
    logger.info(f"  Geographic graph: {geo_adj.nnz} edges")

    # Step 2: Commodity connectivity (per crop)
    logger.info("Step 2: Building commodity connectivity matrices")
    for crop in config["data"]["commodities"]:
        crop_data = features_df[features_df["commodity"] == crop]
        commodity_adj = builder.build_commodity_connectivity(crop_data, crop)
        sparse.save_npz(graph_dir / f"adjacency_commodity_{crop.lower()}.npz", commodity_adj)
        logger.info(f"  {crop} commodity graph: {commodity_adj.nnz} edges")

    # Step 3: Transportation connectivity
    logger.info("Step 3: Building transportation connectivity matrix")
    transport_adj = builder.build_transport_connectivity(fips_list)
    sparse.save_npz(graph_dir / "adjacency_transport.npz", transport_adj)
    logger.info(f"  Transport graph: {transport_adj.nnz} edges")

    # Step 4: Dynamic graph snapshots
    logger.info("Step 4: Building dynamic monthly graph snapshots")
    dynamic_dir = graph_dir / "dynamic_graphs"
    dynamic_dir.mkdir(parents=True, exist_ok=True)

    years = range(
        min(config["data"]["train_years"]),
        max(config["data"]["test_years"]) + 1,
    )
    for year in years:
        for month in range(1, 13):
            month_data = features_df[
                (features_df["year"] == year) & (features_df["month"] == month)
            ]
            if len(month_data) == 0:
                continue

            for crop in config["data"]["commodities"]:
                crop_month_data = month_data[month_data["commodity"] == crop]
                if len(crop_month_data) == 0:
                    continue

                edge_index, edge_attr = builder.build_dynamic_graph(
                    geo_adj=geo_adj,
                    commodity_adj=sparse.load_npz(
                        graph_dir / f"adjacency_commodity_{crop.lower()}.npz"
                    ),
                    transport_adj=transport_adj,
                    top_k=config["graph"]["sparsify_top_k"],
                )

                np.savez(
                    dynamic_dir / f"graph_{crop.lower()}_{year}_{month:02d}.npz",
                    edge_index=edge_index,
                    edge_attr=edge_attr,
                )

    # Step 5: Graph statistics
    logger.info("Step 5: Computing graph statistics")
    stats = builder.compute_graph_statistics(
        edge_index=np.array([[0], [1]]),  # placeholder
        num_nodes=len(fips_list),
    )

    stats_lines = [
        "# Graph Statistics",
        f"\n## Summary",
        f"- Number of counties (nodes): {len(fips_list):,}",
        f"- Geographic edges: {geo_adj.nnz:,}",
        f"- Transport edges: {transport_adj.nnz:,}",
        f"- Top-K neighbors: {config['graph']['sparsify_top_k']}",
    ]

    with open(graph_dir / "graph_statistics.md", "w") as f:
        f.write("\n".join(stats_lines))

    logger.info("Graph construction complete")


if __name__ == "__main__":
    main()
