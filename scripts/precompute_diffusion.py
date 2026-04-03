#!/usr/bin/env python3
"""Pre-compute asymmetric multi-hop shock diffusion features.

For each county-commodity-month sample, computes how price shocks
propagate through the agricultural network at 1, 2, and 3 hops,
split into positive and negative channels.

This replaces online GNN message passing with pre-computed graph features —
no mini-batch subgraph construction needed at training time.

Output: data/processed/features_diffusion.parquet
    Original features + 6 new columns:
        diff_1hop_pos, diff_1hop_neg,
        diff_2hop_pos, diff_2hop_neg,
        diff_3hop_pos, diff_3hop_neg
"""

import json, logging, sys
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.sparse as sp

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DATA = Path("data/processed")
GRAPH = Path("data/graphs")
N_HOPS = 3


def build_normalized_adjacency():
    """Load combined graph and build row-normalized sparse adjacency."""
    graph = np.load(GRAPH / "combined_graph.npz")
    edge_index = graph["edge_index"]
    edge_weight = graph.get("edge_weight", np.ones(edge_index.shape[1]))

    with open(GRAPH / "fips_index.json") as f:
        fips_to_idx = json.load(f)
    N = len(fips_to_idx)

    A = sp.coo_matrix(
        (edge_weight, (edge_index[0], edge_index[1])),
        shape=(N, N),
    ).tocsr()

    # Row-normalize: each row sums to 1 (diffusion conserves mass)
    row_sums = np.array(A.sum(axis=1)).flatten()
    row_sums[row_sums == 0] = 1.0
    D_inv = sp.diags(1.0 / row_sums)
    A_norm = D_inv @ A

    logger.info(f"Adjacency: {N} nodes, {A.nnz} edges, avg degree {A.nnz / N:.1f}")
    return A_norm, fips_to_idx


def compute_diffusion_features(features: pd.DataFrame, A_norm, fips_to_idx: dict, n_hops: int = 3):
    """Compute multi-hop asymmetric diffusion for all samples.

    For each (commodity, year, month) group:
        1. Build county-level shock vector from price_change_1m
        2. Split into positive (price rise) and negative (price drop)
        3. Diffuse each channel through A_norm at hops 1..K
        4. Assign diffused values back to each sample
    """
    N = len(fips_to_idx)
    shock_col = "county_shock" if "county_shock" in features.columns else "price_change_1m"
    logger.info(f"Using shock column: {shock_col}")

    # Pre-allocate output columns
    diff_cols = {}
    for k in range(1, n_hops + 1):
        diff_cols[f"diff_{k}hop_pos"] = np.zeros(len(features), dtype=np.float32)
        diff_cols[f"diff_{k}hop_neg"] = np.zeros(len(features), dtype=np.float32)

    # Map FIPS to adjacency index
    features = features.copy()
    features["_node_idx"] = features["fips"].map(fips_to_idx).fillna(-1).astype(int)

    # Group by time step and commodity
    groups = features.groupby(["commodity", "year", "month"])
    total = len(groups)

    for i, ((commodity, year, month), group) in enumerate(groups):
        if i % 200 == 0:
            logger.info(f"  [{i}/{total}] {commodity} {year}-{month:02d}")

        idx = group.index.values
        node_ids = group["_node_idx"].values
        shocks = group[shock_col].fillna(0).values.astype(np.float32)

        # Build county-level shock vector
        shock_vec = np.zeros(N, dtype=np.float32)
        for j in range(len(node_ids)):
            if node_ids[j] >= 0:
                shock_vec[node_ids[j]] = shocks[j]

        # Split into positive/negative
        pos = np.maximum(shock_vec, 0)
        neg = np.maximum(-shock_vec, 0)  # magnitude of negative shocks

        # Multi-hop diffusion
        h_pos, h_neg = pos, neg
        for k in range(1, n_hops + 1):
            h_pos = A_norm @ h_pos
            h_neg = A_norm @ h_neg

            # Assign back to samples
            for j in range(len(node_ids)):
                nid = node_ids[j]
                if nid >= 0:
                    diff_cols[f"diff_{k}hop_pos"][idx[j]] = h_pos[nid]
                    diff_cols[f"diff_{k}hop_neg"][idx[j]] = h_neg[nid]

    features.drop(columns=["_node_idx"], inplace=True)

    # Add diffusion columns
    for col, vals in diff_cols.items():
        features[col] = vals

    return features


def main():
    logger.info("Loading data...")
    features = pd.read_parquet(DATA / "features_monthly.parquet")
    logger.info(f"Features: {features.shape}")

    logger.info("Building adjacency...")
    A_norm, fips_to_idx = build_normalized_adjacency()

    logger.info("Computing diffusion features...")
    enriched = compute_diffusion_features(features, A_norm, fips_to_idx, N_HOPS)

    # Update feature groups
    with open(DATA / "feature_groups_monthly.json") as f:
        groups = json.load(f)
    groups["diffusion"] = [f"diff_{k}hop_{p}" for k in range(1, N_HOPS + 1) for p in ("pos", "neg")]

    # Save
    out_path = DATA / "features_diffusion.parquet"
    enriched.to_parquet(out_path, index=False)
    with open(DATA / "feature_groups_diffusion.json", "w") as f:
        json.dump(groups, f, indent=2)

    logger.info(f"Saved: {out_path} ({enriched.shape})")
    logger.info(f"Diffusion features: {groups['diffusion']}")

    # Quick stats
    for col in groups["diffusion"]:
        vals = enriched[col]
        logger.info(f"  {col}: mean={vals.mean():.6f}, std={vals.std():.6f}, "
                     f"nonzero={(vals != 0).mean():.1%}")


if __name__ == "__main__":
    main()
