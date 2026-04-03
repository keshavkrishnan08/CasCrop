#!/usr/bin/env python3
"""Pre-compute Polarity-Routed Cascade features.

Novel pre-computation:
    1. Polarity-Routed Diffusion: negative shocks propagate through
       commodity-specific graphs (corn→corn neighbors); positive shocks
       through geographic proximity. The contagion TOPOLOGY changes
       based on shock polarity.
    2. Cascade Decay Signatures: ratio of diffusion at hop k vs hop k-1.
       Flat decay = systemic crisis. Steep decay = local noise.

Output: data/processed/features_cascade.parquet
    Original 35 features + 10 new:
        neg_1hop, neg_2hop, neg_3hop  (commodity-routed)
        pos_1hop, pos_2hop, pos_3hop  (geo-routed)
        decay_neg_2, decay_neg_3      (negative cascade persistence)
        decay_pos_2, decay_pos_3      (positive cascade persistence)
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
TOP_K = 20
EPS = 1e-8

COMMODITY_GRAPH_MAP = {
    "CORN": "adjacency_commodity_corn.npz",
    "SOYBEANS": "adjacency_commodity_soybeans.npz",
    "WHEAT": "adjacency_commodity_wheat.npz",
}


def row_normalize(A: sp.spmatrix) -> sp.spmatrix:
    row_sums = np.array(A.sum(axis=1)).flatten()
    row_sums[row_sums == 0] = 1.0
    return sp.diags(1.0 / row_sums) @ A


def sparsify_top_k(A: sp.spmatrix, k: int = 20) -> sp.spmatrix:
    """Keep only top-K neighbors per row (densify is fine for N=2130)."""
    dense = A.toarray()
    N = dense.shape[0]
    result = np.zeros_like(dense)
    for i in range(N):
        row = dense[i]
        nonzero = np.where(row > 0)[0]
        if len(nonzero) <= k:
            result[i, nonzero] = row[nonzero]
        else:
            topk = nonzero[np.argpartition(row[nonzero], -k)[-k:]]
            result[i, topk] = row[topk]
    return sp.csr_matrix(result)


def load_graphs(fips_to_idx: dict):
    """Load geo graph + per-commodity graphs (sparsified + normalized)."""
    N = len(fips_to_idx)

    # Geographic graph (already sparse)
    A_geo = sp.load_npz(GRAPH / "adjacency_geo.npz")
    A_geo_norm = row_normalize(A_geo)
    logger.info(f"Geo graph: {A_geo.nnz:,} edges")

    # Per-commodity graphs (dense → sparsify → normalize)
    commodity_graphs = {}
    for commodity, fname in COMMODITY_GRAPH_MAP.items():
        path = GRAPH / fname
        if path.exists():
            A_raw = sp.load_npz(path)
            A_sparse = sparsify_top_k(A_raw, TOP_K)
            A_norm = row_normalize(A_sparse)
            commodity_graphs[commodity] = A_norm
            logger.info(f"{commodity} graph: {A_raw.nnz:,} → {A_sparse.nnz:,} edges (top-{TOP_K})")
        else:
            logger.warning(f"Missing: {path}")
            commodity_graphs[commodity] = A_geo_norm  # fallback

    return A_geo_norm, commodity_graphs


def compute_cascade_features(features: pd.DataFrame, A_geo, commodity_graphs, fips_to_idx):
    """Compute polarity-routed diffusion + cascade decay signatures."""
    N = len(fips_to_idx)
    shock_col = "county_shock" if "county_shock" in features.columns else "price_change_1m"
    logger.info(f"Shock column: {shock_col}")

    # Pre-allocate output arrays
    col_names = []
    for k in range(1, N_HOPS + 1):
        col_names.extend([f"neg_{k}hop", f"pos_{k}hop"])
    col_names.extend(["decay_neg_2", "decay_neg_3", "decay_pos_2", "decay_pos_3"])

    out = {c: np.zeros(len(features), dtype=np.float32) for c in col_names}

    features = features.copy()
    features["_node_idx"] = features["fips"].map(fips_to_idx).fillna(-1).astype(int)

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

        # Split by polarity
        neg = np.maximum(-shock_vec, 0)  # magnitude of negative shocks
        pos = np.maximum(shock_vec, 0)   # magnitude of positive shocks

        # === POLARITY-ROUTED DIFFUSION ===
        # Negative shocks → commodity-specific graph (supply chain contagion)
        A_comm = commodity_graphs.get(commodity, A_geo)
        h_neg = neg.copy()
        neg_hops = []
        for k in range(1, N_HOPS + 1):
            h_neg = A_comm @ h_neg
            neg_hops.append(h_neg.copy())
            for j in range(len(node_ids)):
                if node_ids[j] >= 0:
                    out[f"neg_{k}hop"][idx[j]] = h_neg[node_ids[j]]

        # Positive shocks → geographic graph (proximity/substitution)
        h_pos = pos.copy()
        pos_hops = []
        for k in range(1, N_HOPS + 1):
            h_pos = A_geo @ h_pos
            pos_hops.append(h_pos.copy())
            for j in range(len(node_ids)):
                if node_ids[j] >= 0:
                    out[f"pos_{k}hop"][idx[j]] = h_pos[node_ids[j]]

        # === CASCADE DECAY SIGNATURES ===
        # Ratio of consecutive hops — only meaningful when denominator > threshold
        DECAY_THRESH = 1e-4
        for j in range(len(node_ids)):
            nid = node_ids[j]
            if nid < 0:
                continue
            n1, n2, n3 = neg_hops[0][nid], neg_hops[1][nid], neg_hops[2][nid]
            p1, p2, p3 = pos_hops[0][nid], pos_hops[1][nid], pos_hops[2][nid]
            out["decay_neg_2"][idx[j]] = (n2 / n1) if n1 > DECAY_THRESH else 0.0
            out["decay_neg_3"][idx[j]] = (n3 / n2) if n2 > DECAY_THRESH else 0.0
            out["decay_pos_2"][idx[j]] = (p2 / p1) if p1 > DECAY_THRESH else 0.0
            out["decay_pos_3"][idx[j]] = (p3 / p2) if p2 > DECAY_THRESH else 0.0

    features.drop(columns=["_node_idx"], inplace=True)
    for c in col_names:
        features[c] = out[c]

    return features, col_names


def main():
    logger.info("Loading data...")
    features = pd.read_parquet(DATA / "features_monthly.parquet")
    logger.info(f"Features: {features.shape}")

    with open(GRAPH / "fips_index.json") as f:
        fips_to_idx = json.load(f)

    logger.info("Loading graphs (polarity routing: commodity + geo)...")
    A_geo, commodity_graphs = load_graphs(fips_to_idx)

    logger.info("Computing polarity-routed cascade features...")
    enriched, cascade_cols = compute_cascade_features(
        features, A_geo, commodity_graphs, fips_to_idx
    )

    # === CROSS-COMMODITY INTERFERENCE ===
    logger.info("Computing cross-commodity interference...")
    price_col = "county_shock" if "county_shock" in enriched.columns else "price_change_1m"
    agg = enriched.groupby(["fips", "year", "month"])[price_col].agg(
        _total_sum="sum", _total_count="count", _total_min="min", _total_max="max"
    ).reset_index()
    enriched = enriched.merge(agg, on=["fips", "year", "month"])
    # Mean of OTHER commodities' prices in same county-month
    enriched["cross_other_mean"] = np.where(
        enriched["_total_count"] > 1,
        (enriched["_total_sum"] - enriched[price_col].fillna(0)) / (enriched["_total_count"] - 1),
        0.0,
    ).astype(np.float32)
    # Spread across commodities: high = commodity-specific, low = systemic
    enriched["cross_spread"] = (enriched["_total_max"] - enriched["_total_min"]).astype(np.float32)
    enriched.drop(columns=["_total_sum", "_total_count", "_total_min", "_total_max"], inplace=True)
    cascade_cols += ["cross_other_mean", "cross_spread"]
    logger.info(f"  cross_other_mean: mean={enriched['cross_other_mean'].mean():.6f}")
    logger.info(f"  cross_spread: mean={enriched['cross_spread'].mean():.6f}")

    # Update feature groups
    with open(DATA / "feature_groups_monthly.json") as f:
        groups = json.load(f)
    groups["cascade"] = cascade_cols

    out_path = DATA / "features_cascade.parquet"
    enriched.to_parquet(out_path, index=False)
    with open(DATA / "feature_groups_cascade.json", "w") as f:
        json.dump(groups, f, indent=2)

    logger.info(f"Saved: {out_path} ({enriched.shape})")
    logger.info(f"Cascade features ({len(cascade_cols)}): {cascade_cols}")

    for c in cascade_cols:
        v = enriched[c]
        logger.info(f"  {c}: mean={v.mean():.6f} std={v.std():.6f} nonzero={(v!=0).mean():.1%}")


if __name__ == "__main__":
    main()
