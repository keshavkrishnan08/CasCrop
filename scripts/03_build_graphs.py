#!/usr/bin/env python3
"""Script 03: Build county graphs from actual Census adjacency data.

Produces:
  - data/graphs/adjacency_geo.npz       (geographic adjacency)
  - data/graphs/adjacency_commodity.npz  (per-crop commodity similarity)
  - data/graphs/fips_index.json          (FIPS → node index mapping)
  - data/graphs/graph_stats.json         (graph statistics)

Estimated runtime: ~2 minutes.
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.spatial.distance import pdist, squareform

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

PROC = Path("data/processed")
RAW = Path("data/raw")
GRAPH_DIR = Path("data/graphs")


def load_adjacency() -> dict[str, set[str]]:
    """Parse Census county adjacency file into a dict: fips → set of neighbor fips."""
    path = RAW / "geographic" / "county_adjacency.txt"
    adj = {}
    current_fips = None

    with open(path, encoding="latin-1") as f:
        for line in f:
            parts = line.strip().split("\t")
            # Lines with county name in first field start a new county
            # Lines with empty first field are continuation neighbors
            fips_fields = [p.strip().strip('"') for p in parts if p.strip().strip('"').isdigit() and len(p.strip().strip('"')) == 5]

            if len(parts) >= 4 and parts[0].strip():
                # New county line: has name + fips + neighbor name + neighbor fips
                if len(fips_fields) >= 1:
                    current_fips = fips_fields[0]
                    if current_fips not in adj:
                        adj[current_fips] = set()
                if len(fips_fields) >= 2:
                    neighbor = fips_fields[1]
                    if neighbor != current_fips:
                        adj[current_fips].add(neighbor)
            elif current_fips and fips_fields:
                # Continuation line: just neighbor
                for neighbor in fips_fields:
                    if neighbor != current_fips:
                        adj[current_fips].add(neighbor)

    logger.info(f"  Parsed adjacency: {len(adj)} counties, "
                f"{sum(len(v) for v in adj.values())} edges")
    return adj


def build_geographic_matrix(
    fips_list: list[str],
    adjacency: dict[str, set[str]],
    centroids: pd.DataFrame,
    sigma: float = None,
) -> sparse.csr_matrix:
    """Build geographic adjacency matrix with distance-weighted edges.

    w_ij = exp(-d_ij / sigma) if counties are adjacent, else 0.
    """
    n = len(fips_list)
    fips_to_idx = {f: i for i, f in enumerate(fips_list)}

    # Build centroid lookup
    centroid_map = {}
    for _, row in centroids.iterrows():
        centroid_map[row["fips"]] = (row["latitude"], row["longitude"])

    rows, cols, vals = [], [], []

    for fips_i, neighbors in adjacency.items():
        if fips_i not in fips_to_idx:
            continue
        i = fips_to_idx[fips_i]

        for fips_j in neighbors:
            if fips_j not in fips_to_idx:
                continue
            j = fips_to_idx[fips_j]

            # Distance weight
            if fips_i in centroid_map and fips_j in centroid_map:
                lat1, lon1 = centroid_map[fips_i]
                lat2, lon2 = centroid_map[fips_j]
                d = _haversine(lat1, lon1, lat2, lon2)
            else:
                d = 100  # default 100 km

            rows.append(i)
            cols.append(j)
            vals.append(d)

    # Convert distances to weights
    vals = np.array(vals)
    if sigma is None:
        sigma = np.median(vals[vals > 0]) if len(vals) > 0 else 100
    weights = np.exp(-vals / sigma)

    matrix = sparse.csr_matrix((weights, (rows, cols)), shape=(n, n))
    # Make symmetric
    matrix = (matrix + matrix.T) / 2

    logger.info(f"  Geographic matrix: {matrix.nnz} edges, sigma={sigma:.1f} km")
    return matrix


def _haversine(lat1, lon1, lat2, lon2):
    """Haversine distance in km."""
    R = 6371
    lat1, lat2, lon1, lon2 = map(np.radians, [lat1, lat2, lon1, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return R * 2 * np.arcsin(np.sqrt(a))


def build_commodity_matrix(
    fips_list: list[str],
    features: pd.DataFrame,
    crop: str,
) -> sparse.csr_matrix:
    """Build commodity similarity matrix for a crop.

    Similarity based on production profile (area, yield) cosine similarity.
    """
    n = len(fips_list)
    fips_to_idx = {f: i for i, f in enumerate(fips_list)}

    # Get crop features per county (averaged over years)
    crop_data = features[features["commodity"] == crop].copy()
    if len(crop_data) == 0:
        return sparse.csr_matrix((n, n))

    county_profiles = crop_data.groupby("fips").agg({
        "yield_value": "mean",
        "area_planted": "mean",
        "production": "mean",
    }).reset_index()

    # Only counties in our fips_list
    county_profiles = county_profiles[county_profiles["fips"].isin(fips_to_idx)]

    if len(county_profiles) < 2:
        return sparse.csr_matrix((n, n))

    # Normalize features
    profile_cols = ["yield_value", "area_planted", "production"]
    X = county_profiles[profile_cols].fillna(0).values
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1
    X_norm = X / norms

    # Cosine similarity
    sim = X_norm @ X_norm.T

    # Map to full matrix
    county_fips = county_profiles["fips"].values
    rows, cols, vals = [], [], []
    for i_local in range(len(county_fips)):
        for j_local in range(len(county_fips)):
            if i_local == j_local:
                continue
            if sim[i_local, j_local] < 0.1:
                continue  # sparsify
            i_global = fips_to_idx.get(county_fips[i_local])
            j_global = fips_to_idx.get(county_fips[j_local])
            if i_global is not None and j_global is not None:
                rows.append(i_global)
                cols.append(j_global)
                vals.append(sim[i_local, j_local])

    matrix = sparse.csr_matrix((vals, (rows, cols)), shape=(n, n))
    logger.info(f"  Commodity matrix ({crop}): {matrix.nnz} edges")
    return matrix


def sparsify_to_top_k(matrix: sparse.csr_matrix, k: int = 20) -> tuple:
    """Sparsify to top-K neighbors per node. Returns edge_index, edge_weight."""
    n = matrix.shape[0]
    rows, cols, vals = [], [], []

    dense = matrix.toarray()
    for i in range(n):
        neighbors = dense[i]
        if neighbors.sum() == 0:
            continue
        top_k_idx = np.argsort(neighbors)[-k:]
        for j in top_k_idx:
            if neighbors[j] > 0:
                rows.append(i)
                cols.append(j)
                vals.append(neighbors[j])

    edge_index = np.array([rows, cols])
    edge_weight = np.array(vals)
    return edge_index, edge_weight


def main():
    parser = argparse.ArgumentParser(description="Build CasCrop county graphs")
    parser.add_argument("--top-k", type=int, default=20)
    args = parser.parse_args()

    GRAPH_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("CasCrop Graph Construction")
    logger.info("=" * 60)

    # Load processed features to get FIPS list
    features = pd.read_parquet(PROC / "features.parquet")
    fips_list = sorted(features["fips"].unique())
    logger.info(f"Building graphs for {len(fips_list)} counties")

    # FIPS index mapping
    fips_to_idx = {f: i for i, f in enumerate(fips_list)}
    with open(GRAPH_DIR / "fips_index.json", "w") as f:
        json.dump(fips_to_idx, f)

    # Load centroids
    gaz = pd.read_csv(
        RAW / "geographic" / "2023_Gaz_counties_national.txt",
        sep="\t", dtype={"GEOID": str},
    )
    gaz.columns = gaz.columns.str.strip()
    gaz.rename(columns={"GEOID": "fips", "INTPTLAT": "latitude", "INTPTLONG": "longitude"}, inplace=True)
    gaz["latitude"] = pd.to_numeric(gaz["latitude"].astype(str).str.strip(), errors="coerce")
    gaz["longitude"] = pd.to_numeric(gaz["longitude"].astype(str).str.strip(), errors="coerce")

    # Step 1: Geographic adjacency
    logger.info("\n[1/3] Building geographic adjacency...")
    adjacency = load_adjacency()
    geo_matrix = build_geographic_matrix(fips_list, adjacency, gaz)
    sparse.save_npz(GRAPH_DIR / "adjacency_geo.npz", geo_matrix)

    # Step 2: Commodity similarity (per crop)
    logger.info("\n[2/3] Building commodity similarity matrices...")
    for crop in ["CORN", "SOYBEANS", "WHEAT"]:
        comm_matrix = build_commodity_matrix(fips_list, features, crop)
        sparse.save_npz(GRAPH_DIR / f"adjacency_commodity_{crop.lower()}.npz", comm_matrix)

    # Step 3: Combined + sparsified graph
    logger.info("\n[3/3] Building combined sparsified graph...")
    # Simple combination: geo + average commodity
    corn_m = sparse.load_npz(GRAPH_DIR / "adjacency_commodity_corn.npz")
    soy_m = sparse.load_npz(GRAPH_DIR / "adjacency_commodity_soybeans.npz")
    wheat_m = sparse.load_npz(GRAPH_DIR / "adjacency_commodity_wheat.npz")

    # Normalize each to [0, 1]
    def normalize_sparse(m):
        if m.nnz == 0:
            return m
        m_max = m.max()
        if m_max > 0:
            return m / m_max
        return m

    geo_norm = normalize_sparse(geo_matrix)
    comm_avg = normalize_sparse((corn_m + soy_m + wheat_m) / 3)

    combined = 0.5 * geo_norm + 0.5 * comm_avg
    edge_index, edge_weight = sparsify_to_top_k(combined, k=args.top_k)

    np.savez(
        GRAPH_DIR / "combined_graph.npz",
        edge_index=edge_index,
        edge_weight=edge_weight,
    )

    # Stats
    n = len(fips_list)
    stats = {
        "num_nodes": n,
        "geo_edges": geo_matrix.nnz,
        "commodity_corn_edges": corn_m.nnz,
        "commodity_soy_edges": soy_m.nnz,
        "commodity_wheat_edges": wheat_m.nnz,
        "combined_edges": len(edge_weight),
        "avg_degree": len(edge_weight) / n if n > 0 else 0,
        "top_k": args.top_k,
    }
    with open(GRAPH_DIR / "graph_stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    logger.info(f"\nGraph stats:")
    for k, v in stats.items():
        logger.info(f"  {k}: {v:,.1f}" if isinstance(v, float) else f"  {k}: {v:,}")
    logger.info(f"\nOutput: {GRAPH_DIR}/")


if __name__ == "__main__":
    main()
