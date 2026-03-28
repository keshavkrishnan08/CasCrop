"""
graph_builder.py — Build dynamic county graphs for ECMP.

Constructs three adjacency layers (geographic, commodity-market,
transport) and fuses them into a single time-varying graph where
counties are nodes and edges encode economic + spatial connectivity.

The final graph is sparsified to top-K neighbors per node and
exported as PyTorch tensors (edge_index, edge_attr) for consumption
by the ECMP message-passing module.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.spatial.distance import cdist

logger = logging.getLogger(__name__)


def _normalize_sparse(mat: sp.csr_matrix) -> sp.csr_matrix:
    """Row-normalize a sparse matrix so each row sums to 1 (or stays 0)."""
    row_sums = np.array(mat.sum(axis=1)).flatten()
    row_sums[row_sums == 0] = 1.0  # avoid division by zero
    diag_inv = sp.diags(1.0 / row_sums)
    return diag_inv @ mat


def _cosine_similarity_sparse(
    A: np.ndarray,
    B: np.ndarray | None = None,
) -> np.ndarray:
    """Pairwise cosine similarity. Returns dense (n, n) matrix."""
    if B is None:
        B = A
    # Normalize rows
    A_norm = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-12)
    B_norm = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-12)
    return A_norm @ B_norm.T


class GraphBuilder:
    """Build and manage the multi-layer dynamic county graph.

    Three adjacency layers:
    1. Geographic — distance-weighted or binary contiguity
    2. Commodity — per-crop market connectivity (production similarity,
       price correlation)
    3. Transport — road/rail distance proxy, grain elevator proximity

    These get fused via learnable coefficients (alpha, beta, gamma)
    into a single edge-weighted graph per time step.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config

        graph_cfg = config.get("graph", {})
        self.alpha: float = graph_cfg.get("alpha_init", 1.0 / 3)
        self.beta: float = graph_cfg.get("beta_init", 1.0 / 3)
        self.gamma: float = graph_cfg.get("gamma_init", 1.0 / 3)
        self.top_k: int = graph_cfg.get("sparsify_top_k", 20)
        self.distance_sigma: str | float = graph_cfg.get("distance_sigma", "auto")

        data_cfg = config.get("data", {})
        self.commodities: list[str] = data_cfg.get(
            "commodities", ["CORN", "SOYBEANS", "WHEAT"]
        )

        paths_cfg = config.get("paths", {})
        self.graphs_dir = Path(paths_cfg.get("graphs", "data/graphs"))

        # Caches (populated lazily)
        self._fips_list: list[str] = []
        self._fips_to_idx: dict[str, int] = {}
        self._geo_adj: sp.csr_matrix | None = None
        self._commodity_adj: dict[str, sp.csr_matrix] = {}
        self._transport_adj: sp.csr_matrix | None = None

    # ── Index management ────────────────────────────────────────────────

    @property
    def num_nodes(self) -> int:
        return len(self._fips_list)

    def _ensure_fips_index(self, fips_list: list[str] | np.ndarray) -> None:
        """Set the canonical FIPS ordering if not yet established."""
        if not self._fips_list:
            self._fips_list = sorted(set(fips_list))
            self._fips_to_idx = {f: i for i, f in enumerate(self._fips_list)}
            logger.info("FIPS index: %d counties.", self.num_nodes)

    # ── Geographic adjacency ────────────────────────────────────────────

    def build_geographic_adjacency(
        self,
        fips_metadata: pd.DataFrame,
        adjacency_file: Path | None = None,
    ) -> sp.csr_matrix:
        """Build geographic adjacency from county centroid distances.

        Parameters
        ----------
        fips_metadata : pd.DataFrame
            Must contain columns: ``fips``, ``lat``, ``lon``.
            Optionally ``neighbor_fips`` for Census contiguity data.
        adjacency_file : Path, optional
            Path to Census Bureau county adjacency CSV. If provided,
            binary contiguity is loaded from it directly.

        Returns
        -------
        scipy.sparse.csr_matrix
            Distance-weighted adjacency with w_ij = exp(-d_ij / sigma).
        """
        meta = fips_metadata.copy()
        meta["fips"] = meta["fips"].astype(str).str.zfill(5)
        meta = meta.drop_duplicates(subset=["fips"]).sort_values("fips")
        meta.reset_index(drop=True, inplace=True)

        self._ensure_fips_index(meta["fips"].tolist())
        n = self.num_nodes

        # ── Binary contiguity from adjacency file ───────────────────────
        binary_adj = sp.lil_matrix((n, n), dtype=np.float32)

        if adjacency_file is not None and Path(adjacency_file).exists():
            adj_raw = pd.read_csv(adjacency_file, dtype=str)
            # Expect columns like fips, neighbor_fips
            for col_pair in [("fips", "neighbor_fips"), ("fipscounty", "fipsneighbor")]:
                if col_pair[0] in adj_raw.columns and col_pair[1] in adj_raw.columns:
                    for _, row in adj_raw.iterrows():
                        fi = row[col_pair[0]].zfill(5)
                        fj = row[col_pair[1]].zfill(5)
                        if fi in self._fips_to_idx and fj in self._fips_to_idx:
                            i, j = self._fips_to_idx[fi], self._fips_to_idx[fj]
                            binary_adj[i, j] = 1.0
                            binary_adj[j, i] = 1.0
                    break
            logger.info("Loaded binary contiguity from %s.", adjacency_file)

        # ── Distance-weighted adjacency ─────────────────────────────────
        # Build ordered coordinate matrix
        coords = np.zeros((n, 2), dtype=np.float64)
        for _, row in meta.iterrows():
            idx = self._fips_to_idx.get(row["fips"])
            if idx is not None:
                coords[idx] = [row["lat"], row["lon"]]

        # Haversine-approximate distances in km
        dist_matrix = self._haversine_matrix(coords)

        # Sigma selection
        if self.distance_sigma == "auto":
            # Use median positive inter-county distance
            upper_tri = dist_matrix[np.triu_indices(n, k=1)]
            positive = upper_tri[upper_tri > 0]
            sigma = float(np.median(positive)) if len(positive) > 0 else 100.0
            logger.info("Auto sigma for distance kernel: %.1f km.", sigma)
        else:
            sigma = float(self.distance_sigma)

        # w_ij = exp(-d_ij / sigma), zero on diagonal
        weights = np.exp(-dist_matrix / sigma)
        np.fill_diagonal(weights, 0.0)

        # Combine: where we have binary contiguity, keep distance weight;
        # otherwise use distance weight if it's above a threshold
        geo_dense = weights.astype(np.float32)

        self._geo_adj = sp.csr_matrix(geo_dense)
        logger.info(
            "Geographic adjacency: %d nodes, %d non-zero edges.",
            n, self._geo_adj.nnz,
        )
        return self._geo_adj

    @staticmethod
    def _haversine_matrix(coords: np.ndarray) -> np.ndarray:
        """Pairwise haversine distance (km) from (lat, lon) in degrees."""
        lat = np.radians(coords[:, 0])
        lon = np.radians(coords[:, 1])

        n = len(lat)
        dist = np.zeros((n, n), dtype=np.float64)

        for i in range(n):
            dlat = lat - lat[i]
            dlon = lon - lon[i]
            a = (
                np.sin(dlat / 2) ** 2
                + np.cos(lat[i]) * np.cos(lat) * np.sin(dlon / 2) ** 2
            )
            c = 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
            dist[i] = 6371.0 * c  # Earth radius in km

        return dist

    # ── Commodity connectivity ──────────────────────────────────────────

    def build_commodity_connectivity(
        self,
        production_data: pd.DataFrame,
        crop: str,
        price_data: pd.DataFrame | None = None,
    ) -> sp.csr_matrix:
        """Per-crop pairwise market connectivity between counties.

        Combines:
        - Production similarity: cosine(crop_area_fraction, yield_history)
        - Price correlation: rolling 12-month Pearson r of prices received

        Parameters
        ----------
        production_data : pd.DataFrame
            Columns: fips, commodity, year, crop_area_fraction,
            historical_yield_mean (or yield).
        crop : str
            Single commodity name (e.g. "CORN").
        price_data : pd.DataFrame, optional
            Columns: fips, commodity, year, month, price.
            Used for rolling price correlation.

        Returns
        -------
        scipy.sparse.csr_matrix
        """
        prod = production_data.copy()
        prod["fips"] = prod["fips"].astype(str).str.zfill(5)
        prod["commodity"] = prod["commodity"].str.upper().str.strip()
        prod = prod[prod["commodity"] == crop.upper()]

        if self.num_nodes == 0:
            self._ensure_fips_index(prod["fips"].unique().tolist())
        n = self.num_nodes

        # ── Production similarity ───────────────────────────────────────
        # Average across years for a static profile per county
        yield_col = "historical_yield_mean" if "historical_yield_mean" in prod.columns else "yield"
        area_col = "crop_area_fraction" if "crop_area_fraction" in prod.columns else "area_planted"

        feature_cols = []
        for c in [yield_col, area_col]:
            if c in prod.columns:
                feature_cols.append(c)

        if not feature_cols:
            logger.warning(
                "No production feature columns found for crop %s. "
                "Returning identity commodity matrix.", crop,
            )
            self._commodity_adj[crop] = sp.eye(n, format="csr", dtype=np.float32)
            return self._commodity_adj[crop]

        county_avg = (
            prod.groupby("fips")[feature_cols]
            .mean()
            .reindex(self._fips_list)
            .fillna(0.0)
        )
        prod_features = county_avg.values.astype(np.float64)
        prod_sim = _cosine_similarity_sparse(prod_features)

        # Clip negative cosine similarities to 0
        prod_sim = np.clip(prod_sim, 0.0, 1.0)
        np.fill_diagonal(prod_sim, 0.0)

        # ── Price correlation ───────────────────────────────────────────
        price_corr = np.zeros((n, n), dtype=np.float64)
        if price_data is not None and "price" in price_data.columns:
            px = price_data.copy()
            px["fips"] = px["fips"].astype(str).str.zfill(5)
            px["commodity"] = px["commodity"].str.upper().str.strip()
            px = px[px["commodity"] == crop.upper()]

            # Pivot to (time, county) matrix
            px["time"] = px["year"] * 12 + px["month"]
            pivot = px.pivot_table(
                index="time", columns="fips", values="price", aggfunc="mean",
            )
            # Reindex to our FIPS ordering
            pivot = pivot.reindex(columns=self._fips_list)

            # Rolling 12-month correlation
            if len(pivot) >= 12:
                corr_mat = pivot.corr(min_periods=6).fillna(0.0).values
                np.fill_diagonal(corr_mat, 0.0)
                price_corr = np.clip(corr_mat, 0.0, 1.0)

        # Combine: equal weight production similarity + price correlation
        combined = 0.5 * prod_sim + 0.5 * price_corr
        combined = combined.astype(np.float32)

        self._commodity_adj[crop] = sp.csr_matrix(combined)
        logger.info(
            "Commodity adjacency [%s]: %d nodes, %d non-zero.",
            crop, n, self._commodity_adj[crop].nnz,
        )
        return self._commodity_adj[crop]

    # ── Transport connectivity ──────────────────────────────────────────

    def build_transport_connectivity(
        self,
        fips_metadata: pd.DataFrame,
        elevator_data: pd.DataFrame | None = None,
    ) -> sp.csr_matrix:
        """Road distance approximation and grain elevator proximity.

        In the absence of real road network data, we approximate
        transport connectivity as an inverse-distance kernel of
        county centroids (tighter kernel than geography) combined
        with shared grain-elevator access.

        Parameters
        ----------
        fips_metadata : pd.DataFrame
            Must contain fips, lat, lon.
        elevator_data : pd.DataFrame, optional
            Grain elevator locations with lat, lon, capacity columns.
            If absent, transport connectivity degrades to distance only.

        Returns
        -------
        scipy.sparse.csr_matrix
        """
        meta = fips_metadata.copy()
        meta["fips"] = meta["fips"].astype(str).str.zfill(5)
        meta = meta.drop_duplicates(subset=["fips"]).sort_values("fips")
        meta.reset_index(drop=True, inplace=True)

        self._ensure_fips_index(meta["fips"].tolist())
        n = self.num_nodes

        coords = np.zeros((n, 2), dtype=np.float64)
        for _, row in meta.iterrows():
            idx = self._fips_to_idx.get(row["fips"])
            if idx is not None:
                coords[idx] = [row["lat"], row["lon"]]

        # Transport distance: tighter kernel (sigma / 3) than geography
        dist = self._haversine_matrix(coords)
        if self.distance_sigma == "auto":
            upper_tri = dist[np.triu_indices(n, k=1)]
            positive = upper_tri[upper_tri > 0]
            sigma = float(np.median(positive)) / 3.0 if len(positive) > 0 else 50.0
        else:
            sigma = float(self.distance_sigma) / 3.0

        transport_weight = np.exp(-dist / sigma).astype(np.float32)
        np.fill_diagonal(transport_weight, 0.0)

        # ── Grain elevator proximity bonus ──────────────────────────────
        if elevator_data is not None and len(elevator_data) > 0:
            elev = elevator_data.copy()
            elev_coords = elev[["lat", "lon"]].values.astype(np.float64)

            # For each county, find nearest elevator distance
            county_elev_dist = cdist(
                np.radians(coords), np.radians(elev_coords),
                metric="euclidean",
            )
            # Assign each elevator to its nearest county
            nearest_county = county_elev_dist.argmin(axis=0)

            # Counties sharing access to the same elevator cluster
            # (within 50 km) get a bonus
            for e_idx in range(len(elev)):
                close_counties = np.where(
                    county_elev_dist[:, e_idx] < 0.5  # ~50 km in radians approx
                )[0]
                for i in close_counties:
                    for j in close_counties:
                        if i != j:
                            transport_weight[i, j] += 0.1

            # Re-clip to [0, 1]
            transport_weight = np.clip(transport_weight, 0.0, 1.0)
            logger.info(
                "Added grain elevator proximity for %d elevators.", len(elev),
            )

        self._transport_adj = sp.csr_matrix(transport_weight)
        logger.info(
            "Transport adjacency: %d nodes, %d non-zero.",
            n, self._transport_adj.nnz,
        )
        return self._transport_adj

    # ── Dynamic graph fusion ────────────────────────────────────────────

    def build_dynamic_graph(
        self,
        geo_adj: sp.csr_matrix | None = None,
        commodity_adj: sp.csr_matrix | None = None,
        transport_adj: sp.csr_matrix | None = None,
        month: int | None = None,
        alpha: float | None = None,
        beta: float | None = None,
        gamma: float | None = None,
        top_k: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Fuse adjacency layers into a single sparsified graph.

        w_ij(t) = alpha * geo_ij + beta * commodity_ij(t) + gamma * transport_ij

        Then keep only the top-K neighbors per node.

        Parameters
        ----------
        geo_adj, commodity_adj, transport_adj : sparse matrices, optional
            Fall back to cached versions from prior build calls.
        month : int, optional
            Calendar month (unused currently; future: index time-varying
            commodity adjacency snapshots).
        alpha, beta, gamma : float, optional
            Fusion weights. Default to ``self.alpha``, etc.
        top_k : int, optional
            Number of neighbors to keep per node. Default: ``self.top_k``.

        Returns
        -------
        edge_index : np.ndarray, shape (2, num_edges)
            Source and target node indices.
        edge_attr : np.ndarray, shape (num_edges,)
            Edge weights.
        """
        if geo_adj is None:
            geo_adj = self._geo_adj
        if commodity_adj is None:
            # Grab the first cached crop if available
            if self._commodity_adj:
                commodity_adj = next(iter(self._commodity_adj.values()))
        if transport_adj is None:
            transport_adj = self._transport_adj

        a = alpha if alpha is not None else self.alpha
        b = beta if beta is not None else self.beta
        g = gamma if gamma is not None else self.gamma
        k = top_k if top_k is not None else self.top_k

        # Start with zeros
        n = max(
            geo_adj.shape[0] if geo_adj is not None else 0,
            commodity_adj.shape[0] if commodity_adj is not None else 0,
            transport_adj.shape[0] if transport_adj is not None else 0,
        )
        if n == 0:
            logger.warning("All adjacency matrices are empty. Returning empty graph.")
            return (
                np.zeros((2, 0), dtype=np.int64),
                np.zeros((0,), dtype=np.float32),
            )

        fused = sp.csr_matrix((n, n), dtype=np.float32)

        if geo_adj is not None:
            # Ensure matching dimensions by slicing or padding
            geo_adj = self._match_shape(geo_adj, n)
            fused = fused + a * geo_adj

        if commodity_adj is not None:
            commodity_adj = self._match_shape(commodity_adj, n)
            fused = fused + b * commodity_adj

        if transport_adj is not None:
            transport_adj = self._match_shape(transport_adj, n)
            fused = fused + g * transport_adj

        # ── Sparsify to top-K per node ──────────────────────────────────
        fused_dense = fused.toarray()
        np.fill_diagonal(fused_dense, 0.0)

        edge_rows, edge_cols, edge_vals = [], [], []

        for i in range(n):
            row = fused_dense[i]
            nonzero_idx = np.nonzero(row)[0]

            if len(nonzero_idx) == 0:
                continue

            if len(nonzero_idx) <= k:
                # Keep all
                for j in nonzero_idx:
                    edge_rows.append(i)
                    edge_cols.append(j)
                    edge_vals.append(row[j])
            else:
                # Top-K by weight
                top_idx = np.argpartition(row[nonzero_idx], -k)[-k:]
                selected = nonzero_idx[top_idx]
                for j in selected:
                    edge_rows.append(i)
                    edge_cols.append(j)
                    edge_vals.append(row[j])

        edge_index = np.array([edge_rows, edge_cols], dtype=np.int64)
        edge_attr = np.array(edge_vals, dtype=np.float32)

        logger.info(
            "Dynamic graph: %d nodes, %d edges (top-%d per node), "
            "weights [alpha=%.3f, beta=%.3f, gamma=%.3f].",
            n, len(edge_vals), k, a, b, g,
        )

        return edge_index, edge_attr

    @staticmethod
    def _match_shape(mat: sp.csr_matrix, n: int) -> sp.csr_matrix:
        """Resize a sparse matrix to (n, n), padding or truncating."""
        if mat.shape == (n, n):
            return mat
        current = mat.shape[0]
        if current >= n:
            return mat[:n, :n].tocsr()
        # Pad with zeros
        padded = sp.lil_matrix((n, n), dtype=mat.dtype)
        padded[:current, :current] = mat
        return padded.tocsr()

    # ── Graph statistics ────────────────────────────────────────────────

    @staticmethod
    def compute_graph_statistics(
        edge_index: np.ndarray,
        num_nodes: int,
    ) -> dict[str, Any]:
        """Compute summary statistics for a graph.

        Returns dict with: num_nodes, num_edges, avg_degree,
        density, clustering_coefficient (approximate).
        """
        num_edges = edge_index.shape[1] if edge_index.ndim == 2 else 0
        avg_degree = num_edges / max(num_nodes, 1)
        density = num_edges / max(num_nodes * (num_nodes - 1), 1)

        # Degree distribution
        if num_edges > 0:
            src = edge_index[0]
            degrees = np.bincount(src, minlength=num_nodes)
            degree_mean = float(degrees.mean())
            degree_std = float(degrees.std())
            degree_max = int(degrees.max())
            degree_min = int(degrees.min())
            isolated = int((degrees == 0).sum())
        else:
            degree_mean = degree_std = 0.0
            degree_max = degree_min = isolated = 0

        # Approximate clustering coefficient via sampling
        clustering = 0.0
        if num_edges > 0 and num_nodes > 2:
            adj_set: dict[int, set[int]] = {}
            for idx in range(num_edges):
                s, t = int(edge_index[0, idx]), int(edge_index[1, idx])
                adj_set.setdefault(s, set()).add(t)

            # Sample up to 500 nodes for clustering
            sample_size = min(num_nodes, 500)
            sample_nodes = np.random.choice(num_nodes, sample_size, replace=False)
            coeffs = []

            for node in sample_nodes:
                neighbors = adj_set.get(node, set())
                k = len(neighbors)
                if k < 2:
                    coeffs.append(0.0)
                    continue
                # Count edges among neighbors
                triangles = 0
                neighbor_list = list(neighbors)
                for ii in range(k):
                    for jj in range(ii + 1, k):
                        if neighbor_list[jj] in adj_set.get(neighbor_list[ii], set()):
                            triangles += 1
                possible = k * (k - 1) / 2
                coeffs.append(triangles / possible)

            clustering = float(np.mean(coeffs))

        stats = {
            "num_nodes": num_nodes,
            "num_edges": num_edges,
            "avg_degree": avg_degree,
            "density": density,
            "degree_mean": degree_mean,
            "degree_std": degree_std,
            "degree_max": degree_max,
            "degree_min": degree_min,
            "isolated_nodes": isolated,
            "clustering_coefficient": clustering,
        }

        logger.info(
            "Graph stats: %d nodes, %d edges, avg_degree=%.2f, "
            "clustering=%.4f.",
            num_nodes, num_edges, avg_degree, clustering,
        )

        return stats

    # ── Build all layers from data directory ────────────────────────────

    def build_all(
        self,
        fips_metadata: pd.DataFrame,
        production_data: pd.DataFrame,
        price_data: pd.DataFrame | None = None,
        elevator_data: pd.DataFrame | None = None,
        adjacency_file: Path | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Convenience: build every adjacency layer then fuse.

        Returns the final (edge_index, edge_attr) for the fused graph.
        """
        geo = self.build_geographic_adjacency(fips_metadata, adjacency_file)
        transport = self.build_transport_connectivity(fips_metadata, elevator_data)

        # Build commodity adjacency per crop, then average
        crop_adjs = []
        for crop in self.commodities:
            crop_prod = production_data[
                production_data["commodity"].str.upper() == crop.upper()
            ]
            if len(crop_prod) > 0:
                adj = self.build_commodity_connectivity(
                    production_data, crop, price_data,
                )
                crop_adjs.append(adj)

        if crop_adjs:
            # Average across crops
            n = crop_adjs[0].shape[0]
            avg_commodity = sum(
                self._match_shape(a, n) for a in crop_adjs
            ) / len(crop_adjs)
            avg_commodity = sp.csr_matrix(avg_commodity)
        else:
            avg_commodity = None

        edge_index, edge_attr = self.build_dynamic_graph(
            geo_adj=geo,
            commodity_adj=avg_commodity,
            transport_adj=transport,
        )

        return edge_index, edge_attr

    # ── Persistence ─────────────────────────────────────────────────────

    def save_graphs(self, output_dir: Path | str | None = None) -> None:
        """Write all adjacency matrices and graph metadata to disk."""
        out = Path(output_dir) if output_dir else self.graphs_dir
        out.mkdir(parents=True, exist_ok=True)

        # Geographic
        if self._geo_adj is not None:
            sp.save_npz(out / "adjacency_geo.npz", self._geo_adj)
            logger.info("Saved geographic adjacency -> %s", out / "adjacency_geo.npz")

        # Commodity (per-crop)
        for crop, adj in self._commodity_adj.items():
            fname = f"adjacency_commodity_{crop.lower()}.npz"
            sp.save_npz(out / fname, adj)
            logger.info("Saved commodity adjacency [%s] -> %s", crop, out / fname)

        # Transport
        if self._transport_adj is not None:
            sp.save_npz(out / "adjacency_transport.npz", self._transport_adj)
            logger.info("Saved transport adjacency -> %s", out / "adjacency_transport.npz")

        # FIPS mapping
        fips_map = {
            "fips_list": self._fips_list,
            "fips_to_idx": self._fips_to_idx,
        }
        with open(out / "fips_mapping.json", "w") as f:
            json.dump(fips_map, f, indent=2)

        # Build and save a fused graph snapshot
        edge_index, edge_attr = self.build_dynamic_graph()
        np.savez(
            out / "fused_graph.npz",
            edge_index=edge_index,
            edge_attr=edge_attr,
        )

        # Statistics
        stats = self.compute_graph_statistics(edge_index, self.num_nodes)
        with open(out / "graph_statistics.json", "w") as f:
            json.dump(stats, f, indent=2)
        logger.info("Saved graph statistics -> %s", out / "graph_statistics.json")

    # ── Load from disk ──────────────────────────────────────────────────

    def load_graphs(self, input_dir: Path | str | None = None) -> None:
        """Reload previously saved adjacency matrices."""
        src = Path(input_dir) if input_dir else self.graphs_dir

        geo_path = src / "adjacency_geo.npz"
        if geo_path.exists():
            self._geo_adj = sp.load_npz(geo_path)
            logger.info("Loaded geographic adjacency from %s.", geo_path)

        transport_path = src / "adjacency_transport.npz"
        if transport_path.exists():
            self._transport_adj = sp.load_npz(transport_path)
            logger.info("Loaded transport adjacency from %s.", transport_path)

        # Commodity adjacencies
        for path in sorted(src.glob("adjacency_commodity_*.npz")):
            crop = path.stem.replace("adjacency_commodity_", "").upper()
            self._commodity_adj[crop] = sp.load_npz(path)
            logger.info("Loaded commodity adjacency [%s] from %s.", crop, path)

        # FIPS mapping
        fips_path = src / "fips_mapping.json"
        if fips_path.exists():
            with open(fips_path) as f:
                mapping = json.load(f)
            self._fips_list = mapping["fips_list"]
            self._fips_to_idx = mapping["fips_to_idx"]
            logger.info("Loaded FIPS mapping: %d counties.", self.num_nodes)

    # ── Utility: get edge tensors for a specific crop + time ────────────

    def get_graph_for_crop_month(
        self,
        crop: str,
        year: int,
        month: int,
        alpha: float | None = None,
        beta: float | None = None,
        gamma: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return (edge_index, edge_attr) for a specific crop and time.

        Uses the crop-specific commodity adjacency if available.
        """
        crop_key = crop.upper()
        commodity_adj = self._commodity_adj.get(crop_key)

        return self.build_dynamic_graph(
            geo_adj=self._geo_adj,
            commodity_adj=commodity_adj,
            transport_adj=self._transport_adj,
            month=month,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
        )
