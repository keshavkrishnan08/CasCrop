"""
dataset.py — PyTorch Dataset and DataLoader for CasCrop.

Provides ``CasCropDataset`` (a standard ``torch.utils.data.Dataset``)
and ``CasCropGraphBatch`` (a collation helper that groups samples by
time period so they share graph structure for efficient ECMP).

Usage::

    from data.dataset import get_dataloaders
    train_dl, val_dl, test_dl = get_dataloaders(config)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from .graph_builder import GraphBuilder
from .matcher import (
    BIOPHYSICAL_FEATURES,
    ECONOMIC_FEATURES,
    HISTORICAL_FEATURES,
    MERGE_KEYS,
    TEMPORAL_FEATURES,
)

logger = logging.getLogger(__name__)

# Optional torch_geometric support
try:
    from torch_geometric.data import Data as PyGData

    HAS_PYG = True
except ImportError:
    HAS_PYG = False
    logger.debug("torch_geometric not available; graph batching will use plain dicts.")


# ── Feature column helpers ──────────────────────────────────────────────────

def _resolve_columns(df: pd.DataFrame, requested: list[str]) -> list[str]:
    """Return the subset of *requested* columns that actually exist in *df*."""
    return [c for c in requested if c in df.columns]


# ── Dataset ─────────────────────────────────────────────────────────────────

class CasCropDataset(Dataset):
    """PyTorch Dataset for county-crop-month observations.

    Each sample is a dict containing:

    - ``x_bio``:  biophysical feature tensor   [d_bio]
    - ``x_econ``: economic feature tensor       [d_econ]
    - ``x_hist``: historical feature tensor     [d_hist]
    - ``y_waste``: binary waste label           [1]
    - ``y_cause``: multi-class cause label      [1]
    - ``fips``:    county FIPS code (str)
    - ``commodity``: commodity name (str)
    - ``year``:    observation year (int)
    - ``month``:   observation month (int)
    - ``price_shock``: 1-month price change     [1]
    - ``node_idx``: index into the graph        [1]

    Feature tensors are z-score normalized using **train-set statistics**
    regardless of split, preventing information leakage.
    """

    def __init__(
        self,
        features_df: pd.DataFrame,
        labels_df: pd.DataFrame,
        graph_builder: GraphBuilder | None = None,
        config: dict[str, Any] | None = None,
        split: str = "train",
        split_indices: np.ndarray | None = None,
        feature_stats: dict[str, dict[str, float]] | None = None,
    ) -> None:
        """
        Parameters
        ----------
        features_df : pd.DataFrame
            Full feature matrix (all splits).
        labels_df : pd.DataFrame
            Full label matrix (all splits).
        graph_builder : GraphBuilder, optional
            Pre-built graph for edge_index / edge_attr lookup.
        config : dict, optional
            Project configuration.
        split : str
            One of 'train', 'val', 'test'.
        split_indices : np.ndarray, optional
            Row indices for this split. If None the full DataFrame is used.
        feature_stats : dict, optional
            Pre-computed {col: {mean, std}} from the training set.
            If None and split == 'train', stats are computed on the fly.
        """
        super().__init__()

        self.config = config or {}
        self.split = split
        self.graph_builder = graph_builder

        # ── Subset to split ─────────────────────────────────────────────
        if split_indices is not None:
            self.features = features_df.iloc[split_indices].reset_index(drop=True)
            self.labels = labels_df.iloc[split_indices].reset_index(drop=True)
        else:
            self.features = features_df.reset_index(drop=True)
            self.labels = labels_df.reset_index(drop=True)

        # ── Resolve available columns per feature group ─────────────────
        self.bio_cols = _resolve_columns(self.features, BIOPHYSICAL_FEATURES)
        self.econ_cols = _resolve_columns(self.features, ECONOMIC_FEATURES)
        self.hist_cols = _resolve_columns(self.features, HISTORICAL_FEATURES)
        self.temp_cols = _resolve_columns(self.features, TEMPORAL_FEATURES)

        # Merge temporal into bio for convenience (they're low-dim)
        self.bio_cols_full = self.bio_cols + self.temp_cols

        logger.info(
            "CasCropDataset [%s]: %d samples, d_bio=%d, d_econ=%d, d_hist=%d.",
            split,
            len(self.features),
            len(self.bio_cols_full),
            len(self.econ_cols),
            len(self.hist_cols),
        )

        # ── Normalization ───────────────────────────────────────────────
        self.feature_stats = feature_stats or {}
        if not self.feature_stats and split == "train":
            self.feature_stats = self._compute_stats(self.features)

        self._apply_normalization()

        # ── FIPS -> node index mapping ──────────────────────────────────
        if graph_builder is not None and graph_builder._fips_to_idx:
            self._fips_to_node = graph_builder._fips_to_idx
        else:
            unique_fips = sorted(self.features["fips"].unique())
            self._fips_to_node = {f: i for i, f in enumerate(unique_fips)}

        # ── Pre-compute tensors for speed ───────────────────────────────
        all_feat_cols = self.bio_cols_full + self.econ_cols + self.hist_cols
        self._feature_tensor = torch.tensor(
            self.features[all_feat_cols].values.astype(np.float32),
            dtype=torch.float32,
        )

        # Slice boundaries
        self._bio_end = len(self.bio_cols_full)
        self._econ_end = self._bio_end + len(self.econ_cols)
        # hist goes from _econ_end to end

        # Labels
        self._y_waste = torch.tensor(
            self.labels["waste"].values.astype(np.float32),
            dtype=torch.float32,
        )
        self._y_cause = torch.tensor(
            self.labels["cause"].values.astype(np.int64),
            dtype=torch.long,
        )

        # Price shock (1-month price change) — used by ECMP
        if "price_change_1m" in self.features.columns:
            self._price_shock = torch.tensor(
                self.features["price_change_1m"]
                .fillna(0.0)
                .values.astype(np.float32),
                dtype=torch.float32,
            )
        else:
            self._price_shock = torch.zeros(len(self.features), dtype=torch.float32)

        # Metadata arrays
        self._fips = self.features["fips"].values
        self._commodity = self.features["commodity"].values
        self._year = self.features["year"].values.astype(int)
        self._month = self.features["month"].values.astype(int)

    # ── Normalization helpers ───────────────────────────────────────────

    def _compute_stats(self, df: pd.DataFrame) -> dict[str, dict[str, float]]:
        """Compute column-wise mean/std for z-score normalization."""
        all_cols = self.bio_cols_full + self.econ_cols + self.hist_cols
        stats: dict[str, dict[str, float]] = {}
        for col in all_cols:
            if col in df.columns:
                vals = df[col].dropna()
                stats[col] = {
                    "mean": float(vals.mean()) if len(vals) > 0 else 0.0,
                    "std": float(vals.std()) if len(vals) > 0 else 1.0,
                }
                # Guard against zero std
                if stats[col]["std"] < 1e-8:
                    stats[col]["std"] = 1.0
        return stats

    def _apply_normalization(self) -> None:
        """Z-score normalize feature columns in-place."""
        if not self.feature_stats:
            return

        all_cols = self.bio_cols_full + self.econ_cols + self.hist_cols
        for col in all_cols:
            if col in self.features.columns and col in self.feature_stats:
                mu = self.feature_stats[col]["mean"]
                sigma = self.feature_stats[col]["std"]
                self.features[col] = (self.features[col] - mu) / sigma

    # ── Dataset interface ───────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self._feature_tensor[idx]

        x_bio = row[: self._bio_end]
        x_econ = row[self._bio_end : self._econ_end]
        x_hist = row[self._econ_end :]

        fips_str = str(self._fips[idx])
        node_idx = self._fips_to_node.get(fips_str, -1)

        return {
            "x_bio": x_bio,
            "x_econ": x_econ,
            "x_hist": x_hist,
            "waste_target": self._y_waste[idx].unsqueeze(0),
            "cause_target": self._y_cause[idx].unsqueeze(0),
            "fips": fips_str,
            "commodity": str(self._commodity[idx]),
            "year": int(self._year[idx]),
            "month": int(self._month[idx]),
            "price_shocks": self._price_shock[idx].unsqueeze(0),
            "node_idx": torch.tensor([node_idx], dtype=torch.long),
        }

    def get_class_weights(self) -> torch.Tensor:
        """Compute inverse-frequency class weights for the waste label.

        Returns a 2-element tensor [w_negative, w_positive].
        """
        pos = float(self._y_waste.sum())
        neg = float(len(self._y_waste) - pos)
        total = pos + neg

        if pos == 0 or neg == 0:
            return torch.ones(2, dtype=torch.float32)

        w_neg = total / (2.0 * neg)
        w_pos = total / (2.0 * pos)
        return torch.tensor([w_neg, w_pos], dtype=torch.float32)

    def get_cause_class_weights(self) -> torch.Tensor:
        """Inverse-frequency weights for the 6 cause classes."""
        counts = torch.bincount(self._y_cause, minlength=6).float()
        counts = counts.clamp(min=1.0)
        total = counts.sum()
        weights = total / (6.0 * counts)
        return weights

    @property
    def d_bio(self) -> int:
        return len(self.bio_cols_full)

    @property
    def d_econ(self) -> int:
        return len(self.econ_cols)

    @property
    def d_hist(self) -> int:
        return len(self.hist_cols)


# ── Graph-aware batch collation ─────────────────────────────────────────────

class CasCropGraphBatch:
    """Collation helper that groups samples into batched graph data.

    Samples sharing the same (commodity, year, month) triple are
    grouped so they use a single graph structure.  This is essential
    for ECMP, where message passing happens across counties within
    the same time period.
    """

    def __init__(self, graph_builder: GraphBuilder | None = None) -> None:
        self.graph_builder = graph_builder

    def collate_fn(self, samples: list[dict[str, Any]]) -> dict[str, Any]:
        """Collate a list of sample dicts into a batched dict.

        Returns
        -------
        dict with keys:
            x_bio:      [B, d_bio]
            x_econ:     [B, d_econ]
            x_hist:     [B, d_hist]
            y_waste:    [B, 1]
            y_cause:    [B, 1]
            price_shock:[B, 1]
            node_idx:   [B, 1]
            fips:       list[str]
            commodity:  list[str]
            year:       list[int]
            month:      list[int]
            edge_index: [2, E]  (graph edges, may be empty)
            edge_attr:  [E]     (edge weights)
            batch_idx:  [B]     (maps each sample to its graph subgraph)
        """
        batch_size = len(samples)

        # Stack tensors
        x_bio = torch.stack([s["x_bio"] for s in samples])
        x_econ = torch.stack([s["x_econ"] for s in samples])
        x_hist = torch.stack([s["x_hist"] for s in samples])
        waste_target = torch.stack([s["waste_target"] for s in samples])
        cause_target = torch.stack([s["cause_target"] for s in samples])
        price_shocks = torch.stack([s["price_shocks"] for s in samples])
        node_idx = torch.stack([s["node_idx"] for s in samples])

        # Metadata
        fips_list = [s["fips"] for s in samples]
        commodity_list = [s["commodity"] for s in samples]
        year_list = [s["year"] for s in samples]
        month_list = [s["month"] for s in samples]

        # ── Build graph for this batch ──────────────────────────────────
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.zeros((0,), dtype=torch.float32)
        batch_idx = torch.zeros(batch_size, dtype=torch.long)

        if self.graph_builder is not None:
            # Group samples by (commodity, year, month) for shared graphs
            groups: dict[tuple[str, int, int], list[int]] = {}
            for i, s in enumerate(samples):
                key = (s["commodity"], s["year"], s["month"])
                groups.setdefault(key, []).append(i)

            all_edge_src: list[int] = []
            all_edge_dst: list[int] = []
            all_edge_wt: list[float] = []
            node_offset = 0

            for group_id, ((crop, yr, mo), sample_indices) in enumerate(groups.items()):
                # Get graph for this crop-month
                ei, ea = self.graph_builder.get_graph_for_crop_month(
                    crop, yr, mo,
                )

                # Map graph node indices to batch-local indices
                # Only keep edges between nodes that are in this batch
                batch_node_ids = set()
                for si in sample_indices:
                    nid = samples[si]["node_idx"].item()
                    if nid >= 0:
                        batch_node_ids.add(nid)
                    batch_idx[si] = group_id

                if len(batch_node_ids) < 2 or ei.shape[1] == 0:
                    continue

                # Create a local mapping for nodes in this batch group
                batch_node_list = sorted(batch_node_ids)
                local_map = {nid: local_i + node_offset for local_i, nid in enumerate(batch_node_list)}

                # Filter edges to only those between batch nodes
                for e_idx in range(ei.shape[1]):
                    src, dst = int(ei[0, e_idx]), int(ei[1, e_idx])
                    if src in local_map and dst in local_map:
                        all_edge_src.append(local_map[src])
                        all_edge_dst.append(local_map[dst])
                        all_edge_wt.append(float(ea[e_idx]))

                node_offset += len(batch_node_list)

            if all_edge_src:
                edge_index = torch.tensor(
                    [all_edge_src, all_edge_dst], dtype=torch.long,
                )
                edge_attr = torch.tensor(all_edge_wt, dtype=torch.float32)

        result = {
            "x_bio": x_bio,
            "x_econ": x_econ,
            "x_hist": x_hist,
            "waste_target": waste_target,
            "cause_target": cause_target,
            "price_shocks": price_shocks,
            "node_idx": node_idx,
            "fips": fips_list,
            "commodity": commodity_list,
            "year": year_list,
            "month": month_list,
            "edge_index": edge_index,
            "edge_attr": edge_attr,
            "batch_idx": batch_idx,
        }

        # Optionally wrap in PyG Data object
        if HAS_PYG:
            result["pyg_data"] = PyGData(
                x=torch.cat([x_bio, x_econ, x_hist], dim=1),
                edge_index=edge_index,
                edge_attr=edge_attr.unsqueeze(-1) if edge_attr.ndim == 1 else edge_attr,
                y=waste_target.squeeze(-1),
            )

        return result


# ── DataLoader factory ──────────────────────────────────────────────────────

def get_dataloaders(
    config: dict[str, Any],
    features_df: pd.DataFrame | None = None,
    labels_df: pd.DataFrame | None = None,
    graph_builder: GraphBuilder | None = None,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Create train, validation, and test DataLoaders.

    If ``features_df`` / ``labels_df`` are not provided they are loaded
    from the paths specified in ``config['paths']``.

    Parameters
    ----------
    config : dict
        Project configuration (must contain ``data`` and ``paths`` keys).
    features_df, labels_df : pd.DataFrame, optional
        Pre-loaded DataFrames.
    graph_builder : GraphBuilder, optional
        Pre-built graph builder for edge construction.

    Returns
    -------
    train_loader, val_loader, test_loader : DataLoader
    """
    paths = config.get("paths", {})
    data_cfg = config.get("data", {})
    training_cfg = config.get("training", {})

    processed_dir = Path(paths.get("processed_data", "data/processed"))
    splits_dir = Path(paths.get("splits", "data/splits"))

    batch_size = training_cfg.get("batch_size", 256)
    num_workers = training_cfg.get("num_workers", 0)

    # ── Load data if not provided ───────────────────────────────────────
    if features_df is None:
        feat_path = processed_dir / "features.parquet"
        if not feat_path.exists():
            raise FileNotFoundError(
                f"Features not found at {feat_path}. "
                "Run DataMatcher.match_all() and .save() first."
            )
        features_df = pd.read_parquet(feat_path)
        logger.info("Loaded features: %d rows from %s.", len(features_df), feat_path)

    if labels_df is None:
        lab_path = processed_dir / "labels.parquet"
        if not lab_path.exists():
            raise FileNotFoundError(
                f"Labels not found at {lab_path}. "
                "Run DataMatcher.match_all() and .save() first."
            )
        labels_df = pd.read_parquet(lab_path)
        logger.info("Loaded labels: %d rows from %s.", len(labels_df), lab_path)

    # ── Load split indices ──────────────────────────────────────────────
    splits: dict[str, np.ndarray] = {}
    for name in ("train", "val", "test"):
        idx_path = splits_dir / f"{name}_indices.npy"
        if idx_path.exists():
            splits[name] = np.load(idx_path)
            logger.info("Loaded %s split: %d indices.", name, len(splits[name]))
        else:
            logger.warning(
                "Split file %s not found. Falling back to year-based split.",
                idx_path,
            )

    # Fallback: year-based split
    if not splits:
        train_years = data_cfg.get("train_years", list(range(2008, 2020)))
        val_years = data_cfg.get("val_years", [2020, 2021])
        test_years = data_cfg.get("test_years", [2022, 2023, 2024])

        splits["train"] = features_df.index[features_df["year"].isin(train_years)].values
        splits["val"] = features_df.index[features_df["year"].isin(val_years)].values
        splits["test"] = features_df.index[features_df["year"].isin(test_years)].values
        logger.info(
            "Year-based split: train=%d, val=%d, test=%d.",
            len(splits["train"]), len(splits["val"]), len(splits["test"]),
        )

    # ── Load or build graph ─────────────────────────────────────────────
    if graph_builder is None:
        graphs_dir = Path(paths.get("graphs", "data/graphs"))
        if (graphs_dir / "fips_mapping.json").exists():
            graph_builder = GraphBuilder(config)
            graph_builder.load_graphs(graphs_dir)
            logger.info("Loaded pre-built graph from %s.", graphs_dir)

    # ── Load normalization stats from train set ─────────────────────────
    stats_path = processed_dir / "feature_statistics.json"
    feature_stats: dict[str, dict[str, float]] | None = None
    if stats_path.exists():
        with open(stats_path) as f:
            feature_stats = json.load(f)
        logger.info("Loaded feature statistics from %s.", stats_path)

    # ── Build datasets ──────────────────────────────────────────────────
    train_ds = CasCropDataset(
        features_df=features_df,
        labels_df=labels_df,
        graph_builder=graph_builder,
        config=config,
        split="train",
        split_indices=splits.get("train"),
        feature_stats=feature_stats,
    )

    # Use train stats for val and test normalization
    shared_stats = train_ds.feature_stats or feature_stats

    val_ds = CasCropDataset(
        features_df=features_df,
        labels_df=labels_df,
        graph_builder=graph_builder,
        config=config,
        split="val",
        split_indices=splits.get("val"),
        feature_stats=shared_stats,
    )

    test_ds = CasCropDataset(
        features_df=features_df,
        labels_df=labels_df,
        graph_builder=graph_builder,
        config=config,
        split="test",
        split_indices=splits.get("test"),
        feature_stats=shared_stats,
    )

    # ── Collation ───────────────────────────────────────────────────────
    collator = CasCropGraphBatch(graph_builder=graph_builder)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collator.collate_fn,
        pin_memory=True,
        drop_last=False,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collator.collate_fn,
        pin_memory=True,
        drop_last=False,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collator.collate_fn,
        pin_memory=True,
        drop_last=False,
    )

    logger.info(
        "DataLoaders: train=%d batches, val=%d batches, test=%d batches "
        "(batch_size=%d).",
        len(train_loader), len(val_loader), len(test_loader), batch_size,
    )

    return train_loader, val_loader, test_loader
