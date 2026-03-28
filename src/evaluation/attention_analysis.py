"""Extract and analyze ECMP attention weights.

Provides tools for examining which county-to-county connections
the model learns to attend to, and how attention correlates
with price shocks and economic contagion.
"""

import numpy as np
import torch
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class AttentionAnalyzer:
    """Analyze ECMP attention weights from trained CasCrop model.

    Extracts attention patterns, identifies top contagion edges,
    and computes attention statistics for case study visualization.
    """

    def __init__(self, model, device: str = "cpu"):
        self.model = model
        self.device = torch.device(device)
        self.model.to(self.device)
        self.model.eval()

    def extract_attention_weights(self, batch: dict) -> dict:
        """Extract attention weights from ECMP layers for a batch.

        Returns:
            dict with:
                attention_weights: [num_edges, num_heads] per layer
                edge_index: [2, num_edges]
                price_shocks: [num_nodes]
                node_fips: list of FIPS codes
        """
        batch_device = {
            k: v.to(self.device) if hasattr(v, "to") else v
            for k, v in batch.items()
        }

        with torch.no_grad():
            outputs = self.model(batch_device)

        result = {
            "edge_index": batch["edge_index"].cpu().numpy(),
        }

        if "attention_weights" in outputs:
            attn = outputs["attention_weights"]
            if isinstance(attn, list):
                result["attention_weights"] = [a.cpu().numpy() for a in attn]
            else:
                result["attention_weights"] = attn.cpu().numpy()

        if "price_shock" in batch:
            result["price_shocks"] = batch["price_shock"].cpu().numpy()

        if "fips" in batch:
            result["node_fips"] = batch["fips"]

        return result

    def get_top_k_edges(
        self,
        attention_weights: np.ndarray,
        edge_index: np.ndarray,
        node_fips: list,
        k: int = 20,
        head: int = 0,
    ) -> list[dict]:
        """Get the top-K highest attention edges.

        Args:
            attention_weights: [num_edges, num_heads]
            edge_index: [2, num_edges]
            node_fips: FIPS codes for each node
            k: number of top edges to return
            head: which attention head to use

        Returns:
            list of dicts with source_fips, target_fips, attention_weight
        """
        if attention_weights.ndim == 2:
            weights = attention_weights[:, head]
        else:
            weights = attention_weights

        top_idx = np.argsort(weights)[-k:][::-1]

        edges = []
        for idx in top_idx:
            src = int(edge_index[0, idx])
            tgt = int(edge_index[1, idx])
            edges.append(
                {
                    "source_node": src,
                    "target_node": tgt,
                    "source_fips": node_fips[src] if node_fips else str(src),
                    "target_fips": node_fips[tgt] if node_fips else str(tgt),
                    "attention_weight": float(weights[idx]),
                }
            )
        return edges

    def compute_attention_statistics(
        self, attention_weights: np.ndarray
    ) -> dict:
        """Compute summary statistics of attention distribution.

        Returns:
            dict with mean, std, max, min, entropy, sparsity
        """
        if attention_weights.ndim == 2:
            # Average across heads
            weights = attention_weights.mean(axis=1)
        else:
            weights = attention_weights

        # Entropy of attention distribution
        weights_norm = weights / (weights.sum() + 1e-10)
        entropy = -np.sum(weights_norm * np.log(weights_norm + 1e-10))

        # Sparsity: fraction of weights below threshold
        sparsity = np.mean(weights < 0.01)

        return {
            "mean": float(np.mean(weights)),
            "std": float(np.std(weights)),
            "max": float(np.max(weights)),
            "min": float(np.min(weights)),
            "median": float(np.median(weights)),
            "entropy": float(entropy),
            "sparsity": float(sparsity),
        }

    def attention_by_shock_direction(
        self,
        attention_weights: np.ndarray,
        edge_index: np.ndarray,
        price_shocks: np.ndarray,
    ) -> dict:
        """Analyze how attention differs based on price shock direction.

        Tests the asymmetry hypothesis: do positive and negative shocks
        lead to different attention patterns?

        Returns:
            dict with mean attention for positive vs negative source shocks
        """
        source_nodes = edge_index[0]
        source_shocks = price_shocks[source_nodes]

        if attention_weights.ndim == 2:
            weights = attention_weights.mean(axis=1)
        else:
            weights = attention_weights

        pos_mask = source_shocks > 0
        neg_mask = source_shocks < 0
        zero_mask = source_shocks == 0

        results = {
            "positive_shock_mean_attention": (
                float(np.mean(weights[pos_mask])) if pos_mask.any() else np.nan
            ),
            "negative_shock_mean_attention": (
                float(np.mean(weights[neg_mask])) if neg_mask.any() else np.nan
            ),
            "zero_shock_mean_attention": (
                float(np.mean(weights[zero_mask])) if zero_mask.any() else np.nan
            ),
            "n_positive_edges": int(pos_mask.sum()),
            "n_negative_edges": int(neg_mask.sum()),
        }

        # Statistical test for asymmetry
        if pos_mask.any() and neg_mask.any():
            from scipy.stats import mannwhitneyu

            stat, p_val = mannwhitneyu(
                weights[pos_mask], weights[neg_mask], alternative="two-sided"
            )
            results["asymmetry_p_value"] = float(p_val)
            results["asymmetry_significant"] = p_val < 0.05

        return results

    def temporal_attention_evolution(
        self,
        model,
        data_loader,
        target_fips: str,
        time_steps: list,
    ) -> list[dict]:
        """Track how attention to a target county evolves over time.

        Used for case study: shows how incoming attention increases
        as a cascade propagates.

        Args:
            model: trained CasCrop model
            data_loader: data loader with temporal batches
            target_fips: FIPS code of target county
            time_steps: list of (year, month) tuples

        Returns:
            list of dicts with time_step, incoming_attention_sum,
            top_sources, waste_probability
        """
        evolution = []

        for batch in data_loader:
            if "fips" not in batch:
                continue

            year_month = (
                batch.get("year", [None])[0],
                batch.get("month", [None])[0],
            )
            if year_month not in time_steps:
                continue

            attn_data = self.extract_attention_weights(batch)

            if "attention_weights" not in attn_data:
                continue

            node_fips = attn_data.get("node_fips", [])
            if target_fips not in node_fips:
                continue

            target_idx = node_fips.index(target_fips)
            edge_index = attn_data["edge_index"]
            weights = attn_data["attention_weights"]
            if isinstance(weights, list):
                weights = weights[-1]  # last ECMP layer

            # Incoming edges to target
            incoming_mask = edge_index[1] == target_idx
            incoming_attention = (
                weights[incoming_mask].mean(axis=-1)
                if weights.ndim == 2
                else weights[incoming_mask]
            )

            # Get waste probability for target
            with torch.no_grad():
                batch_device = {
                    k: v.to(self.device) if hasattr(v, "to") else v
                    for k, v in batch.items()
                }
                outputs = model(batch_device)
                waste_prob = (
                    torch.sigmoid(outputs["waste_logits"])[target_idx].cpu().item()
                )

            evolution.append(
                {
                    "year": year_month[0],
                    "month": year_month[1],
                    "incoming_attention_sum": float(incoming_attention.sum()),
                    "incoming_attention_mean": float(incoming_attention.mean())
                    if len(incoming_attention) > 0
                    else 0.0,
                    "num_incoming_edges": int(incoming_mask.sum()),
                    "waste_probability": waste_prob,
                }
            )

        return sorted(evolution, key=lambda x: (x["year"], x["month"]))
