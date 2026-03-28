"""Attention weight visualization — heatmaps and distributions.

Generates Figure S4 (attention weight distributions) and
supporting attention visualizations for case studies.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


def plot_attention_heatmap(
    attention_matrix: np.ndarray,
    source_labels: list[str],
    target_labels: list[str],
    title: str = "ECMP Attention Weights",
    output_path: str = "paper/figures/attention_heatmap.pdf",
    top_k: int = 30,
):
    """Plot attention weight heatmap between county pairs.

    Shows which source counties most strongly influence which targets.

    Args:
        attention_matrix: [n_targets, n_sources] attention weights
        source_labels: labels for source counties
        target_labels: labels for target counties
        title: figure title
        output_path: where to save
        top_k: only show top-K most attended source/target pairs
    """
    # Select top-K most interesting rows and columns
    col_importance = attention_matrix.sum(axis=0)
    row_importance = attention_matrix.sum(axis=1)
    top_cols = np.argsort(col_importance)[-top_k:]
    top_rows = np.argsort(row_importance)[-top_k:]

    submatrix = attention_matrix[np.ix_(top_rows, top_cols)]
    sub_src = [source_labels[i] for i in top_cols]
    sub_tgt = [target_labels[i] for i in top_rows]

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        submatrix,
        xticklabels=sub_src,
        yticklabels=sub_tgt,
        cmap="YlOrRd",
        ax=ax,
        cbar_kws={"label": "Attention Weight"},
    )
    ax.set_xlabel("Source Counties")
    ax.set_ylabel("Target Counties")
    ax.set_title(title)
    plt.xticks(fontsize=5, rotation=90)
    plt.yticks(fontsize=5)
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, format="pdf", dpi=300)
    plt.close(fig)
    logger.info(f"Attention heatmap saved to {output_path}")


def plot_attention_distribution(
    attention_weights_list: dict[str, np.ndarray],
    output_path: str = "paper/figures/figS4_attention_dist.pdf",
):
    """Supplementary Figure S4: Attention weight distributions.

    Compares attention distributions across models (Geo GAT, Symmetric ECMP,
    Full CasCrop) and across attention heads.

    Args:
        attention_weights_list: dict mapping model_name -> flattened attention weights
        output_path: where to save
    """
    fig, axes = plt.subplots(1, len(attention_weights_list), figsize=(10, 3),
                             sharey=True)
    if len(attention_weights_list) == 1:
        axes = [axes]

    for ax, (model_name, weights) in zip(axes, attention_weights_list.items()):
        ax.hist(weights, bins=50, density=True, alpha=0.7,
                color="#3498db", edgecolor="black", linewidth=0.3)
        ax.set_xlabel("Attention Weight")
        ax.set_title(model_name, fontsize=8)
        ax.set_xlim(0, max(0.1, np.percentile(weights, 99)))

        # Stats overlay
        ax.axvline(np.mean(weights), color="red", linestyle="--",
                  linewidth=0.8, label=f"Mean: {np.mean(weights):.4f}")
        ax.axvline(np.median(weights), color="green", linestyle="--",
                  linewidth=0.8, label=f"Median: {np.median(weights):.4f}")
        ax.legend(fontsize=5, frameon=False)

    axes[0].set_ylabel("Density")
    plt.tight_layout()
    fig.savefig(output_path, format="pdf", dpi=300)
    plt.close(fig)
    logger.info(f"Figure S4 saved to {output_path}")


def plot_attention_by_head(
    attention_weights: np.ndarray,
    num_heads: int = 4,
    output_path: str = "paper/figures/attention_per_head.pdf",
):
    """Plot attention distribution per head to see if heads specialize.

    Args:
        attention_weights: [num_edges, num_heads]
        num_heads: number of attention heads
        output_path: where to save
    """
    fig, axes = plt.subplots(1, num_heads, figsize=(10, 2.5), sharey=True)

    for h in range(num_heads):
        ax = axes[h]
        w = attention_weights[:, h]
        ax.hist(w, bins=50, density=True, alpha=0.7, color=f"C{h}",
                edgecolor="black", linewidth=0.3)
        ax.set_title(f"Head {h+1}", fontsize=8)
        ax.set_xlabel("Weight")
        ax.text(0.95, 0.95, f"Entropy: {_entropy(w):.2f}",
                transform=ax.transAxes, ha="right", va="top", fontsize=6)

    axes[0].set_ylabel("Density")
    plt.tight_layout()
    fig.savefig(output_path, format="pdf", dpi=300)
    plt.close(fig)
    logger.info(f"Attention per head plot saved to {output_path}")


def _entropy(weights: np.ndarray) -> float:
    """Compute entropy of attention distribution."""
    w = weights / (weights.sum() + 1e-10)
    return -np.sum(w * np.log(w + 1e-10))


def plot_shock_vs_attention(
    price_shocks: np.ndarray,
    attention_weights: np.ndarray,
    edge_index: np.ndarray,
    output_path: str = "paper/figures/shock_vs_attention.pdf",
):
    """Scatter plot of source node price shock vs outgoing attention.

    Demonstrates the asymmetry: negative shocks should lead to
    higher attention (more contagion risk).
    """
    source_nodes = edge_index[0]
    source_shocks = price_shocks[source_nodes]

    if attention_weights.ndim == 2:
        mean_attn = attention_weights.mean(axis=1)
    else:
        mean_attn = attention_weights

    fig, ax = plt.subplots(figsize=(4, 3))

    # Bin by shock direction
    neg_mask = source_shocks < 0
    pos_mask = source_shocks > 0

    ax.scatter(source_shocks[neg_mask], mean_attn[neg_mask],
              s=1, alpha=0.3, c="#e74c3c", label="Negative Shock", rasterized=True)
    ax.scatter(source_shocks[pos_mask], mean_attn[pos_mask],
              s=1, alpha=0.3, c="#3498db", label="Positive Shock", rasterized=True)

    # Add trend lines
    if neg_mask.sum() > 10:
        z = np.polyfit(source_shocks[neg_mask], mean_attn[neg_mask], 1)
        p = np.poly1d(z)
        x_neg = np.linspace(source_shocks[neg_mask].min(), 0, 100)
        ax.plot(x_neg, p(x_neg), "--", color="#e74c3c", linewidth=1.5)

    if pos_mask.sum() > 10:
        z = np.polyfit(source_shocks[pos_mask], mean_attn[pos_mask], 1)
        p = np.poly1d(z)
        x_pos = np.linspace(0, source_shocks[pos_mask].max(), 100)
        ax.plot(x_pos, p(x_pos), "--", color="#3498db", linewidth=1.5)

    ax.axvline(0, color="gray", linestyle=":", linewidth=0.5)
    ax.set_xlabel("Source Node Price Shock (Δp)")
    ax.set_ylabel("Outgoing Attention Weight")
    ax.set_title("Asymmetric Shock-Attention Relationship")
    ax.legend(fontsize=6, frameon=False)

    plt.tight_layout()
    fig.savefig(output_path, format="pdf", dpi=300)
    plt.close(fig)
    logger.info(f"Shock vs attention plot saved to {output_path}")
