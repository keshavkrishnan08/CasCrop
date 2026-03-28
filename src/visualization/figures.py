"""Publication-quality figure generation for CasCrop paper.

All figures exported at 300dpi as PDF/SVG.
Style: clean, professional, Nature-family compatible.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
import seaborn as sns
from pathlib import Path
from typing import Optional
import logging

logger = logging.getLogger(__name__)

# Publication style
PAPER_RC = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica"],
    "font.size": 8,
    "axes.labelsize": 9,
    "axes.titlesize": 10,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
    "axes.linewidth": 0.5,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "lines.linewidth": 1.0,
}

MODEL_COLORS = {
    "Row 1: Local Only": "#7f8c8d",
    "Row 2: Local + Econ": "#3498db",
    "Row 3: Geo GAT": "#e67e22",
    "Row 4: Symmetric ECMP": "#9b59b6",
    "Row 5: CasCrop": "#e74c3c",
    "Random Forest": "#95a5a6",
    "XGBoost": "#2ecc71",
    "LSTM": "#1abc9c",
}

SIGNIFICANCE_STARS = {0.001: "***", 0.01: "**", 0.05: "*", 1.0: "n.s."}


def _get_stars(p_value: float) -> str:
    for threshold, stars in SIGNIFICANCE_STARS.items():
        if p_value < threshold:
            return stars
    return "n.s."


def figure_3_ablation_results(
    summary_df: pd.DataFrame,
    p_values: Optional[dict] = None,
    output_path: str = "paper/figures/fig3_ablation.pdf",
):
    """Figure 3: Main ablation results — grouped bar chart.

    Shows AUC-ROC and AUC-PR for all ablation rows with error bars
    and significance stars.
    """
    with plt.rc_context(PAPER_RC):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 3.0))

        models = summary_df["model"].tolist()
        x = np.arange(len(models))
        width = 0.6

        # Panel A: AUC-ROC
        means = summary_df["auc_roc_mean"].values
        stds = summary_df["auc_roc_std"].values
        colors = [MODEL_COLORS.get(m.split(": ")[-1], "#7f8c8d") for m in models]

        bars = ax1.bar(x, means, width, yerr=stds, capsize=3, color=colors,
                       edgecolor="black", linewidth=0.5, zorder=3)
        ax1.set_ylabel("AUC-ROC")
        ax1.set_xticks(x)
        ax1.set_xticklabels(
            [m.replace("Row ", "R") for m in models], rotation=45, ha="right"
        )
        ax1.set_ylim(0.5, 1.0)
        ax1.grid(axis="y", alpha=0.3, zorder=0)
        ax1.set_title("(a) AUC-ROC", fontweight="bold")

        # Add significance stars
        if p_values:
            ref_idx = len(models) - 1  # CasCrop is last
            for i, model in enumerate(models[:-1]):
                key = f"cascrop_vs_{model.lower().replace(' ', '_')}"
                if key in p_values:
                    stars = _get_stars(p_values[key])
                    ax1.text(
                        i, means[i] + stds[i] + 0.01, stars,
                        ha="center", va="bottom", fontsize=6,
                    )

        # Panel B: AUC-PR
        if "auc_pr_mean" in summary_df.columns:
            means_pr = summary_df["auc_pr_mean"].values
            stds_pr = summary_df["auc_pr_std"].values

            ax2.bar(x, means_pr, width, yerr=stds_pr, capsize=3, color=colors,
                    edgecolor="black", linewidth=0.5, zorder=3)
            ax2.set_ylabel("AUC-PR")
            ax2.set_xticks(x)
            ax2.set_xticklabels(
                [m.replace("Row ", "R") for m in models], rotation=45, ha="right"
            )
            ax2.set_ylim(0, 1.0)
            ax2.grid(axis="y", alpha=0.3, zorder=0)
            ax2.set_title("(b) AUC-PR", fontweight="bold")

        plt.tight_layout()
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, format="pdf")
        plt.close(fig)
        logger.info(f"Figure 3 saved to {output_path}")


def figure_4_subgroup_heatmap(
    subgroup_results: dict,
    output_path: str = "paper/figures/fig4_subgroup.pdf",
):
    """Figure 4: Sub-group analysis heatmap.

    Rows = models, Columns = crop types / causes / regions.
    Cell color = AUC or F1.
    """
    with plt.rc_context(PAPER_RC):
        # Build matrix from results
        models = list(subgroup_results.keys())
        subgroups = list(subgroup_results[models[0]].keys())
        matrix = np.zeros((len(models), len(subgroups)))

        for i, model in enumerate(models):
            for j, sg in enumerate(subgroups):
                val = subgroup_results[model].get(sg, {}).get("auc_roc", np.nan)
                matrix[i, j] = val

        fig, ax = plt.subplots(figsize=(7.0, 3.5))
        im = ax.imshow(matrix, cmap="RdYlGn", aspect="auto", vmin=0.5, vmax=1.0)

        ax.set_xticks(range(len(subgroups)))
        ax.set_xticklabels(subgroups, rotation=45, ha="right")
        ax.set_yticks(range(len(models)))
        ax.set_yticklabels(models)

        # Annotate cells
        for i in range(len(models)):
            for j in range(len(subgroups)):
                val = matrix[i, j]
                if not np.isnan(val):
                    color = "white" if val < 0.65 else "black"
                    ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                            color=color, fontsize=6)

        plt.colorbar(im, ax=ax, label="AUC-ROC", shrink=0.8)
        ax.set_title("Sub-group Performance Analysis")
        plt.tight_layout()
        fig.savefig(output_path, format="pdf")
        plt.close(fig)
        logger.info(f"Figure 4 saved to {output_path}")


def figure_6_disentanglement(
    z_bio: np.ndarray,
    z_econ: np.ndarray,
    weather_labels: np.ndarray,
    price_labels: np.ndarray,
    output_path: str = "paper/figures/fig6_disentanglement.pdf",
):
    """Figure 6: Disentanglement visualization.

    (a) t-SNE of z_bio colored by weather conditions
    (b) t-SNE of z_econ colored by price levels
    Shows the two encoders capture different information.
    """
    from sklearn.manifold import TSNE

    with plt.rc_context(PAPER_RC):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 3.0))

        # Subsample for speed
        n = min(5000, len(z_bio))
        idx = np.random.choice(len(z_bio), n, replace=False)

        # t-SNE of z_bio
        tsne_bio = TSNE(n_components=2, random_state=42, perplexity=30)
        z_bio_2d = tsne_bio.fit_transform(z_bio[idx])

        scatter1 = ax1.scatter(
            z_bio_2d[:, 0], z_bio_2d[:, 1],
            c=weather_labels[idx], cmap="coolwarm",
            s=1, alpha=0.5, rasterized=True,
        )
        ax1.set_title("(a) Biophysical Encoder (z_bio)", fontweight="bold")
        ax1.set_xlabel("t-SNE 1")
        ax1.set_ylabel("t-SNE 2")
        plt.colorbar(scatter1, ax=ax1, label="Weather Condition")

        # t-SNE of z_econ
        tsne_econ = TSNE(n_components=2, random_state=42, perplexity=30)
        z_econ_2d = tsne_econ.fit_transform(z_econ[idx])

        scatter2 = ax2.scatter(
            z_econ_2d[:, 0], z_econ_2d[:, 1],
            c=price_labels[idx], cmap="RdYlGn_r",
            s=1, alpha=0.5, rasterized=True,
        )
        ax2.set_title("(b) Economic Encoder (z_econ)", fontweight="bold")
        ax2.set_xlabel("t-SNE 1")
        ax2.set_ylabel("t-SNE 2")
        plt.colorbar(scatter2, ax=ax2, label="Price Level")

        plt.tight_layout()
        fig.savefig(output_path, format="pdf")
        plt.close(fig)
        logger.info(f"Figure 6 saved to {output_path}")


def figure_7_economic_impact(
    lead_time_df: pd.DataFrame,
    output_path: str = "paper/figures/fig7_economic_impact.pdf",
):
    """Figure 7: Economic impact — cumulative preventable waste vs lead time.

    Shows CasCrop vs local-only model: how much more money is
    capturable with earlier warning.
    """
    with plt.rc_context(PAPER_RC):
        fig, ax = plt.subplots(figsize=(3.5, 3.0))

        ax.plot(
            lead_time_df["lead_weeks"],
            lead_time_df["cascrop_capturable"] / 1e9,
            "o-", color="#e74c3c", label="CasCrop", markersize=4,
        )
        ax.plot(
            lead_time_df["lead_weeks"],
            lead_time_df["baseline_capturable"] / 1e9,
            "s--", color="#7f8c8d", label="Local Only", markersize=4,
        )

        ax.fill_between(
            lead_time_df["lead_weeks"],
            lead_time_df["baseline_capturable"] / 1e9,
            lead_time_df["cascrop_capturable"] / 1e9,
            alpha=0.2, color="#e74c3c", label="Improvement",
        )

        ax.set_xlabel("Lead Time (weeks)")
        ax.set_ylabel("Capturable Waste ($B)")
        ax.legend(frameon=False)
        ax.grid(alpha=0.3)
        ax.set_title("Early Warning Value")

        plt.tight_layout()
        fig.savefig(output_path, format="pdf")
        plt.close(fig)
        logger.info(f"Figure 7 saved to {output_path}")


def figure_s2_roc_curves(
    model_roc_data: dict[str, tuple[np.ndarray, np.ndarray, float]],
    output_path: str = "paper/figures/figS2_roc_curves.pdf",
):
    """Supplementary Figure S2: Overlaid ROC curves for all models."""
    with plt.rc_context(PAPER_RC):
        fig, ax = plt.subplots(figsize=(3.5, 3.5))

        for model_name, (fpr, tpr, auc) in model_roc_data.items():
            color = MODEL_COLORS.get(model_name, "#333333")
            ax.plot(fpr, tpr, color=color, label=f"{model_name} ({auc:.3f})")

        ax.plot([0, 1], [0, 1], "k--", alpha=0.3, linewidth=0.5)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curves")
        ax.legend(fontsize=5, frameon=False, loc="lower right")
        ax.set_aspect("equal")

        plt.tight_layout()
        fig.savefig(output_path, format="pdf")
        plt.close(fig)
        logger.info(f"Figure S2 saved to {output_path}")


def figure_s3_calibration(
    model_calibration_data: dict[str, tuple[np.ndarray, np.ndarray]],
    output_path: str = "paper/figures/figS3_calibration.pdf",
):
    """Supplementary Figure S3: Calibration plots for all models."""
    with plt.rc_context(PAPER_RC):
        fig, ax = plt.subplots(figsize=(3.5, 3.5))

        for model_name, (pred_probs, true_freqs) in model_calibration_data.items():
            color = MODEL_COLORS.get(model_name, "#333333")
            ax.plot(pred_probs, true_freqs, "o-", color=color,
                    label=model_name, markersize=3)

        ax.plot([0, 1], [0, 1], "k--", alpha=0.3, linewidth=0.5)
        ax.set_xlabel("Predicted Probability")
        ax.set_ylabel("Observed Frequency")
        ax.set_title("Calibration Plot")
        ax.legend(fontsize=5, frameon=False)

        plt.tight_layout()
        fig.savefig(output_path, format="pdf")
        plt.close(fig)
        logger.info(f"Figure S3 saved to {output_path}")
