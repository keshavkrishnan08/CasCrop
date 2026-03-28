"""Geographic contagion maps for case study visualization.

Generates Figure 5: cascade propagation maps showing waste probability
evolution and ECMP attention edges across US counties.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import matplotlib.patches as mpatches
from pathlib import Path
from typing import Optional
import logging

logger = logging.getLogger(__name__)


def _load_county_geometries():
    """Load US county geometries for mapping.

    Uses Census Bureau TIGER/Line shapefiles via geopandas.
    Falls back to centroid-only plotting if shapefiles unavailable.
    """
    try:
        import geopandas as gpd
        url = (
            "https://www2.census.gov/geo/tiger/GENZ2021/shp/"
            "cb_2021_us_county_500k.zip"
        )
        counties = gpd.read_file(url)
        counties["FIPS"] = counties["STATEFP"] + counties["COUNTYFP"]
        # Filter to continental US
        exclude_states = {"02", "15", "60", "66", "69", "72", "78"}
        counties = counties[~counties["STATEFP"].isin(exclude_states)]
        return counties
    except Exception as e:
        logger.warning(f"Could not load county geometries: {e}")
        return None


def figure_5_cascade_map(
    attention_evolution: list[dict],
    county_waste_probs: dict,
    county_centroids: dict,
    top_edges: list[dict],
    source_fips: str,
    target_fips: str,
    time_labels: list[str],
    output_path: str = "paper/figures/fig5_cascade_map.pdf",
):
    """Figure 5: Case study cascade map — 4-6 panel temporal progression.

    Each panel shows a time step with:
    - Counties colored by waste probability (heatmap)
    - Arrows showing top-K attention edges (width ~ alpha_ij)
    - Source county highlighted with star
    - Target county highlighted with diamond

    Args:
        attention_evolution: list of dicts per time step with attention data
        county_waste_probs: dict mapping (fips, time_idx) -> waste probability
        county_centroids: dict mapping fips -> (lat, lon)
        top_edges: list of edge dicts per time step
        source_fips: FIPS code of source county
        target_fips: FIPS code of target county
        time_labels: labels for each time panel (e.g., "2019/03", "2019/04")
        output_path: where to save
    """
    n_panels = min(len(time_labels), 6)

    fig, axes = plt.subplots(
        2, 3 if n_panels > 3 else n_panels,
        figsize=(10, 6),
        subplot_kw={"aspect": "equal"},
    )
    if n_panels <= 3:
        axes = np.array([axes]).reshape(1, -1)
    axes_flat = axes.flatten()

    norm = Normalize(vmin=0, vmax=1)
    cmap = plt.cm.YlOrRd

    for panel_idx in range(n_panels):
        ax = axes_flat[panel_idx]
        time_label = time_labels[panel_idx]
        ax.set_title(time_label, fontsize=9, fontweight="bold")

        # Plot county centroids colored by waste probability
        for fips, (lat, lon) in county_centroids.items():
            prob = county_waste_probs.get((fips, panel_idx), 0.0)
            color = cmap(norm(prob))
            size = 3 if fips not in (source_fips, target_fips) else 15
            marker = "o"
            if fips == source_fips:
                marker = "*"
                size = 60
            elif fips == target_fips:
                marker = "D"
                size = 30

            ax.scatter(lon, lat, c=[color], s=size, marker=marker,
                      edgecolors="black", linewidths=0.3, zorder=3)

        # Plot attention edges as arrows
        if panel_idx < len(top_edges):
            for edge in top_edges[panel_idx]:
                src_fips = edge.get("source_fips", "")
                tgt_fips = edge.get("target_fips", "")
                weight = edge.get("attention_weight", 0)

                if src_fips in county_centroids and tgt_fips in county_centroids:
                    src_lat, src_lon = county_centroids[src_fips]
                    tgt_lat, tgt_lon = county_centroids[tgt_fips]

                    ax.annotate(
                        "",
                        xy=(tgt_lon, tgt_lat),
                        xytext=(src_lon, src_lat),
                        arrowprops=dict(
                            arrowstyle="->",
                            color="red",
                            alpha=min(weight * 5, 0.8),
                            linewidth=max(weight * 10, 0.5),
                        ),
                        zorder=2,
                    )

        ax.set_xlim(-130, -65)
        ax.set_ylim(24, 50)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    # Hide unused panels
    for i in range(n_panels, len(axes_flat)):
        axes_flat[i].set_visible(False)

    # Colorbar
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes_flat[:n_panels].tolist(), shrink=0.6,
                        label="Waste Probability", pad=0.02)

    # Legend
    legend_elements = [
        plt.scatter([], [], marker="*", c="gold", s=60, edgecolors="black",
                   linewidths=0.5, label="Source County"),
        plt.scatter([], [], marker="D", c="cyan", s=30, edgecolors="black",
                   linewidths=0.5, label="Target County"),
    ]

    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, format="pdf", dpi=300)
    plt.close(fig)
    logger.info(f"Figure 5 saved to {output_path}")


def figure_5b_timeseries(
    attention_evolution: list[dict],
    price_trajectory: pd.DataFrame,
    source_fips: str,
    target_fips: str,
    output_path: str = "paper/figures/fig5b_timeseries.pdf",
):
    """Figure 5b: Case study time series.

    Shows price trajectory overlaid with waste probability curves
    and attention weight evolution.
    """
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(6, 5), sharex=True)

    time_idx = range(len(attention_evolution))
    time_labels = [
        f"{e['year']}/{e['month']:02d}" for e in attention_evolution
    ]

    # Panel 1: Price trajectory
    if len(price_trajectory) > 0:
        ax1.plot(price_trajectory.index, price_trajectory.values,
                "k-", linewidth=1.5)
        ax1.set_ylabel("Commodity Price ($)")
        ax1.set_title("Price Trajectory", fontsize=9)
        ax1.grid(alpha=0.3)

    # Panel 2: Waste probability
    waste_probs = [e["waste_probability"] for e in attention_evolution]
    ax2.plot(time_idx, waste_probs, "o-", color="#e74c3c",
            label="CasCrop", markersize=4)
    ax2.axhline(0.5, color="gray", linestyle="--", alpha=0.5, linewidth=0.5)
    ax2.set_ylabel("Waste Probability")
    ax2.set_title(f"Target County ({target_fips})", fontsize=9)
    ax2.legend(frameon=False, fontsize=7)
    ax2.grid(alpha=0.3)
    ax2.set_ylim(0, 1)

    # Panel 3: Incoming attention
    attn_sums = [e["incoming_attention_sum"] for e in attention_evolution]
    ax3.bar(time_idx, attn_sums, color="#3498db", alpha=0.7)
    ax3.set_ylabel("Incoming Attention")
    ax3.set_title("ECMP Attention from Source", fontsize=9)
    ax3.set_xticks(time_idx)
    ax3.set_xticklabels(time_labels, rotation=45, ha="right", fontsize=6)
    ax3.grid(alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_path, format="pdf", dpi=300)
    plt.close(fig)
    logger.info(f"Figure 5b saved to {output_path}")
