#!/usr/bin/env python3
"""
Generate two publication-quality overview figures for the CasCrop paper.

Produces:
    fig0_graphical_abstract.pdf  — Single-page visual summary of the paper
    fig1_architecture_v2.pdf     — Clean architecture diagram (replaces clipped version)

Run:
    python generate_overview_figures.py
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, PathPatch
import matplotlib.patheffects as pe
import matplotlib.path as mpath
import numpy as np
import os

OUT_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Shared style ──────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Arial", "DejaVu Sans"],
    "font.size": 10,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.1,
})

# Palette
RED    = "#C0392B"
BLUE   = "#2471A3"
GOLD   = "#D4AC0D"
GREEN  = "#1E8449"
ORANGE = "#D35400"
PURPLE = "#7D3C98"
GRAY   = "#566573"
LIGHT_RED  = "#FADBD8"
LIGHT_BLUE = "#D6EAF8"
LIGHT_GOLD = "#FEF9E7"
PANEL_BG   = "#F8F9FA"


# ═════════════════════════════════════════════════════════════════════════════
# Helper: draw a rounded box and return its center
# ═════════════════════════════════════════════════════════════════════════════

def rounded_box(ax, x, y, w, h, color, alpha=0.15, lw=1.5, zorder=2):
    """Draw a rounded rectangle, return (cx, cy)."""
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02",
        linewidth=lw,
        edgecolor=color,
        facecolor=color,
        alpha=alpha,
        zorder=zorder,
        transform=ax.transData,
        clip_on=False,
    )
    ax.add_patch(box)
    return x + w / 2, y + h / 2


def solid_box(ax, x, y, w, h, facecolor, edgecolor, lw=1.5, alpha=1.0, zorder=2):
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02",
        linewidth=lw,
        edgecolor=edgecolor,
        facecolor=facecolor,
        alpha=alpha,
        zorder=zorder,
        clip_on=False,
    )
    ax.add_patch(box)
    return x + w / 2, y + h / 2


def arrow(ax, x0, y0, x1, y1, color=GRAY, lw=1.5, zorder=3,
          arrowstyle="-|>", mutation_scale=12):
    ax.annotate(
        "", xy=(x1, y1), xytext=(x0, y0),
        arrowprops=dict(
            arrowstyle=arrowstyle,
            color=color,
            lw=lw,
            mutation_scale=mutation_scale,
        ),
        zorder=zorder,
        annotation_clip=False,
    )


# ═════════════════════════════════════════════════════════════════════════════
# Figure 0: Graphical Abstract
# ═════════════════════════════════════════════════════════════════════════════

def make_graphical_abstract():
    fig = plt.figure(figsize=(12, 4.5))
    fig.patch.set_facecolor("white")

    # ── Top banner ────────────────────────────────────────────────────────────
    fig.text(
        0.5, 0.965,
        "Polarity-Routed Cascade Diffusion for Crop Waste Prediction",
        ha="center", va="top",
        fontsize=13, fontweight="bold", color="#1a1a2e",
        transform=fig.transFigure,
    )

    # ── Three panel axes ─────────────────────────────────────────────────────
    # [left-panel] [center-panel] [right-panel]
    ax_l = fig.add_axes([0.01, 0.04, 0.31, 0.87])   # LEFT
    ax_c = fig.add_axes([0.35, 0.04, 0.30, 0.87])   # CENTER
    ax_r = fig.add_axes([0.68, 0.04, 0.31, 0.87])   # RIGHT

    for ax in (ax_l, ax_c, ax_r):
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")

    # ── dividers ─────────────────────────────────────────────────────────────
    for xpos in (0.335, 0.665):
        fig.add_artist(plt.Line2D(
            [xpos, xpos], [0.06, 0.93],
            transform=fig.transFigure,
            color="#CCCCCC", lw=1.0, zorder=0,
        ))

    # =========================================================================
    # LEFT PANEL — "The Problem"
    # =========================================================================
    ax_l.text(0.5, 0.96, "The Problem", ha="center", va="top",
              fontsize=11, fontweight="bold", color=RED)

    # -- Simple US silhouette (polygon approximation) -------------------------
    # Rough outline: (x, y) in axes coords, scaled to a small region
    us_pts = np.array([
        [0.08, 0.68], [0.12, 0.72], [0.22, 0.74], [0.30, 0.76],
        [0.38, 0.75], [0.50, 0.77], [0.60, 0.76], [0.72, 0.74],
        [0.80, 0.70], [0.85, 0.65], [0.82, 0.60], [0.75, 0.57],
        [0.65, 0.55], [0.55, 0.53], [0.45, 0.54], [0.35, 0.53],
        [0.22, 0.55], [0.12, 0.58], [0.07, 0.62], [0.08, 0.68],
    ])
    us_poly = plt.Polygon(us_pts, closed=True,
                          facecolor="#EBF5FB", edgecolor="#2471A3",
                          linewidth=1.2, zorder=1, alpha=0.7)
    ax_l.add_patch(us_poly)

    # County dots
    county_xy = [
        (0.20, 0.65), (0.35, 0.68), (0.50, 0.63),
        (0.60, 0.67), (0.45, 0.57), (0.70, 0.61),
    ]
    for (cx, cy) in county_xy:
        ax_l.plot(cx, cy, "o", color=RED, ms=5, zorder=4)

    # Cascade arrows between counties
    shock_pairs = [
        (0, 1), (1, 2), (2, 3), (1, 4), (4, 5),
    ]
    for i, j in shock_pairs:
        x0, y0 = county_xy[i]
        x1, y1 = county_xy[j]
        ax_l.annotate(
            "", xy=(x1, y1), xytext=(x0, y0),
            arrowprops=dict(arrowstyle="-|>", color=RED,
                            lw=1.0, mutation_scale=8),
            annotation_clip=False, zorder=3,
        )

    # Label one arrow
    ax_l.text(0.38, 0.73, "economic\nshock", ha="center", va="bottom",
              fontsize=7, color=RED, style="italic")

    # Stats text
    ax_l.text(0.5, 0.48, "$19B in annual crop waste", ha="center", va="top",
              fontsize=9, fontweight="bold", color="#1a1a2e")
    ax_l.text(0.5, 0.40,
              "Counties aren't independent —\nshocks cascade across networks",
              ha="center", va="top", fontsize=8, color=GRAY,
              multialignment="center")

    # =========================================================================
    # CENTER PANEL — "Our Insight"
    # =========================================================================
    ax_c.text(0.5, 0.96, "Our Insight", ha="center", va="top",
              fontsize=11, fontweight="bold", color=BLUE)

    # ── Sub-panel top: Negative shocks ───────────────────────────────────────
    top_y_top = 0.88
    top_y_bot = 0.52

    # Background box
    top_bg = FancyBboxPatch((0.04, top_y_bot), 0.92, top_y_top - top_y_bot,
                             boxstyle="round,pad=0.01",
                             facecolor=LIGHT_RED, edgecolor=RED,
                             linewidth=1.0, alpha=0.5, zorder=1)
    ax_c.add_patch(top_bg)

    ax_c.text(0.5, top_y_top - 0.01, "Negative shocks (oversupply)",
              ha="center", va="top", fontsize=8, fontweight="bold", color=RED)

    # Mini commodity graph nodes
    n1 = (0.25, 0.67); n2 = (0.50, 0.72); n3 = (0.75, 0.67)
    for nx, ny in [n1, n2, n3]:
        ax_c.plot(nx, ny, "s", color=RED, ms=7, zorder=4)
    for (ax0, ay0), (ax1, ay1) in [(n1, n2), (n2, n3), (n1, n3)]:
        ax_c.annotate("", xy=(ax1, ay1), xytext=(ax0, ay0),
                      arrowprops=dict(arrowstyle="-|>", color=RED,
                                      lw=0.9, mutation_scale=7),
                      annotation_clip=False, zorder=3)
    ax_c.text(0.50, 0.59, "Commodity Network\n(corn -> corn)",
              ha="center", va="top", fontsize=7, color=RED,
              multialignment="center")

    # ── Sub-panel bottom: Positive signals ───────────────────────────────────
    bot_y_top = 0.49
    bot_y_bot = 0.14

    bot_bg = FancyBboxPatch((0.04, bot_y_bot), 0.92, bot_y_top - bot_y_bot,
                             boxstyle="round,pad=0.01",
                             facecolor=LIGHT_BLUE, edgecolor=BLUE,
                             linewidth=1.0, alpha=0.5, zorder=1)
    ax_c.add_patch(bot_bg)

    ax_c.text(0.5, bot_y_top - 0.01, "Positive signals (demand)",
              ha="center", va="top", fontsize=8, fontweight="bold", color=BLUE)

    # Mini geographic graph nodes
    m1 = (0.22, 0.31); m2 = (0.50, 0.38); m3 = (0.78, 0.31)
    m4 = (0.36, 0.22); m5 = (0.64, 0.22)
    geo_nodes = [m1, m2, m3, m4, m5]
    geo_edges = [(0,1),(1,2),(0,3),(1,3),(1,4),(2,4),(3,4)]
    for mx, my in geo_nodes:
        ax_c.plot(mx, my, "o", color=BLUE, ms=5, zorder=4)
    for i, j in geo_edges:
        ax0, ay0 = geo_nodes[i]; ax1, ay1 = geo_nodes[j]
        ax_c.plot([ax0, ax1], [ay0, ay1], "-", color=BLUE,
                  lw=0.7, alpha=0.5, zorder=2)
    ax_c.text(0.50, 0.15, "Geographic Network\n(proximity)",
              ha="center", va="top", fontsize=7, color=BLUE,
              multialignment="center")

    # Key insight label
    ax_c.text(0.5, 0.06,
              "Different polarities -> Different topologies",
              ha="center", va="bottom", fontsize=8.5,
              fontweight="bold", color="#1a1a2e")

    # =========================================================================
    # RIGHT PANEL — "Results"
    # =========================================================================
    ax_r.text(0.5, 0.96, "Results", ha="center", va="top",
              fontsize=11, fontweight="bold", color=GREEN)

    # Bar chart
    bar_ax = fig.add_axes([0.70, 0.47, 0.27, 0.36])
    bar_ax.set_facecolor("white")
    models  = ["Local\nOnly", "Polarity-\nRouted"]
    aucs    = [0.917, 0.935]
    colors  = [GRAY, GOLD]
    bars = bar_ax.bar(models, aucs, color=colors, width=0.45,
                      edgecolor="white", linewidth=0.5, zorder=3)
    bar_ax.set_ylim(0.900, 0.945)
    bar_ax.set_ylabel("AUC-ROC", fontsize=8)
    bar_ax.yaxis.set_tick_params(labelsize=7)
    bar_ax.xaxis.set_tick_params(labelsize=7.5)
    bar_ax.spines["top"].set_visible(False)
    bar_ax.spines["right"].set_visible(False)
    bar_ax.set_title("Model Comparison", fontsize=8.5, pad=4)
    bar_ax.yaxis.grid(True, color="#EEEEEE", zorder=0)

    # Value labels on bars
    for bar, val in zip(bars, aucs):
        bar_ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.0005,
                    f"{val:.3f}", ha="center", va="bottom",
                    fontsize=7.5, fontweight="bold")

    # Improvement arrow
    x0_bar = bars[0].get_x() + bars[0].get_width()
    x1_bar = bars[1].get_x()
    y_arr  = 0.938
    bar_ax.annotate(
        "", xy=(x1_bar, y_arr), xytext=(x0_bar, y_arr),
        arrowprops=dict(arrowstyle="-|>", color=GREEN,
                        lw=1.2, mutation_scale=8),
    )
    bar_ax.text((x0_bar + x1_bar) / 2, y_arr + 0.001,
                "+1.8 AUC\n(p<0.001)",
                ha="center", va="bottom", fontsize=6.5,
                color=GREEN, fontweight="bold")

    # Stats text
    ax_r.text(0.5, 0.40,
              "638,622 USDA records",
              ha="center", va="top", fontsize=8.5,
              fontweight="bold", color="#1a1a2e")
    ax_r.text(0.5, 0.32,
              "2,130 counties  •  3 commodities",
              ha="center", va="top", fontsize=8, color=GRAY)
    ax_r.text(0.5, 0.24,
              "Temporal out-of-sample validation",
              ha="center", va="top", fontsize=8, color=GRAY,
              style="italic")

    # ── Panel titles (background strip) ──────────────────────────────────────
    # already placed as text above

    fig.savefig(os.path.join(OUT_DIR, "fig0_graphical_abstract.pdf"),
                dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("Saved: fig0_graphical_abstract.pdf")


# ═════════════════════════════════════════════════════════════════════════════
# Figure 1: Architecture Diagram (v2, clean)
# ═════════════════════════════════════════════════════════════════════════════

def make_architecture():
    fig, ax = plt.subplots(figsize=(14, 5))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 5)
    ax.axis("off")

    # ── Layout constants ──────────────────────────────────────────────────────
    BOX_H   = 3.6    # tall boxes height
    BOX_Y   = 0.7    # bottom edge of main boxes
    ARROW_Y = BOX_Y + BOX_H / 2   # vertical center for horizontal arrows

    # BOX 1: Local Features (green)
    B1_X, B1_W = 0.3, 2.2
    # BOX 2: Polarity-Routed (orange, larger)
    B2_X, B2_W = 3.6, 6.0
    # BOX 3: MLP Classifier (purple)
    B3_X, B3_W = 10.7, 1.9
    # BOX 4: Output (red/green gradient-ish)
    B4_X, B4_W = 13.15, 0.65

    # ── BOX 1 — Local Features ────────────────────────────────────────────────
    b1_face = "#D5F5E3"
    b1_edge = GREEN
    solid_box(ax, B1_X, BOX_Y, B1_W, BOX_H,
              facecolor=b1_face, edgecolor=b1_edge, lw=2.0, alpha=1.0)

    ax.text(B1_X + B1_W / 2, BOX_Y + BOX_H - 0.22,
            "Local Features",
            ha="center", va="top", fontsize=11, fontweight="bold",
            color=GREEN)

    items_b1 = [
        ("Bio", "19 features", "#A9DFBF"),
        ("Econ", "8 features",  "#A9DFBF"),
        ("Hist", "3 features",  "#A9DFBF"),
    ]
    item_y = BOX_Y + BOX_H - 0.75
    for label, sub, fc in items_b1:
        chip = FancyBboxPatch(
            (B1_X + 0.18, item_y - 0.28), B1_W - 0.36, 0.44,
            boxstyle="round,pad=0.04",
            facecolor=fc, edgecolor=GREEN, linewidth=0.8,
            zorder=3,
        )
        ax.add_patch(chip)
        ax.text(B1_X + B1_W / 2, item_y,
                f"{label}  ({sub})",
                ha="center", va="center", fontsize=9,
                color="#1a4a1a", fontweight="bold")
        item_y -= 0.62

    ax.text(B1_X + B1_W / 2, BOX_Y + 0.22,
            "30 features",
            ha="center", va="bottom", fontsize=9,
            color=GREEN, fontweight="bold",
            style="italic")

    # ── ARROW 1 → 2 ───────────────────────────────────────────────────────────
    arrow(ax, B1_X + B1_W + 0.05, ARROW_Y,
          B2_X - 0.05, ARROW_Y,
          color=GRAY, lw=2.0, mutation_scale=14)

    # ── BOX 2 — Polarity-Routed Cascade Diffusion ─────────────────────────────
    b2_face = "#FEF9E7"
    b2_edge = ORANGE
    solid_box(ax, B2_X, BOX_Y, B2_W, BOX_H,
              facecolor=b2_face, edgecolor=b2_edge, lw=2.0, alpha=1.0)

    ax.text(B2_X + B2_W / 2, BOX_Y + BOX_H - 0.22,
            "Polarity-Routed Cascade Diffusion",
            ha="center", va="top", fontsize=11, fontweight="bold",
            color=ORANGE)

    # Divider line inside box
    mid_y = BOX_Y + BOX_H * 0.5
    ax.plot([B2_X + 0.2, B2_X + B2_W - 0.2], [mid_y, mid_y],
            "--", color="#E59866", lw=0.8, zorder=3)

    # ─ Top path: Negative shocks (red) ─
    top_path_y = BOX_Y + BOX_H * 0.73
    # Red background strip
    red_strip = FancyBboxPatch(
        (B2_X + 0.15, top_path_y - 0.30), B2_W - 0.30, 0.54,
        boxstyle="round,pad=0.03",
        facecolor=LIGHT_RED, edgecolor=RED, linewidth=0.9, zorder=2,
    )
    ax.add_patch(red_strip)

    ax.text(B2_X + 0.40, top_path_y + 0.06,
            "Negative shocks",
            ha="left", va="center", fontsize=8.5, fontweight="bold",
            color=RED)
    ax.text(B2_X + 0.40, top_path_y - 0.17,
            "Commodity Graph  •  3 hops",
            ha="left", va="center", fontsize=8, color=RED)

    # Small arrow + graph icon
    _draw_mini_graph(ax, B2_X + B2_W - 1.20, top_path_y - 0.04,
                     node_color=RED, size=0.08)

    # ─ Bottom path: Positive signals (blue) ─
    bot_path_y = BOX_Y + BOX_H * 0.27
    blue_strip = FancyBboxPatch(
        (B2_X + 0.15, bot_path_y - 0.30), B2_W - 0.30, 0.54,
        boxstyle="round,pad=0.03",
        facecolor=LIGHT_BLUE, edgecolor=BLUE, linewidth=0.9, zorder=2,
    )
    ax.add_patch(blue_strip)

    ax.text(B2_X + 0.40, bot_path_y + 0.06,
            "Positive signals",
            ha="left", va="center", fontsize=8.5, fontweight="bold",
            color=BLUE)
    ax.text(B2_X + 0.40, bot_path_y - 0.17,
            "Geographic Graph  •  3 hops",
            ha="left", va="center", fontsize=8, color=BLUE)

    _draw_mini_graph(ax, B2_X + B2_W - 1.20, bot_path_y - 0.04,
                     node_color=BLUE, size=0.08, shape="o")

    # Features produced
    ax.text(B2_X + B2_W / 2, BOX_Y + 0.52,
            "Decay Signatures  +  Cross-Commodity  +  Neighbor Features",
            ha="center", va="center", fontsize=8, color=GRAY,
            style="italic")
    ax.text(B2_X + B2_W / 2, BOX_Y + 0.20,
            "20 cascade features  (output)",
            ha="center", va="center", fontsize=9,
            fontweight="bold", color=ORANGE)

    # ── ARROW 2 → 3 ───────────────────────────────────────────────────────────
    arrow(ax, B2_X + B2_W + 0.05, ARROW_Y,
          B3_X - 0.05, ARROW_Y,
          color=GRAY, lw=2.0, mutation_scale=14)

    # ── BOX 3 — MLP Classifier ────────────────────────────────────────────────
    b3_face = "#EBD5F8"
    b3_edge = PURPLE
    solid_box(ax, B3_X, BOX_Y, B3_W, BOX_H,
              facecolor=b3_face, edgecolor=b3_edge, lw=2.0, alpha=1.0)

    ax.text(B3_X + B3_W / 2, BOX_Y + BOX_H - 0.22,
            "MLP Classifier",
            ha="center", va="top", fontsize=11, fontweight="bold",
            color=PURPLE)

    # Layer representation
    layer_info = [
        ("50",  "in"),
        ("128", ""),
        ("64",  ""),
        ("1",   "out"),
    ]
    lx_start = B3_X + 0.22
    lx_step  = (B3_W - 0.44) / (len(layer_info) - 1)
    prev_lx = None
    for i, (n, tag) in enumerate(layer_info):
        lx = lx_start + i * lx_step
        ly = ARROW_Y + 0.05
        radius = 0.20 + 0.04 * (1 - abs(i - 1.5) / 1.5)
        c = plt.Circle((lx, ly), radius,
                        color=PURPLE, alpha=0.30 + 0.15 * (i % 2),
                        zorder=3)
        ax.add_patch(c)
        ax.text(lx, ly, n, ha="center", va="center",
                fontsize=8, fontweight="bold", color=PURPLE, zorder=4)
        if tag:
            ax.text(lx, ly - radius - 0.14, tag,
                    ha="center", va="top", fontsize=7, color=PURPLE)
        if prev_lx is not None:
            ax.plot([prev_lx + 0.20, lx - 0.20], [ly, ly],
                    "-", color=PURPLE, lw=0.8, alpha=0.4, zorder=2)
        prev_lx = lx

    ax.text(B3_X + B3_W / 2, BOX_Y + 0.36,
            "13,703 params",
            ha="center", va="center", fontsize=8.5,
            color=PURPLE, fontweight="bold", style="italic")

    # ── ARROW 3 → 4 ───────────────────────────────────────────────────────────
    arrow(ax, B3_X + B3_W + 0.04, ARROW_Y,
          B4_X - 0.02, ARROW_Y,
          color=GRAY, lw=2.0, mutation_scale=14)

    # ── BOX 4 — Output ────────────────────────────────────────────────────────
    # Two stacked half-boxes
    half_h = BOX_H / 2
    # Waste (red top)
    solid_box(ax, B4_X, BOX_Y + half_h, B4_W, half_h,
              facecolor="#FADBD8", edgecolor=RED, lw=1.5, alpha=1.0)
    ax.text(B4_X + B4_W / 2, BOX_Y + half_h + half_h / 2,
            "Waste",
            ha="center", va="center", fontsize=9,
            fontweight="bold", color=RED)

    # No Waste (green bottom)
    solid_box(ax, B4_X, BOX_Y, B4_W, half_h,
              facecolor="#D5F5E3", edgecolor=GREEN, lw=1.5, alpha=1.0)
    ax.text(B4_X + B4_W / 2, BOX_Y + half_h / 2,
            "No\nWaste",
            ha="center", va="center", fontsize=9,
            fontweight="bold", color=GREEN)

    # ── Total features label on connector ─────────────────────────────────────
    mid12_x = (B1_X + B1_W + B2_X) / 2
    ax.text(mid12_x, ARROW_Y + 0.30, "30 feat.",
            ha="center", va="bottom", fontsize=7.5, color=GRAY)

    mid23_x = (B2_X + B2_W + B3_X) / 2
    ax.text(mid23_x, ARROW_Y + 0.30, "50 feat.",
            ha="center", va="bottom", fontsize=7.5, color=GRAY)

    # ── Title ─────────────────────────────────────────────────────────────────
    ax.text(7.0, 4.82,
            "CasCrop Architecture",
            ha="center", va="top", fontsize=13,
            fontweight="bold", color="#1a1a2e")

    fig.savefig(os.path.join(OUT_DIR, "fig1_architecture_v2.pdf"),
                dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("Saved: fig1_architecture_v2.pdf")


def _draw_mini_graph(ax, cx, cy, node_color, size=0.10, shape="s"):
    """Draw a tiny 3-node graph as a visual icon."""
    offsets = [(-size * 1.8, 0), (0, size * 1.2), (size * 1.8, 0)]
    edges   = [(0, 1), (1, 2)]
    pts = [(cx + dx, cy + dy) for dx, dy in offsets]
    for i, j in edges:
        ax.plot([pts[i][0], pts[j][0]], [pts[i][1], pts[j][1]],
                "-", color=node_color, lw=0.9, alpha=0.6, zorder=3)
    for px, py in pts:
        if shape == "s":
            ax.plot(px, py, "s", color=node_color, ms=5, zorder=4)
        else:
            ax.plot(px, py, "o", color=node_color, ms=5, zorder=4)


# ═════════════════════════════════════════════════════════════════════════════
# Entry point
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    make_graphical_abstract()
    make_architecture()
    print("Done.")
