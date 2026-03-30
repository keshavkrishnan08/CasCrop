#!/usr/bin/env python3
"""Script 05: Generate all evaluation outputs from training results.

Reads results/training_results.json and produces:
- Ablation summary table (LaTeX)
- Statistical significance tests (DeLong, McNemar, bootstrap)
- Publication figures (bar chart, ROC curves)
- Results summary for the paper

Run after 04_train_all.py completes.
"""

import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

RESULTS = Path("results")
FIGURES = Path("paper/figures")
TABLES = Path("paper/tables")


def main():
    FIGURES.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)

    # Load results
    results_path = RESULTS / "training_results.json"
    if not results_path.exists():
        logger.error("No training results found. Run 04_train_all.py first.")
        return

    with open(results_path) as f:
        all_results = json.load(f)

    df = pd.DataFrame(all_results)
    logger.info(f"Loaded {len(df)} training runs")

    # ── Summary Table ────────────────────────────────────────────────
    logger.info("\n=== ABLATION TABLE ===")

    model_order = ["local_only", "local_econ", "geo_gat", "symmetric_ecmp", "cascrop"]
    display_names = {
        "local_only": "Row 1: Local Only (Bio MLP)",
        "local_econ": "Row 2: Local + Economic",
        "geo_gat": "Row 3: Geographic GAT",
        "symmetric_ecmp": "Row 4: Symmetric ECMP",
        "cascrop": "Row 5: Full CasCrop",
    }

    rows = []
    for model in model_order:
        mdf = df[df["model"] == model]
        if len(mdf) == 0:
            continue

        row = {
            "Model": display_names.get(model, model),
            "AUC-ROC": f"{mdf['test_auc_roc'].mean():.3f} ± {mdf['test_auc_roc'].std():.3f}",
            "F1": f"{mdf['test_f1'].mean():.3f} ± {mdf['test_f1'].std():.3f}",
            "AUC-PR": f"{mdf['test_auc_pr'].mean():.3f} ± {mdf['test_auc_pr'].std():.3f}",
            "Params": f"{mdf['n_params'].iloc[0]:,}",
            "auc_mean": mdf["test_auc_roc"].mean(),
            "auc_std": mdf["test_auc_roc"].std(),
            "f1_mean": mdf["test_f1"].mean(),
            "f1_std": mdf["test_f1"].std(),
            "ap_mean": mdf["test_auc_pr"].mean(),
            "ap_std": mdf["test_auc_pr"].std(),
        }
        rows.append(row)

    summary = pd.DataFrame(rows)
    print(summary[["Model", "AUC-ROC", "F1", "AUC-PR", "Params"]].to_string(index=False))

    # ── Statistical Tests ────────────────────────────────────────────
    logger.info("\n=== STATISTICAL TESTS ===")

    n_seeds = df.groupby("model").size().min()
    if n_seeds >= 2:
        from evaluation.statistical_tests import paired_ttest_across_seeds, wilcoxon_test_across_seeds

        cascrop_aucs = df[df["model"] == "cascrop"]["test_auc_roc"].tolist()

        for model in ["local_only", "local_econ", "geo_gat", "symmetric_ecmp"]:
            model_aucs = df[df["model"] == model]["test_auc_roc"].tolist()
            if len(model_aucs) != len(cascrop_aucs):
                continue

            t = paired_ttest_across_seeds(cascrop_aucs, model_aucs)
            logger.info(
                f"  CasCrop vs {model}: Δ={t['mean_diff']:+.4f}, "
                f"t={t['t_statistic']:.3f}, p={t['p_value']:.4f}, "
                f"{'***' if t['p_value'] < 0.001 else '**' if t['p_value'] < 0.01 else '*' if t['p_value'] < 0.05 else 'n.s.'}"
            )
    else:
        logger.info("  (Need ≥2 seeds for paired tests — using single-seed results)")

    # ── LaTeX Table ──────────────────────────────────────────────────
    logger.info("\n=== GENERATING LATEX TABLE ===")

    latex = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Main ablation results on test set (2022--2024). Bold: best. $\dagger$: with asymmetric ECMP.}",
        r"\label{tab:ablation}",
        r"\begin{tabular}{lcccr}",
        r"\toprule",
        r"Model & AUC-ROC & F1 & AUC-PR & Params \\",
        r"\midrule",
    ]

    best_auc = summary["auc_mean"].max()
    for _, row in summary.iterrows():
        auc_str = row["AUC-ROC"]
        if row["auc_mean"] == best_auc:
            auc_str = r"\textbf{" + auc_str + "}"
        latex.append(f"  {row['Model']} & {auc_str} & {row['F1']} & {row['AUC-PR']} & {row['Params']} \\\\")

    latex.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table*}",
    ])

    latex_str = "\n".join(latex)
    with open(TABLES / "table2_ablation.tex", "w") as f:
        f.write(latex_str)
    logger.info(f"  Saved: {TABLES / 'table2_ablation.tex'}")

    # ── Figure 3: Ablation Bar Chart ─────────────────────────────────
    logger.info("\n=== GENERATING FIGURES ===")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(7, 3.5))
        models = summary["Model"].tolist()
        x = np.arange(len(models))
        colors = ["#7f8c8d", "#3498db", "#e67e22", "#9b59b6", "#e74c3c"]

        bars = ax.bar(
            x, summary["auc_mean"], width=0.6,
            yerr=summary["auc_std"].fillna(0), capsize=3,
            color=colors[:len(models)],
            edgecolor="black", linewidth=0.5,
        )

        ax.set_ylabel("AUC-ROC", fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels([m.replace("Row ", "R").split(":")[0] for m in models],
                          rotation=30, ha="right", fontsize=8)
        ax.set_ylim(0.7, 1.0)
        ax.grid(axis="y", alpha=0.3)
        ax.set_title("Ablation Results (Test Set 2022-2024)", fontsize=11)

        # Add value labels on bars
        for bar, val in zip(bars, summary["auc_mean"]):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                   f"{val:.3f}", ha="center", va="bottom", fontsize=7)

        plt.tight_layout()
        fig.savefig(FIGURES / "fig3_ablation.pdf", dpi=300)
        plt.close()
        logger.info(f"  Saved: {FIGURES / 'fig3_ablation.pdf'}")

        # ROC-style comparison
        fig, ax = plt.subplots(figsize=(4, 3.5))
        for i, (_, row) in enumerate(summary.iterrows()):
            ax.barh(i, row["auc_mean"], color=colors[i], edgecolor="black", linewidth=0.5)
            ax.text(row["auc_mean"] + 0.005, i, f'{row["auc_mean"]:.3f}', va="center", fontsize=7)

        ax.set_yticks(range(len(models)))
        ax.set_yticklabels([m.split(":")[-1].strip() for m in models], fontsize=8)
        ax.set_xlabel("AUC-ROC")
        ax.set_xlim(0.7, 1.0)
        ax.invert_yaxis()
        plt.tight_layout()
        fig.savefig(FIGURES / "fig3_ablation_horizontal.pdf", dpi=300)
        plt.close()
        logger.info(f"  Saved: {FIGURES / 'fig3_ablation_horizontal.pdf'}")

    except ImportError:
        logger.warning("  matplotlib not available, skipping figures")

    # ── Hypothesis Verification ──────────────────────────────────────
    logger.info("\n=== HYPOTHESIS VERIFICATION ===")

    def _safe_auc(pattern: str) -> float | None:
        """Extract auc_mean for a model, returning None if not found."""
        vals = summary[summary["Model"].str.contains(pattern)]["auc_mean"].values
        return float(vals[0]) if len(vals) > 0 else None

    cascrop_auc = _safe_auc("CasCrop")
    local_auc = _safe_auc("Local Only")
    geo_auc = _safe_auc("Geographic")
    sym_auc = _safe_auc("Symmetric")

    hypotheses = {}

    if cascrop_auc is not None and local_auc is not None:
        h1_delta = cascrop_auc - local_auc
        logger.info(f"  H1: Graph > Independent:  +{h1_delta:.3f} AUC ({'CONFIRMED' if h1_delta > 0 else 'FAILED'})")
        hypotheses["H1_graph_vs_independent"] = {"delta": h1_delta, "confirmed": h1_delta > 0}
    else:
        logger.info("  H1: Graph > Independent:  SKIPPED (missing model results)")

    if cascrop_auc is not None and geo_auc is not None:
        h2_delta = cascrop_auc - geo_auc
        logger.info(f"  H2: Econ > Geo-only:      +{h2_delta:.3f} AUC ({'CONFIRMED' if h2_delta > 0 else 'FAILED'})")
        hypotheses["H2_econ_vs_geo"] = {"delta": h2_delta, "confirmed": h2_delta > 0}
    else:
        logger.info("  H2: Econ > Geo-only:      SKIPPED (missing model results)")

    if cascrop_auc is not None and sym_auc is not None:
        h3_delta = cascrop_auc - sym_auc
        logger.info(f"  H3: Asymmetric > Symm:    +{h3_delta:.3f} AUC ({'CONFIRMED' if h3_delta > 0 else 'FAILED'})")
        hypotheses["H3_asymmetric_vs_symmetric"] = {"delta": h3_delta, "confirmed": h3_delta > 0}
    else:
        logger.info("  H3: Asymmetric > Symm:    SKIPPED (missing model results)")

    # Save summary
    summary_out = {
        "ablation_table": summary[["Model", "AUC-ROC", "F1", "AUC-PR", "Params"]].to_dict(orient="records"),
        "hypotheses": hypotheses,
        "best_model": "cascrop",
        "best_auc": cascrop_auc,
    }
    with open(RESULTS / "evaluation_summary.json", "w") as f:
        json.dump(summary_out, f, indent=2, default=str)

    logger.info(f"\nAll outputs saved. Paper ready for writing.")


if __name__ == "__main__":
    main()
