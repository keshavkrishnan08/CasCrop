"""LaTeX table generation for CasCrop paper.

Generates Tables 1-4 (main text) and Tables S1-S5 (supplementary).
"""

import numpy as np
import pandas as pd
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


def _bold(text: str) -> str:
    return f"\\textbf{{{text}}}"


def _underline(text: str) -> str:
    return f"\\underline{{{text}}}"


def _format_metric(mean: float, std: float, is_best: bool = False,
                   is_second: bool = False) -> str:
    """Format a metric as mean ± std with optional bold/underline."""
    formatted = f"{mean:.3f} ± {std:.3f}"
    if is_best:
        return _bold(formatted)
    if is_second:
        return _underline(formatted)
    return formatted


def table_1_dataset_summary(
    data_sources: list[dict],
    output_path: str = "paper/tables/table1_dataset.tex",
):
    """Table 1: Dataset summary.

    Rows: each data source.
    Columns: Source, Temporal Coverage, Spatial Resolution, Records, Features.
    """
    header = (
        "\\begin{table}[t]\n"
        "\\centering\n"
        "\\caption{Dataset summary. All data sources are freely available "
        "and combined at county-crop-month granularity.}\n"
        "\\label{tab:dataset}\n"
        "\\small\n"
        "\\begin{tabular}{llllrl}\n"
        "\\toprule\n"
        "Source & Coverage & Resolution & Type & Records & Features \\\\\n"
        "\\midrule\n"
    )

    rows = []
    for src in data_sources:
        row = (
            f"{src['name']} & {src['coverage']} & {src['resolution']} & "
            f"{src['type']} & {src['records']:,} & {src['features']} \\\\"
        )
        rows.append(row)

    footer = (
        "\n\\bottomrule\n"
        "\\end{tabular}\n"
        "\\end{table}\n"
    )

    latex = header + "\n".join(rows) + footer
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write(latex)
    logger.info(f"Table 1 saved to {output_path}")
    return latex


def table_2_ablation_results(
    summary_df: pd.DataFrame,
    p_values: dict = None,
    output_path: str = "paper/tables/table2_ablation.tex",
):
    """Table 2: Main ablation results — THE central table.

    5 ablation rows + 3 additional baselines.
    Columns: AUC-ROC, AUC-PR, F1, Precision, Recall.
    Mean ± std across seeds. Bold best, underline second best.
    p-values from DeLong test vs Row 5.
    """
    metric_cols = ["auc_roc", "auc_pr", "f1_binary", "precision", "recall"]
    col_labels = ["AUC-ROC", "AUC-PR", "F1", "Precision", "Recall"]

    # Find best and second-best for each metric
    best_idx = {}
    second_idx = {}
    for col in metric_cols:
        mean_col = f"{col}_mean"
        if mean_col in summary_df.columns:
            sorted_idx = summary_df[mean_col].argsort()[::-1]
            best_idx[col] = sorted_idx.iloc[0] if len(sorted_idx) > 0 else -1
            second_idx[col] = sorted_idx.iloc[1] if len(sorted_idx) > 1 else -1

    header = (
        "\\begin{table*}[t]\n"
        "\\centering\n"
        "\\caption{Main ablation results on test set (2022--2024). "
        "Values are mean $\\pm$ std across 5 seeds. "
        "\\textbf{Bold}: best. \\underline{Underline}: second best. "
        "$^*p < 0.05$, $^{**}p < 0.01$, $^{***}p < 0.001$ "
        "(DeLong test vs.~Row 5).}\n"
        "\\label{tab:ablation}\n"
        "\\small\n"
        f"\\begin{{tabular}}{{l{'c' * len(metric_cols)}}}\n"
        "\\toprule\n"
        f"Model & {' & '.join(col_labels)} \\\\\n"
        "\\midrule\n"
    )

    rows = []
    for idx, row in summary_df.iterrows():
        model_name = row["model"]
        cells = [model_name]

        for col in metric_cols:
            mean_col = f"{col}_mean"
            std_col = f"{col}_std"
            if mean_col in row and std_col in row:
                is_best = (idx == best_idx.get(col, -1))
                is_second = (idx == second_idx.get(col, -1))
                cell = _format_metric(row[mean_col], row[std_col], is_best, is_second)
            else:
                cell = row.get(col, "---")

            # Add significance stars
            if p_values and model_name != "Row 5: Full CasCrop":
                key = model_name.lower().replace(" ", "_").replace(":", "")
                p = p_values.get(f"cascrop_vs_{key}", {}).get("p_value", 1.0)
                if p < 0.001:
                    cell += "$^{***}$"
                elif p < 0.01:
                    cell += "$^{**}$"
                elif p < 0.05:
                    cell += "$^{*}$"

            cells.append(cell)

        rows.append(" & ".join(cells) + " \\\\")

    # Add midrule between ablation rows and additional baselines
    if len(rows) >= 5:
        rows.insert(5, "\\midrule")

    footer = (
        "\n\\bottomrule\n"
        "\\end{tabular}\n"
        "\\end{table*}\n"
    )

    latex = header + "\n".join(rows) + footer
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write(latex)
    logger.info(f"Table 2 saved to {output_path}")
    return latex


def table_3_per_crop_cause(
    crop_results: dict,
    cause_results: dict,
    output_path: str = "paper/tables/table3_subgroup.tex",
):
    """Table 3: CasCrop performance by crop type and cause of loss.

    Rows: crop types (corn, soybeans, wheat) + cause categories.
    Columns: AUC-ROC, F1, n_samples.
    """
    header = (
        "\\begin{table}[t]\n"
        "\\centering\n"
        "\\caption{CasCrop performance by crop type and cause of loss.}\n"
        "\\label{tab:subgroup}\n"
        "\\small\n"
        "\\begin{tabular}{lccr}\n"
        "\\toprule\n"
        "Category & AUC-ROC & F1 & $n$ \\\\\n"
        "\\midrule\n"
        "\\multicolumn{4}{l}{\\textit{By Crop}} \\\\\n"
    )

    rows = []
    for crop, metrics in crop_results.items():
        auc = metrics.get("auc_roc", np.nan)
        f1 = metrics.get("f1_binary", np.nan)
        n = metrics.get("n_samples", 0)
        rows.append(f"\\quad {crop} & {auc:.3f} & {f1:.3f} & {n:,} \\\\")

    rows.append("\\midrule")
    rows.append("\\multicolumn{4}{l}{\\textit{By Cause of Loss}} \\\\")

    for cause, metrics in cause_results.items():
        auc = metrics.get("auc_roc", np.nan)
        f1 = metrics.get("f1_binary", np.nan)
        n = metrics.get("n_samples", 0)
        rows.append(f"\\quad {cause} & {auc:.3f} & {f1:.3f} & {n:,} \\\\")

    footer = (
        "\n\\bottomrule\n"
        "\\end{tabular}\n"
        "\\end{table}\n"
    )

    latex = header + "\n".join(rows) + footer
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write(latex)
    logger.info(f"Table 3 saved to {output_path}")
    return latex


def table_4_edge_type_ablation(
    edge_results: dict,
    output_path: str = "paper/tables/table4_edges.tex",
):
    """Table 4: Graph edge type ablation.

    Rows: different edge type combinations.
    Columns: AUC-ROC, AUC-PR.
    """
    header = (
        "\\begin{table}[t]\n"
        "\\centering\n"
        "\\caption{Effect of graph edge types on prediction performance.}\n"
        "\\label{tab:edges}\n"
        "\\small\n"
        "\\begin{tabular}{lccc}\n"
        "\\toprule\n"
        "Edge Types & AUC-ROC & AUC-PR & $\\Delta$ AUC \\\\\n"
        "\\midrule\n"
    )

    rows = []
    best_auc = max(r.get("auc_roc", 0) for r in edge_results.values())

    for config_name, metrics in edge_results.items():
        auc = metrics.get("auc_roc", np.nan)
        auc_pr = metrics.get("auc_pr", np.nan)
        delta = auc - best_auc
        auc_str = f"{auc:.3f}"
        if auc == best_auc:
            auc_str = _bold(auc_str)
        rows.append(
            f"{config_name} & {auc_str} & {auc_pr:.3f} & "
            f"{delta:+.3f} \\\\"
        )

    footer = (
        "\n\\bottomrule\n"
        "\\end{tabular}\n"
        "\\end{table}\n"
    )

    latex = header + "\n".join(rows) + footer
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write(latex)
    logger.info(f"Table 4 saved to {output_path}")
    return latex


def table_s1_hyperparameter_search(
    search_results: pd.DataFrame,
    output_path: str = "paper/tables/tableS1_hyperparam.tex",
):
    """Supplementary Table S1: Full hyperparameter search results."""
    latex = search_results.to_latex(
        index=False,
        caption="Hyperparameter search results (top 20 configurations by validation AUC-ROC).",
        label="tab:hyperparam",
        float_format="%.4f",
    )
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write(latex)
    logger.info(f"Table S1 saved to {output_path}")
    return latex


def table_s5_mcnemar_matrix(
    pairwise_results: dict,
    model_names: list[str],
    output_path: str = "paper/tables/tableS5_mcnemar.tex",
):
    """Supplementary Table S5: Full pairwise McNemar test matrix."""
    n = len(model_names)
    matrix = np.ones((n, n))

    for i, m1 in enumerate(model_names):
        for j, m2 in enumerate(model_names):
            if i == j:
                continue
            key = f"{m1}_vs_{m2}"
            if key in pairwise_results:
                matrix[i, j] = pairwise_results[key].get("p_value", 1.0)

    df = pd.DataFrame(matrix, index=model_names, columns=model_names)

    # Format p-values with significance markers
    def fmt_p(p):
        if p < 0.001:
            return f"{p:.1e}***"
        elif p < 0.01:
            return f"{p:.3f}**"
        elif p < 0.05:
            return f"{p:.3f}*"
        return f"{p:.3f}"

    formatted = df.map(fmt_p)
    latex = formatted.to_latex(
        caption="Pairwise McNemar test p-values. *$p<0.05$, **$p<0.01$, ***$p<0.001$.",
        label="tab:mcnemar",
    )
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write(latex)
    logger.info(f"Table S5 saved to {output_path}")
    return latex
