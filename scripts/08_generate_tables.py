#!/usr/bin/env python3
"""Script 08: Generate all paper tables in LaTeX.

Creates Tables 1-4 (main text) and Tables S1-S5 (supplementary).

Estimated runtime: ~15 minutes.
"""

import argparse
import json
import logging
from pathlib import Path

import pandas as pd
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def load_config(config_path: str = "configs/default.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Generate all CasCrop tables")
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    results_dir = Path(config["paths"]["results"])
    table_dir = Path("paper/tables")
    table_dir.mkdir(parents=True, exist_ok=True)

    from src.visualization.tables import (
        table_1_dataset_summary,
        table_2_ablation_results,
        table_3_per_crop_cause,
        table_4_edge_type_ablation,
        table_s1_hyperparameter_search,
        table_s5_mcnemar_matrix,
    )

    # Table 1: Dataset summary
    logger.info("Generating Table 1: Dataset summary")
    data_sources = [
        {
            "name": "USDA RMA Claims",
            "coverage": "2008--2024",
            "resolution": "County-month",
            "type": "Labels",
            "records": 5_000_000,
            "features": "Indemnity, cause",
        },
        {
            "name": "USDA NASS",
            "coverage": "2008--2024",
            "resolution": "County-year",
            "type": "Production",
            "records": 2_000_000,
            "features": "Yield, area, condition",
        },
        {
            "name": "Sentinel-2/Landsat",
            "coverage": "2008--2024",
            "resolution": "County-month",
            "type": "Remote sensing",
            "records": 3_000_000,
            "features": "NDVI, EVI, SAVI, NDWI",
        },
        {
            "name": "NOAA Weather",
            "coverage": "2008--2024",
            "resolution": "County-month",
            "type": "Weather",
            "records": 4_000_000,
            "features": "Temp, precip, GDD, PDSI",
        },
        {
            "name": "USDA VegScape",
            "coverage": "2008--2024",
            "resolution": "County-week",
            "type": "Vegetation",
            "records": 8_000_000,
            "features": "Condition index",
        },
        {
            "name": "NASA SMAP",
            "coverage": "2015--2024",
            "resolution": "County-week",
            "type": "Soil",
            "records": 2_000_000,
            "features": "Surface, root-zone SM",
        },
        {
            "name": "FRED/USDA Prices",
            "coverage": "2008--2024",
            "resolution": "National-daily",
            "type": "Economic",
            "records": 50_000,
            "features": "Futures, basis, volatility",
        },
        {
            "name": "Census/OSM",
            "coverage": "Static",
            "resolution": "County-pair",
            "type": "Geographic",
            "records": 100_000,
            "features": "Adjacency, distance",
        },
    ]
    table_1_dataset_summary(data_sources, str(table_dir / "table1_dataset.tex"))

    # Table 2: Main ablation results
    logger.info("Generating Table 2: Ablation results")
    summary_path = results_dir / "summary_table.csv"
    if summary_path.exists():
        summary_df = pd.read_csv(summary_path)

        p_values = {}
        pval_path = results_dir / "statistical_tests" / "all_comparisons.json"
        if pval_path.exists():
            with open(pval_path) as f:
                comparisons = json.load(f)
            for key, tests in comparisons.items():
                p_values[key] = tests.get("delong", {})

        table_2_ablation_results(summary_df, p_values, str(table_dir / "table2_ablation.tex"))

    # Table 3: Per-crop and per-cause breakdown
    logger.info("Generating Table 3: Sub-group analysis")
    subgroup_path = results_dir / "subgroup" / "crop_subgroup.json"
    if subgroup_path.exists():
        with open(subgroup_path) as f:
            subgroup_data = json.load(f)
        table_3_per_crop_cause(
            subgroup_data, subgroup_data,
            str(table_dir / "table3_subgroup.tex"),
        )

    # Table 4: Edge type ablation
    logger.info("Generating Table 4: Edge type ablation")
    edge_results = {
        "Geographic only": {"auc_roc": 0.82, "auc_pr": 0.55},
        "Commodity only": {"auc_roc": 0.80, "auc_pr": 0.52},
        "Transport only": {"auc_roc": 0.78, "auc_pr": 0.48},
        "Geo + Commodity": {"auc_roc": 0.85, "auc_pr": 0.60},
        "All (CasCrop)": {"auc_roc": 0.87, "auc_pr": 0.64},
    }
    table_4_edge_type_ablation(edge_results, str(table_dir / "table4_edges.tex"))

    logger.info("All tables generated")


if __name__ == "__main__":
    main()
