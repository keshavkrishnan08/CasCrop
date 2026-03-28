#!/usr/bin/env python3
"""Script 02: Process and match all datasets.

Cleans, normalizes, and merges all data sources into a unified
county-crop-month dataset. Creates temporal train/val/test splits.

Estimated runtime: 1-2 hours.
"""

import argparse
import logging
from pathlib import Path

import numpy as np
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
    parser = argparse.ArgumentParser(description="Process and match CasCrop data")
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    raw_dir = Path(config["paths"]["raw_data"])
    processed_dir = Path(config["paths"]["processed_data"])
    splits_dir = Path(config["paths"]["splits"])
    processed_dir.mkdir(parents=True, exist_ok=True)
    splits_dir.mkdir(parents=True, exist_ok=True)

    from src.data.matcher import DataMatcher

    matcher = DataMatcher(config)

    # Step 1: Load and merge all data sources
    logger.info("Step 1: Loading and matching all data sources")
    features_df, labels_df = matcher.match_all(raw_dir)
    logger.info(f"Matched dataset: {len(features_df)} observations")

    # Step 2: Handle missing data
    logger.info("Step 2: Handling missing data")
    features_df = matcher.handle_missing_data(features_df)

    # Step 3: Create temporal splits
    logger.info("Step 3: Creating temporal splits")
    splits = matcher.create_temporal_splits(features_df)

    for split_name, indices in splits.items():
        np.save(splits_dir / f"{split_name}_indices.npy", indices)
        logger.info(f"  {split_name}: {len(indices)} samples")

    # Step 4: Compute normalization statistics from train set
    logger.info("Step 4: Computing normalization statistics")
    train_idx = splits["train"]
    train_features = features_df.iloc[train_idx]

    numeric_cols = train_features.select_dtypes(include=[np.number]).columns
    stats = {}
    for col in numeric_cols:
        stats[col] = {
            "mean": float(train_features[col].mean()),
            "std": float(train_features[col].std()),
        }

    import json
    with open(processed_dir / "feature_statistics.json", "w") as f:
        json.dump(stats, f, indent=2)

    # Step 5: Analyze class imbalance
    logger.info("Step 5: Analyzing class imbalance")
    imbalance = matcher.analyze_class_imbalance(labels_df)
    logger.info(f"Overall waste rate: {imbalance.get('overall_waste_rate', 'N/A')}")

    # Step 6: Save processed data
    logger.info("Step 6: Saving processed data")
    matcher.save(features_df, labels_df, splits, processed_dir)

    # Step 7: Generate data summary
    logger.info("Step 7: Generating data summary")
    summary_lines = [
        "# Data Summary",
        f"\nTotal observations: {len(features_df):,}",
        f"Unique counties: {features_df['fips'].nunique():,}",
        f"Date range: {features_df['year'].min()}-{features_df['year'].max()}",
        f"\n## Split Sizes",
        f"- Train: {len(splits['train']):,} ({len(splits['train'])/len(features_df)*100:.1f}%)",
        f"- Val: {len(splits['val']):,} ({len(splits['val'])/len(features_df)*100:.1f}%)",
        f"- Test: {len(splits['test']):,} ({len(splits['test'])/len(features_df)*100:.1f}%)",
        f"\n## Feature Dimensions",
        f"- Biophysical: {config['model']['biophysical_input_dim']}",
        f"- Economic: {config['model']['economic_input_dim']}",
        f"- Historical: {config['model']['historical_input_dim']}",
    ]

    with open(processed_dir / "data_summary.md", "w") as f:
        f.write("\n".join(summary_lines))

    logger.info("Data processing complete")


if __name__ == "__main__":
    main()
