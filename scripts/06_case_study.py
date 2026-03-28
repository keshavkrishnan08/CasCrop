#!/usr/bin/env python3
"""Script 06: Generate case study analysis.

Identifies the most compelling cascade event, reconstructs it
using ECMP attention weights, and generates the narrative
for Section 7 of the paper.

Estimated runtime: ~1 hour.
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
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
    parser = argparse.ArgumentParser(description="Generate CasCrop case study")
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    from src.data.dataset import get_dataloaders
    from src.evaluation.case_study import CaseStudyAnalyzer
    from src.evaluation.attention_analysis import AttentionAnalyzer

    # Load data
    processed_dir = Path(config["paths"]["processed_data"])
    features_df = pd.read_parquet(processed_dir / "features.parquet")
    labels_df = pd.read_parquet(processed_dir / "labels.parquet")
    _, _, test_loader = get_dataloaders(config)

    # Load trained CasCrop model
    from scripts.train_all import import_class
    ModelClass = import_class("src.models.cascrop.CasCrop")
    model = ModelClass(config).to(device)

    ckpt_path = Path(config["paths"]["checkpoints"]) / "row5_cascrop_seed42.pt"
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # Load local-only model for comparison
    LocalClass = import_class("src.models.baselines.local_only.LocalOnlyModel")
    local_model = LocalClass(config).to(device)
    local_ckpt = Path(config["paths"]["checkpoints"]) / "row1_local_only_seed42.pt"
    if local_ckpt.exists():
        ckpt = torch.load(local_ckpt, map_location=device)
        local_model.load_state_dict(ckpt["model_state_dict"])
    local_model.eval()

    # Step 1: Find cascade candidates
    logger.info("Step 1: Searching for cascade candidates")
    analyzer = CaseStudyAnalyzer(labels_df, features_df, config)
    candidates = analyzer.find_cascade_candidates()
    logger.info(f"Found {len(candidates)} candidates")

    if len(candidates) == 0:
        logger.warning("No cascade candidates found. Using synthetic example.")
        return

    # Step 2: Select best candidate
    logger.info("Step 2: Selecting best cascade event")
    best_event = analyzer.select_best_candidate(candidates)
    logger.info(f"Selected: {best_event['source_fips']} -> {best_event['target_fips']}")

    # Step 3: Reconstruct cascade with attention weights
    logger.info("Step 3: Reconstructing cascade with ECMP attention")
    attn_analyzer = AttentionAnalyzer(model, device=str(device))
    reconstruction = analyzer.reconstruct_cascade(
        event=best_event,
        attention_analyzer=attn_analyzer,
        model=model,
        data_loader=test_loader,
    )

    # Step 4: Generate counterfactual comparison
    logger.info("Step 4: Computing counterfactual comparison")
    # Collect predictions from both models for target county
    cascrop_preds = {}
    local_preds = {}

    with torch.no_grad():
        for batch in test_loader:
            batch_device = {
                k: v.to(device) if hasattr(v, "to") else v
                for k, v in batch.items()
            }

            cascrop_out = model(batch_device)
            local_out = local_model(batch_device)

            cascrop_prob = torch.sigmoid(cascrop_out["waste_logits"]).cpu().numpy()
            local_prob = torch.sigmoid(local_out["waste_logits"]).cpu().numpy()

            if "fips" in batch:
                for i in range(len(batch["fips"])):
                    fips = batch["fips"][i] if isinstance(batch["fips"], list) else str(batch["fips"][i].item())
                    commodity = batch.get("commodity", [""])[i] if isinstance(batch.get("commodity", [""]), list) else ""
                    year = batch["year"][i].item() if hasattr(batch.get("year", 0), "__getitem__") else 0
                    month = batch["month"][i].item() if hasattr(batch.get("month", 0), "__getitem__") else 0
                    key = (fips, commodity, year, month)
                    cascrop_preds[key] = float(cascrop_prob[i])
                    local_preds[key] = float(local_prob[i])

    comparison = analyzer.compute_counterfactual_comparison(
        best_event, cascrop_preds, local_preds
    )

    # Step 5: Save case study
    logger.info("Step 5: Saving case study")
    case_dir = Path(config["paths"]["results"]) / "case_study"
    reconstruction["counterfactual"] = comparison
    analyzer.save_case_study(reconstruction, case_dir)

    # Save candidates for supplementary
    candidates.to_csv(case_dir / "all_candidates.csv", index=False)

    logger.info("Case study generation complete")
    logger.info(f"Narrative: {reconstruction['narrative'][:200]}...")


if __name__ == "__main__":
    main()
