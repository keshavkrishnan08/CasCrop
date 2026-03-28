#!/usr/bin/env python3
"""Script 05: Run all evaluations and statistical tests.

Evaluates all trained models on the test set, computes all metrics,
runs DeLong/McNemar/bootstrap tests, performs sub-group analysis,
and verifies disentanglement.

Estimated runtime: ~2 hours.
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


def load_model(model_name: str, seed: int, config: dict, device: torch.device):
    """Load a trained model checkpoint."""
    from scripts.train_all import import_class, MODEL_CONFIGS

    model_info = MODEL_CONFIGS.get(model_name, {})
    if not model_info:
        return None

    ModelClass = import_class(model_info["model_class"])
    model = ModelClass(config).to(device)

    ckpt_path = Path(config["paths"]["checkpoints"]) / f"{model_name}_seed{seed}.pt"
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])

    model.eval()
    return model


def evaluate_model(model, test_loader, device):
    """Run model on test set and collect predictions."""
    all_y_true, all_y_prob = [], []
    all_cause_true, all_cause_pred = [], []
    all_z_bio, all_z_econ = [], []
    all_metadata = []

    with torch.no_grad():
        for batch in test_loader:
            batch_device = {
                k: v.to(device) if hasattr(v, "to") else v
                for k, v in batch.items()
            }
            outputs = model(batch_device)

            waste_prob = torch.sigmoid(outputs["waste_logits"]).cpu().numpy()
            all_y_true.append(batch["waste_target"].cpu().numpy())
            all_y_prob.append(waste_prob)

            if "cause_logits" in outputs:
                all_cause_pred.append(
                    outputs["cause_logits"].argmax(dim=-1).cpu().numpy()
                )
                all_cause_true.append(batch["cause_target"].cpu().numpy())

            if "z_bio" in outputs:
                all_z_bio.append(outputs["z_bio"].cpu().numpy())
            if "z_econ" in outputs:
                all_z_econ.append(outputs["z_econ"].cpu().numpy())

            # Collect metadata for sub-group analysis
            if "fips" in batch:
                for i in range(len(batch["fips"])):
                    all_metadata.append({
                        "fips": batch["fips"][i] if isinstance(batch["fips"], list) else batch["fips"][i].item(),
                        "commodity": batch.get("commodity", [""])[i] if isinstance(batch.get("commodity", [""]), list) else "",
                        "year": batch["year"][i].item() if hasattr(batch.get("year", 0), "__getitem__") else 0,
                        "month": batch["month"][i].item() if hasattr(batch.get("month", 0), "__getitem__") else 0,
                    })

    return {
        "y_true": np.concatenate(all_y_true).flatten(),
        "y_prob": np.concatenate(all_y_prob).flatten(),
        "cause_true": np.concatenate(all_cause_true).flatten() if all_cause_true else None,
        "cause_pred": np.concatenate(all_cause_pred).flatten() if all_cause_pred else None,
        "z_bio": np.concatenate(all_z_bio) if all_z_bio else None,
        "z_econ": np.concatenate(all_z_econ) if all_z_econ else None,
        "metadata": pd.DataFrame(all_metadata) if all_metadata else None,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate all CasCrop models")
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results_dir = Path(config["paths"]["results"])

    from src.data.dataset import get_dataloaders
    from src.evaluation.metrics import compute_all_metrics, compute_subgroup_metrics, aggregate_seed_metrics
    from src.evaluation.statistical_tests import run_all_comparisons
    from scripts.train_all import MODEL_CONFIGS

    _, _, test_loader = get_dataloaders(config)
    seeds = config.get("seeds", [42, 123, 456, 789, 1024])

    # Evaluate all models
    all_seed_metrics = {}
    all_predictions = {}

    for model_name in MODEL_CONFIGS:
        logger.info(f"Evaluating: {model_name}")
        seed_metrics = []

        for seed in seeds:
            model = load_model(model_name, seed, config, device)
            if model is None:
                continue

            eval_result = evaluate_model(model, test_loader, device)
            metrics = compute_all_metrics(
                eval_result["y_true"],
                eval_result["y_prob"],
                eval_result["cause_true"],
                eval_result["cause_pred"],
            )
            metrics["seed"] = seed
            seed_metrics.append(metrics)

            # Store last seed's predictions for statistical tests
            all_predictions[model_name] = eval_result["y_prob"]

        all_seed_metrics[model_name] = seed_metrics
        agg = aggregate_seed_metrics(seed_metrics)
        logger.info(f"  AUC-ROC: {agg.get('auc_roc', {}).get('formatted', 'N/A')}")

    # Save raw metrics
    metrics_dir = results_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    with open(metrics_dir / "all_seed_metrics.json", "w") as f:
        json.dump(all_seed_metrics, f, indent=2, default=str)

    # Statistical tests
    logger.info("Running statistical significance tests")
    y_true = eval_result["y_true"]  # Same for all models
    comparisons = run_all_comparisons(
        y_true, all_predictions, all_seed_metrics, reference_model="row5_cascrop"
    )

    test_dir = results_dir / "statistical_tests"
    test_dir.mkdir(parents=True, exist_ok=True)
    with open(test_dir / "all_comparisons.json", "w") as f:
        json.dump(comparisons, f, indent=2, default=str)

    # Sub-group analysis (for CasCrop only)
    logger.info("Running sub-group analysis")
    cascrop_model = load_model("row5_cascrop", seeds[0], config, device)
    if cascrop_model:
        cascrop_eval = evaluate_model(cascrop_model, test_loader, device)
        metadata = cascrop_eval["metadata"]

        if metadata is not None and "commodity" in metadata.columns:
            # By crop
            crop_labels = metadata["commodity"].values
            crop_metrics = compute_subgroup_metrics(
                cascrop_eval["y_true"], cascrop_eval["y_prob"],
                pd.Categorical(crop_labels).codes,
                group_names=sorted(metadata["commodity"].unique()),
            )

            subgroup_dir = results_dir / "subgroup"
            subgroup_dir.mkdir(parents=True, exist_ok=True)
            with open(subgroup_dir / "crop_subgroup.json", "w") as f:
                json.dump(crop_metrics, f, indent=2, default=str)

    # Generate summary table
    logger.info("Generating summary ablation table")
    summary_rows = []
    for model_name, seed_metrics in all_seed_metrics.items():
        agg = aggregate_seed_metrics(seed_metrics)
        row = {"model": model_name}
        for metric_name, vals in agg.items():
            row[f"{metric_name}_mean"] = vals["mean"]
            row[f"{metric_name}_std"] = vals["std"]
            row[metric_name] = vals["formatted"]
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(results_dir / "summary_table.csv", index=False)
    logger.info(f"Summary table saved to {results_dir / 'summary_table.csv'}")

    # Disentanglement evaluation
    logger.info("Evaluating disentanglement")
    if cascrop_eval.get("z_bio") is not None and cascrop_eval.get("z_econ") is not None:
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler

        z_bio = cascrop_eval["z_bio"]
        z_econ = cascrop_eval["z_econ"]

        # Linear probe: predict z_econ cluster from z_bio
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
        econ_labels = kmeans.fit_predict(z_econ)

        scaler = StandardScaler()
        z_bio_scaled = scaler.fit_transform(z_bio)

        probe = LogisticRegression(max_iter=1000, random_state=42)
        probe.fit(z_bio_scaled[:len(z_bio)//2], econ_labels[:len(z_bio)//2])
        probe_acc = probe.score(z_bio_scaled[len(z_bio)//2:], econ_labels[len(z_bio)//2:])

        disentangle_result = {
            "linear_probe_accuracy": float(probe_acc),
            "target_random_accuracy": 0.2,  # 1/5 clusters
            "well_disentangled": probe_acc < 0.55,
        }
        logger.info(f"  Linear probe accuracy: {probe_acc:.3f} (target: < 0.55)")

        with open(test_dir / "disentanglement_eval.json", "w") as f:
            json.dump(disentangle_result, f, indent=2)

    logger.info("All evaluations complete")


if __name__ == "__main__":
    main()
