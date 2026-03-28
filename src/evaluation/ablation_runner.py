"""Automated ablation experiment runner.

Trains all model variants across all seeds and collects results
into the main ablation table.
"""

import logging
import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger(__name__)

# Model registry mapping ablation row names to their module paths
MODEL_REGISTRY = {
    "row1_local_only": "src.models.baselines.local_only.LocalOnlyModel",
    "row2_local_econ": "src.models.baselines.local_econ.LocalEconModel",
    "row3_geo_gat": "src.models.baselines.geo_gat.GeoGATModel",
    "row4_symmetric_ecmp": "src.models.baselines.symmetric_ecmp.SymmetricECMPModel",
    "row5_cascrop": "src.models.cascrop.CasCrop",
}

ADDITIONAL_BASELINES = {
    "random_forest": "sklearn_baseline",
    "xgboost": "sklearn_baseline",
    "lstm": "src.models.baselines.lstm_baseline.LSTMBaseline",
}

ABLATION_DISPLAY_NAMES = {
    "row1_local_only": "Row 1: Local Only (Bio MLP)",
    "row2_local_econ": "Row 2: Local + Economic",
    "row3_geo_gat": "Row 3: Geographic GAT",
    "row4_symmetric_ecmp": "Row 4: Symmetric ECMP",
    "row5_cascrop": "Row 5: Full CasCrop",
    "random_forest": "Random Forest",
    "xgboost": "XGBoost",
    "lstm": "LSTM",
}


class AblationRunner:
    """Runs all ablation experiments and collects results.

    Orchestrates training of all model variants across seeds,
    evaluates on test set, runs statistical tests, and generates
    the main ablation table.
    """

    def __init__(self, config: dict, output_dir: Path):
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.seeds = config.get("seeds", [42, 123, 456, 789, 1024])
        self.results = {}

    def _import_model_class(self, model_path: str):
        """Dynamically import a model class from its module path."""
        parts = model_path.rsplit(".", 1)
        module_path, class_name = parts[0], parts[1]
        import importlib

        module = importlib.import_module(module_path)
        return getattr(module, class_name)

    def run_single_experiment(
        self,
        model_name: str,
        model_class,
        train_loader,
        val_loader,
        test_loader,
        seed: int,
    ) -> dict:
        """Train and evaluate a single model with a single seed.

        Returns dict with all metrics and predictions.
        """
        import torch
        from src.training.trainer import CasCropTrainer
        from src.evaluation.metrics import compute_all_metrics

        # Set seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        # Initialize model
        model = model_class(self.config)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)

        # Train
        trainer = CasCropTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=self.config,
        )
        train_results = trainer.train(self.config["training"]["epochs"])

        # Evaluate on test set
        model.eval()
        all_y_true, all_y_prob = [], []
        all_cause_true, all_cause_pred = [], []

        with torch.no_grad():
            for batch in test_loader:
                batch = {
                    k: v.to(device) if hasattr(v, "to") else v
                    for k, v in batch.items()
                }
                outputs = model(batch)
                waste_prob = torch.sigmoid(outputs["waste_logits"]).cpu().numpy()
                all_y_true.append(batch["y_waste"].cpu().numpy())
                all_y_prob.append(waste_prob)

                if "cause_logits" in outputs:
                    cause_pred = outputs["cause_logits"].argmax(dim=-1).cpu().numpy()
                    all_cause_pred.append(cause_pred)
                    all_cause_true.append(batch["y_cause"].cpu().numpy())

        y_true = np.concatenate(all_y_true).flatten()
        y_prob = np.concatenate(all_y_prob).flatten()

        cause_true = (
            np.concatenate(all_cause_true).flatten() if all_cause_true else None
        )
        cause_pred = (
            np.concatenate(all_cause_pred).flatten() if all_cause_pred else None
        )

        metrics = compute_all_metrics(y_true, y_prob, cause_true, cause_pred)
        metrics["seed"] = seed
        metrics["model"] = model_name
        metrics["best_val_auc"] = train_results.get("best_val_auc", np.nan)

        # Save checkpoint
        ckpt_path = self.output_dir / f"{model_name}_seed{seed}.pt"
        torch.save(model.state_dict(), ckpt_path)

        return {
            "metrics": metrics,
            "y_true": y_true,
            "y_prob": y_prob,
            "cause_true": cause_true,
            "cause_pred": cause_pred,
        }

    def run_all_ablations(
        self, train_loader, val_loader, test_loader
    ) -> pd.DataFrame:
        """Run all ablation experiments (all models x all seeds).

        Returns DataFrame with the main ablation table.
        """
        all_results = []

        for model_name, model_path in MODEL_REGISTRY.items():
            logger.info(f"Running ablation: {model_name}")
            model_class = self._import_model_class(model_path)
            seed_results = []

            for seed in self.seeds:
                logger.info(f"  Seed {seed}")
                result = self.run_single_experiment(
                    model_name, model_class, train_loader, val_loader, test_loader, seed
                )
                seed_results.append(result["metrics"])

                # Store predictions for statistical tests (last seed)
                self.results[f"{model_name}_seed{seed}"] = result

            all_results.extend(seed_results)

        df = pd.DataFrame(all_results)

        # Save raw results
        df.to_csv(self.output_dir / "raw_ablation_results.csv", index=False)

        # Generate summary table
        summary = self._generate_summary_table(df)
        summary.to_csv(self.output_dir / "summary_table.csv")

        return summary

    def _generate_summary_table(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate the main ablation table with mean +/- std.

        Formats as: "0.847 ± 0.012" with bold best and underline second-best.
        """
        metric_cols = [
            "auc_roc",
            "auc_pr",
            "f1_binary",
            "precision",
            "recall",
            "brier_score",
        ]
        rows = []

        for model_name in MODEL_REGISTRY.keys():
            model_df = df[df["model"] == model_name]
            row = {"model": ABLATION_DISPLAY_NAMES.get(model_name, model_name)}

            for col in metric_cols:
                if col in model_df.columns:
                    mean = model_df[col].mean()
                    std = model_df[col].std()
                    row[f"{col}_mean"] = mean
                    row[f"{col}_std"] = std
                    row[col] = f"{mean:.3f} ± {std:.3f}"

            rows.append(row)

        return pd.DataFrame(rows)

    def save_results(self):
        """Save all results to disk."""
        results_path = self.output_dir / "ablation_results.json"
        serializable = {}
        for key, val in self.results.items():
            serializable[key] = {
                "metrics": val["metrics"],
            }
        with open(results_path, "w") as f:
            json.dump(serializable, f, indent=2, default=str)
        logger.info(f"Results saved to {results_path}")
