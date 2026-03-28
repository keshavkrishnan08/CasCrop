#!/usr/bin/env python3
"""Script 04: Train all models (ablation rows + baselines).

Trains 8 models × 5 seeds = 40 training runs.
Uses early stopping on validation AUC-ROC.
Saves checkpoints for best validation performance.

Estimated runtime: 20-40 hours on single GPU.
"""

import argparse
import logging
import json
from pathlib import Path

import numpy as np
import torch
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

MODEL_CONFIGS = {
    "row1_local_only": {
        "config_file": "configs/ablation_row1.yaml",
        "model_class": "src.models.baselines.local_only.LocalOnlyModel",
    },
    "row2_local_econ": {
        "config_file": "configs/ablation_row2.yaml",
        "model_class": "src.models.baselines.local_econ.LocalEconModel",
    },
    "row3_geo_gat": {
        "config_file": "configs/ablation_row3.yaml",
        "model_class": "src.models.baselines.geo_gat.GeoGATModel",
    },
    "row4_symmetric_ecmp": {
        "config_file": "configs/ablation_row4.yaml",
        "model_class": "src.models.baselines.symmetric_ecmp.SymmetricECMPModel",
    },
    "row5_cascrop": {
        "config_file": "configs/ablation_row5.yaml",
        "model_class": "src.models.cascrop.CasCrop",
    },
}


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        config = yaml.safe_load(f)
    # Load base config if specified
    if "_base_" in config:
        base_path = Path(config_path).parent / config["_base_"]
        with open(base_path) as f:
            base = yaml.safe_load(f)
        base.update({k: v for k, v in config.items() if k != "_base_"})
        config = base
    return config


def import_class(class_path: str):
    """Dynamically import a class from module path."""
    parts = class_path.rsplit(".", 1)
    module_path, class_name = parts[0], parts[1]
    import importlib
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


def set_seed(seed: int):
    """Set all random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def train_single(
    model_name: str,
    model_info: dict,
    seed: int,
    device: torch.device,
) -> dict:
    """Train a single model with a single seed."""
    from src.data.dataset import get_dataloaders
    from src.training.trainer import CasCropTrainer

    config = load_config(model_info["config_file"])
    set_seed(seed)

    logger.info(f"Training {model_name} with seed {seed}")

    # Load data
    train_loader, val_loader, test_loader = get_dataloaders(config)

    # Initialize model
    ModelClass = import_class(model_info["model_class"])
    model = ModelClass(config).to(device)

    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"  Model parameters: {param_count:,}")

    # Train
    trainer = CasCropTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
    )

    results = trainer.train(config["training"]["epochs"])

    # Save checkpoint
    ckpt_dir = Path(config["paths"]["checkpoints"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / f"{model_name}_seed{seed}.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": config,
            "seed": seed,
            "best_val_auc": results.get("best_val_auc", None),
            "epoch": results.get("best_epoch", None),
        },
        ckpt_path,
    )
    logger.info(f"  Checkpoint saved: {ckpt_path}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Train all CasCrop models")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--models", nargs="*", default=None,
                        help="Specific models to train (default: all)")
    parser.add_argument("--seeds", nargs="*", type=int, default=None,
                        help="Specific seeds to use (default: from config)")
    parser.add_argument("--gpu", type=int, default=0, help="GPU device ID")
    args = parser.parse_args()

    base_config = load_config(args.config)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    seeds = args.seeds or base_config.get("seeds", [42, 123, 456, 789, 1024])
    models_to_train = args.models or list(MODEL_CONFIGS.keys())

    all_results = {}

    for model_name in models_to_train:
        if model_name not in MODEL_CONFIGS:
            logger.warning(f"Unknown model: {model_name}, skipping")
            continue

        model_results = []
        for seed in seeds:
            try:
                result = train_single(
                    model_name, MODEL_CONFIGS[model_name], seed, device
                )
                result["seed"] = seed
                result["model"] = model_name
                model_results.append(result)
            except Exception as e:
                logger.error(f"Failed: {model_name} seed {seed}: {e}")
                model_results.append({"model": model_name, "seed": seed, "error": str(e)})

        all_results[model_name] = model_results

    # Save all training results
    results_dir = Path(base_config["paths"]["results"])
    results_dir.mkdir(parents=True, exist_ok=True)
    with open(results_dir / "training_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    logger.info("All training complete")

    # Print summary
    for model_name, results in all_results.items():
        aucs = [r.get("best_val_auc", 0) for r in results if "error" not in r]
        if aucs:
            logger.info(
                f"  {model_name}: Val AUC = {np.mean(aucs):.3f} ± {np.std(aucs):.3f}"
            )


if __name__ == "__main__":
    main()
