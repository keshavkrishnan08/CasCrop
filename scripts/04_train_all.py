#!/usr/bin/env python3
"""Script 04: Train all CasCrop models on processed data.

Trains 5 ablation rows × 5 seeds = 25 model runs.
Uses the actual processed data from scripts/02 and graphs from scripts/03.

Usage:
    python scripts/04_train_all.py                    # all models, all seeds
    python scripts/04_train_all.py --models cascrop   # just CasCrop
    python scripts/04_train_all.py --seeds 42         # single seed
    python scripts/04_train_all.py --epochs 10        # quick test
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

PROC = Path("data/processed")
GRAPH = Path("data/graphs")
CKPT = Path("checkpoints")
RESULTS = Path("results")

# ── Data Loading ─────────────────────────────────────────────────────

def load_data():
    """Load processed features, labels, splits, and graph."""
    import pandas as pd

    features = pd.read_parquet(PROC / "features.parquet")
    labels = pd.read_parquet(PROC / "labels.parquet")
    with open(PROC / "splits.json") as f:
        splits = json.load(f)
    with open(PROC / "stats.json") as f:
        stats = json.load(f)
    with open(PROC / "feature_groups.json") as f:
        groups = json.load(f)

    # Load graph
    graph_data = np.load(GRAPH / "combined_graph.npz")
    edge_index = torch.from_numpy(graph_data["edge_index"]).long()
    edge_weight = torch.from_numpy(graph_data["edge_weight"]).float()

    with open(GRAPH / "fips_index.json") as f:
        fips_to_idx = json.load(f)

    return features, labels, splits, stats, groups, edge_index, edge_weight, fips_to_idx


def build_tensors(features, labels, stats, groups, fips_to_idx):
    """Convert DataFrames to normalized tensors."""
    # Biophysical features
    bio_cols = groups["biophysical"]
    econ_cols = groups["economic"]
    hist_cols = groups["historical"]

    def normalize_and_tensorize(df, cols, stats):
        X = df[cols].values.astype(np.float32)
        for i, col in enumerate(cols):
            if col in stats:
                X[:, i] = (X[:, i] - stats[col]["mean"]) / stats[col]["std"]
        return torch.from_numpy(np.nan_to_num(X, 0.0))

    x_bio = normalize_and_tensorize(features, bio_cols, stats)
    x_econ = normalize_and_tensorize(features, econ_cols, stats)
    x_hist = normalize_and_tensorize(features, hist_cols, stats)

    # Labels
    y_waste = torch.from_numpy(labels["waste"].values.astype(np.float32)).unsqueeze(1)
    y_cause = torch.from_numpy(labels["cause_idx"].values.astype(np.int64))

    # Price shocks (year-over-year price change as shock signal)
    if "price_change_pct" in features.columns:
        price_shocks = torch.from_numpy(
            features["price_change_pct"].fillna(0).values.astype(np.float32)
        ).unsqueeze(1)
    else:
        price_shocks = torch.zeros(len(features), 1)

    # Node indices for graph lookup
    node_idx = torch.from_numpy(
        features["fips"].map(fips_to_idx).fillna(0).values.astype(np.int64)
    )

    logger.info(f"  Tensors: bio={x_bio.shape}, econ={x_econ.shape}, hist={x_hist.shape}")
    logger.info(f"  Waste rate: {y_waste.mean():.1%}")

    return x_bio, x_econ, x_hist, y_waste, y_cause, price_shocks, node_idx


class CropDataset(torch.utils.data.Dataset):
    def __init__(self, x_bio, x_econ, x_hist, y_waste, y_cause, price_shocks, node_idx):
        self.x_bio = x_bio
        self.x_econ = x_econ
        self.x_hist = x_hist
        self.y_waste = y_waste
        self.y_cause = y_cause
        self.price_shocks = price_shocks
        self.node_idx = node_idx

    def __len__(self):
        return len(self.x_bio)

    def __getitem__(self, idx):
        return {
            "x_bio": self.x_bio[idx],
            "x_econ": self.x_econ[idx],
            "x_hist": self.x_hist[idx],
            "waste_target": self.y_waste[idx],
            "cause_target": self.y_cause[idx],
            "price_shocks": self.price_shocks[idx],
            "node_idx": self.node_idx[idx],
        }


# ── Model Factory ────────────────────────────────────────────────────

MODEL_REGISTRY = {
    "local_only": ("models.baselines.local_only", "LocalOnlyModel"),
    "local_econ": ("models.baselines.local_econ", "LocalEconModel"),
    "geo_gat": ("models.baselines.geo_gat", "GeoGATModel"),
    "symmetric_ecmp": ("models.baselines.symmetric_ecmp", "SymmetricECMPModel"),
    "cascrop": ("models.cascrop", "CasCrop"),
}


def create_model(name: str, bio_dim: int, econ_dim: int, hist_dim: int):
    """Instantiate a model by name."""
    import importlib
    module_path, class_name = MODEL_REGISTRY[name]
    mod = importlib.import_module(module_path)
    cls = getattr(mod, class_name)

    kwargs = {"bio_input_dim": bio_dim, "latent_dim": 64, "dropout": 0.3}

    if name != "local_only":
        kwargs["econ_input_dim"] = econ_dim
    if name in ("geo_gat", "symmetric_ecmp", "cascrop"):
        kwargs["hist_dim"] = hist_dim
        kwargs["num_heads"] = 4

    return cls(**kwargs)


# ── Training Loop ────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, loss_fn, edge_index, edge_weight,
                    device, clip_norm=1.0):
    """Train for one epoch.

    Note: For graph models, we build a local subgraph for each batch
    by mapping node_idx to local indices and filtering edges.
    """
    model.train()
    total_loss = 0
    n_batches = 0

    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        B = batch["x_bio"].shape[0]

        # Build local subgraph for this batch
        # node_idx maps each sample to its position in the full graph
        batch_node_ids = batch["node_idx"]  # (B,) indices into full graph
        unique_nodes, inverse = torch.unique(batch_node_ids, return_inverse=True)

        # Filter edges to only those between nodes in this batch
        src, dst = edge_index[0], edge_index[1]
        node_set = set(unique_nodes.cpu().tolist())
        edge_mask = torch.tensor(
            [s.item() in node_set and d.item() in node_set for s, d in zip(src, dst)],
            dtype=torch.bool,
        )

        if edge_mask.any():
            # Remap edge indices to local batch indices
            global_to_local = {g.item(): l for l, g in enumerate(unique_nodes)}
            local_src = torch.tensor([global_to_local[s.item()] for s in src[edge_mask]])
            local_dst = torch.tensor([global_to_local[d.item()] for d in dst[edge_mask]])
            local_edge_index = torch.stack([local_src, local_dst]).to(device)
        else:
            # No edges in this batch — create self-loops so ECMP doesn't crash
            local_edge_index = torch.stack([
                torch.arange(B), torch.arange(B)
            ]).to(device)

        batch["edge_index"] = local_edge_index
        batch["edge_attr"] = None

        # Price shocks need to be indexed by local node position
        # The batch already has per-sample price_shocks, which works directly
        # since the model indexes shocks by the edge source indices

        optimizer.zero_grad()
        outputs = model(batch)

        # Compute loss
        waste_loss = loss_fn(outputs["waste_logits"], batch["waste_target"])

        # Add cause loss if available
        cause_loss = torch.tensor(0.0, device=device)
        if "cause_logits" in outputs:
            cause_loss = nn.CrossEntropyLoss()(outputs["cause_logits"], batch["cause_target"])

        # Add disentanglement loss if available
        dis_loss = outputs.get("disentangle_loss", torch.tensor(0.0, device=device))

        loss = waste_loss + 0.3 * cause_loss + 0.1 * dis_loss
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate(model, loader, edge_index, device):
    """Evaluate on val/test set."""
    model.eval()
    all_probs, all_targets = [], []

    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        B = batch["x_bio"].shape[0]

        # Build local subgraph (same as training)
        batch_node_ids = batch["node_idx"]
        unique_nodes, inverse = torch.unique(batch_node_ids, return_inverse=True)
        src, dst = edge_index[0], edge_index[1]
        node_set = set(unique_nodes.cpu().tolist())
        edge_mask = torch.tensor(
            [s.item() in node_set and d.item() in node_set for s, d in zip(src, dst)],
            dtype=torch.bool,
        )
        if edge_mask.any():
            global_to_local = {g.item(): l for l, g in enumerate(unique_nodes)}
            local_src = torch.tensor([global_to_local[s.item()] for s in src[edge_mask]])
            local_dst = torch.tensor([global_to_local[d.item()] for d in dst[edge_mask]])
            local_edge_index = torch.stack([local_src, local_dst]).to(device)
        else:
            local_edge_index = torch.stack([torch.arange(B), torch.arange(B)]).to(device)

        batch["edge_index"] = local_edge_index
        batch["edge_attr"] = None

        outputs = model(batch)
        probs = torch.sigmoid(outputs["waste_logits"]).cpu()
        all_probs.append(probs)
        all_targets.append(batch["waste_target"].cpu())

    y_prob = torch.cat(all_probs).numpy().flatten()
    y_true = torch.cat(all_targets).numpy().flatten()

    from sklearn.metrics import roc_auc_score, f1_score, average_precision_score

    auc = roc_auc_score(y_true, y_prob)
    f1 = f1_score(y_true, (y_prob >= 0.5).astype(int), zero_division=0)
    ap = average_precision_score(y_true, y_prob)

    return {"auc_roc": auc, "f1": f1, "auc_pr": ap, "y_prob": y_prob, "y_true": y_true}


def train_model(
    model_name: str,
    seed: int,
    train_loader, val_loader, test_loader,
    edge_index, edge_weight,
    bio_dim, econ_dim, hist_dim,
    epochs: int = 200,
    patience: int = 20,
    lr: float = 0.001,
    device: str = "cpu",
):
    """Full training pipeline for one model + one seed."""
    # Seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Model
    model = create_model(model_name, bio_dim, econ_dim, hist_dim).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"  {model_name} | seed={seed} | params={n_params:,}")

    # Optimizer + loss
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=50)

    # Focal loss
    from training.losses import FocalLoss
    loss_fn = FocalLoss(gamma=2.0, alpha=0.75)

    # Training loop
    best_auc = 0
    best_epoch = 0
    wait = 0
    best_state = None

    for epoch in range(epochs):
        t0 = time.time()
        train_loss = train_one_epoch(
            model, train_loader, optimizer, loss_fn,
            edge_index, edge_weight, device,
        )
        scheduler.step()

        val_metrics = evaluate(model, val_loader, edge_index, device)
        val_auc = val_metrics["auc_roc"]

        if val_auc > best_auc:
            best_auc = val_auc
            best_epoch = epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1

        if epoch % 10 == 0 or wait == 0:
            logger.info(
                f"    Epoch {epoch:3d} | loss={train_loss:.4f} | "
                f"val_AUC={val_auc:.4f} | best={best_auc:.4f} (ep{best_epoch}) | "
                f"{time.time()-t0:.1f}s"
            )

        if wait >= patience:
            logger.info(f"    Early stopping at epoch {epoch}")
            break

    # Load best and evaluate on test
    if best_state:
        model.load_state_dict(best_state)
    model.to(device)

    test_metrics = evaluate(model, test_loader, edge_index, device)
    logger.info(
        f"  TEST: AUC={test_metrics['auc_roc']:.4f} | "
        f"F1={test_metrics['f1']:.4f} | AUC-PR={test_metrics['auc_pr']:.4f}"
    )

    # Save checkpoint
    CKPT.mkdir(parents=True, exist_ok=True)
    ckpt_path = CKPT / f"{model_name}_seed{seed}.pt"
    torch.save({
        "model_state_dict": best_state or model.state_dict(),
        "model_name": model_name,
        "seed": seed,
        "best_val_auc": best_auc,
        "best_epoch": best_epoch,
        "test_metrics": {k: v for k, v in test_metrics.items() if k not in ("y_prob", "y_true")},
    }, ckpt_path)

    return {
        "model": model_name,
        "seed": seed,
        "best_val_auc": best_auc,
        "best_epoch": best_epoch,
        "test_auc_roc": test_metrics["auc_roc"],
        "test_f1": test_metrics["f1"],
        "test_auc_pr": test_metrics["auc_pr"],
        "n_params": n_params,
    }


# ── Main ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train all CasCrop models")
    parser.add_argument("--models", nargs="*", default=None)
    parser.add_argument("--seeds", nargs="*", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device}")

    # Load data
    logger.info("Loading data...")
    features, labels, splits, stats, groups, edge_index, edge_weight, fips_to_idx = load_data()

    bio_dim = len(groups["biophysical"])
    econ_dim = len(groups["economic"])
    hist_dim = len(groups["historical"])
    logger.info(f"Feature dims: bio={bio_dim}, econ={econ_dim}, hist={hist_dim}")

    # Build tensors
    x_bio, x_econ, x_hist, y_waste, y_cause, price_shocks, node_idx = \
        build_tensors(features, labels, stats, groups, fips_to_idx)

    # Create datasets
    def make_loader(indices, shuffle=False):
        idx = torch.tensor(indices)
        ds = CropDataset(
            x_bio[idx], x_econ[idx], x_hist[idx],
            y_waste[idx], y_cause[idx], price_shocks[idx], node_idx[idx],
        )
        return DataLoader(ds, batch_size=args.batch_size, shuffle=shuffle, drop_last=False)

    train_loader = make_loader(splits["train"], shuffle=True)
    val_loader = make_loader(splits["val"])
    test_loader = make_loader(splits["test"])

    logger.info(f"Train: {len(splits['train']):,} | Val: {len(splits['val']):,} | "
                f"Test: {len(splits['test']):,}")

    # Models and seeds
    models = args.models or list(MODEL_REGISTRY.keys())
    seeds = args.seeds or [42, 123, 456, 789, 1024]

    logger.info(f"Models: {models}")
    logger.info(f"Seeds: {seeds}")
    logger.info(f"Total runs: {len(models) * len(seeds)}")

    # Train all
    all_results = []
    for model_name in models:
        logger.info(f"\n{'='*60}")
        logger.info(f"Model: {model_name}")
        logger.info(f"{'='*60}")

        for seed in seeds:
            result = train_model(
                model_name, seed,
                train_loader, val_loader, test_loader,
                edge_index, edge_weight,
                bio_dim, econ_dim, hist_dim,
                epochs=args.epochs,
                patience=args.patience,
                lr=args.lr,
                device=device,
            )
            all_results.append(result)

    # Save results
    RESULTS.mkdir(parents=True, exist_ok=True)
    with open(RESULTS / "training_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    # Print summary table
    logger.info(f"\n{'='*60}")
    logger.info("FINAL RESULTS")
    logger.info(f"{'='*60}")
    logger.info(f"{'Model':<20} {'AUC-ROC':>12} {'F1':>12} {'AUC-PR':>12} {'Params':>10}")
    logger.info("-" * 70)

    import pandas as pd
    results_df = pd.DataFrame(all_results)
    for model_name in models:
        model_results = results_df[results_df["model"] == model_name]
        auc_mean = model_results["test_auc_roc"].mean()
        auc_std = model_results["test_auc_roc"].std()
        f1_mean = model_results["test_f1"].mean()
        f1_std = model_results["test_f1"].std()
        ap_mean = model_results["test_auc_pr"].mean()
        ap_std = model_results["test_auc_pr"].std()
        params = model_results["n_params"].iloc[0]
        logger.info(
            f"{model_name:<20} {auc_mean:.3f}±{auc_std:.3f}  "
            f"{f1_mean:.3f}±{f1_std:.3f}  {ap_mean:.3f}±{ap_std:.3f}  {params:>10,}"
        )

    results_df.to_csv(RESULTS / "training_results.csv", index=False)
    logger.info(f"\nResults saved to {RESULTS}/")


if __name__ == "__main__":
    main()
