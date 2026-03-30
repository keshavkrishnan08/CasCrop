#!/usr/bin/env python3
"""Script 04: Train all CasCrop models on processed data.

Trains 5 ablation rows × 5 seeds = 25 model runs.
Features: checkpoint resume, incremental saves, NaN detection, GPU OOM handling.

Usage:
    python scripts/04_train_all.py                    # all models, all seeds
    python scripts/04_train_all.py --models cascrop   # just CasCrop
    python scripts/04_train_all.py --seeds 42         # single seed
    python scripts/04_train_all.py --epochs 10        # quick test
    python scripts/04_train_all.py --resume            # skip completed runs
"""

import argparse
import json
import logging
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

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
    graph_path = GRAPH / "combined_graph.npz"
    if not graph_path.exists():
        logger.warning("No graph file found. Using self-loops only.")
        n = features["fips"].nunique()
        edge_index = torch.stack([torch.arange(n), torch.arange(n)]).long()
        edge_weight = torch.ones(n)
    else:
        graph_data = np.load(graph_path)
        edge_index = torch.from_numpy(graph_data["edge_index"]).long()
        edge_weight = torch.from_numpy(graph_data["edge_weight"]).float()

    fips_idx_path = GRAPH / "fips_index.json"
    if fips_idx_path.exists():
        with open(fips_idx_path) as f:
            fips_to_idx = json.load(f)
    else:
        fips_list = sorted(features["fips"].unique())
        fips_to_idx = {f: i for i, f in enumerate(fips_list)}

    return features, labels, splits, stats, groups, edge_index, edge_weight, fips_to_idx


def build_tensors(features, labels, stats, groups, fips_to_idx):
    """Convert DataFrames to normalized tensors."""
    bio_cols = groups["biophysical"]
    econ_cols = groups["economic"]
    hist_cols = groups["historical"]

    def normalize_and_tensorize(df, cols, stats):
        if not cols:
            return torch.zeros(len(df), 1)
        X = df[cols].values.astype(np.float32)
        for i, col in enumerate(cols):
            if col in stats:
                std = stats[col]["std"]
                if std < 1e-8:
                    std = 1.0  # Prevent division by zero
                X[:, i] = (X[:, i] - stats[col]["mean"]) / std
        return torch.from_numpy(np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0))

    x_bio = normalize_and_tensorize(features, bio_cols, stats)
    x_econ = normalize_and_tensorize(features, econ_cols, stats)
    x_hist = normalize_and_tensorize(features, hist_cols, stats)

    # Labels
    y_waste = torch.from_numpy(labels["waste"].values.astype(np.float32)).unsqueeze(1)
    y_cause = torch.from_numpy(labels["cause_idx"].values.astype(np.int64))

    # Price shocks
    if "price_change_pct" in features.columns:
        shock_vals = features["price_change_pct"].fillna(0).values.astype(np.float32)
        # Clip extreme shocks to prevent NaN in attention
        shock_vals = np.clip(shock_vals, -5.0, 5.0)
        price_shocks = torch.from_numpy(shock_vals).unsqueeze(1)
    else:
        price_shocks = torch.zeros(len(features), 1)

    # Node indices
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


# ── Vectorized Edge Filtering ───────────────────────────────────────

def build_local_subgraph(batch_node_ids, edge_index, device, batch_size):
    """Build a local subgraph for the batch using vectorized ops.

    Much faster than the Python-loop version.
    """
    unique_nodes = torch.unique(batch_node_ids)

    # Vectorized membership check using broadcasting
    src, dst = edge_index[0], edge_index[1]
    src_in = torch.isin(src, unique_nodes)
    dst_in = torch.isin(dst, unique_nodes)
    edge_mask = src_in & dst_in

    if edge_mask.any():
        # Build global-to-local mapping
        max_node = unique_nodes.max().item() + 1
        g2l = torch.full((max_node,), -1, dtype=torch.long)
        g2l[unique_nodes] = torch.arange(len(unique_nodes))

        local_src = g2l[src[edge_mask]]
        local_dst = g2l[dst[edge_mask]]

        # Clamp to batch size (safety)
        local_src = local_src.clamp(0, batch_size - 1)
        local_dst = local_dst.clamp(0, batch_size - 1)

        return torch.stack([local_src, local_dst]).to(device)
    else:
        # Self-loops fallback
        return torch.stack([
            torch.arange(batch_size), torch.arange(batch_size)
        ]).to(device)


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
    """Train for one epoch with vectorized edge filtering and NaN detection."""
    model.train()
    total_loss = 0
    n_batches = 0

    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        B = batch["x_bio"].shape[0]

        # Vectorized subgraph construction
        local_edge_index = build_local_subgraph(
            batch["node_idx"], edge_index, device, B
        )
        batch["edge_index"] = local_edge_index
        batch["edge_attr"] = None

        optimizer.zero_grad()

        try:
            outputs = model(batch)
        except RuntimeError as e:
            if "out of memory" in str(e):
                logger.warning("GPU OOM in forward pass, skipping batch")
                torch.cuda.empty_cache()
                continue
            raise

        # Compute loss
        waste_loss = loss_fn(outputs["waste_logits"], batch["waste_target"])

        cause_loss = torch.tensor(0.0, device=device)
        if "cause_logits" in outputs:
            cause_loss = nn.CrossEntropyLoss()(outputs["cause_logits"], batch["cause_target"])

        dis_loss = outputs.get("disentangle_loss", torch.tensor(0.0, device=device))
        if isinstance(dis_loss, (int, float)):
            dis_loss = torch.tensor(dis_loss, device=device)

        loss = waste_loss + 0.3 * cause_loss + 0.1 * dis_loss

        # NaN detection
        if torch.isnan(loss) or torch.isinf(loss):
            logger.warning(f"NaN/Inf loss detected, skipping batch. "
                         f"waste={waste_loss.item():.4f}, cause={cause_loss.item():.4f}")
            optimizer.zero_grad()
            continue

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

        local_edge_index = build_local_subgraph(
            batch["node_idx"], edge_index, device, B
        )
        batch["edge_index"] = local_edge_index
        batch["edge_attr"] = None

        try:
            outputs = model(batch)
        except RuntimeError as e:
            if "out of memory" in str(e):
                torch.cuda.empty_cache()
                continue
            raise

        probs = torch.sigmoid(outputs["waste_logits"]).cpu()

        # NaN safety
        probs = torch.nan_to_num(probs, nan=0.5)

        all_probs.append(probs)
        all_targets.append(batch["waste_target"].cpu())

    if not all_probs:
        return {"auc_roc": 0.5, "f1": 0.0, "auc_pr": 0.5}

    y_prob = torch.cat(all_probs).numpy().flatten()
    y_true = torch.cat(all_targets).numpy().flatten()

    from sklearn.metrics import roc_auc_score, f1_score, average_precision_score

    try:
        auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        auc = 0.5  # Only one class present
    f1 = f1_score(y_true, (y_prob >= 0.5).astype(int), zero_division=0)
    try:
        ap = average_precision_score(y_true, y_prob)
    except ValueError:
        ap = 0.5

    return {"auc_roc": auc, "f1": f1, "auc_pr": ap, "y_prob": y_prob, "y_true": y_true}


def check_existing_checkpoint(model_name: str, seed: int) -> dict:
    """Check if a completed checkpoint exists for this model+seed."""
    ckpt_path = CKPT / f"{model_name}_seed{seed}.pt"
    if ckpt_path.exists():
        try:
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            if "test_metrics" in ckpt and ckpt["test_metrics"].get("auc_roc", 0) > 0:
                return {
                    "model": model_name,
                    "seed": seed,
                    "best_val_auc": ckpt.get("best_val_auc", 0),
                    "best_epoch": ckpt.get("best_epoch", 0),
                    "test_auc_roc": ckpt["test_metrics"]["auc_roc"],
                    "test_f1": ckpt["test_metrics"]["f1"],
                    "test_auc_pr": ckpt["test_metrics"]["auc_pr"],
                    "n_params": ckpt.get("n_params", 0),
                    "resumed": True,
                }
        except Exception:
            pass
    return None


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
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    model = create_model(model_name, bio_dim, econ_dim, hist_dim).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"  {model_name} | seed={seed} | params={n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=50)

    from training.losses import FocalLoss
    loss_fn = FocalLoss(gamma=2.0, alpha=0.75)

    best_auc = 0
    best_epoch = 0
    wait = 0
    best_state = None

    for epoch in range(epochs):
        t0 = time.time()

        try:
            train_loss = train_one_epoch(
                model, train_loader, optimizer, loss_fn,
                edge_index, edge_weight, device,
            )
        except RuntimeError as e:
            if "out of memory" in str(e):
                logger.error(f"GPU OOM at epoch {epoch}. Try reducing batch size.")
                torch.cuda.empty_cache()
                break
            raise

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
        "n_params": n_params,
        "test_metrics": {k: v for k, v in test_metrics.items() if k not in ("y_prob", "y_true")},
    }, ckpt_path)

    # Free GPU memory
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

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


def save_results_incremental(all_results: list):
    """Save results after each model run (crash-safe)."""
    RESULTS.mkdir(parents=True, exist_ok=True)
    with open(RESULTS / "training_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)


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
    parser.add_argument("--resume", action="store_true",
                        help="Skip model+seed combos that already have checkpoints")
    parser.add_argument("--graph", type=str, default=None,
                        help="Path to alternative graph .npz file (for edge ablation)")
    args = parser.parse_args()

    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device}")
    if torch.cuda.is_available():
        mem = torch.cuda.get_device_properties(args.gpu).total_mem / 1e9
        logger.info(f"GPU: {torch.cuda.get_device_name(args.gpu)} ({mem:.1f} GB)")
        if mem < 8:
            logger.info("Low GPU memory detected, reducing batch size to 256")
            args.batch_size = min(args.batch_size, 256)

    # Load data
    logger.info("Loading data...")
    features, labels, splits, stats, groups, edge_index, edge_weight, fips_to_idx = load_data()

    # Override graph if specified (for edge ablation)
    if args.graph:
        graph_data = np.load(args.graph)
        edge_index = torch.from_numpy(graph_data["edge_index"]).long()
        edge_weight = torch.from_numpy(graph_data["edge_weight"]).float()
        logger.info(f"Using custom graph: {args.graph}")

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
        return DataLoader(ds, batch_size=args.batch_size, shuffle=shuffle,
                         drop_last=False, num_workers=0, pin_memory=torch.cuda.is_available())

    train_loader = make_loader(splits["train"], shuffle=True)
    val_loader = make_loader(splits["val"])
    test_loader = make_loader(splits["test"])

    logger.info(f"Train: {len(splits['train']):,} | Val: {len(splits['val']):,} | "
                f"Test: {len(splits['test']):,}")

    models = args.models or list(MODEL_REGISTRY.keys())
    seeds = args.seeds or [42, 123, 456, 789, 1024]

    logger.info(f"Models: {models}")
    logger.info(f"Seeds: {seeds}")
    logger.info(f"Total runs: {len(models) * len(seeds)}")
    if args.resume:
        logger.info("Resume mode: skipping completed runs")

    # Train all
    all_results = []
    completed = 0
    total = len(models) * len(seeds)

    for model_name in models:
        if model_name not in MODEL_REGISTRY:
            logger.warning(f"Unknown model: {model_name}, skipping")
            continue

        logger.info(f"\n{'='*60}")
        logger.info(f"Model: {model_name}")
        logger.info(f"{'='*60}")

        for seed in seeds:
            completed += 1
            logger.info(f"\n--- Run {completed}/{total}: {model_name} seed={seed} ---")

            # Check for existing checkpoint
            if args.resume:
                existing = check_existing_checkpoint(model_name, seed)
                if existing:
                    logger.info(f"  Checkpoint found: AUC={existing['test_auc_roc']:.4f}, skipping")
                    all_results.append(existing)
                    save_results_incremental(all_results)
                    continue

            try:
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
            except Exception as e:
                logger.error(f"FAILED: {model_name} seed={seed}: {e}")
                traceback.print_exc()
                all_results.append({
                    "model": model_name, "seed": seed, "error": str(e),
                    "test_auc_roc": 0, "test_f1": 0, "test_auc_pr": 0, "n_params": 0,
                })

            # Save after every run (crash-safe)
            save_results_incremental(all_results)

    # Final save + summary
    save_results_incremental(all_results)

    import pandas as pd
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(RESULTS / "training_results.csv", index=False)

    logger.info(f"\n{'='*60}")
    logger.info("FINAL RESULTS")
    logger.info(f"{'='*60}")
    logger.info(f"{'Model':<20} {'AUC-ROC':>12} {'F1':>12} {'AUC-PR':>12} {'Params':>10}")
    logger.info("-" * 70)

    for model_name in models:
        model_results = results_df[
            (results_df["model"] == model_name) & (results_df["test_auc_roc"] > 0)
        ]
        if len(model_results) == 0:
            continue
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

    logger.info(f"\nResults saved to {RESULTS}/")


if __name__ == "__main__":
    main()
