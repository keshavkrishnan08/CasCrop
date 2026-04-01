#!/usr/bin/env python3
"""Train temporal cascade models on monthly data.

Usage:
    python scripts/04_train_temporal.py --epochs 100 --patience 20 --seeds 42 123 456
    python scripts/04_train_temporal.py --models temporal_cascrop --seeds 42 --epochs 5  # quick test
"""

import argparse, json, logging, os, sys, time, traceback
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PROC = Path("data/processed")
GRAPH = Path("data/graphs")
CKPT = Path("checkpoints")
RESULTS = Path("results")

# ── Data ─────────────────────────────────────────────────────────────

def load_monthly_data():
    import pandas as pd
    features = pd.read_parquet(PROC / "features_monthly.parquet")
    labels = pd.read_parquet(PROC / "labels_monthly.parquet")
    with open(PROC / "splits_monthly.json") as f: splits = json.load(f)
    with open(PROC / "stats_monthly.json") as f: stats = json.load(f)
    with open(PROC / "feature_groups_monthly.json") as f: groups = json.load(f)
    graph = np.load(GRAPH / "combined_graph.npz")
    edge_index = torch.from_numpy(graph["edge_index"]).long()
    with open(GRAPH / "fips_index.json") as f: fips_to_idx = json.load(f)
    return features, labels, splits, stats, groups, edge_index, fips_to_idx


def build_tensors(features, labels, stats, groups, fips_to_idx):
    bio_cols = groups["biophysical"]
    econ_cols = groups["economic"]
    hist_cols = groups["historical"]

    def norm(df, cols):
        X = df[cols].values.astype(np.float32)
        for i, c in enumerate(cols):
            if c in stats:
                s = max(stats[c]["std"], 1e-8)
                X[:, i] = (X[:, i] - stats[c]["mean"]) / s
        return torch.from_numpy(np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0))

    x_bio = norm(features, bio_cols)
    x_econ = norm(features, econ_cols)
    x_hist = norm(features, hist_cols)
    y_waste = torch.from_numpy(labels["waste"].values.astype(np.float32)).unsqueeze(1)
    y_cause = torch.from_numpy(labels["cause_idx"].values.astype(np.int64))

    shock_col = "county_shock" if "county_shock" in features.columns else "price_change_1m"
    if shock_col in features.columns:
        shocks = torch.from_numpy(np.clip(features[shock_col].fillna(0).values.astype(np.float32), -5, 5)).unsqueeze(1)
    else:
        shocks = torch.zeros(len(features), 1)

    node_idx = torch.from_numpy(features["fips"].map(fips_to_idx).fillna(0).values.astype(np.int64))
    years = features["year"].values
    months = features["month"].values

    logger.info(f"  Tensors: bio={x_bio.shape}, econ={x_econ.shape}, shocks unique={shocks.unique().shape[0]}")
    return x_bio, x_econ, x_hist, y_waste, y_cause, shocks, node_idx, years, months


class MonthlyDataset(Dataset):
    """Dataset that returns single-month samples for non-temporal models,
    or can be used with a temporal collator for sequence models."""
    def __init__(self, x_bio, x_econ, x_hist, y_waste, y_cause, shocks, node_idx):
        self.x_bio = x_bio; self.x_econ = x_econ; self.x_hist = x_hist
        self.y_waste = y_waste; self.y_cause = y_cause
        self.shocks = shocks; self.node_idx = node_idx

    def __len__(self): return len(self.x_bio)

    def __getitem__(self, idx):
        return {
            "x_bio": self.x_bio[idx], "x_econ": self.x_econ[idx], "x_hist": self.x_hist[idx],
            "waste_target": self.y_waste[idx], "cause_target": self.y_cause[idx],
            "price_shocks": self.shocks[idx], "node_idx": self.node_idx[idx],
        }


# ── Subgraph ─────────────────────────────────────────────────────────

def build_local_subgraph(batch_node_ids, edge_index, device, batch_size):
    unique_nodes = torch.unique(batch_node_ids)
    src, dst = edge_index[0], edge_index[1]
    src_in = torch.isin(src, unique_nodes)
    dst_in = torch.isin(dst, unique_nodes)
    edge_mask = src_in & dst_in
    if edge_mask.any():
        max_node = unique_nodes.max().item() + 1
        g2l = torch.full((max_node,), -1, dtype=torch.long)
        g2l[unique_nodes] = torch.arange(len(unique_nodes))
        local_src = g2l[src[edge_mask]].clamp(0, batch_size - 1)
        local_dst = g2l[dst[edge_mask]].clamp(0, batch_size - 1)
        return torch.stack([local_src, local_dst]).to(device)
    return torch.stack([torch.arange(batch_size), torch.arange(batch_size)]).to(device)


# ── Models ───────────────────────────────────────────────────────────

MODEL_REGISTRY = {
    "temporal_local": ("models.baselines.temporal_local", "TemporalLocalModel"),
    "temporal_gat": ("models.baselines.temporal_gat", "TemporalGATModel"),
    "temporal_symmetric": ("models.baselines.temporal_symmetric", "TemporalSymmetricModel"),
    "temporal_cascrop": ("models.temporal_cascrop", "TemporalCasCrop"),
}

def create_model(name, bio_dim, econ_dim, hist_dim):
    import importlib
    mod_path, cls_name = MODEL_REGISTRY[name]
    cls = getattr(importlib.import_module(mod_path), cls_name)
    return cls(bio_input_dim=bio_dim, econ_input_dim=econ_dim, hist_dim=hist_dim,
               latent_dim=64, hidden_dim=64, num_heads=4, dropout=0.3)


# ── Training ─────────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, loss_fn, edge_index, device, dis_lambda=0.1):
    model.train()
    total_loss, n = 0, 0
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        B = batch["x_bio"].shape[0]
        local_ei = build_local_subgraph(batch["node_idx"], edge_index, device, B)
        batch["edge_index"] = local_ei; batch["edge_attr"] = None
        optimizer.zero_grad()
        try:
            out = model(batch)
        except RuntimeError as e:
            if "out of memory" in str(e): torch.cuda.empty_cache(); continue
            raise
        waste_loss = loss_fn(out["waste_logits"], batch["waste_target"])
        cause_loss = nn.CrossEntropyLoss()(out["cause_logits"], batch["cause_target"]) if "cause_logits" in out else torch.tensor(0.0, device=device)
        dis_loss = out.get("disentangle_loss", torch.tensor(0.0, device=device))
        if isinstance(dis_loss, (int, float)): dis_loss = torch.tensor(dis_loss, device=device)
        loss = waste_loss + 0.3 * cause_loss + dis_lambda * dis_loss
        if torch.isnan(loss): continue
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item(); n += 1
    return total_loss / max(n, 1)


@torch.no_grad()
def evaluate(model, loader, edge_index, device):
    model.eval()
    all_p, all_t = [], []
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        B = batch["x_bio"].shape[0]
        batch["edge_index"] = build_local_subgraph(batch["node_idx"], edge_index, device, B)
        batch["edge_attr"] = None
        try: out = model(batch)
        except: continue
        all_p.append(torch.sigmoid(out["waste_logits"]).cpu())
        all_t.append(batch["waste_target"].cpu())
    if not all_p: return {"auc_roc": 0.5, "f1": 0.0, "auc_pr": 0.5}
    yp = torch.cat(all_p).numpy().flatten()
    yt = torch.cat(all_t).numpy().flatten()
    from sklearn.metrics import roc_auc_score, f1_score, average_precision_score
    try: auc = roc_auc_score(yt, yp)
    except: auc = 0.5
    f1 = f1_score(yt, (yp >= 0.5).astype(int), zero_division=0)
    try: ap = average_precision_score(yt, yp)
    except: ap = 0.5
    return {"auc_roc": auc, "f1": f1, "auc_pr": ap}


def train_model(model_name, seed, train_loader, val_loader, test_loader,
                edge_index, bio_dim, econ_dim, hist_dim, epochs, patience, lr, device):
    torch.manual_seed(seed); np.random.seed(seed)
    model = create_model(model_name, bio_dim, econ_dim, hist_dim).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"  {model_name} | seed={seed} | params={n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=50)
    # Compute pos_weight from train loader for class imbalance
    all_y = torch.cat([b["waste_target"] for b in train_loader])
    waste_rate = all_y.mean().item()
    pos_weight = torch.tensor([(1 - waste_rate) / max(waste_rate, 0.01)]).to(device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_auc, best_epoch, wait, best_state = 0, 0, 0, None
    WARMUP = 5

    for epoch in range(epochs):
        t0 = time.time()
        dis_lambda = 0.0 if epoch < WARMUP else 0.1
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, edge_index, device, dis_lambda)
        scheduler.step()
        val = evaluate(model, val_loader, edge_index, device)
        if val["auc_roc"] > best_auc:
            best_auc = val["auc_roc"]; best_epoch = epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}; wait = 0
        else: wait += 1
        if epoch % 10 == 0 or wait == 0:
            logger.info(f"    Ep {epoch:3d} | loss={train_loss:.4f} | val={val['auc_roc']:.4f} | best={best_auc:.4f}(ep{best_epoch}) | {time.time()-t0:.1f}s")
        if wait >= patience: logger.info(f"    Early stop ep {epoch}"); break

    if best_state: model.load_state_dict(best_state)
    model.to(device)
    test = evaluate(model, test_loader, edge_index, device)
    logger.info(f"  TEST: AUC={test['auc_roc']:.4f} | F1={test['f1']:.4f} | AP={test['auc_pr']:.4f}")

    CKPT.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": best_state or model.state_dict(), "model_name": model_name,
                "seed": seed, "best_val_auc": best_auc, "best_epoch": best_epoch, "n_params": n_params,
                "test_metrics": test}, CKPT / f"{model_name}_seed{seed}.pt")
    del model
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return {"model": model_name, "seed": seed, "best_val_auc": best_auc, "best_epoch": best_epoch,
            "test_auc_roc": test["auc_roc"], "test_f1": test["f1"], "test_auc_pr": test["auc_pr"], "n_params": n_params}


# ── Main ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="*", default=None)
    parser.add_argument("--seeds", nargs="*", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device}")

    features, labels, splits, stats, groups, edge_index, fips_to_idx = load_monthly_data()
    bio_dim = len(groups["biophysical"]); econ_dim = len(groups["economic"]); hist_dim = len(groups["historical"])
    logger.info(f"Monthly data: {len(features):,} samples, bio={bio_dim}, econ={econ_dim}, hist={hist_dim}")

    x_bio, x_econ, x_hist, y_waste, y_cause, shocks, node_idx, years, months = \
        build_tensors(features, labels, stats, groups, fips_to_idx)

    def make_loader(indices, shuffle=False):
        idx = torch.tensor(indices)
        ds = MonthlyDataset(x_bio[idx], x_econ[idx], x_hist[idx], y_waste[idx], y_cause[idx], shocks[idx], node_idx[idx])
        return DataLoader(ds, batch_size=args.batch_size, shuffle=shuffle, drop_last=shuffle, num_workers=0)

    train_loader = make_loader(splits["train"], shuffle=True)
    val_loader = make_loader(splits["val"])
    test_loader = make_loader(splits["test"])
    logger.info(f"Train: {len(splits['train']):,} | Val: {len(splits['val']):,} | Test: {len(splits['test']):,}")

    models = args.models or list(MODEL_REGISTRY.keys())
    seeds = args.seeds or [42, 123, 456, 789, 1024]
    all_results = []

    for model_name in models:
        logger.info(f"\n{'='*50}\nModel: {model_name}\n{'='*50}")
        for seed in seeds:
            # Resume check
            ckpt_path = CKPT / f"{model_name}_seed{seed}.pt"
            if args.resume and ckpt_path.exists():
                try:
                    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
                    if "test_metrics" in ckpt and ckpt["test_metrics"].get("auc_roc", 0) > 0:
                        logger.info(f"  Checkpoint: AUC={ckpt['test_metrics']['auc_roc']:.4f}, skip")
                        all_results.append({"model": model_name, "seed": seed,
                            "test_auc_roc": ckpt["test_metrics"]["auc_roc"],
                            "test_f1": ckpt["test_metrics"]["f1"],
                            "test_auc_pr": ckpt["test_metrics"]["auc_pr"],
                            "n_params": ckpt.get("n_params", 0), "best_val_auc": ckpt.get("best_val_auc", 0),
                            "best_epoch": ckpt.get("best_epoch", 0)})
                        continue
                except: pass

            try:
                r = train_model(model_name, seed, train_loader, val_loader, test_loader,
                                edge_index, bio_dim, econ_dim, hist_dim, args.epochs, args.patience, args.lr, device)
                all_results.append(r)
            except Exception as e:
                logger.error(f"FAILED: {model_name} seed={seed}: {e}")
                traceback.print_exc()
                all_results.append({"model": model_name, "seed": seed, "test_auc_roc": 0, "test_f1": 0, "test_auc_pr": 0, "n_params": 0})

            RESULTS.mkdir(parents=True, exist_ok=True)
            with open(RESULTS / "temporal_results.json", "w") as f:
                json.dump(all_results, f, indent=2, default=str)

    # Summary
    import pandas as pd
    df = pd.DataFrame(all_results)
    df.to_csv(RESULTS / "temporal_results.csv", index=False)
    logger.info(f"\n{'='*60}\nFINAL RESULTS\n{'='*60}")
    for m in models:
        d = df[(df["model"]==m) & (df["test_auc_roc"]>0)]
        if len(d): logger.info(f"{m:<25} AUC={d['test_auc_roc'].mean():.4f}±{d['test_auc_roc'].std():.4f}  F1={d['test_f1'].mean():.3f}")


if __name__ == "__main__":
    main()
