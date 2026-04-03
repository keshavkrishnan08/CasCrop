#!/usr/bin/env python3
"""Train Cascade Diffusion models on pre-computed diffusion features.

No online GNN — graph information is baked into diffusion features.
Standard mini-batch training, fast and reliable on any hardware.

Usage:
    python scripts/precompute_diffusion.py          # run once
    python scripts/train_cascade.py --epochs 100    # train all models
    python scripts/train_cascade.py --models cascade_net --seeds 42 --epochs 5  # quick test
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
CKPT = Path("checkpoints")
RESULTS = Path("results")


# ── Data ─────────────────────────────────────────────────────────────

def load_data():
    import pandas as pd
    feat_path = PROC / "features_diffusion.parquet"
    if not feat_path.exists():
        raise FileNotFoundError(
            f"{feat_path} not found. Run: python scripts/precompute_diffusion.py"
        )
    features = pd.read_parquet(feat_path)
    labels = pd.read_parquet(PROC / "labels_monthly.parquet")
    with open(PROC / "splits_monthly.json") as f:
        splits = json.load(f)
    with open(PROC / "stats_monthly.json") as f:
        stats = json.load(f)
    with open(PROC / "feature_groups_diffusion.json") as f:
        groups = json.load(f)
    return features, labels, splits, stats, groups


def build_tensors(features, labels, stats, groups):
    bio_cols = groups["biophysical"]
    econ_cols = groups["economic"]
    hist_cols = groups["historical"]
    diff_cols = groups["diffusion"]

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

    # Diffusion features: normalize by their own stats
    x_diff_raw = features[diff_cols].values.astype(np.float32)
    for i, c in enumerate(diff_cols):
        m = np.nanmean(x_diff_raw[:, i])
        s = max(np.nanstd(x_diff_raw[:, i]), 1e-8)
        x_diff_raw[:, i] = (x_diff_raw[:, i] - m) / s
    x_diff = torch.from_numpy(np.nan_to_num(x_diff_raw, nan=0.0))

    y_waste = torch.from_numpy(labels["waste"].values.astype(np.float32)).unsqueeze(1)
    y_cause = torch.from_numpy(labels["cause_idx"].values.astype(np.int64))

    logger.info(f"  Tensors: bio={x_bio.shape}, econ={x_econ.shape}, "
                f"diff={x_diff.shape}, hist={x_hist.shape}")
    return x_bio, x_econ, x_hist, x_diff, y_waste, y_cause


class CascadeDataset(Dataset):
    def __init__(self, x_bio, x_econ, x_hist, x_diff, y_waste, y_cause):
        self.x_bio = x_bio
        self.x_econ = x_econ
        self.x_hist = x_hist
        self.x_diff = x_diff
        self.y_waste = y_waste
        self.y_cause = y_cause

    def __len__(self):
        return len(self.x_bio)

    def __getitem__(self, idx):
        return {
            "x_bio": self.x_bio[idx],
            "x_econ": self.x_econ[idx],
            "x_hist": self.x_hist[idx],
            "x_diff": self.x_diff[idx],
            "waste_target": self.y_waste[idx],
            "cause_target": self.y_cause[idx],
        }


# ── Models ───────────────────────────────────────────────────────────

MODEL_REGISTRY = {
    # Ablation baselines
    "local_only": "LocalOnlyBaseline",
    "local_econ": "LocalEconBaseline",
    "symmetric_diff": "SymmetricDiffBaseline",
    "cascade_net": "CascadeNet",
}


def create_model(name, bio_dim, econ_dim, hist_dim, diff_dim):
    if name == "local_only":
        return _LocalOnly(bio_dim, hist_dim)
    elif name == "local_econ":
        return _LocalEcon(bio_dim, econ_dim, hist_dim)
    elif name == "symmetric_diff":
        return _SymmetricDiff(bio_dim, econ_dim, hist_dim, diff_dim)
    elif name == "cascade_net":
        from models.cascade_net import CascadeNet
        return CascadeNet(bio_dim, econ_dim, hist_dim, diff_dim)
    else:
        raise ValueError(f"Unknown model: {name}")


class _LocalOnly(nn.Module):
    """Ablation Row 1: biophysical features only."""
    def __init__(self, bio_dim, hist_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(bio_dim + hist_dim, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.3),
        )
        self.waste = nn.Linear(64, 1)
        self.cause = nn.Linear(64, 6)

    def forward(self, batch):
        h = self.net(torch.cat([batch["x_bio"], batch["x_hist"]], dim=-1))
        return {"waste_logits": self.waste(h), "cause_logits": self.cause(h),
                "disentangle_loss": torch.tensor(0.0, device=h.device)}


class _LocalEcon(nn.Module):
    """Ablation Row 2: bio + econ, no graph."""
    def __init__(self, bio_dim, econ_dim, hist_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(bio_dim + econ_dim + hist_dim, 128), nn.BatchNorm1d(128),
            nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.3),
        )
        self.waste = nn.Linear(64, 1)
        self.cause = nn.Linear(64, 6)

    def forward(self, batch):
        h = self.net(torch.cat([batch["x_bio"], batch["x_econ"], batch["x_hist"]], dim=-1))
        return {"waste_logits": self.waste(h), "cause_logits": self.cause(h),
                "disentangle_loss": torch.tensor(0.0, device=h.device)}


class _SymmetricDiff(nn.Module):
    """Ablation Row 3: symmetric diffusion (collapse pos/neg channels)."""
    def __init__(self, bio_dim, econ_dim, hist_dim, diff_dim):
        super().__init__()
        # Collapse pos+neg per hop → symmetric
        self.n_hops = diff_dim // 2
        input_dim = bio_dim + econ_dim + hist_dim + self.n_hops
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.3),
        )
        self.waste = nn.Linear(64, 1)
        self.cause = nn.Linear(64, 6)

    def forward(self, batch):
        x_diff = batch["x_diff"]
        # Collapse: sum pos+neg for each hop
        symmetric = []
        for k in range(self.n_hops):
            symmetric.append(x_diff[:, 2 * k] + x_diff[:, 2 * k + 1])
        x_sym = torch.stack(symmetric, dim=-1)

        h = self.net(torch.cat([batch["x_bio"], batch["x_econ"], batch["x_hist"], x_sym], dim=-1))
        return {"waste_logits": self.waste(h), "cause_logits": self.cause(h),
                "disentangle_loss": torch.tensor(0.0, device=h.device)}


# ── Training ─────────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, loss_fn, device, dis_lambda=0.1):
    model.train()
    total_loss, n = 0, 0
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        optimizer.zero_grad()
        out = model(batch)
        waste_loss = loss_fn(out["waste_logits"], batch["waste_target"])
        cause_loss = nn.CrossEntropyLoss()(out["cause_logits"], batch["cause_target"])
        dis_loss = out.get("disentangle_loss", torch.tensor(0.0, device=device))
        if isinstance(dis_loss, (int, float)):
            dis_loss = torch.tensor(dis_loss, device=device)
        loss = waste_loss + 0.3 * cause_loss + dis_lambda * dis_loss
        if torch.isnan(loss):
            continue
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
        n += 1
    return total_loss / max(n, 1)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_p, all_t = [], []
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(batch)
        all_p.append(torch.sigmoid(out["waste_logits"]).cpu())
        all_t.append(batch["waste_target"].cpu())
    yp = torch.cat(all_p).numpy().flatten()
    yt = torch.cat(all_t).numpy().flatten()
    from sklearn.metrics import roc_auc_score, f1_score, average_precision_score
    try:
        auc = roc_auc_score(yt, yp)
    except:
        auc = 0.5
    f1 = f1_score(yt, (yp >= 0.5).astype(int), zero_division=0)
    try:
        ap = average_precision_score(yt, yp)
    except:
        ap = 0.5
    return {"auc_roc": auc, "f1": f1, "auc_pr": ap}


def train_model(name, seed, train_loader, val_loader, test_loader,
                bio_dim, econ_dim, hist_dim, diff_dim, epochs, patience, lr, device):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = create_model(name, bio_dim, econ_dim, hist_dim, diff_dim).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"  {name} | seed={seed} | params={n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=50)

    all_y = torch.cat([b["waste_target"] for b in train_loader])
    waste_rate = all_y.mean().item()
    pos_weight = torch.tensor([(1 - waste_rate) / max(waste_rate, 0.01)]).to(device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_auc, best_epoch, wait, best_state = 0, 0, 0, None
    WARMUP = 5

    for epoch in range(epochs):
        t0 = time.time()
        dis_lambda = 0.0 if epoch < WARMUP else 0.1
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device, dis_lambda)
        scheduler.step()
        val = evaluate(model, val_loader, device)

        if val["auc_roc"] > best_auc:
            best_auc = val["auc_roc"]
            best_epoch = epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1

        if epoch % 10 == 0 or wait == 0:
            logger.info(f"    Ep {epoch:3d} | loss={train_loss:.4f} | val={val['auc_roc']:.4f} | "
                        f"best={best_auc:.4f}(ep{best_epoch}) | {time.time()-t0:.1f}s")

        if wait >= patience:
            logger.info(f"    Early stop ep {epoch}")
            break

    if best_state:
        model.load_state_dict(best_state)
    model.to(device)
    test = evaluate(model, test_loader, device)
    logger.info(f"  TEST: AUC={test['auc_roc']:.4f} | F1={test['f1']:.4f} | AP={test['auc_pr']:.4f}")

    CKPT.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state_dict": best_state or model.state_dict(),
        "model_name": name, "seed": seed, "n_params": n_params,
        "best_val_auc": best_auc, "best_epoch": best_epoch,
        "test_metrics": test,
    }, CKPT / f"cascade_{name}_seed{seed}.pt")

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "model": name, "seed": seed, "n_params": n_params,
        "best_val_auc": best_auc, "best_epoch": best_epoch,
        "test_auc_roc": test["auc_roc"], "test_f1": test["f1"], "test_auc_pr": test["auc_pr"],
    }


# ── Main ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="*", default=None)
    parser.add_argument("--seeds", nargs="*", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device}")

    features, labels, splits, stats, groups = load_data()
    bio_dim = len(groups["biophysical"])
    econ_dim = len(groups["economic"])
    hist_dim = len(groups["historical"])
    diff_dim = len(groups["diffusion"])
    logger.info(f"Data: {len(features):,} samples | bio={bio_dim} econ={econ_dim} "
                f"hist={hist_dim} diff={diff_dim}")

    x_bio, x_econ, x_hist, x_diff, y_waste, y_cause = \
        build_tensors(features, labels, stats, groups)

    def make_loader(indices, shuffle=False):
        idx = torch.tensor(indices)
        ds = CascadeDataset(x_bio[idx], x_econ[idx], x_hist[idx], x_diff[idx],
                            y_waste[idx], y_cause[idx])
        return DataLoader(ds, batch_size=args.batch_size, shuffle=shuffle,
                          drop_last=shuffle, num_workers=0)

    train_loader = make_loader(splits["train"], shuffle=True)
    val_loader = make_loader(splits["val"])
    test_loader = make_loader(splits["test"])
    logger.info(f"Train: {len(splits['train']):,} | Val: {len(splits['val']):,} | "
                f"Test: {len(splits['test']):,}")

    models = args.models or list(MODEL_REGISTRY.keys())
    seeds = args.seeds or [42, 123, 456, 789, 1024]
    all_results = []

    for model_name in models:
        logger.info(f"\n{'=' * 50}\nModel: {model_name}\n{'=' * 50}")
        for seed in seeds:
            ckpt_path = CKPT / f"cascade_{model_name}_seed{seed}.pt"
            if args.resume and ckpt_path.exists():
                try:
                    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
                    if "test_metrics" in ckpt and ckpt["test_metrics"].get("auc_roc", 0) > 0.5:
                        logger.info(f"  Resume: AUC={ckpt['test_metrics']['auc_roc']:.4f}, skip")
                        all_results.append({
                            "model": model_name, "seed": seed,
                            "test_auc_roc": ckpt["test_metrics"]["auc_roc"],
                            "test_f1": ckpt["test_metrics"]["f1"],
                            "test_auc_pr": ckpt["test_metrics"]["auc_pr"],
                            "n_params": ckpt.get("n_params", 0),
                            "best_val_auc": ckpt.get("best_val_auc", 0),
                            "best_epoch": ckpt.get("best_epoch", 0),
                        })
                        continue
                except:
                    pass

            try:
                r = train_model(model_name, seed, train_loader, val_loader, test_loader,
                                bio_dim, econ_dim, hist_dim, diff_dim,
                                args.epochs, args.patience, args.lr, device)
                all_results.append(r)
            except Exception as e:
                logger.error(f"FAILED: {model_name} seed={seed}: {e}")
                traceback.print_exc()
                all_results.append({
                    "model": model_name, "seed": seed,
                    "test_auc_roc": 0, "test_f1": 0, "test_auc_pr": 0, "n_params": 0,
                    "best_val_auc": 0, "best_epoch": 0,
                })

            RESULTS.mkdir(parents=True, exist_ok=True)
            with open(RESULTS / "cascade_results.json", "w") as f:
                json.dump(all_results, f, indent=2, default=str)

    # Summary
    import pandas as pd
    df = pd.DataFrame(all_results)
    df.to_csv(RESULTS / "cascade_results.csv", index=False)
    logger.info(f"\n{'=' * 60}\nFINAL RESULTS\n{'=' * 60}")
    for m in models:
        d = df[(df["model"] == m) & (df["test_auc_roc"] > 0)]
        if len(d):
            logger.info(f"{m:<20} AUC={d['test_auc_roc'].mean():.4f}"
                        f"±{d['test_auc_roc'].std():.4f}  F1={d['test_f1'].mean():.3f}")


if __name__ == "__main__":
    main()
