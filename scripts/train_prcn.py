#!/usr/bin/env python3
"""Train PRCN (Polarity-Routed Cascade Network) ablation suite.

5-row ablation — each row adds ONE novel component:
    R1: local_only        — bio + hist, no graph
    R2: local_econ        — bio + econ + hist, no graph
    R3: symmetric_diff    — symmetric diffusion (same graph, collapse +/-)
    R4: polarity_routed   — neg→commodity graph, pos→geo graph (novel routing)
    R5: prcn              — + decay signatures + vulnerability routing (full model)

Usage:
    python scripts/precompute_cascade.py                    # run once
    python scripts/train_prcn.py --epochs 100               # train all
    python scripts/train_prcn.py --models prcn --seeds 42 --epochs 5  # quick
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
    feat_path = PROC / "features_cascade.parquet"
    if not feat_path.exists():
        raise FileNotFoundError(f"{feat_path} not found. Run: python scripts/precompute_cascade.py")
    features = pd.read_parquet(feat_path)
    labels = pd.read_parquet(PROC / "labels_monthly.parquet")
    with open(PROC / "splits_monthly.json") as f: splits = json.load(f)
    with open(PROC / "stats_monthly.json") as f: stats = json.load(f)
    with open(PROC / "feature_groups_cascade.json") as f: groups = json.load(f)
    return features, labels, splits, stats, groups


def build_tensors(features, labels, stats, groups):
    def norm(df, cols):
        X = df[cols].values.astype(np.float32)
        for i, c in enumerate(cols):
            if c in stats:
                s = max(stats[c]["std"], 1e-8)
                X[:, i] = (X[:, i] - stats[c]["mean"]) / s
        return torch.from_numpy(np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0))

    x_bio = norm(features, groups["biophysical"])
    x_econ = norm(features, groups["economic"])
    x_hist = norm(features, groups["historical"])

    # Cascade features: normalize by own stats
    cascade_cols = groups["cascade"]
    x_raw = features[cascade_cols].values.astype(np.float32)
    for i in range(x_raw.shape[1]):
        m, s = np.nanmean(x_raw[:, i]), max(np.nanstd(x_raw[:, i]), 1e-8)
        x_raw[:, i] = (x_raw[:, i] - m) / s
    x_cascade = torch.from_numpy(np.nan_to_num(x_raw, nan=0.0))

    y_waste = torch.from_numpy(labels["waste"].values.astype(np.float32)).unsqueeze(1)
    y_cause = torch.from_numpy(labels["cause_idx"].values.astype(np.int64))

    logger.info(f"  bio={x_bio.shape} econ={x_econ.shape} cascade={x_cascade.shape}")
    return x_bio, x_econ, x_hist, x_cascade, y_waste, y_cause


class PRCNDataset(Dataset):
    def __init__(self, x_bio, x_econ, x_hist, x_cascade, y_waste, y_cause):
        self.x_bio = x_bio; self.x_econ = x_econ; self.x_hist = x_hist
        self.x_cascade = x_cascade; self.y_waste = y_waste; self.y_cause = y_cause

    def __len__(self): return len(self.x_bio)

    def __getitem__(self, idx):
        return {
            "x_bio": self.x_bio[idx], "x_econ": self.x_econ[idx],
            "x_hist": self.x_hist[idx], "x_cascade": self.x_cascade[idx],
            "waste_target": self.y_waste[idx], "cause_target": self.y_cause[idx],
        }


# ── Ablation Baselines ──────────────────────────────────────────────

class _LocalOnly(nn.Module):
    """R1: bio + hist only."""
    def __init__(self, bio_dim, hist_dim, **kw):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(bio_dim + hist_dim, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.3))
        self.waste = nn.Linear(64, 1); self.cause = nn.Linear(64, 6)

    def forward(self, batch):
        h = self.net(torch.cat([batch["x_bio"], batch["x_hist"]], dim=-1))
        return {"waste_logits": self.waste(h), "cause_logits": self.cause(h),
                "disentangle_loss": torch.tensor(0.0, device=h.device)}


class _LocalEcon(nn.Module):
    """R2: bio + econ + hist, no graph."""
    def __init__(self, bio_dim, econ_dim, hist_dim, **kw):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(bio_dim + econ_dim + hist_dim, 128), nn.BatchNorm1d(128),
            nn.ReLU(), nn.Dropout(0.3), nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.3))
        self.waste = nn.Linear(64, 1); self.cause = nn.Linear(64, 6)

    def forward(self, batch):
        h = self.net(torch.cat([batch["x_bio"], batch["x_econ"], batch["x_hist"]], dim=-1))
        return {"waste_logits": self.waste(h), "cause_logits": self.cause(h),
                "disentangle_loss": torch.tensor(0.0, device=h.device)}


class _CascadeDirect(nn.Module):
    """R3: bio + hist + cascade features (skip raw econ — graph-diffused is cleaner)."""
    def __init__(self, bio_dim, hist_dim, cascade_dim, **kw):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(bio_dim + hist_dim + cascade_dim, 128), nn.BatchNorm1d(128),
            nn.ReLU(), nn.Dropout(0.3), nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.3))
        self.waste = nn.Linear(64, 1); self.cause = nn.Linear(64, 6)

    def forward(self, batch):
        h = self.net(torch.cat([batch["x_bio"], batch["x_hist"], batch["x_cascade"]], dim=-1))
        return {"waste_logits": self.waste(h), "cause_logits": self.cause(h),
                "disentangle_loss": torch.tensor(0.0, device=h.device)}


class _SymmetricDiff(nn.Module):
    """R3: symmetric diffusion — collapse pos+neg per hop, single graph."""
    def __init__(self, bio_dim, econ_dim, hist_dim, cascade_dim, **kw):
        super().__init__()
        n_hops = 3  # 3 hop distances, 2 channels each (pos+neg) = first 6 features
        self.n_hops = n_hops
        self.net = nn.Sequential(
            nn.Linear(bio_dim + econ_dim + hist_dim + n_hops, 128), nn.BatchNorm1d(128),
            nn.ReLU(), nn.Dropout(0.3), nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.3))
        self.waste = nn.Linear(64, 1); self.cause = nn.Linear(64, 6)

    def forward(self, batch):
        x = batch["x_cascade"][:, :6]  # raw diffusion (first 6)
        sym = []
        for k in range(self.n_hops):
            sym.append(x[:, 2*k] + x[:, 2*k+1])  # neg + pos per hop
        x_sym = torch.stack(sym, dim=-1)
        h = self.net(torch.cat([batch["x_bio"], batch["x_econ"], batch["x_hist"], x_sym], dim=-1))
        return {"waste_logits": self.waste(h), "cause_logits": self.cause(h),
                "disentangle_loss": torch.tensor(0.0, device=h.device)}


class _PolarityRouted(nn.Module):
    """R4: polarity-routed diffusion (novel routing) but no decay/vulnerability."""
    def __init__(self, bio_dim, econ_dim, hist_dim, cascade_dim, latent_dim=64, **kw):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(bio_dim + econ_dim + hist_dim + 6, 128), nn.BatchNorm1d(128),
            nn.ReLU(), nn.Dropout(0.3), nn.Linear(128, latent_dim), nn.ReLU(), nn.Dropout(0.3))
        self.waste = nn.Linear(latent_dim, 1); self.cause = nn.Linear(latent_dim, 6)

    def forward(self, batch):
        x_diff = batch["x_cascade"][:, :6]  # 6 polarity-routed features (no decay)
        h = self.net(torch.cat([batch["x_bio"], batch["x_econ"], batch["x_hist"], x_diff], dim=-1))
        return {"waste_logits": self.waste(h), "cause_logits": self.cause(h),
                "disentangle_loss": torch.tensor(0.0, device=h.device)}


# ── Model Registry ──────────────────────────────────────────────────

MODEL_REGISTRY = {
    "local_only": "R1",
    "local_econ": "R2",
    "cascade_direct": "R3-Cascade",
    "symmetric_diff": "R4-Symmetric",
    "polarity_routed": "R5-PolarityRouted",
    "prcn": "R6-PRCN",
    "temporal_prcn": "R7-Temporal",
}

TEMPORAL_WINDOW = 6  # months of lookback


class SequenceDataset(Dataset):
    """Returns (T, D) sequences per sample for temporal models.
    Uses FULL tensors + global indices to avoid subset/global mismatch."""
    def __init__(self, x_bio, x_econ, x_hist, x_cascade, y_waste, y_cause, seq_map, indices):
        self.x_bio = x_bio; self.x_econ = x_econ; self.x_hist = x_hist
        self.x_cascade = x_cascade; self.y_waste = y_waste; self.y_cause = y_cause
        self.seq_map = seq_map  # (N_total, W) — global indices, -1 for padding
        self.indices = indices  # global indices for this split

    def __len__(self): return len(self.indices)

    def __getitem__(self, idx):
        gi = self.indices[idx]  # global index
        seq = self.seq_map[gi]  # (W,) — global indices
        W = len(seq)
        bio_seq = torch.zeros(W, self.x_bio.size(1))
        econ_seq = torch.zeros(W, self.x_econ.size(1))
        casc_seq = torch.zeros(W, self.x_cascade.size(1))
        for t in range(W):
            if seq[t] >= 0:
                bio_seq[t] = self.x_bio[seq[t]]
                econ_seq[t] = self.x_econ[seq[t]]
                casc_seq[t] = self.x_cascade[seq[t]]
        return {
            "x_bio": bio_seq, "x_econ": econ_seq, "x_hist": self.x_hist[gi],
            "x_cascade": casc_seq, "waste_target": self.y_waste[gi],
            "cause_target": self.y_cause[gi],
        }


def build_sequence_map(features_df, window=6):
    """Build (N, W) index array: for each sample, its temporal lookback window."""
    import pandas as pd
    df = features_df[["fips", "commodity", "year", "month"]].copy()
    df["_orig"] = np.arange(len(df))
    df = df.sort_values(["fips", "commodity", "year", "month"]).reset_index(drop=True)
    key = df["fips"].astype(str) + "_" + df["commodity"]
    is_new = (key != key.shift()).values
    orig = df["_orig"].values

    seq_map = np.full((len(features_df), window), -1, dtype=np.int64)
    buf = []
    for i in range(len(df)):
        if is_new[i]:
            buf = []
        buf.append(orig[i])
        start = max(0, len(buf) - window)
        win = buf[start:]
        offset = window - len(win)
        seq_map[orig[i], offset:] = win
    return torch.from_numpy(seq_map)


def create_model(name, bio_dim, econ_dim, hist_dim, cascade_dim):
    if name == "local_only":
        return _LocalOnly(bio_dim, hist_dim)
    elif name == "local_econ":
        return _LocalEcon(bio_dim, econ_dim, hist_dim)
    elif name == "cascade_direct":
        return _CascadeDirect(bio_dim, hist_dim, cascade_dim)
    elif name == "symmetric_diff":
        return _SymmetricDiff(bio_dim, econ_dim, hist_dim, cascade_dim)
    elif name == "polarity_routed":
        return _PolarityRouted(bio_dim, econ_dim, hist_dim, cascade_dim)
    elif name == "prcn":
        from models.prcn import PRCN
        return PRCN(bio_dim, econ_dim, hist_dim, cascade_dim)
    elif name == "temporal_prcn":
        from models.prcn import TemporalPRCN
        return TemporalPRCN(bio_dim, econ_dim, hist_dim, cascade_dim)
    else:
        raise ValueError(f"Unknown model: {name}")


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
        if torch.isnan(loss): continue
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item(); n += 1
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
    try: auc = roc_auc_score(yt, yp)
    except: auc = 0.5
    f1 = f1_score(yt, (yp >= 0.5).astype(int), zero_division=0)
    try: ap = average_precision_score(yt, yp)
    except: ap = 0.5
    return {"auc_roc": auc, "f1": f1, "auc_pr": ap}


def train_model(name, seed, train_loader, val_loader, test_loader,
                bio_dim, econ_dim, hist_dim, cascade_dim, epochs, patience, lr, device):
    torch.manual_seed(seed); np.random.seed(seed)
    model = create_model(name, bio_dim, econ_dim, hist_dim, cascade_dim).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"  {name} | seed={seed} | params={n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=50)

    all_y = torch.cat([b["waste_target"] for b in train_loader])
    waste_rate = all_y.mean().item()
    pos_weight = torch.tensor([(1 - waste_rate) / max(waste_rate, 0.01)]).to(device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_auc, best_epoch, wait, best_state = 0, 0, 0, None

    for epoch in range(epochs):
        t0 = time.time()
        dis_lambda = 0.0 if epoch < 5 else 0.1
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device, dis_lambda)
        scheduler.step()
        val = evaluate(model, val_loader, device)

        if val["auc_roc"] > best_auc:
            best_auc = val["auc_roc"]; best_epoch = epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}; wait = 0
        else:
            wait += 1

        if epoch % 10 == 0 or wait == 0:
            logger.info(f"    Ep {epoch:3d} | loss={train_loss:.4f} | val={val['auc_roc']:.4f} | "
                        f"best={best_auc:.4f}(ep{best_epoch}) | {time.time()-t0:.1f}s")
        if wait >= patience:
            logger.info(f"    Early stop ep {epoch}"); break

    if best_state: model.load_state_dict(best_state)
    model.to(device)
    test = evaluate(model, test_loader, device)
    logger.info(f"  TEST: AUC={test['auc_roc']:.4f} | F1={test['f1']:.4f} | AP={test['auc_pr']:.4f}")

    CKPT.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": best_state or model.state_dict(),
                "model_name": name, "seed": seed, "n_params": n_params,
                "best_val_auc": best_auc, "best_epoch": best_epoch,
                "test_metrics": test}, CKPT / f"prcn_{name}{args.output_suffix}_seed{seed}.pt")
    del model
    if torch.cuda.is_available(): torch.cuda.empty_cache()

    return {"model": name, "seed": seed, "n_params": n_params,
            "best_val_auc": best_auc, "best_epoch": best_epoch,
            "test_auc_roc": test["auc_roc"], "test_f1": test["f1"], "test_auc_pr": test["auc_pr"]}


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
    parser.add_argument("--random-split", action="store_true",
                        help="Use 70/15/15 random split instead of temporal split")
    parser.add_argument("--output-suffix", type=str, default="",
                        help="Suffix for output files (e.g. '_random')")
    args = parser.parse_args()

    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device}")

    features, labels, splits, stats, groups = load_data()
    bio_dim = len(groups["biophysical"])
    econ_dim = len(groups["economic"])
    hist_dim = len(groups["historical"])
    cascade_dim = len(groups["cascade"])
    logger.info(f"Data: {len(features):,} | bio={bio_dim} econ={econ_dim} hist={hist_dim} cascade={cascade_dim}")

    x_bio, x_econ, x_hist, x_cascade, y_waste, y_cause = build_tensors(features, labels, stats, groups)

    # Override splits if random split requested
    if args.random_split:
        logger.info("Using RANDOM 70/15/15 split (not temporal)")
        N = len(features)
        perm = np.random.RandomState(42).permutation(N)
        n_train = int(0.7 * N)
        n_val = int(0.15 * N)
        splits = {
            "train": perm[:n_train].tolist(),
            "val": perm[n_train:n_train + n_val].tolist(),
            "test": perm[n_train + n_val:].tolist(),
        }
        logger.info(f"Random split: train={n_train:,} val={n_val:,} test={N-n_train-n_val:,}")

    # Standard loaders (snapshot models)
    def make_loader(indices, shuffle=False):
        idx = torch.tensor(indices)
        ds = PRCNDataset(x_bio[idx], x_econ[idx], x_hist[idx], x_cascade[idx], y_waste[idx], y_cause[idx])
        return DataLoader(ds, batch_size=args.batch_size, shuffle=shuffle, drop_last=shuffle, num_workers=0)

    # Temporal loaders (sequence models)
    seq_map = None
    def make_seq_loader(indices, shuffle=False):
        nonlocal seq_map
        if seq_map is None:
            logger.info("Building temporal sequence map...")
            seq_map = build_sequence_map(features, TEMPORAL_WINDOW)
            logger.info(f"Sequence map: {seq_map.shape}")
        ds = SequenceDataset(x_bio, x_econ, x_hist, x_cascade,
                             y_waste, y_cause, seq_map, indices)
        return DataLoader(ds, batch_size=args.batch_size, shuffle=shuffle, drop_last=shuffle, num_workers=0)

    train_loader = make_loader(splits["train"], shuffle=True)
    val_loader = make_loader(splits["val"])
    test_loader = make_loader(splits["test"])

    models = args.models or list(MODEL_REGISTRY.keys())
    seeds = args.seeds or [42, 123, 456, 789, 1024]
    all_results = []

    for model_name in models:
        # Switch to temporal loaders for temporal models
        if model_name == "temporal_prcn":
            train_loader = make_seq_loader(splits["train"], shuffle=True)
            val_loader = make_seq_loader(splits["val"])
            test_loader = make_seq_loader(splits["test"])
        elif model_name != "temporal_prcn" and seq_map is not None:
            # Switch back to standard loaders
            train_loader = make_loader(splits["train"], shuffle=True)
            val_loader = make_loader(splits["val"])
            test_loader = make_loader(splits["test"])

        logger.info(f"\n{'='*50}\n{MODEL_REGISTRY.get(model_name, '??')}: {model_name}\n{'='*50}")
        for seed in seeds:
            ckpt_path = CKPT / f"prcn_{model_name}_seed{seed}.pt"
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
                            "best_epoch": ckpt.get("best_epoch", 0)})
                        continue
                except: pass

            try:
                r = train_model(model_name, seed, train_loader, val_loader, test_loader,
                                bio_dim, econ_dim, hist_dim, cascade_dim,
                                args.epochs, args.patience, args.lr, device)
                all_results.append(r)
            except Exception as e:
                logger.error(f"FAILED: {model_name} seed={seed}: {e}")
                traceback.print_exc()
                all_results.append({"model": model_name, "seed": seed,
                    "test_auc_roc": 0, "test_f1": 0, "test_auc_pr": 0, "n_params": 0,
                    "best_val_auc": 0, "best_epoch": 0})

            RESULTS.mkdir(parents=True, exist_ok=True)
            with open(RESULTS / f"prcn_results{args.output_suffix}.json", "w") as f:
                json.dump(all_results, f, indent=2, default=str)

    import pandas as pd
    df = pd.DataFrame(all_results)
    df.to_csv(RESULTS / f"prcn_results{args.output_suffix}.csv", index=False)
    logger.info(f"\n{'='*60}\nFINAL RESULTS\n{'='*60}")
    for m in models:
        d = df[(df["model"]==m) & (df["test_auc_roc"]>0)]
        if len(d):
            logger.info(f"{m:<20} AUC={d['test_auc_roc'].mean():.4f}±{d['test_auc_roc'].std():.4f}  "
                        f"F1={d['test_f1'].mean():.3f}")


if __name__ == "__main__":
    main()
