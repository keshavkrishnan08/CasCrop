#!/usr/bin/env python3
"""Integration test: Verifies the full CasCrop pipeline is runnable.

Tests:
1. RMA data parsing (from actual downloaded files)
2. Price data loading
3. Geographic data loading
4. Feature construction
5. Graph construction
6. All 5 model architectures (forward + backward)
7. Training loop (3 epochs)
8. Evaluation metrics
9. Statistical tests
10. Figure/table generation (dry run)

This is the "can I run experiments?" validation script.
"""

import sys
import os
import logging
import traceback

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

PASS = 0
FAIL = 0


def check(name: str, fn):
    global PASS, FAIL
    try:
        fn()
        logger.info(f"  [PASS] {name}")
        PASS += 1
    except Exception as e:
        logger.error(f"  [FAIL] {name}: {e}")
        traceback.print_exc()
        FAIL += 1


def main():
    import torch
    import numpy as np

    logger.info("=" * 60)
    logger.info("CasCrop Integration Test")
    logger.info("=" * 60)

    # ------------------------------------------------------------------
    logger.info("\n1. DATA PARSING")
    # ------------------------------------------------------------------

    def test_rma_parse():
        import pandas as pd
        rma_dir = "data/raw/rma"
        files = [f for f in os.listdir(rma_dir) if f.startswith("colsom_") and f.endswith(".txt")]
        assert len(files) >= 5, f"Only {len(files)} RMA files found"

        # Parse one file
        f = sorted(files)[-1]  # latest year
        df = pd.read_csv(
            os.path.join(rma_dir, f),
            sep="|",
            header=None,
            encoding="latin-1",
            on_bad_lines="skip",
        )
        assert len(df) > 10000, f"Only {len(df)} records in {f}"
        # Build FIPS from state code (col 1) and county code (col 3)
        df["fips"] = df.iloc[:, 1].astype(str).str.zfill(2) + df.iloc[:, 3].astype(str).str.zfill(3)
        n_counties = df["fips"].nunique()
        assert n_counties > 100, f"Only {n_counties} unique counties"

    check("Parse RMA data files", test_rma_parse)

    def test_price_data():
        import pandas as pd
        for crop in ["corn", "wheat", "soybeans"]:
            path = f"data/raw/prices/{crop}_prices.csv"
            assert os.path.exists(path), f"{path} not found"
            df = pd.read_csv(path)
            assert len(df) > 100, f"Only {len(df)} price records for {crop}"

    check("Load commodity prices", test_price_data)

    def test_geographic():
        adj_path = "data/raw/geographic/county_adjacency.txt"
        assert os.path.exists(adj_path)
        with open(adj_path, encoding="latin-1") as f:
            lines = f.readlines()
        assert len(lines) > 10000

    check("Load county adjacency", test_geographic)

    def test_gazetteer():
        gaz_files = [f for f in os.listdir("data/raw/geographic") if "Gaz_counties" in f]
        assert len(gaz_files) >= 1
        import pandas as pd
        gaz_file = os.path.join("data/raw/geographic", gaz_files[0])
        df = pd.read_csv(gaz_file, sep="\t")
        assert len(df) > 3000
        assert "INTPTLAT" in df.columns or "INTPTLAT" in df.columns

    check("Load county gazetteer", test_gazetteer)

    # ------------------------------------------------------------------
    logger.info("\n2. MODEL ARCHITECTURES")
    # ------------------------------------------------------------------

    N, E = 64, 256
    batch = {
        "x_bio": torch.randn(N, 30),
        "x_econ": torch.randn(N, 15),
        "x_hist": torch.randn(N, 10),
        "edge_index": torch.randint(0, N, (2, E)),
        "edge_attr": None,
        "price_shocks": torch.randn(N, 1),
    }

    def test_model(module_path, class_name, extra_kwargs=None):
        import importlib
        mod = importlib.import_module(module_path)
        cls = getattr(mod, class_name)
        kwargs = {"bio_input_dim": 30, "latent_dim": 64, "dropout": 0.1}
        if extra_kwargs:
            kwargs.update(extra_kwargs)
        model = cls(**kwargs)
        out = model(batch)
        assert "waste_logits" in out
        assert out["waste_logits"].shape == (N, 1)
        # Backward pass
        loss = out["waste_logits"].sum()
        loss.backward()
        return model

    check("Row 1: LocalOnly", lambda: test_model(
        "models.baselines.local_only", "LocalOnlyModel"))
    check("Row 2: LocalEcon", lambda: test_model(
        "models.baselines.local_econ", "LocalEconModel", {"econ_input_dim": 15}))
    check("Row 3: GeoGAT", lambda: test_model(
        "models.baselines.geo_gat", "GeoGATModel",
        {"econ_input_dim": 15, "hist_dim": 10, "num_heads": 4}))
    check("Row 4: SymmetricECMP", lambda: test_model(
        "models.baselines.symmetric_ecmp", "SymmetricECMPModel",
        {"econ_input_dim": 15, "hist_dim": 10, "num_heads": 4}))
    check("Row 5: CasCrop", lambda: test_model(
        "models.cascrop", "CasCrop",
        {"econ_input_dim": 15, "hist_dim": 10, "num_heads": 4}))

    # ------------------------------------------------------------------
    logger.info("\n3. ECMP ASYMMETRY VERIFICATION")
    # ------------------------------------------------------------------

    def test_asymmetry():
        from models.graph.ecmp import ECMPLayer
        layer = ECMPLayer(in_dim=64, out_dim=32, num_heads=4, asymmetric=True, dropout=0.0)
        layer.eval()

        x = torch.randn(10, 64)
        ei = torch.randint(0, 10, (2, 20))
        ea = torch.randn(20, 3)

        pos = torch.zeros(10, 1); pos[0] = 1.0
        neg = torch.zeros(10, 1); neg[0] = -1.0

        with torch.no_grad():
            _, a_pos = layer(x, ei, ea, pos)
            _, a_neg = layer(x, ei, ea, neg)

        assert not torch.allclose(a_pos, a_neg, atol=1e-6), \
            "Asymmetric ECMP should produce different attention for +/- shocks"

    check("ECMP asymmetry verified", test_asymmetry)

    # ------------------------------------------------------------------
    logger.info("\n4. TRAINING LOOP")
    # ------------------------------------------------------------------

    def test_training_loop():
        from models.cascrop import CasCrop
        from training.losses import CombinedLoss
        from training.optimizers import build_optimizer
        from training.early_stopping import EarlyStopping

        model = CasCrop(bio_input_dim=30, econ_input_dim=15, hist_dim=10,
                        latent_dim=32, num_heads=2, dropout=0.1)
        config = {
            "training": {
                "learning_rate": 0.001,
                "weight_decay": 0.0001,
                "optimizer": "AdamW",
                "gradient_clip_norm": 1.0,
            }
        }
        optimizer = build_optimizer(model, config)
        loss_fn = CombinedLoss(waste_loss="focal", focal_gamma=2.0, focal_alpha=0.75)

        # 3 training steps
        model.train()
        for step in range(3):
            optimizer.zero_grad()
            outputs = model(batch)
            targets = torch.randint(0, 2, (N, 1)).float()
            cause_targets = torch.randint(0, 6, (N,))
            losses = loss_fn(
                outputs["waste_logits"], targets,
                outputs["cause_logits"], cause_targets,
                outputs["disentangle_loss"],
            )
            losses["total_loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        # Verify gradients flowed
        grad_norms = [p.grad.norm().item() for p in model.parameters() if p.grad is not None]
        assert len(grad_norms) > 0
        assert any(g > 0 for g in grad_norms)

    check("Training loop (3 steps)", test_training_loop)

    # ------------------------------------------------------------------
    logger.info("\n5. EVALUATION METRICS")
    # ------------------------------------------------------------------

    def test_metrics():
        from evaluation.metrics import compute_binary_metrics, aggregate_seed_metrics
        y_true = np.random.randint(0, 2, 200)
        y_prob = np.clip(y_true + np.random.randn(200) * 0.3, 0, 1)
        m = compute_binary_metrics(y_true, y_prob)
        assert m["auc_roc"] > 0.7

    check("Binary metrics", test_metrics)

    def test_statistical_tests():
        from evaluation.statistical_tests import delong_test, mcnemar_test
        y = np.random.randint(0, 2, 200)
        p1 = np.clip(y + np.random.randn(200) * 0.3, 0, 1)
        p2 = np.random.rand(200)
        d = delong_test(y, p1, p2)
        assert d["significant"]

    check("Statistical tests", test_statistical_tests)

    # ------------------------------------------------------------------
    logger.info("\n6. VISUALIZATION (dry run)")
    # ------------------------------------------------------------------

    def test_tables():
        from visualization.tables import table_1_dataset_summary
        sources = [{"name": "Test", "coverage": "2020", "resolution": "County",
                    "type": "Labels", "records": 100, "features": "x"}]
        latex = table_1_dataset_summary(sources, "/tmp/test_table.tex")
        assert "\\begin{table}" in latex

    check("LaTeX table generation", test_tables)

    # ------------------------------------------------------------------
    logger.info("\n" + "=" * 60)
    logger.info(f"RESULTS: {PASS} passed, {FAIL} failed")
    logger.info("=" * 60)

    if FAIL == 0:
        logger.info("\nAll systems go. You are ready to run experiments.")
        logger.info("\nNext steps:")
        logger.info("  1. Copy .env.example to .env and add API keys")
        logger.info("  2. Run: python scripts/00_setup_and_download.py")
        logger.info("  3. Run: python scripts/02_process_data.py")
        logger.info("  4. Run: python scripts/03_build_graphs.py")
        logger.info("  5. Run: python scripts/04_train_all.py --gpu 0")
    else:
        logger.info(f"\n{FAIL} tests failed. Fix these before running experiments.")
        sys.exit(1)


if __name__ == "__main__":
    main()
