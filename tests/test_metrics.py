"""Tests for evaluation metrics and statistical tests."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pytest
import numpy as np
from evaluation.metrics import (
    compute_binary_metrics,
    compute_cause_metrics,
    expected_calibration_error,
    aggregate_seed_metrics,
)
from evaluation.statistical_tests import (
    delong_test,
    mcnemar_test,
    bootstrap_ci,
    paired_ttest_across_seeds,
)


class TestBinaryMetrics:
    def test_perfect_predictions(self):
        y_true = np.array([0, 0, 1, 1, 1])
        y_prob = np.array([0.1, 0.2, 0.8, 0.9, 0.95])
        metrics = compute_binary_metrics(y_true, y_prob)

        assert metrics["auc_roc"] == 1.0
        assert metrics["f1_binary"] == 1.0
        assert metrics["precision"] == 1.0
        assert metrics["recall"] == 1.0

    def test_random_predictions(self):
        np.random.seed(42)
        y_true = np.random.randint(0, 2, 1000)
        y_prob = np.random.rand(1000)
        metrics = compute_binary_metrics(y_true, y_prob)

        # Random predictions should give AUC ~0.5
        assert 0.4 < metrics["auc_roc"] < 0.6

    def test_all_metrics_present(self):
        y_true = np.array([0, 1, 0, 1])
        y_prob = np.array([0.3, 0.7, 0.4, 0.8])
        metrics = compute_binary_metrics(y_true, y_prob)

        expected_keys = [
            "auc_roc", "auc_pr", "f1_binary", "precision",
            "recall", "brier_score", "log_loss", "ece",
        ]
        for key in expected_keys:
            assert key in metrics


class TestECE:
    def test_perfectly_calibrated(self):
        # Perfectly calibrated: predicted prob = true frequency
        y_true = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        y_prob = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9, 1.0])
        ece = expected_calibration_error(y_true, y_prob, n_bins=5)
        # Should be relatively low
        assert ece < 0.3

    def test_overconfident(self):
        # All predictions at 0.9 but only 50% are positive
        y_true = np.array([0, 1, 0, 1, 0, 1])
        y_prob = np.array([0.9, 0.9, 0.9, 0.9, 0.9, 0.9])
        ece = expected_calibration_error(y_true, y_prob, n_bins=5)
        assert ece > 0.3  # Should be high


class TestCauseMetrics:
    def test_perfect_cause_prediction(self):
        y_true = np.array([0, 1, 2, 3, 4, 5])
        y_pred = np.array([0, 1, 2, 3, 4, 5])
        metrics = compute_cause_metrics(y_true, y_pred)
        assert metrics["cause_accuracy"] == 1.0
        assert metrics["f1_macro_cause"] == 1.0


class TestDeLongTest:
    def test_identical_models(self):
        y_true = np.random.randint(0, 2, 500)
        y_pred = np.random.rand(500)
        result = delong_test(y_true, y_pred, y_pred)
        # Same predictions → not significant
        assert result["p_value"] > 0.05
        assert abs(result["auc_diff"]) < 0.001

    def test_different_models(self):
        np.random.seed(42)
        y_true = np.random.randint(0, 2, 500)
        # Good model
        y_pred_good = y_true.astype(float) + np.random.randn(500) * 0.3
        y_pred_good = np.clip(y_pred_good, 0, 1)
        # Bad model
        y_pred_bad = np.random.rand(500)

        result = delong_test(y_true, y_pred_good, y_pred_bad)
        assert result["auc_diff"] > 0  # Good model has higher AUC


class TestMcNemar:
    def test_returns_valid_result(self):
        y_true = np.random.randint(0, 2, 200)
        y_pred1 = np.random.rand(200)
        y_pred2 = np.random.rand(200)
        result = mcnemar_test(y_true, y_pred1, y_pred2)
        assert "chi2_statistic" in result
        assert "p_value" in result


class TestBootstrapCI:
    def test_better_model_positive_ci(self):
        np.random.seed(42)
        y_true = np.random.randint(0, 2, 200)
        y_pred_good = y_true.astype(float) * 0.8 + np.random.rand(200) * 0.2
        y_pred_bad = np.random.rand(200)

        from sklearn.metrics import roc_auc_score
        result = bootstrap_ci(
            y_true, y_pred_good, y_pred_bad,
            roc_auc_score, n_bootstrap=1000, seed=42,
        )
        assert result["mean_diff"] > 0


class TestAggregation:
    def test_aggregate_seed_metrics(self):
        seed_metrics = [
            {"auc_roc": 0.85, "f1_binary": 0.70},
            {"auc_roc": 0.87, "f1_binary": 0.72},
            {"auc_roc": 0.84, "f1_binary": 0.69},
        ]
        agg = aggregate_seed_metrics(seed_metrics)
        assert "auc_roc" in agg
        assert abs(agg["auc_roc"]["mean"] - 0.8533) < 0.01
        assert "formatted" in agg["auc_roc"]
