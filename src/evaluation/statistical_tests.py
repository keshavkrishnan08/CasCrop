"""Statistical significance tests for model comparison.

Implements DeLong test for AUC comparison, McNemar's test,
bootstrap confidence intervals, paired t-test, and Wilcoxon signed-rank.
"""

import numpy as np
from scipy import stats
from statsmodels.stats.contingency_tables import mcnemar
from typing import Optional
import logging

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# DeLong Test for AUC Comparison
# Based on: DeLong et al. (1988), "Comparing the Areas under Two or More
# Correlated Receiver Operating Characteristic Curves"
# ---------------------------------------------------------------------------


def _compute_midrank(x: np.ndarray) -> np.ndarray:
    """Compute midranks for DeLong test."""
    j = np.argsort(x)
    z = x[j]
    n = len(x)
    rank = np.zeros(n)
    i = 0
    while i < n:
        j_start = i
        while i < n - 1 and z[i] == z[i + 1]:
            i += 1
        midrank = (j_start + i) / 2.0
        for k in range(j_start, i + 1):
            rank[j[k]] = midrank
        i += 1
    return rank


def _fast_delong(predictions_sorted_transposed: np.ndarray, label_1_count: int):
    """Fast DeLong AUC variance computation.

    Args:
        predictions_sorted_transposed: [2, n] array of sorted predictions
        label_1_count: number of positive labels
    """
    m = label_1_count
    n = predictions_sorted_transposed.shape[1] - m
    positive_examples = predictions_sorted_transposed[:, :m]
    negative_examples = predictions_sorted_transposed[:, m:]
    k = predictions_sorted_transposed.shape[0]

    aucs = np.zeros(k)
    score = np.zeros((k, m + n))
    for j in range(k):
        score[j] = _compute_midrank(predictions_sorted_transposed[j])
        aucs[j] = np.sum(score[j, :m]) / (m * n) - (m + 1.0) / (2.0 * n)

    v01 = (1.0 / n) * (score[:, :m] - np.arange(1, m + 1) / n)
    v10 = 1.0 / m - (1.0 / m) * score[:, m:]

    sx = np.cov(v01) if v01.shape[1] > 1 else np.var(v01, ddof=1).reshape(k, k)
    sy = np.cov(v10) if v10.shape[1] > 1 else np.var(v10, ddof=1).reshape(k, k)
    s = sx / m + sy / n

    return aucs, s


def delong_test(
    y_true: np.ndarray,
    y_pred_1: np.ndarray,
    y_pred_2: np.ndarray,
) -> dict:
    """DeLong test comparing two AUC-ROC values.

    Tests H0: AUC_1 = AUC_2 against H1: AUC_1 != AUC_2.

    Args:
        y_true: ground truth binary labels
        y_pred_1: predicted probabilities from model 1
        y_pred_2: predicted probabilities from model 2

    Returns:
        dict with z_statistic, p_value, auc_1, auc_2, auc_diff, ci_lower, ci_upper
    """
    order = np.argsort(-y_true)  # positive labels first
    label_1_count = int(np.sum(y_true))

    predictions_sorted = np.vstack([y_pred_1, y_pred_2])[:, order]
    aucs, cov_matrix = _fast_delong(predictions_sorted, label_1_count)

    auc_diff = aucs[0] - aucs[1]
    var_diff = cov_matrix[0, 0] + cov_matrix[1, 1] - 2 * cov_matrix[0, 1]
    se = np.sqrt(max(var_diff, 1e-10))
    z = auc_diff / se
    p_value = 2 * stats.norm.sf(abs(z))

    ci_lower = auc_diff - 1.96 * se
    ci_upper = auc_diff + 1.96 * se

    return {
        "z_statistic": z,
        "p_value": p_value,
        "auc_1": aucs[0],
        "auc_2": aucs[1],
        "auc_diff": auc_diff,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "significant": p_value < 0.05,
    }


# ---------------------------------------------------------------------------
# McNemar's Test
# ---------------------------------------------------------------------------


def mcnemar_test(
    y_true: np.ndarray,
    y_pred_1: np.ndarray,
    y_pred_2: np.ndarray,
    threshold: float = 0.5,
) -> dict:
    """McNemar's test comparing two classifiers' errors.

    Tests whether two models make different types of errors.

    Args:
        y_true: ground truth binary labels
        y_pred_1: predicted probabilities from model 1
        y_pred_2: predicted probabilities from model 2
        threshold: classification threshold

    Returns:
        dict with chi2_statistic, p_value, contingency_table
    """
    pred_1 = (y_pred_1 >= threshold).astype(int)
    pred_2 = (y_pred_2 >= threshold).astype(int)

    correct_1 = (pred_1 == y_true).astype(int)
    correct_2 = (pred_2 == y_true).astype(int)

    # Contingency table:
    # [both correct, only 1 correct]
    # [only 2 correct, both wrong]
    n_both_correct = np.sum((correct_1 == 1) & (correct_2 == 1))
    n_only_1_correct = np.sum((correct_1 == 1) & (correct_2 == 0))
    n_only_2_correct = np.sum((correct_1 == 0) & (correct_2 == 1))
    n_both_wrong = np.sum((correct_1 == 0) & (correct_2 == 0))

    table = np.array(
        [[n_both_correct, n_only_1_correct], [n_only_2_correct, n_both_wrong]]
    )

    result = mcnemar(table, exact=False, correction=True)

    return {
        "chi2_statistic": result.statistic,
        "p_value": result.pvalue,
        "contingency_table": table.tolist(),
        "significant": result.pvalue < 0.05,
    }


# ---------------------------------------------------------------------------
# Bootstrap Confidence Intervals
# ---------------------------------------------------------------------------


def bootstrap_ci(
    y_true: np.ndarray,
    y_pred_model1: np.ndarray,
    y_pred_model2: np.ndarray,
    metric_fn: callable,
    n_bootstrap: int = 10000,
    confidence: float = 0.95,
    seed: int = 42,
) -> dict:
    """Compute bootstrap confidence interval for metric difference.

    Args:
        y_true: ground truth labels
        y_pred_model1: predictions from model 1 (CasCrop)
        y_pred_model2: predictions from model 2 (baseline)
        metric_fn: function(y_true, y_pred) -> float
        n_bootstrap: number of bootstrap samples
        confidence: confidence level (default 0.95)
        seed: random seed

    Returns:
        dict with ci_lower, ci_upper, mean_diff, p_value
    """
    rng = np.random.RandomState(seed)
    n = len(y_true)
    diffs = np.zeros(n_bootstrap)

    for i in range(n_bootstrap):
        idx = rng.choice(n, n, replace=True)
        try:
            m1 = metric_fn(y_true[idx], y_pred_model1[idx])
            m2 = metric_fn(y_true[idx], y_pred_model2[idx])
            diffs[i] = m1 - m2
        except ValueError:
            diffs[i] = np.nan

    diffs = diffs[~np.isnan(diffs)]
    alpha = 1 - confidence
    ci_lower = np.percentile(diffs, 100 * alpha / 2)
    ci_upper = np.percentile(diffs, 100 * (1 - alpha / 2))
    p_value = np.mean(diffs <= 0)

    return {
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "mean_diff": np.mean(diffs),
        "p_value": p_value,
        "significant": (ci_lower > 0) or (ci_upper < 0),
    }


# ---------------------------------------------------------------------------
# Paired t-test and Wilcoxon across seeds
# ---------------------------------------------------------------------------


def paired_ttest_across_seeds(
    metrics_model1: list[float],
    metrics_model2: list[float],
) -> dict:
    """Paired t-test comparing metric values across seeds.

    Args:
        metrics_model1: metric values from model 1 for each seed
        metrics_model2: metric values from model 2 for each seed

    Returns:
        dict with t_statistic, p_value
    """
    t_stat, p_value = stats.ttest_rel(metrics_model1, metrics_model2)
    return {
        "t_statistic": t_stat,
        "p_value": p_value,
        "significant": p_value < 0.05,
        "mean_diff": np.mean(np.array(metrics_model1) - np.array(metrics_model2)),
    }


def wilcoxon_test_across_seeds(
    metrics_model1: list[float],
    metrics_model2: list[float],
) -> dict:
    """Wilcoxon signed-rank test (non-parametric alternative to paired t-test).

    Use when normality assumption may not hold with only 5 seeds.

    Args:
        metrics_model1: metric values from model 1 for each seed
        metrics_model2: metric values from model 2 for each seed

    Returns:
        dict with statistic, p_value
    """
    diffs = np.array(metrics_model1) - np.array(metrics_model2)
    if np.all(diffs == 0):
        return {"statistic": 0.0, "p_value": 1.0, "significant": False}
    try:
        stat, p_value = stats.wilcoxon(diffs)
    except ValueError:
        return {"statistic": np.nan, "p_value": np.nan, "significant": False}
    return {
        "statistic": stat,
        "p_value": p_value,
        "significant": p_value < 0.05,
    }


# ---------------------------------------------------------------------------
# Full comparison pipeline
# ---------------------------------------------------------------------------


def run_all_comparisons(
    y_true: np.ndarray,
    model_predictions: dict[str, np.ndarray],
    seed_metrics: dict[str, list[dict]],
    reference_model: str = "cascrop",
) -> dict:
    """Run all statistical tests comparing reference model to all baselines.

    Args:
        y_true: ground truth labels on test set
        model_predictions: dict mapping model_name -> predicted probabilities
        seed_metrics: dict mapping model_name -> list of metric dicts per seed
        reference_model: name of the reference model (CasCrop)

    Returns:
        Nested dict: comparison_name -> test_name -> results
    """
    ref_preds = model_predictions[reference_model]
    results = {}

    for model_name, preds in model_predictions.items():
        if model_name == reference_model:
            continue

        comparison_key = f"{reference_model}_vs_{model_name}"
        results[comparison_key] = {}

        # DeLong test
        results[comparison_key]["delong"] = delong_test(y_true, ref_preds, preds)

        # McNemar test
        results[comparison_key]["mcnemar"] = mcnemar_test(y_true, ref_preds, preds)

        # Bootstrap CI for AUC difference
        from sklearn.metrics import roc_auc_score

        results[comparison_key]["bootstrap_auc"] = bootstrap_ci(
            y_true, ref_preds, preds, roc_auc_score
        )

        # Paired t-test across seeds (on AUC)
        if model_name in seed_metrics and reference_model in seed_metrics:
            ref_aucs = [m["auc_roc"] for m in seed_metrics[reference_model]]
            model_aucs = [m["auc_roc"] for m in seed_metrics[model_name]]
            results[comparison_key]["paired_ttest"] = paired_ttest_across_seeds(
                ref_aucs, model_aucs
            )
            results[comparison_key]["wilcoxon"] = wilcoxon_test_across_seeds(
                ref_aucs, model_aucs
            )

        logger.info(
            f"{comparison_key}: DeLong p={results[comparison_key]['delong']['p_value']:.4f}"
        )

    return results
