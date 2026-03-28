"""Evaluation metrics for CasCrop models.

Computes AUC-ROC, AUC-PR, F1, precision, recall, Brier score,
log loss, expected calibration error, and cause-of-loss metrics.
"""

import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    brier_score_loss,
    log_loss,
    accuracy_score,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
)
from typing import Optional


def expected_calibration_error(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Compute Expected Calibration Error (ECE).

    Bins predictions by confidence and measures the gap between
    predicted probability and actual frequency in each bin.
    """
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (y_prob >= bin_boundaries[i]) & (y_prob < bin_boundaries[i + 1])
        if mask.sum() == 0:
            continue
        bin_acc = y_true[mask].mean()
        bin_conf = y_prob[mask].mean()
        bin_weight = mask.sum() / len(y_true)
        ece += bin_weight * abs(bin_acc - bin_conf)
    return ece


def compute_binary_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> dict:
    """Compute all binary classification metrics for waste prediction.

    Args:
        y_true: ground truth binary labels
        y_prob: predicted probabilities
        threshold: classification threshold

    Returns:
        dict with all metrics
    """
    y_pred = (y_prob >= threshold).astype(int)

    metrics = {
        "auc_roc": roc_auc_score(y_true, y_prob),
        "auc_pr": average_precision_score(y_true, y_prob),
        "f1_binary": f1_score(y_true, y_pred, zero_division=0),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "brier_score": brier_score_loss(y_true, y_prob),
        "log_loss": log_loss(y_true, y_prob, eps=1e-7),
        "ece": expected_calibration_error(y_true, y_prob),
    }
    return metrics


def compute_cause_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: int = 6,
) -> dict:
    """Compute multi-class cause-of-loss metrics.

    Args:
        y_true: ground truth cause labels (0-5)
        y_pred: predicted cause labels (0-5)
        num_classes: number of cause classes

    Returns:
        dict with macro/weighted F1 and accuracy
    """
    return {
        "f1_macro_cause": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_weighted_cause": f1_score(
            y_true, y_pred, average="weighted", zero_division=0
        ),
        "cause_accuracy": accuracy_score(y_true, y_pred),
    }


def compute_all_metrics(
    y_true_waste: np.ndarray,
    y_prob_waste: np.ndarray,
    y_true_cause: Optional[np.ndarray] = None,
    y_pred_cause: Optional[np.ndarray] = None,
    threshold: float = 0.5,
) -> dict:
    """Compute all metrics for a model's predictions.

    Returns combined dict of binary waste metrics and cause metrics.
    """
    metrics = compute_binary_metrics(y_true_waste, y_prob_waste, threshold)
    if y_true_cause is not None and y_pred_cause is not None:
        cause_metrics = compute_cause_metrics(y_true_cause, y_pred_cause)
        metrics.update(cause_metrics)
    return metrics


def compute_subgroup_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    group_labels: np.ndarray,
    group_names: Optional[list] = None,
) -> dict:
    """Compute metrics broken down by subgroup.

    Args:
        y_true: binary waste labels
        y_prob: predicted probabilities
        group_labels: integer group assignments for each sample
        group_names: optional names for each group

    Returns:
        dict mapping group name -> metrics dict
    """
    unique_groups = np.unique(group_labels)
    results = {}
    for g in unique_groups:
        mask = group_labels == g
        if mask.sum() < 10:
            continue
        name = group_names[g] if group_names else str(g)
        try:
            results[name] = compute_binary_metrics(y_true[mask], y_prob[mask])
        except ValueError:
            # Skip if only one class present in subgroup
            results[name] = {"auc_roc": np.nan, "note": "single_class"}
    return results


def aggregate_seed_metrics(
    seed_metrics: list[dict],
) -> dict:
    """Aggregate metrics across seeds as mean +/- std.

    Args:
        seed_metrics: list of metric dicts, one per seed

    Returns:
        dict mapping metric_name -> {"mean": float, "std": float, "formatted": str}
    """
    all_keys = seed_metrics[0].keys()
    aggregated = {}
    for key in all_keys:
        values = [m[key] for m in seed_metrics if isinstance(m.get(key), (int, float))]
        if not values:
            continue
        mean = np.mean(values)
        std = np.std(values)
        aggregated[key] = {
            "mean": mean,
            "std": std,
            "formatted": f"{mean:.3f} ± {std:.3f}",
        }
    return aggregated


def get_roc_curve_data(
    y_true: np.ndarray, y_prob: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Get ROC curve data for plotting."""
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    return fpr, tpr, thresholds


def get_pr_curve_data(
    y_true: np.ndarray, y_prob: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Get precision-recall curve data for plotting."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    return precision, recall, thresholds
