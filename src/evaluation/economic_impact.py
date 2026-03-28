"""Economic impact estimation.

Translates model predictions into dollar values for the paper's
discussion section. Estimates preventable waste and national
extrapolation with confidence intervals.
"""

import numpy as np
import pandas as pd
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class EconomicImpactEstimator:
    """Estimate the economic value of CasCrop's predictions.

    For each correctly predicted waste event in the test set,
    sums the actual RMA indemnity amounts to estimate
    "potentially preventable waste."
    """

    def __init__(
        self,
        test_labels: pd.DataFrame,
        test_years: list[int] = [2022, 2023, 2024],
    ):
        """
        Args:
            test_labels: DataFrame with columns [fips, commodity, year, month,
                         waste, indemnity_amount, cause_category]
            test_years: years in the test set
        """
        self.test_labels = test_labels[
            test_labels["year"].isin(test_years)
        ].copy()
        self.test_years = test_years

    def estimate_preventable_waste(
        self,
        y_true: np.ndarray,
        y_prob_cascrop: np.ndarray,
        y_prob_baseline: np.ndarray,
        indemnity_amounts: np.ndarray,
        threshold: float = 0.5,
        lead_weeks: int = 4,
    ) -> dict:
        """Estimate dollar value of preventable waste.

        For each true positive (correctly predicted waste event):
        1. The model predicted waste > threshold at least lead_weeks in advance
        2. Sum the actual indemnity amounts for these events

        Args:
            y_true: ground truth waste labels
            y_prob_cascrop: CasCrop predicted probabilities
            y_prob_baseline: local-only baseline predicted probabilities
            indemnity_amounts: actual $ indemnity for each sample
            threshold: prediction threshold
            lead_weeks: minimum lead time (in weeks) for prediction to count

        Returns:
            dict with total, cascrop capturable, baseline capturable amounts
        """
        # True waste events
        waste_mask = y_true == 1
        total_indemnity = indemnity_amounts[waste_mask].sum()

        # CasCrop true positives
        cascrop_tp = waste_mask & (y_prob_cascrop >= threshold)
        cascrop_capturable = indemnity_amounts[cascrop_tp].sum()

        # Baseline true positives
        baseline_tp = waste_mask & (y_prob_baseline >= threshold)
        baseline_capturable = indemnity_amounts[baseline_tp].sum()

        improvement = cascrop_capturable - baseline_capturable

        results = {
            "total_test_indemnity": float(total_indemnity),
            "cascrop_capturable": float(cascrop_capturable),
            "cascrop_capture_rate": float(
                cascrop_capturable / total_indemnity if total_indemnity > 0 else 0
            ),
            "baseline_capturable": float(baseline_capturable),
            "baseline_capture_rate": float(
                baseline_capturable / total_indemnity if total_indemnity > 0 else 0
            ),
            "improvement_dollars": float(improvement),
            "improvement_pct_points": float(
                (cascrop_capturable - baseline_capturable)
                / total_indemnity
                * 100
                if total_indemnity > 0
                else 0
            ),
            "num_waste_events": int(waste_mask.sum()),
            "num_cascrop_tp": int(cascrop_tp.sum()),
            "num_baseline_tp": int(baseline_tp.sum()),
        }

        logger.info(
            f"Preventable waste: CasCrop ${cascrop_capturable:,.0f} "
            f"({results['cascrop_capture_rate']:.1%}) vs "
            f"Baseline ${baseline_capturable:,.0f} "
            f"({results['baseline_capture_rate']:.1%})"
        )

        return results

    def national_extrapolation(
        self,
        test_results: dict,
        national_annual_indemnity: float = 12e9,
        test_coverage_fraction: float = 0.8,
    ) -> dict:
        """Scale test-set results to national annual estimates.

        Args:
            test_results: output from estimate_preventable_waste
            national_annual_indemnity: total US annual crop indemnity (~$12B)
            test_coverage_fraction: fraction of national crop area in our dataset

        Returns:
            dict with national annual estimates
        """
        # Scale factor: test covers 3 years and a fraction of national crop area
        test_annual = test_results["total_test_indemnity"] / len(self.test_years)
        scale_factor = national_annual_indemnity / (
            test_annual / test_coverage_fraction
        ) if test_annual > 0 else 1.0

        national = {
            "national_annual_indemnity": national_annual_indemnity,
            "scale_factor": scale_factor,
            "cascrop_annual_preventable": (
                test_results["cascrop_capturable"]
                / len(self.test_years)
                * scale_factor
            ),
            "baseline_annual_preventable": (
                test_results["baseline_capturable"]
                / len(self.test_years)
                * scale_factor
            ),
            "annual_improvement": (
                test_results["improvement_dollars"]
                / len(self.test_years)
                * scale_factor
            ),
        }

        logger.info(
            f"National extrapolation: CasCrop could enable early warning for "
            f"${national['cascrop_annual_preventable']:,.0f} in annual crop waste"
        )

        return national

    def bootstrap_impact_ci(
        self,
        y_true: np.ndarray,
        y_prob_cascrop: np.ndarray,
        y_prob_baseline: np.ndarray,
        indemnity_amounts: np.ndarray,
        n_bootstrap: int = 10000,
        confidence: float = 0.95,
        seed: int = 42,
    ) -> dict:
        """Bootstrap confidence intervals for impact estimates.

        Returns:
            dict with CI for preventable waste improvement
        """
        rng = np.random.RandomState(seed)
        n = len(y_true)
        improvements = np.zeros(n_bootstrap)

        for i in range(n_bootstrap):
            idx = rng.choice(n, n, replace=True)
            waste_mask = y_true[idx] == 1
            cascrop_tp = waste_mask & (y_prob_cascrop[idx] >= 0.5)
            baseline_tp = waste_mask & (y_prob_baseline[idx] >= 0.5)
            improvements[i] = (
                indemnity_amounts[idx][cascrop_tp].sum()
                - indemnity_amounts[idx][baseline_tp].sum()
            )

        alpha = 1 - confidence
        return {
            "mean_improvement": float(np.mean(improvements)),
            "ci_lower": float(np.percentile(improvements, 100 * alpha / 2)),
            "ci_upper": float(np.percentile(improvements, 100 * (1 - alpha / 2))),
            "std": float(np.std(improvements)),
        }

    def lead_time_analysis(
        self,
        predictions_by_lead: dict[int, tuple[np.ndarray, np.ndarray]],
        y_true: np.ndarray,
        indemnity_amounts: np.ndarray,
        threshold: float = 0.5,
    ) -> pd.DataFrame:
        """Analyze prediction performance at different lead times.

        Args:
            predictions_by_lead: dict mapping lead_weeks -> (cascrop_probs, baseline_probs)
            y_true: ground truth labels
            indemnity_amounts: actual indemnity amounts
            threshold: prediction threshold

        Returns:
            DataFrame with columns: lead_weeks, cascrop_recall, baseline_recall,
            cascrop_capturable, baseline_capturable
        """
        rows = []
        waste_mask = y_true == 1
        total_indemnity = indemnity_amounts[waste_mask].sum()

        for lead_weeks, (cascrop_prob, baseline_prob) in sorted(
            predictions_by_lead.items()
        ):
            cascrop_tp = waste_mask & (cascrop_prob >= threshold)
            baseline_tp = waste_mask & (baseline_prob >= threshold)

            rows.append(
                {
                    "lead_weeks": lead_weeks,
                    "cascrop_recall": float(cascrop_tp.sum() / waste_mask.sum())
                    if waste_mask.sum() > 0
                    else 0,
                    "baseline_recall": float(baseline_tp.sum() / waste_mask.sum())
                    if waste_mask.sum() > 0
                    else 0,
                    "cascrop_capturable": float(indemnity_amounts[cascrop_tp].sum()),
                    "baseline_capturable": float(indemnity_amounts[baseline_tp].sum()),
                    "cascrop_capture_rate": float(
                        indemnity_amounts[cascrop_tp].sum() / total_indemnity
                    )
                    if total_indemnity > 0
                    else 0,
                    "baseline_capture_rate": float(
                        indemnity_amounts[baseline_tp].sum() / total_indemnity
                    )
                    if total_indemnity > 0
                    else 0,
                }
            )

        return pd.DataFrame(rows)

    def save_impact_report(self, results: dict, output_path: str):
        """Save economic impact results as JSON."""
        import json

        with open(output_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        logger.info(f"Impact report saved to {output_path}")
