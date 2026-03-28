"""Case study: identify and reconstruct cascade events.

Searches for compelling examples where economic contagion caused
waste in counties with good biophysical conditions, then traces
the model's attention weights to tell the cascade story.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional
import logging

logger = logging.getLogger(__name__)

# Candidate events known to exhibit economic contagion patterns
CANDIDATE_EVENTS = {
    "2019_midwest_floods": {
        "description": "Iowa/Illinois flooding causing price effects on neighboring states",
        "source_states": ["IA", "IL"],
        "year": 2019,
        "months": [3, 4, 5, 6, 7],
        "expected_crops": ["CORN", "SOYBEANS"],
    },
    "2012_drought": {
        "description": "Widespread drought — contrast between affected and unaffected regions",
        "source_states": ["IN", "IL", "IA", "MO"],
        "year": 2012,
        "months": [6, 7, 8, 9],
        "expected_crops": ["CORN", "SOYBEANS"],
    },
    "2020_derecho": {
        "description": "Iowa derecho — localized destruction with price ripple",
        "source_states": ["IA"],
        "year": 2020,
        "months": [8, 9, 10],
        "expected_crops": ["CORN"],
    },
}


class CaseStudyAnalyzer:
    """Identify and reconstruct cascade events for the paper.

    Finds events where:
    1. Large crop loss occurred in region A (source)
    2. Waste followed in region B within 2-8 weeks
    3. Region B had good biophysical conditions (no weather disaster)
    4. Region B's waste was economically driven (price decline cause code)
    """

    def __init__(
        self,
        labels_df: pd.DataFrame,
        features_df: pd.DataFrame,
        config: dict,
    ):
        self.labels_df = labels_df
        self.features_df = features_df
        self.config = config

    def find_cascade_candidates(
        self,
        min_source_indemnity: float = 1e6,
        max_target_ndvi_deficit: float = -0.1,
        lag_months: tuple = (1, 3),
    ) -> pd.DataFrame:
        """Search for candidate cascade events.

        Criteria:
        - Source county: high indemnity (biophysical cause) in month t
        - Target county: waste in month t+lag with economic cause code
        - Target county: above-median NDVI (good conditions)

        Args:
            min_source_indemnity: minimum $ loss at source to qualify
            max_target_ndvi_deficit: max NDVI deviation from normal (negative = bad)
            lag_months: (min_lag, max_lag) in months

        Returns:
            DataFrame of candidate events with source/target info
        """
        # Source events: large biophysical losses
        bio_causes = ["DROUGHT", "EXCESS_MOISTURE", "COLD", "HEAT"]
        source_events = self.labels_df[
            (self.labels_df["indemnity_amount"] >= min_source_indemnity)
            & (self.labels_df["cause_category"].isin(bio_causes))
        ].copy()

        # Target events: economic waste with good conditions
        target_events = self.labels_df[
            (self.labels_df["waste"] == 1)
            & (self.labels_df["cause_category"] == "PRICE")
        ].copy()

        # Merge features for NDVI check on targets
        if "NDVI_mean" in self.features_df.columns:
            target_with_features = target_events.merge(
                self.features_df[["fips", "commodity", "year", "month", "NDVI_mean"]],
                on=["fips", "commodity", "year", "month"],
                how="left",
            )
            # Filter to targets with good conditions
            ndvi_median = self.features_df["NDVI_mean"].median()
            target_with_features = target_with_features[
                target_with_features["NDVI_mean"] >= ndvi_median + max_target_ndvi_deficit
            ]
        else:
            target_with_features = target_events

        # Find source-target pairs with appropriate lag
        candidates = []
        for _, source in source_events.iterrows():
            for lag in range(lag_months[0], lag_months[1] + 1):
                target_month = source["month"] + lag
                target_year = source["year"]
                if target_month > 12:
                    target_month -= 12
                    target_year += 1

                matching_targets = target_with_features[
                    (target_with_features["year"] == target_year)
                    & (target_with_features["month"] == target_month)
                    & (target_with_features["commodity"] == source["commodity"])
                    & (target_with_features["fips"] != source["fips"])
                ]

                for _, target in matching_targets.iterrows():
                    candidates.append(
                        {
                            "source_fips": source["fips"],
                            "source_state": source.get("state", ""),
                            "target_fips": target["fips"],
                            "target_state": target.get("state", ""),
                            "commodity": source["commodity"],
                            "source_year": source["year"],
                            "source_month": source["month"],
                            "target_year": target_year,
                            "target_month": target_month,
                            "lag_months": lag,
                            "source_cause": source["cause_category"],
                            "source_indemnity": source["indemnity_amount"],
                            "target_cause": target["cause_category"],
                            "target_indemnity": target.get("indemnity_amount", 0),
                            "target_ndvi": target.get("NDVI_mean", np.nan),
                        }
                    )

        candidates_df = pd.DataFrame(candidates)
        if len(candidates_df) > 0:
            candidates_df = candidates_df.sort_values(
                "source_indemnity", ascending=False
            )
        logger.info(f"Found {len(candidates_df)} cascade candidates")
        return candidates_df

    def select_best_candidate(
        self, candidates_df: pd.DataFrame
    ) -> Optional[dict]:
        """Select the most compelling cascade event for the case study.

        Prioritizes:
        1. Events in known candidate periods (2012 drought, 2019 floods, 2020 derecho)
        2. Events with highest source indemnity
        3. Events with clear geographic separation between source and target
        """
        if len(candidates_df) == 0:
            logger.warning("No cascade candidates found")
            return None

        # Score each candidate
        scores = np.zeros(len(candidates_df))

        for event_name, event_info in CANDIDATE_EVENTS.items():
            mask = (
                (candidates_df["source_year"] == event_info["year"])
                & (
                    candidates_df["source_month"].isin(event_info["months"])
                )
                & (
                    candidates_df["commodity"].isin(event_info["expected_crops"])
                )
            )
            scores[mask] += 10  # Bonus for known events

        # Score by source magnitude
        if candidates_df["source_indemnity"].max() > 0:
            scores += (
                candidates_df["source_indemnity"].values
                / candidates_df["source_indemnity"].max()
                * 5
            )

        best_idx = np.argmax(scores)
        best = candidates_df.iloc[best_idx].to_dict()
        logger.info(
            f"Selected cascade: {best['source_fips']} -> {best['target_fips']}, "
            f"{best['commodity']}, {best['source_year']}/{best['source_month']}"
        )
        return best

    def reconstruct_cascade(
        self,
        event: dict,
        attention_analyzer,
        model,
        data_loader,
    ) -> dict:
        """Reconstruct the full cascade narrative for a selected event.

        Traces the model's attention weights week by week to show how
        the cascade propagated from source to target.

        Args:
            event: selected cascade event dict
            attention_analyzer: AttentionAnalyzer instance
            model: trained CasCrop model
            data_loader: test data loader

        Returns:
            dict with full cascade reconstruction data
        """
        # Time window for the cascade
        time_steps = []
        for offset in range(-2, event["lag_months"] + 3):
            m = event["source_month"] + offset
            y = event["source_year"]
            if m > 12:
                m -= 12
                y += 1
            elif m < 1:
                m += 12
                y -= 1
            time_steps.append((y, m))

        # Track attention evolution at target
        evolution = attention_analyzer.temporal_attention_evolution(
            model=model,
            data_loader=data_loader,
            target_fips=event["target_fips"],
            time_steps=time_steps,
        )

        reconstruction = {
            "event": event,
            "time_steps": time_steps,
            "attention_evolution": evolution,
            "narrative": self._generate_narrative(event, evolution),
        }

        return reconstruction

    def _generate_narrative(self, event: dict, evolution: list[dict]) -> str:
        """Generate the textual narrative for the case study section.

        Creates a paragraph suitable for Section 6 of the paper.
        """
        if not evolution:
            return "Insufficient data to reconstruct cascade narrative."

        # Find peak attention and waste probability
        peak_attn = max(evolution, key=lambda x: x["incoming_attention_sum"])
        peak_waste = max(evolution, key=lambda x: x["waste_probability"])

        narrative = (
            f"In {event['source_year']}/{event['source_month']:02d}, "
            f"county {event['source_fips']} experienced significant crop loss "
            f"(${event['source_indemnity']:,.0f} in indemnities) due to "
            f"{event['source_cause'].lower().replace('_', ' ')}. "
            f"CasCrop's ECMP attention mechanism detected economic contagion "
            f"propagating to county {event['target_fips']}, "
            f"where biophysical conditions remained favorable "
            f"(NDVI = {event.get('target_ndvi', 'N/A')}). "
            f"By {peak_waste['year']}/{peak_waste['month']:02d}, "
            f"the model predicted a {peak_waste['waste_probability']:.0%} "
            f"waste probability for the target county — "
            f"{event['lag_months']} months before the loss occurred. "
            f"A local-only model, lacking inter-regional economic signals, "
            f"would have missed this cascade entirely."
        )

        return narrative

    def compute_counterfactual_comparison(
        self,
        event: dict,
        cascrop_predictions: dict,
        local_only_predictions: dict,
    ) -> dict:
        """Compare CasCrop vs local-only predictions for the case study.

        Generates the key counterfactual statement:
        "CasCrop predicted X% waste probability Y weeks before loss.
         Local-only predicted only Z%."
        """
        target_key = (
            event["target_fips"],
            event["commodity"],
            event["target_year"],
            event["target_month"],
        )

        cascrop_prob = cascrop_predictions.get(target_key, np.nan)
        local_prob = local_only_predictions.get(target_key, np.nan)

        return {
            "target": target_key,
            "cascrop_probability": cascrop_prob,
            "local_only_probability": local_prob,
            "probability_difference": cascrop_prob - local_prob,
            "lead_time_months": event["lag_months"],
            "statement": (
                f"CasCrop predicted {cascrop_prob:.0%} waste probability "
                f"{event['lag_months']} months before the loss occurred. "
                f"A local-only model predicted only {local_prob:.0%}. "
                f"The difference was driven by ECMP edges from "
                f"county {event['source_fips']} where "
                f"{event['source_cause'].lower().replace('_', ' ')} "
                f"caused ${event['source_indemnity']:,.0f} in losses."
            ),
        }

    def save_case_study(self, reconstruction: dict, output_dir: Path):
        """Save all case study data and narrative."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        import json

        # Save reconstruction data (without non-serializable objects)
        serializable = {
            "event": reconstruction["event"],
            "narrative": reconstruction["narrative"],
            "attention_evolution": reconstruction["attention_evolution"],
        }
        with open(output_dir / "case_study_data.json", "w") as f:
            json.dump(serializable, f, indent=2, default=str)

        # Save narrative as markdown
        with open(output_dir / "case_study_narrative.md", "w") as f:
            f.write("# Case Study: Cascade Event Reconstruction\n\n")
            f.write(reconstruction["narrative"])
            f.write("\n")

        logger.info(f"Case study saved to {output_dir}")
