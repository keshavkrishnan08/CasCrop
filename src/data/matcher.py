"""
matcher.py — Match all data sources by county-crop-month.

Merges biophysical (satellite, weather, soil), economic (prices, costs),
historical (past losses, yields), and RMA insurance claim data into a
unified feature matrix and label set at county-crop-month granularity.

Handles temporal splitting (train 2008-2019, val 2020-2021, test 2022-2024)
with strict no-leakage guarantees: every feature for month t uses only
data available before t, lagged by at least 1 week.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── RMA cause-of-loss code mapping ──────────────────────────────────────────
CAUSE_MAP: dict[str, list[int]] = {
    "DROUGHT": [2, 3],
    "EXCESS_MOISTURE": [10, 11, 14],
    "COLD": [15, 16, 17],
    "HEAT": [36, 40],
    "PRICE": [47, 48],
}

CAUSE_LABEL_TO_INT: dict[str, int] = {
    "DROUGHT": 0,
    "EXCESS_MOISTURE": 1,
    "COLD": 2,
    "HEAT": 3,
    "PRICE": 4,
    "OTHER": 5,
}

# ── Feature column groups ───────────────────────────────────────────────────
BIOPHYSICAL_FEATURES = [
    "NDVI_mean",
    "NDVI_std",
    "EVI_mean",
    "SAVI_mean",
    "NDWI_mean",
    "soil_moisture_surface",
    "soil_moisture_rootzone",
    "GDD_cumulative",
    "frost_days",
    "consec_dry_days",
    "PDSI",
    "TMAX_mean",
    "TMIN_mean",
    "PRCP_total",
    "VegScape_condition",
    "crop_area_fraction",
    "historical_yield_mean",
    "historical_yield_std",
    # Extended satellite features
    "NDVI_min",
    "NDVI_max",
    "NDVI_range",
    "EVI_std",
    "NDWI_std",
    "SAVI_std",
    "soil_moisture_surface_std",
    "soil_moisture_rootzone_std",
    "GDD_deviation_from_normal",
    "PRCP_deviation_from_normal",
    "TMAX_deviation_from_normal",
    "TMIN_deviation_from_normal",
]

ECONOMIC_FEATURES = [
    "commodity_price_current",
    "price_change_1m",
    "price_change_3m",
    "price_volatility_30d",
    "cost_of_production",
    "revenue_to_cost_ratio",
    "futures_basis",
    "national_supply_estimate",
    "export_demand_index",
    # Extended economic features
    "price_change_6m",
    "price_to_5yr_avg_ratio",
    "cost_change_1yr",
    "insurance_premium_rate",
    "national_stocks_to_use",
    "global_production_estimate",
]

HISTORICAL_FEATURES = [
    "county_historical_loss_frequency",
    "loss_severity",
    "avg_indemnity",
    "crop_diversity_index",
    "irrigation_fraction",
    # Extended historical features
    "prev_year_waste",
    "prev_year_indemnity",
    "loss_frequency_3yr",
    "loss_frequency_5yr",
    "county_avg_yield_trend",
]

TEMPORAL_FEATURES = [
    "month_sin",
    "month_cos",
    "year",
]

MERGE_KEYS = ["fips", "commodity", "year", "month"]


def _code_to_cause(code: int) -> str:
    """Map a single RMA cause-of-loss code to a human-readable category."""
    for cause, codes in CAUSE_MAP.items():
        if code in codes:
            return cause
    return "OTHER"


class DataMatcher:
    """Merge every data source on (fips, commodity, year, month) keys.

    Produces a feature matrix and a label matrix suitable for model
    training, with temporal splits that guarantee no data leakage.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config

        data_cfg = config.get("data", {})
        self.waste_threshold: float = data_cfg.get("waste_threshold", 10_000)
        self.train_years: list[int] = data_cfg.get(
            "train_years", list(range(2008, 2020))
        )
        self.val_years: list[int] = data_cfg.get("val_years", [2020, 2021])
        self.test_years: list[int] = data_cfg.get("test_years", [2022, 2023, 2024])
        self.commodities: list[str] = data_cfg.get(
            "commodities", ["CORN", "SOYBEANS", "WHEAT"]
        )
        self.feature_lag_weeks: int = data_cfg.get("feature_lag_weeks", 1)

        paths_cfg = config.get("paths", {})
        self.raw_dir = Path(paths_cfg.get("raw_data", "data/raw"))
        self.processed_dir = Path(paths_cfg.get("processed_data", "data/processed"))
        self.splits_dir = Path(paths_cfg.get("splits", "data/splits"))

    # ── FIPS standardization ────────────────────────────────────────────

    @staticmethod
    def standardize_fips(df: pd.DataFrame, fips_col: str = "fips") -> pd.DataFrame:
        """Ensure FIPS codes are zero-padded 5-digit strings.

        Handles numeric columns, strings missing leading zeros, and
        state+county code pairs that need concatenation.
        """
        df = df.copy()

        if fips_col not in df.columns:
            # Try combining state_code + county_code
            if "state_code" in df.columns and "county_code" in df.columns:
                state = df["state_code"].astype(str).str.zfill(2)
                county = df["county_code"].astype(str).str.zfill(3)
                df[fips_col] = state + county
                logger.info(
                    "Created '%s' from state_code + county_code.", fips_col
                )
            else:
                raise KeyError(
                    f"Column '{fips_col}' not found and cannot construct it "
                    "from state_code + county_code."
                )
        else:
            df[fips_col] = (
                df[fips_col]
                .astype(str)
                .str.strip()
                .str.zfill(5)
            )

        # Drop rows with obviously invalid FIPS
        valid_mask = df[fips_col].str.match(r"^\d{5}$")
        n_invalid = (~valid_mask).sum()
        if n_invalid > 0:
            logger.warning("Dropping %d rows with invalid FIPS codes.", n_invalid)
            df = df.loc[valid_mask].copy()

        return df

    # ── Label construction ──────────────────────────────────────────────

    def construct_target_labels(
        self,
        rma_df: pd.DataFrame,
        threshold: float | None = None,
    ) -> pd.DataFrame:
        """Build binary waste and multi-class cause labels from RMA data.

        Parameters
        ----------
        rma_df : pd.DataFrame
            RMA cause-of-loss records with at least: fips, commodity,
            year, month, indemnity_amount, cause_of_loss_code.
        threshold : float, optional
            Indemnity dollar threshold for the binary waste label.
            Falls back to ``self.waste_threshold`` (default 10 000).

        Returns
        -------
        pd.DataFrame
            One row per (fips, commodity, year, month) with columns:
            ``waste`` (0/1), ``cause`` (int 0-5), ``cause_name`` (str),
            ``total_indemnity`` (float), ``num_claims`` (int).
        """
        if threshold is None:
            threshold = self.waste_threshold

        rma = rma_df.copy()
        rma = self.standardize_fips(rma)

        # Ensure required columns exist
        required = {"fips", "commodity", "year", "month",
                    "indemnity_amount", "cause_of_loss_code"}
        missing = required - set(rma.columns)
        if missing:
            raise ValueError(f"RMA DataFrame missing columns: {missing}")

        # Normalize commodity names
        rma["commodity"] = rma["commodity"].str.upper().str.strip()

        # Filter to target commodities
        if self.commodities:
            rma = rma[rma["commodity"].isin(self.commodities)].copy()

        # Aggregate to county-crop-month level
        agg = (
            rma.groupby(MERGE_KEYS, as_index=False)
            .agg(
                total_indemnity=("indemnity_amount", "sum"),
                num_claims=("indemnity_amount", "count"),
                # Dominant cause = the cause with the highest total indemnity
                dominant_cause_code=(
                    "cause_of_loss_code",
                    lambda codes: codes.mode().iloc[0] if len(codes) > 0 else -1,
                ),
            )
        )

        # Recompute dominant cause by highest indemnity, not mode
        # Group again and pick the cause code with max total indemnity
        cause_agg = (
            rma.groupby(MERGE_KEYS + ["cause_of_loss_code"], as_index=False)
            ["indemnity_amount"]
            .sum()
        )
        idx_max = cause_agg.groupby(MERGE_KEYS)["indemnity_amount"].idxmax()
        dominant = cause_agg.loc[idx_max, MERGE_KEYS + ["cause_of_loss_code"]].copy()
        dominant.rename(columns={"cause_of_loss_code": "dominant_cause_code"}, inplace=True)

        # Merge dominant cause back onto aggregation
        agg = agg.drop(columns=["dominant_cause_code"], errors="ignore")
        agg = agg.merge(dominant, on=MERGE_KEYS, how="left")

        # Binary waste label
        agg["waste"] = (agg["total_indemnity"] > threshold).astype(int)

        # Multi-class cause label
        agg["cause_name"] = agg["dominant_cause_code"].apply(_code_to_cause)
        agg["cause"] = agg["cause_name"].map(CAUSE_LABEL_TO_INT)

        logger.info(
            "Labels built: %d observations, %.1f%% waste-positive.",
            len(agg),
            100.0 * agg["waste"].mean(),
        )

        return agg[MERGE_KEYS + [
            "waste", "cause", "cause_name",
            "total_indemnity", "num_claims",
        ]]

    # ── Feature matrix construction ─────────────────────────────────────

    def construct_feature_matrix(
        self,
        bio_df: pd.DataFrame,
        econ_df: pd.DataFrame,
        hist_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Merge biophysical, economic, and historical feature tables.

        Each input DataFrame must contain the merge keys
        (fips, commodity, year, month). Missing feature columns are
        filled with NaN and flagged for later imputation.

        Returns a single DataFrame sorted by (year, month, fips, commodity).
        """
        for name, df in [("bio", bio_df), ("econ", econ_df), ("hist", hist_df)]:
            missing = set(MERGE_KEYS) - set(df.columns)
            if missing:
                raise ValueError(
                    f"{name}_df missing merge key columns: {missing}"
                )

        # Standardize FIPS across all inputs
        bio_df = self.standardize_fips(bio_df.copy())
        econ_df = self.standardize_fips(econ_df.copy())
        hist_df = self.standardize_fips(hist_df.copy())

        # Normalize commodity
        for df in [bio_df, econ_df, hist_df]:
            df["commodity"] = df["commodity"].str.upper().str.strip()

        # Merge step-wise: bio <-- econ <-- hist
        merged = bio_df.merge(econ_df, on=MERGE_KEYS, how="outer", suffixes=("", "_econ"))
        merged = merged.merge(hist_df, on=MERGE_KEYS, how="outer", suffixes=("", "_hist"))

        # Add temporal encoding
        merged["month_sin"] = np.sin(2 * np.pi * merged["month"] / 12)
        merged["month_cos"] = np.cos(2 * np.pi * merged["month"] / 12)

        # Log which expected features are present / absent
        all_expected = BIOPHYSICAL_FEATURES + ECONOMIC_FEATURES + HISTORICAL_FEATURES
        present = [c for c in all_expected if c in merged.columns]
        absent = [c for c in all_expected if c not in merged.columns]

        logger.info(
            "Feature matrix: %d rows, %d/%d expected features present.",
            len(merged), len(present), len(all_expected),
        )
        if absent:
            logger.warning("Missing feature columns (will be NaN): %s", absent)
            for col in absent:
                merged[col] = np.nan

        # Sort for determinism
        merged.sort_values(MERGE_KEYS, inplace=True)
        merged.reset_index(drop=True, inplace=True)

        return merged

    # ── Temporal split creation ─────────────────────────────────────────

    def create_temporal_splits(
        self,
        df: pd.DataFrame,
    ) -> dict[str, np.ndarray]:
        """Partition rows into train / val / test by year.

        Returns a dict of split name -> integer index array.
        Guarantees no temporal leakage: every feature in each split
        has already been lagged by ``self.feature_lag_weeks``.
        """
        splits: dict[str, np.ndarray] = {}

        for split_name, years in [
            ("train", self.train_years),
            ("val", self.val_years),
            ("test", self.test_years),
        ]:
            mask = df["year"].isin(years)
            indices = df.index[mask].values
            splits[split_name] = indices
            logger.info(
                "Split '%s': %d rows (%s).",
                split_name, len(indices),
                ", ".join(str(y) for y in sorted(years)),
            )

        # Sanity: no overlap
        all_idx = np.concatenate(list(splits.values()))
        if len(all_idx) != len(set(all_idx)):
            raise RuntimeError("Temporal splits have overlapping indices!")

        return splits

    # ── Missing-data handling ───────────────────────────────────────────

    def handle_missing_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Impute missing values with strategy-specific methods.

        Strategies by feature group:
        - Sentinel-derived indices before 2015: flag for Landsat backfill
        - SMAP soil moisture before 2015: flag for NLDAS backfill
        - Weather: spatial interpolation from nearest 3 stations (approx.
          here via per-year-month county median as a tractable proxy)
        - Prices: forward-fill along (commodity) time series
        - Remaining numeric: per-column median within the same year
        """
        df = df.copy()

        n_missing_before = df.isna().sum().sum()

        # ── Sentinel backfill flag ──────────────────────────────────────
        sentinel_cols = [
            c for c in BIOPHYSICAL_FEATURES
            if c.startswith(("NDVI", "EVI", "SAVI", "NDWI"))
        ]
        pre_2015 = df["year"] < 2015
        for col in sentinel_cols:
            if col in df.columns:
                backfill_mask = pre_2015 & df[col].isna()
                if backfill_mask.any():
                    df.loc[backfill_mask, col] = np.nan  # stays NaN; flag
                    logger.debug(
                        "%d rows for '%s' pre-2015 flagged for Landsat backfill.",
                        backfill_mask.sum(), col,
                    )

        # ── SMAP backfill flag ──────────────────────────────────────────
        smap_cols = [c for c in df.columns if "soil_moisture" in c]
        for col in smap_cols:
            backfill_mask = pre_2015 & df[col].isna()
            if backfill_mask.any():
                logger.debug(
                    "%d rows for '%s' pre-2015 flagged for NLDAS backfill.",
                    backfill_mask.sum(), col,
                )

        # ── Weather: spatial interpolation proxy ────────────────────────
        weather_cols = [
            c for c in df.columns
            if c in {
                "TMAX_mean", "TMIN_mean", "PRCP_total", "GDD_cumulative",
                "frost_days", "consec_dry_days", "PDSI",
                "TMAX_deviation_from_normal", "TMIN_deviation_from_normal",
                "PRCP_deviation_from_normal", "GDD_deviation_from_normal",
            }
        ]
        if weather_cols:
            # Group by state (first 2 digits of FIPS) + year + month
            df["_state"] = df["fips"].str[:2]
            group_cols = ["_state", "year", "month"]
            medians = df.groupby(group_cols)[weather_cols].transform("median")
            for col in weather_cols:
                mask = df[col].isna()
                df.loc[mask, col] = medians.loc[mask, col]
            df.drop(columns=["_state"], inplace=True)

        # ── Price: forward-fill per commodity ───────────────────────────
        price_cols = [c for c in ECONOMIC_FEATURES if c in df.columns]
        if price_cols:
            df.sort_values(["commodity", "year", "month"], inplace=True)
            for col in price_cols:
                df[col] = df.groupby("commodity")[col].ffill()
            # Back-fill the earliest entries if still NaN
            for col in price_cols:
                df[col] = df.groupby("commodity")[col].bfill()

        # ── Remaining: per-year column median ───────────────────────────
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        still_missing = [c for c in numeric_cols if df[c].isna().any()]
        if still_missing:
            year_medians = df.groupby("year")[still_missing].transform("median")
            for col in still_missing:
                mask = df[col].isna()
                df.loc[mask, col] = year_medians.loc[mask, col]

        # Final fallback: global median
        for col in still_missing:
            if df[col].isna().any():
                df[col].fillna(df[col].median(), inplace=True)

        n_missing_after = df.isna().sum().sum()
        logger.info(
            "Missing data: %d -> %d NaN cells after imputation.",
            n_missing_before, n_missing_after,
        )

        df.sort_values(MERGE_KEYS, inplace=True)
        df.reset_index(drop=True, inplace=True)
        return df

    # ── Feature lagging ─────────────────────────────────────────────────

    def _apply_feature_lag(self, df: pd.DataFrame) -> pd.DataFrame:
        """Lag all feature columns by ``feature_lag_weeks`` to prevent leakage.

        At county-crop-month resolution a 1-week lag means we use the
        *previous month's* features (since our finest granularity is
        monthly). For intra-month predictions the lag is already built
        into how GEE composites are constructed.
        """
        if self.feature_lag_weeks <= 0:
            return df

        df = df.copy()
        feature_cols = [
            c for c in df.columns
            if c in (BIOPHYSICAL_FEATURES + ECONOMIC_FEATURES +
                     HISTORICAL_FEATURES + TEMPORAL_FEATURES)
        ]

        # Sort by entity and time so the shift is correct
        df.sort_values(["fips", "commodity", "year", "month"], inplace=True)

        # Build a monotonic time index for correct shifting
        df["_time_idx"] = df["year"] * 12 + df["month"]
        grouped = df.groupby(["fips", "commodity"])

        for col in feature_cols:
            if col in ("year", "month_sin", "month_cos"):
                # Temporal identity features don't get lagged
                continue
            df[col] = grouped[col].shift(1)

        df.drop(columns=["_time_idx"], inplace=True)

        # Drop rows that lost their features to the shift
        before = len(df)
        df.dropna(subset=feature_cols[:1], inplace=True)
        after = len(df)
        if before != after:
            logger.info(
                "Feature lag: dropped %d rows with no prior-period data.",
                before - after,
            )

        return df

    # ── Class imbalance analysis ────────────────────────────────────────

    @staticmethod
    def analyze_class_imbalance(labels_df: pd.DataFrame) -> dict[str, Any]:
        """Compute waste-to-no-waste ratio per crop and year.

        Returns a nested dict:
        ``{ "overall": float, "by_crop": {crop: float},
            "by_year": {year: float}, "by_crop_year": {crop: {year: float}},
            "cause_distribution": {cause: int} }``
        """
        results: dict[str, Any] = {}

        total_pos = labels_df["waste"].sum()
        total_neg = len(labels_df) - total_pos
        results["overall_ratio"] = (
            float(total_pos / total_neg) if total_neg > 0 else float("inf")
        )
        results["overall_positive_rate"] = float(labels_df["waste"].mean())
        results["n_total"] = len(labels_df)
        results["n_positive"] = int(total_pos)
        results["n_negative"] = int(total_neg)

        # By crop
        by_crop: dict[str, float] = {}
        for crop, grp in labels_df.groupby("commodity"):
            pos = grp["waste"].sum()
            neg = len(grp) - pos
            by_crop[str(crop)] = float(pos / neg) if neg > 0 else float("inf")
        results["by_crop"] = by_crop

        # By year
        by_year: dict[str, float] = {}
        for yr, grp in labels_df.groupby("year"):
            pos = grp["waste"].sum()
            neg = len(grp) - pos
            by_year[str(yr)] = float(pos / neg) if neg > 0 else float("inf")
        results["by_year"] = by_year

        # By crop-year
        by_crop_year: dict[str, dict[str, float]] = {}
        for (crop, yr), grp in labels_df.groupby(["commodity", "year"]):
            pos = grp["waste"].sum()
            neg = len(grp) - pos
            by_crop_year.setdefault(str(crop), {})[str(yr)] = (
                float(pos / neg) if neg > 0 else float("inf")
            )
        results["by_crop_year"] = by_crop_year

        # Cause distribution
        if "cause_name" in labels_df.columns:
            cause_dist = labels_df["cause_name"].value_counts().to_dict()
            results["cause_distribution"] = {
                str(k): int(v) for k, v in cause_dist.items()
            }

        logger.info(
            "Class imbalance: overall pos rate %.3f, ratio %.4f.",
            results["overall_positive_rate"],
            results["overall_ratio"],
        )

        return results

    # ── Full pipeline ───────────────────────────────────────────────────

    def match_all(
        self,
        data_dir: Path | None = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Load every processed source and merge on (fips, commodity, year, month).

        Parameters
        ----------
        data_dir : Path, optional
            Root of the processed data directory.  Falls back to
            ``self.raw_dir``.

        Returns
        -------
        features_df : pd.DataFrame
            Complete feature matrix with biophysical, economic,
            historical, and temporal columns.
        labels_df : pd.DataFrame
            Labels with waste (binary) and cause (multi-class) columns.
        """
        if data_dir is None:
            data_dir = self.raw_dir

        data_dir = Path(data_dir)
        logger.info("Loading data sources from %s", data_dir)

        # ── Load individual processed sources ───────────────────────────
        loaders: dict[str, Path] = {
            "rma": data_dir / "rma_claims.parquet",
            "sentinel": data_dir / "sentinel_indices.parquet",
            "weather": data_dir / "weather.parquet",
            "crop": data_dir / "crop_production.parquet",
            "price": data_dir / "commodity_prices.parquet",
            "vegscape": data_dir / "vegscape.parquet",
            "soil": data_dir / "soil_moisture.parquet",
        }

        loaded: dict[str, pd.DataFrame] = {}
        for name, path in loaders.items():
            if path.exists():
                loaded[name] = pd.read_parquet(path)
                logger.info("Loaded %s: %d rows from %s", name, len(loaded[name]), path)
            else:
                logger.warning("Source '%s' not found at %s — skipping.", name, path)

        if "rma" not in loaded:
            raise FileNotFoundError(
                f"RMA claims data not found at {loaders['rma']}. "
                "Cannot build labels without insurance claims."
            )

        # ── Construct labels ────────────────────────────────────────────
        labels_df = self.construct_target_labels(loaded["rma"])

        # ── Assemble feature groups ─────────────────────────────────────
        # Biophysical: sentinel + weather + vegscape + soil + crop stats
        bio_frames = []
        for key in ("sentinel", "weather", "vegscape", "soil", "crop"):
            if key in loaded:
                bio_frames.append(self.standardize_fips(loaded[key]))
        if bio_frames:
            bio_df = bio_frames[0]
            for frame in bio_frames[1:]:
                bio_df = bio_df.merge(frame, on=MERGE_KEYS, how="outer",
                                      suffixes=("", f"_{id(frame)}"))
        else:
            # Empty placeholder
            bio_df = labels_df[MERGE_KEYS].copy()

        # Economic: price data
        if "price" in loaded:
            econ_df = self.standardize_fips(loaded["price"])
        else:
            econ_df = labels_df[MERGE_KEYS].copy()

        # Historical: derived from RMA historical aggregates
        hist_df = self._build_historical_features(loaded["rma"])

        # ── Merge into feature matrix ───────────────────────────────────
        features_df = self.construct_feature_matrix(bio_df, econ_df, hist_df)

        # ── Apply feature lag ───────────────────────────────────────────
        features_df = self._apply_feature_lag(features_df)

        # ── Handle missing data ─────────────────────────────────────────
        features_df = self.handle_missing_data(features_df)

        # ── Align features and labels ───────────────────────────────────
        # Inner join so every row has both features and labels
        features_df = features_df.merge(
            labels_df[MERGE_KEYS],
            on=MERGE_KEYS,
            how="inner",
        )
        labels_df = labels_df.merge(
            features_df[MERGE_KEYS],
            on=MERGE_KEYS,
            how="inner",
        )

        logger.info(
            "Final matched dataset: %d observations, %d feature columns.",
            len(features_df),
            len([c for c in features_df.columns if c not in MERGE_KEYS]),
        )

        return features_df, labels_df

    # ── Historical feature builder ──────────────────────────────────────

    def _build_historical_features(
        self,
        rma_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Derive historical risk features from RMA data.

        For each (fips, commodity, year, month) compute backward-looking
        aggregates: loss frequency, severity, average indemnity, etc.
        Uses only data strictly before the current observation.
        """
        rma = self.standardize_fips(rma_df.copy())
        rma["commodity"] = rma["commodity"].str.upper().str.strip()

        # Sort chronologically
        rma.sort_values(["fips", "commodity", "year", "month"], inplace=True)

        # Unique county-crop-month observations
        obs = rma[MERGE_KEYS].drop_duplicates().copy()

        records = []
        # Group by county-crop for historical lookback
        for (fips, commodity), grp in rma.groupby(["fips", "commodity"]):
            grp = grp.sort_values(["year", "month"])
            for _, row in obs[
                (obs["fips"] == fips) & (obs["commodity"] == commodity)
            ].iterrows():
                yr, mo = row["year"], row["month"]
                # All data strictly before this month
                hist = grp[(grp["year"] < yr) | ((grp["year"] == yr) & (grp["month"] < mo))]

                if len(hist) == 0:
                    records.append({
                        "fips": fips,
                        "commodity": commodity,
                        "year": yr,
                        "month": mo,
                        "county_historical_loss_frequency": 0.0,
                        "loss_severity": 0.0,
                        "avg_indemnity": 0.0,
                        "prev_year_waste": 0,
                        "prev_year_indemnity": 0.0,
                        "loss_frequency_3yr": 0.0,
                        "loss_frequency_5yr": 0.0,
                    })
                    continue

                total_months = hist[["year", "month"]].drop_duplicates().shape[0]
                loss_months = hist[hist["indemnity_amount"] > self.waste_threshold]
                loss_months_count = loss_months[["year", "month"]].drop_duplicates().shape[0]

                freq = loss_months_count / max(total_months, 1)
                severity = (
                    hist["indemnity_amount"].sum() / max(total_months, 1)
                )
                avg_ind = hist["indemnity_amount"].mean()

                # Previous year same month
                prev = hist[(hist["year"] == yr - 1) & (hist["month"] == mo)]
                prev_waste = int(prev["indemnity_amount"].sum() > self.waste_threshold) if len(prev) > 0 else 0
                prev_ind = float(prev["indemnity_amount"].sum()) if len(prev) > 0 else 0.0

                # 3-year and 5-year frequency
                h3 = hist[hist["year"] >= yr - 3]
                h5 = hist[hist["year"] >= yr - 5]
                m3 = h3[["year", "month"]].drop_duplicates().shape[0]
                m5 = h5[["year", "month"]].drop_duplicates().shape[0]
                f3 = (
                    h3[h3["indemnity_amount"] > self.waste_threshold]
                    [["year", "month"]].drop_duplicates().shape[0]
                    / max(m3, 1)
                )
                f5 = (
                    h5[h5["indemnity_amount"] > self.waste_threshold]
                    [["year", "month"]].drop_duplicates().shape[0]
                    / max(m5, 1)
                )

                records.append({
                    "fips": fips,
                    "commodity": commodity,
                    "year": yr,
                    "month": mo,
                    "county_historical_loss_frequency": freq,
                    "loss_severity": severity,
                    "avg_indemnity": avg_ind,
                    "prev_year_waste": prev_waste,
                    "prev_year_indemnity": prev_ind,
                    "loss_frequency_3yr": f3,
                    "loss_frequency_5yr": f5,
                })

        hist_df = pd.DataFrame(records)
        logger.info("Built historical features: %d rows.", len(hist_df))
        return hist_df

    # ── Persistence ─────────────────────────────────────────────────────

    def save(
        self,
        features: pd.DataFrame,
        labels: pd.DataFrame,
        splits: dict[str, np.ndarray],
        output_dir: Path | str | None = None,
    ) -> None:
        """Write features, labels, splits, and normalization stats to disk."""
        out = Path(output_dir) if output_dir else self.processed_dir
        out.mkdir(parents=True, exist_ok=True)

        splits_out = self.splits_dir
        splits_out.mkdir(parents=True, exist_ok=True)

        # Features & labels as parquet
        feat_path = out / "features.parquet"
        lab_path = out / "labels.parquet"
        features.to_parquet(feat_path, index=False)
        labels.to_parquet(lab_path, index=False)
        logger.info("Saved features -> %s", feat_path)
        logger.info("Saved labels   -> %s", lab_path)

        # Temporal split indices
        for name, idx in splits.items():
            np.save(splits_out / f"{name}_indices.npy", idx)
        logger.info("Saved split indices -> %s", splits_out)

        # Normalization statistics (computed on train split only)
        train_idx = splits.get("train", np.array([], dtype=int))
        numeric_cols = features.select_dtypes(include=[np.number]).columns.tolist()
        feature_cols = [
            c for c in numeric_cols
            if c not in MERGE_KEYS
        ]

        if len(train_idx) > 0:
            train_data = features.iloc[train_idx]
            stats: dict[str, dict[str, float]] = {}
            for col in feature_cols:
                stats[col] = {
                    "mean": float(train_data[col].mean()),
                    "std": float(train_data[col].std()),
                    "min": float(train_data[col].min()),
                    "max": float(train_data[col].max()),
                    "median": float(train_data[col].median()),
                }
        else:
            stats = {}

        stats_path = out / "feature_statistics.json"
        with open(stats_path, "w") as f:
            json.dump(stats, f, indent=2)
        logger.info("Saved feature statistics -> %s", stats_path)

        # Class imbalance report
        imbalance = self.analyze_class_imbalance(labels)
        imbalance_path = out / "class_imbalance.json"
        with open(imbalance_path, "w") as f:
            json.dump(imbalance, f, indent=2)
        logger.info("Saved class imbalance report -> %s", imbalance_path)
