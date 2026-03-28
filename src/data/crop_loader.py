"""
USDA NASS Quick Stats crop production data loader.

Queries the NASS Quick Stats API for county-level crop statistics:
yield, production, area planted, area harvested, and crop condition.
Also downloads Cropland Data Layer (CDL) summaries for county-level
crop composition vectors.

API docs: https://quickstats.nass.usda.gov/api
Free API key: https://quickstats.nass.usda.gov/api#param_define

Typical usage:
    loader = NASSLoader(api_key="YOUR_KEY")
    df = loader.download_production_data(
        years=range(2008, 2025),
        commodities=["CORN", "SOYBEANS", "WHEAT"],
        output_dir=Path("data/raw/nass"),
    )
    waste = loader.calculate_waste_proxy(df)
"""

import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Union

import pandas as pd

from cascrop.src.data.utils import (
    TARGET_COMMODITIES,
    create_retry_session,
    ensure_directory,
    format_fips,
    save_parquet,
    validate_fips,
)

logger = logging.getLogger(__name__)

# NASS Quick Stats API base
API_BASE = "https://quickstats.nass.usda.gov/api/api_GET/"

# Maximum rows the API returns in one call
API_ROW_LIMIT = 50_000

# Statistics we need
TARGET_STATS = [
    "YIELD",
    "PRODUCTION",
    "AREA PLANTED",
    "AREA HARVESTED",
]

# Condition is reported differently (weekly, categorical)
CONDITION_STATS = ["CONDITION"]


class NASSLoader:
    """Download county-level crop data from the USDA NASS Quick Stats API."""

    def __init__(self, api_key: str):
        """
        Args:
            api_key: Free API key from https://quickstats.nass.usda.gov/api
        """
        self.api_key = api_key
        self.session = create_retry_session(retries=5, backoff_factor=1.5)

    # ------------------------------------------------------------------
    # Core API query
    # ------------------------------------------------------------------

    def query(
        self,
        commodity: str,
        statistic: str,
        year: int,
        geo_level: str = "COUNTY",
        extra_params: Optional[Dict[str, str]] = None,
    ) -> pd.DataFrame:
        """Issue a single Quick Stats API query.

        Args:
            commodity: e.g. "CORN", "SOYBEANS", "WHEAT"
            statistic: e.g. "YIELD", "PRODUCTION", "AREA PLANTED"
            year: calendar year
            geo_level: geographic aggregation level
            extra_params: additional API parameters

        Returns:
            DataFrame of matching records.
        """
        params = {
            "key": self.api_key,
            "commodity_desc": commodity.upper(),
            "statisticcat_desc": statistic.upper(),
            "year": str(year),
            "agg_level_desc": geo_level.upper(),
            "format": "JSON",
        }
        if extra_params:
            params.update(extra_params)

        logger.debug(
            "NASS query: commodity=%s stat=%s year=%d geo=%s",
            commodity, statistic, year, geo_level,
        )

        try:
            resp = self.session.get(API_BASE, params=params, timeout=120)
            resp.raise_for_status()
        except Exception as exc:
            logger.error("NASS API request failed: %s", exc)
            return pd.DataFrame()

        data = resp.json()

        if "data" not in data:
            # API returns error messages in different fields
            error_msg = data.get("error", [str(data)])
            logger.warning("NASS API returned no data: %s", error_msg)
            return pd.DataFrame()

        df = pd.DataFrame(data["data"])
        logger.debug("  -> %d rows returned", len(df))
        return df

    # ------------------------------------------------------------------
    # Batch download: production data
    # ------------------------------------------------------------------

    def download_production_data(
        self,
        years: Union[range, List[int]],
        commodities: Optional[List[str]] = None,
        output_dir: Optional[Path] = None,
    ) -> pd.DataFrame:
        """Download yield, production, area planted, and area harvested
        for target commodities across all years.

        Args:
            years: Years to query.
            commodities: List of commodity names. Defaults to TARGET_COMMODITIES.
            output_dir: If provided, saves per-year parquet files here.

        Returns:
            Combined DataFrame of all results.
        """
        if commodities is None:
            commodities = list(TARGET_COMMODITIES)

        if output_dir:
            output_dir = ensure_directory(output_dir)

        all_frames = []

        for year in years:
            year_frames = []
            for commodity in commodities:
                for stat in TARGET_STATS:
                    # Check cache
                    if output_dir:
                        cache_file = (
                            output_dir
                            / f"{commodity.lower()}_{stat.lower().replace(' ', '_')}_{year}.parquet"
                        )
                        if cache_file.exists():
                            logger.info("Cache hit: %s", cache_file.name)
                            year_frames.append(pd.read_parquet(cache_file))
                            continue

                    df = self.query(commodity, stat, year)
                    if df.empty:
                        continue

                    # Clean the result
                    df = self._clean_api_result(df, commodity, stat, year)
                    year_frames.append(df)

                    if output_dir:
                        df.to_parquet(cache_file, index=False)

                    # Respect rate limits — NASS allows ~100 requests/min
                    time.sleep(0.7)

            if year_frames:
                year_df = pd.concat(year_frames, ignore_index=True)
                all_frames.append(year_df)
                logger.info("Year %d: %d total records", year, len(year_df))

        if not all_frames:
            logger.warning("No NASS data downloaded")
            return pd.DataFrame()

        combined = pd.concat(all_frames, ignore_index=True)
        logger.info("Total NASS records: %d", len(combined))
        return combined

    def _clean_api_result(
        self, df: pd.DataFrame, commodity: str, stat: str, year: int
    ) -> pd.DataFrame:
        """Standardize a single API response into our schema."""
        result = pd.DataFrame()

        # Build 5-digit FIPS
        if "state_fips_code" in df.columns and "county_code" in df.columns:
            result["fips"] = df.apply(
                lambda r: format_fips(
                    r.get("state_fips_code", "00"),
                    r.get("county_code", "000"),
                ),
                axis=1,
            )
        else:
            # Some responses use different column names
            result["fips"] = "00000"

        result["state_code"] = df.get("state_fips_code", "00").astype(str).str.zfill(2)
        result["county_name"] = df.get("county_name", "")
        result["commodity"] = commodity.upper()
        result["statistic"] = stat.upper()
        result["year"] = year

        # Value field — NASS uses commas in numbers and (D) for suppressed
        if "Value" in df.columns:
            val = df["Value"].astype(str).str.replace(",", "", regex=False)
            val = val.str.strip()
            val = val.replace({"(D)": None, "(Z)": None, "(NA)": None, "": None})
            result["value"] = pd.to_numeric(val, errors="coerce")
        elif "value" in df.columns:
            val = df["value"].astype(str).str.replace(",", "", regex=False)
            val = val.replace({"(D)": None, "(Z)": None, "(NA)": None, "": None})
            result["value"] = pd.to_numeric(val, errors="coerce")
        else:
            result["value"] = None

        # Unit
        result["unit"] = df.get("unit_desc", "")

        # Reference period (useful for sub-annual stats like CONDITION)
        result["reference_period"] = df.get("reference_period_desc", "YEAR")

        return result

    # ------------------------------------------------------------------
    # Waste proxy calculation
    # ------------------------------------------------------------------

    def calculate_waste_proxy(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute abandoned fraction: (area_planted - area_harvested) / area_planted.

        This proxy captures fields that were planted but never harvested,
        which is a strong signal of crop waste.

        Args:
            df: DataFrame from download_production_data() with columns
                fips, commodity, year, statistic, value.

        Returns:
            DataFrame with columns: fips, commodity, year, area_planted,
            area_harvested, abandoned_fraction.
        """
        # Pivot so each stat is a column
        planted = df[df["statistic"] == "AREA PLANTED"][
            ["fips", "commodity", "year", "value"]
        ].rename(columns={"value": "area_planted"})

        harvested = df[df["statistic"] == "AREA HARVESTED"][
            ["fips", "commodity", "year", "value"]
        ].rename(columns={"value": "area_harvested"})

        merged = planted.merge(
            harvested, on=["fips", "commodity", "year"], how="inner"
        )

        # Calculate abandoned fraction; clamp to [0, 1]
        merged["abandoned_fraction"] = (
            (merged["area_planted"] - merged["area_harvested"])
            / merged["area_planted"]
        ).clip(0.0, 1.0)

        # Handle zero-planted edge case
        merged.loc[merged["area_planted"] == 0, "abandoned_fraction"] = 0.0
        merged["abandoned_fraction"] = merged["abandoned_fraction"].fillna(0.0)

        logger.info(
            "Waste proxy: mean abandoned fraction = %.4f across %d observations",
            merged["abandoned_fraction"].mean(),
            len(merged),
        )
        return merged

    # ------------------------------------------------------------------
    # Crop condition data
    # ------------------------------------------------------------------

    def download_condition_data(
        self,
        years: Union[range, List[int]],
        commodities: Optional[List[str]] = None,
        output_dir: Optional[Path] = None,
    ) -> pd.DataFrame:
        """Download weekly crop condition reports (Good/Excellent %, etc.).

        Condition is reported at STATE level (not county). We'll use it
        as a state-level feature downstream.
        """
        if commodities is None:
            commodities = list(TARGET_COMMODITIES)
        if output_dir:
            output_dir = ensure_directory(output_dir)

        frames = []
        for year in years:
            for commodity in commodities:
                cache = None
                if output_dir:
                    cache = output_dir / f"condition_{commodity.lower()}_{year}.parquet"
                    if cache.exists():
                        frames.append(pd.read_parquet(cache))
                        continue

                df = self.query(
                    commodity,
                    "CONDITION",
                    year,
                    geo_level="STATE",
                    extra_params={"unit_desc": "PCT OF ACREAGE"},
                )
                if df.empty:
                    time.sleep(0.5)
                    continue

                df = self._clean_condition(df, commodity, year)
                frames.append(df)

                if cache:
                    df.to_parquet(cache, index=False)
                time.sleep(0.7)

        if not frames:
            return pd.DataFrame()

        return pd.concat(frames, ignore_index=True)

    def _clean_condition(
        self, df: pd.DataFrame, commodity: str, year: int
    ) -> pd.DataFrame:
        """Parse crop condition categories into a pivot-ready frame."""
        result = pd.DataFrame()
        result["state_code"] = df.get("state_fips_code", "00").astype(str).str.zfill(2)
        result["commodity"] = commodity.upper()
        result["year"] = year

        # Week number from reference_period_desc (e.g. "WEEK #20")
        week_str = df.get("reference_period_desc", "").astype(str)
        result["week"] = pd.to_numeric(
            week_str.str.extract(r"(\d+)", expand=False), errors="coerce"
        )

        # Condition category (EXCELLENT, GOOD, FAIR, POOR, VERY POOR)
        result["condition"] = df.get("domaincat_desc", "").astype(str).str.strip()
        if result["condition"].str.len().sum() == 0:
            # Try unit_desc for category info
            result["condition"] = df.get("unit_desc", "")

        # Value
        val = df.get("Value", df.get("value", "")).astype(str)
        val = val.str.replace(",", "", regex=False)
        val = val.replace({"(D)": None, "": None})
        result["pct"] = pd.to_numeric(val, errors="coerce")

        return result.dropna(subset=["week", "pct"])

    # ------------------------------------------------------------------
    # CDL county-level summaries
    # ------------------------------------------------------------------

    def download_cdl_summaries(
        self,
        years: Union[range, List[int]],
        output_dir: Path,
    ) -> pd.DataFrame:
        """Download county-level crop composition from NASS Quick Stats.

        Instead of raw CDL rasters, we query the API for AREA PLANTED
        by commodity at county level, then compute each crop's share of
        total planted area. This gives us a crop composition vector per
        county-year without needing raster downloads.

        Returns:
            DataFrame with columns: fips, year, crop, area, crop_fraction.
        """
        output_dir = ensure_directory(output_dir)
        all_commodities = [
            "CORN", "SOYBEANS", "WHEAT", "COTTON", "RICE",
            "SORGHUM", "BARLEY", "OATS", "HAY",
        ]

        frames = []
        for year in years:
            cache = output_dir / f"cdl_summary_{year}.parquet"
            if cache.exists():
                frames.append(pd.read_parquet(cache))
                continue

            year_data = []
            for crop in all_commodities:
                df = self.query(crop, "AREA PLANTED", year)
                if not df.empty:
                    cleaned = self._clean_api_result(df, crop, "AREA PLANTED", year)
                    cleaned = cleaned.rename(columns={"value": "area"})
                    cleaned["crop"] = crop
                    year_data.append(cleaned[["fips", "year", "crop", "area"]])
                time.sleep(0.7)

            if year_data:
                ydf = pd.concat(year_data, ignore_index=True)
                # Compute per-county total and fraction
                totals = ydf.groupby("fips")["area"].sum().rename("total_area")
                ydf = ydf.merge(totals, on="fips", how="left")
                ydf["crop_fraction"] = (ydf["area"] / ydf["total_area"]).fillna(0)
                ydf = ydf.drop(columns=["total_area"])
                ydf.to_parquet(cache, index=False)
                frames.append(ydf)
                logger.info("CDL summary year %d: %d records", year, len(ydf))

        if not frames:
            return pd.DataFrame()

        return pd.concat(frames, ignore_index=True)

    # ------------------------------------------------------------------
    # Pivot to wide format
    # ------------------------------------------------------------------

    @staticmethod
    def pivot_to_features(df: pd.DataFrame) -> pd.DataFrame:
        """Pivot long-form NASS data into a wide feature matrix.

        Each row becomes one county-commodity-year observation, with columns
        for each statistic: yield, production, area_planted, area_harvested.
        """
        if df.empty:
            return df

        pivot = df.pivot_table(
            index=["fips", "commodity", "year"],
            columns="statistic",
            values="value",
            aggfunc="mean",
        ).reset_index()

        # Flatten column names
        pivot.columns = [
            c.lower().replace(" ", "_") if isinstance(c, str) else c
            for c in pivot.columns
        ]

        return pivot

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate(self, df: pd.DataFrame) -> None:
        """Run basic integrity checks on downloaded NASS data."""
        assert len(df) > 0, "DataFrame is empty"

        if "fips" in df.columns:
            validate_fips(df, "fips")

        # Check for reasonable value ranges
        if "value" in df.columns:
            negatives = (df["value"] < 0).sum()
            if negatives > 0:
                logger.warning("%d negative values found", negatives)

        # Commodity coverage
        if "commodity" in df.columns:
            comms = set(df["commodity"].unique())
            for target in TARGET_COMMODITIES:
                assert target in comms, f"Missing commodity: {target}"

        logger.info("NASS validation passed: %d rows", len(df))

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    def save(self, df: pd.DataFrame, output_path: Path) -> None:
        """Save NASS data as Parquet."""
        save_parquet(df, output_path)
