"""
USDA VegScape vegetation condition index loader.

VegScape provides weekly NDVI-based vegetation condition data at 250m
resolution, derived from MODIS satellite imagery. We download county-level
summary statistics and aggregate to monthly means for use as features.

Data source: https://nassgeodata.gmu.edu/VegScape/
Alternative: CropScape CDL + MODIS NDVI

Typical usage:
    loader = VegScapeLoader()
    loader.download(years=range(2008, 2025), output_dir=Path("data/raw/vegscape"))
    df = loader.aggregate_to_county_month(Path("data/raw/vegscape"))
    loader.validate(df)
"""

import io
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd

from .utils import (
    STATE_FIPS_TO_NAME,
    create_retry_session,
    ensure_directory,
    format_fips,
    save_parquet,
    validate_fips,
)

logger = logging.getLogger(__name__)

# VegScape API / download endpoints
VEGSCAPE_BASE_URL = "https://nassgeodata.gmu.edu/VegScape"

# MODIS NDVI product (MOD13Q1) — 250m, 16-day composites
# Used as fallback when VegScape API is unavailable
MODIS_NDVI_COLLECTION = "MODIS/061/MOD13Q1"

# Week-to-month mapping: approximate DOY ranges per month
MONTH_DOY_RANGES = {
    1: (1, 31), 2: (32, 59), 3: (60, 90), 4: (91, 120),
    5: (121, 151), 6: (152, 181), 7: (182, 212), 8: (213, 243),
    9: (244, 273), 10: (274, 304), 11: (305, 334), 12: (335, 365),
}


class VegScapeLoader:
    """Download and aggregate USDA VegScape vegetation condition data."""

    def __init__(self):
        self.session = create_retry_session(retries=5, backoff_factor=2.0)

    # ------------------------------------------------------------------
    # Download
    # ------------------------------------------------------------------

    def download(
        self,
        years: Union[range, List[int]],
        output_dir: Path,
    ) -> List[Path]:
        """Download VegScape county-level NDVI summaries.

        VegScape provides a WMS/WCS interface. We query county-level
        summary statistics for each 16-day period. If the VegScape
        API is unavailable, we fall back to downloading MODIS NDVI
        data via Google Earth Engine.

        Args:
            years: Calendar years to download.
            output_dir: Local directory for caching files.

        Returns:
            List of saved file paths.
        """
        output_dir = ensure_directory(output_dir)
        saved_files = []

        for year in years:
            cache = output_dir / f"vegscape_{year}.parquet"
            if cache.exists():
                logger.info("Cache hit: %s", cache.name)
                saved_files.append(cache)
                continue

            logger.info("Downloading VegScape data for %d", year)
            year_data = self._download_year_vegscape_api(year)

            if year_data is None or year_data.empty:
                logger.info("VegScape API unavailable for %d, trying MODIS fallback", year)
                year_data = self._download_year_modis_fallback(year)

            if year_data is not None and not year_data.empty:
                year_data.to_parquet(cache, index=False)
                saved_files.append(cache)
                logger.info("  Year %d: %d records saved", year, len(year_data))
            else:
                logger.warning("No VegScape/MODIS data available for %d", year)

        return saved_files

    def _download_year_vegscape_api(self, year: int) -> Optional[pd.DataFrame]:
        """Attempt to download from VegScape WCS/API for one year.

        The VegScape service provides county-level vegetation condition
        as percent of normal NDVI. We query each state for 16-day periods
        throughout the growing season.
        """
        records = []

        # VegScape data URLs — try the county statistics endpoint
        for week_start_doy in range(1, 366, 16):
            date_str = f"{year}{week_start_doy:03d}"

            # VegScape county stats URL pattern
            url = (
                f"{VEGSCAPE_BASE_URL}/servlet/VegScapeServlet?"
                f"type=countystat&date={date_str}"
            )

            try:
                resp = self.session.get(url, timeout=60)
                if resp.status_code != 200:
                    continue

                # Parse CSV response
                try:
                    df = pd.read_csv(io.StringIO(resp.text))
                except Exception:
                    # Try JSON
                    try:
                        data = resp.json()
                        if isinstance(data, list):
                            df = pd.DataFrame(data)
                        elif isinstance(data, dict) and "data" in data:
                            df = pd.DataFrame(data["data"])
                        else:
                            continue
                    except Exception:
                        continue

                if df.empty:
                    continue

                df["year"] = year
                df["doy"] = week_start_doy
                records.append(df)

            except Exception as exc:
                logger.debug("VegScape API request failed: %s", exc)

            time.sleep(0.3)

        if not records:
            return None

        combined = pd.concat(records, ignore_index=True)
        return self._standardize_vegscape(combined)

    def _download_year_modis_fallback(self, year: int) -> Optional[pd.DataFrame]:
        """Fall back to MODIS NDVI via Google Earth Engine when VegScape
        API is unreachable.

        Computes county-level mean and median NDVI from MOD13Q1 (250m,
        16-day composites) for each month.
        """
        try:
            import ee
            ee.Initialize()
        except Exception as exc:
            logger.warning("Earth Engine unavailable for MODIS fallback: %s", exc)
            return self._download_year_modis_appeeears(year)

        records = []
        counties = ee.FeatureCollection("TIGER/2018/Counties")

        for month in range(1, 13):
            start = f"{year}-{month:02d}-01"
            if month == 12:
                end = f"{year + 1}-01-01"
            else:
                end = f"{year}-{month + 1:02d}-01"

            try:
                ndvi_collection = (
                    ee.ImageCollection(MODIS_NDVI_COLLECTION)
                    .filterDate(start, end)
                    .select("NDVI")
                )

                if ndvi_collection.size().getInfo() == 0:
                    continue

                # Monthly median composite, scaled to [-1, 1]
                composite = ndvi_collection.median().multiply(0.0001)

                # Reduce to county means
                def _reduce(feature):
                    stats = composite.reduceRegion(
                        reducer=ee.Reducer.mean().combine(
                            ee.Reducer.stdDev(), sharedInputs=True
                        ),
                        geometry=feature.geometry(),
                        scale=250,
                        maxPixels=1e8,
                        bestEffort=True,
                    )
                    return feature.set(stats).set(
                        "fips", feature.get("GEOID")
                    ).set("year", year).set("month", month)

                result = counties.map(_reduce)
                info = result.getInfo()

                for feat in info.get("features", []):
                    props = feat.get("properties", {})
                    records.append({
                        "fips": str(props.get("fips", "")).zfill(5),
                        "year": year,
                        "month": month,
                        "ndvi_mean": props.get("NDVI_mean", props.get("NDVI")),
                        "ndvi_std": props.get("NDVI_stdDev"),
                        "vegscape_condition": None,  # Not available from MODIS directly
                    })

                logger.info("  MODIS fallback: %d-%02d extracted", year, month)

            except Exception as exc:
                logger.error("MODIS extraction failed for %d-%02d: %s", year, month, exc)

            time.sleep(1.0)

        if not records:
            return None

        return pd.DataFrame(records)

    def _download_year_modis_appeeears(self, year: int) -> Optional[pd.DataFrame]:
        """Last-resort fallback: download MODIS NDVI county stats from
        NASA AppEEARS API.

        This works without Google Earth Engine but requires a NASA
        Earthdata login.
        """
        logger.info("Attempting AppEEARS MODIS download for %d", year)

        # AppEEARS API base
        base = "https://appeears.earthdatacloud.nasa.gov/api"

        # This would require an authenticated session with Earthdata.
        # For now, return None and let downstream handle missing data.
        logger.warning(
            "AppEEARS fallback not fully automated — requires Earthdata "
            "login. Year %d VegScape data will be missing.",
            year,
        )
        return None

    def _standardize_vegscape(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize VegScape API responses into our schema."""
        result = pd.DataFrame()

        # Find FIPS column
        fips_col = None
        for cand in ["FIPS", "fips", "county_fips", "GEOID", "geoid"]:
            if cand in df.columns:
                fips_col = cand
                break

        if fips_col:
            result["fips"] = df[fips_col].astype(str).str.zfill(5)
        elif "state_code" in df.columns and "county_code" in df.columns:
            result["fips"] = df.apply(
                lambda r: format_fips(r["state_code"], r["county_code"]),
                axis=1,
            )
        else:
            logger.warning("No FIPS column found in VegScape data")
            return pd.DataFrame()

        result["year"] = df.get("year", 0)

        # DOY -> month mapping
        if "doy" in df.columns:
            result["doy"] = pd.to_numeric(df["doy"], errors="coerce")
            result["month"] = result["doy"].apply(self._doy_to_month)
        elif "month" in df.columns:
            result["month"] = df["month"]
        elif "date" in df.columns:
            result["month"] = pd.to_datetime(df["date"], errors="coerce").dt.month

        # NDVI / vegetation condition values
        for cand in ["ndvi", "NDVI", "ndvi_mean", "mean_ndvi", "vci", "VCI"]:
            if cand in df.columns:
                result["ndvi_mean"] = pd.to_numeric(df[cand], errors="coerce")
                break

        for cand in ["ndvi_std", "std", "stddev"]:
            if cand in df.columns:
                result["ndvi_std"] = pd.to_numeric(df[cand], errors="coerce")
                break

        # Vegetation condition index (percent of normal)
        for cand in ["condition", "pct_normal", "vhi", "VHI"]:
            if cand in df.columns:
                result["vegscape_condition"] = pd.to_numeric(df[cand], errors="coerce")
                break

        return result

    @staticmethod
    def _doy_to_month(doy: int) -> int:
        """Convert day-of-year to month number."""
        if pd.isna(doy):
            return 0
        doy = int(doy)
        for month, (start, end) in MONTH_DOY_RANGES.items():
            if start <= doy <= end:
                return month
        return 12  # Edge case: doy > 365

    # ------------------------------------------------------------------
    # Aggregate to county-month
    # ------------------------------------------------------------------

    def aggregate_to_county_month(
        self, raw_dir: Path
    ) -> pd.DataFrame:
        """Load all cached VegScape files and aggregate to county-month.

        Multiple 16-day composites within a month get averaged into
        a single monthly observation per county.

        Args:
            raw_dir: Directory containing vegscape_YYYY.parquet files.

        Returns:
            DataFrame with columns: fips, year, month, ndvi_mean,
            ndvi_std, vegscape_condition.
        """
        raw_dir = Path(raw_dir)
        frames = []

        for fp in sorted(raw_dir.glob("vegscape_*.parquet")):
            frames.append(pd.read_parquet(fp))

        if not frames:
            # Try MODIS fallback files
            for fp in sorted(raw_dir.glob("modis_*.parquet")):
                frames.append(pd.read_parquet(fp))

        if not frames:
            raise FileNotFoundError(f"No VegScape/MODIS files in {raw_dir}")

        df = pd.concat(frames, ignore_index=True)

        # Aggregate sub-monthly observations to monthly means
        agg_cols = {}
        if "ndvi_mean" in df.columns:
            agg_cols["ndvi_mean"] = "mean"
        if "ndvi_std" in df.columns:
            agg_cols["ndvi_std"] = "mean"
        if "vegscape_condition" in df.columns:
            agg_cols["vegscape_condition"] = "mean"

        if not agg_cols:
            logger.warning("No aggregatable columns found")
            return df

        county_month = (
            df.groupby(["fips", "year", "month"])
            .agg(agg_cols)
            .reset_index()
        )

        logger.info(
            "Aggregated VegScape to %d county-month observations",
            len(county_month),
        )
        return county_month

    # ------------------------------------------------------------------
    # Compute derived vegetation features
    # ------------------------------------------------------------------

    @staticmethod
    def compute_anomalies(df: pd.DataFrame) -> pd.DataFrame:
        """Compute NDVI anomalies relative to the county-month climatology.

        For each county and calendar month, we calculate the long-term
        mean NDVI, then express each observation as a deviation.

        New columns:
        - ndvi_climatology: long-term average NDVI for this county-month
        - ndvi_anomaly: ndvi_mean - ndvi_climatology
        - ndvi_zscore: standardized anomaly
        """
        df = df.copy()

        if "ndvi_mean" not in df.columns:
            return df

        # Climatology = mean across all years for each county-month
        clim = (
            df.groupby(["fips", "month"])["ndvi_mean"]
            .agg(["mean", "std"])
            .rename(columns={"mean": "ndvi_climatology", "std": "ndvi_clim_std"})
            .reset_index()
        )

        df = df.merge(clim, on=["fips", "month"], how="left")
        df["ndvi_anomaly"] = df["ndvi_mean"] - df["ndvi_climatology"]
        df["ndvi_zscore"] = np.where(
            df["ndvi_clim_std"] > 0,
            df["ndvi_anomaly"] / df["ndvi_clim_std"],
            0.0,
        )

        return df

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate(self, df: pd.DataFrame) -> None:
        """Basic integrity checks on VegScape data."""
        assert len(df) > 0, "VegScape DataFrame is empty"

        if "fips" in df.columns:
            validate_fips(df, "fips")

        if "ndvi_mean" in df.columns:
            valid_ndvi = df["ndvi_mean"].dropna()
            if len(valid_ndvi) > 0:
                out_of_range = ~valid_ndvi.between(-0.5, 1.0)
                if out_of_range.any():
                    pct = out_of_range.mean() * 100
                    logger.warning("%.1f%% of NDVI values outside [-0.5, 1.0]", pct)

        if "year" in df.columns:
            years = sorted(df["year"].unique())
            logger.info("VegScape years: %d - %d", years[0], years[-1])

        n_counties = df["fips"].nunique() if "fips" in df.columns else 0
        logger.info(
            "VegScape validation passed: %d rows, %d counties",
            len(df), n_counties,
        )

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    def save(self, df: pd.DataFrame, output_path: Path) -> None:
        """Save VegScape data as Parquet."""
        save_parquet(df, output_path)
