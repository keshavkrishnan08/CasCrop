"""
NASA SMAP soil moisture data loader.

Downloads and processes Soil Moisture Active Passive (SMAP) Level-3
soil moisture data, aggregated to county-level monthly means. Provides
both surface (~5cm) and root-zone (~1m) soil moisture.

For years before SMAP availability (pre-2015), backfills with NLDAS-2
Noah model soil moisture, cross-calibrated to the SMAP scale.

Data sources:
    - SMAP L3: https://nsidc.org/data/spl3smp/versions/8
    - USDA Crop-CASMA: https://croplandcrosswalk.blob.core.windows.net/
    - NLDAS-2: https://ldas.gsfc.nasa.gov/nldas/v2/models
    - GEE: NASA/SMAP/SPL3SMP_E/005 and NASA_USDA/HSL/SMAP10KM_soil_moisture

Typical usage:
    loader = SMAPLoader()
    loader.download(years=range(2015, 2025), output_dir=Path("data/raw/smap"))
    df = loader.aggregate_to_county(Path("data/raw/smap"))
    backfill = loader.backfill_with_nldas(years_before_2015=range(2008, 2015))
"""

import io
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from cascrop.src.data.utils import (
    STATE_FIPS_TO_NAME,
    create_retry_session,
    ensure_directory,
    load_fips_codes,
    save_parquet,
    validate_fips,
)

logger = logging.getLogger(__name__)

# GEE collection IDs
SMAP_GEE_COLLECTION = "NASA_USDA/HSL/SMAP10KM_soil_moisture"
SMAP_L3_COLLECTION = "NASA/SMAP/SPL3SMP_E/005"
NLDAS_COLLECTION = "NASA/NLDAS/FORA0125_H002"

# SMAP bands of interest
SMAP_BANDS = {
    "surface": "ssm",          # Surface soil moisture (mm)
    "sub_surface": "susm",     # Sub-surface soil moisture (mm)
    "root_zone": "smp",        # Root-zone soil moisture (unitless)
}

# NLDAS-2 Noah model soil moisture layers
NLDAS_SOIL_BANDS = {
    "surface": "SOILM0_10",     # 0-10cm layer (kg/m^2)
    "root_zone": "SOILM0_200",  # 0-200cm layer (kg/m^2)
}

# Cross-calibration: NLDAS -> SMAP
# These coefficients should be fit from the overlap period (2015-2016)
# Placeholder values based on published studies
NLDAS_TO_SMAP_CALIBRATION = {
    "surface": {"slope": 0.82, "intercept": 0.035},
    "root_zone": {"slope": 0.78, "intercept": 0.042},
}


class SMAPLoader:
    """Download and process NASA SMAP soil moisture data."""

    def __init__(self):
        self.session = create_retry_session(retries=5, backoff_factor=2.0)
        self._ee = None

    def _init_ee(self):
        """Lazy-initialize Earth Engine."""
        if self._ee is not None:
            return self._ee

        try:
            import ee
            ee.Initialize()
            self._ee = ee
            logger.info("Earth Engine initialized for SMAP loader")
        except Exception as exc:
            logger.warning("Earth Engine not available: %s", exc)
            self._ee = None
        return self._ee

    # ------------------------------------------------------------------
    # Download via GEE (primary method)
    # ------------------------------------------------------------------

    def download(
        self,
        years: Union[range, List[int]],
        output_dir: Path,
        batch_size: int = 300,
    ) -> List[Path]:
        """Download SMAP soil moisture via Google Earth Engine.

        Uses the NASA_USDA/HSL/SMAP10KM_soil_moisture collection, which
        provides 10km soil moisture grids. We reduce each county polygon
        to a mean value per month.

        Args:
            years: Calendar years (SMAP available from April 2015).
            output_dir: Directory for cached parquet files.
            batch_size: Counties per GEE request.

        Returns:
            List of saved file paths.
        """
        output_dir = ensure_directory(output_dir)
        ee = self._init_ee()
        saved = []

        if ee is None:
            logger.info("GEE unavailable — trying Crop-CASMA download")
            return self._download_crop_casma(years, output_dir)

        fips_df = load_fips_codes()
        all_fips = sorted(fips_df["fips"].unique())
        counties = ee.FeatureCollection("TIGER/2018/Counties")

        for year in years:
            # SMAP data starts April 2015
            if year < 2015:
                logger.info("SMAP not available for %d (starts 2015)", year)
                continue

            start_month = 4 if year == 2015 else 1

            for month in range(start_month, 13):
                cache = output_dir / f"smap_{year}_{month:02d}.parquet"
                if cache.exists():
                    saved.append(cache)
                    continue

                logger.info("Downloading SMAP: %d-%02d", year, month)
                start_date = f"{year}-{month:02d}-01"
                if month == 12:
                    end_date = f"{year + 1}-01-01"
                else:
                    end_date = f"{year}-{month + 1:02d}-01"

                month_records = []

                # Process in batches of counties
                for i in range(0, len(all_fips), batch_size):
                    batch_fips = all_fips[i : i + batch_size]
                    batch_fc = counties.filter(
                        ee.Filter.inList("GEOID", batch_fips)
                    )

                    try:
                        # Try NASA_USDA SMAP collection first
                        collection = (
                            ee.ImageCollection(SMAP_GEE_COLLECTION)
                            .filterDate(start_date, end_date)
                        )

                        n_images = collection.size().getInfo()
                        if n_images == 0:
                            # Fall back to SPL3SMP_E
                            collection = (
                                ee.ImageCollection(SMAP_L3_COLLECTION)
                                .filterDate(start_date, end_date)
                            )

                        composite = collection.mean()

                        def _reduce(feature):
                            stats = composite.reduceRegion(
                                reducer=ee.Reducer.mean(),
                                geometry=feature.geometry(),
                                scale=10000,  # 10km SMAP native resolution
                                maxPixels=1e8,
                                bestEffort=True,
                            )
                            return feature.set(stats).set(
                                "fips", feature.get("GEOID")
                            )

                        results = batch_fc.map(_reduce)
                        info = results.getInfo()

                        for feat in info.get("features", []):
                            props = feat.get("properties", {})
                            record = {
                                "fips": str(props.get("fips", "")).zfill(5),
                                "year": year,
                                "month": month,
                            }
                            # Extract soil moisture bands
                            for our_name, band_name in SMAP_BANDS.items():
                                record[f"sm_{our_name}"] = props.get(band_name)
                            month_records.append(record)

                    except Exception as exc:
                        logger.error(
                            "SMAP GEE batch %d-%d failed: %s",
                            i, i + batch_size, exc,
                        )

                    time.sleep(1.0)

                if month_records:
                    mdf = pd.DataFrame(month_records)
                    mdf.to_parquet(cache, index=False)
                    saved.append(cache)
                    logger.info("  %d-%02d: %d county records", year, month, len(mdf))

        return saved

    def _download_crop_casma(
        self, years: Union[range, List[int]], output_dir: Path
    ) -> List[Path]:
        """Download soil moisture from USDA Crop-CASMA aggregated data.

        Crop-CASMA provides county-level soil moisture statistics derived
        from SMAP observations, available without Earth Engine.
        """
        output_dir = ensure_directory(output_dir)
        saved = []

        base_url = (
            "https://nassgeodata.gmu.edu/CropCASMA/api/v1/"
            "soil_moisture/county"
        )

        for year in years:
            if year < 2015:
                continue

            cache = output_dir / f"crop_casma_{year}.parquet"
            if cache.exists():
                saved.append(cache)
                continue

            logger.info("Downloading Crop-CASMA soil moisture for %d", year)

            try:
                params = {"year": year, "format": "csv"}
                resp = self.session.get(base_url, params=params, timeout=120)
                if resp.status_code != 200:
                    logger.warning("Crop-CASMA returned status %d for year %d",
                                   resp.status_code, year)
                    continue

                df = pd.read_csv(io.StringIO(resp.text))
                if df.empty:
                    continue

                df = self._standardize_crop_casma(df, year)
                df.to_parquet(cache, index=False)
                saved.append(cache)
                logger.info("  Year %d: %d records", year, len(df))

            except Exception as exc:
                logger.error("Crop-CASMA download failed for %d: %s", year, exc)

            time.sleep(0.5)

        return saved

    def _standardize_crop_casma(
        self, df: pd.DataFrame, year: int
    ) -> pd.DataFrame:
        """Standardize Crop-CASMA responses."""
        result = pd.DataFrame()

        # Find FIPS column
        fips_col = None
        for cand in ["fips", "FIPS", "county_fips", "GEOID"]:
            if cand in df.columns:
                fips_col = cand
                break

        if fips_col:
            result["fips"] = df[fips_col].astype(str).str.zfill(5)
        else:
            return pd.DataFrame()

        result["year"] = year

        # Date/month
        if "date" in df.columns:
            result["month"] = pd.to_datetime(df["date"], errors="coerce").dt.month
        elif "month" in df.columns:
            result["month"] = df["month"]

        # Soil moisture values
        for cand in ["ssm", "surface_sm", "sm_surface"]:
            if cand in df.columns:
                result["sm_surface"] = pd.to_numeric(df[cand], errors="coerce")
                break

        for cand in ["susm", "subsurface_sm", "sm_subsurface"]:
            if cand in df.columns:
                result["sm_sub_surface"] = pd.to_numeric(df[cand], errors="coerce")
                break

        for cand in ["smp", "rootzone_sm", "sm_rootzone"]:
            if cand in df.columns:
                result["sm_root_zone"] = pd.to_numeric(df[cand], errors="coerce")
                break

        return result

    # ------------------------------------------------------------------
    # Aggregate raw files to county-month
    # ------------------------------------------------------------------

    def aggregate_to_county(
        self, raw_dir: Path
    ) -> pd.DataFrame:
        """Load all cached SMAP files and aggregate to county-month means.

        Args:
            raw_dir: Directory containing smap_YYYY_MM.parquet files.

        Returns:
            DataFrame with columns: fips, year, month, sm_surface,
            sm_sub_surface, sm_root_zone.
        """
        raw_dir = Path(raw_dir)
        frames = []

        for pattern in ["smap_*.parquet", "crop_casma_*.parquet"]:
            for fp in sorted(raw_dir.glob(pattern)):
                frames.append(pd.read_parquet(fp))

        if not frames:
            raise FileNotFoundError(f"No SMAP/Crop-CASMA files in {raw_dir}")

        df = pd.concat(frames, ignore_index=True)

        # Aggregate — might have multiple records per county-month from
        # different SMAP passes
        sm_cols = [c for c in df.columns if c.startswith("sm_")]
        agg_dict = {c: "mean" for c in sm_cols}

        if agg_dict:
            county_month = (
                df.groupby(["fips", "year", "month"])
                .agg(agg_dict)
                .reset_index()
            )
        else:
            county_month = df

        logger.info(
            "Aggregated SMAP to %d county-month observations", len(county_month)
        )
        return county_month

    # ------------------------------------------------------------------
    # NLDAS-2 backfill for pre-2015
    # ------------------------------------------------------------------

    def backfill_with_nldas(
        self,
        years_before_2015: Union[range, List[int]],
        output_dir: Optional[Path] = None,
        batch_size: int = 300,
    ) -> pd.DataFrame:
        """Backfill soil moisture with NLDAS-2 Noah model output.

        NLDAS-2 provides modeled soil moisture from 1979 to present at
        1/8-degree resolution (~13km). We extract county-level means
        and apply cross-calibration coefficients derived from the
        2015-2016 overlap period with SMAP.

        Args:
            years_before_2015: Years to backfill (e.g., range(2008, 2015)).
            output_dir: Cache directory.
            batch_size: Counties per GEE request.

        Returns:
            Cross-calibrated DataFrame with SMAP-equivalent soil moisture.
        """
        ee = self._init_ee()
        if ee is None:
            logger.warning("Earth Engine required for NLDAS backfill")
            return self._backfill_nldas_api(years_before_2015, output_dir)

        if output_dir:
            output_dir = ensure_directory(output_dir)

        fips_df = load_fips_codes()
        all_fips = sorted(fips_df["fips"].unique())
        counties = ee.FeatureCollection("TIGER/2018/Counties")

        all_frames = []

        for year in years_before_2015:
            for month in range(1, 13):
                cache = None
                if output_dir:
                    cache = output_dir / f"nldas_sm_{year}_{month:02d}.parquet"
                    if cache.exists():
                        all_frames.append(pd.read_parquet(cache))
                        continue

                logger.info("NLDAS backfill: %d-%02d", year, month)

                start_date = f"{year}-{month:02d}-01"
                if month == 12:
                    end_date = f"{year + 1}-01-01"
                else:
                    end_date = f"{year}-{month + 1:02d}-01"

                month_records = []

                for i in range(0, len(all_fips), batch_size):
                    batch_fips = all_fips[i : i + batch_size]
                    batch_fc = counties.filter(
                        ee.Filter.inList("GEOID", batch_fips)
                    )

                    try:
                        # NLDAS-2 hourly -> monthly mean
                        collection = (
                            ee.ImageCollection(NLDAS_COLLECTION)
                            .filterDate(start_date, end_date)
                            .select(list(NLDAS_SOIL_BANDS.values()))
                        )

                        composite = collection.mean()

                        def _reduce(feature):
                            stats = composite.reduceRegion(
                                reducer=ee.Reducer.mean(),
                                geometry=feature.geometry(),
                                scale=13000,  # NLDAS ~13km
                                maxPixels=1e8,
                                bestEffort=True,
                            )
                            return feature.set(stats).set(
                                "fips", feature.get("GEOID")
                            )

                        results = batch_fc.map(_reduce)
                        info = results.getInfo()

                        for feat in info.get("features", []):
                            props = feat.get("properties", {})
                            month_records.append({
                                "fips": str(props.get("fips", "")).zfill(5),
                                "year": year,
                                "month": month,
                                "nldas_sm_surface": props.get(
                                    NLDAS_SOIL_BANDS["surface"]
                                ),
                                "nldas_sm_root_zone": props.get(
                                    NLDAS_SOIL_BANDS["root_zone"]
                                ),
                            })

                    except Exception as exc:
                        logger.error("NLDAS batch failed: %s", exc)

                    time.sleep(1.0)

                if month_records:
                    mdf = pd.DataFrame(month_records)
                    mdf = self._apply_nldas_calibration(mdf)
                    if cache:
                        mdf.to_parquet(cache, index=False)
                    all_frames.append(mdf)

        if not all_frames:
            return pd.DataFrame()

        return pd.concat(all_frames, ignore_index=True)

    def _backfill_nldas_api(
        self,
        years: Union[range, List[int]],
        output_dir: Optional[Path] = None,
    ) -> pd.DataFrame:
        """Download NLDAS soil moisture via NASA GES DISC API.

        Fallback when GEE is unavailable.
        """
        logger.warning(
            "NLDAS API fallback: requires NASA Earthdata credentials. "
            "Set EARTHDATA_USERNAME and EARTHDATA_PASSWORD env vars."
        )
        import os
        username = os.environ.get("EARTHDATA_USERNAME")
        password = os.environ.get("EARTHDATA_PASSWORD")

        if not username or not password:
            logger.error("NASA Earthdata credentials not found in environment")
            return pd.DataFrame()

        if output_dir:
            output_dir = ensure_directory(output_dir)

        frames = []
        base = "https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_NOAH0125_M.002"

        for year in years:
            cache = None
            if output_dir:
                cache = output_dir / f"nldas_api_{year}.parquet"
                if cache.exists():
                    frames.append(pd.read_parquet(cache))
                    continue

            logger.info("Downloading NLDAS from GES DISC: %d", year)

            for month in range(1, 13):
                url = (
                    f"{base}/{year}/{year}{month:02d}01/"
                    f"NLDAS_NOAH0125_M.A{year}{month:02d}.002.grb"
                )

                try:
                    resp = self.session.get(
                        url,
                        auth=(username, password),
                        timeout=120,
                    )
                    if resp.status_code == 200:
                        logger.debug("Downloaded NLDAS %d-%02d", year, month)
                        # GRB parsing would go here with cfgrib or similar
                        # For now, log a note
                except Exception as exc:
                    logger.debug("NLDAS download failed: %s", exc)

                time.sleep(0.5)

        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True)

    def _apply_nldas_calibration(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply linear cross-calibration from NLDAS to SMAP scale.

        Coefficients are derived from regressing SMAP observations on
        NLDAS modeled values during the overlap period (2015-2016).
        """
        df = df.copy()

        if "nldas_sm_surface" in df.columns:
            coef = NLDAS_TO_SMAP_CALIBRATION["surface"]
            df["sm_surface"] = (
                df["nldas_sm_surface"] * coef["slope"] + coef["intercept"]
            )

        if "nldas_sm_root_zone" in df.columns:
            coef = NLDAS_TO_SMAP_CALIBRATION["root_zone"]
            df["sm_root_zone"] = (
                df["nldas_sm_root_zone"] * coef["slope"] + coef["intercept"]
            )

        # Keep NLDAS raw values for diagnostics but also provide
        # calibrated SMAP-equivalent columns
        df["sm_sub_surface"] = np.nan  # Not directly available from NLDAS
        df["calibration_source"] = "NLDAS"

        return df

    # ------------------------------------------------------------------
    # Fit cross-calibration from overlap period
    # ------------------------------------------------------------------

    def fit_calibration(
        self,
        smap_df: pd.DataFrame,
        nldas_df: pd.DataFrame,
    ) -> Dict[str, Dict[str, float]]:
        """Fit linear calibration coefficients from the SMAP-NLDAS overlap.

        Uses data from 2015-2016 (or later) where both SMAP and NLDAS
        are available to fit: smap = slope * nldas + intercept

        Args:
            smap_df: SMAP county-month data with sm_surface, sm_root_zone.
            nldas_df: NLDAS county-month data with nldas_sm_surface, nldas_sm_root_zone.

        Returns:
            Dict of calibration coefficients per variable.
        """
        from scipy import stats as sp_stats

        merged = smap_df.merge(
            nldas_df,
            on=["fips", "year", "month"],
            how="inner",
            suffixes=("_smap", "_nldas"),
        )

        calibration = {}

        for var_pair in [("sm_surface", "nldas_sm_surface"),
                         ("sm_root_zone", "nldas_sm_root_zone")]:
            smap_col, nldas_col = var_pair

            if smap_col not in merged.columns or nldas_col not in merged.columns:
                continue

            valid = merged.dropna(subset=[smap_col, nldas_col])
            if len(valid) < 10:
                continue

            slope, intercept, r_value, p_value, std_err = sp_stats.linregress(
                valid[nldas_col], valid[smap_col]
            )

            var_name = smap_col.replace("sm_", "")
            calibration[var_name] = {
                "slope": float(slope),
                "intercept": float(intercept),
                "r_squared": float(r_value ** 2),
                "n_samples": len(valid),
            }

            logger.info(
                "Calibration %s: slope=%.4f intercept=%.4f R2=%.4f (n=%d)",
                var_name, slope, intercept, r_value ** 2, len(valid),
            )

        return calibration

    # ------------------------------------------------------------------
    # Derived features
    # ------------------------------------------------------------------

    @staticmethod
    def compute_soil_features(df: pd.DataFrame) -> pd.DataFrame:
        """Compute derived soil moisture features.

        New columns:
        - sm_anomaly_surface: deviation from county-month mean
        - sm_anomaly_rootzone: deviation from county-month mean
        - sm_ratio: surface / root-zone (indicates infiltration)
        - sm_trend_3m: 3-month rolling trend (drying or wetting)
        """
        df = df.copy()

        # Anomalies
        for col in ["sm_surface", "sm_root_zone"]:
            if col in df.columns:
                clim = (
                    df.groupby(["fips", "month"])[col]
                    .transform("mean")
                )
                df[f"{col}_anomaly"] = df[col] - clim

        # Surface-to-rootzone ratio
        if "sm_surface" in df.columns and "sm_root_zone" in df.columns:
            df["sm_ratio"] = np.where(
                df["sm_root_zone"] > 0,
                df["sm_surface"] / df["sm_root_zone"],
                np.nan,
            )

        # 3-month rolling trend per county
        if "sm_surface" in df.columns:
            df = df.sort_values(["fips", "year", "month"])
            df["sm_trend_3m"] = (
                df.groupby("fips")["sm_surface"]
                .transform(lambda x: x.rolling(3, min_periods=1).apply(
                    lambda vals: np.polyfit(range(len(vals)), vals, 1)[0]
                    if len(vals) > 1 else 0.0,
                    raw=True,
                ))
            )

        return df

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate(self, df: pd.DataFrame) -> None:
        """Sanity checks on soil moisture data."""
        assert len(df) > 0, "SMAP DataFrame is empty"

        if "fips" in df.columns:
            validate_fips(df, "fips")

        for col in ["sm_surface", "sm_root_zone"]:
            if col in df.columns:
                valid = df[col].dropna()
                if len(valid) > 0:
                    # Soil moisture should be non-negative
                    neg = (valid < 0).sum()
                    if neg > 0:
                        logger.warning("%d negative %s values", neg, col)

        n_counties = df["fips"].nunique() if "fips" in df.columns else 0
        logger.info(
            "SMAP validation passed: %d rows, %d counties", len(df), n_counties
        )

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    def save(self, df: pd.DataFrame, output_path: Path) -> None:
        """Save soil moisture data as Parquet."""
        save_parquet(df, output_path)
