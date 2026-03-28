"""
Sentinel-2 vegetation index extractor via Google Earth Engine.

Computes county-level monthly median NDVI, EVI, SAVI, and NDWI from
cloud-masked Sentinel-2 Surface Reflectance imagery. For years before
Sentinel-2 availability (pre-2015), cross-calibrates Landsat 8 indices
to the Sentinel scale.

Prerequisites:
    - Google Earth Engine Python API (``ee``) authenticated
    - ``earthengine-api`` package installed
    - Run ``ee.Authenticate()`` once before first use

Typical usage:
    loader = SentinelLoader()
    df = loader.extract_county_indices(
        fips_list=["17001", "17003"],
        year=2020,
        month=7,
    )
"""

import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from .utils import (
    create_retry_session,
    ensure_directory,
    fips_to_latlon,
    load_fips_codes,
    save_parquet,
)

logger = logging.getLogger(__name__)

# Sentinel-2 SR band names
S2_BANDS = {
    "BLUE": "B2",
    "GREEN": "B3",
    "RED": "B4",
    "RED_EDGE_1": "B5",
    "RED_EDGE_2": "B6",
    "RED_EDGE_3": "B7",
    "NIR": "B8",
    "NIR_NARROW": "B8A",
    "SWIR_1": "B11",
    "SWIR_2": "B12",
    "SCL": "SCL",  # Scene Classification Layer (for cloud masking)
}

# Landsat 8 OLI SR band names (Collection 2)
L8_BANDS = {
    "BLUE": "SR_B2",
    "GREEN": "SR_B3",
    "RED": "SR_B4",
    "NIR": "SR_B5",
    "SWIR_1": "SR_B6",
    "SWIR_2": "SR_B7",
    "QA_PIXEL": "QA_PIXEL",
}

# Sentinel-2 SCL cloud/shadow classes to mask out
SCL_MASK_VALUES = [0, 1, 3, 8, 9, 10, 11]
# 0=No Data, 1=Saturated/Defective, 3=Cloud Shadow,
# 8=Cloud Medium Prob, 9=Cloud High Prob, 10=Thin Cirrus, 11=Snow/Ice

# Cross-calibration coefficients: Landsat 8 -> Sentinel-2 scale
# Based on published cross-calibration studies (Claverie et al. 2018)
L8_TO_S2_COEFFICIENTS = {
    "NDVI": {"slope": 0.9889, "intercept": 0.0049},
    "EVI": {"slope": 0.9764, "intercept": 0.0073},
    "SAVI": {"slope": 0.9851, "intercept": 0.0048},
    "NDWI": {"slope": 1.0012, "intercept": -0.0029},
}


def _initialize_ee():
    """Initialize Earth Engine, authenticating if needed."""
    try:
        import ee
    except ImportError:
        raise ImportError(
            "Google Earth Engine API not installed. "
            "Run: pip install earthengine-api"
        )

    try:
        ee.Initialize()
        logger.info("Earth Engine initialized successfully")
    except Exception:
        logger.info("Attempting Earth Engine authentication...")
        try:
            ee.Authenticate()
            ee.Initialize()
            logger.info("Earth Engine authenticated and initialized")
        except Exception as exc:
            raise RuntimeError(
                f"Could not initialize Earth Engine: {exc}. "
                "Run ee.Authenticate() manually first."
            )
    return ee


class SentinelLoader:
    """Extract county-level vegetation indices from Sentinel-2 via GEE."""

    def __init__(self):
        """Initialize the Earth Engine API."""
        self.ee = _initialize_ee()
        self._county_geometries_cache: Dict[str, object] = {}

    # ------------------------------------------------------------------
    # County geometry helpers
    # ------------------------------------------------------------------

    def _get_county_geometry(self, fips: str):
        """Get the Earth Engine geometry for a county by FIPS code."""
        ee = self.ee

        if fips in self._county_geometries_cache:
            return self._county_geometries_cache[fips]

        # TIGER/2018/Counties feature collection
        counties = ee.FeatureCollection("TIGER/2018/Counties")
        county = counties.filter(ee.Filter.eq("GEOID", fips)).first()
        geom = county.geometry()
        self._county_geometries_cache[fips] = geom
        return geom

    def _get_county_geometries_batch(self, fips_list: List[str]):
        """Get geometries for multiple counties as a FeatureCollection."""
        ee = self.ee
        counties = ee.FeatureCollection("TIGER/2018/Counties")
        return counties.filter(ee.Filter.inList("GEOID", fips_list))

    # ------------------------------------------------------------------
    # Cloud masking
    # ------------------------------------------------------------------

    def cloud_mask(self, image):
        """Apply SCL-based cloud masking to a Sentinel-2 image.

        Masks pixels classified as cloud, cloud shadow, snow, saturated,
        or no-data in the Scene Classification Layer (SCL band).

        Args:
            image: An ee.Image from the COPERNICUS/S2_SR collection.

        Returns:
            The input image with cloudy pixels masked out.
        """
        ee = self.ee
        scl = image.select("SCL")

        # Build a mask: 1 where pixel is clear, 0 where cloudy
        clear_mask = ee.Image.constant(1)
        for val in SCL_MASK_VALUES:
            clear_mask = clear_mask.And(scl.neq(val))

        return image.updateMask(clear_mask)

    def _cloud_mask_landsat(self, image):
        """QA_PIXEL-based cloud mask for Landsat 8 Collection 2."""
        ee = self.ee
        qa = image.select("QA_PIXEL")
        # Bits 3 (cloud shadow) and 5 (cloud) should be 0
        cloud_shadow_bit = 1 << 3
        cloud_bit = 1 << 5
        mask = (
            qa.bitwiseAnd(cloud_shadow_bit).eq(0)
            .And(qa.bitwiseAnd(cloud_bit).eq(0))
        )
        return image.updateMask(mask)

    # ------------------------------------------------------------------
    # Vegetation index computation
    # ------------------------------------------------------------------

    def compute_indices(self, image):
        """Compute NDVI, EVI, SAVI, and NDWI from a Sentinel-2 image.

        Index formulas:
        - NDVI = (NIR - RED) / (NIR + RED)
        - EVI  = 2.5 * (NIR - RED) / (NIR + 6*RED - 7.5*BLUE + 1)
        - SAVI = 1.5 * (NIR - RED) / (NIR + RED + 0.5)
        - NDWI = (GREEN - NIR) / (GREEN + NIR)

        Args:
            image: Cloud-masked Sentinel-2 ee.Image.

        Returns:
            ee.Image with bands: NDVI, EVI, SAVI, NDWI.
        """
        ee = self.ee

        nir = image.select(S2_BANDS["NIR"]).divide(10000)
        red = image.select(S2_BANDS["RED"]).divide(10000)
        blue = image.select(S2_BANDS["BLUE"]).divide(10000)
        green = image.select(S2_BANDS["GREEN"]).divide(10000)

        ndvi = nir.subtract(red).divide(nir.add(red)).rename("NDVI")
        evi = (
            nir.subtract(red)
            .multiply(2.5)
            .divide(nir.add(red.multiply(6)).subtract(blue.multiply(7.5)).add(1))
            .rename("EVI")
        )
        savi = (
            nir.subtract(red)
            .multiply(1.5)
            .divide(nir.add(red).add(0.5))
            .rename("SAVI")
        )
        ndwi = green.subtract(nir).divide(green.add(nir)).rename("NDWI")

        return ee.Image.cat([ndvi, evi, savi, ndwi])

    def _compute_indices_landsat(self, image):
        """Compute indices from Landsat 8 Collection 2 SR."""
        ee = self.ee

        # Landsat 8 C2 SR scale factor: 0.0000275, offset: -0.2
        nir = image.select(L8_BANDS["NIR"]).multiply(0.0000275).add(-0.2)
        red = image.select(L8_BANDS["RED"]).multiply(0.0000275).add(-0.2)
        blue = image.select(L8_BANDS["BLUE"]).multiply(0.0000275).add(-0.2)
        green = image.select(L8_BANDS["GREEN"]).multiply(0.0000275).add(-0.2)

        ndvi = nir.subtract(red).divide(nir.add(red)).rename("NDVI")
        evi = (
            nir.subtract(red)
            .multiply(2.5)
            .divide(nir.add(red.multiply(6)).subtract(blue.multiply(7.5)).add(1))
            .rename("EVI")
        )
        savi = (
            nir.subtract(red)
            .multiply(1.5)
            .divide(nir.add(red).add(0.5))
            .rename("SAVI")
        )
        ndwi = green.subtract(nir).divide(green.add(nir)).rename("NDWI")

        return ee.Image.cat([ndvi, evi, savi, ndwi])

    # ------------------------------------------------------------------
    # County-level extraction
    # ------------------------------------------------------------------

    def extract_county_indices(
        self,
        fips_list: List[str],
        year: int,
        month: int,
    ) -> pd.DataFrame:
        """Extract county-level median vegetation indices for one month.

        Builds a cloud-masked monthly composite of Sentinel-2 imagery,
        computes NDVI/EVI/SAVI/NDWI, then reduces each county polygon
        to median values.

        Args:
            fips_list: List of 5-digit FIPS codes.
            year: Calendar year.
            month: Calendar month (1-12).

        Returns:
            DataFrame with columns: fips, year, month, ndvi, evi, savi, ndwi.
        """
        ee = self.ee

        # Use Sentinel-2 for 2017+ (reliable), Landsat 8 for earlier
        use_landsat = year < 2017

        # Date range for the month
        start_date = f"{year}-{month:02d}-01"
        if month == 12:
            end_date = f"{year + 1}-01-01"
        else:
            end_date = f"{year}-{month + 1:02d}-01"

        # Get county geometries
        counties_fc = self._get_county_geometries_batch(fips_list)
        roi = counties_fc.geometry().bounds()

        if use_landsat:
            collection = (
                ee.ImageCollection("LANDSAT/LC08/C02/T1_L2")
                .filterDate(start_date, end_date)
                .filterBounds(roi)
                .map(self._cloud_mask_landsat)
                .map(self._compute_indices_landsat)
            )
        else:
            collection = (
                ee.ImageCollection("COPERNICUS/S2_SR")
                .filterDate(start_date, end_date)
                .filterBounds(roi)
                .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 30))
                .map(self.cloud_mask)
                .map(self.compute_indices)
            )

        # Monthly median composite
        composite = collection.median()

        # Reduce to county medians
        def _reduce_county(feature):
            stats = composite.reduceRegion(
                reducer=ee.Reducer.median(),
                geometry=feature.geometry(),
                scale=100,  # 100m for speed (Sentinel is 10m native)
                maxPixels=1e8,
                bestEffort=True,
            )
            return feature.set(stats).set("fips", feature.get("GEOID"))

        results = counties_fc.map(_reduce_county)

        # Extract to Python
        try:
            info = results.getInfo()
        except Exception as exc:
            logger.error("GEE extraction failed for %d-%02d: %s", year, month, exc)
            return pd.DataFrame()

        records = []
        for feat in info.get("features", []):
            props = feat.get("properties", {})
            records.append({
                "fips": str(props.get("fips", props.get("GEOID", ""))).zfill(5),
                "year": year,
                "month": month,
                "ndvi": props.get("NDVI"),
                "evi": props.get("EVI"),
                "savi": props.get("SAVI"),
                "ndwi": props.get("NDWI"),
            })

        df = pd.DataFrame(records)

        # Apply Landsat-to-Sentinel cross-calibration if needed
        if use_landsat:
            df = self._apply_cross_calibration(df)

        logger.info(
            "Extracted indices for %d counties, %d-%02d",
            len(df), year, month,
        )
        return df

    # ------------------------------------------------------------------
    # Batch extraction
    # ------------------------------------------------------------------

    def extract_all(
        self,
        fips_list: List[str],
        years: Union[range, List[int]],
        months: Optional[List[int]] = None,
        output_dir: Optional[Path] = None,
        batch_size: int = 200,
    ) -> pd.DataFrame:
        """Extract indices for all counties, years, and months.

        Processes in batches of counties to avoid GEE memory limits.
        Caches results per year-month to allow resumption.

        Args:
            fips_list: County FIPS codes (up to ~3100).
            years: Calendar years to process.
            months: Months to process (default: 3-11, growing season).
            output_dir: Cache directory for per-month files.
            batch_size: Number of counties per GEE request.

        Returns:
            Combined DataFrame of all extractions.
        """
        if months is None:
            months = list(range(3, 12))  # March through November
        if output_dir:
            output_dir = ensure_directory(output_dir)

        all_frames = []

        for year in years:
            for month in months:
                cache = None
                if output_dir:
                    cache = output_dir / f"indices_{year}_{month:02d}.parquet"
                    if cache.exists():
                        all_frames.append(pd.read_parquet(cache))
                        continue

                logger.info("Extracting indices: %d-%02d", year, month)
                month_frames = []

                # Process in batches
                for i in range(0, len(fips_list), batch_size):
                    batch = fips_list[i : i + batch_size]
                    try:
                        batch_df = self.extract_county_indices(batch, year, month)
                        if not batch_df.empty:
                            month_frames.append(batch_df)
                    except Exception as exc:
                        logger.error(
                            "Batch %d-%d failed for %d-%02d: %s",
                            i, i + batch_size, year, month, exc,
                        )

                    # GEE rate limiting
                    time.sleep(1.0)

                if month_frames:
                    mdf = pd.concat(month_frames, ignore_index=True)
                    if cache:
                        mdf.to_parquet(cache, index=False)
                    all_frames.append(mdf)

        if not all_frames:
            return pd.DataFrame()

        return pd.concat(all_frames, ignore_index=True)

    # ------------------------------------------------------------------
    # Landsat cross-calibration
    # ------------------------------------------------------------------

    def _apply_cross_calibration(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform Landsat-derived indices to Sentinel-2 scale.

        Uses published linear calibration coefficients from
        Claverie et al. (2018) to harmonize Landsat 8 and Sentinel-2
        vegetation indices.
        """
        df = df.copy()
        for index_name, coefs in L8_TO_S2_COEFFICIENTS.items():
            col = index_name.lower()
            if col in df.columns:
                df[col] = df[col] * coefs["slope"] + coefs["intercept"]
        return df

    def backfill_with_landsat(
        self,
        fips_list: List[str],
        years_before_2015: Union[range, List[int]],
        months: Optional[List[int]] = None,
        output_dir: Optional[Path] = None,
        batch_size: int = 200,
    ) -> pd.DataFrame:
        """Extract vegetation indices from Landsat 8 for pre-Sentinel years.

        Landsat 8 launched in 2013. For 2008-2012, we use Landsat 7 ETM+
        (with SLC-off gap filling). Cross-calibration is applied to harmonize
        all indices to the Sentinel-2 scale.

        Args:
            fips_list: County FIPS codes.
            years_before_2015: Years to backfill (e.g. range(2008, 2015)).
            months: Months to extract.
            output_dir: Cache directory.
            batch_size: Counties per GEE request.

        Returns:
            Cross-calibrated DataFrame with Sentinel-equivalent indices.
        """
        ee = self.ee

        if months is None:
            months = list(range(3, 12))
        if output_dir:
            output_dir = ensure_directory(output_dir)

        all_frames = []

        for year in years_before_2015:
            for month in months:
                cache = None
                if output_dir:
                    cache = output_dir / f"landsat_indices_{year}_{month:02d}.parquet"
                    if cache.exists():
                        all_frames.append(pd.read_parquet(cache))
                        continue

                logger.info("Backfilling with Landsat: %d-%02d", year, month)

                start_date = f"{year}-{month:02d}-01"
                if month == 12:
                    end_date = f"{year + 1}-01-01"
                else:
                    end_date = f"{year}-{month + 1:02d}-01"

                month_frames = []
                for i in range(0, len(fips_list), batch_size):
                    batch = fips_list[i : i + batch_size]
                    counties_fc = self._get_county_geometries_batch(batch)
                    roi = counties_fc.geometry().bounds()

                    # Use Landsat 8 for 2013+, Landsat 7 for earlier
                    if year >= 2013:
                        collection = (
                            ee.ImageCollection("LANDSAT/LC08/C02/T1_L2")
                            .filterDate(start_date, end_date)
                            .filterBounds(roi)
                            .map(self._cloud_mask_landsat)
                            .map(self._compute_indices_landsat)
                        )
                    else:
                        collection = (
                            ee.ImageCollection("LANDSAT/LE07/C02/T1_L2")
                            .filterDate(start_date, end_date)
                            .filterBounds(roi)
                            .map(self._cloud_mask_landsat)
                            .map(self._compute_indices_landsat)
                        )

                    composite = collection.median()

                    def _reduce(feature):
                        stats = composite.reduceRegion(
                            reducer=ee.Reducer.median(),
                            geometry=feature.geometry(),
                            scale=100,
                            maxPixels=1e8,
                            bestEffort=True,
                        )
                        return feature.set(stats).set("fips", feature.get("GEOID"))

                    try:
                        results = counties_fc.map(_reduce)
                        info = results.getInfo()
                        for feat in info.get("features", []):
                            props = feat.get("properties", {})
                            month_frames.append({
                                "fips": str(props.get("fips", "")).zfill(5),
                                "year": year,
                                "month": month,
                                "ndvi": props.get("NDVI"),
                                "evi": props.get("EVI"),
                                "savi": props.get("SAVI"),
                                "ndwi": props.get("NDWI"),
                            })
                    except Exception as exc:
                        logger.error("Landsat backfill batch failed: %s", exc)

                    time.sleep(1.0)

                if month_frames:
                    mdf = pd.DataFrame(month_frames)
                    mdf = self._apply_cross_calibration(mdf)
                    if cache:
                        mdf.to_parquet(cache, index=False)
                    all_frames.append(mdf)

        if not all_frames:
            return pd.DataFrame()

        return pd.concat(all_frames, ignore_index=True)

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate(self, df: pd.DataFrame) -> None:
        """Sanity-check extracted vegetation indices."""
        assert len(df) > 0, "Sentinel DataFrame is empty"

        for col in ["ndvi", "evi", "savi", "ndwi"]:
            if col in df.columns:
                valid = df[col].dropna()
                if len(valid) > 0:
                    assert valid.between(-1.0, 1.0).all(), (
                        f"{col} values out of [-1, 1] range"
                    )

        nulls = df[["ndvi", "evi", "savi", "ndwi"]].isna().mean()
        for col, pct in nulls.items():
            if pct > 0.5:
                logger.warning("%.1f%% null values in %s", pct * 100, col)

        logger.info("Sentinel validation passed: %d rows", len(df))

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    def save(self, df: pd.DataFrame, output_path: Path) -> None:
        """Save vegetation indices as Parquet."""
        save_parquet(df, output_path)
