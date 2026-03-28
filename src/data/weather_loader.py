"""
NOAA Climate Data Online (CDO) weather data loader.

Downloads daily station-level weather observations (TMAX, TMIN, PRCP, SNOW),
aggregates them to county-month level, and computes derived agricultural
features like growing degree days, frost days, and consecutive dry days.
Also downloads Palmer Drought Severity Index (PDSI) from drought.gov.

API docs: https://www.ncdc.noaa.gov/cdo-web/api/v2/
Free token: https://www.ncdc.noaa.gov/cdo-web/token

Typical usage:
    loader = NOAAWeatherLoader(api_token="YOUR_TOKEN")
    loader.download_station_data(
        years=range(2008, 2025),
        variables=["TMAX", "TMIN", "PRCP", "SNOW"],
        output_dir=Path("data/raw/weather"),
    )
    county_df = loader.aggregate_to_county(station_df, station_county_map)
    features = loader.compute_derived_features(county_df)
"""

import io
import logging
import time
from datetime import date, datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from cascrop.src.data.utils import (
    STATE_FIPS_TO_NAME,
    create_retry_session,
    ensure_directory,
    load_fips_codes,
    haversine_distance,
    save_parquet,
)

logger = logging.getLogger(__name__)

# NOAA CDO API base
CDO_API_BASE = "https://www.ncdc.noaa.gov/cdo-web/api/v2/"

# Dataset IDs
DATASET_GHCND = "GHCND"  # Global Historical Climatology Network - Daily

# Target variables
DEFAULT_VARIABLES = ["TMAX", "TMIN", "PRCP", "SNOW"]

# NOAA API limits: 1000 records per request, 5 requests per second
API_LIMIT = 1000
API_RATE_LIMIT_DELAY = 0.25  # seconds between requests

# Growing Degree Day base temperature (Fahrenheit for corn)
GDD_BASE_F = 50.0
GDD_CEILING_F = 86.0


class NOAAWeatherLoader:
    """Download and process NOAA weather data for county-level analysis."""

    def __init__(self, api_token: str):
        """
        Args:
            api_token: NOAA CDO API token from
                https://www.ncdc.noaa.gov/cdo-web/token
        """
        self.api_token = api_token
        self.session = create_retry_session(retries=5, backoff_factor=2.0)
        self.session.headers.update({"token": api_token})

    # ------------------------------------------------------------------
    # Station metadata
    # ------------------------------------------------------------------

    def get_stations(
        self,
        state_fips: Optional[str] = None,
        dataset: str = DATASET_GHCND,
    ) -> pd.DataFrame:
        """Fetch metadata for all GHCND stations, optionally filtered by state.

        Returns DataFrame with columns: station_id, name, latitude, longitude,
        elevation, min_date, max_date.
        """
        params = {
            "datasetid": dataset,
            "limit": API_LIMIT,
            "offset": 1,
        }
        if state_fips:
            params["locationid"] = f"FIPS:{state_fips}"

        all_stations = []
        while True:
            resp = self.session.get(
                f"{CDO_API_BASE}stations", params=params, timeout=60
            )
            resp.raise_for_status()
            body = resp.json()

            results = body.get("results", [])
            if not results:
                break

            all_stations.extend(results)
            logger.debug("Fetched %d stations (offset %d)", len(results), params["offset"])

            # Check if there are more pages
            metadata = body.get("metadata", {}).get("resultset", {})
            total = metadata.get("count", 0)
            if params["offset"] + API_LIMIT > total:
                break
            params["offset"] += API_LIMIT
            time.sleep(API_RATE_LIMIT_DELAY)

        if not all_stations:
            return pd.DataFrame()

        df = pd.DataFrame(all_stations)
        df = df.rename(columns={
            "id": "station_id",
            "latitude": "latitude",
            "longitude": "longitude",
            "mindate": "min_date",
            "maxdate": "max_date",
        })
        return df

    def build_station_county_map(
        self, stations_df: pd.DataFrame, max_distance_km: float = 50.0
    ) -> Dict[str, str]:
        """Map each weather station to its nearest county by centroid distance.

        Args:
            stations_df: Station metadata with latitude, longitude columns.
            max_distance_km: Maximum distance to assign a station to a county.

        Returns:
            Dict mapping station_id -> 5-digit FIPS code.
        """
        fips_df = load_fips_codes()
        county_coords = list(
            zip(fips_df["fips"], fips_df["lat"], fips_df["lon"])
        )

        mapping = {}
        unmapped = 0

        for _, row in stations_df.iterrows():
            slat, slon = row.get("latitude"), row.get("longitude")
            if pd.isna(slat) or pd.isna(slon):
                unmapped += 1
                continue

            best_fips = None
            best_dist = float("inf")

            for fips, clat, clon in county_coords:
                if pd.isna(clat) or pd.isna(clon):
                    continue
                d = haversine_distance(slat, slon, clat, clon)
                if d < best_dist:
                    best_dist = d
                    best_fips = fips

            if best_dist <= max_distance_km and best_fips is not None:
                mapping[row["station_id"]] = best_fips
            else:
                unmapped += 1

        logger.info(
            "Mapped %d stations to counties (%d unmapped beyond %.0f km)",
            len(mapping), unmapped, max_distance_km,
        )
        return mapping

    # ------------------------------------------------------------------
    # Download station data
    # ------------------------------------------------------------------

    def download_station_data(
        self,
        years: Union[range, List[int]],
        variables: Optional[List[str]] = None,
        output_dir: Optional[Path] = None,
        states: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """Download daily weather observations from NOAA CDO.

        Because the CDO API limits responses to 1000 records, we query
        state-by-state, month-by-month for each year.

        Args:
            years: Calendar years to download.
            variables: GHCND variable IDs (default: TMAX, TMIN, PRCP, SNOW).
            output_dir: Cache directory for per-state-year files.
            states: State FIPS codes to download. Defaults to all 50 states.

        Returns:
            Combined DataFrame of daily station observations.
        """
        if variables is None:
            variables = list(DEFAULT_VARIABLES)
        if states is None:
            states = sorted(STATE_FIPS_TO_NAME.keys())
        if output_dir:
            output_dir = ensure_directory(output_dir)

        all_frames = []
        datatypes = ",".join(variables)

        for year in years:
            for state in states:
                cache_path = None
                if output_dir:
                    cache_path = output_dir / f"weather_{state}_{year}.parquet"
                    if cache_path.exists():
                        all_frames.append(pd.read_parquet(cache_path))
                        continue

                logger.info("Downloading weather: state=%s year=%d", state, year)
                state_frames = []

                # Query in monthly chunks to stay under API row limits
                for month in range(1, 13):
                    start_date = f"{year}-{month:02d}-01"
                    if month == 12:
                        end_date = f"{year}-12-31"
                    else:
                        end_date = f"{year}-{month + 1:02d}-01"

                    params = {
                        "datasetid": DATASET_GHCND,
                        "datatypeid": datatypes,
                        "locationid": f"FIPS:{state}",
                        "startdate": start_date,
                        "enddate": end_date,
                        "units": "standard",
                        "limit": API_LIMIT,
                        "offset": 1,
                    }

                    month_data = self._paginated_query("data", params)
                    if month_data:
                        state_frames.extend(month_data)

                    time.sleep(API_RATE_LIMIT_DELAY)

                if state_frames:
                    sdf = pd.DataFrame(state_frames)
                    sdf["state_fips"] = state
                    if cache_path:
                        sdf.to_parquet(cache_path, index=False)
                    all_frames.append(sdf)
                    logger.info(
                        "  State %s year %d: %d observations",
                        state, year, len(sdf),
                    )

        if not all_frames:
            logger.warning("No weather data downloaded")
            return pd.DataFrame()

        combined = pd.concat(all_frames, ignore_index=True)
        logger.info("Total weather observations: %d", len(combined))
        return combined

    def _paginated_query(
        self, endpoint: str, params: dict
    ) -> List[dict]:
        """Fetch all pages from a CDO API endpoint."""
        all_results = []
        while True:
            try:
                resp = self.session.get(
                    f"{CDO_API_BASE}{endpoint}", params=params, timeout=60
                )
                if resp.status_code == 429:
                    # Rate limited — back off
                    logger.warning("Rate limited, sleeping 5 seconds")
                    time.sleep(5)
                    continue
                resp.raise_for_status()
            except Exception as exc:
                logger.debug("CDO API error: %s", exc)
                break

            body = resp.json()
            results = body.get("results", [])
            if not results:
                break

            all_results.extend(results)

            metadata = body.get("metadata", {}).get("resultset", {})
            total = metadata.get("count", 0)
            if params["offset"] + API_LIMIT > total:
                break
            params["offset"] += API_LIMIT
            time.sleep(API_RATE_LIMIT_DELAY)

        return all_results

    # ------------------------------------------------------------------
    # Aggregate station data to county-month
    # ------------------------------------------------------------------

    def aggregate_to_county(
        self,
        station_df: pd.DataFrame,
        station_county_map: Dict[str, str],
    ) -> pd.DataFrame:
        """Aggregate daily station observations to county-month means.

        For each county-month, we average across all stations assigned
        to that county. Missing county-months are left as NaN (to be
        spatially interpolated later).

        Args:
            station_df: Daily observations with station, date, datatype, value.
            station_county_map: Dict mapping station_id -> 5-digit FIPS.

        Returns:
            DataFrame with columns: fips, year, month, TMAX, TMIN, PRCP, SNOW
            (monthly averages/totals).
        """
        df = station_df.copy()

        # Map stations to counties
        df["fips"] = df["station"].map(station_county_map)
        df = df.dropna(subset=["fips"])

        # Parse date
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df["year"] = df["date"].dt.year
        df["month"] = df["date"].dt.month

        # Pivot to get one column per variable
        pivot = df.pivot_table(
            index=["fips", "year", "month", "date"],
            columns="datatype",
            values="value",
            aggfunc="mean",
        ).reset_index()

        # Aggregate to county-month
        # Temperature: monthly mean; Precipitation/Snow: monthly sum
        agg_dict = {}
        for var in DEFAULT_VARIABLES:
            if var in pivot.columns:
                if var in ("PRCP", "SNOW"):
                    agg_dict[var] = "sum"
                else:
                    agg_dict[var] = "mean"

        county_month = (
            pivot.groupby(["fips", "year", "month"])
            .agg(agg_dict)
            .reset_index()
        )

        logger.info(
            "Aggregated to %d county-month observations", len(county_month)
        )
        return county_month

    # ------------------------------------------------------------------
    # Derived features
    # ------------------------------------------------------------------

    def compute_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute agriculture-relevant weather features from raw observations.

        New columns added:
        - gdd: Growing Degree Days (base 50F, ceiling 86F)
        - frost_days: Number of days with TMIN <= 32F in the month
        - consec_dry_days: Max consecutive days with PRCP < 0.01 inches
        - tavg: Average of TMAX and TMIN

        Args:
            df: County-month weather data with TMAX, TMIN, PRCP columns.
                Can be daily (for precise GDD) or monthly (estimated).

        Returns:
            DataFrame with added derived feature columns.
        """
        df = df.copy()

        # Average temperature
        if "TMAX" in df.columns and "TMIN" in df.columns:
            df["tavg"] = (df["TMAX"] + df["TMIN"]) / 2.0
        elif "TMAX" in df.columns:
            df["tavg"] = df["TMAX"]

        # Growing Degree Days (simplified monthly estimate)
        # GDD = max(0, min(TMAX, 86) - max(TMIN, 50)) for corn
        if "TMAX" in df.columns and "TMIN" in df.columns:
            tmax_capped = df["TMAX"].clip(upper=GDD_CEILING_F)
            tmin_floored = df["TMIN"].clip(lower=GDD_BASE_F)
            daily_gdd = ((tmax_capped + tmin_floored) / 2.0 - GDD_BASE_F).clip(lower=0)
            # For monthly data, multiply by ~30 days
            df["gdd"] = daily_gdd * 30.0
        else:
            df["gdd"] = np.nan

        # Frost days estimate
        # If we have monthly TMIN, estimate as: 1 if TMIN <= 32, else 0
        # For daily data accumulated to monthly, this would be a count.
        # Since our aggregated data is monthly mean TMIN, we use a proxy:
        if "TMIN" in df.columns:
            df["frost_risk"] = (df["TMIN"] <= 32.0).astype(float)
            # More nuanced: probability based on how far below freezing
            df["frost_days_est"] = np.where(
                df["TMIN"] <= 32.0,
                np.clip(30 * (32.0 - df["TMIN"]) / 20.0, 1, 30),
                0,
            )
        else:
            df["frost_risk"] = np.nan
            df["frost_days_est"] = np.nan

        # Consecutive dry days proxy
        # From monthly PRCP total, estimate: low precip -> more dry days
        if "PRCP" in df.columns:
            # If total monthly precip < 0.5 inches, estimate high dry streak
            df["dry_month_flag"] = (df["PRCP"] < 0.5).astype(float)
            # Rough proxy: consec dry days ~ 30 * (1 - PRCP/normal)
            # Using 3 inches as "normal" monthly precipitation
            df["consec_dry_days_est"] = np.clip(
                30 * (1 - df["PRCP"] / 3.0), 0, 30
            )
        else:
            df["dry_month_flag"] = np.nan
            df["consec_dry_days_est"] = np.nan

        # Temperature range (diurnal)
        if "TMAX" in df.columns and "TMIN" in df.columns:
            df["temp_range"] = df["TMAX"] - df["TMIN"]

        logger.info("Computed derived weather features for %d rows", len(df))
        return df

    # ------------------------------------------------------------------
    # PDSI / Drought Index
    # ------------------------------------------------------------------

    def download_drought_index(
        self,
        years: Union[range, List[int]],
        output_dir: Path,
    ) -> pd.DataFrame:
        """Download Palmer Drought Severity Index (PDSI) from drought.gov.

        Uses the US Drought Monitor API to get county-level drought
        classifications (D0-D4) by week.

        The USDM provides weekly % area in each drought category per county.
        We convert these to a single severity score:
            pdsi_proxy = 0*D0 + 1*D1 + 2*D2 + 3*D3 + 4*D4

        Args:
            years: Calendar years.
            output_dir: Directory to cache downloaded files.

        Returns:
            DataFrame with columns: fips, year, month, drought_score, pct_d0-d4.
        """
        output_dir = ensure_directory(output_dir)
        frames = []

        for year in years:
            cache = output_dir / f"drought_{year}.parquet"
            if cache.exists():
                frames.append(pd.read_parquet(cache))
                continue

            logger.info("Downloading drought data for %d", year)
            year_frames = []

            # USDM comprehensive statistics CSV (weekly snapshots)
            for month in range(1, 13):
                # Use the last day of each month as the snapshot date
                if month == 12:
                    end_day = 31
                else:
                    # Approximate month end
                    end_day = [0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31][month]

                date_str = f"{year}{month:02d}{end_day:02d}"
                url = (
                    f"https://droughtmonitor.unl.edu/DmData/DataDownload/"
                    f"ComprehensiveStatistics.aspx?"
                    f"fips=0&date={date_str}&statisticsType=2"
                )

                try:
                    resp = self.session.get(url, timeout=60)
                    if resp.status_code != 200:
                        continue

                    # Try to parse as CSV
                    try:
                        mdf = pd.read_csv(io.StringIO(resp.text))
                    except Exception:
                        continue

                    if mdf.empty:
                        continue

                    mdf["year"] = year
                    mdf["month"] = month
                    year_frames.append(mdf)

                except Exception as exc:
                    logger.debug("Drought download failed: %s — %s", date_str, exc)

                time.sleep(0.5)

            if year_frames:
                ydf = pd.concat(year_frames, ignore_index=True)
                ydf = self._clean_drought_data(ydf)
                ydf.to_parquet(cache, index=False)
                frames.append(ydf)
                logger.info("  Year %d: %d drought records", year, len(ydf))

        if not frames:
            logger.warning("No drought data downloaded — will use fallback")
            return pd.DataFrame()

        return pd.concat(frames, ignore_index=True)

    def _clean_drought_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize drought monitor data into our schema."""
        result = pd.DataFrame()

        # FIPS column might be named differently
        fips_col = None
        for candidate in ["FIPS", "fips", "CountyFIPS", "county_fips"]:
            if candidate in df.columns:
                fips_col = candidate
                break

        if fips_col is None:
            logger.warning("No FIPS column found in drought data")
            return pd.DataFrame()

        result["fips"] = df[fips_col].astype(str).str.zfill(5)
        result["year"] = df.get("year", 0)
        result["month"] = df.get("month", 0)

        # Drought categories D0-D4 (percent area)
        for level in range(5):
            col_candidates = [f"D{level}", f"d{level}", f"pctD{level}"]
            for cc in col_candidates:
                if cc in df.columns:
                    result[f"pct_d{level}"] = pd.to_numeric(
                        df[cc], errors="coerce"
                    ).fillna(0)
                    break
            else:
                result[f"pct_d{level}"] = 0.0

        # Composite drought severity score
        result["drought_score"] = (
            0 * result["pct_d0"]
            + 1 * result["pct_d1"]
            + 2 * result["pct_d2"]
            + 3 * result["pct_d3"]
            + 4 * result["pct_d4"]
        ) / 100.0  # Normalize to 0-4 scale

        return result

    # ------------------------------------------------------------------
    # Spatial interpolation for missing stations
    # ------------------------------------------------------------------

    @staticmethod
    def spatial_interpolate(
        df: pd.DataFrame,
        fips_all: List[str],
        k_nearest: int = 3,
    ) -> pd.DataFrame:
        """Fill missing county-months using inverse-distance-weighted
        interpolation from the k nearest counties with data.

        Args:
            df: County-month weather data (may have missing counties).
            fips_all: Complete list of FIPS codes to interpolate for.
            k_nearest: Number of nearest neighbors for IDW interpolation.

        Returns:
            DataFrame with all fips_all x year x month combinations filled.
        """
        from cascrop.src.data.utils import fips_to_latlon

        coords = fips_to_latlon(fips_all)
        weather_cols = [c for c in df.columns if c not in ("fips", "year", "month")]

        # Build full grid
        years = sorted(df["year"].unique())
        months = sorted(df["month"].unique())

        existing = set(zip(df["fips"], df["year"], df["month"]))
        records_to_fill = []

        for fips in fips_all:
            if fips not in coords:
                continue
            for year in years:
                for month in months:
                    if (fips, year, month) not in existing:
                        records_to_fill.append((fips, year, month))

        if not records_to_fill:
            return df

        logger.info("Interpolating %d missing county-month observations", len(records_to_fill))

        fill_rows = []
        for fips, year, month in records_to_fill:
            lat, lon = coords[fips]
            # Find counties with data for this year-month
            subset = df[(df["year"] == year) & (df["month"] == month)]
            if subset.empty:
                continue

            # Compute distances to all available counties
            distances = []
            for _, row in subset.iterrows():
                if row["fips"] in coords:
                    nlat, nlon = coords[row["fips"]]
                    d = haversine_distance(lat, lon, nlat, nlon)
                    distances.append((d, row))

            if not distances:
                continue

            distances.sort(key=lambda x: x[0])
            nearest = distances[:k_nearest]

            # IDW interpolation
            weights = [1.0 / max(d, 1.0) for d, _ in nearest]
            w_sum = sum(weights)
            weights = [w / w_sum for w in weights]

            row_dict = {"fips": fips, "year": year, "month": month}
            for col in weather_cols:
                values = [r[col] for _, r in nearest if pd.notna(r.get(col))]
                w_valid = [weights[i] for i, (_, r) in enumerate(nearest) if pd.notna(r.get(col))]
                if values and w_valid:
                    w_norm = [w / sum(w_valid) for w in w_valid]
                    row_dict[col] = sum(v * w for v, w in zip(values, w_norm))
                else:
                    row_dict[col] = np.nan

            fill_rows.append(row_dict)

        if fill_rows:
            fill_df = pd.DataFrame(fill_rows)
            df = pd.concat([df, fill_df], ignore_index=True)

        return df

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate(self, df: pd.DataFrame) -> None:
        """Basic sanity checks on weather data."""
        assert len(df) > 0, "Weather DataFrame is empty"

        if "TMAX" in df.columns:
            # Temperature in Fahrenheit: should be between -60 and 140
            assert df["TMAX"].dropna().between(-60, 140).all(), \
                "TMAX values out of range [-60, 140]F"

        if "PRCP" in df.columns:
            assert (df["PRCP"].dropna() >= 0).all(), "Negative precipitation values"

        if "fips" in df.columns:
            n_counties = df["fips"].nunique()
            logger.info("Weather data covers %d counties", n_counties)

        logger.info("Weather validation passed: %d rows", len(df))

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    def save(self, df: pd.DataFrame, output_path: Path) -> None:
        """Save weather data as Parquet."""
        save_parquet(df, output_path)
