"""
FIPS code utilities, geographic helpers, and shared constants for CasCrop data loaders.

Provides:
- FIPS code loading, validation, and geocoding
- Commodity code mappings for USDA systems
- Cause-of-loss code categorization for RMA data
- Retry-enabled HTTP session factory
"""

import io
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Commodity code mappings (USDA NASS / RMA commodity codes)
# ---------------------------------------------------------------------------

COMMODITY_CODES: Dict[str, int] = {
    "CORN": 41,
    "SOYBEANS": 81,
    "WHEAT": 11,      # Winter wheat (most common)
    "WHEAT_SPRING": 12,
    "WHEAT_DURUM": 13,
}

COMMODITY_NAMES: Dict[int, str] = {v: k for k, v in COMMODITY_CODES.items()}

# Target commodities for the CasCrop study
TARGET_COMMODITIES = ["CORN", "SOYBEANS", "WHEAT"]

# NASS commodity_desc strings
NASS_COMMODITY_MAP: Dict[str, str] = {
    "CORN": "CORN",
    "SOYBEANS": "SOYBEANS",
    "WHEAT": "WHEAT",
}

# ---------------------------------------------------------------------------
# RMA Cause-of-Loss code categorization
# See: https://www.rma.usda.gov/data-tools/summary-of-business/cause-of-loss
# ---------------------------------------------------------------------------

CAUSE_OF_LOSS_MAPPING: Dict[str, List[int]] = {
    "DROUGHT": [2, 3],                    # Drought, failure of irrigation water
    "EXCESS_MOISTURE": [10, 11, 14],      # Excess moisture, flood, rain
    "COLD": [15, 16, 17],                 # Freeze, frost, cold weather
    "HEAT": [36, 40],                     # Heat, hot wind
    "PRICE": [47, 48],                    # Decline in price, insufficient revenue
    "OTHER": [],                          # Catch-all (handled programmatically)
}

# Inverted mapping: code -> category
_CAUSE_CODE_TO_CATEGORY: Dict[int, str] = {}
for category, codes in CAUSE_OF_LOSS_MAPPING.items():
    for code in codes:
        _CAUSE_CODE_TO_CATEGORY[code] = category


def map_cause_code(code: int) -> str:
    """Map an RMA cause-of-loss code to one of six categories.

    Returns one of: DROUGHT, EXCESS_MOISTURE, COLD, HEAT, PRICE, OTHER.
    """
    return _CAUSE_CODE_TO_CATEGORY.get(code, "OTHER")


# ---------------------------------------------------------------------------
# State FIPS -> Name mapping (all 50 states + DC)
# ---------------------------------------------------------------------------

STATE_FIPS_TO_NAME: Dict[str, str] = {
    "01": "Alabama", "02": "Alaska", "04": "Arizona", "05": "Arkansas",
    "06": "California", "08": "Colorado", "09": "Connecticut", "10": "Delaware",
    "11": "District of Columbia", "12": "Florida", "13": "Georgia", "15": "Hawaii",
    "16": "Idaho", "17": "Illinois", "18": "Indiana", "19": "Iowa",
    "20": "Kansas", "21": "Kentucky", "22": "Louisiana", "23": "Maine",
    "24": "Maryland", "25": "Massachusetts", "26": "Michigan", "27": "Minnesota",
    "28": "Mississippi", "29": "Missouri", "30": "Montana", "31": "Nebraska",
    "32": "Nevada", "33": "New Hampshire", "34": "New Jersey", "35": "New Mexico",
    "36": "New York", "37": "North Carolina", "38": "North Dakota", "39": "Ohio",
    "40": "Oklahoma", "41": "Oregon", "42": "Pennsylvania", "44": "Rhode Island",
    "45": "South Carolina", "46": "South Dakota", "47": "Tennessee", "48": "Texas",
    "49": "Utah", "50": "Vermont", "51": "Virginia", "53": "Washington",
    "54": "West Virginia", "55": "Wisconsin", "56": "Wyoming",
}

STATE_NAME_TO_FIPS: Dict[str, str] = {v: k for k, v in STATE_FIPS_TO_NAME.items()}

# States with significant crop production (continental US ag states)
AG_STATES = {
    "17", "18", "19", "20", "21", "26", "27", "28", "29", "31",  # Midwest/Plains
    "38", "39", "46", "55",                                        # Upper Midwest
    "01", "05", "13", "22", "28", "37", "45", "47", "48", "51",  # South
    "08", "30", "35", "40", "56",                                  # West/Mountain
    "06", "16", "41", "53",                                        # Pacific
    "36", "42",                                                    # Northeast
}

# ---------------------------------------------------------------------------
# HTTP session with retry logic
# ---------------------------------------------------------------------------


def create_retry_session(
    retries: int = 5,
    backoff_factor: float = 1.0,
    status_forcelist: Tuple[int, ...] = (429, 500, 502, 503, 504),
    timeout: int = 60,
) -> requests.Session:
    """Build a requests.Session with exponential-backoff retry.

    Args:
        retries: Maximum number of retry attempts.
        backoff_factor: Multiplier for exponential backoff between retries.
        status_forcelist: HTTP status codes that trigger a retry.
        timeout: Default request timeout in seconds.

    Returns:
        Configured ``requests.Session`` instance.
    """
    session = requests.Session()
    retry = Retry(
        total=retries,
        read=retries,
        connect=retries,
        backoff_factor=backoff_factor,
        status_forcelist=status_forcelist,
        allowed_methods=["HEAD", "GET", "OPTIONS", "POST"],
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    session.timeout = timeout
    return session


# ---------------------------------------------------------------------------
# FIPS code loading and geocoding
# ---------------------------------------------------------------------------


def load_fips_codes(cache_path: Optional[Path] = None) -> pd.DataFrame:
    """Download the master FIPS code list from the Census Bureau.

    Returns a DataFrame with columns: state_fips, county_fips, fips (5-digit),
    state_name, county_name, lat, lon.

    The file is cached locally after the first download.
    """
    if cache_path is None:
        cache_path = Path.home() / ".cascrop" / "fips_master.parquet"

    if cache_path.exists():
        logger.info("Loading cached FIPS codes from %s", cache_path)
        return pd.read_parquet(cache_path)

    logger.info("Downloading FIPS codes from Census Bureau gazetteer files")
    session = create_retry_session()

    # Census Bureau county gazetteer — tab-delimited, contains FIPS + lat/lon
    url = (
        "https://www2.census.gov/geo/docs/maps-data/data/gazetteer/"
        "2023_Gazetteer/2023_Gaz_counties_national.zip"
    )

    try:
        resp = session.get(url, timeout=120)
        resp.raise_for_status()
    except requests.RequestException:
        # Fallback: try previous year's gazetteer
        url = (
            "https://www2.census.gov/geo/docs/maps-data/data/gazetteer/"
            "2022_Gazetteer/2022_Gaz_counties_national.zip"
        )
        resp = session.get(url, timeout=120)
        resp.raise_for_status()

    import zipfile

    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        # The zip contains a single tab-delimited text file
        names = zf.namelist()
        txt_name = [n for n in names if n.endswith(".txt")][0]
        with zf.open(txt_name) as f:
            df = pd.read_csv(f, sep="\t", dtype=str)

    # Standardize column names (Census uses variable whitespace)
    df.columns = df.columns.str.strip()

    # Build clean output
    result = pd.DataFrame()
    result["state_fips"] = df["GEOID"].str[:2]
    result["county_fips"] = df["GEOID"].str[2:]
    result["fips"] = df["GEOID"].str.zfill(5)
    result["state_name"] = result["state_fips"].map(STATE_FIPS_TO_NAME)
    result["county_name"] = df["NAME"].str.strip()
    result["lat"] = pd.to_numeric(df["INTPTLAT"].str.strip(), errors="coerce")
    result["lon"] = pd.to_numeric(df["INTPTLONG"].str.strip(), errors="coerce")

    # Drop territories — keep only 50 states + DC
    result = result[result["state_fips"].isin(STATE_FIPS_TO_NAME.keys())].copy()
    result = result.reset_index(drop=True)

    # Cache
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(cache_path, index=False)
    logger.info("Cached %d FIPS codes to %s", len(result), cache_path)

    return result


def fips_to_latlon(
    fips_codes: Optional[List[str]] = None,
) -> Dict[str, Tuple[float, float]]:
    """Return a dict mapping 5-digit FIPS codes to (lat, lon) centroids.

    If *fips_codes* is None, returns the full mapping for all counties.
    """
    df = load_fips_codes()
    if fips_codes is not None:
        df = df[df["fips"].isin(fips_codes)]
    return dict(zip(df["fips"], zip(df["lat"], df["lon"])))


def state_fips_to_name() -> Dict[str, str]:
    """Return the canonical state-FIPS-to-name mapping dict."""
    return dict(STATE_FIPS_TO_NAME)


# ---------------------------------------------------------------------------
# FIPS code formatting and validation
# ---------------------------------------------------------------------------


def format_fips(state_code, county_code) -> str:
    """Construct a 5-digit FIPS from state and county codes.

    Handles both numeric and string inputs; zero-pads as needed.
    """
    s = str(int(state_code)).zfill(2)
    c = str(int(county_code)).zfill(3)
    return f"{s}{c}"


def validate_fips(
    df: pd.DataFrame,
    fips_col: str = "fips",
    strict: bool = False,
) -> pd.DataFrame:
    """Validate FIPS codes in a DataFrame.

    Checks:
    - Column exists and has no nulls.
    - Every value is a 5-digit string with a valid state prefix.
    - (If strict) every FIPS matches the Census master list.

    Returns the same DataFrame (or raises ValueError on failure).
    """
    if fips_col not in df.columns:
        raise ValueError(f"Column '{fips_col}' not found in DataFrame")

    nulls = df[fips_col].isna().sum()
    if nulls > 0:
        raise ValueError(f"{nulls} null values in FIPS column '{fips_col}'")

    # Ensure string type and 5-char length
    fips_series = df[fips_col].astype(str).str.zfill(5)
    bad_length = fips_series.str.len() != 5
    if bad_length.any():
        n_bad = bad_length.sum()
        examples = fips_series[bad_length].head(5).tolist()
        raise ValueError(
            f"{n_bad} FIPS codes with wrong length. Examples: {examples}"
        )

    # Valid state prefixes
    state_prefixes = fips_series.str[:2]
    invalid_states = ~state_prefixes.isin(STATE_FIPS_TO_NAME.keys())
    if invalid_states.any():
        n_inv = invalid_states.sum()
        examples = fips_series[invalid_states].head(5).tolist()
        logger.warning(
            "%d FIPS codes with unrecognized state prefix. Examples: %s",
            n_inv,
            examples,
        )
        if strict:
            raise ValueError(
                f"{n_inv} FIPS codes with invalid state prefix: {examples}"
            )

    if strict:
        master = load_fips_codes()
        valid_fips = set(master["fips"])
        unknown = set(fips_series) - valid_fips
        if unknown:
            logger.warning(
                "%d FIPS codes not in Census master list. Examples: %s",
                len(unknown),
                list(unknown)[:10],
            )

    logger.info("FIPS validation passed: %d records", len(df))
    return df


# ---------------------------------------------------------------------------
# County adjacency
# ---------------------------------------------------------------------------


def load_county_adjacency(cache_path: Optional[Path] = None) -> pd.DataFrame:
    """Download the Census Bureau county adjacency file.

    Returns a DataFrame with columns: fips, neighbor_fips (one row per pair).
    """
    if cache_path is None:
        cache_path = Path.home() / ".cascrop" / "county_adjacency.parquet"

    if cache_path.exists():
        logger.info("Loading cached adjacency from %s", cache_path)
        return pd.read_parquet(cache_path)

    url = (
        "https://www2.census.gov/geo/docs/reference/"
        "county_adjacency2023.txt"
    )
    session = create_retry_session()
    try:
        resp = session.get(url, timeout=120)
        resp.raise_for_status()
    except requests.RequestException:
        # Fallback URL
        url = (
            "https://www2.census.gov/geo/docs/reference/"
            "county_adjacency.txt"
        )
        resp = session.get(url, timeout=120)
        resp.raise_for_status()

    # Parse the pipe-delimited adjacency file.  Format:
    #   "County Name, ST" | FIPS | "Neighbor Name, ST" | Neighbor FIPS
    # When a county has multiple neighbors, subsequent rows omit the first
    # two columns (they're empty until a new source county appears).
    lines = resp.text.strip().split("\n")
    records = []
    current_fips = None

    for line in lines:
        parts = line.split("\t")
        # Clean up parts
        parts = [p.strip().strip('"') for p in parts]

        if len(parts) >= 4 and parts[1].strip():
            current_fips = parts[1].strip().zfill(5)
            neighbor_fips = parts[3].strip().zfill(5)
        elif len(parts) >= 4:
            neighbor_fips = parts[3].strip().zfill(5)
        elif len(parts) >= 2:
            neighbor_fips = parts[-1].strip().zfill(5)
        else:
            continue

        if current_fips and neighbor_fips and current_fips != neighbor_fips:
            records.append({"fips": current_fips, "neighbor_fips": neighbor_fips})

    adj_df = pd.DataFrame(records).drop_duplicates()

    # Filter to 50 states + DC
    valid_states = set(STATE_FIPS_TO_NAME.keys())
    mask = (
        adj_df["fips"].str[:2].isin(valid_states)
        & adj_df["neighbor_fips"].str[:2].isin(valid_states)
    )
    adj_df = adj_df[mask].reset_index(drop=True)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    adj_df.to_parquet(cache_path, index=False)
    logger.info("Cached %d adjacency pairs to %s", len(adj_df), cache_path)
    return adj_df


# ---------------------------------------------------------------------------
# Haversine distance
# ---------------------------------------------------------------------------


def haversine_distance(
    lat1: float, lon1: float, lat2: float, lon2: float
) -> float:
    """Great-circle distance in kilometers between two points."""
    R = 6371.0  # Earth radius in km
    lat1_r, lon1_r = np.radians(lat1), np.radians(lon1)
    lat2_r, lon2_r = np.radians(lat2), np.radians(lon2)
    dlat = lat2_r - lat1_r
    dlon = lon2_r - lon1_r
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1_r) * np.cos(lat2_r) * np.sin(dlon / 2) ** 2
    return R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))


def compute_distance_matrix(fips_list: List[str]) -> pd.DataFrame:
    """Build a pairwise distance matrix (km) for a list of FIPS codes.

    Returns a square DataFrame indexed and columned by FIPS.
    """
    coords = fips_to_latlon(fips_list)
    n = len(fips_list)
    dist = np.zeros((n, n))

    for i, f1 in enumerate(fips_list):
        for j, f2 in enumerate(fips_list):
            if i < j:
                lat1, lon1 = coords[f1]
                lat2, lon2 = coords[f2]
                d = haversine_distance(lat1, lon1, lat2, lon2)
                dist[i, j] = d
                dist[j, i] = d

    return pd.DataFrame(dist, index=fips_list, columns=fips_list)


# ---------------------------------------------------------------------------
# Parquet I/O helpers
# ---------------------------------------------------------------------------


def save_parquet(
    df: pd.DataFrame,
    path: Path,
    partition_cols: Optional[List[str]] = None,
) -> None:
    """Save a DataFrame as Parquet, optionally partitioned."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if partition_cols:
        df.to_parquet(path, index=False, partition_cols=partition_cols, engine="pyarrow")
    else:
        df.to_parquet(path, index=False, engine="pyarrow")

    logger.info("Saved %d rows to %s", len(df), path)


def ensure_directory(path: Path) -> Path:
    """Create a directory (and parents) if it doesn't exist. Returns the path."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path
