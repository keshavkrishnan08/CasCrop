"""
USDA Risk Management Agency (RMA) crop insurance cause-of-loss data loader.

Downloads, parses, cleans, and validates annual cause-of-loss files from the
RMA website. These records form the primary training labels for CasCrop: each
row is an insurance indemnity claim tied to a specific county, commodity, month,
and cause of loss.

Data source:
    https://www.rma.usda.gov/data-tools/summary-of-business/cause-of-loss

Typical usage:
    loader = RMALoader()
    loader.download(years=range(2008, 2025), output_dir=Path("data/raw/rma"))
    df = loader.parse_raw(Path("data/raw/rma"))
    df = loader.clean(df)
    loader.validate(df)
    loader.save(df, Path("data/processed/rma_losses.parquet"))
"""

import io
import logging
import zipfile
from pathlib import Path
from typing import List, Optional, Union

import pandas as pd

from cascrop.src.data.utils import (
    CAUSE_OF_LOSS_MAPPING,
    COMMODITY_CODES,
    STATE_FIPS_TO_NAME,
    TARGET_COMMODITIES,
    create_retry_session,
    ensure_directory,
    format_fips,
    map_cause_code,
    save_parquet,
    validate_fips,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Column spec for RMA cause-of-loss files
# The raw files are pipe-delimited (|) with no header. Column order changes
# slightly across years, so we handle the two major layouts.
# ---------------------------------------------------------------------------

# Post-2016 layout (most common)
COL_NAMES_V2 = [
    "commodity_year_identifier",
    "state_code",
    "state_abbreviation",
    "county_code",
    "county_name",
    "commodity_code",
    "commodity_name",
    "insurance_plan_code",
    "insurance_plan_abbreviation",
    "stage_code",
    "cause_of_loss_code",
    "cause_of_loss_description",
    "month_of_loss",
    "month_of_loss_name",
    "year_of_loss",
    "policies_earning_premium",
    "policies_indemnified",
    "net_planted_quantity",
    "net_endorsed_acres",
    "indemnity_amount",
    "loss_ratio",
]

# Pre-2016 layout
COL_NAMES_V1 = [
    "commodity_year_identifier",
    "state_code",
    "state_abbreviation",
    "county_code",
    "county_name",
    "commodity_code",
    "commodity_name",
    "insurance_plan_code",
    "insurance_plan_abbreviation",
    "cause_of_loss_code",
    "cause_of_loss_description",
    "month_of_loss",
    "month_of_loss_name",
    "year_of_loss",
    "policies_earning_premium",
    "policies_indemnified",
    "net_planted_quantity",
    "net_endorsed_acres",
    "indemnity_amount",
    "loss_ratio",
]

# Columns we actually need downstream
KEEP_COLS = [
    "state_code",
    "county_code",
    "commodity_code",
    "commodity_name",
    "insurance_plan_code",
    "cause_of_loss_code",
    "cause_of_loss_description",
    "month_of_loss",
    "year_of_loss",
    "policies_earning_premium",
    "policies_indemnified",
    "net_planted_quantity",
    "indemnity_amount",
]


class RMALoader:
    """Download, parse, clean, and validate USDA RMA cause-of-loss data."""

    # URL patterns — RMA changes these occasionally, so we try several
    _URL_PATTERNS = [
        "https://www.rma.usda.gov/-/media/RMA/Cause-of-Loss/ColData{year}.ashx",
        "https://www.rma.usda.gov/-/media/RMA/Cause-of-Loss/coldat{year}.zip",
        "https://www.rma.usda.gov/data-tools/summary-of-business/cause-of-loss/coldat{year}.zip",
        "https://www.rma.usda.gov/-/media/RMAWeb/Cause-of-Loss/ColData{year}.ashx",
    ]

    def __init__(self, session=None):
        self.session = session or create_retry_session(retries=5, backoff_factor=2.0)

    # ------------------------------------------------------------------
    # Download
    # ------------------------------------------------------------------

    def download(
        self,
        years: Union[range, List[int]],
        output_dir: Path,
    ) -> List[Path]:
        """Download cause-of-loss zip/text files for each year.

        Tries multiple URL patterns per year. Downloaded files are saved to
        *output_dir* and a list of local paths is returned.
        """
        output_dir = ensure_directory(output_dir)
        downloaded = []

        for year in years:
            dest = output_dir / f"coldata_{year}.zip"
            if dest.exists():
                logger.info("Already downloaded: %s", dest.name)
                downloaded.append(dest)
                continue

            success = False
            for pattern in self._URL_PATTERNS:
                url = pattern.format(year=year)
                logger.info("Trying %s", url)
                try:
                    resp = self.session.get(url, timeout=180)
                    if resp.status_code == 200 and len(resp.content) > 1000:
                        dest.write_bytes(resp.content)
                        logger.info("Downloaded year %d (%d bytes)", year, len(resp.content))
                        downloaded.append(dest)
                        success = True
                        break
                except Exception as exc:
                    logger.debug("URL failed: %s — %s", url, exc)

            if not success:
                logger.warning("Could not download RMA data for year %d", year)

        logger.info("Downloaded %d / %d year files", len(downloaded), len(list(years)))
        return downloaded

    # ------------------------------------------------------------------
    # Parse raw files
    # ------------------------------------------------------------------

    def parse_raw(self, raw_dir: Path) -> pd.DataFrame:
        """Parse all downloaded RMA files in *raw_dir* into one DataFrame.

        Handles both zip archives and bare text/CSV files. Automatically
        detects the column layout version.
        """
        raw_dir = Path(raw_dir)
        frames = []

        for fp in sorted(raw_dir.glob("coldata_*")):
            logger.info("Parsing %s", fp.name)
            try:
                df = self._read_single_file(fp)
                frames.append(df)
                logger.info("  -> %d rows", len(df))
            except Exception as exc:
                logger.error("Failed to parse %s: %s", fp.name, exc)

        if not frames:
            raise FileNotFoundError(f"No parseable RMA files found in {raw_dir}")

        combined = pd.concat(frames, ignore_index=True)
        logger.info("Total raw rows parsed: %d", len(combined))
        return combined

    def _read_single_file(self, path: Path) -> pd.DataFrame:
        """Read a single RMA file (zip or text)."""
        raw_text = None

        if path.suffix in (".zip", ".ashx"):
            try:
                with zipfile.ZipFile(path) as zf:
                    # Pick the first text/csv file inside the archive
                    candidates = [
                        n for n in zf.namelist()
                        if n.lower().endswith((".txt", ".csv"))
                    ]
                    if not candidates:
                        candidates = zf.namelist()
                    with zf.open(candidates[0]) as f:
                        raw_text = f.read().decode("latin-1", errors="replace")
            except zipfile.BadZipFile:
                # Maybe it was served as plain text despite the extension
                raw_text = path.read_text(encoding="latin-1", errors="replace")
        else:
            raw_text = path.read_text(encoding="latin-1", errors="replace")

        # Detect delimiter — RMA uses pipe (|) in most years
        first_line = raw_text.split("\n")[0]
        if "|" in first_line:
            sep = "|"
        elif "\t" in first_line:
            sep = "\t"
        else:
            sep = ","

        df = pd.read_csv(
            io.StringIO(raw_text),
            sep=sep,
            header=None,
            dtype=str,
            on_bad_lines="skip",
        )

        # Assign column names based on number of columns
        if df.shape[1] == len(COL_NAMES_V2):
            df.columns = COL_NAMES_V2
        elif df.shape[1] == len(COL_NAMES_V1):
            df.columns = COL_NAMES_V1
        else:
            # Best-effort: trim or pad to V2 layout
            logger.warning(
                "Unexpected column count %d in %s (expected %d or %d). "
                "Attempting best-effort parse.",
                df.shape[1], path.name, len(COL_NAMES_V2), len(COL_NAMES_V1),
            )
            if df.shape[1] > len(COL_NAMES_V2):
                df = df.iloc[:, : len(COL_NAMES_V2)]
                df.columns = COL_NAMES_V2
            else:
                col_names = COL_NAMES_V2[: df.shape[1]]
                df.columns = col_names

        return df

    # ------------------------------------------------------------------
    # Clean
    # ------------------------------------------------------------------

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize types, filter commodities, build FIPS, categorize causes.

        Steps:
        1. Cast numeric columns from strings.
        2. Build 5-digit FIPS from state_code + county_code.
        3. Filter to target commodities (CORN, SOYBEANS, WHEAT).
        4. Map cause-of-loss codes to six categories.
        5. Drop records with missing indemnity.
        """
        df = df.copy()

        # --- Numeric conversions ---
        int_cols = [
            "state_code", "county_code", "commodity_code",
            "cause_of_loss_code", "month_of_loss", "year_of_loss",
            "policies_earning_premium", "policies_indemnified",
        ]
        for col in int_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        float_cols = ["indemnity_amount", "net_planted_quantity"]
        for col in float_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # --- Drop rows without critical fields ---
        before = len(df)
        df = df.dropna(subset=["state_code", "county_code", "year_of_loss"])
        logger.info("Dropped %d rows with missing state/county/year", before - len(df))

        # --- Build 5-digit FIPS ---
        df["fips"] = df.apply(
            lambda r: format_fips(r["state_code"], r["county_code"]), axis=1
        )

        # --- Filter to target commodities ---
        # Match on commodity_name (case-insensitive) or commodity_code
        target_codes = {COMMODITY_CODES.get(c) for c in TARGET_COMMODITIES}
        # Also handle wheat variants
        target_codes.update({11, 12, 13})

        df["commodity_name_clean"] = (
            df["commodity_name"]
            .astype(str)
            .str.strip()
            .str.upper()
        )

        name_mask = df["commodity_name_clean"].str.contains(
            "CORN|SOYBEAN|WHEAT", na=False
        )
        code_mask = df["commodity_code"].isin(target_codes)
        df = df[name_mask | code_mask].copy()
        logger.info("Filtered to target commodities: %d rows", len(df))

        # Normalize commodity names
        def _normalize_commodity(name: str) -> str:
            name = str(name).upper().strip()
            if "CORN" in name:
                return "CORN"
            if "SOYBEAN" in name:
                return "SOYBEANS"
            if "WHEAT" in name:
                return "WHEAT"
            return name

        df["commodity"] = df["commodity_name_clean"].apply(_normalize_commodity)

        # --- Map cause-of-loss codes ---
        df["cause_category"] = df["cause_of_loss_code"].apply(
            lambda x: map_cause_code(int(x)) if pd.notna(x) else "OTHER"
        )

        # --- Clean year and month ---
        df["year_of_loss"] = df["year_of_loss"].astype(int)
        df["month_of_loss"] = df["month_of_loss"].fillna(0).astype(int)

        # --- Fill missing indemnity with 0 ---
        df["indemnity_amount"] = df["indemnity_amount"].fillna(0.0)

        # --- Select final columns ---
        keep = [
            "fips", "state_code", "county_code", "commodity", "commodity_code",
            "insurance_plan_code", "cause_of_loss_code", "cause_of_loss_description",
            "cause_category", "month_of_loss", "year_of_loss",
            "policies_earning_premium", "policies_indemnified",
            "net_planted_quantity", "indemnity_amount",
        ]
        available = [c for c in keep if c in df.columns]
        df = df[available].reset_index(drop=True)

        logger.info("Cleaned RMA data: %d rows, %d columns", *df.shape)
        return df

    # ------------------------------------------------------------------
    # Validate
    # ------------------------------------------------------------------

    def validate(self, df: pd.DataFrame) -> None:
        """Run validation assertions on cleaned RMA data.

        Checks:
        - No null indemnity amounts.
        - All expected states are present (at least 40 of 50).
        - FIPS codes are well-formed.
        - Commodity set is exactly {CORN, SOYBEANS, WHEAT}.
        - Year range spans the expected period.
        """
        # Null indemnities
        null_indem = df["indemnity_amount"].isna().sum()
        assert null_indem == 0, f"Found {null_indem} null indemnity amounts"

        # State coverage
        states_present = df["fips"].str[:2].nunique()
        assert states_present >= 40, (
            f"Only {states_present} states present — expected >= 40. "
            f"Missing: {set(STATE_FIPS_TO_NAME.keys()) - set(df['fips'].str[:2].unique())}"
        )
        logger.info("State coverage: %d states", states_present)

        # FIPS validity
        validate_fips(df, fips_col="fips")

        # Commodity set
        commodities = set(df["commodity"].unique())
        expected = {"CORN", "SOYBEANS", "WHEAT"}
        assert commodities == expected, (
            f"Commodity mismatch. Got {commodities}, expected {expected}"
        )

        # Year range
        years = sorted(df["year_of_loss"].unique())
        logger.info("Year range: %d - %d (%d years)", years[0], years[-1], len(years))

        # Basic row-count sanity
        assert len(df) > 10_000, f"Suspiciously few rows: {len(df)}"

        logger.info("All RMA validation checks passed")

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    def save(self, df: pd.DataFrame, output_path: Path) -> None:
        """Save cleaned RMA data as Parquet, partitioned by year_of_loss."""
        save_parquet(df, output_path, partition_cols=["year_of_loss"])
        logger.info("Saved RMA data to %s", output_path)

    # ------------------------------------------------------------------
    # Convenience: full pipeline
    # ------------------------------------------------------------------

    def run(
        self,
        years: Union[range, List[int]],
        raw_dir: Path,
        output_path: Path,
    ) -> pd.DataFrame:
        """Download, parse, clean, validate, and save in one call."""
        self.download(years, raw_dir)
        df = self.parse_raw(raw_dir)
        df = self.clean(df)
        self.validate(df)
        self.save(df, output_path)
        return df
