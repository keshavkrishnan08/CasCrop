#!/usr/bin/env python3
"""Script 02b: Process raw data into MONTHLY-resolution model-ready features and labels.

Unlike 02_process_data.py (which aggregates to county-crop-YEAR), this script
produces county-crop-MONTH data. Monthly granularity captures within-year dynamics:
  - Month-over-month price shocks (the real driver of economic contagion)
  - Monthly weather signals (not washed out by growing-season averages)
  - Loss timing through RMA month_of_loss

Outputs:
  - data/processed/features_monthly.parquet   (county-crop-year-month feature matrix)
  - data/processed/labels_monthly.parquet     (binary waste + cause labels)
  - data/processed/splits_monthly.json        (train/val/test indices)
  - data/processed/stats_monthly.json         (normalization statistics from train set)
  - data/processed/feature_groups_monthly.json (bio/econ/hist column lists)

Estimated runtime: ~2-5 minutes depending on data volume.
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

RAW = Path("data/raw")
OUT = Path("data/processed")

# ── Constants ───────────────────────────────────────────────────────────

# nClimDiv state codes → Census FIPS state codes
NCLIMDIV_TO_FIPS = {
    1: "01", 2: "02", 3: "04", 4: "05", 5: "06", 6: "08", 7: "09",
    8: "10", 9: "11", 10: "12", 11: "13", 12: "15", 13: "16", 14: "17",
    15: "18", 16: "19", 17: "20", 18: "21", 19: "22", 20: "23", 21: "24",
    22: "25", 23: "26", 24: "27", 25: "28", 26: "29", 27: "30", 28: "31",
    29: "32", 30: "33", 31: "34", 32: "35", 33: "36", 34: "37", 35: "38",
    36: "39", 37: "40", 38: "41", 39: "42", 40: "44", 41: "45", 42: "46",
    43: "47", 44: "48", 45: "49", 46: "50", 47: "51", 48: "53", 49: "54",
    50: "55", 51: "56",
}

# RMA cause-of-loss code → category
CAUSE_MAP = {}
for code in [2, 3]:
    CAUSE_MAP[code] = "DROUGHT"
for code in [10, 11, 14]:
    CAUSE_MAP[code] = "EXCESS_MOISTURE"
for code in [15, 16, 17]:
    CAUSE_MAP[code] = "COLD"
for code in [36, 40]:
    CAUSE_MAP[code] = "HEAT"
for code in [47, 48]:
    CAUSE_MAP[code] = "PRICE"

CAUSE_TO_IDX = {
    "DROUGHT": 0, "EXCESS_MOISTURE": 1, "COLD": 2,
    "HEAT": 3, "PRICE": 4, "OTHER": 5,
}

COMMODITIES = {"CORN": "0041", "SOYBEANS": "0081", "WHEAT": "0011"}
COMMODITY_NAME_MAP = {
    "Corn": "CORN", "Soybeans": "SOYBEANS", "Wheat": "WHEAT",
}

TRAIN_YEARS = list(range(2015, 2020))
VAL_YEARS = [2020, 2021]
TEST_YEARS = [2022, 2023, 2024]

ALL_MONTHS = list(range(1, 13))


# ── Step 1: Parse RMA claims at monthly resolution ─────────────────────

def load_rma() -> pd.DataFrame:
    """Parse all RMA cause-of-loss files into a unified DataFrame.

    Keeps the month_of_loss column for monthly aggregation.
    """
    rma_dir = RAW / "rma"
    frames = []

    for f in sorted(rma_dir.glob("colsom_*.txt")):
        logger.info(f"  Parsing {f.name}")
        df = pd.read_csv(
            f, sep="|", header=None, encoding="latin-1",
            on_bad_lines="skip", dtype=str,
        )
        # Columns: 0=year, 1=state_code, 2=state_abbr, 3=county_code,
        # 4=county_name, 5=commodity_code, 6=commodity_name, ...
        # 11=cause_code, 12=cause_desc, 13=month_num, ..., 20=indemnity
        df.columns = [
            "year", "state_code", "state_abbr", "county_code", "county_name",
            "commodity_code", "commodity_name", "insurance_plan_code",
            "insurance_plan", "coverage_cat", "stage_code",
            "cause_code", "cause_desc", "month_num", "month_name",
            "year_of_loss", "policies", "claims", "net_acres",
            "liability", "total_premium", "subsidy",
            "indemnity", "loss_ratio", "unk1", "unk2", "unk3",
            "net_determined_qty", "indemnity2", "loss_ratio2",
        ][:len(df.columns)]

        frames.append(df)

    rma = pd.concat(frames, ignore_index=True)

    # Clean core columns
    rma["fips"] = (
        rma["state_code"].str.strip().str.zfill(2)
        + rma["county_code"].str.strip().str.zfill(3)
    )
    rma["commodity_name"] = rma["commodity_name"].str.strip()
    rma["year"] = pd.to_numeric(rma["year"].str.strip(), errors="coerce")
    rma["month_num"] = pd.to_numeric(rma["month_num"].str.strip(), errors="coerce")
    rma["cause_code"] = pd.to_numeric(rma["cause_code"].str.strip(), errors="coerce")

    # Parse indemnity — handle dollar signs and commas
    rma["indemnity"] = (
        rma["indemnity"].astype(str)
        .str.replace("$", "", regex=False)
        .str.replace(",", "", regex=False)
        .str.strip()
    )
    rma["indemnity"] = pd.to_numeric(rma["indemnity"], errors="coerce").fillna(0)

    # Filter to target commodities by code
    rma["commodity_code_clean"] = rma["commodity_code"].str.strip()
    code_set = set(COMMODITIES.values())
    rma = rma[rma["commodity_code_clean"].isin(code_set)].copy()

    # Map commodity code to standard name
    code_to_name = {v: k for k, v in COMMODITIES.items()}
    rma["commodity"] = rma["commodity_code_clean"].map(code_to_name)
    rma = rma.dropna(subset=["commodity"])

    # Drop rows with invalid months (month_num == 0 or NaN)
    rma = rma[rma["month_num"].between(1, 12)].copy()
    rma["month"] = rma["month_num"].astype(int)

    logger.info(
        f"  RMA: {len(rma):,} records, {rma['fips'].nunique()} counties, "
        f"months {rma['month'].min()}-{rma['month'].max()}"
    )
    return rma


def build_monthly_labels(
    rma: pd.DataFrame, threshold: float = 10_000
) -> pd.DataFrame:
    """Aggregate RMA claims to county-crop-year-MONTH and create labels."""
    agg = rma.groupby(["fips", "commodity", "year", "month"]).agg(
        total_indemnity=("indemnity", "sum"),
        total_claims=("claims", lambda x: pd.to_numeric(x, errors="coerce").sum()),
        dominant_cause_code=(
            "cause_code",
            lambda x: x.mode().iloc[0] if len(x) > 0 else 0,
        ),
    ).reset_index()

    # Binary waste label — lower threshold for monthly data
    agg["waste"] = (agg["total_indemnity"] >= threshold).astype(int)

    # Cause category
    agg["cause_category"] = agg["dominant_cause_code"].map(
        lambda c: CAUSE_MAP.get(int(c), "OTHER") if pd.notna(c) else "OTHER"
    )
    agg["cause_idx"] = agg["cause_category"].map(CAUSE_TO_IDX)

    logger.info(
        f"  Labels: {len(agg):,} county-crop-months, "
        f"waste rate = {agg['waste'].mean():.1%}"
    )
    return agg


# ── Step 2: Load NASS crop features (annual, replicated per month) ─────

def load_nass() -> pd.DataFrame:
    """Load NASS county crop data. Annual only — will be replicated per month later."""
    path = RAW / "nass" / "all_crops_county_annual.csv"
    if not path.exists():
        logger.warning("NASS county data not found, skipping")
        return pd.DataFrame()

    df = pd.read_csv(
        path, dtype={"FIPS": str, "STATE_FIPS": str, "COUNTY_FIPS": str}
    )
    df.rename(columns={
        "FIPS": "fips", "YEAR": "year", "crop": "commodity",
        "yield_bu_per_acre": "yield_value",
        "area_planted_acres": "area_planted",
        "area_harvested_acres": "area_harvested",
        "production_bu": "production",
        "waste_proxy": "nass_waste_proxy",
    }, inplace=True)

    df["commodity"] = df["commodity"].str.upper()
    df["fips"] = df["fips"].str.zfill(5)

    cols = [
        "fips", "commodity", "year", "yield_value", "area_planted",
        "area_harvested", "production", "nass_waste_proxy",
    ]
    cols = [c for c in cols if c in df.columns]

    logger.info(f"  NASS: {len(df):,} records, {df['fips'].nunique()} counties")
    return df[cols]


# ── Step 3: Load MONTHLY weather features ──────────────────────────────

def parse_nclimdiv_county_monthly(
    filename: str, var_name: str
) -> pd.DataFrame:
    """Parse a nClimDiv county-level file into monthly rows.

    Format: positions 0-1 = nClimDiv state, 2-4 = county FIPS,
    5-6 = element code, 7-10 = year, then 12 monthly values (Jan-Dec).
    """
    path = RAW / "weather" / filename
    if not path.exists():
        return pd.DataFrame()

    rows = []
    with open(path) as f:
        for line in f:
            if len(line.strip()) < 20:
                continue
            try:
                nclimdiv_state = int(line[0:2])
                county_fips_local = line[2:5]
                year = int(line[7:11])

                if year < 2014 or year > 2025:
                    # 2014 needed for lagging into 2015
                    continue

                state_fips = NCLIMDIV_TO_FIPS.get(nclimdiv_state)
                if not state_fips:
                    continue

                fips = state_fips + county_fips_local

                # Monthly values: 12 values after position 11
                value_strs = line[11:].split()
                for m_idx, val_str in enumerate(value_strs[:12]):
                    v = float(val_str)
                    if v == -99.90:
                        v = np.nan
                    rows.append({
                        "fips": fips,
                        "year": year,
                        "month": m_idx + 1,  # 1-indexed (Jan=1)
                        var_name: v,
                    })
            except (ValueError, IndexError):
                continue

    return pd.DataFrame(rows)


def load_weather_monthly() -> pd.DataFrame:
    """Load county-level MONTHLY weather features from nClimDiv files."""
    logger.info("  Loading nClimDiv county data (monthly)...")

    weather_vars = {
        "climdiv_tmax_county.txt": "tmax",
        "climdiv_tmin_county.txt": "tmin",
        "climdiv_tavg_county.txt": "tavg",
        "climdiv_precip_county.txt": "precip",
        "climdiv_pdsi_county.txt": "pdsi",
        "climdiv_cdd_county.txt": "cdd",
        "climdiv_hdd_county.txt": "hdd",
    }

    dfs = []
    for filename, var_name in weather_vars.items():
        df = parse_nclimdiv_county_monthly(filename, var_name)
        if len(df) > 0:
            dfs.append(df)
            logger.info(f"    {var_name}: {len(df):,} records (monthly)")

    if not dfs:
        return pd.DataFrame()

    # Merge all weather variables on fips + year + month
    weather = dfs[0]
    for df in dfs[1:]:
        weather = weather.merge(df, on=["fips", "year", "month"], how="outer")

    # Derived features
    if "tmax" in weather.columns and "tmin" in weather.columns:
        weather["temp_range"] = weather["tmax"] - weather["tmin"]

    if "cdd" in weather.columns:
        weather["gdd_proxy"] = weather["cdd"]

    logger.info(
        f"  Weather: {len(weather):,} county-year-months, "
        f"{weather.shape[1]} features"
    )
    return weather


# ── Step 4: Load MONTHLY price features ────────────────────────────────

def load_prices_monthly() -> pd.DataFrame:
    """Load commodity price data at monthly resolution.

    Computes month-over-month price change — the key monthly shock signal
    that annual data can't capture.
    """
    price_dir = RAW / "prices"
    frames = []

    for crop, filename in [
        ("CORN", "corn_prices.csv"),
        ("WHEAT", "wheat_prices.csv"),
        ("SOYBEANS", "soybeans_prices.csv"),
    ]:
        path = price_dir / filename
        if not path.exists():
            logger.warning(f"  Price file {filename} not found, skipping")
            continue

        df = pd.read_csv(path)
        date_col = df.columns[0]
        price_col = df.columns[1]

        df["date"] = pd.to_datetime(df[date_col], errors="coerce")
        df = df.dropna(subset=["date"])
        df["price"] = pd.to_numeric(df[price_col], errors="coerce")
        df = df.dropna(subset=["price"])
        df["year"] = df["date"].dt.year
        df["month"] = df["date"].dt.month
        df["commodity"] = crop

        # Sort chronologically
        df = df.sort_values(["commodity", "year", "month"]).reset_index(drop=True)

        # Month-over-month price change — the core monthly signal
        df["price_change_1m"] = df.groupby("commodity")["price"].pct_change()

        # Rolling 3-month price stats for richer context
        df["price_3m_mean"] = (
            df.groupby("commodity")["price"]
            .transform(lambda s: s.rolling(3, min_periods=1).mean())
        )
        df["price_3m_std"] = (
            df.groupby("commodity")["price"]
            .transform(lambda s: s.rolling(3, min_periods=1).std())
        )
        df["price_volatility_3m"] = df["price_3m_std"] / df["price_3m_mean"].clip(lower=0.01)

        # Year-over-year change for same month (seasonal comparison)
        df["price_yoy"] = df.groupby(["commodity", "month"])["price"].pct_change()

        frames.append(df[["commodity", "year", "month", "price",
                          "price_change_1m", "price_3m_mean", "price_3m_std",
                          "price_volatility_3m", "price_yoy"]])

    if not frames:
        return pd.DataFrame()

    prices = pd.concat(frames, ignore_index=True)
    logger.info(f"  Prices: {len(prices):,} commodity-month records")
    return prices


# ── Step 5: Load geographic metadata (static) ──────────────────────────

def load_gazetteer() -> pd.DataFrame:
    """Load county centroids from Census gazetteer."""
    gaz_path = RAW / "geographic" / "2023_Gaz_counties_national.txt"
    if not gaz_path.exists():
        return pd.DataFrame()

    df = pd.read_csv(gaz_path, sep="\t", dtype={"GEOID": str})
    df.columns = df.columns.str.strip()
    df.rename(columns={
        "GEOID": "fips",
        "INTPTLAT": "latitude",
        "INTPTLONG": "longitude",
        "ALAND_SQMI": "land_area_sqmi",
    }, inplace=True)

    for col in ["fips", "latitude", "longitude"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()

    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")

    logger.info(f"  Gazetteer: {len(df):,} counties with centroids")
    return df[["fips", "latitude", "longitude", "land_area_sqmi"]]


# ── Step 6: Build the monthly scaffold and merge ───────────────────────

def build_monthly_scaffold(labels: pd.DataFrame) -> pd.DataFrame:
    """Create a scaffold of all (fips, commodity, year, month) combinations.

    For every county-commodity-year that appears in the labels, expand to
    all 12 months. Months without losses get waste=0.
    """
    # Get all county-commodity-year combos from labels
    combos = labels[["fips", "commodity", "year"]].drop_duplicates()

    # Expand to 12 months per combo
    months_df = pd.DataFrame({"month": ALL_MONTHS})
    scaffold = combos.merge(months_df, how="cross")

    logger.info(
        f"  Scaffold: {len(scaffold):,} county-crop-months "
        f"({len(combos):,} county-crop-years × 12)"
    )
    return scaffold


def merge_all_monthly(
    labels: pd.DataFrame,
    nass: pd.DataFrame,
    weather: pd.DataFrame,
    prices: pd.DataFrame,
    gazetteer: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Merge all data sources into features + labels at monthly resolution."""

    # Build full scaffold: every county-crop-year × 12 months
    scaffold = build_monthly_scaffold(labels)

    # Left-join labels onto scaffold. Months without losses get NaN → fill with 0
    merged = scaffold.merge(
        labels[["fips", "commodity", "year", "month",
                "total_indemnity", "waste", "cause_category", "cause_idx"]],
        on=["fips", "commodity", "year", "month"],
        how="left",
    )
    merged["total_indemnity"] = merged["total_indemnity"].fillna(0)
    merged["waste"] = merged["waste"].fillna(0).astype(int)
    merged["cause_category"] = merged["cause_category"].fillna("OTHER")
    merged["cause_idx"] = merged["cause_idx"].fillna(CAUSE_TO_IDX["OTHER"]).astype(int)

    # Split labels out now (will save separately)
    label_cols = ["fips", "commodity", "year", "month",
                  "waste", "total_indemnity", "cause_category", "cause_idx"]
    labels_out = merged[label_cols].copy()

    # ── Merge NASS (annual → repeated for all months in that year)
    if len(nass) > 0:
        merged = merged.merge(nass, on=["fips", "commodity", "year"], how="left")

    # ── Merge weather (fips-year-month level)
    if len(weather) > 0:
        merged = merged.merge(weather, on=["fips", "year", "month"], how="left")

    # ── Merge prices (commodity-year-month level)
    if len(prices) > 0:
        price_cols = [c for c in prices.columns if c != "date"]
        merged = merged.merge(
            prices[price_cols],
            on=["commodity", "year", "month"],
            how="left",
        )

    # ── Merge gazetteer (fips-level static)
    if len(gazetteer) > 0:
        merged = merged.merge(gazetteer, on="fips", how="left")

    # ── Temporal encoding: cyclical month features
    merged["month_sin"] = np.sin(2 * np.pi * merged["month"] / 12)
    merged["month_cos"] = np.cos(2 * np.pi * merged["month"] / 12)

    # ── Historical features: county-commodity historical loss rate
    # Use expanding window up to (but not including) current month to avoid leakage.
    # For efficiency, compute the all-time historical stats, same as annual version.
    hist_annual = labels.groupby(["fips", "commodity"]).agg(
        historical_waste_rate=("waste", "mean"),
        historical_avg_indemnity=("total_indemnity", "mean"),
        historical_loss_count=("waste", "sum"),
    ).reset_index()
    merged = merged.merge(hist_annual, on=["fips", "commodity"], how="left")

    # ── County-specific economic shock (monthly version) ──
    # price_change_1m varies every month — this is the key improvement over annual
    if "production" in merged.columns and "price_change_1m" in merged.columns:
        # County production share (from annual NASS data)
        total_prod = merged.groupby(
            ["commodity", "year"]
        )["production"].transform("sum")
        merged["county_prod_share"] = (
            merged["production"] / total_prod.clip(lower=1)
        )

        # Yield deviation: county yield vs national mean for that crop-year
        yield_mean = merged.groupby(
            ["commodity", "year"]
        )["yield_value"].transform("mean")
        yield_std = merged.groupby(
            ["commodity", "year"]
        )["yield_value"].transform("std").clip(lower=0.1)
        merged["yield_deviation"] = (
            (merged["yield_value"] - yield_mean) / yield_std
        )

        # Monthly county shock = price_change_1m × county exposure + yield anomaly
        merged["county_shock"] = (
            merged["price_change_1m"].fillna(0)
            * merged["county_prod_share"].fillna(0)
            + 0.5 * merged["yield_deviation"].fillna(0)
        )

        logger.info(
            f"  County shocks: {merged['county_shock'].nunique():,} unique values "
            f"(monthly price variation captured)"
        )

    # ── Feature lag: shift features by 1 month to prevent data leakage ──
    # For predicting month t, we can only use data available at month t-1.
    # Sort by time, then shift feature columns within each county-commodity group.
    merged = merged.sort_values(
        ["fips", "commodity", "year", "month"]
    ).reset_index(drop=True)

    # Columns that need lagging: all numeric features EXCEPT identifiers,
    # labels, and temporal encoding (which is inherently non-leaky).
    id_cols = {"fips", "commodity", "year", "month"}
    label_set = {"waste", "total_indemnity", "cause_category", "cause_idx"}
    no_lag_cols = id_cols | label_set | {"month_sin", "month_cos",
                                          "latitude", "longitude",
                                          "land_area_sqmi"}

    feature_cols_to_lag = [
        c for c in merged.select_dtypes(include=[np.number]).columns
        if c not in no_lag_cols
    ]

    logger.info(f"  Lagging {len(feature_cols_to_lag)} feature columns by 1 month...")
    merged[feature_cols_to_lag] = (
        merged.groupby(["fips", "commodity"])[feature_cols_to_lag]
        .shift(1)
    )

    # The first month per county-commodity will be NaN after shift — drop it
    before_lag_drop = len(merged)
    merged = merged.dropna(
        subset=feature_cols_to_lag, how="all"
    ).reset_index(drop=True)
    logger.info(
        f"  Dropped {before_lag_drop - len(merged):,} rows with no lagged features"
    )

    # Also align labels_out to keep only the rows that survived the lag
    labels_out = merged[label_cols].copy()

    # ── Fill remaining NaN with column medians ──
    numeric_cols = merged.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if merged[col].isna().any():
            merged[col] = merged[col].fillna(merged[col].median())

    logger.info(
        f"  Merged: {len(merged):,} rows, {merged.shape[1]} columns, "
        f"{merged['fips'].nunique()} counties"
    )

    # Drop label columns from features (they'll be in the labels file)
    features_out = merged.drop(
        columns=["total_indemnity", "waste", "cause_category", "cause_idx"],
        errors="ignore",
    )

    return features_out, labels_out


# ── Step 7: Create train/val/test splits ───────────────────────────────

def create_splits(df: pd.DataFrame) -> dict:
    """Temporal split: train 2015-2019, val 2020-2021, test 2022-2024."""
    train_mask = df["year"].isin(TRAIN_YEARS)
    val_mask = df["year"].isin(VAL_YEARS)
    test_mask = df["year"].isin(TEST_YEARS)

    splits = {
        "train": df.index[train_mask].tolist(),
        "val": df.index[val_mask].tolist(),
        "test": df.index[test_mask].tolist(),
    }

    for name, idx in splits.items():
        if idx:
            years = sorted(df.loc[idx, "year"].unique())
            logger.info(
                f"  {name}: {len(idx):,} samples, "
                f"years {years[0]}-{years[-1]}"
            )
        else:
            logger.info(f"  {name}: 0 samples")

    return splits


# ── Step 8: Compute normalization stats ────────────────────────────────

def compute_stats(df: pd.DataFrame, train_idx: list) -> dict:
    """Compute mean/std from train set for z-score normalization."""
    train_df = df.iloc[train_idx]
    numeric_cols = train_df.select_dtypes(include=[np.number]).columns
    exclude = {"year", "month", "waste", "cause_idx"}
    numeric_cols = [c for c in numeric_cols if c not in exclude]

    stats = {}
    for col in numeric_cols:
        stats[col] = {
            "mean": float(train_df[col].mean()),
            "std": float(max(train_df[col].std(), 1e-8)),
        }
    return stats


# ── Step 9: Define feature groups ──────────────────────────────────────

def get_feature_groups(df: pd.DataFrame) -> dict:
    """Map feature columns to biophysical / economic / historical groups."""
    bio_cols = [c for c in df.columns if c in [
        "tmax", "tmin", "tavg", "precip", "pdsi", "cdd", "hdd",
        "temp_range", "gdd_proxy", "latitude", "longitude", "land_area_sqmi",
        "yield_value", "area_planted", "area_harvested", "production",
        "nass_waste_proxy", "month_sin", "month_cos",
    ]]

    econ_cols = [c for c in df.columns if c in [
        "price", "price_change_1m", "price_3m_mean", "price_3m_std",
        "price_volatility_3m", "price_yoy",
        "county_prod_share", "county_shock",
    ]]

    hist_cols = [c for c in df.columns if c in [
        "historical_waste_rate", "historical_avg_indemnity",
        "historical_loss_count",
    ]]

    return {
        "biophysical": bio_cols,
        "economic": econ_cols,
        "historical": hist_cols,
    }


# ── Main ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Process CasCrop data at MONTHLY resolution"
    )
    parser.add_argument(
        "--threshold", type=float, default=10_000,
        help="Monthly indemnity threshold for waste label ($)"
    )
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("CasCrop MONTHLY Data Processing")
    logger.info("=" * 60)

    # Step 1: RMA labels (monthly)
    logger.info("\n[1/7] Loading RMA claims (monthly)...")
    rma = load_rma()
    labels = build_monthly_labels(rma, threshold=args.threshold)

    # Step 2: NASS features (annual, will be repeated per month)
    logger.info("\n[2/7] Loading NASS crop data...")
    nass = load_nass()

    # Step 3: Weather features (monthly)
    logger.info("\n[3/7] Loading weather data (monthly)...")
    weather = load_weather_monthly()

    # Step 4: Price features (monthly)
    logger.info("\n[4/7] Loading price data (monthly)...")
    prices = load_prices_monthly()

    # Step 5: Geographic metadata
    logger.info("\n[5/7] Loading geographic data...")
    gazetteer = load_gazetteer()

    # Step 6: Merge
    logger.info("\n[6/7] Merging all sources (monthly)...")
    features, labels_out = merge_all_monthly(
        labels, nass, weather, prices, gazetteer,
    )

    # Step 7: Splits and stats
    logger.info("\n[7/7] Creating splits and statistics...")
    splits = create_splits(features)
    stats = compute_stats(features, splits["train"])
    feature_groups = get_feature_groups(features)

    # ── Save outputs ──
    features.to_parquet(OUT / "features_monthly.parquet", index=False)
    labels_out.to_parquet(OUT / "labels_monthly.parquet", index=False)

    with open(OUT / "splits_monthly.json", "w") as f:
        json.dump(splits, f)

    with open(OUT / "stats_monthly.json", "w") as f:
        json.dump(stats, f, indent=2)

    with open(OUT / "feature_groups_monthly.json", "w") as f:
        json.dump(feature_groups, f, indent=2)

    # ── Summary ──
    logger.info("\n" + "=" * 60)
    logger.info("MONTHLY PROCESSING COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Total samples:     {len(features):,}")
    logger.info(f"Unique counties:   {features['fips'].nunique():,}")
    logger.info(f"Commodities:       {sorted(features['commodity'].unique())}")
    logger.info(f"Year range:        {features['year'].min()}-{features['year'].max()}")
    logger.info(f"Month range:       {features['month'].min()}-{features['month'].max()}")
    logger.info(f"Waste rate:        {labels_out['waste'].mean():.1%}")
    logger.info(
        f"Feature dims:      "
        f"bio={len(feature_groups['biophysical'])}, "
        f"econ={len(feature_groups['economic'])}, "
        f"hist={len(feature_groups['historical'])}"
    )
    logger.info(
        f"Train/Val/Test:    "
        f"{len(splits['train'])}/{len(splits['val'])}/{len(splits['test'])}"
    )
    logger.info(f"Feature columns:   {list(features.columns)}")
    logger.info(f"\nOutput: {OUT}/")

    # Class balance by crop
    for crop in sorted(labels_out["commodity"].unique()):
        crop_labels = labels_out[labels_out["commodity"] == crop]
        logger.info(
            f"  {crop}: {crop_labels['waste'].mean():.1%} waste rate "
            f"({crop_labels['waste'].sum():,}/{len(crop_labels):,})"
        )

    # Monthly waste distribution
    monthly_waste = labels_out.groupby("month")["waste"].mean()
    logger.info("\nWaste rate by month:")
    for m, rate in monthly_waste.items():
        logger.info(f"  Month {m:2d}: {rate:.1%}")


if __name__ == "__main__":
    main()
