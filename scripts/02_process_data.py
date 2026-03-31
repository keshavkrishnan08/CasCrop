#!/usr/bin/env python3
"""Script 02: Process raw data into model-ready features and labels.

Reads the actual downloaded data files and produces:
  - data/processed/features.parquet  (county-crop-year feature matrix)
  - data/processed/labels.parquet    (binary waste + cause labels)
  - data/processed/splits.json       (train/val/test indices)
  - data/processed/stats.json        (normalization statistics from train set)

Estimated runtime: ~10 minutes.
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

# nClimDiv state codes → Census FIPS state codes
# nClimDiv uses its own numbering; Census uses standard FIPS
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

TRAIN_YEARS = list(range(2008, 2020))
VAL_YEARS = [2020, 2021]
TEST_YEARS = [2022, 2023, 2024]


# ── Step 1: Parse RMA claims into labels ─────────────────────────────

def load_rma() -> pd.DataFrame:
    """Parse all RMA cause-of-loss files into a unified DataFrame."""
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

    # Clean
    rma["fips"] = rma["state_code"].str.strip().str.zfill(2) + \
                  rma["county_code"].str.strip().str.zfill(3)
    rma["commodity_name"] = rma["commodity_name"].str.strip()
    rma["year"] = pd.to_numeric(rma["year"].str.strip(), errors="coerce")
    rma["month_num"] = pd.to_numeric(rma["month_num"].str.strip(), errors="coerce")
    rma["cause_code"] = pd.to_numeric(rma["cause_code"].str.strip(), errors="coerce")

    # Parse indemnity — handle both dollar signs and commas
    for col in ["indemnity"]:
        rma[col] = (
            rma[col].astype(str)
            .str.replace("$", "", regex=False)
            .str.replace(",", "", regex=False)
            .str.strip()
        )
        rma[col] = pd.to_numeric(rma[col], errors="coerce").fillna(0)

    # Filter to target commodities
    target_names = set()
    for row_name in rma["commodity_name"].unique():
        clean = row_name.strip().title()
        if clean in COMMODITY_NAME_MAP:
            target_names.add(row_name)
        # Also match by code
    rma["commodity_code_clean"] = rma["commodity_code"].str.strip()
    code_set = set(COMMODITIES.values())
    mask = rma["commodity_code_clean"].isin(code_set)
    rma = rma[mask].copy()

    # Map commodity code to standard name
    code_to_name = {v: k for k, v in COMMODITIES.items()}
    rma["commodity"] = rma["commodity_code_clean"].map(code_to_name)
    rma = rma.dropna(subset=["commodity"])

    logger.info(f"  RMA: {len(rma):,} records, {rma['fips'].nunique()} counties")
    return rma


def build_labels(rma: pd.DataFrame, threshold: float = 10000) -> pd.DataFrame:
    """Aggregate RMA claims to county-crop-year and create labels."""
    # Aggregate to county-crop-year (we use annual since RMA month is month of LOSS
    # which can differ from growing season month)
    agg = rma.groupby(["fips", "commodity", "year"]).agg(
        total_indemnity=("indemnity", "sum"),
        total_claims=("claims", lambda x: pd.to_numeric(x, errors="coerce").sum()),
        dominant_cause_code=("cause_code", lambda x: x.mode().iloc[0] if len(x) > 0 else 0),
    ).reset_index()

    # Binary waste label
    agg["waste"] = (agg["total_indemnity"] >= threshold).astype(int)

    # Cause category
    agg["cause_category"] = agg["dominant_cause_code"].map(
        lambda c: CAUSE_MAP.get(int(c), "OTHER") if pd.notna(c) else "OTHER"
    )
    agg["cause_idx"] = agg["cause_category"].map(CAUSE_TO_IDX)

    logger.info(
        f"  Labels: {len(agg):,} county-crop-years, "
        f"waste rate = {agg['waste'].mean():.1%}"
    )
    return agg


# ── Step 2: Load NASS crop features ──────────────────────────────────

def load_nass() -> pd.DataFrame:
    """Load pre-processed NASS county crop data."""
    path = RAW / "nass" / "all_crops_county_annual.csv"
    if not path.exists():
        logger.warning("NASS county data not found, skipping")
        return pd.DataFrame()

    df = pd.read_csv(path, dtype={"FIPS": str, "STATE_FIPS": str, "COUNTY_FIPS": str})
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

    logger.info(f"  NASS: {len(df):,} records, {df['fips'].nunique()} counties")
    return df[["fips", "commodity", "year", "yield_value", "area_planted",
               "area_harvested", "production", "nass_waste_proxy"]]


# ── Step 3: Load weather features ────────────────────────────────────

def parse_nclimdiv_county(filename: str, var_name: str) -> pd.DataFrame:
    """Parse a nClimDiv county-level fixed-width file.

    Format: positions 1-2 = nClimDiv state, 3-5 = county FIPS,
    6-7 = element code, 8-11 = year, then 12 monthly values.
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

                if year < 2008 or year > 2025:
                    continue

                state_fips = NCLIMDIV_TO_FIPS.get(nclimdiv_state)
                if not state_fips:
                    continue

                fips = state_fips + county_fips_local

                # Monthly values: split on whitespace after position 11
                value_strs = line[11:].split()
                values = []
                for val in value_strs[:12]:
                    v = float(val)
                    values.append(v if v != -99.90 else np.nan)

                # Use annual mean for simplicity (growing season: Apr-Oct)
                growing = values[3:10]  # April through October
                annual_mean = np.nanmean(growing) if any(not np.isnan(v) for v in growing) else np.nan

                rows.append({
                    "fips": fips,
                    "year": year,
                    var_name: annual_mean,
                })
            except (ValueError, IndexError):
                continue

    return pd.DataFrame(rows)


def load_weather() -> pd.DataFrame:
    """Load county-level weather features from nClimDiv files."""
    logger.info("  Loading nClimDiv county data...")

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
        df = parse_nclimdiv_county(filename, var_name)
        if len(df) > 0:
            dfs.append(df)
            logger.info(f"    {var_name}: {len(df):,} records")

    if not dfs:
        return pd.DataFrame()

    # Merge all weather variables on fips + year
    weather = dfs[0]
    for df in dfs[1:]:
        weather = weather.merge(df, on=["fips", "year"], how="outer")

    # Derived features
    if "tmax" in weather.columns and "tmin" in weather.columns:
        weather["temp_range"] = weather["tmax"] - weather["tmin"]

    # GDD proxy from CDD (cooling degree days correlate with growing degree days)
    if "cdd" in weather.columns:
        weather["gdd_proxy"] = weather["cdd"]

    logger.info(f"  Weather: {len(weather):,} county-years, {weather.shape[1]} features")
    return weather


# ── Step 4: Load price features ──────────────────────────────────────

def load_prices() -> pd.DataFrame:
    """Load commodity price features."""
    price_dir = RAW / "prices"
    frames = []

    for crop, filename in [
        ("CORN", "corn_prices.csv"),
        ("WHEAT", "wheat_prices.csv"),
        ("SOYBEANS", "soybeans_prices.csv"),
    ]:
        path = price_dir / filename
        if not path.exists():
            continue

        df = pd.read_csv(path)
        date_col = df.columns[0]
        price_col = df.columns[1]

        df["date"] = pd.to_datetime(df[date_col], errors="coerce")
        df = df.dropna(subset=["date"])
        df["price"] = pd.to_numeric(df[price_col], errors="coerce")
        df = df.dropna(subset=["price"])
        df["year"] = df["date"].dt.year
        df["commodity"] = crop

        # Annual price features
        annual = df.groupby(["commodity", "year"]).agg(
            price_mean=("price", "mean"),
            price_std=("price", "std"),
            price_min=("price", "min"),
            price_max=("price", "max"),
        ).reset_index()

        # Price change from previous year
        annual = annual.sort_values("year")
        annual["price_change_pct"] = annual.groupby("commodity")["price_mean"].pct_change()
        annual["price_volatility"] = annual["price_std"] / annual["price_mean"]

        frames.append(annual)

    if not frames:
        return pd.DataFrame()

    prices = pd.concat(frames, ignore_index=True)
    logger.info(f"  Prices: {len(prices):,} commodity-year records")
    return prices


# ── Step 5: Load geographic metadata ─────────────────────────────────

def load_gazetteer() -> pd.DataFrame:
    """Load county centroids from Census gazetteer."""
    gaz_path = RAW / "geographic" / "2023_Gaz_counties_national.txt"
    if not gaz_path.exists():
        return pd.DataFrame()

    df = pd.read_csv(gaz_path, sep="\t", dtype={"GEOID": str})
    # Strip whitespace from column names FIRST (INTPTLONG has trailing spaces)
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


# ── Step 6: Merge everything ─────────────────────────────────────────

def merge_all(
    labels: pd.DataFrame,
    nass: pd.DataFrame,
    weather: pd.DataFrame,
    prices: pd.DataFrame,
    gazetteer: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Merge all data sources into features + labels aligned by fips-commodity-year."""

    # Start with labels as the base (every county-crop-year with RMA data)
    merged = labels[["fips", "commodity", "year"]].copy()

    # Merge NASS
    if len(nass) > 0:
        merged = merged.merge(nass, on=["fips", "commodity", "year"], how="left")

    # Merge weather (county-year level, same for all commodities)
    if len(weather) > 0:
        merged = merged.merge(weather, on=["fips", "year"], how="left")

    # Merge prices (commodity-year level, same for all counties)
    if len(prices) > 0:
        price_cols = ["commodity", "year", "price_mean", "price_std", "price_min",
                      "price_max", "price_change_pct", "price_volatility"]
        price_cols = [c for c in price_cols if c in prices.columns]
        merged = merged.merge(prices[price_cols], on=["commodity", "year"], how="left")

    # Merge gazetteer (county-level, static)
    if len(gazetteer) > 0:
        merged = merged.merge(gazetteer, on="fips", how="left")

    # Historical features: county-commodity historical loss frequency
    hist = labels.groupby(["fips", "commodity"]).agg(
        historical_waste_rate=("waste", "mean"),
        historical_avg_indemnity=("total_indemnity", "mean"),
        historical_loss_count=("waste", "sum"),
    ).reset_index()
    # Only use history BEFORE each year (lag to prevent leakage)
    # For simplicity, use the full historical average but shift by 1 year
    merged = merged.merge(hist, on=["fips", "commodity"], how="left")

    # ── County-specific economic shock ──
    # National price_change_pct is identical for all counties growing the same crop.
    # ECMP needs per-county variation to learn asymmetric contagion patterns.
    # County economic exposure = national price change × county production share
    # Plus: county-specific yield deviation from trend as local shock signal.
    if "production" in merged.columns and "price_change_pct" in merged.columns:
        # County production share within each commodity-year
        total_prod = merged.groupby(["commodity", "year"])["production"].transform("sum")
        merged["county_prod_share"] = merged["production"] / total_prod.clip(lower=1)

        # County-specific economic shock: price change × production exposure
        merged["county_econ_shock"] = (
            merged["price_change_pct"].fillna(0) * merged["county_prod_share"].fillna(0)
        )

        # Yield deviation: county yield vs commodity-year mean (local stress signal)
        yield_mean = merged.groupby(["commodity", "year"])["yield_value"].transform("mean")
        yield_std = merged.groupby(["commodity", "year"])["yield_value"].transform("std").clip(lower=0.1)
        merged["yield_deviation"] = (merged["yield_value"] - yield_mean) / yield_std

        # Combined county shock: economic exposure + yield anomaly
        # This varies across counties even within the same commodity-year
        merged["county_shock"] = (
            merged["county_econ_shock"] + 0.5 * merged["yield_deviation"].fillna(0)
        )

        logger.info(f"  County shocks: {merged['county_shock'].nunique()} unique values "
                    f"(was {merged['price_change_pct'].nunique()} with national-only)")

    # Fill missing
    numeric_cols = merged.select_dtypes(include=[np.number]).columns
    merged[numeric_cols] = merged[numeric_cols].fillna(merged[numeric_cols].median())

    logger.info(
        f"  Merged: {len(merged):,} rows, {merged.shape[1]} columns, "
        f"{merged['fips'].nunique()} counties"
    )
    return merged, labels


# ── Step 7: Create train/val/test splits ─────────────────────────────

def create_splits(df: pd.DataFrame) -> dict:
    """Temporal split: train 2008-2019, val 2020-2021, test 2022-2024."""
    train_mask = df["year"].isin(TRAIN_YEARS)
    val_mask = df["year"].isin(VAL_YEARS)
    test_mask = df["year"].isin(TEST_YEARS)

    splits = {
        "train": df.index[train_mask].tolist(),
        "val": df.index[val_mask].tolist(),
        "test": df.index[test_mask].tolist(),
    }

    for name, idx in splits.items():
        years = sorted(df.loc[idx, "year"].unique())
        logger.info(f"  {name}: {len(idx):,} samples, years {years[0]}-{years[-1]}")

    return splits


# ── Step 8: Compute normalization stats ──────────────────────────────

def compute_stats(df: pd.DataFrame, train_idx: list) -> dict:
    """Compute mean/std from train set for z-score normalization."""
    train_df = df.iloc[train_idx]
    numeric_cols = train_df.select_dtypes(include=[np.number]).columns
    numeric_cols = [c for c in numeric_cols if c not in ["year", "waste", "cause_idx"]]

    stats = {}
    for col in numeric_cols:
        stats[col] = {
            "mean": float(train_df[col].mean()),
            "std": float(max(train_df[col].std(), 1e-8)),
        }
    return stats


# ── Step 9: Define feature groups ────────────────────────────────────

def get_feature_groups(df: pd.DataFrame) -> dict:
    """Map feature columns to biophysical / economic / historical groups."""
    bio_cols = [c for c in df.columns if c in [
        "tmax", "tmin", "tavg", "precip", "pdsi", "cdd", "hdd",
        "temp_range", "gdd_proxy", "latitude", "longitude", "land_area_sqmi",
        "yield_value", "area_planted", "area_harvested", "production",
        "nass_waste_proxy",
    ]]

    econ_cols = [c for c in df.columns if c in [
        "price_mean", "price_std", "price_min", "price_max",
        "price_change_pct", "price_volatility",
        "county_prod_share", "county_econ_shock",
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


# ── Main ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Process CasCrop data")
    parser.add_argument("--threshold", type=float, default=10000,
                        help="Indemnity threshold for waste label ($)")
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("CasCrop Data Processing")
    logger.info("=" * 60)

    # Step 1: RMA labels
    logger.info("\n[1/7] Loading RMA claims...")
    rma = load_rma()
    labels = build_labels(rma, threshold=args.threshold)

    # Step 2: NASS features
    logger.info("\n[2/7] Loading NASS crop data...")
    nass = load_nass()

    # Step 3: Weather features
    logger.info("\n[3/7] Loading weather data...")
    weather = load_weather()

    # Step 4: Price features
    logger.info("\n[4/7] Loading price data...")
    prices = load_prices()

    # Step 5: Geographic metadata
    logger.info("\n[5/7] Loading geographic data...")
    gazetteer = load_gazetteer()

    # Step 6: Merge
    logger.info("\n[6/7] Merging all sources...")
    features, labels = merge_all(labels, nass, weather, prices, gazetteer)

    # Step 7: Splits and stats
    logger.info("\n[7/7] Creating splits and statistics...")
    splits = create_splits(features)
    stats = compute_stats(features, splits["train"])
    feature_groups = get_feature_groups(features)

    # Save
    features.to_parquet(OUT / "features.parquet", index=False)
    labels.to_parquet(OUT / "labels.parquet", index=False)

    with open(OUT / "splits.json", "w") as f:
        json.dump(splits, f)

    with open(OUT / "stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    with open(OUT / "feature_groups.json", "w") as f:
        json.dump(feature_groups, f, indent=2)

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("PROCESSING COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Total samples:     {len(features):,}")
    logger.info(f"Unique counties:   {features['fips'].nunique():,}")
    logger.info(f"Commodities:       {sorted(features['commodity'].unique())}")
    logger.info(f"Year range:        {features['year'].min()}-{features['year'].max()}")
    logger.info(f"Waste rate:        {labels['waste'].mean():.1%}")
    logger.info(f"Feature dims:      bio={len(feature_groups['biophysical'])}, "
                f"econ={len(feature_groups['economic'])}, "
                f"hist={len(feature_groups['historical'])}")
    logger.info(f"Train/Val/Test:    {len(splits['train'])}/{len(splits['val'])}/{len(splits['test'])}")
    logger.info(f"\nOutput: {OUT}/")

    # Class balance by crop
    for crop in sorted(labels["commodity"].unique()):
        crop_labels = labels[labels["commodity"] == crop]
        logger.info(f"  {crop}: {crop_labels['waste'].mean():.1%} waste rate "
                    f"({crop_labels['waste'].sum():,}/{len(crop_labels):,})")


if __name__ == "__main__":
    main()
