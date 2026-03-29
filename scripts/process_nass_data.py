#!/usr/bin/env python3
"""
Process NASS Quick Stats bulk county-level crop data into analysis-ready CSVs.

Input: TSV files extracted from qs.crops bulk download
Output: Clean CSV files per crop with FIPS codes, year, and pivoted statistics
"""

import pandas as pd
import os
import sys

RAW_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw', 'nass')
OUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw', 'nass')

# Columns we care about
KEEP_COLS = [
    'COMMODITY_DESC', 'CLASS_DESC', 'UTIL_PRACTICE_DESC',
    'STATISTICCAT_DESC', 'UNIT_DESC',
    'STATE_FIPS_CODE', 'STATE_ALPHA', 'STATE_NAME',
    'COUNTY_CODE', 'COUNTY_NAME',
    'YEAR', 'VALUE'
]


def load_and_clean(filepath, crop_name):
    """Load a TSV and clean values."""
    print(f"Loading {crop_name} from {os.path.basename(filepath)}...")
    df = pd.read_csv(filepath, sep='\t', dtype=str, low_memory=False)
    df = df[KEEP_COLS].copy()

    # Clean VALUE: remove commas, handle (D) suppressed, (Z) zero, etc.
    df['VALUE'] = df['VALUE'].str.strip()
    df = df[~df['VALUE'].isin(['(D)', '(Z)', '(NA)', '(S)', '(L)', '(H)', ''])]
    df['VALUE'] = df['VALUE'].str.replace(',', '', regex=False)
    df['VALUE'] = pd.to_numeric(df['VALUE'], errors='coerce')
    df = df.dropna(subset=['VALUE'])

    # Build FIPS code (5-digit: state + county)
    df['STATE_FIPS'] = df['STATE_FIPS_CODE'].str.zfill(2)
    df['COUNTY_FIPS'] = df['COUNTY_CODE'].str.zfill(3)
    df['FIPS'] = df['STATE_FIPS'] + df['COUNTY_FIPS']
    df['YEAR'] = df['YEAR'].astype(int)

    # Drop rows where county code is 998 or 999 (OTHER/COMBINED counties)
    df = df[~df['COUNTY_CODE'].isin(['998', '999'])]

    print(f"  {len(df):,} clean rows for {crop_name}")
    return df


def pivot_crop(df, crop_name):
    """Pivot from long format to wide: one row per county-year with columns for each stat."""
    # For the pivot, we need a unique identifier per stat
    stat_col = 'STATISTICCAT_DESC'

    # Create pivot
    pivot = df.pivot_table(
        index=['FIPS', 'STATE_FIPS', 'STATE_ALPHA', 'STATE_NAME',
               'COUNTY_FIPS', 'COUNTY_NAME', 'YEAR'],
        columns=stat_col,
        values='VALUE',
        aggfunc='first'  # Should be unique per county-year-stat
    ).reset_index()

    # Flatten column names
    pivot.columns.name = None

    # Rename stat columns to be more descriptive
    rename_map = {
        'AREA PLANTED': 'area_planted_acres',
        'AREA HARVESTED': 'area_harvested_acres',
        'YIELD': 'yield_bu_per_acre',
        'PRODUCTION': 'production_bu',
    }
    pivot = pivot.rename(columns=rename_map)

    # Compute waste proxy: (planted - harvested) / planted
    if 'area_planted_acres' in pivot.columns and 'area_harvested_acres' in pivot.columns:
        mask = (pivot['area_planted_acres'] > 0) & pivot['area_harvested_acres'].notna()
        pivot.loc[mask, 'waste_proxy'] = (
            (pivot.loc[mask, 'area_planted_acres'] - pivot.loc[mask, 'area_harvested_acres'])
            / pivot.loc[mask, 'area_planted_acres']
        )

    # Sort
    pivot = pivot.sort_values(['FIPS', 'YEAR']).reset_index(drop=True)

    print(f"  {len(pivot):,} county-year rows after pivoting")
    print(f"  Columns: {list(pivot.columns)}")
    print(f"  Year range: {pivot['YEAR'].min()}-{pivot['YEAR'].max()}")
    print(f"  Unique counties: {pivot['FIPS'].nunique()}")

    return pivot


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # Process CORN
    print("\n" + "="*60)
    print("CORN")
    print("="*60)
    corn = load_and_clean(os.path.join(RAW_DIR, 'county_corn_clean.tsv'), 'CORN')
    corn_pivot = pivot_crop(corn, 'CORN')
    outpath = os.path.join(OUT_DIR, 'corn_county_annual.csv')
    corn_pivot.to_csv(outpath, index=False)
    print(f"  Saved: {outpath}")

    # Summary stats
    if 'waste_proxy' in corn_pivot.columns:
        wp = corn_pivot['waste_proxy'].dropna()
        print(f"  Waste proxy: mean={wp.mean():.4f}, median={wp.median():.4f}, "
              f"std={wp.std():.4f}, n={len(wp):,}")

    # Process SOYBEANS
    print("\n" + "="*60)
    print("SOYBEANS")
    print("="*60)
    soy = load_and_clean(os.path.join(RAW_DIR, 'county_soybeans_clean.tsv'), 'SOYBEANS')
    soy_pivot = pivot_crop(soy, 'SOYBEANS')
    outpath = os.path.join(OUT_DIR, 'soybeans_county_annual.csv')
    soy_pivot.to_csv(outpath, index=False)
    print(f"  Saved: {outpath}")

    if 'waste_proxy' in soy_pivot.columns:
        wp = soy_pivot['waste_proxy'].dropna()
        print(f"  Waste proxy: mean={wp.mean():.4f}, median={wp.median():.4f}, "
              f"std={wp.std():.4f}, n={len(wp):,}")

    # Process WHEAT (WINTER only -- consistent stats across all 4 categories)
    # Note: ALL CLASSES wheat has no AREA PLANTED data, so we use WINTER wheat
    # which covers ~85% of US wheat production and has all 4 stat categories.
    print("\n" + "="*60)
    print("WHEAT (WINTER)")
    print("="*60)
    wheat = load_and_clean(os.path.join(RAW_DIR, 'county_wheat_winter_clean.tsv'), 'WHEAT')
    wheat_pivot = pivot_crop(wheat, 'WHEAT')
    outpath = os.path.join(OUT_DIR, 'wheat_county_annual.csv')
    wheat_pivot.to_csv(outpath, index=False)
    print(f"  Saved: {outpath}")

    if 'waste_proxy' in wheat_pivot.columns:
        wp = wheat_pivot['waste_proxy'].dropna()
        print(f"  Waste proxy: mean={wp.mean():.4f}, median={wp.median():.4f}, "
              f"std={wp.std():.4f}, n={len(wp):,}")

    # Combined file
    print("\n" + "="*60)
    print("COMBINED")
    print("="*60)
    corn_pivot['crop'] = 'CORN'
    soy_pivot['crop'] = 'SOYBEANS'
    wheat_pivot['crop'] = 'WHEAT'
    combined = pd.concat([corn_pivot, soy_pivot, wheat_pivot], ignore_index=True)
    outpath = os.path.join(OUT_DIR, 'all_crops_county_annual.csv')
    combined.to_csv(outpath, index=False)
    print(f"  Saved: {outpath}")
    print(f"  Total rows: {len(combined):,}")
    print(f"  Unique FIPS: {combined['FIPS'].nunique()}")
    print(f"  Year range: {combined['YEAR'].min()}-{combined['YEAR'].max()}")

    # Print final summary
    print("\n" + "="*60)
    print("DOWNLOAD & PROCESSING SUMMARY")
    print("="*60)
    print(f"Source: USDA NASS Quick Stats bulk download (qs.crops)")
    print(f"URL: https://www.nass.usda.gov/datasets/qs.crops_20260328.txt.gz")
    print(f"")
    print("Files created in data/raw/nass/:")
    for f in ['corn_county_annual.csv', 'soybeans_county_annual.csv',
              'wheat_county_annual.csv', 'all_crops_county_annual.csv']:
        fpath = os.path.join(OUT_DIR, f)
        if os.path.exists(fpath):
            size = os.path.getsize(fpath)
            rows = sum(1 for _ in open(fpath)) - 1
            print(f"  {f}: {rows:,} rows, {size/1024/1024:.1f} MB")


if __name__ == '__main__':
    main()
