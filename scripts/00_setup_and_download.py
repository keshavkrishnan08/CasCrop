#!/usr/bin/env python3
"""Script 00: Setup environment and download all datasets.

This is the first script to run. It:
1. Checks for required API keys (from .env file)
2. Downloads all freely available datasets (no key required)
3. Downloads API-keyed datasets where keys are available
4. Reports what succeeded and what needs manual intervention

Datasets that DON'T need API keys:
- USDA RMA crop insurance claims (direct download)
- Census Bureau county adjacency + FIPS gazetteer
- FRED commodity prices (CSV endpoint)
- US Drought Monitor (direct download)

Datasets that NEED API keys:
- USDA NASS Quick Stats (NASS_API_KEY)
- NOAA weather (NOAA_API_TOKEN)
- Google Earth Engine imagery (GEE_PROJECT + OAuth)
- NASA SMAP soil moisture (NASA_EARTHDATA_USER)

Usage:
    python scripts/00_setup_and_download.py
    python scripts/00_setup_and_download.py --skip-api  # only free data
    python scripts/00_setup_and_download.py --only rma prices  # specific sources
"""

import argparse
import hashlib
import json
import logging
import os
import sys
import time
import zipfile
from datetime import datetime
from io import BytesIO
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_env(env_path: str = ".env") -> dict:
    """Load environment variables from .env file."""
    env = {}
    path = Path(env_path)
    if path.exists():
        for line in path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, value = line.partition("=")
                env[key.strip()] = value.strip()
                os.environ.setdefault(key.strip(), value.strip())
    return env


def download_file(url: str, output_path: Path, description: str = "",
                  max_retries: int = 3, timeout: int = 120) -> bool:
    """Download a file with retries and validation."""
    import requests

    output_path.parent.mkdir(parents=True, exist_ok=True)

    for attempt in range(max_retries):
        try:
            logger.info(f"  Downloading: {description or url}")
            headers = {
                "User-Agent": "CasCrop-Research/1.0 (academic research; contact: researcher@university.edu)"
            }
            resp = requests.get(url, headers=headers, timeout=timeout, stream=True)
            resp.raise_for_status()

            total = int(resp.headers.get("content-length", 0))
            with open(output_path, "wb") as f:
                downloaded = 0
                for chunk in resp.iter_content(chunk_size=8192):
                    f.write(chunk)
                    downloaded += len(chunk)

            size = output_path.stat().st_size
            if size == 0:
                logger.warning(f"  Empty file downloaded: {output_path}")
                output_path.unlink()
                continue

            logger.info(f"  Saved: {output_path} ({size:,} bytes)")
            return True

        except Exception as e:
            logger.warning(f"  Attempt {attempt + 1}/{max_retries} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)

    logger.error(f"  FAILED: {description or url}")
    return False


def unzip_file(zip_path: Path, extract_to: Path) -> list:
    """Extract a zip file and return list of extracted files."""
    extracted = []
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(extract_to)
            extracted = zf.namelist()
            logger.info(f"  Extracted {len(extracted)} files from {zip_path.name}")
    except zipfile.BadZipFile:
        logger.warning(f"  Not a valid zip file: {zip_path}")
    return extracted


# ---------------------------------------------------------------------------
# Data Source Downloaders
# ---------------------------------------------------------------------------

def download_rma_data(raw_dir: Path) -> dict:
    """Download USDA RMA Cause of Loss data (no API key needed).

    These are the training labels — the most critical dataset.
    """
    rma_dir = raw_dir / "rma"
    rma_dir.mkdir(parents=True, exist_ok=True)
    result = {"name": "USDA RMA Cause of Loss", "status": "partial", "files": []}

    # RMA provides annual cause-of-loss files
    # Try multiple URL patterns as they change occasionally
    url_patterns = [
        "https://www.rma.usda.gov/-/media/RMA/Cause-of-Loss/ColData{year}.ashx",
        "https://www.rma.usda.gov/-/media/RMA/Cause-of-Loss/coldat{year}.zip",
        "https://www.rma.usda.gov/data/cause-of-loss/coldat{year}.zip",
        "https://prodwebnlb.rma.usda.gov/apps/sob/current_week/causeloss{year}.zip",
    ]

    years_downloaded = []
    for year in range(2015, 2025):
        downloaded = False
        for pattern in url_patterns:
            url = pattern.format(year=year)
            ext = ".zip" if "zip" in url.lower() else ".txt"
            outfile = rma_dir / f"coldata_{year}{ext}"

            if outfile.exists() and outfile.stat().st_size > 1000:
                logger.info(f"  Already exists: {outfile.name}")
                years_downloaded.append(year)
                downloaded = True
                break

            if download_file(url, outfile, f"RMA {year}"):
                # If it's a zip, extract it
                if ext == ".zip":
                    unzip_file(outfile, rma_dir)
                years_downloaded.append(year)
                result["files"].append(str(outfile))
                downloaded = True
                break

        if not downloaded:
            logger.warning(f"  Could not download RMA data for {year}")

    result["years_downloaded"] = years_downloaded
    result["status"] = "complete" if len(years_downloaded) >= 5 else "partial"
    return result


def download_geographic_data(raw_dir: Path) -> dict:
    """Download county adjacency, FIPS codes, centroids (no API key)."""
    geo_dir = raw_dir / "geographic"
    geo_dir.mkdir(parents=True, exist_ok=True)
    result = {"name": "Geographic & FIPS Data", "status": "pending", "files": []}

    # County adjacency
    adj_urls = [
        "https://www2.census.gov/geo/docs/reference/county_adjacency.txt",
        "https://www2.census.gov/programs-surveys/geography/relateddocs/county-adjacency.txt",
    ]
    adj_file = geo_dir / "county_adjacency.txt"
    for url in adj_urls:
        if download_file(url, adj_file, "County adjacency"):
            result["files"].append(str(adj_file))
            break

    # County gazetteer (FIPS + centroids)
    gaz_urls = [
        "https://www2.census.gov/geo/docs/maps-data/data/gazetteer/2023_Gazetteer/2023_Gaz_counties_national.zip",
        "https://www2.census.gov/geo/docs/maps-data/data/gazetteer/2022_Gazetteer/2022_Gaz_counties_national.zip",
        "https://www2.census.gov/geo/docs/maps-data/data/gazetteer/2021_Gazetteer/2021_Gaz_counties_national.zip",
    ]
    gaz_file = geo_dir / "county_gazetteer.zip"
    for url in gaz_urls:
        if download_file(url, gaz_file, "County gazetteer"):
            unzip_file(gaz_file, geo_dir)
            result["files"].append(str(gaz_file))
            break

    # County shapefile (for maps)
    shp_url = "https://www2.census.gov/geo/tiger/GENZ2021/shp/cb_2021_us_county_500k.zip"
    shp_file = geo_dir / "county_shapefile.zip"
    if download_file(shp_url, shp_file, "County shapefile"):
        unzip_file(shp_file, geo_dir)
        result["files"].append(str(shp_file))

    result["status"] = "complete" if len(result["files"]) >= 2 else "partial"
    return result


def download_price_data(raw_dir: Path) -> dict:
    """Download commodity prices from FRED (CSV endpoint, no API key)."""
    price_dir = raw_dir / "prices"
    price_dir.mkdir(parents=True, exist_ok=True)
    result = {"name": "Commodity Prices (FRED)", "status": "pending", "files": []}

    # FRED CSV endpoint works without API key for individual series
    series = {
        "corn": [
            "https://fred.stlouisfed.org/graph/fredgraph.csv?bgcolor=%23e1e9f0&chart_type=line&drp=0&fo=open%20sans&graph_bgcolor=%23ffffff&height=450&mode=fred&recession_bars=on&txtcolor=%23444444&ts=12&tts=12&width=1318&nt=0&thu=0&trc=0&show_legend=yes&show_axis_titles=yes&show_tooltip=yes&id=PMAIZMTUSDM&scale=left&cosd=2008-01-01&coed=2025-01-01&line_color=%234572a7&link_values=false&line_style=solid&mark_type=none&mw=3&lw=2&ost=-99999&oet=99999&mma=0&fml=a&fq=Monthly&fam=avg&fgst=lin&fgsnd=2020-02-01&line_index=1&transformation=lin&vintage_date={today}&revision_date={today}&nd=1990-01-01",
            "https://fred.stlouisfed.org/graph/fredgraph.csv?id=PMAIZMTUSDM",
            "https://fred.stlouisfed.org/graph/fredgraph.csv?id=WPU012202",
        ],
        "wheat": [
            "https://fred.stlouisfed.org/graph/fredgraph.csv?id=PWHEAMTUSDM",
            "https://fred.stlouisfed.org/graph/fredgraph.csv?id=WPU0121",
        ],
        "soybeans": [
            "https://fred.stlouisfed.org/graph/fredgraph.csv?id=PSOYBUSDM",
            "https://fred.stlouisfed.org/graph/fredgraph.csv?id=WPU01830101",
        ],
    }

    today = datetime.now().strftime("%Y-%m-%d")

    for commodity, urls in series.items():
        outfile = price_dir / f"{commodity}_prices.csv"
        if outfile.exists() and outfile.stat().st_size > 100:
            logger.info(f"  Already exists: {outfile.name}")
            result["files"].append(str(outfile))
            continue

        for url in urls:
            url = url.format(today=today)
            if download_file(url, outfile, f"{commodity} prices"):
                result["files"].append(str(outfile))
                break

    result["status"] = "complete" if len(result["files"]) >= 2 else "partial"
    return result


def download_drought_data(raw_dir: Path) -> dict:
    """Download US Drought Monitor data (no API key)."""
    drought_dir = raw_dir / "weather"
    drought_dir.mkdir(parents=True, exist_ok=True)
    result = {"name": "US Drought Monitor", "status": "pending", "files": []}

    # USDM comprehensive statistics (county-level)
    urls = [
        "https://droughtmonitor.unl.edu/DmData/DataDownload/DSCI.aspx",
        "https://usdm.unl.edu/DmData/DataTables.aspx?mode=table&aession=County&flag=1",
    ]

    # Try direct CSV downloads for recent years
    for year in range(2015, 2025):
        url = f"https://droughtmonitor.unl.edu/DmData/DataDownload/ComprehensiveStatistics.aspx?fips=&daterange=01/01/{year}-12/31/{year}&stattype=county"
        outfile = drought_dir / f"drought_{year}.csv"
        if outfile.exists() and outfile.stat().st_size > 100:
            result["files"].append(str(outfile))
            continue
        # This URL may not work directly — log it
        download_file(url, outfile, f"Drought data {year}", max_retries=1)
        if outfile.exists() and outfile.stat().st_size > 100:
            result["files"].append(str(outfile))

    result["status"] = "complete" if len(result["files"]) >= 3 else "partial"
    return result


def download_nass_data(raw_dir: Path, api_key: str) -> dict:
    """Download USDA NASS data (requires API key)."""
    nass_dir = raw_dir / "nass"
    nass_dir.mkdir(parents=True, exist_ok=True)
    result = {"name": "USDA NASS Quick Stats", "status": "pending", "files": []}

    if not api_key:
        result["status"] = "skipped"
        result["reason"] = "No NASS_API_KEY in .env — get one at https://quickstats.nass.usda.gov/api"
        return result

    import requests

    base_url = "https://quickstats.nass.usda.gov/api/api_GET/"

    for commodity in ["CORN", "SOYBEANS", "WHEAT"]:
        for stat in ["YIELD", "PRODUCTION", "AREA PLANTED", "AREA HARVESTED"]:
            params = {
                "key": api_key,
                "commodity_desc": commodity,
                "statisticcat_desc": stat,
                "agg_level_desc": "COUNTY",
                "year__GE": 2008,
                "year__LE": 2024,
                "format": "JSON",
            }
            outfile = nass_dir / f"{commodity.lower()}_{stat.lower().replace(' ', '_')}.json"

            if outfile.exists() and outfile.stat().st_size > 100:
                result["files"].append(str(outfile))
                continue

            try:
                logger.info(f"  Querying NASS: {commodity} {stat}")
                resp = requests.get(base_url, params=params, timeout=60)
                resp.raise_for_status()
                data = resp.json()

                with open(outfile, "w") as f:
                    json.dump(data, f)

                record_count = len(data.get("data", []))
                logger.info(f"  Got {record_count:,} records")
                result["files"].append(str(outfile))

                time.sleep(1)  # Rate limit

            except Exception as e:
                logger.warning(f"  NASS query failed: {e}")

    result["status"] = "complete" if len(result["files"]) >= 6 else "partial"
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Download CasCrop datasets")
    parser.add_argument("--skip-api", action="store_true",
                        help="Skip datasets requiring API keys")
    parser.add_argument("--only", nargs="*", default=None,
                        help="Only download specific sources (rma, geo, prices, drought, nass)")
    parser.add_argument("--env", default=".env", help="Path to .env file")
    args = parser.parse_args()

    logger.info("CasCrop Data Download")
    logger.info("=" * 60)

    # Load API keys
    env = load_env(args.env)
    logger.info(f"API keys found: {sum(1 for v in env.values() if v)}/{len(env)}")

    raw_dir = Path("data/raw")
    raw_dir.mkdir(parents=True, exist_ok=True)

    sources = args.only or ["rma", "geo", "prices", "drought", "nass"]
    results = []

    # Free downloads (no API key)
    if "rma" in sources:
        logger.info("\n--- USDA RMA Cause of Loss Data (training labels) ---")
        results.append(download_rma_data(raw_dir))

    if "geo" in sources:
        logger.info("\n--- Geographic & FIPS Data ---")
        results.append(download_geographic_data(raw_dir))

    if "prices" in sources:
        logger.info("\n--- Commodity Prices ---")
        results.append(download_price_data(raw_dir))

    if "drought" in sources:
        logger.info("\n--- US Drought Monitor ---")
        results.append(download_drought_data(raw_dir))

    # API-keyed downloads
    if not args.skip_api:
        if "nass" in sources:
            logger.info("\n--- USDA NASS Quick Stats ---")
            results.append(download_nass_data(raw_dir, env.get("NASS_API_KEY", "")))

    # Save manifest
    manifest = {
        "download_timestamp": datetime.now().isoformat(),
        "datasets": results,
    }
    with open(raw_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2, default=str)

    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("DOWNLOAD SUMMARY")
    logger.info("=" * 60)

    for r in results:
        status_icon = {"complete": "+", "partial": "~", "skipped": "-", "pending": "?"}.get(
            r["status"], "?"
        )
        n_files = len(r.get("files", []))
        logger.info(f"  [{status_icon}] {r['name']}: {r['status']} ({n_files} files)")
        if r.get("reason"):
            logger.info(f"      {r['reason']}")

    # Report what's needed next
    missing = [r for r in results if r["status"] in ("skipped", "pending")]
    if missing:
        logger.info("\nTO COMPLETE DATA DOWNLOAD:")
        logger.info("  1. Copy .env.example to .env")
        logger.info("  2. Fill in missing API keys (all free)")
        logger.info("  3. Re-run: python scripts/00_setup_and_download.py")
        for r in missing:
            if r.get("reason"):
                logger.info(f"  - {r['reason']}")

    logger.info(f"\nManifest saved to {raw_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
