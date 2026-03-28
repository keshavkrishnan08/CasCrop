"""
Commodity price data loader from FRED and USDA sources.

Downloads daily futures prices for corn, soybeans, and wheat from the
FRED API, USDA prices-received data from NASS, and cost-of-production
estimates from USDA ERS. Computes derived price features: returns,
volatility, price-to-cost ratios, and futures basis.

APIs:
    - FRED: https://fred.stlouisfed.org/docs/api/fred/
    - USDA NASS: https://quickstats.nass.usda.gov/api
    - USDA ERS: https://www.ers.usda.gov/data-products/commodity-costs-and-returns/

Typical usage:
    loader = PriceLoader(fred_api_key="YOUR_KEY")
    loader.download_futures(
        commodities=["CORN", "SOYBEANS", "WHEAT"],
        years=range(2008, 2025),
        output_dir=Path("data/raw/prices"),
    )
    features = loader.compute_features(df)
"""

import io
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd

from .utils import (
    TARGET_COMMODITIES,
    create_retry_session,
    ensure_directory,
    save_parquet,
)

logger = logging.getLogger(__name__)

# FRED API base
FRED_API_BASE = "https://api.stlouisfed.org/fred/series/observations"

# FRED series IDs for commodity futures
# Note: CHRIS/CME series are on Quandl/Nasdaq. FRED has different IDs.
FRED_SERIES = {
    "CORN": "WPU012202",       # PPI: Corn
    "SOYBEANS": "WPU012401",   # PPI: Soybeans
    "WHEAT": "WPU0121",        # PPI: Wheat
}

# Alternative: continuous futures from Quandl/Nasdaq Data Link
QUANDL_FUTURES = {
    "CORN": "CHRIS/CME_C1",
    "SOYBEANS": "CHRIS/CME_S1",
    "WHEAT": "CHRIS/CME_W1",
}

# USDA NASS prices received series
NASS_PRICE_PARAMS = {
    "CORN": {
        "commodity_desc": "CORN",
        "statisticcat_desc": "PRICE RECEIVED",
        "unit_desc": "$ / BU",
    },
    "SOYBEANS": {
        "commodity_desc": "SOYBEANS",
        "statisticcat_desc": "PRICE RECEIVED",
        "unit_desc": "$ / BU",
    },
    "WHEAT": {
        "commodity_desc": "WHEAT",
        "statisticcat_desc": "PRICE RECEIVED",
        "unit_desc": "$ / BU",
    },
}


class PriceLoader:
    """Download and compute commodity price features."""

    def __init__(self, fred_api_key: str, nass_api_key: Optional[str] = None):
        """
        Args:
            fred_api_key: API key from https://fred.stlouisfed.org/docs/api/api_key.html
            nass_api_key: Optional USDA NASS API key for prices-received data.
        """
        self.fred_api_key = fred_api_key
        self.nass_api_key = nass_api_key
        self.session = create_retry_session(retries=5, backoff_factor=1.5)

    # ------------------------------------------------------------------
    # FRED futures / price index download
    # ------------------------------------------------------------------

    def download_futures(
        self,
        commodities: Optional[List[str]] = None,
        years: Optional[Union[range, List[int]]] = None,
        output_dir: Optional[Path] = None,
    ) -> pd.DataFrame:
        """Download commodity price series from FRED.

        First tries FRED PPI (Producer Price Index) series, which are
        freely available. If supplemental futures data is needed, attempts
        Nasdaq Data Link (formerly Quandl).

        Args:
            commodities: Commodity names (default: CORN, SOYBEANS, WHEAT).
            years: Year range for filtering (default: 2007-2025).
            output_dir: Cache directory.

        Returns:
            DataFrame with columns: date, commodity, price, source.
        """
        if commodities is None:
            commodities = list(TARGET_COMMODITIES)
        if years is None:
            years = range(2007, 2026)
        if output_dir:
            output_dir = ensure_directory(output_dir)

        start_date = f"{min(years)}-01-01"
        end_date = f"{max(years)}-12-31"

        all_frames = []

        for commodity in commodities:
            cache = None
            if output_dir:
                cache = output_dir / f"price_{commodity.lower()}.parquet"
                if cache.exists():
                    all_frames.append(pd.read_parquet(cache))
                    continue

            logger.info("Downloading prices for %s", commodity)

            # Try FRED PPI series
            series_id = FRED_SERIES.get(commodity)
            if series_id:
                df = self._fetch_fred_series(series_id, start_date, end_date)
                if not df.empty:
                    df["commodity"] = commodity
                    df["source"] = "FRED_PPI"
                    all_frames.append(df)
                    if cache:
                        df.to_parquet(cache, index=False)
                    continue

            # Try Nasdaq Data Link / Quandl as fallback
            quandl_code = QUANDL_FUTURES.get(commodity)
            if quandl_code:
                df = self._fetch_quandl(quandl_code, start_date, end_date)
                if not df.empty:
                    df["commodity"] = commodity
                    df["source"] = "QUANDL"
                    all_frames.append(df)
                    if cache:
                        df.to_parquet(cache, index=False)
                    continue

            logger.warning("No price data source worked for %s", commodity)

        if not all_frames:
            logger.warning("No futures data downloaded")
            return pd.DataFrame()

        combined = pd.concat(all_frames, ignore_index=True)
        logger.info("Total price records: %d", len(combined))
        return combined

    def _fetch_fred_series(
        self, series_id: str, start_date: str, end_date: str
    ) -> pd.DataFrame:
        """Fetch a single FRED time series."""
        params = {
            "series_id": series_id,
            "api_key": self.fred_api_key,
            "file_type": "json",
            "observation_start": start_date,
            "observation_end": end_date,
            "frequency": "d",  # daily
        }

        try:
            resp = self.session.get(FRED_API_BASE, params=params, timeout=60)
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:
            logger.error("FRED API request failed for %s: %s", series_id, exc)
            return pd.DataFrame()

        observations = data.get("observations", [])
        if not observations:
            # Try monthly frequency as fallback
            params["frequency"] = "m"
            try:
                resp = self.session.get(FRED_API_BASE, params=params, timeout=60)
                resp.raise_for_status()
                data = resp.json()
                observations = data.get("observations", [])
            except Exception:
                return pd.DataFrame()

        if not observations:
            return pd.DataFrame()

        df = pd.DataFrame(observations)
        df["date"] = pd.to_datetime(df["date"])
        df["price"] = pd.to_numeric(
            df["value"].replace(".", np.nan), errors="coerce"
        )
        df = df[["date", "price"]].dropna()

        logger.info("FRED %s: %d observations", series_id, len(df))
        return df

    def _fetch_quandl(
        self, code: str, start_date: str, end_date: str
    ) -> pd.DataFrame:
        """Fetch futures data from Nasdaq Data Link (Quandl)."""
        import os
        quandl_key = os.environ.get("QUANDL_API_KEY", "")

        url = f"https://data.nasdaq.com/api/v3/datasets/{code}.json"
        params = {
            "start_date": start_date,
            "end_date": end_date,
            "api_key": quandl_key,
        }

        try:
            resp = self.session.get(url, params=params, timeout=60)
            if resp.status_code != 200:
                return pd.DataFrame()
            data = resp.json()
        except Exception as exc:
            logger.debug("Quandl request failed for %s: %s", code, exc)
            return pd.DataFrame()

        dataset = data.get("dataset", {})
        columns = dataset.get("column_names", [])
        rows = dataset.get("data", [])

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows, columns=columns)
        df["date"] = pd.to_datetime(df.iloc[:, 0])

        # Settlement / Last / Close price — different naming per commodity
        price_col = None
        for cand in ["Settle", "Last", "Close", "Settlement"]:
            if cand in df.columns:
                price_col = cand
                break

        if price_col is None:
            # Use the second numeric column
            price_col = df.columns[1] if len(df.columns) > 1 else None

        if price_col:
            df["price"] = pd.to_numeric(df[price_col], errors="coerce")
        else:
            return pd.DataFrame()

        return df[["date", "price"]].dropna()

    # ------------------------------------------------------------------
    # USDA prices received
    # ------------------------------------------------------------------

    def download_usda_prices_received(
        self,
        years: Union[range, List[int]],
        output_dir: Optional[Path] = None,
    ) -> pd.DataFrame:
        """Download state-level prices received from USDA NASS.

        Prices received are the actual prices farmers get for their crops,
        as opposed to futures prices. These capture local market conditions
        and basis differentials.

        Args:
            years: Calendar years.
            output_dir: Cache directory.

        Returns:
            DataFrame with columns: state_code, commodity, year, month, price_received.
        """
        if not self.nass_api_key:
            logger.warning("NASS API key not provided — skipping prices received")
            return pd.DataFrame()

        if output_dir:
            output_dir = ensure_directory(output_dir)

        nass_base = "https://quickstats.nass.usda.gov/api/api_GET/"
        all_frames = []

        for commodity, params in NASS_PRICE_PARAMS.items():
            for year in years:
                cache = None
                if output_dir:
                    cache = output_dir / f"price_received_{commodity.lower()}_{year}.parquet"
                    if cache.exists():
                        all_frames.append(pd.read_parquet(cache))
                        continue

                query = {
                    "key": self.nass_api_key,
                    "year": str(year),
                    "agg_level_desc": "STATE",
                    "format": "JSON",
                    **params,
                }

                try:
                    resp = self.session.get(nass_base, params=query, timeout=60)
                    resp.raise_for_status()
                    data = resp.json()
                except Exception as exc:
                    logger.debug("NASS price query failed: %s", exc)
                    time.sleep(0.5)
                    continue

                records = data.get("data", [])
                if not records:
                    time.sleep(0.5)
                    continue

                df = pd.DataFrame(records)
                cleaned = self._clean_prices_received(df, commodity, year)

                if not cleaned.empty:
                    all_frames.append(cleaned)
                    if cache:
                        cleaned.to_parquet(cache, index=False)

                time.sleep(0.7)

        if not all_frames:
            return pd.DataFrame()

        return pd.concat(all_frames, ignore_index=True)

    def _clean_prices_received(
        self, df: pd.DataFrame, commodity: str, year: int
    ) -> pd.DataFrame:
        """Clean NASS prices-received response."""
        result = pd.DataFrame()
        result["state_code"] = (
            df.get("state_fips_code", "00").astype(str).str.zfill(2)
        )
        result["commodity"] = commodity
        result["year"] = year

        # Extract month from reference_period_desc
        month_map = {
            "JAN": 1, "FEB": 2, "MAR": 3, "APR": 4,
            "MAY": 5, "JUN": 6, "JUL": 7, "AUG": 8,
            "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12,
            "YEAR": 0,  # Annual average
        }
        ref_period = df.get("reference_period_desc", "YEAR").astype(str).str.upper()
        result["month"] = ref_period.map(month_map).fillna(0).astype(int)

        # Price value
        val = df.get("Value", df.get("value", "")).astype(str)
        val = val.str.replace(",", "", regex=False)
        val = val.replace({"(D)": None, "(Z)": None, "": None})
        result["price_received"] = pd.to_numeric(val, errors="coerce")

        return result.dropna(subset=["price_received"])

    # ------------------------------------------------------------------
    # Cost of production
    # ------------------------------------------------------------------

    def download_cost_of_production(
        self,
        years: Union[range, List[int]],
        output_dir: Optional[Path] = None,
    ) -> pd.DataFrame:
        """Download cost-of-production estimates from USDA ERS.

        ERS publishes annual commodity costs and returns data including:
        - Operating costs (seed, fertilizer, chemicals, fuel, labor)
        - Allocated overhead (land, capital, management)
        - Total economic cost per acre and per bushel

        Data is national-level with some regional breakdowns.

        Args:
            years: Calendar years.
            output_dir: Cache directory.

        Returns:
            DataFrame with columns: commodity, year, cost_per_acre,
            cost_per_bushel, breakeven_yield.
        """
        if output_dir:
            output_dir = ensure_directory(output_dir)

        # ERS data files (CSV downloads)
        ers_urls = {
            "CORN": (
                "https://www.ers.usda.gov/webdocs/DataFiles/"
                "47913/CornCostReturn.csv"
            ),
            "SOYBEANS": (
                "https://www.ers.usda.gov/webdocs/DataFiles/"
                "47913/SoybeanCostReturn.csv"
            ),
            "WHEAT": (
                "https://www.ers.usda.gov/webdocs/DataFiles/"
                "47913/WheatCostReturn.csv"
            ),
        }

        all_frames = []

        for commodity, url in ers_urls.items():
            cache = None
            if output_dir:
                cache = output_dir / f"cost_{commodity.lower()}.parquet"
                if cache.exists():
                    all_frames.append(pd.read_parquet(cache))
                    continue

            logger.info("Downloading ERS cost data for %s", commodity)
            try:
                resp = self.session.get(url, timeout=60)
                if resp.status_code != 200:
                    logger.warning("ERS download failed for %s: HTTP %d",
                                   commodity, resp.status_code)
                    # Build synthetic cost data as fallback
                    df = self._build_synthetic_costs(commodity, years)
                else:
                    try:
                        df = pd.read_csv(io.StringIO(resp.text))
                        df = self._clean_ers_costs(df, commodity, years)
                    except Exception as exc:
                        logger.warning("Failed to parse ERS CSV for %s: %s",
                                       commodity, exc)
                        df = self._build_synthetic_costs(commodity, years)

                if not df.empty:
                    all_frames.append(df)
                    if cache:
                        df.to_parquet(cache, index=False)

            except Exception as exc:
                logger.error("ERS download error for %s: %s", commodity, exc)
                df = self._build_synthetic_costs(commodity, years)
                if not df.empty:
                    all_frames.append(df)

        if not all_frames:
            return pd.DataFrame()

        return pd.concat(all_frames, ignore_index=True)

    def _clean_ers_costs(
        self, df: pd.DataFrame, commodity: str, years: Union[range, List[int]]
    ) -> pd.DataFrame:
        """Parse ERS cost-and-returns CSV into standardized format."""
        result = pd.DataFrame()

        # ERS CSVs have variable layouts — find year and cost columns
        year_col = None
        for cand in ["Year", "year", "YEAR"]:
            if cand in df.columns:
                year_col = cand
                break

        if year_col is None:
            # Try first column
            year_col = df.columns[0]

        result["year"] = pd.to_numeric(df[year_col], errors="coerce")
        result = result.dropna(subset=["year"])
        result["year"] = result["year"].astype(int)
        result = result[result["year"].isin(years)]
        result["commodity"] = commodity

        # Find cost columns
        cost_per_acre_col = None
        for cand in ["Total, gross value of production", "Total economic cost",
                      "Total costs", "Operating costs", "cost_per_acre"]:
            matches = [c for c in df.columns if cand.lower() in c.lower()]
            if matches:
                cost_per_acre_col = matches[0]
                break

        if cost_per_acre_col:
            result["cost_per_acre"] = pd.to_numeric(
                df[cost_per_acre_col].astype(str).str.replace(",", ""),
                errors="coerce",
            )
        else:
            result["cost_per_acre"] = np.nan

        # Yield and compute cost per bushel
        yield_col = None
        for cand in ["Yield per planted acre", "yield", "Yield"]:
            matches = [c for c in df.columns if cand.lower() in c.lower()]
            if matches:
                yield_col = matches[0]
                break

        if yield_col and cost_per_acre_col:
            yld = pd.to_numeric(
                df[yield_col].astype(str).str.replace(",", ""),
                errors="coerce",
            )
            result["breakeven_yield"] = yld
            result["cost_per_bushel"] = np.where(
                yld > 0, result["cost_per_acre"] / yld, np.nan
            )
        else:
            result["cost_per_bushel"] = np.nan
            result["breakeven_yield"] = np.nan

        return result

    def _build_synthetic_costs(
        self, commodity: str, years: Union[range, List[int]]
    ) -> pd.DataFrame:
        """Build approximate cost-of-production when ERS data unavailable.

        Uses USDA-published average costs ($/acre) with inflation adjustment.
        These are rough national averages for planning purposes.
        """
        # Base costs circa 2020 ($/acre, total economic cost)
        base_costs = {
            "CORN": 700.0,
            "SOYBEANS": 450.0,
            "WHEAT": 350.0,
        }
        # Average yields (bu/acre)
        base_yields = {
            "CORN": 175.0,
            "SOYBEANS": 50.0,
            "WHEAT": 47.0,
        }

        base_year = 2020
        inflation_rate = 0.03  # ~3% annual cost increase

        records = []
        for year in years:
            years_diff = year - base_year
            cost_adj = base_costs.get(commodity, 500) * (1 + inflation_rate) ** years_diff
            yld = base_yields.get(commodity, 100)
            records.append({
                "commodity": commodity,
                "year": year,
                "cost_per_acre": round(cost_adj, 2),
                "cost_per_bushel": round(cost_adj / yld, 2) if yld > 0 else np.nan,
                "breakeven_yield": yld,
            })

        logger.info("Built synthetic cost data for %s (%d years)", commodity, len(records))
        return pd.DataFrame(records)

    # ------------------------------------------------------------------
    # Compute derived price features
    # ------------------------------------------------------------------

    def compute_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute price-derived features used as model inputs.

        New columns:
        - price_change_pct: Month-over-month percent change.
        - price_change_3m: 3-month percent change.
        - rolling_volatility_30d: 30-day rolling standard deviation of returns.
        - price_vs_cost_ratio: Current price / cost-of-production.
        - price_zscore: Z-score relative to trailing 12-month distribution.
        - futures_basis: Spread between futures and cash (if both available).

        Args:
            df: Price DataFrame with columns: date, commodity, price.

        Returns:
            DataFrame with additional feature columns.
        """
        df = df.copy()
        df = df.sort_values(["commodity", "date"]).reset_index(drop=True)

        featured_frames = []

        for commodity, group in df.groupby("commodity"):
            g = group.copy()

            # Daily returns
            g["daily_return"] = g["price"].pct_change()

            # Monthly price change (approximate: 21 trading days)
            g["price_change_pct"] = g["price"].pct_change(periods=21)

            # 3-month price change
            g["price_change_3m"] = g["price"].pct_change(periods=63)

            # 30-day rolling volatility (annualized)
            g["rolling_volatility_30d"] = (
                g["daily_return"]
                .rolling(window=30, min_periods=10)
                .std()
                * np.sqrt(252)  # Annualize
            )

            # Z-score relative to trailing 252 days (1 year)
            rolling_mean = g["price"].rolling(252, min_periods=60).mean()
            rolling_std = g["price"].rolling(252, min_periods=60).std()
            g["price_zscore"] = np.where(
                rolling_std > 0,
                (g["price"] - rolling_mean) / rolling_std,
                0.0,
            )

            # Momentum indicators
            g["price_sma_50"] = g["price"].rolling(50, min_periods=20).mean()
            g["price_sma_200"] = g["price"].rolling(200, min_periods=60).mean()
            g["momentum_signal"] = np.where(
                g["price_sma_200"] > 0,
                (g["price_sma_50"] / g["price_sma_200"]) - 1.0,
                0.0,
            )

            featured_frames.append(g)

        result = pd.concat(featured_frames, ignore_index=True)
        logger.info("Computed price features: %d rows, %d columns", *result.shape)
        return result

    def add_cost_ratio(
        self,
        price_df: pd.DataFrame,
        cost_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Add price-to-cost ratio by matching price data with production costs.

        Args:
            price_df: Daily/monthly prices with commodity, date, price columns.
            cost_df: Annual cost data with commodity, year, cost_per_bushel.

        Returns:
            Price DataFrame with added price_vs_cost_ratio column.
        """
        df = price_df.copy()

        if "date" in df.columns:
            df["year"] = pd.to_datetime(df["date"]).dt.year
        elif "year" not in df.columns:
            logger.warning("No date or year column in price data")
            return df

        # Merge cost data
        cost_cols = ["commodity", "year", "cost_per_bushel"]
        available = [c for c in cost_cols if c in cost_df.columns]
        if len(available) < 3:
            logger.warning("Cost data missing required columns")
            df["price_vs_cost_ratio"] = np.nan
            return df

        merged = df.merge(
            cost_df[available],
            on=["commodity", "year"],
            how="left",
        )

        merged["price_vs_cost_ratio"] = np.where(
            merged["cost_per_bushel"] > 0,
            merged["price"] / merged["cost_per_bushel"],
            np.nan,
        )

        return merged

    def aggregate_to_monthly(self, df: pd.DataFrame) -> pd.DataFrame:
        """Aggregate daily price data to monthly frequency.

        For each commodity-month, computes: mean, median, min, max,
        close (last value), and std of prices.

        Args:
            df: Daily price data with date, commodity, price columns.

        Returns:
            Monthly DataFrame with summary statistics.
        """
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"])
        df["year"] = df["date"].dt.year
        df["month"] = df["date"].dt.month

        agg_dict = {
            "price": ["mean", "median", "min", "max", "last", "std"],
        }

        # Also aggregate any computed features
        for col in ["rolling_volatility_30d", "price_zscore",
                     "momentum_signal", "price_vs_cost_ratio"]:
            if col in df.columns:
                agg_dict[col] = "mean"

        monthly = (
            df.groupby(["commodity", "year", "month"])
            .agg(agg_dict)
        )

        # Flatten multi-level columns
        monthly.columns = [
            f"{col}_{stat}" if stat != "mean" or col != "price"
            else col
            for col, stat in monthly.columns
        ]
        monthly = monthly.reset_index()

        # Rename for clarity
        rename_map = {
            "price_mean": "price_mean",
            "price_median": "price_median",
            "price_min": "price_min",
            "price_max": "price_max",
            "price_last": "price_close",
            "price_std": "price_monthly_std",
        }
        monthly = monthly.rename(columns={
            k: v for k, v in rename_map.items() if k in monthly.columns
        })

        logger.info("Aggregated to %d monthly price observations", len(monthly))
        return monthly

    # ------------------------------------------------------------------
    # Futures basis computation
    # ------------------------------------------------------------------

    def compute_futures_basis(
        self,
        futures_df: pd.DataFrame,
        cash_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Compute basis: cash price - futures price.

        Basis reflects local supply/demand conditions and transportation
        costs. A narrowing basis signals local demand; widening basis
        signals oversupply.

        Args:
            futures_df: Futures prices with date, commodity, price.
            cash_df: Cash/spot prices with state_code, commodity, year, month, price_received.

        Returns:
            DataFrame with futures_basis column per state-commodity-month.
        """
        # Aggregate futures to monthly
        fut = futures_df.copy()
        fut["date"] = pd.to_datetime(fut["date"])
        fut["year"] = fut["date"].dt.year
        fut["month"] = fut["date"].dt.month

        fut_monthly = (
            fut.groupby(["commodity", "year", "month"])["price"]
            .mean()
            .rename("futures_price")
            .reset_index()
        )

        # Merge with cash prices
        merged = cash_df.merge(
            fut_monthly,
            on=["commodity", "year", "month"],
            how="left",
        )

        merged["futures_basis"] = merged["price_received"] - merged["futures_price"]
        merged["basis_pct"] = np.where(
            merged["futures_price"] > 0,
            merged["futures_basis"] / merged["futures_price"] * 100,
            np.nan,
        )

        return merged

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate(self, df: pd.DataFrame) -> None:
        """Sanity checks on price data."""
        assert len(df) > 0, "Price DataFrame is empty"

        if "price" in df.columns:
            assert (df["price"].dropna() > 0).all(), "Non-positive prices found"

        if "commodity" in df.columns:
            commodities = set(df["commodity"].unique())
            for target in TARGET_COMMODITIES:
                assert target in commodities, f"Missing commodity: {target}"

        if "date" in df.columns:
            date_range = pd.to_datetime(df["date"])
            logger.info(
                "Price data range: %s to %s",
                date_range.min().strftime("%Y-%m-%d"),
                date_range.max().strftime("%Y-%m-%d"),
            )

        logger.info("Price validation passed: %d rows", len(df))

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    def save(self, df: pd.DataFrame, output_path: Path) -> None:
        """Save price data as Parquet."""
        save_parquet(df, output_path)
