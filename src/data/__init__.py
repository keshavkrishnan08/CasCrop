"""CasCrop data acquisition, processing, matching, graph construction, and dataset modules.

Data acquisition loaders:
- ``rma_loader``: USDA RMA crop insurance cause-of-loss claims (training labels)
- ``crop_loader``: USDA NASS Quick Stats crop production data
- ``weather_loader``: NOAA CDO weather observations + drought index
- ``sentinel_loader``: Sentinel-2 vegetation indices via Google Earth Engine
- ``vegscape_loader``: USDA VegScape vegetation condition (MODIS-derived)
- ``soil_loader``: NASA SMAP soil moisture + NLDAS-2 backfill
- ``price_loader``: Commodity futures (FRED) + USDA prices received + ERS costs
- ``utils``: FIPS codes, geographic helpers, shared constants
"""

# --- Data acquisition loaders ---
from cascrop.src.data.utils import (
    CAUSE_OF_LOSS_MAPPING,
    COMMODITY_CODES,
    TARGET_COMMODITIES,
    STATE_FIPS_TO_NAME,
    create_retry_session,
    format_fips,
    load_fips_codes,
    fips_to_latlon,
    validate_fips,
    map_cause_code,
)
from cascrop.src.data.rma_loader import RMALoader
from cascrop.src.data.crop_loader import NASSLoader
from cascrop.src.data.weather_loader import NOAAWeatherLoader
from cascrop.src.data.sentinel_loader import SentinelLoader
from cascrop.src.data.vegscape_loader import VegScapeLoader
from cascrop.src.data.soil_loader import SMAPLoader
from cascrop.src.data.price_loader import PriceLoader

# --- Data processing / matching / dataset ---
from cascrop.src.data.dataset import CasCropDataset, CasCropGraphBatch, get_dataloaders
from cascrop.src.data.graph_builder import GraphBuilder
from cascrop.src.data.matcher import DataMatcher

__all__ = [
    # Loaders
    "RMALoader",
    "NASSLoader",
    "NOAAWeatherLoader",
    "SentinelLoader",
    "VegScapeLoader",
    "SMAPLoader",
    "PriceLoader",
    # Utilities
    "CAUSE_OF_LOSS_MAPPING",
    "COMMODITY_CODES",
    "TARGET_COMMODITIES",
    "STATE_FIPS_TO_NAME",
    "create_retry_session",
    "format_fips",
    "load_fips_codes",
    "fips_to_latlon",
    "validate_fips",
    "map_cause_code",
    # Processing
    "DataMatcher",
    "GraphBuilder",
    "CasCropDataset",
    "CasCropGraphBatch",
    "get_dataloaders",
]
