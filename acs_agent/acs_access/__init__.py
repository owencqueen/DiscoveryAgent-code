"""
ACS Access - Comprehensive American Community Survey API Client

A full-featured, async-capable Python client for accessing the U.S. Census Bureau's
American Community Survey (ACS) API.

Features:
- All ACS products (1-year, 5-year, subject tables, data profiles, comparison profiles)
- All available years (2009-present for 5-year, 2005-present for 1-year)
- All geography levels (nation to block group, plus special geographies)
- Full table downloads and individual variable queries
- Built-in geographic name resolution
- Intelligent metadata caching
- Async batch operations for high performance
- Rate limiting to respect Census API limits

Basic Usage:
    from acs_agent.acs_access import ACSClient
    
    async with ACSClient() as client:
        # Download a table
        result = await client.download_table(
            table_id="B01001",
            geography={"state": "Michigan"},
            year=2022,
            output_dir="./data"
        )
        
        # Search for tables
        tables = await client.search_tables("median income")

Quick Download:
    from acs_agent.acs_access import quick_download
    
    result = await quick_download("B01001", "Michigan", county="Wayne County")

Environment Variables:
    CENSUS_API_KEY: Optional API key for higher rate limits
    ACS_CACHE_PATH: Path to cache directory for metadata storage
"""

from importlib import metadata

__author__ = "ACS Agent Maintainers"

try:
    __version__ = metadata.version("acs-agent")
except metadata.PackageNotFoundError:
    __version__ = "0.1.0"

# Core client
from .client import ACSClient

# Configuration
from .config import (
    ACSConfig,
    ACSDataset,
    GeographyLevel,
    DATASET_YEARS,
    CENSUS_API_BASE,
)

# Data models
from .models import (
    GeographySpec,
    TableInfo,
    VariableInfo,
    DownloadResult,
)

# Geography resolution
from .geography import GeographyResolver

# Caching
from .cache import MetadataCache

# Convenience functions
from .utils import (
    quick_download,
    search_and_download,
)

# CLI entry point
from .cli import main as cli_main

__all__ = [
    # Client
    "ACSClient",
    # Config
    "ACSConfig",
    "ACSDataset", 
    "GeographyLevel",
    "DATASET_YEARS",
    "CENSUS_API_BASE",
    # Models
    "GeographySpec",
    "TableInfo",
    "VariableInfo",
    "DownloadResult",
    # Geography
    "GeographyResolver",
    # Cache
    "MetadataCache",
    # Utils
    "quick_download",
    "search_and_download",
    # CLI
    "cli_main",
]
