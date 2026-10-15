"""
Configuration constants, enums, and settings for ACS Access.

This module contains:
- ACS dataset type definitions
- Geography level enums
- Year availability mappings
- State/territory FIPS codes
- Rate limit settings
- Default configuration
"""

import os
import logging
from pathlib import Path
from enum import Enum
from dataclasses import dataclass
from typing import Optional

from dotenv import load_dotenv
from rich.console import Console
from rich.logging import RichHandler

# Load environment variables
load_dotenv()

# Configure logging
console = Console()
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[RichHandler(console=console, rich_tracebacks=True)]
)
logger = logging.getLogger("acs_access")


# =============================================================================
# ENUMS
# =============================================================================

class ACSDataset(str, Enum):
    """
    Available ACS dataset types.
    
    Each dataset provides different types of data:
    - ACS1/ACS5: Detailed tables with individual estimates
    - SUBJECT: Pre-computed subject tables with related metrics
    - PROFILE: Data profiles with key indicators
    - CPROFILE: Comparison profiles for time comparisons
    - SPP: Supplemental estimates for smaller populations
    """
    ACS1 = "acs/acs1"                    # 1-year estimates (populations 65,000+)
    ACS5 = "acs/acs5"                    # 5-year estimates (all geographies)
    ACS1_SUBJECT = "acs/acs1/subject"    # 1-year subject tables
    ACS5_SUBJECT = "acs/acs5/subject"    # 5-year subject tables
    ACS1_PROFILE = "acs/acs1/profile"    # 1-year data profiles
    ACS5_PROFILE = "acs/acs5/profile"    # 5-year data profiles
    ACS5_CPROFILE = "acs/acs5/cprofile"  # 5-year comparison profiles
    ACS1_SPP = "acs/acs1/spp"            # 1-year supplemental estimates
    ACS5_SPP = "acs/acs5/spp"            # 5-year supplemental estimates


class GeographyLevel(str, Enum):
    """
    Supported geography levels with their Census API identifiers.
    
    Geography levels are hierarchical:
    - US (nation) is the top level
    - States contain counties, places, congressional districts
    - Counties contain tracts, county subdivisions
    - Tracts contain block groups
    
    Some geographies cross state boundaries (metro areas, etc.)
    """
    US = "us"
    REGION = "region"
    DIVISION = "division"
    STATE = "state"
    COUNTY = "county"
    COUNTY_SUBDIVISION = "county subdivision"
    PLACE = "place"
    TRACT = "tract"
    BLOCK_GROUP = "block group"
    CONGRESSIONAL_DISTRICT = "congressional district"
    SCHOOL_DISTRICT_ELEMENTARY = "school district (elementary)"
    SCHOOL_DISTRICT_SECONDARY = "school district (secondary)"
    SCHOOL_DISTRICT_UNIFIED = "school district (unified)"
    ZCTA = "zip code tabulation area"
    METRO_AREA = "metropolitan statistical area/micropolitan statistical area"
    CBSA = "combined statistical area"
    PUMA = "public use microdata area"
    NECTA = "new england city and town area"
    URBAN_AREA = "urban area"
    AIANNH = "american indian area/alaska native area/hawaiian home land"


# =============================================================================
# CONSTANTS
# =============================================================================

# Year availability by dataset type
DATASET_YEARS = {
    ACSDataset.ACS1: range(2005, 2025),           # 2005-2024
    ACSDataset.ACS5: range(2009, 2025),           # 2009-2024
    ACSDataset.ACS1_SUBJECT: range(2010, 2025),
    ACSDataset.ACS5_SUBJECT: range(2010, 2025),
    ACSDataset.ACS1_PROFILE: range(2010, 2025),
    ACSDataset.ACS5_PROFILE: range(2010, 2025),
    ACSDataset.ACS5_CPROFILE: range(2010, 2025),
    ACSDataset.ACS1_SPP: range(2014, 2025),
    ACSDataset.ACS5_SPP: range(2014, 2025),
}

# Base URL for Census API
CENSUS_API_BASE = "https://api.census.gov/data"

# Rate limits (requests per second)
RATE_LIMIT_ANONYMOUS = 50    # Conservative limit without API key
RATE_LIMIT_WITH_KEY = 500    # Higher limit with API key


# =============================================================================
# STATE & TERRITORY DATA
# =============================================================================

# Complete state/territory FIPS codes and names
STATE_FIPS = {
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
    # Territories
    "60": "American Samoa", "66": "Guam", "69": "Northern Mariana Islands",
    "72": "Puerto Rico", "78": "Virgin Islands"
}

# Reverse lookup: name to FIPS (lowercase)
STATE_TO_FIPS = {v.lower(): k for k, v in STATE_FIPS.items()}

# State abbreviation to FIPS code
STATE_ABBREV_TO_FIPS = {
    "al": "01", "ak": "02", "az": "04", "ar": "05", "ca": "06", "co": "08",
    "ct": "09", "de": "10", "dc": "11", "fl": "12", "ga": "13", "hi": "15",
    "id": "16", "il": "17", "in": "18", "ia": "19", "ks": "20", "ky": "21",
    "la": "22", "me": "23", "md": "24", "ma": "25", "mi": "26", "mn": "27",
    "ms": "28", "mo": "29", "mt": "30", "ne": "31", "nv": "32", "nh": "33",
    "nj": "34", "nm": "35", "ny": "36", "nc": "37", "nd": "38", "oh": "39",
    "ok": "40", "or": "41", "pa": "42", "ri": "44", "sc": "45", "sd": "46",
    "tn": "47", "tx": "48", "ut": "49", "vt": "50", "va": "51", "wa": "53",
    "wv": "54", "wi": "55", "wy": "56",
    # Territories
    "as": "60", "gu": "66", "mp": "69", "pr": "72", "vi": "78"
}

# FIPS to abbreviation (reverse of above)
FIPS_TO_STATE_ABBREV = {v: k.upper() for k, v in STATE_ABBREV_TO_FIPS.items()}


# =============================================================================
# CONFIGURATION CLASS
# =============================================================================

@dataclass
class ACSConfig:
    """
    Configuration for ACS client.
    
    Attributes:
        api_key: Census API key (optional, from env CENSUS_API_KEY)
        cache_path: Path to cache directory (from env ACS_CACHE_PATH)
        rate_limit: Requests per second limit
        max_concurrent_requests: Max parallel requests
        request_timeout: HTTP timeout in seconds
        max_retries: Number of retry attempts on failure
        cache_ttl_days: How long to cache metadata
    """
    api_key: Optional[str] = None
    cache_path: Optional[Path] = None
    rate_limit: int = RATE_LIMIT_ANONYMOUS
    max_concurrent_requests: int = 10
    request_timeout: int = 60
    max_retries: int = 3
    cache_ttl_days: int = 30
    
    def __post_init__(self):
        """Load from environment and set up directories."""
        # Load API key from environment if not provided
        if self.api_key is None:
            self.api_key = os.getenv("CENSUS_API_KEY")
        
        # Load cache path from environment if not provided
        if self.cache_path is None:
            cache_env = os.getenv("ACS_CACHE_PATH")
            if cache_env:
                self.cache_path = Path(cache_env)
            else:
                self.cache_path = Path.home() / ".cache" / "acs_data"
        
        # Adjust rate limit based on API key presence
        if self.api_key:
            self.rate_limit = RATE_LIMIT_WITH_KEY
            logger.info("Census API key detected - using higher rate limits")
        else:
            logger.warning(
                "No Census API key found - using conservative rate limits. "
                "Set CENSUS_API_KEY environment variable for higher limits."
            )
        
        # Ensure cache directory exists
        self.cache_path.mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def from_env(cls) -> "ACSConfig":
        """Create configuration from environment variables only."""
        return cls()
    
    def with_api_key(self, api_key: str) -> "ACSConfig":
        """Return a new config with a different API key."""
        return ACSConfig(
            api_key=api_key,
            cache_path=self.cache_path,
            rate_limit=RATE_LIMIT_WITH_KEY,
            max_concurrent_requests=self.max_concurrent_requests,
            request_timeout=self.request_timeout,
            max_retries=self.max_retries,
            cache_ttl_days=self.cache_ttl_days,
        )


