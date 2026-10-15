"""
Geographic name resolution for ACS Access.

This module provides the GeographyResolver class which converts
human-readable location names to Census FIPS codes.

Supported conversions:
- State names ("Michigan") and abbreviations ("MI") to FIPS codes
- County names ("Wayne County") to county FIPS codes
- Place/city names ("Detroit") to place FIPS codes
- Wildcard queries ("*" for all of a geography type)

Examples:
    resolver = GeographyResolver(cache)
    
    # Resolve a state
    spec = await resolver.resolve({"state": "Michigan"}, session)
    # -> GeographySpec(level=STATE, fips_code="26", name="Michigan")
    
    # Resolve a county
    spec = await resolver.resolve(
        {"state": "MI", "county": "Wayne County"}, 
        session
    )
    # -> GeographySpec(level=COUNTY, fips_code="163", parent_fips={"state": "26"})
    
    # All counties in a state
    spec = await resolver.resolve({"state": "TX", "county": "*"}, session)
    # -> GeographySpec(level=COUNTY, fips_code="*", parent_fips={"state": "48"})
"""

import re
import logging
from typing import Optional

import aiohttp

from .config import (
    GeographyLevel,
    STATE_FIPS,
    STATE_TO_FIPS,
    STATE_ABBREV_TO_FIPS,
    CENSUS_API_BASE,
)
from .models import GeographySpec
from .cache import MetadataCache, GeographyCache

logger = logging.getLogger("acs_access")


class GeographyResolver:
    """
    Resolves geographic names to Census FIPS codes.
    
    Handles:
    - State names and abbreviations
    - County names (with state context)
    - Place/city names (with state context)
    - Census tracts and block groups (pass-through)
    - Wildcards for "all" queries
    
    Uses caching to avoid redundant API lookups.
    """
    
    def __init__(self, cache: MetadataCache):
        """
        Initialize the resolver with a cache.
        
        Args:
            cache: MetadataCache instance for storing lookups
        """
        self._metadata_cache = cache
        self._geo_cache = GeographyCache(cache)
    
    async def resolve(
        self,
        geography: dict[str, str],
        session: aiohttp.ClientSession,
        year: int = 2022,
        dataset: str = "acs/acs5"
    ) -> GeographySpec:
        """
        Resolve a geography specification to FIPS codes.
        
        Args:
            geography: Dict with geographic identifiers, e.g.:
                {"state": "Michigan"}
                {"state": "MI", "county": "Wayne County"}
                {"state": "26", "place": "Detroit"}
                {"state": "*"}  # All states
            session: aiohttp session for API calls
            year: ACS year for geography lookup
            dataset: ACS dataset
            
        Returns:
            GeographySpec with resolved FIPS codes
            
        Raises:
            ValueError: If geography cannot be resolved
        """
        resolved = {}
        name_parts = []
        
        # Handle US-level query
        if "us" in geography:
            return GeographySpec(
                level=GeographyLevel.US,
                fips_code="*" if geography["us"] == "*" else "1",
                name="United States"
            )
        
        # Resolve state (required for most queries)
        if "state" in geography:
            state_val = geography["state"]
            if state_val == "*":
                resolved["state"] = "*"
                name_parts.append("All States")
            else:
                state_fips = self._resolve_state(state_val)
                if state_fips is None:
                    raise ValueError(f"Could not resolve state: {state_val}")
                resolved["state"] = state_fips
                name_parts.append(STATE_FIPS.get(state_fips, state_val))
        
        # Resolve county
        if "county" in geography:
            county_val = geography["county"]
            if county_val == "*":
                resolved["county"] = "*"
                name_parts.append("All Counties")
            else:
                if "state" not in resolved:
                    raise ValueError("County requires state specification")
                county_fips = await self._resolve_county(
                    county_val, resolved["state"], session, year, dataset
                )
                if county_fips is None:
                    raise ValueError(f"Could not resolve county: {county_val}")
                resolved["county"] = county_fips
                name_parts.append(county_val)
        
        # Resolve place/city
        if "place" in geography:
            place_val = geography["place"]
            if place_val == "*":
                resolved["place"] = "*"
                name_parts.append("All Places")
            else:
                if "state" not in resolved:
                    raise ValueError("Place requires state specification")
                place_fips = await self._resolve_place(
                    place_val, resolved["state"], session, year, dataset
                )
                if place_fips is None:
                    raise ValueError(f"Could not resolve place: {place_val}")
                resolved["place"] = place_fips
                name_parts.append(place_val)
        
        # Pass-through for tract (requires state and county)
        if "tract" in geography:
            tract_val = geography["tract"]
            if "state" not in resolved or "county" not in resolved:
                raise ValueError("Tract requires state and county specification")
            resolved["tract"] = tract_val
            if tract_val != "*":
                name_parts.append(f"Tract {tract_val}")
            else:
                name_parts.append("All Tracts")
        
        # Pass-through for block group
        bg_key = "block_group" if "block_group" in geography else "block group"
        if bg_key in geography:
            bg_val = geography[bg_key]
            if "tract" not in resolved:
                raise ValueError("Block group requires tract specification")
            resolved["block group"] = bg_val
            if bg_val != "*":
                name_parts.append(f"Block Group {bg_val}")
            else:
                name_parts.append("All Block Groups")
        
        # ZCTA (zip code tabulation area)
        zcta_key = next((k for k in ["zcta", "zip", "zip code tabulation area"] if k in geography), None)
        if zcta_key:
            resolved["zip code tabulation area"] = geography[zcta_key]
            if geography[zcta_key] != "*":
                name_parts.append(f"ZCTA {geography[zcta_key]}")
            else:
                name_parts.append("All ZCTAs")
        
        # Congressional district
        cd_key = next((k for k in ["congressional_district", "congressional district"] if k in geography), None)
        if cd_key:
            resolved["congressional district"] = geography[cd_key]
            if geography[cd_key] != "*":
                name_parts.append(f"CD-{geography[cd_key]}")
            else:
                name_parts.append("All Congressional Districts")
        
        # PUMA (public use microdata area)
        if "puma" in geography:
            resolved["public use microdata area"] = geography["puma"]
            if geography["puma"] != "*":
                name_parts.append(f"PUMA {geography['puma']}")
            else:
                name_parts.append("All PUMAs")
        
        # Determine target level and build spec
        target_level = self._get_target_level(resolved)
        target_code = self._get_target_code(resolved, target_level)
        parent_fips = self._get_parent_fips(resolved, target_level)
        
        return GeographySpec(
            level=target_level,
            fips_code=target_code,
            parent_fips=parent_fips,
            name=", ".join(name_parts)
        )
    
    def _resolve_state(self, state: str) -> Optional[str]:
        """
        Resolve state name/abbreviation to FIPS code.
        
        Args:
            state: State name, abbreviation, or FIPS code
            
        Returns:
            2-digit FIPS code or None if not found
        """
        state_lower = state.lower().strip()
        
        # Already a FIPS code?
        if state_lower.isdigit() and len(state_lower) <= 2:
            return state_lower.zfill(2)
        
        # Check abbreviation
        if state_lower in STATE_ABBREV_TO_FIPS:
            return STATE_ABBREV_TO_FIPS[state_lower]
        
        # Check full name
        if state_lower in STATE_TO_FIPS:
            return STATE_TO_FIPS[state_lower]
        
        # Fuzzy match - find partial matches
        for name, fips in STATE_TO_FIPS.items():
            if state_lower in name or name in state_lower:
                return fips
        
        return None
    
    async def _resolve_county(
        self,
        county: str,
        state_fips: str,
        session: aiohttp.ClientSession,
        year: int,
        dataset: str
    ) -> Optional[str]:
        """
        Resolve county name to FIPS code.
        
        Args:
            county: County name
            state_fips: State FIPS code
            session: aiohttp session
            year: ACS year
            dataset: ACS dataset
            
        Returns:
            County FIPS code or None if not found
        """
        # Check cache first
        counties = self._geo_cache.get_counties(state_fips, year)
        
        if counties is None:
            # Fetch from API
            counties = await self._fetch_geography_list(
                session, year, dataset, "county", {"state": state_fips}
            )
            if counties:
                self._geo_cache.set_counties(state_fips, year, counties)
        
        if not counties:
            return None
        
        return self._match_name(county, counties, strip_suffix="county")
    
    async def _resolve_place(
        self,
        place: str,
        state_fips: str,
        session: aiohttp.ClientSession,
        year: int,
        dataset: str
    ) -> Optional[str]:
        """
        Resolve place/city name to FIPS code.
        
        Args:
            place: Place/city name
            state_fips: State FIPS code
            session: aiohttp session
            year: ACS year
            dataset: ACS dataset
            
        Returns:
            Place FIPS code or None if not found
        """
        # Check cache first
        places = self._geo_cache.get_places(state_fips, year)
        
        if places is None:
            # Fetch from API
            places = await self._fetch_geography_list(
                session, year, dataset, "place", {"state": state_fips}
            )
            if places:
                self._geo_cache.set_places(state_fips, year, places)
        
        if not places:
            return None
        
        return self._match_name(place, places, strip_suffix=r"(city|town|village|cdp|borough)")
    
    def _match_name(
        self,
        query: str,
        name_to_fips: dict[str, str],
        strip_suffix: Optional[str] = None
    ) -> Optional[str]:
        """
        Match a query string against a dictionary of names.
        
        Args:
            query: The name to search for
            name_to_fips: Dict mapping names to FIPS codes
            strip_suffix: Regex pattern for suffix to strip before matching
            
        Returns:
            Matching FIPS code or None
        """
        query_lower = query.lower().strip()
        
        # Clean query
        if strip_suffix:
            query_clean = re.sub(rf'\s*{strip_suffix}\s*$', '', query_lower, flags=re.IGNORECASE)
        else:
            query_clean = query_lower
        
        # Exact match first
        for name, fips in name_to_fips.items():
            name_lower = name.lower()
            if strip_suffix:
                name_clean = re.sub(rf'\s*{strip_suffix}\s*$', '', name_lower, flags=re.IGNORECASE)
            else:
                name_clean = name_lower
            
            if query_clean == name_clean or query_lower == name_lower:
                return fips
        
        # Fuzzy match
        for name, fips in name_to_fips.items():
            name_lower = name.lower()
            if strip_suffix:
                name_clean = re.sub(rf'\s*{strip_suffix}\s*$', '', name_lower, flags=re.IGNORECASE)
            else:
                name_clean = name_lower
            
            if query_clean in name_clean or name_clean in query_clean:
                return fips
        
        return None
    
    async def _fetch_geography_list(
        self,
        session: aiohttp.ClientSession,
        year: int,
        dataset: str,
        geo_type: str,
        parent_geo: dict
    ) -> dict[str, str]:
        """
        Fetch list of geographies from Census API.
        
        Args:
            session: aiohttp session
            year: ACS year
            dataset: ACS dataset
            geo_type: Geography type (county, place, etc.)
            parent_geo: Parent geography specification
            
        Returns:
            Dict mapping geography names to FIPS codes
        """
        url = f"{CENSUS_API_BASE}/{year}/{dataset}"
        params = {
            "get": "NAME",
            "for": f"{geo_type}:*"
        }
        
        if parent_geo:
            in_parts = [f"{k}:{v}" for k, v in parent_geo.items()]
            params["in"] = " ".join(in_parts)
        
        try:
            async with session.get(url, params=params) as resp:
                if resp.status != 200:
                    logger.warning(f"Failed to fetch {geo_type} list: HTTP {resp.status}")
                    return {}
                
                data = await resp.json()
                
                if not data or len(data) < 2:
                    return {}
                
                headers = data[0]
                name_idx = headers.index("NAME") if "NAME" in headers else 0
                geo_idx = headers.index(geo_type) if geo_type in headers else -1
                
                if geo_idx == -1:
                    return {}
                
                return {row[name_idx]: row[geo_idx] for row in data[1:]}
        
        except Exception as e:
            logger.error(f"Error fetching geography list: {e}")
            return {}
    
    def _get_target_level(self, resolved: dict) -> GeographyLevel:
        """
        Determine the most granular geography level from resolved dict.
        
        Args:
            resolved: Dict of resolved geography components
            
        Returns:
            The most granular GeographyLevel
        """
        # Order from most to least granular
        level_priority = [
            ("block group", GeographyLevel.BLOCK_GROUP),
            ("tract", GeographyLevel.TRACT),
            ("place", GeographyLevel.PLACE),
            ("county subdivision", GeographyLevel.COUNTY_SUBDIVISION),
            ("county", GeographyLevel.COUNTY),
            ("congressional district", GeographyLevel.CONGRESSIONAL_DISTRICT),
            ("zip code tabulation area", GeographyLevel.ZCTA),
            ("public use microdata area", GeographyLevel.PUMA),
            ("state", GeographyLevel.STATE),
            ("division", GeographyLevel.DIVISION),
            ("region", GeographyLevel.REGION),
        ]
        
        for key, level in level_priority:
            if key in resolved:
                return level
        
        return GeographyLevel.US
    
    def _get_target_code(self, resolved: dict, level: GeographyLevel) -> str:
        """Get the FIPS code for the target level."""
        level_value = level.value
        return resolved.get(level_value, resolved.get(level_value.replace(" ", "_"), "*"))
    
    def _get_parent_fips(self, resolved: dict, target_level: GeographyLevel) -> dict:
        """Build parent FIPS dict excluding the target level."""
        parent_fips = {}
        target_value = target_level.value
        
        for key, val in resolved.items():
            normalized_key = key.replace("_", " ")
            if normalized_key != target_value:
                parent_fips[normalized_key] = val
        
        return parent_fips
    
    async def list_counties(
        self,
        state: str,
        session: aiohttp.ClientSession,
        year: int = 2022,
        dataset: str = "acs/acs5"
    ) -> dict[str, str]:
        """
        List all counties in a state.
        
        Args:
            state: State name, abbreviation, or FIPS
            session: aiohttp session
            year: ACS year
            dataset: ACS dataset
            
        Returns:
            Dict mapping county names to FIPS codes
        """
        state_fips = self._resolve_state(state)
        if not state_fips:
            raise ValueError(f"Could not resolve state: {state}")
        
        counties = self._geo_cache.get_counties(state_fips, year)
        if counties is None:
            counties = await self._fetch_geography_list(
                session, year, dataset, "county", {"state": state_fips}
            )
            if counties:
                self._geo_cache.set_counties(state_fips, year, counties)
        
        return counties or {}
    
    async def list_places(
        self,
        state: str,
        session: aiohttp.ClientSession,
        year: int = 2022,
        dataset: str = "acs/acs5"
    ) -> dict[str, str]:
        """
        List all places/cities in a state.
        
        Args:
            state: State name, abbreviation, or FIPS
            session: aiohttp session
            year: ACS year
            dataset: ACS dataset
            
        Returns:
            Dict mapping place names to FIPS codes
        """
        state_fips = self._resolve_state(state)
        if not state_fips:
            raise ValueError(f"Could not resolve state: {state}")
        
        places = self._geo_cache.get_places(state_fips, year)
        if places is None:
            places = await self._fetch_geography_list(
                session, year, dataset, "place", {"state": state_fips}
            )
            if places:
                self._geo_cache.set_places(state_fips, year, places)
        
        return places or {}


