"""
Metadata caching for ACS Access.

This module provides disk-based caching for ACS metadata including:
- Table listings
- Variable definitions
- Geography hierarchies
- Resolved geographic codes

Caching significantly improves performance by avoiding redundant API calls
for metadata that changes infrequently.
"""

import hashlib
from pathlib import Path
from datetime import timedelta
from typing import Optional, Any

from diskcache import Cache


class MetadataCache:
    """
    Disk-based cache for ACS metadata.
    
    Uses diskcache for persistent storage with automatic TTL expiration.
    
    Attributes:
        cache: The underlying diskcache.Cache instance
        ttl: Time-to-live for cached items
        
    Example:
        cache = MetadataCache(Path("./cache"), ttl_days=30)
        
        # Store a value
        cache.set(tables_list, "tables", 2022, "acs/acs5")
        
        # Retrieve it later
        tables = cache.get("tables", 2022, "acs/acs5")
    """
    
    def __init__(self, cache_path: Path, ttl_days: int = 30):
        """
        Initialize the metadata cache.
        
        Args:
            cache_path: Directory to store cache files
            ttl_days: Number of days before cached items expire
        """
        cache_dir = cache_path / "metadata"
        cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.cache = Cache(str(cache_dir))
        self.ttl = timedelta(days=ttl_days)
    
    def _make_key(self, *args) -> str:
        """
        Create a unique cache key from arguments.
        
        Args:
            *args: Values to include in the key
            
        Returns:
            MD5 hash of the concatenated arguments
        """
        key_str = ":".join(str(a) for a in args)
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def get(self, *args) -> Optional[Any]:
        """
        Get a cached value if it exists and hasn't expired.
        
        Args:
            *args: Key components (same as used in set())
            
        Returns:
            Cached value or None if not found/expired
        """
        key = self._make_key(*args)
        return self.cache.get(key)
    
    def set(self, value: Any, *args) -> None:
        """
        Store a value in the cache.
        
        Args:
            value: The value to cache
            *args: Key components for retrieval
        """
        key = self._make_key(*args)
        self.cache.set(key, value, expire=self.ttl.total_seconds())
    
    def delete(self, *args) -> bool:
        """
        Delete a cached value.
        
        Args:
            *args: Key components
            
        Returns:
            True if the key existed and was deleted
        """
        key = self._make_key(*args)
        return self.cache.delete(key)
    
    def clear(self) -> None:
        """Clear all cached metadata."""
        self.cache.clear()
    
    def stats(self) -> dict:
        """
        Get cache statistics.
        
        Returns:
            Dict with cache size and item count
        """
        return {
            "size_bytes": self.cache.volume(),
            "item_count": len(self.cache),
        }
    
    def __contains__(self, args: tuple) -> bool:
        """Check if a key exists in cache."""
        key = self._make_key(*args)
        return key in self.cache
    
    def __len__(self) -> int:
        """Number of items in cache."""
        return len(self.cache)


class GeographyCache:
    """
    Specialized cache for resolved geography codes.
    
    Stores mappings from location names to FIPS codes to avoid
    repeated API lookups for the same locations.
    """
    
    def __init__(self, base_cache: MetadataCache):
        """
        Initialize with a base metadata cache.
        
        Args:
            base_cache: The underlying MetadataCache to use
        """
        self._cache = base_cache
        self._county_cache: dict[str, dict[str, str]] = {}
        self._place_cache: dict[str, dict[str, str]] = {}
    
    def get_counties(self, state_fips: str, year: int) -> Optional[dict[str, str]]:
        """
        Get cached county name to FIPS mappings for a state.
        
        Args:
            state_fips: State FIPS code
            year: ACS year
            
        Returns:
            Dict mapping county names to FIPS codes, or None
        """
        cache_key = f"counties_{state_fips}_{year}"
        if cache_key in self._county_cache:
            return self._county_cache[cache_key]
        
        # Check disk cache
        result = self._cache.get("geo_counties", state_fips, year)
        if result:
            self._county_cache[cache_key] = result
        return result
    
    def set_counties(self, state_fips: str, year: int, counties: dict[str, str]) -> None:
        """
        Cache county name to FIPS mappings for a state.
        
        Args:
            state_fips: State FIPS code
            year: ACS year
            counties: Dict mapping county names to FIPS codes
        """
        cache_key = f"counties_{state_fips}_{year}"
        self._county_cache[cache_key] = counties
        self._cache.set(counties, "geo_counties", state_fips, year)
    
    def get_places(self, state_fips: str, year: int) -> Optional[dict[str, str]]:
        """
        Get cached place name to FIPS mappings for a state.
        
        Args:
            state_fips: State FIPS code
            year: ACS year
            
        Returns:
            Dict mapping place names to FIPS codes, or None
        """
        cache_key = f"places_{state_fips}_{year}"
        if cache_key in self._place_cache:
            return self._place_cache[cache_key]
        
        result = self._cache.get("geo_places", state_fips, year)
        if result:
            self._place_cache[cache_key] = result
        return result
    
    def set_places(self, state_fips: str, year: int, places: dict[str, str]) -> None:
        """
        Cache place name to FIPS mappings for a state.
        
        Args:
            state_fips: State FIPS code
            year: ACS year
            places: Dict mapping place names to FIPS codes
        """
        cache_key = f"places_{state_fips}_{year}"
        self._place_cache[cache_key] = places
        self._cache.set(places, "geo_places", state_fips, year)
    
    def clear_memory_cache(self) -> None:
        """Clear in-memory caches (disk cache remains)."""
        self._county_cache.clear()
        self._place_cache.clear()


