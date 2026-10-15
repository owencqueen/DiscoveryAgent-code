"""
Main ACS API client for ACS Access.

This module contains the ACSClient class, the primary interface for
interacting with the Census ACS API.

Features:
- Async HTTP requests with rate limiting
- Automatic retries with exponential backoff
- Metadata discovery (tables, variables, geographies)
- Data downloads (tables, variables, time series, batch)
- Built-in geographic resolution

Example:
    async with ACSClient() as client:
        # List available tables
        tables = await client.list_tables(year=2022)
        
        # Download a table
        result = await client.download_table(
            table_id="B01001",
            geography={"state": "Michigan"},
            year=2022,
            output_dir="./data"
        )
"""

import re
import json
import asyncio
import logging
import hashlib
from pathlib import Path
from typing import Optional, Union

import aiohttp
import pandas as pd
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)
from aiolimiter import AsyncLimiter
from tqdm.asyncio import tqdm

from .config import (
    ACSConfig,
    ACSDataset,
    DATASET_YEARS,
    CENSUS_API_BASE,
    logger,
)
from .models import (
    GeographySpec,
    TableInfo,
    VariableInfo,
    DownloadResult,
)
from .cache import MetadataCache
from .geography import GeographyResolver


class ACSClient:
    """
    Comprehensive async client for the American Community Survey API.
    
    This is the main interface for accessing ACS data. It handles:
    - Rate limiting and request throttling
    - Automatic retries on failures
    - Metadata caching
    - Geographic name resolution
    - Data downloads in various formats
    
    Usage:
        # As async context manager (recommended)
        async with ACSClient() as client:
            result = await client.download_table("B01001", {"state": "MI"})
        
        # Manual lifecycle management
        client = ACSClient()
        try:
            result = await client.download_table("B01001", {"state": "MI"})
        finally:
            await client.close()
    
    Configuration:
        Pass an ACSConfig object or let it load from environment variables:
        - CENSUS_API_KEY: Your Census API key (optional but recommended)
        - ACS_CACHE_PATH: Directory for metadata cache
    """
    
    def __init__(self, config: Optional[ACSConfig] = None):
        """
        Initialize the ACS client.
        
        Args:
            config: Configuration object. If None, loads from environment.
        """
        self.config = config or ACSConfig()
        self.cache = MetadataCache(self.config.cache_path, self.config.cache_ttl_days)
        self.geo_resolver = GeographyResolver(self.cache)
        
        # Rate limiter (requests per second)
        self.rate_limiter = AsyncLimiter(
            self.config.rate_limit,
            time_period=1.0
        )
        
        # Semaphore for concurrent request limiting
        self.semaphore = asyncio.Semaphore(self.config.max_concurrent_requests)
        
        # HTTP session (created lazily)
        self._session: Optional[aiohttp.ClientSession] = None
    
    async def __aenter__(self) -> "ACSClient":
        """Async context manager entry."""
        await self._ensure_session()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()
    
    async def _ensure_session(self) -> None:
        """Ensure aiohttp session exists."""
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=self.config.request_timeout)
            self._session = aiohttp.ClientSession(timeout=timeout)
    
    async def close(self) -> None:
        """Close the HTTP session."""
        if self._session and not self._session.closed:
            await self._session.close()
    
    def _build_url(self, year: int, dataset: str) -> str:
        """Build the base API URL for a year/dataset."""
        return f"{CENSUS_API_BASE}/{year}/{dataset}"
    
    def _add_api_key(self, params: dict) -> dict:
        """Add API key to params if available."""
        if self.config.api_key:
            params["key"] = self.config.api_key
        return params
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        retry=retry_if_exception_type((aiohttp.ClientError, asyncio.TimeoutError))
    )
    async def _request(
        self,
        url: str,
        params: dict,
        description: str = "API request"
    ) -> Optional[list]:
        """
        Make a rate-limited, retrying API request.
        
        Args:
            url: API endpoint URL
            params: Query parameters
            description: Description for logging
            
        Returns:
            JSON response as list, or None on failure
        """
        await self._ensure_session()
        
        async with self.semaphore:
            async with self.rate_limiter:
                params = self._add_api_key(params)
                
                try:
                    async with self._session.get(url, params=params) as resp:
                        if resp.status == 204:
                            return None
                        
                        if resp.status == 400:
                            error_text = await resp.text()
                            logger.error(f"Bad request for {description}: {error_text}")
                            return None
                        
                        if resp.status == 404:
                            logger.warning(f"Not found: {description}")
                            return None
                        
                        if resp.status != 200:
                            logger.error(f"HTTP {resp.status} for {description}")
                            return None
                        
                        return await resp.json()
                
                except aiohttp.ContentTypeError as e:
                    logger.error(f"Invalid JSON response for {description}: {e}")
                    return None
    
    # =========================================================================
    # METADATA DISCOVERY
    # =========================================================================
    
    async def list_datasets(self, year: int = 2022) -> list[dict]:
        """
        List all available ACS datasets for a given year.
        
        Args:
            year: The year to query
            
        Returns:
            List of dataset information dicts with keys:
            - identifier: Dataset path (e.g., "acs/acs5")
            - title: Human-readable title
            - description: Dataset description
            - modified: Last modification date
        """
        cached = self.cache.get("datasets", year)
        if cached:
            return cached
        
        await self._ensure_session()
        url = f"{CENSUS_API_BASE}/{year}.json"
        
        async with self._session.get(url) as resp:
            if resp.status != 200:
                return []
            data = await resp.json()
        
        # Filter to ACS datasets
        acs_datasets = []
        for ds in data.get("dataset", []):
            if "acs" in ds.get("c_dataset", [""])[0].lower():
                acs_datasets.append({
                    "identifier": "/".join(ds.get("c_dataset", [])),
                    "title": ds.get("title", ""),
                    "description": ds.get("description", ""),
                    "modified": ds.get("modified", "")
                })
        
        self.cache.set(acs_datasets, "datasets", year)
        return acs_datasets
    
    async def list_tables(
        self,
        year: int = 2022,
        dataset: Union[str, ACSDataset] = ACSDataset.ACS5
    ) -> list[TableInfo]:
        """
        List all available tables for a dataset/year.
        
        Args:
            year: ACS year
            dataset: ACS dataset type
            
        Returns:
            List of TableInfo objects with table_id, title, universe
        """
        ds = dataset.value if isinstance(dataset, ACSDataset) else dataset
        
        cached = self.cache.get("tables", year, ds)
        if cached:
            return cached
        
        await self._ensure_session()
        url = f"{CENSUS_API_BASE}/{year}/{ds}/groups.json"
        
        async with self._session.get(url) as resp:
            if resp.status != 200:
                logger.error(f"Failed to fetch tables: HTTP {resp.status}")
                return []
            data = await resp.json()
        
        tables = []
        for group_id, group_data in data.get("groups", {}).items():
            if group_id == "N/A":
                continue
            tables.append(TableInfo(
                table_id=group_id,
                title=group_data.get("description", ""),
                universe=group_data.get("universe", ""),
                dataset=ds,
                year=year
            ))
        
        self.cache.set(tables, "tables", year, ds)
        return tables
    
    async def get_table_info(
        self,
        table_id: str,
        year: int = 2022,
        dataset: Union[str, ACSDataset] = ACSDataset.ACS5
    ) -> Optional[TableInfo]:
        """
        Get detailed information about a specific table.
        
        Args:
            table_id: Table ID (e.g., "B01001")
            year: ACS year
            dataset: ACS dataset type
            
        Returns:
            TableInfo with variables populated, or None if not found
        """
        ds = dataset.value if isinstance(dataset, ACSDataset) else dataset
        
        cached = self.cache.get("table_info", table_id, year, ds)
        if cached:
            return cached
        
        await self._ensure_session()
        url = f"{CENSUS_API_BASE}/{year}/{ds}/groups/{table_id}.json"
        
        async with self._session.get(url) as resp:
            if resp.status != 200:
                return None
            data = await resp.json()
        
        variables = []
        for var_name, var_data in data.get("variables", {}).items():
            if var_name in ["NAME", "GEO_ID"] or "_" not in var_name:
                continue
            variables.append(VariableInfo(
                name=var_name,
                label=var_data.get("label", ""),
                concept=var_data.get("concept", ""),
                group=var_data.get("group", table_id),
                predicateType=var_data.get("predicateType", "")
            ))
        
        # Sort by variable suffix number
        def var_sort_key(v):
            match = re.search(r'_(\d+)([EM]?)$', v.name)
            if match:
                return (int(match.group(1)), match.group(2))
            return (0, "")
        
        variables.sort(key=var_sort_key)
        
        table_info = TableInfo(
            table_id=table_id,
            title=data.get("description", ""),
            universe=data.get("universe", ""),
            variables=variables,
            dataset=ds,
            year=year
        )
        
        self.cache.set(table_info, "table_info", table_id, year, ds)
        return table_info
    
    async def list_variables(
        self,
        year: int = 2022,
        dataset: Union[str, ACSDataset] = ACSDataset.ACS5,
        limit: Optional[int] = None
    ) -> list[VariableInfo]:
        """
        List all available variables for a dataset/year.
        
        Args:
            year: ACS year
            dataset: ACS dataset type
            limit: Maximum number of variables to return
            
        Returns:
            List of VariableInfo objects
        """
        ds = dataset.value if isinstance(dataset, ACSDataset) else dataset
        
        cached = self.cache.get("variables", year, ds)
        if cached:
            return cached[:limit] if limit else cached
        
        await self._ensure_session()
        url = f"{CENSUS_API_BASE}/{year}/{ds}/variables.json"
        
        async with self._session.get(url) as resp:
            if resp.status != 200:
                return []
            data = await resp.json()
        
        variables = []
        for var_name, var_data in data.get("variables", {}).items():
            if var_name in ["for", "in", "ucgid", "NAME", "GEO_ID"]:
                continue
            if not re.match(r'^[A-Z]\d+', var_name):
                continue
            
            variables.append(VariableInfo(
                name=var_name,
                label=var_data.get("label", ""),
                concept=var_data.get("concept", ""),
                group=var_data.get("group", ""),
                predicateType=var_data.get("predicateType", "")
            ))
        
        self.cache.set(variables, "variables", year, ds)
        return variables[:limit] if limit else variables
    
    async def search_tables(
        self,
        query: str,
        year: int = 2022,
        dataset: Union[str, ACSDataset] = ACSDataset.ACS5,
        max_results: int = 20
    ) -> list[TableInfo]:
        """
        Search for tables matching a query string.
        
        Args:
            query: Search string (e.g., "income", "education")
            year: ACS year
            dataset: ACS dataset type
            max_results: Maximum results to return
            
        Returns:
            List of matching TableInfo objects, ranked by relevance
        """
        tables = await self.list_tables(year, dataset)
        
        query_lower = query.lower()
        query_terms = query_lower.split()
        
        scored_tables = []
        for table in tables:
            title_lower = table.title.lower()
            universe_lower = (table.universe or "").lower()
            
            score = 0
            for term in query_terms:
                if term in title_lower:
                    score += 2
                if term in universe_lower:
                    score += 1
                if term in table.table_id.lower():
                    score += 3
            
            if score > 0:
                scored_tables.append((score, table))
        
        scored_tables.sort(key=lambda x: -x[0])
        return [t for _, t in scored_tables[:max_results]]
    
    async def search_variables(
        self,
        query: str,
        year: int = 2022,
        dataset: Union[str, ACSDataset] = ACSDataset.ACS5,
        max_results: int = 50
    ) -> list[VariableInfo]:
        """
        Search for variables matching a query string.
        
        Args:
            query: Search string
            year: ACS year
            dataset: ACS dataset type
            max_results: Maximum results to return
            
        Returns:
            List of matching VariableInfo objects, ranked by relevance
        """
        variables = await self.list_variables(year, dataset)
        
        query_lower = query.lower()
        query_terms = query_lower.split()
        
        scored_vars = []
        for var in variables:
            label_lower = var.label.lower()
            concept_lower = (var.concept or "").lower()
            
            score = 0
            for term in query_terms:
                if term in label_lower:
                    score += 2
                if term in concept_lower:
                    score += 1
                if term in var.name.lower():
                    score += 3
            
            if score > 0:
                scored_vars.append((score, var))
        
        scored_vars.sort(key=lambda x: -x[0])
        return [v for _, v in scored_vars[:max_results]]
    
    async def list_geographies(
        self,
        year: int = 2022,
        dataset: Union[str, ACSDataset] = ACSDataset.ACS5
    ) -> list[dict]:
        """
        List all available geography levels for a dataset.
        
        Args:
            year: ACS year
            dataset: ACS dataset type
            
        Returns:
            List of geography level specifications from Census API
        """
        ds = dataset.value if isinstance(dataset, ACSDataset) else dataset
        
        cached = self.cache.get("geographies", year, ds)
        if cached:
            return cached
        
        await self._ensure_session()
        url = f"{CENSUS_API_BASE}/{year}/{ds}/geography.json"
        
        async with self._session.get(url) as resp:
            if resp.status != 200:
                return []
            data = await resp.json()
        
        geos = data.get("fips", [])
        self.cache.set(geos, "geographies", year, ds)
        return geos
    
    # =========================================================================
    # DATA DOWNLOAD
    # =========================================================================
    
    async def download_table(
        self,
        table_id: str,
        geography: dict[str, str],
        year: int = 2022,
        dataset: Union[str, ACSDataset] = ACSDataset.ACS5,
        output_dir: Union[str, Path] = ".",
        include_moe: bool = True,
        filename: Optional[str] = None
    ) -> DownloadResult:
        """
        Download a complete ACS table for a geography.
        
        Args:
            table_id: Table ID (e.g., "B01001")
            geography: Geography specification dict, e.g.:
                {"state": "Michigan"}
                {"state": "MI", "county": "Wayne County"}
                {"state": "*"}  # All states
            year: ACS year (default 2022)
            dataset: ACS dataset type (default ACS 5-year)
            output_dir: Directory to save CSV file
            include_moe: Include margin of error columns (default True)
            filename: Custom filename (default: auto-generated)
            
        Returns:
            DownloadResult with success status, file path, and metadata
        """
        ds = dataset.value if isinstance(dataset, ACSDataset) else dataset
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        await self._ensure_session()
        
        try:
            # Resolve geography
            geo_spec = await self.geo_resolver.resolve(
                geography, self._session, year, ds
            )
            
            # Get table metadata
            table_info = await self.get_table_info(table_id, year, ds)
            if not table_info:
                return DownloadResult(
                    success=False,
                    error=f"Table {table_id} not found for {ds} {year}"
                )
            
            # Build API request
            url = self._build_url(year, ds)
            for_param, in_param = geo_spec.to_api_params()
            
            params = {
                "get": f"NAME,group({table_id})",
                "for": for_param
            }
            if in_param:
                params["in"] = in_param
            
            # Make request
            data = await self._request(url, params, f"table {table_id}")
            
            if not data or len(data) < 2:
                return DownloadResult(
                    success=False,
                    error="No data returned from API"
                )
            
            # Convert to DataFrame
            df = pd.DataFrame(data[1:], columns=data[0])
            
            # Filter MOE columns if requested
            if not include_moe:
                moe_cols = [c for c in df.columns if c.endswith("M")]
                df = df.drop(columns=moe_cols)
            
            # Add metadata columns
            df["_year"] = year
            df["_dataset"] = ds
            df["_table_id"] = table_id
            
            # Generate filename
            if filename is None:
                geo_str = geo_spec.name.replace(", ", "_").replace(" ", "_")
                geo_str = re.sub(r'[^\w\-_]', '', geo_str)
                filename = f"{table_id}_{geo_str}_{year}.csv"
            
            file_path = output_path / filename
            
            # Save to CSV
            df.to_csv(file_path, index=False)
            
            # Save data dictionary alongside CSV
            dict_filename = filename.replace(".csv", "_dictionary.json")
            dict_path = output_path / dict_filename
            
            # Build variable dictionary: code -> label mapping
            variable_dict = {
                "_metadata": {
                    "table_id": table_id,
                    "table_title": table_info.title,
                    "universe": table_info.universe,
                    "geography": geo_spec.name,
                    "year": year,
                    "dataset": ds,
                    "source": "U.S. Census Bureau American Community Survey",
                    "notes": "Variables ending in 'E' are estimates, 'M' are margins of error"
                },
                "variables": {
                    v.name: {
                        "label": v.label,
                        "concept": v.concept if hasattr(v, 'concept') else None
                    }
                    for v in table_info.variables
                }
            }
            
            with open(dict_path, 'w') as f:
                json.dump(variable_dict, f, indent=2)
            
            return DownloadResult(
                success=True,
                file_path=file_path,
                rows=len(df),
                columns=len(df.columns),
                metadata={
                    "table_id": table_id,
                    "table_title": table_info.title,
                    "geography": geo_spec.name,
                    "year": year,
                    "dataset": ds,
                    "variable_count": len(table_info.variables),
                    "dictionary_path": str(dict_path)
                }
            )
        
        except Exception as e:
            logger.error(f"Error downloading table {table_id}: {e}")
            return DownloadResult(success=False, error=str(e))
    
    async def download_variables(
        self,
        variables: list[str],
        geography: dict[str, str],
        year: int = 2022,
        dataset: Union[str, ACSDataset] = ACSDataset.ACS5,
        output_dir: Union[str, Path] = ".",
        filename: Optional[str] = None
    ) -> DownloadResult:
        """
        Download specific ACS variables for a geography.
        
        Args:
            variables: List of variable names (e.g., ["B01001_001E", "B19013_001E"])
            geography: Geography specification dict
            year: ACS year
            dataset: ACS dataset type
            output_dir: Directory to save CSV file
            filename: Custom filename
            
        Returns:
            DownloadResult with file path and metadata
        """
        ds = dataset.value if isinstance(dataset, ACSDataset) else dataset
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        await self._ensure_session()
        
        try:
            geo_spec = await self.geo_resolver.resolve(
                geography, self._session, year, ds
            )
            
            url = self._build_url(year, ds)
            for_param, in_param = geo_spec.to_api_params()
            
            # Census API limits variables per request (~50)
            var_chunks = [variables[i:i+50] for i in range(0, len(variables), 50)]
            
            all_dfs = []
            for chunk in var_chunks:
                var_str = ",".join(["NAME"] + chunk)
                
                params = {
                    "get": var_str,
                    "for": for_param
                }
                if in_param:
                    params["in"] = in_param
                
                data = await self._request(url, params, "variables")
                
                if data and len(data) >= 2:
                    chunk_df = pd.DataFrame(data[1:], columns=data[0])
                    all_dfs.append(chunk_df)
            
            if not all_dfs:
                return DownloadResult(
                    success=False,
                    error="No data returned from API"
                )
            
            # Merge chunks
            df = all_dfs[0]
            for chunk_df in all_dfs[1:]:
                merge_cols = [c for c in df.columns if c not in variables and c != "NAME"]
                df = df.merge(chunk_df, on=merge_cols, how="outer")
            
            df["_year"] = year
            df["_dataset"] = ds
            
            if filename is None:
                geo_str = geo_spec.name.replace(", ", "_").replace(" ", "_")
                geo_str = re.sub(r'[^\w\-_]', '', geo_str)
                var_hash = hashlib.md5(",".join(variables).encode()).hexdigest()[:8]
                filename = f"vars_{var_hash}_{geo_str}_{year}.csv"
            
            file_path = output_path / filename
            df.to_csv(file_path, index=False)
            
            # Save data dictionary for the requested variables
            dict_filename = filename.replace(".csv", "_dictionary.json")
            dict_path = output_path / dict_filename
            
            # Look up variable labels
            try:
                var_labels = await self.get_variable_labels(variables, year, ds)
            except Exception:
                var_labels = {v: v for v in variables}  # Fall back to using variable names
            
            variable_dict = {
                "_metadata": {
                    "geography": geo_spec.name,
                    "year": year,
                    "dataset": ds,
                    "source": "U.S. Census Bureau American Community Survey",
                    "notes": "Variables ending in 'E' are estimates, 'M' are margins of error"
                },
                "variables": {v: {"label": var_labels.get(v, v)} for v in variables}
            }
            
            with open(dict_path, 'w') as f:
                json.dump(variable_dict, f, indent=2)
            
            return DownloadResult(
                success=True,
                file_path=file_path,
                rows=len(df),
                columns=len(df.columns),
                metadata={
                    "variables": variables,
                    "geography": geo_spec.name,
                    "year": year,
                    "dataset": ds,
                    "dictionary_path": str(dict_path)
                }
            )
        
        except Exception as e:
            logger.error(f"Error downloading variables: {e}")
            return DownloadResult(success=False, error=str(e))
    
    async def download_multiple_tables(
        self,
        table_ids: list[str],
        geography: dict[str, str],
        year: int = 2022,
        dataset: Union[str, ACSDataset] = ACSDataset.ACS5,
        output_dir: Union[str, Path] = ".",
        include_moe: bool = True
    ) -> list[DownloadResult]:
        """
        Download multiple tables concurrently.
        
        Args:
            table_ids: List of table IDs
            geography: Geography specification
            year: ACS year
            dataset: ACS dataset type
            output_dir: Directory to save files
            include_moe: Include margin of error columns
            
        Returns:
            List of DownloadResult objects
        """
        tasks = [
            self.download_table(
                table_id=tid,
                geography=geography,
                year=year,
                dataset=dataset,
                output_dir=output_dir,
                include_moe=include_moe
            )
            for tid in table_ids
        ]
        
        results = await tqdm.gather(
            *tasks,
            desc="Downloading tables",
            total=len(tasks)
        )
        
        return list(results)
    
    async def download_table_multiple_geographies(
        self,
        table_id: str,
        geographies: list[dict[str, str]],
        year: int = 2022,
        dataset: Union[str, ACSDataset] = ACSDataset.ACS5,
        output_dir: Union[str, Path] = ".",
        include_moe: bool = True,
        combine: bool = True
    ) -> Union[DownloadResult, list[DownloadResult]]:
        """
        Download a table for multiple geographies.
        
        Args:
            table_id: Table ID
            geographies: List of geography specifications
            year: ACS year
            dataset: ACS dataset type
            output_dir: Directory to save files
            include_moe: Include MOE columns
            combine: Combine all results into single file
            
        Returns:
            Single combined DownloadResult if combine=True, else list
        """
        tasks = [
            self.download_table(
                table_id=table_id,
                geography=geo,
                year=year,
                dataset=dataset,
                output_dir=output_dir,
                include_moe=include_moe
            )
            for geo in geographies
        ]
        
        results = await tqdm.gather(
            *tasks,
            desc=f"Downloading {table_id}",
            total=len(tasks)
        )
        
        if not combine:
            return list(results)
        
        successful = [r for r in results if r.success and r.file_path]
        if not successful:
            return DownloadResult(success=False, error="All downloads failed")
        
        dfs = [pd.read_csv(r.file_path) for r in successful]
        combined_df = pd.concat(dfs, ignore_index=True)
        
        output_path = Path(output_dir)
        combined_file = output_path / f"{table_id}_combined_{year}.csv"
        combined_df.to_csv(combined_file, index=False)
        
        for result in successful:
            result.file_path.unlink(missing_ok=True)
        
        return DownloadResult(
            success=True,
            file_path=combined_file,
            rows=len(combined_df),
            columns=len(combined_df.columns),
            metadata={
                "table_id": table_id,
                "geographies": len(geographies),
                "year": year
            }
        )
    
    async def download_table_time_series(
        self,
        table_id: str,
        geography: dict[str, str],
        years: list[int],
        dataset: Union[str, ACSDataset] = ACSDataset.ACS5,
        output_dir: Union[str, Path] = ".",
        include_moe: bool = True,
        combine: bool = True
    ) -> Union[DownloadResult, list[DownloadResult]]:
        """
        Download a table across multiple years.
        
        Args:
            table_id: Table ID
            geography: Geography specification
            years: List of years to download
            dataset: ACS dataset type
            output_dir: Directory to save files
            include_moe: Include MOE columns
            combine: Combine into single file
            
        Returns:
            Combined DownloadResult or list of results
        """
        ds = dataset.value if isinstance(dataset, ACSDataset) else dataset
        ds_enum = ACSDataset(ds) if isinstance(ds, str) else dataset
        
        valid_years = DATASET_YEARS.get(ds_enum, range(2009, 2024))
        invalid = [y for y in years if y not in valid_years]
        if invalid:
            logger.warning(f"Years not available for {ds}: {invalid}")
            years = [y for y in years if y in valid_years]
        
        if not years:
            return DownloadResult(success=False, error="No valid years specified")
        
        tasks = [
            self.download_table(
                table_id=table_id,
                geography=geography,
                year=year,
                dataset=dataset,
                output_dir=output_dir,
                include_moe=include_moe
            )
            for year in years
        ]
        
        results = await tqdm.gather(
            *tasks,
            desc=f"Downloading {table_id} time series",
            total=len(tasks)
        )
        
        if not combine:
            return list(results)
        
        successful = [r for r in results if r.success and r.file_path]
        if not successful:
            return DownloadResult(success=False, error="All downloads failed")
        
        dfs = [pd.read_csv(r.file_path) for r in successful]
        combined_df = pd.concat(dfs, ignore_index=True)
        
        output_path = Path(output_dir)
        year_range = f"{min(years)}-{max(years)}"
        combined_file = output_path / f"{table_id}_timeseries_{year_range}.csv"
        combined_df.to_csv(combined_file, index=False)
        
        for result in successful:
            result.file_path.unlink(missing_ok=True)
        
        return DownloadResult(
            success=True,
            file_path=combined_file,
            rows=len(combined_df),
            columns=len(combined_df.columns),
            metadata={
                "table_id": table_id,
                "years": years,
                "geography": geography
            }
        )
    
    async def download_all_states(
        self,
        table_id: str,
        year: int = 2022,
        dataset: Union[str, ACSDataset] = ACSDataset.ACS5,
        output_dir: Union[str, Path] = ".",
        include_moe: bool = True
    ) -> DownloadResult:
        """
        Download a table for all 50 states + DC.
        
        Args:
            table_id: Table ID
            year: ACS year
            dataset: ACS dataset type
            output_dir: Directory to save file
            include_moe: Include MOE columns
            
        Returns:
            DownloadResult with combined data
        """
        return await self.download_table(
            table_id=table_id,
            geography={"state": "*"},
            year=year,
            dataset=dataset,
            output_dir=output_dir,
            include_moe=include_moe,
            filename=f"{table_id}_all_states_{year}.csv"
        )
    
    async def download_all_counties(
        self,
        table_id: str,
        state: Optional[str] = None,
        year: int = 2022,
        dataset: Union[str, ACSDataset] = ACSDataset.ACS5,
        output_dir: Union[str, Path] = ".",
        include_moe: bool = True
    ) -> DownloadResult:
        """
        Download a table for all counties.
        
        Args:
            table_id: Table ID
            state: State to filter by (None for all states)
            year: ACS year
            dataset: ACS dataset type
            output_dir: Directory to save file
            include_moe: Include MOE columns
            
        Returns:
            DownloadResult
        """
        if state:
            geography = {"state": state, "county": "*"}
            suffix = f"_{state}"
        else:
            geography = {"state": "*", "county": "*"}
            suffix = "_all"
        
        return await self.download_table(
            table_id=table_id,
            geography=geography,
            year=year,
            dataset=dataset,
            output_dir=output_dir,
            include_moe=include_moe,
            filename=f"{table_id}_counties{suffix}_{year}.csv"
        )
    
    async def download_all_tracts(
        self,
        table_id: str,
        state: str,
        county: Optional[str] = None,
        year: int = 2022,
        dataset: Union[str, ACSDataset] = ACSDataset.ACS5,
        output_dir: Union[str, Path] = ".",
        include_moe: bool = True
    ) -> DownloadResult:
        """
        Download a table for all census tracts.
        
        Args:
            table_id: Table ID
            state: State (required)
            county: County (optional, None for all counties in state)
            year: ACS year
            dataset: ACS dataset type
            output_dir: Directory to save file
            include_moe: Include MOE columns
            
        Returns:
            DownloadResult
        """
        geography = {"state": state, "county": county or "*", "tract": "*"}
        
        suffix = f"_{state}"
        if county:
            suffix += f"_{county}"
        
        return await self.download_table(
            table_id=table_id,
            geography=geography,
            year=year,
            dataset=dataset,
            output_dir=output_dir,
            include_moe=include_moe,
            filename=f"{table_id}_tracts{suffix}_{year}.csv"
        )
    
    # =========================================================================
    # UTILITY METHODS
    # =========================================================================
    
    async def get_variable_labels(
        self,
        variables: list[str],
        year: int = 2022,
        dataset: Union[str, ACSDataset] = ACSDataset.ACS5
    ) -> dict[str, str]:
        """
        Get human-readable labels for variable names.
        
        Args:
            variables: List of variable names
            year: ACS year
            dataset: ACS dataset type
            
        Returns:
            Dict mapping variable name to label
        """
        all_vars = await self.list_variables(year, dataset)
        var_map = {v.name: v.label for v in all_vars}
        return {v: var_map.get(v, v) for v in variables}
    
    async def validate_table(
        self,
        table_id: str,
        year: int = 2022,
        dataset: Union[str, ACSDataset] = ACSDataset.ACS5
    ) -> bool:
        """Check if a table exists for a given year/dataset."""
        table_info = await self.get_table_info(table_id, year, dataset)
        return table_info is not None
    
    async def validate_variables(
        self,
        variables: list[str],
        year: int = 2022,
        dataset: Union[str, ACSDataset] = ACSDataset.ACS5
    ) -> dict[str, bool]:
        """
        Check which variables exist for a given year/dataset.
        
        Returns:
            Dict mapping variable name to existence boolean
        """
        all_vars = await self.list_variables(year, dataset)
        existing = {v.name for v in all_vars}
        return {v: v in existing for v in variables}
    
    def clear_cache(self) -> None:
        """Clear all cached metadata."""
        self.cache.clear()
        logger.info("Cache cleared")


