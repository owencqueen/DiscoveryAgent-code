"""
Utility functions for ACS Access.

This module provides convenience functions for common operations:
- quick_download: One-liner table download
- search_and_download: Search and download matching tables

These functions create and manage their own ACSClient instances,
making them suitable for simple scripts and one-off operations.
"""

import logging
from pathlib import Path
from typing import Union

from .client import ACSClient
from .models import DownloadResult

logger = logging.getLogger("acs_access")


async def quick_download(
    table_id: str,
    state: str,
    county: str = None,
    place: str = None,
    year: int = 2022,
    output_dir: Union[str, Path] = "."
) -> DownloadResult:
    """
    Quick one-liner to download ACS data.
    
    Creates a temporary client, downloads the specified table,
    and returns the result. Ideal for simple scripts.
    
    Args:
        table_id: Table ID (e.g., "B01001")
        state: State name or abbreviation
        county: Optional county name
        place: Optional place/city name
        year: ACS year (default 2022)
        output_dir: Output directory (default current directory)
        
    Returns:
        DownloadResult with file path and metadata
        
    Example:
        result = await quick_download("B01001", "Michigan", county="Wayne County")
        if result.success:
            print(f"Downloaded to {result.file_path}")
    """
    geography = {"state": state}
    if county:
        geography["county"] = county
    if place:
        geography["place"] = place
    
    async with ACSClient() as client:
        return await client.download_table(
            table_id=table_id,
            geography=geography,
            year=year,
            output_dir=output_dir
        )


async def search_and_download(
    query: str,
    state: str,
    year: int = 2022,
    output_dir: Union[str, Path] = ".",
    max_tables: int = 3
) -> list[DownloadResult]:
    """
    Search for tables and download the top matches.
    
    Searches for tables matching the query string, then downloads
    the top results for the specified state.
    
    Args:
        query: Search query (e.g., "income", "education attainment")
        state: State to download for
        year: ACS year (default 2022)
        output_dir: Output directory
        max_tables: Maximum tables to download (default 3)
        
    Returns:
        List of DownloadResults for each table
        
    Example:
        results = await search_and_download("median income", "California")
        for r in results:
            if r.success:
                print(f"Downloaded {r.metadata['table_id']} to {r.file_path}")
    """
    async with ACSClient() as client:
        # Search for tables
        tables = await client.search_tables(query, year=year, max_results=max_tables)
        
        if not tables:
            logger.warning(f"No tables found for query: {query}")
            return []
        
        # Download each table
        results = []
        for table in tables:
            result = await client.download_table(
                table_id=table.table_id,
                geography={"state": state},
                year=year,
                output_dir=output_dir
            )
            results.append(result)
        
        return results


async def download_common_tables(
    state: str,
    year: int = 2022,
    output_dir: Union[str, Path] = "."
) -> list[DownloadResult]:
    """
    Download commonly-used ACS tables for a state.
    
    Downloads a curated set of frequently-used tables:
    - B01001: Sex by Age (demographics)
    - B01003: Total Population
    - B19013: Median Household Income
    - B19001: Household Income Distribution
    - B15003: Educational Attainment
    - B25001: Housing Units
    - B23025: Employment Status
    
    Args:
        state: State name or abbreviation
        year: ACS year
        output_dir: Output directory
        
    Returns:
        List of DownloadResults
    """
    common_tables = [
        "B01001",  # Sex by Age
        "B01003",  # Total Population
        "B19013",  # Median Household Income
        "B19001",  # Household Income Distribution
        "B15003",  # Educational Attainment
        "B25001",  # Housing Units
        "B23025",  # Employment Status
    ]
    
    async with ACSClient() as client:
        return await client.download_multiple_tables(
            table_ids=common_tables,
            geography={"state": state},
            year=year,
            output_dir=output_dir
        )


async def get_state_summary(
    state: str,
    year: int = 2022
) -> dict:
    """
    Get a quick summary of key metrics for a state.
    
    Downloads key variables and returns a summary dict.
    Does not save to file.
    
    Args:
        state: State name or abbreviation
        year: ACS year
        
    Returns:
        Dict with key metrics:
        - total_population
        - median_household_income
        - housing_units
        - etc.
    """
    key_variables = [
        "B01003_001E",  # Total population
        "B19013_001E",  # Median household income
        "B25001_001E",  # Housing units
        "B23025_002E",  # In labor force
        "B23025_005E",  # Unemployed
        "B15003_022E",  # Bachelor's degree
        "B15003_023E",  # Master's degree
        "B15003_025E",  # Doctorate degree
    ]
    
    async with ACSClient() as client:
        # Get labels for context
        labels = await client.get_variable_labels(key_variables, year)
        
        # Download the variables
        result = await client.download_variables(
            variables=key_variables,
            geography={"state": state},
            year=year,
            output_dir="/tmp"
        )
        
        if not result.success:
            return {"error": result.error}
        
        # Read the CSV and build summary
        import pandas as pd
        df = pd.read_csv(result.file_path)
        
        # Clean up temp file
        result.file_path.unlink(missing_ok=True)
        
        # Build summary
        row = df.iloc[0]
        summary = {
            "state": row.get("NAME", state),
            "year": year,
            "total_population": int(row.get("B01003_001E", 0)),
            "median_household_income": int(row.get("B19013_001E", 0)),
            "housing_units": int(row.get("B25001_001E", 0)),
            "labor_force": int(row.get("B23025_002E", 0)),
            "unemployed": int(row.get("B23025_005E", 0)),
            "bachelors_degree": int(row.get("B15003_022E", 0)),
            "masters_degree": int(row.get("B15003_023E", 0)),
            "doctorate_degree": int(row.get("B15003_025E", 0)),
        }
        
        # Calculate derived metrics
        if summary["labor_force"] > 0:
            summary["unemployment_rate"] = round(
                summary["unemployed"] / summary["labor_force"] * 100, 2
            )
        
        return summary


