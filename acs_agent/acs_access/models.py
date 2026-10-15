"""
Data models for ACS Access.

This module contains dataclasses representing:
- Geography specifications
- Table metadata
- Variable metadata
- Download operation results
"""

from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Any


@dataclass
class GeographySpec:
    """
    Specification for a geographic query.
    
    Represents a resolved geography ready for API queries.
    
    Attributes:
        level: The geography level (state, county, tract, etc.)
        fips_code: The FIPS code for this geography (or "*" for all)
        parent_fips: FIPS codes for parent geographies (e.g., state for county)
        name: Human-readable name for display
        
    Example:
        # Wayne County, Michigan
        GeographySpec(
            level=GeographyLevel.COUNTY,
            fips_code="163",
            parent_fips={"state": "26"},
            name="Wayne County, Michigan"
        )
    """
    level: Any  # GeographyLevel enum
    fips_code: str
    parent_fips: dict = field(default_factory=dict)
    name: Optional[str] = None
    
    def to_api_params(self) -> tuple[str, Optional[str]]:
        """
        Convert to Census API 'for' and 'in' parameters.
        
        Returns:
            Tuple of (for_param, in_param) strings for the API query
            
        Example:
            >>> spec.to_api_params()
            ("county:163", "state:26")
        """
        for_param = f"{self.level.value}:{self.fips_code}"
        
        in_param = None
        if self.parent_fips:
            in_parts = [f"{k}:{v}" for k, v in self.parent_fips.items()]
            in_param = " ".join(in_parts)
        
        return for_param, in_param
    
    def __str__(self) -> str:
        """Human-readable string representation."""
        return self.name or f"{self.level.value}:{self.fips_code}"


@dataclass
class TableInfo:
    """
    Information about an ACS table.
    
    Attributes:
        table_id: Table identifier (e.g., "B01001")
        title: Human-readable table title
        universe: Population the table covers (e.g., "Total population")
        variables: List of VariableInfo objects in this table
        dataset: Dataset this table belongs to
        year: Year of the data
        
    Example:
        TableInfo(
            table_id="B01001",
            title="SEX BY AGE",
            universe="Total population",
            variables=[...],
            dataset="acs/acs5",
            year=2022
        )
    """
    table_id: str
    title: str
    universe: Optional[str] = None
    variables: list = field(default_factory=list)
    dataset: str = "acs/acs5"
    year: int = 2022
    
    @property
    def variable_count(self) -> int:
        """Number of variables in this table."""
        return len(self.variables)
    
    @property
    def estimate_variables(self) -> list:
        """Variables that are estimates (not margins of error)."""
        return [v for v in self.variables if not v.name.endswith("M")]
    
    @property
    def moe_variables(self) -> list:
        """Variables that are margins of error."""
        return [v for v in self.variables if v.name.endswith("M")]
    
    def __str__(self) -> str:
        return f"{self.table_id}: {self.title}"


@dataclass
class VariableInfo:
    """
    Information about an ACS variable.
    
    Attributes:
        name: Variable name (e.g., "B01001_001E")
        label: Human-readable label
        concept: Concept/category the variable belongs to
        group: Table ID this variable belongs to
        predicateType: Data type (int, float, string)
        
    Variable naming convention:
        - Ends with 'E': Estimate
        - Ends with 'M': Margin of error
        - Ends with 'PE': Percent estimate
        - Ends with 'PM': Percent margin of error
    """
    name: str
    label: str
    concept: Optional[str] = None
    group: Optional[str] = None
    predicateType: Optional[str] = None
    
    @property
    def is_estimate(self) -> bool:
        """Whether this is an estimate (vs margin of error)."""
        return self.name.endswith("E") or self.name.endswith("PE")
    
    @property
    def is_margin_of_error(self) -> bool:
        """Whether this is a margin of error."""
        return self.name.endswith("M") or self.name.endswith("PM")
    
    @property
    def is_percent(self) -> bool:
        """Whether this is a percentage variable."""
        return "PE" in self.name or "PM" in self.name
    
    @property
    def table_id(self) -> Optional[str]:
        """Extract table ID from variable name."""
        if "_" in self.name:
            return self.name.rsplit("_", 1)[0]
        return self.group
    
    def __str__(self) -> str:
        return f"{self.name}: {self.label}"


@dataclass 
class DownloadResult:
    """
    Result of a data download operation.
    
    Attributes:
        success: Whether the download succeeded
        file_path: Path to the downloaded CSV file
        rows: Number of rows in the data
        columns: Number of columns in the data
        error: Error message if download failed
        metadata: Additional metadata about the download
        
    Example:
        DownloadResult(
            success=True,
            file_path=Path("./data/B01001_Michigan_2022.csv"),
            rows=1,
            columns=52,
            metadata={"table_id": "B01001", "geography": "Michigan"}
        )
    """
    success: bool
    file_path: Optional[Path] = None
    rows: int = 0
    columns: int = 0
    error: Optional[str] = None
    metadata: dict = field(default_factory=dict)
    
    def __bool__(self) -> bool:
        """Allow using result in boolean context."""
        return self.success
    
    def __str__(self) -> str:
        if self.success:
            return f"✓ Downloaded {self.rows} rows to {self.file_path}"
        return f"✗ Failed: {self.error}"
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "success": self.success,
            "file_path": str(self.file_path) if self.file_path else None,
            "rows": self.rows,
            "columns": self.columns,
            "error": self.error,
            "metadata": self.metadata,
        }


@dataclass
class BatchDownloadResult:
    """
    Result of a batch download operation.
    
    Attributes:
        total: Total number of downloads attempted
        successful: Number of successful downloads
        failed: Number of failed downloads
        results: List of individual DownloadResult objects
        combined_file: Path to combined output file (if applicable)
    """
    total: int
    successful: int
    failed: int
    results: list[DownloadResult] = field(default_factory=list)
    combined_file: Optional[Path] = None
    
    @classmethod
    def from_results(cls, results: list[DownloadResult], combined_file: Optional[Path] = None) -> "BatchDownloadResult":
        """Create from a list of individual results."""
        successful = sum(1 for r in results if r.success)
        return cls(
            total=len(results),
            successful=successful,
            failed=len(results) - successful,
            results=results,
            combined_file=combined_file,
        )
    
    def __bool__(self) -> bool:
        """True if all downloads succeeded."""
        return self.failed == 0
    
    def __str__(self) -> str:
        status = "✓" if self.failed == 0 else "⚠"
        return f"{status} {self.successful}/{self.total} downloads successful"


