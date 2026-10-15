"""
ACS Agent - AI-powered American Community Survey data assistant.

Uses Claude claude-haiku-4-5-20251001 with tool use to efficiently query and download ACS data.
Unlike traditional multi-step LLM pipelines, this agent uses a single conversation
with tool calling for maximum efficiency.

Usage:
    # Interactive mode
    python -m acs_agent

    # Single query mode
    python -m acs_agent --query "What is the population of Michigan?"

    # With custom output directory
    python -m acs_agent --output ./data

Environment Variables:
    ANTHROPIC_API_KEY: Required - Your Anthropic API key
    CENSUS_API_KEY: Optional - Census API key for higher rate limits
    ACS_CACHE_PATH: Optional - Path to cache directory
"""

import os
import json
import asyncio
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field

import anthropic
from dotenv import load_dotenv
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel


# =============================================================================
# COST TRACKING
# =============================================================================

# Pricing per million tokens, keyed by model ID
MODEL_PRICING = {
    "claude-opus-4-6": {"prompt": 5.0, "completion": 25.0},
    "claude-haiku-4-5-20251001": {"prompt": 1.0, "completion": 5.0},
    "claude-sonnet-4-5-20250929": {"prompt": 3.0, "completion": 15.0},
    "claude-sonnet-4-6": {"prompt": 3.0, "completion": 15.0},
}


@dataclass
class UsageTracker:
    """Tracks token usage and calculates cost for a specific model."""
    model: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    api_calls: int = 0

    def _pricing(self) -> dict:
        """Return pricing dict for the current model, falling back to zeros."""
        return MODEL_PRICING.get(self.model, {"prompt": 0.0, "completion": 0.0})

    def add_usage(self, input_tokens: int, output_tokens: int):
        """Add usage from an API call."""
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens
        self.api_calls += 1

    @property
    def input_cost(self) -> float:
        """Cost for input tokens in dollars."""
        return (self.input_tokens / 1_000_000) * self._pricing()["prompt"]

    @property
    def output_cost(self) -> float:
        """Cost for output tokens in dollars."""
        return (self.output_tokens / 1_000_000) * self._pricing()["completion"]

    @property
    def total_cost(self) -> float:
        """Total cost in dollars."""
        return self.input_cost + self.output_cost

    def reset(self):
        """Reset all counters."""
        self.input_tokens = 0
        self.output_tokens = 0
        self.api_calls = 0

    def summary(self) -> str:
        """Return a formatted summary string."""
        return (
            f"API Calls: {self.api_calls} | "
            f"Model: {self.model} | "
            f"Tokens: {self.input_tokens:,} in / {self.output_tokens:,} out | "
            f"Cost: ${self.total_cost:.4f} "
            f"(${self.input_cost:.4f} in + ${self.output_cost:.4f} out)"
        )

    def detailed_summary(self) -> dict:
        """Return detailed usage statistics."""
        return {
            "model": self.model,
            "api_calls": self.api_calls,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.input_tokens + self.output_tokens,
            "input_cost_usd": round(self.input_cost, 6),
            "output_cost_usd": round(self.output_cost, 6),
            "total_cost_usd": round(self.total_cost, 6),
        }

# Import the acs_access package (local)
from .acs_access import ACSClient
from .pums.client import PUMSClient
from .pums.tools import PUMS_TOOLS
from .ipums.client import IpumsClient
from .ipums.tools import IPUMS_TOOLS

load_dotenv()

console = Console()

# =============================================================================
# CONFIGURATION
# =============================================================================

MODEL = "claude-opus-4-6"
MAX_TOKENS = 4096
MAX_TOOL_ROUNDS = 25          # safety cap on tool-call iterations per chat()
COMPACTION_BETA = "compact-2026-01-12"
COMPACTION_TRIGGER_TOKENS = 150_000  # compact when input exceeds this many tokens

# System prompt for the agent
SYSTEM_PROMPT = """You are an expert data analyst assistant specializing in U.S. Census American Community Survey (ACS) data.

Your role is to help users find, understand, and download ACS data. You have access to tools that can:
- Search for relevant ACS tables by topic
- Get detailed information about specific tables
- Download complete tables or specific variables
- Resolve geographic locations (state names, counties, cities) to proper identifiers
- Retrieve PUMS microdata records and lightweight tabulations via the Census Microdata API

## Key Knowledge

**ACS Datasets:**
- ACS 5-year estimates (acs5): Most comprehensive, available for all geographies, 2009-2024
- ACS 1-year estimates (acs1): More current but only for areas with 65,000+ population, 2005-2024
- Subject tables, data profiles, and comparison profiles are also available

**Common Table Prefixes:**
- B01xxx: Age, Sex, Population
- B02xxx: Race
- B03xxx: Hispanic Origin
- B05xxx: Citizenship, Place of Birth
- B06xxx: Place of Birth (detailed)
- B07xxx: Geographic Mobility
- B08xxx: Commuting/Transportation
- B09xxx: Children characteristics
- B11xxx: Household/Family
- B12xxx: Marital Status
- B15xxx: Education
- B17xxx: Poverty
- B19xxx: Income
- B23xxx: Employment Status
- B25xxx: Housing

**Geography Hierarchy:**
- Nation → Region → Division → State → County → Tract → Block Group
- Places (cities) are within states
- Use "*" for all of a geography type (e.g., all states, all counties in a state)

**PUMS (Public Use Microdata Sample) via Census API:**
- Record types: person (P) and housing (H)
- Supported geographies: nation, region, division, state, PUMA
- PUMS does not support tracts, block groups, or ZIP codes
- Always include a small geography or restrictive filters, and enforce row limits

**IPUMS Data (via ipumspy) – Extended Microdata Access:**

IPUMS provides harmonized individual-level microdata with consistent variable coding across decades. Use IPUMS tools when you need: historical census data (pre-2005), CPS labor force data, or health surveys (NHIS, MEPS). For current ACS summary tables (2009+), always prefer the direct Census API tools — they return instantly.

Available collections: usa, cps, nhis, meps

### IPUMS Collections — When to Use Each

**IPUMS USA (`usa`)** — U.S. Census + ACS microdata, 1850-present.
Use for: historical census analysis, harmonized variables across decades, full-count historical data.
Geography: State, county, metro, PUMA (varies by sample).
Why over Census PUMS API: historical data back to 1850, harmonized variables, linked records.

**IPUMS CPS (`cps`)** — Current Population Survey, 1962-present.
Use for: employment, unemployment, earnings, labor force, industry/occupation.
The March ASEC supplement has the most detailed income/poverty data.
Geography: State, metro area.

**IPUMS NHIS (`nhis`)** — National Health Interview Survey, 1963-present.
Use for: health status, chronic conditions, healthcare access, insurance, disability, mental health.
Geography: Region only (limited detail for confidentiality).
NHIS has its own variable names — they differ from IPUMS USA variables.

**IPUMS MEPS (`meps`)** — Medical Expenditure Panel Survey, 1996-present.
Use for: healthcare expenditures, insurance, medical conditions, utilization, out-of-pocket costs.
Geography: Region only.

### IPUMS Sample ID Formats

Sample IDs are required for ipums_get_microdata. Use ipums_search_samples to discover them, or use these known patterns:

- USA: `us{year}{variant}` — e.g., `us2022a` (ACS 1-year), `us2022b` (ACS 5-year), `us1940a` (1940 Census)
- CPS: `cps{year}_{month}s` — e.g., `cps2023_03s` (March 2023 ASEC), `cps2023_01b` (Jan basic)
- MEPS: Discover via ipums_search_samples (e.g., `mp2022`)
- NHIS: Always use ipums_search_samples first — naming varies.

Common USA samples: us2024a, us2023a, us2022a, us2021a (ACS 1-year); us2024b, us2023b (ACS 5-year); us2010a, us2000a, us1990a, us1980a, us1970a, us1960a, us1950a, us1940a (decennial census).
Common CPS samples: cps2024_03s, cps2023_03s, cps2022_03s (March ASEC supplements).

### IPUMS Variable Names — CRITICAL REFERENCE

IPUMS uses its own harmonized variable names — these are NOT Census Bureau codes. Never use Census codes (B01001_001E, AGEP) with IPUMS tools.

Demographics: AGE (age), SEX (sex), RACE/RACED (race general/detailed), HISPAN/HISPAND (Hispanic origin), BPL (birthplace), CITIZEN (citizenship), MARST (marital status)
Education: EDUC/EDUCD (educational attainment general/detailed), SCHOOL (school attendance)
Employment: EMPSTAT/EMPSTATD (employment status), OCC (occupation), IND (industry), CLASSWKR (class of worker), WKSWORK2 (weeks worked), UHRSWORK (hours/week)
Income: INCTOT (total personal income), INCWAGE (wage income), INCEARN (earned income), POVERTY (% of poverty line), HHINCOME (household income)
Housing: OWNERSHP (own/rent), RENT (contract rent), VALUEH (house value), ROOMS, BEDROOMS, MORTGAGE
Geography: STATEFIP (state FIPS), COUNTYFIP (county FIPS), MET2013 (metro area), PUMA
Weights — ALWAYS include: PERWT (person weight for USA/CPS/NHIS/MEPS), HHWT (household weight)

### IPUMS Extract Workflow

1. If the user specifies a collection, skip to step 2. Otherwise call ipums_list_collections.
2. Call ipums_search_samples to discover sample IDs for the collection (unless you already know the ID from the patterns above).
3. Call ipums_get_microdata with: collection, samples, variables, data_format="csv".
4. The extract takes 2-10 minutes. This is normal — wait for the result.
5. The tool returns a data preview. Present the results to the user.

### IPUMS Error Handling Rules

- If a tool returns `do_not_retry: true`, do NOT call the same tool again. Report the error to the user.
- If an extract succeeds but parsing fails, the data files are on disk. Use ipums_read_extract_data with the file paths from the error response.
- If the API says a variable is "not found" or "invalid", do NOT retry with the same variable. Check the variable name against the reference above.
- Registration errors mean the user needs to register at https://uma.pop.umn.edu/{collection}/registration/new — do not retry.

### IPUMS Performance Tips

- Keep extracts small: 1-2 samples, 5-15 variables. Large multi-decade requests can take 30+ minutes.
- Always use data_format="csv" — it's faster and avoids fixed-width parsing issues.
- Use case_selections to filter records (e.g., {"SEX": ["2"]} for females only) to reduce extract size.

## Guidelines

1. When users ask about data, first search for relevant tables to find the right one
2. For current ACS data (2009+), use the direct Census API tools; for historical, health, or labor microdata, use IPUMS tools
3. Default to 5-year estimates (acs5) and most recent year (2024) unless specified
4. Present results in a clear, organized way
5. When an IPUMS extract is processing, inform the user it may take a few minutes
6. Always include appropriate weight variables (PERWT, HHWT) in IPUMS extracts
7. Always include STATEFIP in USA extracts so results can be disaggregated by state

Always be helpful and accurate."""

AUTONOMOUS_PROMPT = """
CRITICAL: You are operating autonomously. You MUST follow these rules:

## General Rules
1. NEVER ask the user clarifying questions. NEVER ask "Would you like me to...?" or "Should I...?" or "Which option do you prefer?" — just do it.
2. When the request is ambiguous, make the best reasonable assumption and proceed. State your assumption briefly, then act on it.
3. Always call tools immediately to fulfill the request. Do not explain what you could do — just do it.
4. Report your actions and results after completing them, not before.
5. If there are errors or important caveats in the data, note them in your response alongside the results.

## IPUMS-Specific Rules
6. For IPUMS requests: identify the right collection, look up sample IDs if needed, then immediately call ipums_get_microdata. Do not ask which variables or samples the user wants — infer them from the request.
7. ALWAYS include a weight variable (PERWT) in every IPUMS extract. Never submit an extract without a weight.
8. ALWAYS include STATEFIP in USA extracts for geographic disaggregation.
9. ALWAYS use data_format="csv" for all IPUMS extracts. Never use fixed_width.
10. If a tool returns "do_not_retry": true, STOP. Do not call the same tool again with the same or similar parameters. Report the error to the user.
12. If an IPUMS extract fails with a variable-not-found error, do NOT retry with guessed variable names. Instead, inform the user which variables failed and suggest they check the IPUMS website for collection-specific variable names.
13. Never submit more than 2 IPUMS extracts for a single user request. If the first extract fails, try once more with corrected parameters. If that also fails, report the issue.
14. When choosing between Census API and IPUMS: current data (2009+) → Census API. Historical, health, or labor data → IPUMS.
"""


# =============================================================================
# TOOL DEFINITIONS
# =============================================================================

ACS_TOOLS = [
    {
        "name": "search_tables",
        "description": "Search for ACS tables by topic or keyword. Use this to find relevant tables when the user asks about a topic (e.g., 'income', 'education', 'housing'). Returns table IDs and descriptions.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query - topic or keywords to search for (e.g., 'median household income', 'educational attainment', 'poverty')"
                },
                "year": {
                    "type": "integer",
                    "description": "ACS year (2009-2024 for 5-year, 2005-2024 for 1-year). Default: 2024",
                    "default": 2024
                },
                "dataset": {
                    "type": "string",
                    "enum": ["acs/acs5", "acs/acs1", "acs/acs5/subject", "acs/acs5/profile"],
                    "description": "ACS dataset type. Default: acs/acs5 (5-year estimates)",
                    "default": "acs/acs5"
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of tables to return. Default: 10",
                    "default": 10
                }
            },
            "required": ["query"]
        }
    },
    {
        "name": "get_table_info",
        "description": "Get detailed information about a specific ACS table, including all its variables and their descriptions. Use this to understand what data a table contains before downloading.",
        "input_schema": {
            "type": "object",
            "properties": {
                "table_id": {
                    "type": "string",
                    "description": "The table ID (e.g., 'B01001', 'B19013', 'S1501')"
                },
                "year": {
                    "type": "integer",
                    "description": "ACS year. Default: 2024",
                    "default": 2024
                },
                "dataset": {
                    "type": "string",
                    "enum": ["acs/acs5", "acs/acs1", "acs/acs5/subject", "acs/acs5/profile"],
                    "description": "ACS dataset type. Default: acs/acs5",
                    "default": "acs/acs5"
                }
            },
            "required": ["table_id"]
        }
    },
    {
        "name": "download_table",
        "description": "Download a complete ACS table for a specified geography. Saves the data as a CSV file. Use this when the user wants to get actual data for a table.",
        "input_schema": {
            "type": "object",
            "properties": {
                "table_id": {
                    "type": "string",
                    "description": "The table ID to download (e.g., 'B01001', 'B19013')"
                },
                "state": {
                    "type": "string",
                    "description": "State name, abbreviation, or FIPS code (e.g., 'Michigan', 'MI', '26'). Use '*' for all states."
                },
                "county": {
                    "type": "string",
                    "description": "Optional: County name (e.g., 'Wayne County'). Use '*' for all counties in the state."
                },
                "place": {
                    "type": "string",
                    "description": "Optional: City/place name (e.g., 'Detroit', 'Los Angeles city')"
                },
                "year": {
                    "type": "integer",
                    "description": "ACS year. Default: 2024",
                    "default": 2024
                },
                "dataset": {
                    "type": "string",
                    "enum": ["acs/acs5", "acs/acs1"],
                    "description": "ACS dataset type. Default: acs/acs5",
                    "default": "acs/acs5"
                },
                "include_moe": {
                    "type": "boolean",
                    "description": "Include margin of error columns. Default: true",
                    "default": True
                }
            },
            "required": ["table_id", "state"]
        }
    },
    {
        "name": "download_variables",
        "description": "Download specific ACS variables (not a full table) for a geography. Use this when the user only needs specific data points rather than a complete table.",
        "input_schema": {
            "type": "object",
            "properties": {
                "variables": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of variable names to download (e.g., ['B01003_001E', 'B19013_001E'])"
                },
                "state": {
                    "type": "string",
                    "description": "State name, abbreviation, or FIPS code"
                },
                "county": {
                    "type": "string",
                    "description": "Optional: County name"
                },
                "place": {
                    "type": "string",
                    "description": "Optional: City/place name"
                },
                "year": {
                    "type": "integer",
                    "description": "ACS year. Default: 2024",
                    "default": 2024
                }
            },
            "required": ["variables", "state"]
        }
    },
    {
        "name": "search_variables",
        "description": "Search for specific ACS variables by keyword. Use this to find exact variable codes when you need specific data points.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query for variables (e.g., 'total population', 'bachelor degree')"
                },
                "year": {
                    "type": "integer",
                    "description": "ACS year. Default: 2024",
                    "default": 2024
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum results. Default: 20",
                    "default": 20
                }
            },
            "required": ["query"]
        }
    },
    {
        "name": "get_quick_stats",
        "description": "Get quick summary statistics for a state (population, income, etc.) without downloading a full table. Good for answering simple questions quickly.",
        "input_schema": {
            "type": "object",
            "properties": {
                "state": {
                    "type": "string",
                    "description": "State name or abbreviation"
                },
                "year": {
                    "type": "integer",
                    "description": "ACS year. Default: 2024",
                    "default": 2024
                }
            },
            "required": ["state"]
        }
    },
    {
        "name": "compare_geographies",
        "description": "Compare key statistics between multiple states or geographies. Useful for comparative questions.",
        "input_schema": {
            "type": "object",
            "properties": {
                "states": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of state names or abbreviations to compare"
                },
                "metrics": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "enum": ["population", "median_income", "poverty_rate", "unemployment_rate", "bachelors_degree_rate"]
                    },
                    "description": "Metrics to compare. Default: all available",
                    "default": ["population", "median_income"]
                },
                "year": {
                    "type": "integer",
                    "description": "ACS year. Default: 2024",
                    "default": 2024
                }
            },
            "required": ["states"]
        }
    }
]

TOOLS = ACS_TOOLS + PUMS_TOOLS + IPUMS_TOOLS


# =============================================================================
# TOOL IMPLEMENTATIONS
# =============================================================================

class ACSToolExecutor:
    """Executes ACS tools and returns results."""

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.client: Optional[ACSClient] = None
        self.pums_client: Optional[PUMSClient] = None
        self.ipums_client: Optional[IpumsClient] = None

    async def __aenter__(self):
        self.client = ACSClient()
        await self.client._ensure_session()
        self.pums_client = PUMSClient()
        await self.pums_client._ensure_session()
        self.ipums_client = IpumsClient(self.output_dir)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.client:
            await self.client.close()
        if self.pums_client:
            await self.pums_client.close()

    async def execute(self, tool_name: str, tool_input: dict) -> dict:
        """Execute a tool and return the result."""
        try:
            if tool_name == "search_tables":
                return await self._search_tables(**tool_input)
            elif tool_name == "get_table_info":
                return await self._get_table_info(**tool_input)
            elif tool_name == "download_table":
                return await self._download_table(**tool_input)
            elif tool_name == "download_variables":
                return await self._download_variables(**tool_input)
            elif tool_name == "search_variables":
                return await self._search_variables(**tool_input)
            elif tool_name == "get_quick_stats":
                return await self._get_quick_stats(**tool_input)
            elif tool_name == "compare_geographies":
                return await self._compare_geographies(**tool_input)
            elif tool_name == "pums_search_variables":
                return await self._pums_search_variables(**tool_input)
            elif tool_name == "pums_get_records":
                return await self._pums_get_records(**tool_input)
            elif tool_name == "pums_tabulate":
                return await self._pums_tabulate(**tool_input)
            # IPUMS tools
            elif tool_name == "ipums_list_collections":
                return await self._ipums_list_collections(**tool_input)
            elif tool_name == "ipums_search_samples":
                return await self._ipums_search_samples(**tool_input)
            elif tool_name == "ipums_read_extract_data":
                return await self._ipums_read_extract_data(**tool_input)
            elif tool_name == "ipums_check_extract_status":
                return await self._ipums_check_extract_status(**tool_input)
            elif tool_name == "ipums_get_microdata":
                return await self._ipums_get_microdata(**tool_input)
            else:
                return {"error": f"Unknown tool: {tool_name}"}
        except Exception as e:
            return {"error": str(e)}

    async def _search_tables(
        self,
        query: str,
        year: int = 2024,
        dataset: str = "acs/acs5",
        max_results: int = 10
    ) -> dict:
        """Search for tables by topic."""
        tables = await self.client.search_tables(
            query=query,
            year=year,
            dataset=dataset,
            max_results=max_results
        )

        return {
            "query": query,
            "year": year,
            "dataset": dataset,
            "count": len(tables),
            "tables": [
                {
                    "table_id": t.table_id,
                    "title": t.title,
                    "universe": t.universe
                }
                for t in tables
            ]
        }

    async def _get_table_info(
        self,
        table_id: str,
        year: int = 2024,
        dataset: str = "acs/acs5"
    ) -> dict:
        """Get detailed table information."""
        info = await self.client.get_table_info(
            table_id=table_id,
            year=year,
            dataset=dataset
        )

        if not info:
            return {"error": f"Table {table_id} not found"}

        # Group variables by type
        estimates = [v for v in info.variables if v.name.endswith("E")]

        return {
            "table_id": info.table_id,
            "title": info.title,
            "universe": info.universe,
            "year": year,
            "dataset": dataset,
            "total_variables": len(info.variables),
            "estimate_variables": len(estimates),
            "variables": [
                {
                    "name": v.name,
                    "label": v.label
                }
                for v in estimates[:30]  # Limit to first 30 estimates
            ],
            "truncated": len(estimates) > 30
        }

    async def _download_table(
        self,
        table_id: str,
        state: str,
        county: str = None,
        place: str = None,
        year: int = 2024,
        dataset: str = "acs/acs5",
        include_moe: bool = True
    ) -> dict:
        """Download a complete table."""
        geography = {"state": state}
        if county:
            geography["county"] = county
        if place:
            geography["place"] = place

        result = await self.client.download_table(
            table_id=table_id,
            geography=geography,
            year=year,
            dataset=dataset,
            output_dir=self.output_dir,
            include_moe=include_moe
        )

        if result.success:
            # Read first few rows to show preview
            import pandas as pd
            df = pd.read_csv(result.file_path)
            preview = df.head(5).to_dict(orient='records')

            return {
                "success": True,
                "file_path": str(result.file_path),
                "rows": result.rows,
                "columns": result.columns,
                "table_id": table_id,
                "geography": result.metadata.get("geography", ""),
                "preview": preview
            }
        else:
            return {
                "success": False,
                "error": result.error
            }

    async def _download_variables(
        self,
        variables: list[str],
        state: str,
        county: str = None,
        place: str = None,
        year: int = 2024
    ) -> dict:
        """Download specific variables."""
        geography = {"state": state}
        if county:
            geography["county"] = county
        if place:
            geography["place"] = place

        result = await self.client.download_variables(
            variables=variables,
            geography=geography,
            year=year,
            output_dir=self.output_dir
        )

        if result.success:
            import pandas as pd
            df = pd.read_csv(result.file_path)
            data = df.to_dict(orient='records')

            return {
                "success": True,
                "file_path": str(result.file_path),
                "data": data
            }
        else:
            return {
                "success": False,
                "error": result.error
            }

    async def _search_variables(
        self,
        query: str,
        year: int = 2024,
        max_results: int = 20
    ) -> dict:
        """Search for specific variables."""
        variables = await self.client.search_variables(
            query=query,
            year=year,
            max_results=max_results
        )

        return {
            "query": query,
            "count": len(variables),
            "variables": [
                {
                    "name": v.name,
                    "label": v.label,
                    "concept": v.concept,
                    "table_id": v.table_id
                }
                for v in variables
            ]
        }

    async def _get_quick_stats(
        self,
        state: str,
        year: int = 2024
    ) -> dict:
        """Get quick statistics for a state."""
        # Key variables for quick stats
        variables = [
            "B01003_001E",  # Total population
            "B19013_001E",  # Median household income
            "B25001_001E",  # Housing units
            "B23025_002E",  # Labor force
            "B23025_005E",  # Unemployed
            "B17001_002E",  # Below poverty level
            "B17001_001E",  # Poverty universe (total)
            "B15003_022E",  # Bachelor's degree
            "B15003_023E",  # Master's degree
            "B15003_025E",  # Doctorate
            "B15003_001E",  # Education universe (25+)
        ]

        result = await self.client.download_variables(
            variables=variables,
            geography={"state": state},
            year=year,
            output_dir=self.output_dir
        )

        if not result.success:
            return {"error": result.error}

        import pandas as pd
        df = pd.read_csv(result.file_path)
        row = df.iloc[0]

        # Calculate derived metrics
        pop = int(row.get("B01003_001E", 0))
        income = int(row.get("B19013_001E", 0))
        housing = int(row.get("B25001_001E", 0))
        labor_force = int(row.get("B23025_002E", 0))
        unemployed = int(row.get("B23025_005E", 0))
        below_poverty = int(row.get("B17001_002E", 0))
        poverty_universe = int(row.get("B17001_001E", 0))
        bachelors = int(row.get("B15003_022E", 0))
        masters = int(row.get("B15003_023E", 0))
        doctorate = int(row.get("B15003_025E", 0))
        edu_universe = int(row.get("B15003_001E", 0))

        stats = {
            "state": row.get("NAME", state),
            "year": year,
            "total_population": pop,
            "median_household_income": income,
            "housing_units": housing,
            "labor_force": labor_force,
        }

        if labor_force > 0:
            stats["unemployment_rate"] = round(unemployed / labor_force * 100, 1)

        if poverty_universe > 0:
            stats["poverty_rate"] = round(below_poverty / poverty_universe * 100, 1)

        if edu_universe > 0:
            higher_ed = bachelors + masters + doctorate
            stats["bachelors_or_higher_rate"] = round(higher_ed / edu_universe * 100, 1)

        # Clean up temp file
        Path(result.file_path).unlink(missing_ok=True)

        return stats

    async def _compare_geographies(
        self,
        states: list[str],
        metrics: list[str] = None,
        year: int = 2024
    ) -> dict:
        """Compare statistics across multiple states."""
        if metrics is None:
            metrics = ["population", "median_income", "poverty_rate", "unemployment_rate"]

        results = []
        for state in states:
            stats = await self._get_quick_stats(state, year)
            if "error" not in stats:
                results.append(stats)

        if not results:
            return {"error": "Could not retrieve data for any of the specified states"}

        # Build comparison
        comparison = {
            "year": year,
            "states_compared": len(results),
            "data": results
        }

        return comparison

    async def _pums_search_variables(
        self,
        query: str,
        year: int = 2024,
        record_type: str = "person",
        sample: str = "acs5",
    ) -> dict:
        """Search PUMS variables by keyword."""
        variables = await self.pums_client.search_variables(
            query=query,
            year=year,
            record_type=record_type,
            sample=sample,
        )
        return {
            "query": query,
            "year": year,
            "record_type": record_type,
            "sample": sample,
            "count": len(variables),
            "variables": [
                {
                    "name": v.name,
                    "label": v.label,
                    "concept": v.concept,
                    "record_type": v.record_type,
                }
                for v in variables[:50]
            ],
            "truncated": len(variables) > 50,
        }

    async def _pums_get_records(
        self,
        year: int,
        record_type: str,
        variables: list[str],
        filters: dict,
        geography: dict,
        limit: int = 5000,
        sample: str = "acs5",
    ) -> dict:
        """Retrieve bounded PUMS records."""
        records = await self.pums_client.get_records(
            year=year,
            record_type=record_type,
            variables=variables,
            filters=filters,
            geography=geography,
            limit=limit,
            sample=sample,
        )
        return {
            "year": year,
            "record_type": record_type,
            "sample": sample,
            "count": len(records),
            "records": records,
            "truncated": len(records) >= limit,
        }

    async def _pums_tabulate(
        self,
        year: int,
        record_type: str,
        metrics: list[str],
        group_by: list[str],
        filters: dict,
        geography: dict,
        weight: str | None = None,
        sample: str = "acs5",
    ) -> dict:
        """Tabulate PUMS data with lightweight metrics."""
        rows = await self.pums_client.tabulate(
            year=year,
            record_type=record_type,
            metrics=metrics,
            group_by=group_by,
            filters=filters,
            geography=geography,
            weight=weight,
            sample=sample,
        )
        return {
            "year": year,
            "record_type": record_type,
            "sample": sample,
            "rows": rows,
            "count": len(rows),
        }

    # ------------------------------------------------------------------ #
    # IPUMS tool implementations
    # ------------------------------------------------------------------ #

    async def _ipums_list_collections(
        self,
        collection_type: str = "all",
    ) -> dict:
        """List available IPUMS collections."""
        return await self.ipums_client.list_collections(collection_type)

    async def _ipums_search_samples(
        self,
        collection: str,
    ) -> dict:
        """List samples for an IPUMS microdata collection."""
        return await self.ipums_client.search_samples(collection)

    async def _ipums_read_extract_data(
        self,
        ddi_file: str,
        data_file: str | None = None,
        subset: list[str] | None = None,
        head: int = 20,
    ) -> dict:
        """Read a previously downloaded IPUMS extract."""
        return await self.ipums_client.read_extract_data(
            ddi_file, data_file, subset, head,
        )

    async def _ipums_check_extract_status(
        self,
        collection: str,
        extract_id: int,
    ) -> dict:
        """Check status of a previously submitted extract."""
        return await self.ipums_client.check_extract_status(collection, extract_id)

    async def _ipums_get_microdata(
        self,
        collection: str,
        samples: list[str],
        variables: list[str],
        description: str | None = None,
        data_format: str = "csv",
        data_structure: str = "rectangular_on_P",
        case_selections: dict[str, list[str]] | None = None,
    ) -> dict:
        """Request and download IPUMS microdata."""
        return await self.ipums_client.get_microdata(
            collection, samples, variables, description,
            data_format, data_structure, case_selections,
        )


# =============================================================================
# AGENT CLASS
# =============================================================================

class ACSAgent:
    """
    AI-powered ACS data assistant using Claude with tool use.
    """

    def __init__(
        self,
        output_dir: Path = None,
        autonomous: bool = True,
        compaction: bool = True,
        compaction_trigger_tokens: int = COMPACTION_TRIGGER_TOKENS,
    ):
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable is required")

        self.anthropic = anthropic.Anthropic(api_key=api_key)
        self.output_dir = output_dir or Path("./acs_downloads")
        self.conversation_history = []
        self.system_prompt = SYSTEM_PROMPT + (AUTONOMOUS_PROMPT if autonomous else "")
        self.usage = UsageTracker(model=MODEL)
        self.compaction = compaction
        self.compaction_trigger_tokens = compaction_trigger_tokens

    async def chat(self, user_message: str) -> str:
        """
        Process a user message and return the assistant's response.
        Handles tool calls automatically with context compaction.
        """
        # Add user message to history
        self.conversation_history.append({
            "role": "user",
            "content": user_message
        })

        # Build kwargs shared across every call in this loop
        api_kwargs: dict = dict(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            system=self.system_prompt,
            tools=TOOLS,
            messages=self.conversation_history,
        )
        if self.compaction:
            api_kwargs["betas"] = [COMPACTION_BETA]
            api_kwargs["context_management"] = {
                "edits": [
                    {
                        "type": "compact_20260112",
                        "trigger": {
                            "type": "input_tokens",
                            "value": self.compaction_trigger_tokens,
                        },
                    }
                ]
            }

        async with ACSToolExecutor(self.output_dir) as executor:
            rounds = 0
            while rounds < MAX_TOOL_ROUNDS:
                rounds += 1

                # Call Claude (via beta endpoint when compaction is on)
                if self.compaction:
                    response = self.anthropic.beta.messages.create(**api_kwargs)
                else:
                    response = self.anthropic.messages.create(**api_kwargs)

                # Track usage
                self.usage.add_usage(
                    input_tokens=response.usage.input_tokens,
                    output_tokens=response.usage.output_tokens
                )

                # ----- handle compaction stop reason --------------------
                if response.stop_reason == "compaction":
                    # The response content contains a compaction block
                    # summarising earlier context. Keep it in history so the
                    # API can drop the pre-compaction messages.
                    self.conversation_history.append({
                        "role": "assistant",
                        "content": response.content,
                    })
                    console.print("[dim]📦 Context compacted[/dim]")
                    continue

                # ----- handle tool use ----------------------------------
                if response.stop_reason == "tool_use":
                    tool_results = []
                    assistant_content = response.content

                    for block in response.content:
                        if block.type == "tool_use":
                            tool_name = block.name
                            tool_input = block.input
                            tool_id = block.id

                            console.print(f"[dim]🔧 Calling {tool_name}...[/dim]")

                            result = await executor.execute(tool_name, tool_input)

                            tool_results.append({
                                "type": "tool_result",
                                "tool_use_id": tool_id,
                                "content": json.dumps(result, default=str)
                            })

                    self.conversation_history.append({
                        "role": "assistant",
                        "content": assistant_content
                    })
                    self.conversation_history.append({
                        "role": "user",
                        "content": tool_results
                    })
                    continue

                # ----- handle end_turn (final answer) -------------------
                if response.stop_reason == "end_turn":
                    text_response = ""
                    for block in response.content:
                        if hasattr(block, "text"):
                            text_response += block.text

                    self.conversation_history.append({
                        "role": "assistant",
                        "content": response.content
                    })
                    return text_response

                # ----- unexpected stop reason ---------------------------
                return f"Unexpected response: {response.stop_reason}"

            # Exhausted MAX_TOOL_ROUNDS
            return (
                "I was unable to complete the request within the maximum number "
                f"of tool-calling rounds ({MAX_TOOL_ROUNDS}). "
                "Please try narrowing your request."
            )

    def reset_conversation(self):
        """Clear conversation history."""
        self.conversation_history = []

    def get_usage_summary(self) -> str:
        """Get a formatted usage summary."""
        return self.usage.summary()

    def get_usage_details(self) -> dict:
        """Get detailed usage statistics."""
        return self.usage.detailed_summary()

    def reset_usage(self):
        """Reset usage tracking."""
        self.usage.reset()

    async def run_interactive(self):
        """Run an interactive chat session."""
        console.print(Panel(
            "[bold blue]ACS Data Assistant[/bold blue]\n\n"
            "I can help you find and download American Community Survey data.\n"
            "Ask me about demographics, income, education, housing, and more!\n\n"
            "[dim]Commands: 'quit' to exit, 'reset' to clear history, 'cost' for usage, 'help' for examples[/dim]",
            title="Welcome",
            border_style="blue"
        ))

        examples = """
[bold]Example queries:[/bold]
• "What is the population of California?"
• "Download the median household income table for Texas"
• "Compare poverty rates between Michigan and Ohio"
• "Find tables about educational attainment"
• "Get housing data for Detroit, Michigan"
• "What's the unemployment rate in Florida?"
"""

        while True:
            try:
                user_input = console.input("\n[bold green]You:[/bold green] ").strip()

                if not user_input:
                    continue

                if user_input.lower() == "quit":
                    # Show final usage stats
                    console.print(f"\n[dim]Session {self.usage.summary()}[/dim]")
                    console.print("[dim]Goodbye![/dim]")
                    break

                if user_input.lower() == "reset":
                    self.reset_conversation()
                    console.print("[dim]Conversation reset.[/dim]")
                    continue

                if user_input.lower() == "cost":
                    console.print(f"[cyan]{self.usage.summary()}[/cyan]")
                    continue

                if user_input.lower() == "help":
                    console.print(examples)
                    continue

                # Get response
                console.print()
                response = await self.chat(user_input)

                # Display response
                console.print("\n[bold blue]Assistant:[/bold blue]")
                console.print(Markdown(response))

                # Show cost for this query
                console.print(f"\n[dim]💰 {self.usage.summary()}[/dim]")

            except KeyboardInterrupt:
                console.print("\n[dim]Interrupted. Type 'quit' to exit.[/dim]")
            except Exception as e:
                console.print(f"[red]Error: {e}[/red]")


# =============================================================================
# CLI
# =============================================================================

def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="ACS Data Assistant - AI-powered Census data helper"
    )
    parser.add_argument(
        "--query", "-q",
        help="Single query to process (non-interactive mode)"
    )
    parser.add_argument(
        "--output", "-o",
        default="./acs_downloads",
        help="Output directory for downloaded files"
    )

    args = parser.parse_args()

    output_dir = Path(args.output)
    agent = ACSAgent(output_dir=output_dir)

    async def run():
        if args.query:
            # Single query mode
            response = await agent.chat(args.query)
            console.print(Markdown(response))
            console.print(f"\n[dim]💰 {agent.get_usage_summary()}[/dim]")
        else:
            # Interactive mode
            await agent.run_interactive()

    asyncio.run(run())


if __name__ == "__main__":
    main()
