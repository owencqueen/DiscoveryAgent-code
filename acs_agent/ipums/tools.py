"""Tool definitions for IPUMS data access via ipumspy."""

IPUMS_TOOLS = [
    # ------------------------------------------------------------------
    # Tier 1: Instant metadata tools
    # ------------------------------------------------------------------
    {
        "name": "ipums_list_collections",
        "description": (
            "List all available IPUMS data collections and their capabilities. "
            "Use this when the user asks about historical census data, "
            "health surveys, or CPS labor data. Returns collection "
            "names, descriptions, topics, and time spans."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "collection_type": {
                    "type": "string",
                    "enum": ["microdata", "all"],
                    "description": (
                        "Filter by collection type. 'microdata' for individual-level "
                        "records (USA, CPS, NHIS, etc.). Default: all"
                    ),
                    "default": "all",
                },
            },
            "required": [],
        },
    },
    {
        "name": "ipums_search_samples",
        "description": (
            "List available data samples for an IPUMS microdata collection. "
            "Returns sample IDs needed to build extract requests. "
            "For example, IPUMS USA samples include decennial censuses and "
            "annual ACS releases."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "collection": {
                    "type": "string",
                    "enum": [
                        "usa", "cps",
                        "nhis", "meps",
                    ],
                    "description": "IPUMS microdata collection identifier.",
                },
            },
            "required": ["collection"],
        },
    },
    # ------------------------------------------------------------------
    # Tier 1: Instant - read already-downloaded data
    # ------------------------------------------------------------------
    {
        "name": "ipums_read_extract_data",
        "description": (
            "Read data from a previously downloaded IPUMS extract file. "
            "Parses the DDI codebook and loads data into a summary. "
            "Use this to re-examine data that was already downloaded, or "
            "to pull specific variables from a large extract."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "ddi_file": {
                    "type": "string",
                    "description": (
                        "Path to the DDI XML codebook file (.xml) "
                        "that was downloaded alongside the data."
                    ),
                },
                "data_file": {
                    "type": "string",
                    "description": (
                        "Path to the data file (.dat.gz, .csv.gz, or .csv). "
                        "If omitted, inferred from the DDI codebook."
                    ),
                },
                "subset": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Optional list of variable names to load. "
                        "Loads all variables if omitted."
                    ),
                },
                "head": {
                    "type": "integer",
                    "description": "Number of preview rows to return. Default: 20",
                    "default": 20,
                },
            },
            "required": ["ddi_file"],
        },
    },
    # ------------------------------------------------------------------
    # Tier 1: Instant - check on a running extract
    # ------------------------------------------------------------------
    {
        "name": "ipums_check_extract_status",
        "description": (
            "Check the status of a previously submitted IPUMS extract. "
            "Use this if a prior extract request timed out or you want "
            "to resume downloading a completed extract."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "collection": {
                    "type": "string",
                    "description": (
                        "IPUMS collection identifier "
                        "(e.g., 'usa', 'cps')."
                    ),
                },
                "extract_id": {
                    "type": "integer",
                    "description": "The numeric extract ID returned by a previous submission.",
                },
            },
            "required": ["collection", "extract_id"],
        },
    },
    # ------------------------------------------------------------------
    # Tier 3: Minutes (IPUMS microdata extract)
    # ------------------------------------------------------------------
    {
        "name": "ipums_get_microdata",
        "description": (
            "Request and download IPUMS microdata (individual/household records). "
            "This submits an extract, waits for processing (typically 2-10 minutes "
            "for focused requests), downloads the result, and returns a data preview. "
            "Use this for: harmonized historical census microdata, CPS labor data, "
            "or NHIS/MEPS health surveys. "
            "Requires sample IDs and variable names - use "
            "ipums_search_samples and the IPUMS website to discover these. "
            "IMPORTANT: Keep extracts small (few samples, few variables) for "
            "faster processing. Large multi-decade requests may take 30+ minutes."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "collection": {
                    "type": "string",
                    "enum": [
                        "usa", "cps",
                        "nhis", "meps",
                    ],
                    "description": "IPUMS microdata collection identifier.",
                },
                "samples": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Sample IDs to include (e.g., ['us2022a', 'us2021a'] "
                        "for IPUMS USA, ['cps2023_03s'] for CPS). "
                        "Use ipums_search_samples to discover available IDs."
                    ),
                },
                "variables": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "IPUMS harmonized variable names to extract "
                        "(e.g., ['AGE', 'SEX', 'RACE', 'INCTOT', 'EDUC']). "
                        "These are IPUMS-standardized names, not raw Census codes."
                    ),
                },
                "description": {
                    "type": "string",
                    "description": "Short description of the extract purpose.",
                },
                "data_format": {
                    "type": "string",
                    "enum": ["csv", "fixed_width"],
                    "description": "Output format. Default: csv (easier to work with).",
                    "default": "csv",
                },
                "data_structure": {
                    "type": "string",
                    "enum": ["rectangular_on_P", "rectangular_on_H", "hierarchical"],
                    "description": (
                        "Data structure. 'rectangular_on_P' rectangularizes on person "
                        "records (default), 'rectangular_on_H' on household records, "
                        "'hierarchical' preserves the full hierarchy."
                    ),
                    "default": "rectangular_on_P",
                },
                "case_selections": {
                    "type": "object",
                    "description": (
                        "Optional case selection filters. Keys are variable names, "
                        "values are lists of codes to keep. "
                        "Example: {\"SEX\": [\"2\"]} to keep only females."
                    ),
                },
            },
            "required": ["collection", "samples", "variables"],
        },
    },
]
