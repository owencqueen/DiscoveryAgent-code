"""Tool definitions for PUMS access."""

PUMS_TOOLS = [
    {
        "name": "pums_search_variables",
        "description": "Search PUMS variables by keyword. Use this to find microdata fields for person or housing records.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query (e.g., 'household income', 'tenure', 'age')"
                },
                "year": {
                    "type": "integer",
                    "description": "ACS year. Default: 2024",
                    "default": 2024
                },
                "record_type": {
                    "type": "string",
                    "enum": ["person", "housing"],
                    "description": "PUMS record type"
                },
                "sample": {
                    "type": "string",
                    "enum": ["acs1", "acs5"],
                    "description": "ACS sample type (1-year or 5-year). Default: acs5",
                    "default": "acs5"
                }
            },
            "required": ["query", "record_type"]
        }
    },
    {
        "name": "pums_get_records",
        "description": "Retrieve bounded PUMS microdata records. Requires geography and/or restrictive filters.",
        "input_schema": {
            "type": "object",
            "properties": {
                "year": {
                    "type": "integer",
                    "description": "ACS year"
                },
                "record_type": {
                    "type": "string",
                    "enum": ["person", "housing"],
                    "description": "PUMS record type"
                },
                "sample": {
                    "type": "string",
                    "enum": ["acs1", "acs5"],
                    "description": "ACS sample type (1-year or 5-year). Default: acs5",
                    "default": "acs5"
                },
                "variables": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Variables to retrieve (e.g., ['AGEP', 'HINCP', 'TEN'])"
                },
                "filters": {
                    "type": "object",
                    "description": "Filter constraints (equality or range). Example: {\"TEN\": 2, \"AGEP\": [25, 64]}",
                    "default": {}
                },
                "geography": {
                    "type": "object",
                    "description": "Supported keys: nation, region, division, state, puma. Example: {\"state\": \"06\", \"puma\": \"03706\"}"
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum rows to return. Default: 5000",
                    "default": 5000
                }
            },
            "required": ["year", "record_type", "variables", "geography"]
        }
    },
    {
        "name": "pums_tabulate",
        "description": "Lightweight PUMS tabulation (counts/averages) by group. Avoids returning raw rows.",
        "input_schema": {
            "type": "object",
            "properties": {
                "year": {
                    "type": "integer",
                    "description": "ACS year"
                },
                "record_type": {
                    "type": "string",
                    "enum": ["person", "housing"],
                    "description": "PUMS record type"
                },
                "sample": {
                    "type": "string",
                    "enum": ["acs1", "acs5"],
                    "description": "ACS sample type (1-year or 5-year). Default: acs5",
                    "default": "acs5"
                },
                "metrics": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Metrics to compute. Use 'count' for weighted counts or 'avg:VAR' for averages.",
                    "default": ["count"]
                },
                "group_by": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Grouping variables (e.g., ['TEN'] or ['TEN', 'RAC1P'])"
                },
                "filters": {
                    "type": "object",
                    "description": "Filter constraints (equality or range).",
                    "default": {}
                },
                "geography": {
                    "type": "object",
                    "description": "Supported keys: nation, region, division, state, puma"
                },
                "weight": {
                    "type": "string",
                    "description": "Optional weight variable override (defaults to PWGTP or WGTP)."
                }
            },
            "required": ["year", "record_type", "group_by", "geography"]
        }
    }
]
