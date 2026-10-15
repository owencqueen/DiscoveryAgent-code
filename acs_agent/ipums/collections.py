"""
Hardcoded metadata for IPUMS collections.

Since the IPUMS API does not provide a metadata endpoint for microdata
collections (sample IDs, variable names, etc.), we store commonly-used
reference data here so the agent can discover what's available without
requiring users to browse the IPUMS website.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Microdata collection identifiers accepted by ipumspy
# ---------------------------------------------------------------------------

MICRODATA_COLLECTIONS: dict[str, dict] = {
    "usa": {
        "label": "IPUMS USA",
        "description": (
            "U.S. Census and American Community Survey microdata, 1850-present. "
            "Harmonized individual and household records with consistent variable "
            "coding across decades."
        ),
        "topics": [
            "demographics", "income", "education", "employment", "housing",
            "migration", "disability", "health insurance", "ancestry",
            "commuting", "fertility",
        ],
        "time_span": "1850-present",
        "geography_detail": "State, county, metro area, PUMA (varies by sample)",
    },
    "cps": {
        "label": "IPUMS CPS (Current Population Survey)",
        "description": (
            "Monthly labor force survey, 1962-present. "
            "Core labor force data plus supplemental topics: fertility, tobacco, "
            "volunteer activities, voter registration, computer/internet use, "
            "food security."
        ),
        "topics": [
            "employment", "unemployment", "earnings", "labor force",
            "work hours", "industry", "occupation", "union membership",
            "program participation",
        ],
        "time_span": "1962-present",
        "geography_detail": "State, metro area",
    },
    "nhis": {
        "label": "IPUMS NHIS (National Health Interview Survey)",
        "description": (
            "Health information and healthcare access for the U.S. civilian "
            "population, 1963-present. Covers health status, conditions, "
            "healthcare utilization, and health behaviors."
        ),
        "topics": [
            "health status", "chronic conditions", "healthcare access",
            "health insurance", "health behaviors", "disability",
            "mental health", "preventive care",
        ],
        "time_span": "1963-present",
        "geography_detail": "Region (limited geographic detail)",
    },
    "meps": {
        "label": "IPUMS MEPS (Medical Expenditure Panel Survey)",
        "description": (
            "Healthcare utilization, expenditures, and health insurance, "
            "1996-present. Tracks medical spending at the person and event level."
        ),
        "topics": [
            "healthcare expenditures", "health insurance", "medical conditions",
            "prescribed medicines", "healthcare utilization", "out-of-pocket costs",
        ],
        "time_span": "1996-present",
        "geography_detail": "Region (limited geographic detail)",
    },
}

# ---------------------------------------------------------------------------
# Aggregate data collections
# ---------------------------------------------------------------------------

AGGREGATE_COLLECTIONS: dict[str, dict] = {}


# ---------------------------------------------------------------------------
# Common IPUMS USA sample IDs
# These are the most frequently used samples. The full list is much larger
# and can be browsed via get_all_sample_info().
# ---------------------------------------------------------------------------

COMMON_USA_SAMPLES: dict[str, str] = {
    # Recent ACS 1-year
    "us2024a": "2024 ACS 1-year",
    "us2023a": "2023 ACS 1-year",
    "us2022a": "2022 ACS 1-year",
    "us2021a": "2021 ACS 1-year",
    "us2020a": "2020 ACS 1-year",
    "us2019a": "2019 ACS 1-year",
    # Recent ACS 5-year (PRCS)
    "us2024b": "2020-2024 ACS 5-year",
    "us2023b": "2019-2023 ACS 5-year",
    "us2022b": "2018-2022 ACS 5-year",
    # Decennial census
    "us2010a": "2010 Census 10%",
    "us2000a": "2000 Census 5%",
    "us1990a": "1990 Census 5%",
    "us1980a": "1980 Census 5%",
    "us1970a": "1970 Census 1% (Form 1 Metro)",
    "us1960a": "1960 Census 5%",
    "us1950a": "1950 Census 1%",
    "us1940a": "1940 Census 1%",
}

COMMON_CPS_SAMPLES: dict[str, str] = {
    "cps2024_03s": "March 2024 CPS (ASEC)",
    "cps2023_03s": "March 2023 CPS (ASEC)",
    "cps2022_03s": "March 2022 CPS (ASEC)",
    "cps2021_03s": "March 2021 CPS (ASEC)",
    "cps2020_03s": "March 2020 CPS (ASEC)",
}


# ---------------------------------------------------------------------------
# Commonly-used IPUMS harmonized variable names
# These are IPUMS-standardized names, not raw Census Bureau codes.
# ---------------------------------------------------------------------------

COMMON_IPUMS_VARIABLES: dict[str, dict] = {
    # Demographics
    "AGE": {"label": "Age", "record_type": "person"},
    "SEX": {"label": "Sex", "record_type": "person"},
    "RACE": {"label": "Race (general)", "record_type": "person"},
    "RACED": {"label": "Race (detailed)", "record_type": "person"},
    "HISPAN": {"label": "Hispanic origin (general)", "record_type": "person"},
    "HISPAND": {"label": "Hispanic origin (detailed)", "record_type": "person"},
    "BPL": {"label": "Birthplace (general)", "record_type": "person"},
    "CITIZEN": {"label": "Citizenship status", "record_type": "person"},
    "MARST": {"label": "Marital status", "record_type": "person"},
    "FERTYR": {"label": "Children born in past year", "record_type": "person"},
    # Education
    "EDUC": {"label": "Educational attainment (general)", "record_type": "person"},
    "EDUCD": {"label": "Educational attainment (detailed)", "record_type": "person"},
    "SCHOOL": {"label": "School attendance", "record_type": "person"},
    # Employment
    "EMPSTAT": {"label": "Employment status (general)", "record_type": "person"},
    "EMPSTATD": {"label": "Employment status (detailed)", "record_type": "person"},
    "OCC": {"label": "Occupation", "record_type": "person"},
    "IND": {"label": "Industry", "record_type": "person"},
    "CLASSWKR": {"label": "Class of worker", "record_type": "person"},
    "WKSWORK2": {"label": "Weeks worked", "record_type": "person"},
    "UHRSWORK": {"label": "Usual hours worked per week", "record_type": "person"},
    # Income
    "INCTOT": {"label": "Total personal income", "record_type": "person"},
    "INCWAGE": {"label": "Wage and salary income", "record_type": "person"},
    "INCEARN": {"label": "Total personal earned income", "record_type": "person"},
    "POVERTY": {"label": "Poverty status (% of poverty line)", "record_type": "person"},
    "HHINCOME": {"label": "Total household income", "record_type": "household"},
    # Housing
    "OWNERSHP": {"label": "Ownership of dwelling (owned/rented)", "record_type": "household"},
    "RENT": {"label": "Monthly contract rent", "record_type": "household"},
    "VALUEH": {"label": "House value", "record_type": "household"},
    "ROOMS": {"label": "Number of rooms", "record_type": "household"},
    "BEDROOMS": {"label": "Number of bedrooms", "record_type": "household"},
    "MORTGAGE": {"label": "Mortgage status", "record_type": "household"},
    # Geographic identifiers
    "STATEFIP": {"label": "State (FIPS code)", "record_type": "household"},
    "COUNTYFIP": {"label": "County (FIPS code)", "record_type": "household"},
    "MET2013": {"label": "Metropolitan area (2013 delineation)", "record_type": "household"},
    "PUMA": {"label": "Public Use Microdata Area", "record_type": "household"},
    # Weights
    "PERWT": {"label": "Person weight", "record_type": "person"},
    "HHWT": {"label": "Household weight", "record_type": "household"},
}
