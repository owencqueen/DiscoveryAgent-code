================================================================================
                    CENSUS & SURVEY DATA — REFERENCE GUIDE
================================================================================

This document describes all data accessible through the acs_agent package,
which provides three backends: the Census Bureau ACS API, the Census Bureau
PUMS API, and IPUMS microdata via ipumspy.

Downloaded files are stored in the acs_downloads/ directory.
NOTE: Not all data may be available locally — some queries require live API
calls or IPUMS extract submissions.

================================================================================
 1. OVERVIEW
================================================================================

The American Community Survey (ACS) is an ongoing, nationwide survey conducted
by the U.S. Census Bureau to provide detailed, up-to-date information about the
social, economic, housing, and demographic characteristics of the U.S.
population. Unlike the decennial census, which counts every resident once per
decade, the ACS is administered continuously to a sample of households each
year, producing timely estimates that reflect changes in communities over time.

The ACS collects data on age, sex, race and ethnicity, education, income,
employment, commuting patterns, health insurance coverage, housing
characteristics, and housing costs. Data are released as one-year and five-year
estimates, with the latter providing greater statistical reliability for smaller
geographic areas (census tracts, towns, rural communities).

Beyond the ACS, the acs_agent package also provides access to labor force
surveys (CPS) and health surveys (NHIS, MEPS) through IPUMS, as well as
historical census microdata back to 1850.


================================================================================
 2. DATA BACKENDS
================================================================================

                     +-------------------------+
                     |     ACS Agent (Claude)   |
                     |       15 total tools     |
                     +----+--------+--------+---+
                          |        |        |
              +-----------+        |        +-----------+
              v                    v                    v
   +------------------+ +------------------+ +------------------+
   |  Census Bureau   | |  Census Bureau   | |   IPUMS (via     |
   |  ACS API         | |  PUMS API        | |   ipumspy)       |
   |                  | |                  | |                  |
   |  7 tools         | |  3 tools         | |  5 tools         |
   |  Summary tables  | |  Microdata API   | |  Historical +    |
   |  2005-2024       | |  Recent years    | |  health/labor    |
   |  Real-time       | |  Real-time       | |  data            |
   +------------------+ +------------------+ +------------------+


Census Bureau ACS API
  - Real-time access to published ACS summary tables (pre-tabulated).
  - Coverage: 2005-present.
  - Geography: nation, state, county, tract, place, metro, zip code (ZCTA).
  - Instant responses.

Census Bureau PUMS API
  - Real-time access to Public Use Microdata Sample (individual records).
  - Coverage: Recent ACS years.
  - Instant responses for queries and tabulations.

IPUMS (via ipumspy)
  - Extract-based access to harmonized historical microdata.
  - Four collections: USA, CPS, NHIS, MEPS.
  - Responses range from instant (metadata) to minutes (extracts).

================================================================================
 3. DATA SOURCES — SUMMARY TABLES (Census Bureau ACS API)
================================================================================

Summary tables are pre-tabulated statistics published by the Census Bureau.
These cover a wide range of topics at many geographic levels.

Coverage:  2005-present (1-year estimates), 2009-present (5-year estimates)

Key table series:

  B01001   Sex by age
  B01003   Total population
  B02001   Race
  B03003   Hispanic or Latino origin
  B05001   Nativity and citizenship status
  B06001   Place of birth
  B07001   Geographic mobility
  B08301   Means of transportation to work
  B11001   Household type
  B15003   Educational attainment (25+)
  B17001   Poverty status
  B19013   Median household income
  B19001   Household income distribution
  B23025   Employment status (16+)
  B25001   Housing units
  B25003   Tenure (owner/renter)
  B25077   Median home value
  B27001   Health insurance coverage

Thousands of additional tables exist. Use the search_tables tool to discover
tables by topic keyword.


================================================================================
 4. DATA SOURCES — PUMS MICRODATA (Census Bureau PUMS API)
================================================================================

The Public Use Microdata Sample provides individual- and household-level
records from the ACS, allowing custom tabulations and analyses not possible
with pre-tabulated summary tables.

Coverage:  Recent ACS years (real-time API)
Geography: Public Use Microdata Areas (PUMAs), states

Common PUMS variable codes:

  Demographics       AGEP (age), SEX, RAC1P (race), HISP (Hispanic origin)
  Education          SCHL (educational attainment), SCH (school enrollment)
  Employment         ESR (employment status), COW (class of worker),
                     WKHP (hours worked/week), WKW (weeks worked)
  Income             PINCP (personal income), WAGP (wages), HINCP (household
                     income), PERNP (earnings), SEMP (self-employment income)
  Housing            TEN (tenure), RNTP (rent), VALP (property value),
                     RMSP (rooms), BDSP (bedrooms), MHP (mobile home costs)
  Insurance          HINS1-HINS7 (insurance coverage types)
  Commuting          JWTR (transportation mode), JWMNP (travel time)


================================================================================
 5. DATA SOURCES — IPUMS MICRODATA COLLECTIONS
================================================================================

IPUMS provides harmonized individual-level microdata from multiple survey
programs. Variables are coded consistently across years and decades.

Available collections:  usa, cps, nhis, meps

------------------------------------------------------------------------------
 5.1  IPUMS USA  (collection: usa)
------------------------------------------------------------------------------

Harmonized individual and household records from every U.S. Census
(1850-2010) and the American Community Survey (2000-present). Variables are
coded consistently across decades, so "EDUC" means the same thing in 1940 as
in 2022.

Topics:  Demographics, income, education, employment, housing, migration,
         disability, health insurance, ancestry, commuting, fertility

Time span:   1850-present
Geography:   State, county, metro, PUMA

Why use IPUMS USA over the direct Census PUMS API?
  - Historical data back to 1850 (PUMS API only covers recent years)
  - Harmonized variables with consistent coding across decades
  - Full-count data available for some historical censuses

Common samples:
  us2024a    2024 ACS 1-year
  us2023a    2023 ACS 1-year
  us2022a    2022 ACS 1-year
  us2024b    2020-2024 ACS 5-year
  us2023b    2019-2023 ACS 5-year
  us2010a    2010 Census 10%
  us2000a    2000 Census 5%
  us1990a    1990 Census 5%
  us1980a    1980 Census 5%
  us1970a    1970 Census 1%
  us1960a    1960 Census 5%
  us1950a    1950 Census 1%
  us1940a    1940 Census 1%

Hundreds of additional samples exist (full-count files, linked data,
specialized subsets). Use ipums_search_samples to browse the complete list.

------------------------------------------------------------------------------
 5.2  IPUMS CPS  (collection: cps)
------------------------------------------------------------------------------

The Current Population Survey — the U.S. government's primary labor force
survey. Monthly data since 1962. The March Annual Social and Economic
Supplement (ASEC) provides the most detailed income and poverty data.

Topics:  Employment, unemployment, earnings, labor force participation,
         work hours, industry, occupation, union membership, program
         participation

Time span:   1962-present
Geography:   State, metro

Common samples:
  cps2024_03s    March 2024 CPS (ASEC)
  cps2023_03s    March 2023 CPS (ASEC)
  cps2022_03s    March 2022 CPS (ASEC)
  cps2021_03s    March 2021 CPS (ASEC)
  cps2020_03s    March 2020 CPS (ASEC)

Monthly basic files also available (format: cps{year}_{month}b).

------------------------------------------------------------------------------
 5.3  IPUMS NHIS  (collection: nhis)
------------------------------------------------------------------------------

The National Health Interview Survey — the principal source of information
on the health of the U.S. civilian non-institutionalized population.

Topics:  Health status, chronic conditions, healthcare access, health
         insurance coverage, health behaviors, disability, mental health,
         preventive care

Time span:   1963-present
Geography:   Region only (limited geographic detail for confidentiality)

------------------------------------------------------------------------------
 5.4  IPUMS MEPS  (collection: meps)
------------------------------------------------------------------------------

The Medical Expenditure Panel Survey tracks healthcare utilization,
expenditures, and insurance coverage at the person level.

Topics:  Healthcare expenditures, health insurance, medical conditions,
         prescribed medicines, healthcare utilization, out-of-pocket costs

Time span:   1996-present
Geography:   Region only (limited geographic detail)

Special feature: Round-level data structure tracks individuals over time
within the survey panel.


================================================================================
 6. IPUMS VARIABLE REFERENCE
================================================================================

IPUMS uses its own harmonized variable names, distinct from Census Bureau
variable codes. Below are the most commonly used variables for IPUMS USA.
Other collections (CPS, NHIS, MEPS) have their own variable sets.

DEMOGRAPHICS
  Variable    Label                           Record Type
  AGE         Age                             Person
  SEX         Sex                             Person
  RACE        Race (general)                  Person
  RACED       Race (detailed)                 Person
  HISPAN      Hispanic origin (general)       Person
  HISPAND     Hispanic origin (detailed)      Person
  BPL         Birthplace (general)            Person
  CITIZEN     Citizenship status              Person
  MARST       Marital status                  Person
  FERTYR      Children born in past year      Person

EDUCATION
  Variable    Label                           Record Type
  EDUC        Educational attainment (gen.)   Person
  EDUCD       Educational attainment (det.)   Person
  SCHOOL      School attendance               Person

EMPLOYMENT
  Variable    Label                           Record Type
  EMPSTAT     Employment status (general)     Person
  EMPSTATD    Employment status (detailed)    Person
  OCC         Occupation                      Person
  IND         Industry                        Person
  CLASSWKR    Class of worker                 Person
  WKSWORK2    Weeks worked                    Person
  UHRSWORK    Usual hours worked per week     Person

INCOME
  Variable    Label                           Record Type
  INCTOT      Total personal income           Person
  INCWAGE     Wage and salary income          Person
  INCEARN     Total personal earned income    Person
  POVERTY     Poverty status (% of poverty)   Person
  HHINCOME    Total household income          Household

HOUSING
  Variable    Label                           Record Type
  OWNERSHP    Ownership (owned/rented)        Household
  RENT        Monthly contract rent           Household
  VALUEH      House value                     Household
  ROOMS       Number of rooms                 Household
  BEDROOMS    Number of bedrooms              Household
  MORTGAGE    Mortgage status                 Household

GEOGRAPHY
  Variable    Label                           Record Type
  STATEFIP    State (FIPS code)               Household
  COUNTYFIP   County (FIPS code)              Household
  MET2013     Metro area (2013 delineation)   Household
  PUMA        Public Use Microdata Area       Household

WEIGHTS (always include in extracts for correct population estimates)
  Variable    Label                           Record Type
  PERWT       Person weight                   Person
  HHWT        Household weight                Household


================================================================================
 7. SAMPLE ID FORMAT
================================================================================

  Collection   Format                  Examples
  USA          us{year}{variant}       us2022a (ACS 1yr), us2022b (ACS 5yr)
  CPS          cps{year}_{month}s      cps2023_03s (March ASEC)
  NHIS         collection-specific     Use ipums_search_samples to discover
  MEPS         mp{year}                mp2022, mp2018


================================================================================
 8. CHOOSING THE RIGHT DATA SOURCE
================================================================================

  I need...                                Use            Tool(s)
  ---------------------------------------- -------------- -------------------------
  Current ACS summary table (2009+)        Census API     search_tables, download_table
  Quick state-level stats                  Census API     get_quick_stats
  Compare states on key metrics            Census API     compare_geographies
  Recent PUMS microdata (real-time)        Census PUMS    pums_search_variables, pums_get_records
  PUMS tabulations (real-time)             Census PUMS    pums_tabulate
  Historical census microdata (pre-2005)   IPUMS USA      ipums_get_microdata
  Harmonized data across decades           IPUMS USA      ipums_get_microdata
  CPS labor force data                     IPUMS CPS      ipums_get_microdata
  Health survey data (NHIS, MEPS)          IPUMS Health   ipums_get_microdata

Decision flow:

  Is the data from the current ACS (2009+)?
    YES -> Is it a summary table or quick stat?
             YES -> Census API (search_tables, download_table, get_quick_stats)
             NO  -> Need individual records?
                      YES -> Census PUMS API (pums_get_records, pums_tabulate)
                      NO  -> Census API (download_variables)
    NO  -> Is it individual-level records?
             YES -> IPUMS microdata (ipums_get_microdata) -- 2-30 min
             NO  -> Not currently supported (aggregate historical data
                    would require NHGIS, which is not enabled)


================================================================================
 9. LATENCY AND PERFORMANCE
================================================================================

Census Bureau ACS API:    Instant (seconds)
Census Bureau PUMS API:   Instant (seconds)
IPUMS metadata tools:     Instant (seconds)
IPUMS microdata extracts: 2-30 minutes depending on size

  Extract size                          Typical wait
  Small (1 sample, few variables)       2-5 minutes
  Medium (several samples)              5-15 minutes
  Large (many decades)                  15-60 minutes
  Full-count census files               Hours (exceeds agent timeout)

Tips for faster IPUMS extracts:
  - Request fewer samples (1-2 instead of 10+)
  - Request only the variables you need
  - Use case selections to filter records
  - IPUMS extract files expire after 72 hours
