"""
CensusDataAgent - wraps the external acs_agent package to provide broad
U.S. Census Bureau data access (ACS summary tables, PUMS microdata,
IPUMS historical/health data) as a tool agent within the DiscoveryAgent Director workflow.

This class adheres to the BaseAgent interface and is meant to be
registered in STRING_TO_AGENT_MAP for use by the Director.

The ACS/PUMS/IPUMS helper is bundled as the ``acs_agent`` package.
"""

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Any

from ..base import BaseAgent, AgentToolConfig
from ...utils.logger import Logger


@dataclass
class CensusDataAgentConfig(AgentToolConfig):
    """
    Configuration for CensusDataAgent.

    Fields inherited from AgentToolConfig:
        agent_name: registry key used in STRING_TO_AGENT_MAP
        tag: tag used by the Director to invoke this agent (<tag>...</tag>)
        tool_prompt: short description shown to the Director about when to use it
    """
    output_dir: str = "./census_downloads"
    log_dir: str = "logs"
    tool_prompt: str = """Description: Helper agent that downloads data from the U.S. Census Bureau, including ACS summary tables, PUMS microdata, and IPUMS historical datasets.

DATA SOURCES:

1. ACS Summary Tables (2009-2024, default 2024):
   - Demographics: Population, age, sex, race, Hispanic origin, citizenship
   - Economics: Household/family income, poverty status, earnings, employment, industry
   - Education: Educational attainment, school enrollment
   - Housing: Home values, rent, occupancy, costs, tenure (own/rent)
   - Transportation: Commute times, means of transportation
   - Social: Marital status, fertility, veteran status, health insurance, internet access
   - Language: Languages spoken, English proficiency
   Common table prefixes: B01xxx (Age/Sex), B02xxx (Race), B08xxx (Commuting),
   B15xxx (Education), B17xxx (Poverty), B19xxx (Income), B23xxx (Employment), B25xxx (Housing)

2. PUMS Microdata (individual/household-level records):
   - Person records: age, sex, race, income, education, occupation, industry, etc.
   - Housing records: rent, value, rooms, vehicles, heating fuel, etc.
   - Supports tabulation (counts, averages) and bounded record retrieval
   - Geography: Nation, Region, Division, State, PUMA only (no tracts/counties)

3. IPUMS Historical/Survey Data:
   - IPUMS USA (1850-present): Census and ACS microdata with harmonized variables
   - IPUMS CPS (1962-present): Current Population Survey (employment, labor force, earnings)
   - IPUMS NHIS (1963-present): National Health Interview Survey (health status, conditions, insurance)
   - IPUMS MEPS (1996-present): Medical Expenditure Panel Survey (healthcare costs, utilization)
   - Note: IPUMS extracts take 2-10 minutes to process

GEOGRAPHIC LEVELS (ACS):
- Nation, State, County, City/Place, Census Tract, Block Group
- Also: Congressional Districts, School Districts, ZIP Code Areas (ZCTA)
- Use "*" for all (e.g., all states, all counties in a state)

Guidelines:
1. Always specify geography: state required, add county/city if needed
2. Search before downloading: "Find tables about median income"
3. Be specific: "median household income" not just "income"
4. Use PUMS when you need individual-level records or custom cross-tabulations
5. Use IPUMS for historical trends, labor data (CPS), or health data (NHIS/MEPS)
6. The agent downloads data but cannot analyze it or run code on it
7. The more precise your prompt, the more likely you are to get the data you want
8. If your query is not successful the first time, retry with clearer instructions

Outputs:
- CSV files with estimates and optional margins of error (MOE)
- JSON data dictionary saved alongside each CSV (explains all variable codes)

IMPORTANT: You have no data if you don't first request it from this agent! You must request it and confirm download before analyzing anything."""


class CensusDataAgent(BaseAgent):
    """
    Thin wrapper that forwards queries to the ACSAgent from the external acs_agent package.
    Provides access to ACS summary tables, PUMS microdata, and IPUMS historical data.
    """

    def __init__(self, config: CensusDataAgentConfig, logger: Optional[Any] = None):
        super().__init__(config, logger)
        self.config = config
        self.logger = logger if logger is not None else Logger("census_data_agent", log_dir=config.log_dir)

        # Import the bundled acs_agent package.
        from acs_agent import ACSAgent

        output_path = Path(config.output_dir).resolve()
        output_path.mkdir(parents=True, exist_ok=True)
        self.logger.log_action("census_output_dir_init", f"Census agent output directory: {output_path}")
        self._acs_agent = ACSAgent(output_dir=output_path)

    def _run_coro(self, coro):
        """
        Run an async coroutine safely whether or not a loop is already running.
        """
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            new_loop = asyncio.new_event_loop()
            try:
                return new_loop.run_until_complete(coro)
            finally:
                new_loop.close()
        return asyncio.run(coro)

    def execute(self, query: str, **kwargs) -> str:
        """
        Execute a Census data query. Returns the assistant's Markdown/text response.
        """
        try:
            result = self._run_coro(self._acs_agent.chat(query))
            self.logger.log_response(result, "census_data_agent_result")
            return result
        except Exception as e:
            err_msg = f"CensusDataAgent error: {str(e)}"
            try:
                self.logger.log_error(err_msg)
            except Exception:
                pass
            raise

    def cleanup(self):
        """No-op hook for symmetry with other agents."""
        return None
