"""
Lightweight wrapper that exposes the ACS chat agent behind a simple `execute`
method while handling output/log directories.
"""

import asyncio
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .agent import ACSAgent


@dataclass
class ACSDataAgentConfig:
    """Configuration for `ACSDataAgent`."""

    output_dir: str = "./acs_downloads"
    log_dir: str = "logs"
    tool_prompt: str = """Description: Helper agent that downloads data from the U.S. Census American Community Survey (ACS).

DATA AVAILABLE (2009-2024, default 2024):
- Demographics: Population, age, sex, race, Hispanic origin, citizenship
- Economics: Household/family income, poverty status, earnings, employment, industry
- Education: Educational attainment, school enrollment
- Housing: Home values, rent, occupancy, costs, tenure (own/rent)
- Transportation: Commute times, means of transportation
- Social: Marital status, fertility, veteran status, health insurance, internet access
- Language: Languages spoken, English proficiency

GEOGRAPHIC LEVELS:
- Nation, State, County, City/Place, Census Tract, Block Group
- Also: Congressional Districts, School Districts, ZIP Code Areas (ZCTA)
- Use "*" for all (e.g., all states, all counties in a state)

COMMON TABLE PREFIXES:
- B01xxx: Age/Sex/Population  - B15xxx: Education      - B19xxx: Income
- B02xxx: Race                - B17xxx: Poverty        - B23xxx: Employment
- B25xxx: Housing             - B08xxx: Commuting

Guidelines:
1. Always specify geography: state required, add county/city if needed
2. Search before downloading: "Find tables about median income" or "What is table B19013?"
3. Be specific: "median household income" not just "income"
4. For quick facts: "Get quick stats for California"
5. For comparisons: "Compare poverty rates between Michigan, Ohio, and Indiana"
6. Clarify ambiguous places: "Portland, Oregon" not just "Portland"
7. Request bulk downloads to avoid repeatedly fetching individual variables.
8. If a query fails, retry with clearer instructions.

Outputs:
- CSV files with estimates and optional margins of error (MOE)
- JSON data dictionary saved alongside each CSV (explains all variable codes)
"""


class ACSDataAgent:
    """Simple synchronous wrapper around the async `ACSAgent`."""

    def __init__(self, config: Optional[ACSDataAgentConfig] = None, logger: Optional[logging.Logger] = None):
        self.config = config or ACSDataAgentConfig()
        self.logger = logger or self._build_logger(self.config.log_dir)

        output_path = Path(self.config.output_dir).resolve()
        output_path.mkdir(parents=True, exist_ok=True)
        self.logger.info("ACS agent output directory: %s", output_path)
        self._acs_agent = ACSAgent(output_dir=output_path)

    def _build_logger(self, log_dir: str) -> logging.Logger:
        """Create a simple file + console logger."""
        log_dir_path = Path(log_dir)
        log_dir_path.mkdir(parents=True, exist_ok=True)

        logger = logging.getLogger("acs_data_agent")
        if logger.handlers:
            return logger

        logger.setLevel(logging.INFO)

        file_handler = logging.FileHandler(log_dir_path / "acs_data_agent.log")
        formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        return logger

    def _run_coro(self, coro):
        """Run an async coroutine safely whether or not a loop is already running."""
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
        """Execute an ACS query and return the assistant's response text."""
        try:
            result = self._run_coro(self._acs_agent.chat(query))
            self.logger.info("Query completed")
            return result
        except Exception as exc:  # noqa: BLE001
            err_msg = f"ACSDataAgent error: {exc}"
            self.logger.error(err_msg)
            raise

    def cleanup(self):
        """Compatibility no-op."""
        return None
