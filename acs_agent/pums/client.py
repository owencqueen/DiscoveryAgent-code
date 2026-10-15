"""
Async client for the Census Microdata (PUMS) API.

Focuses on bounded retrieval and lightweight tabulations.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Optional, Literal

import aiohttp
from aiolimiter import AsyncLimiter
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from ..acs_access.config import ACSConfig, CENSUS_API_BASE, STATE_ABBREV_TO_FIPS, STATE_TO_FIPS
from .metadata import PUMSMetadata

logger = logging.getLogger("acs_pums")

RecordType = Literal["person", "housing"]


class PUMSClient:
    """Async client for PUMS Microdata API."""

    def __init__(self, config: Optional[ACSConfig] = None):
        self.config = config or ACSConfig()
        self.metadata = PUMSMetadata(self.config)
        self.rate_limiter = AsyncLimiter(self.config.rate_limit, time_period=1.0)
        self.semaphore = asyncio.Semaphore(self.config.max_concurrent_requests)
        self._session: Optional[aiohttp.ClientSession] = None

    async def __aenter__(self) -> "PUMSClient":
        await self._ensure_session()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

    async def _ensure_session(self) -> None:
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=self.config.request_timeout)
            self._session = aiohttp.ClientSession(timeout=timeout)

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()

    # ---------------------------- Public API ---------------------------- #
    async def search_variables(
        self,
        query: str,
        year: int,
        record_type: RecordType,
        sample: str = "acs5",
    ) -> list[VariableMetadata]:
        return await self.metadata.search_async(query, year, record_type, sample)

    async def get_records(
        self,
        year: int,
        record_type: RecordType,
        variables: list[str],
        filters: dict[str, Any],
        geography: dict[str, str],
        limit: int = 5000,
        sample: str = "acs5",
    ) -> list[dict]:
        if not variables:
            raise ValueError("At least one variable is required for PUMS retrieval.")
        if limit <= 0:
            raise ValueError("limit must be a positive integer.")

        normalized_filters = self._normalize_filters(filters or {}, record_type, geography)
        geo_params = self._build_geography_params(geography)
        self._enforce_guardrails(record_type, normalized_filters, geography)

        params = {
            "get": ",".join(self._normalize_variables(variables)),
            **geo_params,
            **normalized_filters,
        }

        url = self._build_url(year, sample)
        data = await self._request(url, params, description="PUMS records")
        rows = self._rows_to_dicts(data)
        if len(rows) > limit:
            return rows[:limit]
        return rows

    async def tabulate(
        self,
        year: int,
        record_type: RecordType,
        metrics: list[str],
        group_by: list[str],
        filters: dict[str, Any],
        geography: dict[str, str],
        weight: str | None = None,
        sample: str = "acs5",
    ) -> list[dict]:
        if not group_by:
            raise ValueError("group_by is required for PUMS tabulations.")

        normalized_filters = self._normalize_filters(filters or {}, record_type, geography)
        geo_params = self._build_geography_params(geography)
        self._enforce_guardrails(record_type, normalized_filters, geography)

        tabulate = self._build_tabulate(metrics, record_type, weight)
        row_vars, col_vars = self._split_group_by(group_by)

        params = {
            "tabulate": tabulate,
            **normalized_filters,
        }
        if geo_params.get("for"):
            params["for"] = geo_params["for"]
        if geo_params.get("in"):
            params["in"] = geo_params["in"]

        for idx, var in enumerate(row_vars, start=1):
            params[f"row+{idx}"] = var
        for idx, var in enumerate(col_vars, start=1):
            params[f"col+{idx}"] = var
        if geo_params.get("for"):
            if not col_vars:
                params["col+for"] = "for"
            else:
                params[f"col+{len(col_vars)+1}"] = "for"

        url = self._build_url(year, sample)
        data = await self._request(url, params, description="PUMS tabulation")
        return self._rows_to_dicts(data)

    # ---------------------------- Internals ---------------------------- #
    def _build_url(self, year: int, sample: str) -> str:
        dataset = "acs/acs1/pums" if sample == "acs1" else "acs/acs5/pums"
        return f"{CENSUS_API_BASE}/{year}/{dataset}"

    def _normalize_variables(self, variables: list[str]) -> list[str]:
        return [v.strip().upper() for v in variables if v and v.strip()]

    def _normalize_filters(
        self,
        filters: dict[str, Any],
        record_type: RecordType,
        geography: dict[str, str],
    ) -> dict[str, str]:
        normalized: dict[str, str] = {}
        for key, value in (filters or {}).items():
            if value is None:
                continue
            normalized_key = key.strip().upper()
            normalized[normalized_key] = self._format_filter_value(value)

        rt_value = "P" if record_type == "person" else "H"
        if "RT" in normalized and normalized["RT"] != rt_value:
            raise ValueError("RT filter conflicts with record_type.")
        normalized["RT"] = rt_value

        if geography and geography.get("puma"):
            normalized.setdefault("PUMA", geography["puma"])

        return normalized

    def _build_geography_params(self, geography: dict[str, str]) -> dict[str, str]:
        if not geography:
            return {}

        allowed = {"nation", "region", "division", "state", "puma"}
        unknown = set(geography.keys()) - allowed
        if unknown:
            raise ValueError(f"Unsupported PUMS geography keys: {sorted(unknown)}")

        if geography.get("nation"):
            return {}

        if geography.get("region"):
            if geography["region"] == "*":
                raise ValueError("Wildcard region is not supported for PUMS queries.")
            return {"for": f"region:{geography['region']}"}
        if geography.get("division"):
            if geography["division"] == "*":
                raise ValueError("Wildcard division is not supported for PUMS queries.")
            return {"for": f"division:{geography['division']}"}
        if geography.get("state"):
            if geography["state"] == "*":
                raise ValueError("Wildcard state is not supported for PUMS queries.")
            state_fips = self._normalize_state(geography["state"])
            return {"for": f"state:{state_fips}"}

        if geography.get("puma"):
            raise ValueError("PUMA geography requires a state code.")

        return {}

    def _normalize_state(self, value: str) -> str:
        val = value.strip().lower()
        if len(val) == 2 and val in STATE_ABBREV_TO_FIPS:
            return STATE_ABBREV_TO_FIPS[val]
        if val in STATE_TO_FIPS:
            return STATE_TO_FIPS[val]
        if val.isdigit() and len(val) == 2:
            return val
        raise ValueError("State must be a 2-digit FIPS, abbreviation, or full name.")

    def _format_filter_value(self, value: Any) -> str:
        if isinstance(value, (list, tuple)) and len(value) == 2:
            return f"{value[0]}:{value[1]}"
        if isinstance(value, dict) and {"min", "max"} <= value.keys():
            return f"{value['min']}:{value['max']}"
        return str(value)

    def _split_group_by(self, group_by: list[str]) -> tuple[list[str], list[str]]:
        variables = self._normalize_variables(group_by)
        if len(variables) == 1:
            return variables, []
        return [variables[0]], variables[1:]

    def _build_tabulate(self, metrics: list[str], record_type: RecordType, weight: str | None) -> str:
        weight_var = weight or ("PWGTP" if record_type == "person" else "WGTP")
        if not metrics:
            return f"weight({weight_var})"

        parts = []
        for metric in metrics:
            if not metric:
                continue
            m = metric.strip()
            if m.lower() in {"count", "weight"}:
                parts.append(f"weight({weight_var})")
            elif m.lower().startswith("avg(") and m.endswith(")"):
                parts.append(m)
            elif m.lower().startswith("avg:"):
                parts.append(f"avg({m.split(':', 1)[1].strip()})")
            else:
                parts.append(f"avg({m})")
        return ",".join(parts)

    def _rows_to_dicts(self, data: Optional[list]) -> list[dict]:
        if not data or len(data) < 2:
            return []
        headers = data[0]
        rows = []
        for row in data[1:]:
            rows.append({headers[i]: row[i] if i < len(row) else None for i in range(len(headers))})
        return rows

    def _enforce_guardrails(
        self,
        record_type: RecordType,
        filters: dict[str, str],
        geography: dict[str, str],
    ) -> None:
        if not geography:
            raise ValueError("PUMS queries require a geography (nation, region, division, state, or PUMA).")

        if geography.get("puma") and not geography.get("state"):
            raise ValueError("PUMA geography must include a state.")
        if geography.get("puma") == "*":
            raise ValueError("Wildcard PUMA is not supported for PUMS queries.")

        is_small_geo = bool(geography.get("puma"))
        restrictive = any(key not in {"RT"} for key in (filters or {}).keys())

        if not is_small_geo and not restrictive:
            raise ValueError("PUMS queries must include a PUMA or at least one restrictive filter.")

        if record_type == "person" and not is_small_geo and not restrictive:
            raise ValueError("Person-level PUMS queries must be filtered or restricted to a PUMA.")

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        retry=retry_if_exception_type((aiohttp.ClientError, asyncio.TimeoutError)),
    )
    async def _request(self, url: str, params: dict, description: str) -> Optional[list]:
        await self._ensure_session()
        params = {k: v for k, v in params.items() if v not in ("", None)}
        if self.config.api_key:
            params["key"] = self.config.api_key

        async with self.semaphore:
            async with self.rate_limiter:
                async with self._session.get(url, params=params) as resp:
                    if resp.status == 204:
                        return None
                    if resp.status == 400:
                        error_text = await resp.text()
                        raise ValueError(f"Bad request for {description}: {error_text}")
                    if resp.status != 200:
                        raise ValueError(f"HTTP {resp.status} for {description}")
                    return await resp.json()
