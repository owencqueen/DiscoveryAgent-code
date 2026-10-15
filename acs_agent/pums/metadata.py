"""
PUMS variable metadata and data dictionary helpers.

Provides variable descriptions, values/codes, and record type availability.
Uses cached downloads of Census metadata sources.
"""

from __future__ import annotations

import asyncio
import csv
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Optional

import aiohttp

from ..acs_access.cache import MetadataCache
from ..acs_access.config import ACSConfig, CENSUS_API_BASE

logger = logging.getLogger("acs_pums")


@dataclass
class VariableMetadata:
    """Metadata about a PUMS variable."""

    name: str
    label: Optional[str] = None
    concept: Optional[str] = None
    predicate_type: Optional[str] = None
    record_type: Optional[str] = None  # "person", "housing", "both", "unknown"
    values: dict[str, str] = field(default_factory=dict)
    suggested_weight: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "label": self.label,
            "concept": self.concept,
            "predicate_type": self.predicate_type,
            "record_type": self.record_type,
            "values": self.values,
            "suggested_weight": self.suggested_weight,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "VariableMetadata":
        return cls(
            name=data.get("name", ""),
            label=data.get("label"),
            concept=data.get("concept"),
            predicate_type=data.get("predicate_type"),
            record_type=data.get("record_type"),
            values=data.get("values") or {},
            suggested_weight=data.get("suggested_weight"),
        )


class PUMSMetadata:
    """Fetch and cache PUMS variable metadata."""

    def __init__(self, config: Optional[ACSConfig] = None):
        self.config = config or ACSConfig()
        self.cache = MetadataCache(self.config.cache_path, self.config.cache_ttl_days)

    # ---------------------------- Public API ---------------------------- #
    def search(self, query: str, year: int, record_type: str, sample: str = "acs5") -> list[VariableMetadata]:
        """Synchronous search wrapper."""
        return self._run_sync(self.search_async(query, year, record_type, sample))

    def get(self, variable: str, year: int, record_type: str, sample: str = "acs5") -> VariableMetadata:
        """Synchronous get wrapper."""
        return self._run_sync(self.get_async(variable, year, record_type, sample))

    async def search_async(
        self,
        query: str,
        year: int,
        record_type: str,
        sample: str = "acs5",
    ) -> list[VariableMetadata]:
        metadata = await self._load_metadata(year, sample)
        query_norm = (query or "").lower().strip()
        if not query_norm:
            return []

        terms = [t for t in re.split(r"\s+", query_norm) if t]
        results = []
        for meta in metadata.values():
            if not self._record_type_match(meta, record_type):
                continue
            haystack = " ".join(
                part
                for part in [
                    meta.name,
                    meta.label or "",
                    meta.concept or "",
                ]
                if part
            ).lower()
            if all(term in haystack for term in terms):
                results.append(meta)
        return results

    async def get_async(
        self,
        variable: str,
        year: int,
        record_type: str,
        sample: str = "acs5",
    ) -> VariableMetadata:
        metadata = await self._load_metadata(year, sample)
        meta = metadata.get(variable.upper())
        if not meta:
            return VariableMetadata(name=variable.upper(), record_type="unknown")
        if not self._record_type_match(meta, record_type):
            return VariableMetadata(name=variable.upper(), record_type="unknown")
        return meta

    # ---------------------------- Internals ---------------------------- #
    def _run_sync(self, coro):
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

    async def _load_metadata(self, year: int, sample: str) -> dict[str, VariableMetadata]:
        cache_key = ("pums_metadata", year, sample)
        cached = self.cache.get(*cache_key)
        if cached:
            return {k: VariableMetadata.from_dict(v) for k, v in cached.items()}

        for candidate_year in self._candidate_years(year):
            variables = await self._fetch_variables_json(candidate_year, sample)
            ddict = await self._fetch_data_dictionary(candidate_year, sample)
            if not variables and not ddict:
                continue

            if candidate_year != year:
                logger.warning(
                    "PUMS metadata for %s (%s) not available; falling back to %s.",
                    year,
                    sample,
                    candidate_year,
                )

            merged = self._merge_metadata(variables, ddict)
            self.cache.set({k: v.to_dict() for k, v in merged.items()}, *cache_key)
            return merged

        logger.warning("No PUMS metadata found for year %s (%s).", year, sample)
        self.cache.set({}, *cache_key)
        return {}

    def _candidate_years(self, year: int, max_back: int = 5) -> list[int]:
        candidates: list[int] = []
        for offset in range(max_back + 1):
            candidate = year - offset
            if candidate <= 0:
                break
            candidates.append(candidate)
        return candidates

    def _merge_metadata(
        self,
        variables: dict[str, dict[str, Any]],
        ddict: dict[str, dict[str, Any]],
    ) -> dict[str, VariableMetadata]:
        merged: dict[str, VariableMetadata] = {}
        for var_name, info in variables.items():
            merged[var_name] = VariableMetadata(
                name=var_name,
                label=info.get("label"),
                concept=info.get("concept"),
                predicate_type=info.get("predicateType"),
                values=info.get("values") or {},
                suggested_weight=info.get("suggested-weight"),
                record_type="unknown",
            )

        if ddict:
            for var_name, dd in ddict.items():
                meta = merged.get(var_name)
                if not meta:
                    meta = VariableMetadata(name=var_name, record_type="unknown")
                    merged[var_name] = meta
                if dd.get("label"):
                    meta.label = meta.label or dd.get("label")
                if dd.get("record_type"):
                    meta.record_type = dd.get("record_type")
                if dd.get("values"):
                    meta.values = meta.values or dd.get("values", {})

        return merged

    async def _fetch_variables_json(self, year: int, sample: str) -> dict[str, dict[str, Any]]:
        url = f"{CENSUS_API_BASE}/{year}/{self._dataset_path(sample)}/variables.json"
        cache_key = ("pums_variables_json", year, sample)
        cached = self.cache.get(*cache_key)
        if cached:
            return cached

        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=60)) as session:
            async with session.get(url) as resp:
                if resp.status != 200:
                    logger.warning("Failed to fetch variables.json: %s (HTTP %s)", url, resp.status)
                    return {}
                payload = await resp.json()

        variables = payload.get("variables", {})
        self.cache.set(variables, *cache_key)
        return variables

    async def _fetch_data_dictionary(self, year: int, sample: str) -> dict[str, dict[str, Any]]:
        cache_key = ("pums_data_dictionary", year, sample)
        cached = self.cache.get(*cache_key)
        if cached:
            return cached

        urls = self._candidate_dictionary_urls(year, sample)
        for url in urls:
            try:
                data = await self._download_csv(url)
                if not data:
                    continue
                self.cache.set(data, *cache_key)
                return data
            except Exception as exc:  # noqa: BLE001
                logger.debug("Failed to parse PUMS data dictionary %s: %s", url, exc)
                continue

        self.cache.set({}, *cache_key)
        return {}

    def _candidate_dictionary_urls(self, year: int, sample: str) -> list[str]:
        base = "https://www2.census.gov/programs-surveys/acs/tech_docs/pums/data_dict"
        urls = []
        if sample == "acs5":
            start_year = year - 4
            urls.append(f"{base}/PUMS_Data_Dictionary_{start_year}-{year}.csv")
        urls.append(f"{base}/PUMS_Data_Dictionary_{year}.csv")
        return urls

    async def _download_csv(self, url: str) -> dict[str, dict[str, Any]]:
        raw = await self._fetch_text(url)
        if not raw:
            return {}
        reader = csv.DictReader(raw.splitlines())
        if not reader.fieldnames:
            return {}

        field_map = self._map_headers(reader.fieldnames)
        if not field_map.get("variable"):
            return {}

        results: dict[str, dict[str, Any]] = {}
        for row in reader:
            var = (row.get(field_map["variable"]) or "").strip().upper()
            if not var:
                continue
            record_type = None
            if field_map.get("record_type"):
                record_type = self._normalize_record_type(row.get(field_map["record_type"]))
            label = None
            if field_map.get("label"):
                label = (row.get(field_map["label"]) or "").strip()

            entry = results.setdefault(var, {"values": {}, "record_type": record_type, "label": label})
            if record_type and not entry.get("record_type"):
                entry["record_type"] = record_type
            if label and not entry.get("label"):
                entry["label"] = label

            value_col = field_map.get("value")
            value_label_col = field_map.get("value_label")
            if value_col and value_label_col:
                value = (row.get(value_col) or "").strip()
                value_label = (row.get(value_label_col) or "").strip()
                if value and value_label:
                    entry["values"][value] = value_label

        return results

    async def _fetch_text(self, url: str) -> str:
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=60)) as session:
                async with session.get(url) as resp:
                    if resp.status != 200:
                        return ""
                    return await resp.text()
        except aiohttp.ClientError:
            return ""

    def _map_headers(self, headers: list[str]) -> dict[str, str]:
        normalized = {self._normalize_header(h): h for h in headers}

        def pick(candidates: list[str]) -> Optional[str]:
            for cand in candidates:
                if cand in normalized:
                    return normalized[cand]
            return None

        return {
            "variable": pick(["variable", "variablename", "var", "name", "variableid", "variable_name"]),
            "record_type": pick(["recordtype", "record", "rectype", "record_type"]),
            "label": pick(["label", "description", "variabledescription", "variablelabel"]),
            "value": pick(["value", "code", "val", "valuecode", "codevalue"]),
            "value_label": pick(["valuelabel", "value_label", "valuedescription", "codedescription", "value description"]),
        }

    def _normalize_header(self, header: str) -> str:
        return re.sub(r"[^a-z0-9]+", "", (header or "").lower())

    def _normalize_record_type(self, value: Optional[str]) -> Optional[str]:
        if not value:
            return None
        val = value.strip().lower()
        if val in {"p", "person", "persons", "p record", "p-record"}:
            return "person"
        if val in {"h", "housing", "household", "h record", "h-record"}:
            return "housing"
        if "person" in val and "housing" in val:
            return "both"
        return None

    def _record_type_match(self, meta: VariableMetadata, record_type: str) -> bool:
        if meta.record_type in (None, "unknown", ""):
            return True
        if meta.record_type == "both":
            return True
        return meta.record_type == record_type

    def _dataset_path(self, sample: str) -> str:
        return "acs/acs1/pums" if sample == "acs1" else "acs/acs5/pums"
