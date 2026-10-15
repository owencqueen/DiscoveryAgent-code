"""
Client wrapping ipumspy for the ACS agent.

Handles the full extract lifecycle (submit -> wait -> download -> read)
inside async tool calls, running blocking ipumspy operations in a thread
executor so they don't block the event loop.
"""

from __future__ import annotations

import asyncio
import logging
import os
import zipfile
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from .collections import (
    MICRODATA_COLLECTIONS,
    COMMON_USA_SAMPLES,
    COMMON_CPS_SAMPLES,
)

logger = logging.getLogger("acs_ipums")

# Maximum time (seconds) to wait for an extract before giving up.
_MICRODATA_TIMEOUT = 1800   # 30 minutes

# Maximum preview rows returned to Claude in a tool result.
_MAX_PREVIEW_ROWS = 30


# ---------------------------------------------------------------------------
# Lazy import helpers – ipumspy is optional and may not be installed.
# ---------------------------------------------------------------------------

def _get_ipumspy():
    """Lazily import ipumspy and return the module."""
    try:
        import ipumspy
        return ipumspy
    except ImportError:
        raise ImportError(
            "ipumspy is required for IPUMS data access. "
            "Install it with: pip install ipumspy"
        )


def _get_readers():
    """Lazily import ipumspy.readers and return the module."""
    try:
        from ipumspy import readers
        return readers
    except ImportError:
        raise ImportError(
            "ipumspy is required for IPUMS data access. "
            "Install it with: pip install ipumspy"
        )


_AUTH_ERROR_KEYWORDS = ["not registered", "registration", "unauthorized", "forbidden", "api key"]
_VALIDATION_ERROR_KEYWORDS = [
    "not found", "invalid", "unrecognized", "not available",
    "does not exist", "bad request", "400",
]


def _is_auth_error(exc: Exception) -> bool:
    """Check if an exception is an authentication/registration error."""
    msg = str(exc).lower()
    return any(kw in msg for kw in _AUTH_ERROR_KEYWORDS)


def _is_validation_error(exc: Exception) -> bool:
    """Check if an exception is a validation/bad-request error."""
    msg = str(exc).lower()
    return any(kw in msg for kw in _VALIDATION_ERROR_KEYWORDS)


def _format_error(exc: Exception, operation: str) -> dict:
    """Format an error dict with retry guidance."""
    error_msg = f"{operation}: {exc}"
    result: dict = {"error": error_msg}
    if _is_auth_error(exc):
        result["do_not_retry"] = True
        result["action_required"] = (
            "This is an authentication or registration error. Do NOT retry this request. "
            "Inform the user they need to register or check their API key."
        )
    elif _is_validation_error(exc):
        result["do_not_retry"] = True
        result["action_required"] = (
            "This is a validation error — the request parameters are incorrect. "
            "Do NOT retry with the same parameters. Check variable names, sample IDs, "
            "and collection names against the IPUMS documentation. Use "
            "ipums_search_samples to discover valid sample IDs."
        )
    return result


def _get_api_client():
    """Create and return an IpumsApiClient with the user's API key."""
    ipumspy = _get_ipumspy()
    api_key = os.getenv("IPUMS_API_KEY")
    if not api_key:
        raise ValueError(
            "IPUMS_API_KEY environment variable is required for IPUMS data access. "
            "Register at https://account.ipums.org/api_keys to get a key."
        )
    return ipumspy.IpumsApiClient(api_key)


# ---------------------------------------------------------------------------
# IpumsClient
# ---------------------------------------------------------------------------

class IpumsClient:
    """
    High-level client for IPUMS data access.

    Wraps ipumspy's synchronous API and exposes async methods that run
    blocking I/O in a thread executor.  Each public method corresponds to
    one agent tool and returns a plain dict ready for JSON serialization.
    """

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir / "ipums_downloads"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    # Tier 1 – instant metadata helpers
    # ------------------------------------------------------------------ #

    async def list_collections(
        self,
        collection_type: str = "all",
    ) -> dict:
        """Return catalogue of available IPUMS collections."""
        result: dict[str, Any] = {}
        if collection_type in ("microdata", "all"):
            result["microdata_collections"] = {
                k: {
                    "label": v["label"],
                    "description": v["description"],
                    "topics": v["topics"],
                    "time_span": v["time_span"],
                }
                for k, v in MICRODATA_COLLECTIONS.items()
            }
        return result

    async def search_samples(
        self,
        collection: str,
    ) -> dict:
        """List available samples for a microdata collection."""
        if collection not in MICRODATA_COLLECTIONS:
            return {"error": f"Unknown collection '{collection}'. "
                    f"Valid: {sorted(MICRODATA_COLLECTIONS.keys())}"}

        # Try the live API first; fall back to hardcoded catalogue.
        try:
            samples = await asyncio.get_event_loop().run_in_executor(
                None, self._sync_get_samples, collection,
            )
            return {
                "collection": collection,
                "label": MICRODATA_COLLECTIONS[collection]["label"],
                "count": len(samples),
                "samples": samples,
                "source": "ipums_api",
            }
        except Exception as exc:
            logger.warning("Failed to fetch samples from API: %s", exc)
            # Fall back to hardcoded common samples
            fallback = self._fallback_samples(collection)
            return {
                "collection": collection,
                "label": MICRODATA_COLLECTIONS[collection]["label"],
                "count": len(fallback),
                "samples": fallback,
                "source": "hardcoded_common_samples",
                "note": (
                    "Live API unavailable; showing common samples only. "
                    "The full list is available at the IPUMS website."
                ),
            }

    async def check_extract_status(
        self,
        collection: str,
        extract_id: int,
    ) -> dict:
        """Check status of a previously submitted extract."""
        try:
            result = await asyncio.get_event_loop().run_in_executor(
                None, self._sync_check_status, collection, extract_id,
            )
            return result
        except Exception as exc:
            return {"error": f"Failed to check extract status: {exc}"}

    async def read_extract_data(
        self,
        ddi_file: str,
        data_file: str | None = None,
        subset: list[str] | None = None,
        head: int = 20,
    ) -> dict:
        """Read a previously downloaded IPUMS extract."""
        try:
            result = await asyncio.get_event_loop().run_in_executor(
                None, self._sync_read_extract, ddi_file, data_file, subset, head,
            )
            return result
        except Exception as exc:
            return {"error": f"Failed to read extract: {exc}"}

    # ------------------------------------------------------------------ #
    # Tier 3 – Microdata (minutes)
    # ------------------------------------------------------------------ #

    async def get_microdata(
        self,
        collection: str,
        samples: list[str],
        variables: list[str],
        description: str | None = None,
        data_format: str = "csv",
        data_structure: str = "rectangular_on_P",
        case_selections: dict[str, list[str]] | None = None,
    ) -> dict:
        """Submit, wait, download, and read a microdata extract."""
        if collection not in MICRODATA_COLLECTIONS:
            return {"error": f"Unknown collection '{collection}'. "
                    f"Valid: {sorted(MICRODATA_COLLECTIONS.keys())}"}
        if not samples:
            return {"error": "At least one sample ID is required."}
        if not variables:
            return {"error": "At least one variable name is required."}

        try:
            result = await asyncio.get_event_loop().run_in_executor(
                None,
                self._sync_microdata_extract,
                collection,
                samples,
                variables,
                description,
                data_format,
                data_structure,
                case_selections,
            )
            return result
        except Exception as exc:
            return _format_error(exc, "Microdata extract failed")

    # ================================================================== #
    #  Synchronous helpers (run in thread executor)
    # ================================================================== #

    def _sync_get_samples(self, collection: str) -> list[dict]:
        client = _get_api_client()
        raw = client.get_all_sample_info(collection)
        samples = []
        if isinstance(raw, dict):
            for sample_id, info in raw.items():
                desc = info if isinstance(info, str) else str(info)
                samples.append({"id": sample_id, "description": desc})
        elif isinstance(raw, list):
            for item in raw:
                if isinstance(item, dict):
                    samples.append({
                        "id": item.get("name", item.get("id", "")),
                        "description": item.get("description", ""),
                    })
                else:
                    samples.append({"id": str(item), "description": ""})
        return samples

    def _fallback_samples(self, collection: str) -> list[dict]:
        mapping = {
            "usa": COMMON_USA_SAMPLES,
            "cps": COMMON_CPS_SAMPLES,
        }
        common = mapping.get(collection, {})
        return [{"id": k, "description": v} for k, v in common.items()]

    def _sync_check_status(self, collection: str, extract_id: int) -> dict:
        client = _get_api_client()
        extract = client.get_extract_by_id(extract_id, collection=collection)
        status = "unknown"
        if hasattr(extract, "status"):
            status = extract.status
        elif isinstance(extract, dict):
            status = extract.get("status", "unknown")
        return {
            "collection": collection,
            "extract_id": extract_id,
            "status": status,
        }

    def _sync_read_extract(
        self,
        ddi_file: str,
        data_file: str | None,
        subset: list[str] | None,
        head: int,
    ) -> dict:
        ipumspy = _get_ipumspy()
        ddi_path = Path(ddi_file)
        if not ddi_path.exists():
            return {"error": f"DDI file not found: {ddi_file}"}

        readers = _get_readers()
        codebook = readers.read_ipums_ddi(ddi_path)

        # Determine data file path
        if data_file:
            dat_path = Path(data_file)
        else:
            dat_path = ddi_path.parent / codebook.file_description.filename
        if not dat_path.exists():
            return {"error": f"Data file not found: {dat_path}"}

        kwargs: dict[str, Any] = {}
        if subset:
            kwargs["subset"] = subset

        df = readers.read_microdata(codebook, dat_path, **kwargs)
        return self._dataframe_to_result(df, head, str(dat_path))

    # ------------------------------------------------------------------ #
    # Microdata extract lifecycle
    # ------------------------------------------------------------------ #

    def _sync_microdata_extract(
        self,
        collection: str,
        samples: list[str],
        variables: list[str],
        description: str | None,
        data_format: str,
        data_structure: str,
        case_selections: dict[str, list[str]] | None,
    ) -> dict:
        ipumspy = _get_ipumspy()
        client = _get_api_client()

        # Map data_structure string to ipumspy dict
        structure_map = {
            "rectangular_on_P": {"rectangular": {"on": "P"}},
            "rectangular_on_H": {"rectangular": {"on": "H"}},
            "hierarchical": {"hierarchical": {}},
        }
        ds = structure_map.get(data_structure, {"rectangular": {"on": "P"}})

        # Build Variable objects with case selections if provided
        if case_selections:
            var_objects = []
            for v in variables:
                if v in case_selections:
                    var_objects.append(
                        ipumspy.Variable(
                            name=v,
                            case_selections={"general": case_selections[v]},
                        )
                    )
                else:
                    var_objects.append(v)
        else:
            var_objects = variables

        extract = ipumspy.MicrodataExtract(
            collection=collection,
            samples=samples,
            variables=var_objects,
            description=description or f"{collection} extract via ACS agent",
            data_format=data_format,
            data_structure=ds,
        )

        logger.info("Submitting %s microdata extract...", collection)
        client.submit_extract(extract)
        extract_id = extract.extract_id
        logger.info(
            "%s extract %s submitted. Waiting for completion (this may take several minutes)...",
            collection,
            extract_id,
        )

        try:
            client.wait_for_extract(
                extract,
                inital_wait_time=2,
                max_wait_time=120,
                timeout=_MICRODATA_TIMEOUT,
            )
        except Exception as exc:
            # Timeout or other failure – return partial info
            return {
                "success": False,
                "extract_id": extract_id,
                "collection": collection,
                "status": "timed_out_or_failed",
                "error": str(exc),
                "note": (
                    f"Extract {extract_id} was submitted but did not complete within "
                    f"the timeout ({_MICRODATA_TIMEOUT}s). You can check its status "
                    f"later with ipums_check_extract_status."
                ),
            }

        logger.info(
            "%s extract %s completed. Downloading...",
            collection,
            extract_id,
        )
        client.download_extract(extract, download_dir=self.output_dir)

        # Find and read the downloaded data
        return self._read_microdata_download(collection, extract_id)

    def _read_microdata_download(self, collection: str, extract_id: int) -> dict:
        """Locate DDI + data files and return a preview."""
        readers = _get_readers()

        # IPUMS files follow the pattern: {collection}_{extract_id:05d}.xml
        # Match specifically on collection + extract_id to avoid grabbing
        # files from other collections that share the download directory.
        expected_prefix = f"{collection}_{extract_id:05d}"
        ddi_candidates = sorted(self.output_dir.glob(f"{expected_prefix}.xml"))

        if not ddi_candidates:
            # Try inside zip files
            for zf in sorted(self.output_dir.glob(f"{expected_prefix}*.zip")):
                extract_dir = self.output_dir / zf.stem
                extract_dir.mkdir(parents=True, exist_ok=True)
                with zipfile.ZipFile(zf, "r") as z:
                    z.extractall(extract_dir)
                ddi_candidates.extend(sorted(extract_dir.glob("*.xml")))

        if not ddi_candidates:
            # Broader fallback: any XML with the collection name
            ddi_candidates = sorted(self.output_dir.glob(f"{collection}_*.xml"))

        if not ddi_candidates:
            return {
                "success": True,
                "extract_id": extract_id,
                "collection": collection,
                "note": "Extract downloaded but no DDI codebook found. "
                        "Check the download directory for files.",
                "download_dir": str(self.output_dir),
            }

        ddi_path = ddi_candidates[-1]
        codebook = readers.read_ipums_ddi(ddi_path)

        # Find the data file — scope search to this specific extract
        data_filename = codebook.file_description.filename
        data_path = ddi_path.parent / data_filename
        if not data_path.exists():
            # Try with .gz extension
            gz_path = ddi_path.parent / (data_filename + ".gz")
            if gz_path.exists():
                data_path = gz_path
            else:
                # Try finding data file matching this extract's prefix
                # (e.g. usa_00005.csv.gz when DDI says usa_00005.dat)
                candidates = (
                    list(ddi_path.parent.glob(f"{expected_prefix}.dat*"))
                    + list(ddi_path.parent.glob(f"{expected_prefix}.csv*"))
                )
                if not candidates:
                    # Broader fallback: any data file with collection name
                    candidates = (
                        list(ddi_path.parent.glob(f"{collection}_*.dat*"))
                        + list(ddi_path.parent.glob(f"{collection}_*.csv*"))
                    )
                if candidates:
                    data_path = candidates[0]
                else:
                    return {
                        "success": True,
                        "extract_id": extract_id,
                        "collection": collection,
                        "ddi_file": str(ddi_path),
                        "note": f"DDI codebook found but data file '{data_filename}' missing.",
                        "download_dir": str(self.output_dir),
                    }

        # Read the data
        try:
            df = readers.read_microdata(codebook, data_path)
        except Exception as exc:
            return {
                "success": False,
                "extract_id": extract_id,
                "collection": collection,
                "ddi_file": str(ddi_path),
                "data_file": str(data_path),
                "error": f"Downloaded but failed to read data: {exc}",
                "do_not_retry": True,
                "note": (
                    "The extract was submitted and downloaded successfully but the data "
                    "could not be parsed. Do NOT re-submit the extract. The data files "
                    "are available on disk. Use ipums_read_extract_data to try reading "
                    "with different options, or inform the user about the file locations."
                ),
            }

        result = self._dataframe_to_result(df, _MAX_PREVIEW_ROWS, str(data_path))
        result.update({
            "success": True,
            "extract_id": extract_id,
            "collection": collection,
            "ddi_file": str(ddi_path),
        })
        return result

    # ------------------------------------------------------------------ #
    # Shared helpers
    # ------------------------------------------------------------------ #

    def _dataframe_to_result(
        self,
        df: pd.DataFrame,
        head: int,
        file_path: str,
    ) -> dict:
        """Convert a DataFrame into a serializable result dict."""
        preview = df.head(head)
        # Convert to records, handling numpy types
        records = []
        for _, row in preview.iterrows():
            record = {}
            for col in preview.columns:
                val = row[col]
                if pd.isna(val):
                    record[col] = None
                elif hasattr(val, "item"):
                    record[col] = val.item()
                else:
                    record[col] = val
            records.append(record)

        return {
            "file_path": file_path,
            "total_rows": len(df),
            "total_columns": len(df.columns),
            "columns": list(df.columns),
            "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
            "preview_rows": len(records),
            "preview": records,
        }
