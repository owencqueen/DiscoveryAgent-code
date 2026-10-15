"""ACS Agent package."""

from importlib import metadata
from typing import Any

__all__ = [
    "ACSAgent",
    "ACSDataAgent",
    "ACSDataAgentConfig",
    "ACSClient",
    "ACSConfig",
    "ACSDataset",
    "GeographyLevel",
    "quick_download",
    "search_and_download",
    "PUMSClient",
    "PUMSMetadata",
    "VariableMetadata",
    "IpumsClient",
    "IPUMS_TOOLS",
]

try:
    __version__ = metadata.version("acs-agent")
except metadata.PackageNotFoundError:
    __version__ = "0.1.0"


def __getattr__(name: str) -> Any:
    if name in {"ACSAgent"}:
        from .agent import ACSAgent
        return ACSAgent
    if name in {"ACSDataAgent", "ACSDataAgentConfig"}:
        from .tool_agent import ACSDataAgent, ACSDataAgentConfig
        return {"ACSDataAgent": ACSDataAgent, "ACSDataAgentConfig": ACSDataAgentConfig}[name]
    if name in {
        "ACSClient",
        "ACSConfig",
        "ACSDataset",
        "GeographyLevel",
        "quick_download",
        "search_and_download",
    }:
        from .acs_access import (
            ACSClient,
            ACSConfig,
            ACSDataset,
            GeographyLevel,
            quick_download,
            search_and_download,
        )
        return {
            "ACSClient": ACSClient,
            "ACSConfig": ACSConfig,
            "ACSDataset": ACSDataset,
            "GeographyLevel": GeographyLevel,
            "quick_download": quick_download,
            "search_and_download": search_and_download,
        }[name]
    if name in {"PUMSClient", "PUMSMetadata", "VariableMetadata"}:
        from .pums import PUMSClient, PUMSMetadata, VariableMetadata
        return {
            "PUMSClient": PUMSClient,
            "PUMSMetadata": PUMSMetadata,
            "VariableMetadata": VariableMetadata,
        }[name]
    if name in {"IpumsClient", "IPUMS_TOOLS"}:
        from .ipums import IpumsClient, IPUMS_TOOLS
        return {"IpumsClient": IpumsClient, "IPUMS_TOOLS": IPUMS_TOOLS}[name]
    raise AttributeError(f"module 'acs_agent' has no attribute '{name}'")
