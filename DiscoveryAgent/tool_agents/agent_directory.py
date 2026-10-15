from importlib import import_module
from typing import Iterator


_AGENT_CLASS_PATHS = {
    "execution_agent": "DiscoveryAgent.tool_agents.execution_agent_cc:ExecutionAgentCC",
    "web_search": "DiscoveryAgent.tool_agents.single_search:SingleSearchAgent",
    "deep_research": "DiscoveryAgent.tool_agents.deep_research:DeepResearchAgent",
    "census_data_agent": "DiscoveryAgent.tool_agents.specialized.census_agent:CensusDataAgent",
    "opentargets_agent": "DiscoveryAgent.tool_agents.specialized.opentargets_agent:OpenTargetsAgent",
}


def _load_symbol(spec: str):
    module_name, symbol_name = spec.split(":", 1)
    module = import_module(module_name)
    return getattr(module, symbol_name)


def get_agent_class(agent_type: str):
    spec = _AGENT_CLASS_PATHS.get(agent_type)
    if spec is None:
        raise KeyError(agent_type)
    return _load_symbol(spec)


class _LazyAgentMap:
    """Resolve agent classes on demand so optional extras stay optional."""

    def __getitem__(self, key: str):
        return get_agent_class(key)

    def get(self, key: str, default=None):
        try:
            return get_agent_class(key)
        except KeyError:
            return default

    def __contains__(self, key: object) -> bool:
        return key in _AGENT_CLASS_PATHS

    def __iter__(self) -> Iterator[str]:
        return iter(_AGENT_CLASS_PATHS)

    def __len__(self) -> int:
        return len(_AGENT_CLASS_PATHS)

    def keys(self):
        return _AGENT_CLASS_PATHS.keys()


STRING_TO_AGENT_MAP = _LazyAgentMap()
