from typing import Any, Optional, Dict

from dataclasses import dataclass

@dataclass
class AgentToolConfig:
    """Configuration for an agent tool."""
    agent_name: str = None
    tag: str = None
    tool_prompt: str = None

class BaseAgent:
    def __init__(self, config: Any, logger: Optional[Any] = None):
        self.config = config
        self.logger = logger

    def config_to_prompt(self) -> str:
        prompt = \
f"""Agent: { self.config.agent_name}
Tag: {self.config.tag}
{self.config.tool_prompt}"""
        return prompt

    def execute(self, query: str):
        raise NotImplementedError("Subclasses must implement this method")
    
    def cleanup(self):
        """Cleanup function that clears any internals - only implement if needed"""
        pass
    
    def save_state(self) -> Dict[str, Any]:
        """
        Save the current state of the agent for checkpointing.
        
        Returns:
            Dict[str, Any]: Serializable state dictionary
        """
        # Default implementation saves basic config info
        return {
            'agent_type': type(self).__name__,
            'config_agent_name': getattr(self.config, 'agent_name', None),
            'config_tag': getattr(self.config, 'tag', None)
        }
    
    def restore_state(self, state: Dict[str, Any]) -> bool:
        """
        Restore the agent state from checkpoint data.
        
        Args:
            state: State dictionary from checkpoint
            
        Returns:
            bool: True if restoration was successful
        """
        # Default implementation does nothing but returns success
        # Subclasses should override to restore their specific state
        return True
    
    def get_state_schema(self) -> Dict[str, str]:
        """
        Get the schema/description of state fields for validation.
        
        Returns:
            Dict[str, str]: Mapping of field names to descriptions
        """
        return {
            'agent_type': 'Class name of the agent',
            'config_agent_name': 'Agent name from config',
            'config_tag': 'Agent tag from config'
        }