"""
Token counting utilities for estimating context token usage.

Uses Anthropic's count_tokens API when available; otherwise falls back to a
simple character-based approximation.
"""
import logging
from typing import List, Dict, Any, Optional

def _approximate_tokens_by_chars(text: str) -> int:
    """Fallback heuristic: ~4 characters per token, rounded up."""
    if not text:
        return 0
    # Clamp to at least 1 for non-empty content
    return max(1, (len(text) + 3) // 4)


def flatten_messages_for_token_count(messages: List[Dict[str, Any]]) -> str:
    """Flatten a messages list to a single string for approximate token counting."""
    parts: List[str] = []
    for m in messages:
        role = m.get("role", "")
        content = m.get("content", "")
        # If content is not str (e.g., structured), coerce to str safely
        if not isinstance(content, str):
            try:
                content = str(content)
            except Exception:
                content = ""
        parts.append(f"{role}: {content}")
    return "\n".join(parts)


def count_anthropic_tokens_via_api(client: Any, model_name: str,
                                   messages: List[Dict[str, Any]],
                                   system: Optional[str] = None) -> Optional[int]:
    """Try Anthropic SDK's token counting if available; return None if unsupported or on error."""
    # Suppress noisy INFO HTTP logs from the Anthropic client during token counting
    suppressed_loggers = ["anthropic", "anthropic._client", "httpx", "httpcore"]
    previous_levels = {}
    try:
        for name in suppressed_loggers:
            logger = logging.getLogger(name)
            previous_levels[name] = logger.getEffectiveLevel()
            logger.setLevel(logging.WARNING)

        # Some SDK versions support messages.count_tokens
        if hasattr(client, "messages") and hasattr(client.messages, "count_tokens"):
            kwargs: Dict[str, Any] = {"model": model_name, "messages": messages}
            if system is not None:
                kwargs["system"] = system
            resp = client.messages.count_tokens(**kwargs)
            # SDK may return object with .input_tokens or dict
            if hasattr(resp, "input_tokens"):
                return int(getattr(resp, "input_tokens"))
            if isinstance(resp, dict) and "input_tokens" in resp:
                return int(resp["input_tokens"])
        return None
    except Exception:
        return None
    finally:
        # Restore logger levels
        for name, level in previous_levels.items():
            logging.getLogger(name).setLevel(level)


def count_anthropic_tokens_approx(model_name: str,
                                  messages: List[Dict[str, Any]],
                                  system: Optional[str]) -> int:
    """Approximate Anthropic tokens with a 4-characters-per-token heuristic."""
    combined = []
    if system:
        combined.append(f"system: {system}")
    combined.append(flatten_messages_for_token_count(messages))
    text = "\n".join(combined)
    return _approximate_tokens_by_chars(text)


def count_anthropic_tokens(client: Any, model_name: str,
                           messages: List[Dict[str, Any]],
                           system: Optional[str] = None) -> Dict[str, Any]:
    """Return token count with method metadata for Anthropic."""
    via_api = count_anthropic_tokens_via_api(client, model_name, messages, system)
    if via_api is not None:
        return {"input_tokens": via_api, "method": "anthropic_api"}
    approx = count_anthropic_tokens_approx(model_name, messages, system)
    return {"input_tokens": approx, "method": "approximation"} 