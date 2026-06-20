"""ToolRegistry — maps tool names to callables and builds the tools prompt."""

from __future__ import annotations

import json
import logging
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


class ToolRegistrationError(ValueError):
    """Raised when a tool cannot be registered."""


class ToolDispatchError(RuntimeError):
    """Raised when a tool call fails."""


class _ToolEntry:
    __slots__ = ("fn", "description", "params_schema")

    def __init__(self, fn: Callable, description: str, params_schema: Dict[str, Any]) -> None:
        self.fn = fn
        self.description = description
        self.params_schema = params_schema


class ToolRegistry:
    """Registry of callable tools available to the ReAct agent.

    Each tool is registered with:
    - name: identifier the LLM uses in "Action:" lines
    - fn: callable that accepts keyword arguments and returns a str observation
    - description: natural-language description injected into the system prompt
    - params_schema: JSON Schema describing the expected Action Input JSON
    """

    def __init__(self) -> None:
        self._tools: Dict[str, _ToolEntry] = {}

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(
        self,
        name: str,
        fn: Callable[..., str],
        description: str,
        params_schema: Optional[Dict[str, Any]] = None,
    ) -> None:
        if not name or not name.strip():
            raise ToolRegistrationError("Tool name cannot be empty")
        if name in self._tools:
            logger.warning("Tool '%s' is already registered; overwriting.", name)
        self._tools[name] = _ToolEntry(fn, description, params_schema or {})
        logger.debug("Registered tool: %s", name)

    def unregister(self, name: str) -> None:
        self._tools.pop(name, None)

    # ------------------------------------------------------------------
    # Dispatch
    # ------------------------------------------------------------------

    def dispatch(self, tool_name: str, tool_input: Dict[str, Any]) -> str:
        """Call a tool and return its observation string.

        Args:
            tool_name: The name registered via register().
            tool_input: Keyword arguments forwarded to the tool function.

        Returns:
            Natural-language observation string.

        Raises:
            ToolDispatchError: If the tool is not found or raises an exception.
        """
        entry = self._tools.get(tool_name)
        if entry is None:
            available = ", ".join(sorted(self._tools)) or "none"
            raise ToolDispatchError(
                f"Unknown tool '{tool_name}'. Available tools: {available}"
            )
        try:
            result = entry.fn(**tool_input)
            return str(result)
        except Exception as exc:
            raise ToolDispatchError(
                f"Tool '{tool_name}' raised an error: {exc}"
            ) from exc

    # ------------------------------------------------------------------
    # Prompt building
    # ------------------------------------------------------------------

    def get_tools_prompt(self) -> str:
        """Return a newline-separated description of all registered tools."""
        if not self._tools:
            return "(no tools registered)"
        parts: List[str] = []
        for name, entry in self._tools.items():
            schema_str = json.dumps(entry.params_schema, ensure_ascii=False)
            parts.append(
                f"- {name}: {entry.description}\n"
                f"  Parameters (JSON): {schema_str}"
            )
        return "\n".join(parts)

    def list_tool_names(self) -> List[str]:
        return list(self._tools.keys())

    def __contains__(self, name: str) -> bool:
        return name in self._tools
