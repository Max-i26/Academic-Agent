"""Tool registry providing centralized registration, discovery, and schema cataloging."""

import threading
from typing import Any, Dict, List, Optional

from academic_agent.agent.models import ToolSideEffectPolicy
from academic_agent.agent.protocols import ToolProtocol, ToolRegistryProtocol


class ToolRegistry(ToolRegistryProtocol):
    """Central registry managing academic agent tool discovery and schemas.

    Guarantees:
    - Thread-safe tool registration and lookup via RLock.
    - Strict validation of tool contract, unique names (case-insensitive), and JSON schema.
    - Formatted schema catalogs formatted for LLM tool calling.
    """

    def __init__(self) -> None:
        self._tools: Dict[str, ToolProtocol] = {}
        self._lock = threading.RLock()

    def register(self, tool: ToolProtocol) -> None:
        """Register a tool instance, validating uniqueness and contract compliance.

        Raises:
            TypeError: If tool does not implement ToolProtocol or invalid side_effect_policy.
            ValueError: If tool name is invalid or already registered, or schema is malformed.
        """
        if not isinstance(tool, ToolProtocol):
            raise TypeError(f"Object {tool} does not satisfy ToolProtocol.")

        name = getattr(tool, "name", None)
        if not name or not isinstance(name, str) or not name.strip():
            raise ValueError(f"Tool name must be a non-empty string, got: {name!r}")

        key = name.lower().strip()

        with self._lock:
            if key in self._tools:
                raise ValueError(f"Tool with name '{name}' is already registered.")

            policy = getattr(tool, "side_effect_policy", None)
            if not isinstance(policy, ToolSideEffectPolicy):
                raise TypeError(f"Tool '{name}' must provide a valid ToolSideEffectPolicy, got: {type(policy)}")

            schema = tool.get_schema()
            if not isinstance(schema, dict) or schema.get("type") != "object":
                raise ValueError(
                    f"Tool '{name}' schema must be a dict with type='object', got: {schema}"
                )

            self._tools[key] = tool

    def get(self, name: str) -> Optional[ToolProtocol]:
        """Retrieve registered tool by name (case-insensitive)."""
        if not name or not isinstance(name, str):
            return None
        with self._lock:
            return self._tools.get(name.lower().strip())

    def has(self, name: str) -> bool:
        """Check if tool name is registered."""
        if not name or not isinstance(name, str):
            return False
        with self._lock:
            return name.lower().strip() in self._tools

    def unregister(self, name: str) -> bool:
        """Unregister a tool by name. Returns True if removed, False otherwise."""
        if not name or not isinstance(name, str):
            return False
        with self._lock:
            key = name.lower().strip()
            if key in self._tools:
                del self._tools[key]
                return True
            return False

    def list_tools(self) -> List[ToolProtocol]:
        """List all currently registered tools."""
        with self._lock:
            return list(self._tools.values())

    def get_schemas(self) -> List[Dict[str, Any]]:
        """Return LLM-compatible tool specifications for all registered tools."""
        with self._lock:
            specs = []
            for tool in self._tools.values():
                specs.append({
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.get_schema(),
                })
            return specs
