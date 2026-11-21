"""Unit tests for Stage 6.2 ToolRegistry."""

from typing import Any, Dict
import pytest
from pydantic import BaseModel, Field

from academic_agent.agent.models import ToolResult, ToolSideEffectPolicy
from academic_agent.agent.protocols import ToolProtocol, ToolRegistryProtocol
from academic_agent.agent.registry import ToolRegistry


class DummyArgs(BaseModel):
    query: str = Field(min_length=1)
    limit: int = Field(default=5, ge=1)


class ValidDummyTool:
    """Mock tool conforming to ToolProtocol."""

    def __init__(self, name: str = "dummy_tool", timeout: float = 10.0) -> None:
        self._name = name
        self._timeout = timeout

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return "A valid dummy tool for registry testing."

    @property
    def side_effect_policy(self) -> ToolSideEffectPolicy:
        return ToolSideEffectPolicy(read_only=True, timeout_seconds=self._timeout)

    def get_schema(self) -> Dict[str, Any]:
        return DummyArgs.model_json_schema()

    def execute(self, arguments: Dict[str, Any]) -> ToolResult:
        return ToolResult(
            call_id="call_test",
            tool_name=self.name,
            success=True,
            output={"echo": arguments},
        )


class TestToolRegistry:
    """Test suite verifying ToolRegistry discovery, validation, and schema cataloging."""

    def test_protocol_conformance(self):
        """ToolRegistry satisfies ToolRegistryProtocol."""
        registry = ToolRegistry()
        assert isinstance(registry, ToolRegistryProtocol)

    def test_tool_registration_and_retrieval(self):
        """Registering a tool makes it retrievable by exact and case-insensitive name."""
        registry = ToolRegistry()
        tool = ValidDummyTool(name="Academic_Search")

        registry.register(tool)
        assert registry.has("academic_search") is True
        assert registry.has("ACADEMIC_SEARCH") is True
        assert registry.has("Academic_Search") is True
        assert registry.has("non_existent") is False

        retrieved = registry.get("academic_search")
        assert retrieved is tool
        assert retrieved.name == "Academic_Search"

    def test_duplicate_registration_rejected(self):
        """Registering two tools with the same name (case-insensitive) raises ValueError."""
        registry = ToolRegistry()
        tool1 = ValidDummyTool(name="search")
        tool2 = ValidDummyTool(name="SEARCH")

        registry.register(tool1)
        with pytest.raises(ValueError, match="already registered"):
            registry.register(tool2)

    def test_invalid_tool_rejections(self):
        """Registry strictly rejects non-protocols, invalid names, missing policies, and malformed schemas."""
        registry = ToolRegistry()

        # Non-tool object
        with pytest.raises(TypeError, match="does not satisfy ToolProtocol"):
            registry.register("not_a_tool")

        # Invalid name
        class MissingNameTool(ValidDummyTool):
            @property
            def name(self) -> str:
                return "   "

        with pytest.raises(ValueError, match="Tool name must be a non-empty string"):
            registry.register(MissingNameTool())

        # Invalid policy
        class BadPolicyTool(ValidDummyTool):
            @property
            def side_effect_policy(self) -> Any:
                return "not_a_policy"

        with pytest.raises(TypeError, match="ToolSideEffectPolicy"):
            registry.register(BadPolicyTool())

        # Malformed schema (must be type='object')
        class BadSchemaTool(ValidDummyTool):
            def get_schema(self) -> Dict[str, Any]:
                return {"type": "array"}

        with pytest.raises(ValueError, match="type='object'"):
            registry.register(BadSchemaTool())

    def test_tool_listing_and_schema_catalog(self):
        """Registry lists all tools and exports LLM-compatible schema specifications."""
        registry = ToolRegistry()
        tool1 = ValidDummyTool(name="tool_a")
        tool2 = ValidDummyTool(name="tool_b")

        registry.register(tool1)
        registry.register(tool2)

        tools = registry.list_tools()
        assert len(tools) == 2
        assert {t.name for t in tools} == {"tool_a", "tool_b"}

        schemas = registry.get_schemas()
        assert len(schemas) == 2
        assert schemas[0]["name"] == "tool_a"
        assert schemas[0]["parameters"]["type"] == "object"
        assert "query" in schemas[0]["parameters"]["properties"]

    def test_tool_unregistration(self):
        """Unregister removes tool and returns True; removing non-existent returns False."""
        registry = ToolRegistry()
        tool = ValidDummyTool(name="temp_tool")

        registry.register(tool)
        assert registry.has("temp_tool") is True

        assert registry.unregister("temp_tool") is True
        assert registry.has("temp_tool") is False
        assert registry.unregister("temp_tool") is False
