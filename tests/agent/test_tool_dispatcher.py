"""Unit tests for Stage 6.2 ToolDispatcher enforcing execution boundaries and safety rules."""

import time
from typing import Any, Dict, Tuple
from pydantic import BaseModel, Field

from academic_agent.agent.budget import ExecutionBudgetTracker
from academic_agent.agent.dispatcher import ToolDispatcher
from academic_agent.agent.models import (
    BudgetConfig,
    ToolCall,
    ToolErrorCategory,
    ToolResult,
    ToolSideEffectPolicy,
)
from academic_agent.agent.protocols import ToolDispatcherProtocol
from academic_agent.agent.registry import ToolRegistry


class SearchArgs(BaseModel):
    query: str = Field(min_length=2)
    top_k: int = Field(default=3, ge=1, le=10)


class MockSearchTool:
    """Mock search tool with schema validation."""

    name = "mock_search"
    description = "Mock search tool."
    side_effect_policy = ToolSideEffectPolicy(read_only=True, timeout_seconds=5.0)

    def get_schema(self) -> Dict[str, Any]:
        return SearchArgs.model_json_schema()

    def validate_arguments(self, arguments: Dict[str, Any]) -> Tuple[bool, str | None]:
        try:
            SearchArgs.model_validate(arguments)
            return True, None
        except Exception as e:
            return False, str(e)

    def execute(self, arguments: Dict[str, Any]) -> ToolResult:
        return ToolResult(
            call_id="",
            tool_name=self.name,
            success=True,
            output={"results": [f"Result for {arguments['query']}"]},
        )


class CrashingTool:
    """Mock tool that raises an unexpected internal exception."""

    name = "crashing_tool"
    description = "A tool that crashes."
    side_effect_policy = ToolSideEffectPolicy(read_only=True, timeout_seconds=5.0)

    def get_schema(self) -> Dict[str, Any]:
        return {"type": "object", "properties": {}}

    def execute(self, arguments: Dict[str, Any]) -> ToolResult:
        raise RuntimeError("Fatal database connection crash!")


class SlowHangingTool:
    """Mock tool that hangs longer than its configured timeout."""

    name = "slow_tool"
    description = "A tool that sleeps."
    side_effect_policy = ToolSideEffectPolicy(read_only=True, timeout_seconds=0.08)  # 80ms timeout

    def get_schema(self) -> Dict[str, Any]:
        return {"type": "object", "properties": {}}

    def execute(self, arguments: Dict[str, Any]) -> ToolResult:
        time.sleep(0.20)  # Sleep 200ms
        return ToolResult(call_id="", tool_name=self.name, success=True, output="Done")


class TestToolDispatcher:
    """Test suite validating ToolDispatcher three non-negotiable rules and budget pre-enforcement."""

    def setup_method(self):
        self.registry = ToolRegistry()
        self.registry.register(MockSearchTool())
        self.registry.register(CrashingTool())
        self.registry.register(SlowHangingTool())
        self.dispatcher = ToolDispatcher(registry=self.registry, max_workers=2)

    def teardown_method(self):
        self.dispatcher.shutdown(wait=False)

    def test_protocol_conformance(self):
        """Dispatcher satisfies ToolDispatcherProtocol."""
        assert isinstance(self.dispatcher, ToolDispatcherProtocol)

    def test_successful_dispatch(self):
        """Valid tool call executes and returns successful ToolResult with measured latency."""
        call = ToolCall(call_id="c1", tool_name="mock_search", arguments={"query": "differencing", "top_k": 3})
        res = self.dispatcher.dispatch(call)

        assert res.success is True
        assert res.call_id == "c1"
        assert res.tool_name == "mock_search"
        assert res.output["results"] == ["Result for differencing"]
        assert res.latency_ms > 0.0
        assert res.error_message is None

    def test_unregistered_tool_returns_not_found(self):
        """Attempting to dispatch a non-existent tool returns NOT_FOUND error category."""
        call = ToolCall(call_id="c2", tool_name="unknown_tool", arguments={})
        res = self.dispatcher.dispatch(call)

        assert res.success is False
        assert res.error_category == ToolErrorCategory.NOT_FOUND
        assert "not registered" in res.error_message

    def test_rule_1_schema_validation_rejection_without_executing(self):
        """Non-Negotiable Rule 1: Invalid arguments are rejected with VALIDATION_ERROR prior to execution."""
        # Query too short (min_length=2 required)
        call = ToolCall(call_id="c3", tool_name="mock_search", arguments={"query": "a", "top_k": 3})
        res = self.dispatcher.dispatch(call)

        assert res.success is False
        assert res.error_category == ToolErrorCategory.VALIDATION_ERROR
        assert "Schema validation failed" in res.error_message

        # Invalid top_k type or range
        call2 = ToolCall(call_id="c4", tool_name="mock_search", arguments={"query": "valid query", "top_k": 99})
        res2 = self.dispatcher.dispatch(call2)
        assert res2.success is False
        assert res2.error_category == ToolErrorCategory.VALIDATION_ERROR

    def test_rule_3_exception_containment(self):
        """Non-Negotiable Rule 3A: Tool exceptions are caught and converted to EXECUTION_ERROR observations."""
        call = ToolCall(call_id="c5", tool_name="crashing_tool", arguments={})
        res = self.dispatcher.dispatch(call)

        assert res.success is False
        assert res.error_category == ToolErrorCategory.EXECUTION_ERROR
        assert "Fatal database connection crash!" in res.error_message
        assert res.latency_ms >= 0.0

    def test_rule_3_timeout_containment(self):
        """Non-Negotiable Rule 3B: Exceeding tool side_effect_policy.timeout_seconds returns TIMEOUT."""
        call = ToolCall(call_id="c6", tool_name="slow_tool", arguments={})
        res = self.dispatcher.dispatch(call)

        assert res.success is False
        assert res.error_category == ToolErrorCategory.TIMEOUT
        assert "timed out after" in res.error_message

    def test_budget_pre_enforcement_duplicate_interception(self):
        """Dispatcher checks tracker for duplicate calls and intercepts with PERMISSION_DENIED."""
        tracker = ExecutionBudgetTracker(config=BudgetConfig(detect_duplicate_calls=True))
        call1 = ToolCall(call_id="c7", tool_name="mock_search", arguments={"query": "AR(1)", "top_k": 2})
        call2 = ToolCall(call_id="c8", tool_name="mock_search", arguments={"query": "AR(1)", "top_k": 2})

        res1 = self.dispatcher.dispatch(call1, budget_tracker=tracker)
        assert res1.success is True

        res2 = self.dispatcher.dispatch(call2, budget_tracker=tracker)
        assert res2.success is False
        assert res2.error_category == ToolErrorCategory.PERMISSION_DENIED
        assert "Duplicate tool invocation rejected" in res2.error_message

    def test_budget_pre_enforcement_max_invocations_exceeded(self):
        """Dispatcher checks tracker invocation limits and blocks call with PERMISSION_DENIED."""
        tracker = ExecutionBudgetTracker(config=BudgetConfig(max_tool_invocations=1))
        call1 = ToolCall(call_id="c9", tool_name="mock_search", arguments={"query": "q1"})
        call2 = ToolCall(call_id="c10", tool_name="mock_search", arguments={"query": "q2"})

        res1 = self.dispatcher.dispatch(call1, budget_tracker=tracker)
        assert res1.success is True

        res2 = self.dispatcher.dispatch(call2, budget_tracker=tracker)
        assert res2.success is False
        assert res2.error_category == ToolErrorCategory.PERMISSION_DENIED
        assert "Maximum tool invocations reached" in res2.error_message

    def test_budget_pre_enforcement_remaining_time_bounding(self):
        """Global deadline smaller than tool timeout short-circuits execution with TIMEOUT."""
        # Tracker configured with only 0.05s timeout
        tracker = ExecutionBudgetTracker(config=BudgetConfig(timeout_seconds=0.05))
        time.sleep(0.06)  # Elapsed beyond tracker budget

        call = ToolCall(call_id="c11", tool_name="mock_search", arguments={"query": "late query"})
        res = self.dispatcher.dispatch(call, budget_tracker=tracker)

        assert res.success is False
        assert res.error_category == ToolErrorCategory.TIMEOUT
        assert "Execution budget exceeded before tool dispatch" in res.error_message
