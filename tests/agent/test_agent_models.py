"""Unit tests for Stage 6.1 Agent and Tool Data Models."""

import pytest
from academic_agent.agent.models import (
    AgentExecutionState,
    AgentTerminalState,
    AgentStep,
    AgentTrace,
    BudgetConfig,
    BudgetConsumption,
    ProposedAction,
    TokenAccounting,
    ToolCall,
    ToolErrorCategory,
    ToolResult,
    ToolSideEffectPolicy,
)


class TestAgentDataModels:
    """Test suite validating core Phase 6 data structures and constraints."""

    def test_state_enumerations(self):
        """Execution states and terminal states must be disjoint and well-defined."""
        exec_values = {s.value for s in AgentExecutionState}
        term_values = {s.value for s in AgentTerminalState}

        assert len(exec_values.intersection(term_values)) == 0
        assert "INITIALIZED" in exec_values
        assert "PLANNING" in exec_values
        assert "TERMINATED_SUCCESS" in term_values
        assert "TERMINATED_ABSTENTION" in term_values
        assert "TERMINATED_BUDGET_EXCEEDED" in term_values

    def test_tool_side_effect_policy_defaults_and_validation(self):
        """Default policy must be strictly read-only and idempotent."""
        policy = ToolSideEffectPolicy()
        assert policy.read_only is True
        assert policy.idempotent is True
        assert policy.mutating is False
        assert policy.requires_confirmation is False
        assert policy.timeout_seconds == 15.0

        # Conflict: mutating cannot be True if read_only is True
        with pytest.raises(ValueError, match="Tool cannot be both read_only and mutating"):
            ToolSideEffectPolicy(read_only=True, mutating=True)

    def test_tool_call_and_tool_result(self):
        """ToolCall and ToolResult serialize and correlate correctly."""
        call = ToolCall(
            call_id="call_123",
            tool_name="academic_search",
            arguments={"query": "stationarity", "top_k": 3},
        )
        assert call.call_id == "call_123"
        assert call.arguments["top_k"] == 3

        res_success = ToolResult(
            call_id="call_123",
            tool_name="academic_search",
            success=True,
            output={"results": []},
            latency_ms=12.5,
        )
        assert res_success.success is True
        assert res_success.error_category is None

        res_fail = ToolResult(
            call_id="call_123",
            tool_name="academic_search",
            success=False,
            error_message="Query timed out",
            error_category=ToolErrorCategory.TIMEOUT,
            latency_ms=15000.0,
        )
        assert res_fail.success is False
        assert res_fail.error_category == ToolErrorCategory.TIMEOUT

    def test_token_accounting_cumulative_addition(self):
        """TokenAccounting adds usage accurately and preserves exactness status."""
        acc1 = TokenAccounting(prompt_tokens=100, completion_tokens=50, total_tokens=150, is_exact=True)
        acc2 = acc1.add(prompt=50, completion=25, is_exact=True)

        assert acc2.prompt_tokens == 150
        assert acc2.completion_tokens == 75
        assert acc2.total_tokens == 225
        assert acc2.is_exact is True

        # If any step is estimated, cumulative exactness flips to False
        acc3 = acc2.add(prompt=20, completion=10, is_exact=False)
        assert acc3.total_tokens == 255
        assert acc3.is_exact is False

    def test_agent_step_and_trace_construction(self):
        """AgentStep and AgentTrace record complete audit history."""
        step = AgentStep(
            step_index=0,
            state="PLANNING",
            thought="Student question requires time series search.",
            proposed_action=ProposedAction(
                thought="Search for AR(1) in course materials",
                action_type="tool_call",
                tool_call=ToolCall(call_id="c1", tool_name="search", arguments={"q": "AR(1)"}),
            ),
            authorized=True,
        )
        assert step.step_index == 0
        assert step.authorized is True

        trace = AgentTrace(
            run_id="run_abc",
            query="Explain AR(1) causality.",
            terminal_state="TERMINATED_SUCCESS",
            steps=[step],
        )
        assert trace.run_id == "run_abc"
        assert len(trace.steps) == 1
