"""Unit tests for Stage 6.1 Agent and Tool Data Models."""

import pytest
from pydantic import ValidationError
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
    """Test suite validating core Phase 6 data structures, immutability, and invariants."""

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
        """Default policy must be strictly read-only, idempotent, and frozen immutable."""
        policy = ToolSideEffectPolicy()
        assert policy.read_only is True
        assert policy.idempotent is True
        assert policy.mutating is False
        assert policy.requires_confirmation is False
        assert policy.timeout_seconds == 15.0

        # Immutability check
        with pytest.raises(ValidationError):
            policy.mutating = True

        # Conflict: mutating cannot be True if read_only is True
        with pytest.raises(ValueError, match="Tool cannot be both read_only and mutating"):
            ToolSideEffectPolicy(read_only=True, mutating=True)

    def test_tool_call_and_tool_result_immutability(self):
        """ToolCall and ToolResult are frozen and validate payload correctly."""
        call = ToolCall(
            call_id="call_123",
            tool_name="academic_search",
            arguments={"query": "stationarity", "top_k": 3},
        )
        with pytest.raises(ValidationError):
            call.tool_name = "other_tool"

        res_success = ToolResult(
            call_id="call_123",
            tool_name="academic_search",
            success=True,
            output={"results": []},
            latency_ms=12.5,
        )
        assert res_success.success is True
        with pytest.raises(ValidationError):
            res_success.success = False

    def test_token_accounting_invariants_and_exactness(self):
        """TokenAccounting validates total_tokens == prompt + completion and rejects negatives."""
        # Valid accounting
        acc = TokenAccounting(prompt_tokens=100, completion_tokens=50, total_tokens=150, is_exact=True)
        assert acc.total_tokens == 150

        # Invariant failure: mismatched total_tokens raises ValueError
        with pytest.raises(ValueError, match="total_tokens"):
            TokenAccounting(prompt_tokens=100, completion_tokens=50, total_tokens=999, is_exact=True)

        # Invariant failure: negative token increments rejected
        with pytest.raises(ValueError, match="non-negative"):
            acc.add(prompt=-10, completion=20)

        # Immutability check
        with pytest.raises(ValidationError):
            acc.prompt_tokens = 200

        # Exactness status propagation
        acc2 = acc.add(prompt=50, completion=25, is_exact=False)
        assert acc2.total_tokens == 225
        assert acc2.is_exact is False

    def test_agent_step_and_trace_strict_immutability(self):
        """AgentStep and AgentTrace use decision summaries and enforce tuple immutability."""
        step = AgentStep(
            step_index=0,
            state="PLANNING",
            action_rationale="Question requires time series search.",
            decision_summary="Search AR(1) in Lecture 2",
            proposed_action=ProposedAction(
                action_rationale="Search for AR(1) in course materials",
                action_type="tool_call",
                decision_summary="Call search tool",
                tool_call=ToolCall(call_id="c1", tool_name="search", arguments={"q": "AR(1)"}),
            ),
            authorized=True,
        )
        # Verify AgentStep is frozen
        with pytest.raises(ValidationError):
            step.state = "ACTION_SELECTION"

        trace = AgentTrace(
            run_id="run_abc",
            query="Explain AR(1) causality.",
            terminal_state="TERMINATED_SUCCESS",
            steps=(step,),
        )
        assert isinstance(trace.steps, tuple)
        # Verify AgentTrace is frozen
        with pytest.raises(ValidationError):
            trace.terminal_state = "TERMINATED_ERROR"
