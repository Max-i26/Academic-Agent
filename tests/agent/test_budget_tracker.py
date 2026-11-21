"""Unit tests for Stage 6.1 ExecutionBudgetTracker."""

import time
import pytest
from academic_agent.agent.budget import BudgetExceededError, ExecutionBudgetTracker
from academic_agent.agent.models import BudgetConfig, ToolCall
from academic_agent.agent.protocols import ExecutionBudgetTrackerProtocol


class TestExecutionBudgetTracker:
    """Test suite validating multi-dimensional resource bounds and duplicate tool interception."""

    def test_protocol_conformance(self):
        """Tracker satisfies ExecutionBudgetTrackerProtocol."""
        tracker = ExecutionBudgetTracker()
        assert isinstance(tracker, ExecutionBudgetTrackerProtocol)

    def test_iteration_limit_enforcement(self):
        """Exceeding max_iterations raises BudgetExceededError and marks exhausted."""
        cfg = BudgetConfig(max_iterations=3)
        tracker = ExecutionBudgetTracker(config=cfg)

        tracker.record_iteration()  # 1
        assert tracker.is_exhausted() is False
        tracker.record_iteration()  # 2
        assert tracker.is_exhausted() is False
        tracker.record_iteration()  # 3
        assert tracker.is_exhausted() is True

        with pytest.raises(BudgetExceededError, match="max_iterations"):
            tracker.record_iteration()  # 4 -> limit exceeded

        assert tracker.is_exhausted() is True
        assert tracker.get_exhausted_resource() == "iterations"
        consumption = tracker.get_consumption()
        assert consumption.limit_exceeded is True
        assert consumption.iterations == 4

    def test_token_ceiling_enforcement(self):
        """Exceeding max_total_tokens raises BudgetExceededError."""
        cfg = BudgetConfig(max_total_tokens=500)
        tracker = ExecutionBudgetTracker(config=cfg)

        tracker.record_tokens(prompt=200, completion=100, is_exact=True)
        assert tracker.token_usage.total_tokens == 300
        assert tracker.is_exhausted() is False

        with pytest.raises(BudgetExceededError, match="max_total_tokens"):
            tracker.record_tokens(prompt=150, completion=100, is_exact=False)  # Total 550 > 500

        assert tracker.is_exhausted() is True
        assert tracker.get_exhausted_resource() == "tokens"
        assert tracker.token_usage.is_exact is False

    def test_timeout_enforcement(self):
        """Elapsed time exceeding timeout_seconds raises BudgetExceededError on next operation."""
        cfg = BudgetConfig(timeout_seconds=0.1)  # 100 ms timeout
        tracker = ExecutionBudgetTracker(config=cfg)

        time.sleep(0.12)  # Wait for timeout to elapse

        with pytest.raises(BudgetExceededError, match="timeout_seconds"):
            tracker.record_iteration()

        assert tracker.is_exhausted() is True
        assert tracker.get_exhausted_resource() == "timeout"

    def test_duplicate_tool_call_detection(self):
        """Identical tool invocations with identical parameters are intercepted and rejected."""
        cfg = BudgetConfig(detect_duplicate_calls=True)
        tracker = ExecutionBudgetTracker(config=cfg)

        call1 = ToolCall(call_id="c1", tool_name="search", arguments={"q": "AR(1)", "k": 3})
        call2 = ToolCall(call_id="c2", tool_name="search", arguments={"k": 3, "q": "AR(1)"})  # Same args, different order

        auth1, reason1 = tracker.record_tool_call(call1)
        assert auth1 is True
        assert reason1 is None
        assert tracker.tool_invocations == 1

        auth2, reason2 = tracker.record_tool_call(call2)
        assert auth2 is False
        assert "Duplicate tool invocation rejected" in reason2
        assert tracker.tool_invocations == 1  # Did not increment
        assert tracker.duplicate_calls_detected == 1

        # Different query should be authorized
        call3 = ToolCall(call_id="c3", tool_name="search", arguments={"q": "MA(1)", "k": 3})
        auth3, reason3 = tracker.record_tool_call(call3)
        assert auth3 is True
        assert tracker.tool_invocations == 2

    def test_tool_invocation_ceiling_enforcement(self):
        """Exceeding max_tool_invocations rejects subsequent tool calls."""
        cfg = BudgetConfig(max_tool_invocations=2)
        tracker = ExecutionBudgetTracker(config=cfg)

        auth1, _ = tracker.record_tool_call(ToolCall(call_id="c1", tool_name="t1"))
        auth2, _ = tracker.record_tool_call(ToolCall(call_id="c2", tool_name="t2"))
        assert auth1 is True and auth2 is True

        auth3, reason3 = tracker.record_tool_call(ToolCall(call_id="c3", tool_name="t3"))
        assert auth3 is False
        assert "Maximum tool invocations reached" in reason3
        assert tracker.is_exhausted() is True
        assert tracker.get_exhausted_resource() == "tool_invocations"
