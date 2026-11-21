"""Unit tests for Stage 6.1 ExecutionBudgetTracker."""

import concurrent.futures
import time
import pytest
from academic_agent.agent.budget import BudgetExceededError, ExecutionBudgetTracker
from academic_agent.agent.models import BudgetConfig, ToolCall
from academic_agent.agent.protocols import ExecutionBudgetTrackerProtocol


class TestExecutionBudgetTracker:
    """Test suite validating multi-dimensional resource bounds, thread safety, and duplicate interception."""

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

    def test_timeout_and_deadline_helpers(self):
        """Elapsed time exceeding timeout_seconds raises BudgetExceededError; deadline is computed."""
        cfg = BudgetConfig(timeout_seconds=0.08)  # 80 ms timeout
        tracker = ExecutionBudgetTracker(config=cfg)

        assert tracker.deadline > tracker.start_time
        assert tracker.get_remaining_time() > 0.0

        time.sleep(0.10)  # Wait for timeout to elapse

        assert tracker.get_remaining_time() == 0.0
        with pytest.raises(BudgetExceededError, match="timeout_seconds"):
            tracker.record_iteration()

        assert tracker.is_exhausted() is True
        assert tracker.get_exhausted_resource() == "timeout"

    def test_duplicate_tool_call_detection(self):
        """Identical tool invocations with identical parameters are intercepted and rejected."""
        cfg = BudgetConfig(detect_duplicate_calls=True)
        tracker = ExecutionBudgetTracker(config=cfg)

        call1 = ToolCall(call_id="c1", tool_name="search", arguments={"q": "AR(1)", "k": 3})
        call2 = ToolCall(call_id="c2", tool_name="search", arguments={"k": 3, "q": "AR(1)"})  # Same args, different key order

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

    def test_multithreaded_concurrency_safety(self):
        """Concurrent updates to tokens and tool calls are synchronized without race conditions."""
        tracker = ExecutionBudgetTracker(config=BudgetConfig(max_total_tokens=100000, max_tool_invocations=1000))

        def worker(w_id: int):
            for i in range(20):
                tracker.record_tokens(prompt=10, completion=5, is_exact=True)
                tracker.record_tool_call(ToolCall(call_id=f"c_{w_id}_{i}", tool_name=f"tool_{w_id}_{i}"))

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            futures = [executor.submit(worker, w) for w in range(8)]
            for f in futures:
                f.result()

        # 8 workers * 20 iterations = 160 operations
        assert tracker.tool_invocations == 160
        assert tracker.token_usage.prompt_tokens == 1600
        assert tracker.token_usage.completion_tokens == 800
        assert tracker.token_usage.total_tokens == 2400

    def test_concurrent_atomic_duplicate_interception(self):
        """Concurrent submission of identical tool calls authorizes exactly one and intercepts the rest."""
        tracker = ExecutionBudgetTracker(config=BudgetConfig(max_tool_invocations=100))
        identical_call = ToolCall(call_id="cid", tool_name="search", arguments={"query": "GARCH(1,1)", "top_k": 5})

        results = []

        def submit():
            auth, reason = tracker.record_tool_call(identical_call)
            return auth

        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(submit) for _ in range(10)]
            for f in futures:
                results.append(f.result())

        # Exactly one authorized, 9 rejected as duplicates
        assert results.count(True) == 1
        assert results.count(False) == 9
        assert tracker.duplicate_calls_detected == 9
        assert tracker.tool_invocations == 1
