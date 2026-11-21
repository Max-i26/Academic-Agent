"""Tests for Stage 6.4.1 Orchestrator execution context, lifecycle contracts, and cancellation."""

import threading
import time
import pytest

from academic_agent.agent.context import (
    AgentExecutionContext,
    AgentRunConfig,
    CancellationToken,
)
from academic_agent.agent.models import (
    ActionType,
    AgentExecutionState,
    AgentStep,
    AgentTerminalState,
    BudgetConfig,
    InvalidStateTransitionError,
    ProposedAction,
    ToolCall,
    ToolResult,
)


class TestCancellationToken:
    """Test suite for thread-safe cooperative cancellation."""

    def test_cancellation_lifecycle(self) -> None:
        """Verify normal cancellation flag and reason semantics."""
        token = CancellationToken()
        assert token.is_cancelled is False
        assert token.reason is None

        token.cancel("User pressed stop")
        assert token.is_cancelled is True
        assert token.reason == "User pressed stop"

        # Subsequent cancel calls do not overwrite original reason
        token.cancel("Second reason")
        assert token.reason == "User pressed stop"

        token.reset()
        assert token.is_cancelled is False
        assert token.reason is None

    def test_multithreaded_cancellation_concurrency(self) -> None:
        """Verify thread-safety when multiple threads check and cancel simultaneously."""
        token = CancellationToken()
        cancelled_signals = []

        def worker(idx: int) -> None:
            if idx == 5:
                token.cancel("Thread 5 cancellation")
            time.sleep(0.005)
            if token.is_cancelled:
                cancelled_signals.append(True)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert token.is_cancelled is True
        assert token.reason == "Thread 5 cancellation"
        assert len(cancelled_signals) > 0


class TestAgentExecutionContext:
    """Test suite for per-run isolated execution context."""

    def test_context_initialization(self) -> None:
        """Verify fresh state machine, budget tracker, and empty history on init."""
        ctx = AgentExecutionContext(query="Explain gradient descent")
        assert ctx.query == "Explain gradient descent"
        assert ctx.run_id is not None
        assert ctx.current_state == AgentExecutionState.INITIALIZED
        assert ctx.is_terminated is False
        assert len(ctx.steps) == 0
        assert len(ctx.transitions) == 0
        assert ctx.final_answer is None
        assert ctx.refusal_explanation is None

    def test_state_transitions_and_recording(self) -> None:
        """Verify state transitions update FSM and record StateTransitionRecord."""
        ctx = AgentExecutionContext(query="Test query")

        # Legal transition: INITIALIZED -> PLANNING
        ctx.transition_to(AgentExecutionState.PLANNING, reason="Query accepted")
        assert ctx.current_state == AgentExecutionState.PLANNING
        assert len(ctx.transitions) == 1
        assert ctx.transitions[0].from_state == AgentExecutionState.INITIALIZED.value
        assert ctx.transitions[0].to_state == AgentExecutionState.PLANNING.value
        assert ctx.transitions[0].reason == "Query accepted"

        # Legal transition: PLANNING -> ACTION_SELECTION
        ctx.transition_to(AgentExecutionState.ACTION_SELECTION, reason="Planner proposed search")
        assert ctx.current_state == AgentExecutionState.ACTION_SELECTION
        assert len(ctx.transitions) == 2

        # Illegal transition: ACTION_SELECTION -> INITIALIZED
        with pytest.raises(InvalidStateTransitionError):
            ctx.transition_to(AgentExecutionState.INITIALIZED, reason="Illegal reset")

    def test_add_step_and_build_trace(self) -> None:
        """Verify step logging and compilation into immutable AgentTrace."""
        ctx = AgentExecutionContext(query="What is SARIMA?")
        ctx.transition_to(AgentExecutionState.PLANNING)
        ctx.transition_to(AgentExecutionState.ACTION_SELECTION)
        ctx.transition_to(AgentExecutionState.TOOL_INVOCATION)

        step = AgentStep(
            step_index=0,
            state=AgentExecutionState.TOOL_INVOCATION.value,
            action_rationale="Search lectures for SARIMA.",
            decision_summary="Dispatched academic_search.",
            proposed_action=ProposedAction(
                action_rationale="Search lectures for SARIMA.",
                action_type=ActionType.SEARCH,
                tool_call=ToolCall(call_id="c_1", tool_name="academic_search", arguments={"query": "SARIMA"}),
            ),
            authorized=True,
            tool_call=ToolCall(call_id="c_1", tool_name="academic_search", arguments={"query": "SARIMA"}),
            tool_result=ToolResult(
                call_id="c_1",
                tool_name="academic_search",
                success=True,
                output={"candidates": [], "count": 0},
            ),
        )
        ctx.add_step(step)

        ctx.transition_to(AgentExecutionState.OBSERVATION_PROCESSING)
        ctx.transition_to(AgentExecutionState.PLANNING, reason="Observation processed, deciding next action")
        ctx.transition_to(AgentTerminalState.TERMINATED_ABSTENTION, reason="Empty search results")
        ctx.refusal_explanation = "Course materials do not discuss SARIMA."

        trace = ctx.build_trace()
        assert trace.run_id == ctx.run_id
        assert trace.query == "What is SARIMA?"
        assert trace.terminal_state == AgentTerminalState.TERMINATED_ABSTENTION.value
        assert len(trace.steps) == 1
        assert trace.steps[0].action_rationale == "Search lectures for SARIMA."
        assert len(trace.transitions) == 6
        assert trace.refusal_explanation == "Course materials do not discuss SARIMA."
        assert trace.total_latency_ms >= 0.0

    def test_concurrency_isolation_between_contexts(self) -> None:
        """Verify separate contexts do not leak or share mutable state."""
        ctx1 = AgentExecutionContext(query="Query 1", run_id="run-1")
        ctx2 = AgentExecutionContext(query="Query 2", run_id="run-2")

        ctx1.transition_to(AgentExecutionState.PLANNING)
        ctx1.add_step(
            AgentStep(
                step_index=0,
                state=AgentExecutionState.PLANNING.value,
                action_rationale="Step in ctx 1",
            )
        )

        assert ctx1.current_state == AgentExecutionState.PLANNING
        assert ctx2.current_state == AgentExecutionState.INITIALIZED
        assert len(ctx1.steps) == 1
        assert len(ctx2.steps) == 0

    def test_cancellation_check_in_context(self) -> None:
        """Verify context detects cooperative cancellation."""
        token = CancellationToken()
        ctx = AgentExecutionContext(query="Long running task", cancellation_token=token)

        assert ctx.check_cancellation() is False

        token.cancel("User aborted run")
        assert ctx.check_cancellation() is True
        assert ctx.refusal_explanation == "User aborted run"
