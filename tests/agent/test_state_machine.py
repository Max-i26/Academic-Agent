"""Unit tests for Stage 6.1 Finite State Machine."""

import pytest
from academic_agent.agent.fsm import AgentStateMachine
from academic_agent.agent.models import (
    AgentExecutionState,
    AgentTerminalState,
    InvalidStateTransitionError,
)


class TestAgentStateMachine:
    """Test suite validating FSM state transitions, guards, and history audit log."""

    def test_initial_state_defaults_to_initialized(self):
        """State machine starts in INITIALIZED and records genesis transition."""
        fsm = AgentStateMachine()
        assert fsm.current_state == AgentExecutionState.INITIALIZED
        assert fsm.is_terminal is False
        assert len(fsm.history) == 1
        assert fsm.history[0].to_state == "INITIALIZED"

    def test_valid_full_lifecycle_transitions(self):
        """Standard lifecycle transitions execute cleanly and update history."""
        fsm = AgentStateMachine()

        fsm.transition_to(AgentExecutionState.PLANNING, reason="Query received")
        assert fsm.current_state == AgentExecutionState.PLANNING

        fsm.transition_to(AgentExecutionState.ACTION_SELECTION, reason="Planner proposed search")
        assert fsm.current_state == AgentExecutionState.ACTION_SELECTION

        fsm.transition_to(AgentExecutionState.TOOL_INVOCATION, reason="Engine authorized search tool")
        assert fsm.current_state == AgentExecutionState.TOOL_INVOCATION

        fsm.transition_to(AgentExecutionState.OBSERVATION_PROCESSING, reason="Search tool returned passages")
        assert fsm.current_state == AgentExecutionState.OBSERVATION_PROCESSING

        fsm.transition_to(AgentExecutionState.PLANNING, reason="Evaluate evidence sufficiency")
        assert fsm.current_state == AgentExecutionState.PLANNING

        fsm.transition_to(AgentExecutionState.ANSWER_SYNTHESIS, reason="Sufficient evidence acquired")
        assert fsm.current_state == AgentExecutionState.ANSWER_SYNTHESIS

        rec = fsm.transition_to(AgentTerminalState.TERMINATED_SUCCESS, reason="Answer verified and grounded")
        assert fsm.current_state == AgentTerminalState.TERMINATED_SUCCESS
        assert fsm.is_terminal is True
        assert rec.to_state == "TERMINATED_SUCCESS"
        assert len(fsm.history) == 8

    def test_valid_abstention_and_budget_exceeded_transitions(self):
        """Direct transitions to TERMINATED_ABSTENTION and TERMINATED_BUDGET_EXCEEDED are valid."""
        fsm1 = AgentStateMachine()
        fsm1.transition_to(AgentExecutionState.PLANNING)
        fsm1.transition_to(AgentTerminalState.TERMINATED_ABSTENTION, reason="Query out of academic scope")
        assert fsm1.current_state == AgentTerminalState.TERMINATED_ABSTENTION
        assert fsm1.is_terminal is True

        fsm2 = AgentStateMachine()
        fsm2.transition_to(AgentExecutionState.PLANNING)
        fsm2.transition_to(AgentTerminalState.TERMINATED_BUDGET_EXCEEDED, reason="Max iterations reached")
        assert fsm2.current_state == AgentTerminalState.TERMINATED_BUDGET_EXCEEDED
        assert fsm2.is_terminal is True

    def test_invalid_transition_raises_error(self):
        """Illegal transition attempts raise InvalidStateTransitionError and leave state unchanged."""
        fsm = AgentStateMachine()

        # Cannot skip from INITIALIZED directly to ANSWER_SYNTHESIS
        with pytest.raises(InvalidStateTransitionError, match="Illegal transition"):
            fsm.transition_to(AgentExecutionState.ANSWER_SYNTHESIS)

        assert fsm.current_state == AgentExecutionState.INITIALIZED

        # Cannot jump from TOOL_INVOCATION directly to PLANNING without OBSERVATION_PROCESSING
        fsm.transition_to(AgentExecutionState.PLANNING)
        fsm.transition_to(AgentExecutionState.ACTION_SELECTION)
        fsm.transition_to(AgentExecutionState.TOOL_INVOCATION)

        with pytest.raises(InvalidStateTransitionError, match="Illegal transition"):
            fsm.transition_to(AgentExecutionState.PLANNING)

        assert fsm.current_state == AgentExecutionState.TOOL_INVOCATION

    def test_terminal_state_is_closed(self):
        """No transitions are permitted out of any terminal state."""
        fsm = AgentStateMachine()
        fsm.transition_to(AgentExecutionState.PLANNING)
        fsm.transition_to(AgentExecutionState.ANSWER_SYNTHESIS)
        fsm.transition_to(AgentTerminalState.TERMINATED_SUCCESS)
        assert fsm.is_terminal is True

        with pytest.raises(InvalidStateTransitionError, match="Cannot transition from terminal state"):
            fsm.transition_to(AgentExecutionState.PLANNING)

        with pytest.raises(InvalidStateTransitionError, match="Cannot transition from terminal state"):
            fsm.transition_to(AgentTerminalState.TERMINATED_ERROR)
