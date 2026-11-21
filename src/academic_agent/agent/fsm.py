"""Finite State Machine engine with strict transition enforcement."""

from typing import List, Optional, Tuple
from datetime import datetime, timezone

from academic_agent.agent.models import (
    AgentExecutionState,
    AgentState,
    AgentTerminalState,
    InvalidStateTransitionError,
    StateTransitionRecord,
    VALID_STATE_TRANSITIONS,
)


class AgentStateMachine:
    """Deterministic finite state machine governing the agent execution lifecycle.

    Guarantees:
    - Initialization is strictly validated: constructor rejects terminal or unknown states.
    - Every state transition is validated against VALID_STATE_TRANSITIONS.
    - Terminal states are strictly closed (cannot transition to any other state).
    - Every transition produces an immutable StateTransitionRecord with timestamp and reason.
    - History log returns an immutable tuple of records.
    """

    def __init__(self, initial_state: AgentState = AgentExecutionState.INITIALIZED) -> None:
        # Validate that initial state is a recognized non-terminal execution state
        allowed_initial = {s.value if hasattr(s, "value") else str(s) for s in AgentExecutionState}
        init_val = initial_state.value if hasattr(initial_state, "value") else str(initial_state)

        if init_val not in allowed_initial:
            raise InvalidStateTransitionError(
                "START",
                initial_state,
                f"Invalid initial state '{initial_state}'. State machine must initialize in an active execution state: {sorted(list(allowed_initial))}",
            )

        self._current_state: AgentState = initial_state
        self._history: List[StateTransitionRecord] = [
            StateTransitionRecord(
                from_state="START",
                to_state=init_val,
                timestamp=datetime.now(timezone.utc).isoformat(),
                reason="Agent session initialized",
            )
        ]

    @property
    def current_state(self) -> AgentState:
        """Current state of the state machine."""
        return self._current_state

    @property
    def is_terminal(self) -> bool:
        """True if current state is one of the terminal exit states."""
        return isinstance(self._current_state, AgentTerminalState) or self._current_state in {
            s.value for s in AgentTerminalState
        }

    @property
    def history(self) -> Tuple[StateTransitionRecord, ...]:
        """Ordered, immutable audit log of all state transitions."""
        return tuple(self._history)

    def transition_to(self, target_state: AgentState, reason: Optional[str] = None) -> StateTransitionRecord:
        """Attempt to transition the state machine to target_state.

        Args:
            target_state: Desired destination state.
            reason: Optional human-readable rationale or condition triggering transition.

        Returns:
            StateTransitionRecord documenting the successful transition.

        Raises:
            InvalidStateTransitionError: If the transition is illegal or if current state is terminal.
        """
        # 1. Terminal state check: terminal states cannot transition to anything
        if self.is_terminal:
            raise InvalidStateTransitionError(
                self._current_state,
                target_state,
                f"Cannot transition from terminal state '{self._current_state}' to '{target_state}'.",
            )

        # 2. Transition matrix check
        allowed_targets = VALID_STATE_TRANSITIONS.get(self._current_state, set())
        allowed_target_values = {
            s.value if hasattr(s, "value") else str(s) for s in allowed_targets
        }
        target_val = target_state.value if hasattr(target_state, "value") else str(target_state)

        if target_val not in allowed_target_values:
            raise InvalidStateTransitionError(
                self._current_state,
                target_state,
                f"Illegal transition from '{self._current_state}' to '{target_state}'. "
                f"Allowed destinations: {sorted(list(allowed_target_values))}",
            )

        # 3. Apply transition
        from_str = self._current_state.value if hasattr(self._current_state, "value") else str(self._current_state)
        to_str = target_val
        record = StateTransitionRecord(
            from_state=from_str,
            to_state=to_str,
            timestamp=datetime.now(timezone.utc).isoformat(),
            reason=reason,
        )
        self._current_state = target_state
        self._history.append(record)
        return record
