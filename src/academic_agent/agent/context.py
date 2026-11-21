"""Execution context, cancellation primitives, and per-run lifecycle containers."""

from datetime import datetime, timezone
import threading
import time
from typing import List, Optional, Tuple
import uuid

from pydantic import BaseModel, ConfigDict, Field

from academic_agent.agent.budget import ExecutionBudgetTracker
from academic_agent.agent.fsm import AgentStateMachine
from academic_agent.agent.models import (
    AgentExecutionState,
    AgentState,
    AgentStep,
    AgentTerminalState,
    AgentTrace,
    BudgetConfig,
    BudgetConsumption,
    StateTransitionRecord,
)
from academic_agent.generation.synthesis.models import AcademicAnswer


class CancellationToken:
    """Thread-safe cooperative cancellation token for interrupting agent runs between steps."""

    def __init__(self) -> None:
        self._event = threading.Event()
        self._reason: Optional[str] = None
        self._lock = threading.Lock()

    @property
    def is_cancelled(self) -> bool:
        """True if cancellation has been requested."""
        return self._event.is_set()

    @property
    def reason(self) -> Optional[str]:
        """Human-readable explanation of why cancellation was requested."""
        with self._lock:
            return self._reason

    def cancel(self, reason: Optional[str] = None) -> None:
        """Request cooperative cancellation.

        Strictly monotonic: once cancelled, cancellation cannot be undone or reset.
        """
        with self._lock:
            if not self._event.is_set():
                self._reason = reason or "Run cancelled by caller."
                self._event.set()


class AgentRunConfig(BaseModel):
    """Configuration options governing an AcademicAgent run."""

    model_config = ConfigDict(frozen=True)

    budget_config: BudgetConfig = Field(
        default_factory=BudgetConfig,
        description="Resource allocation limits (iterations, tokens, timeouts).",
    )
    enable_query_screening: bool = Field(
        default=True,
        description="Whether to run pre-retrieval clarification and domain screening.",
    )
    max_consecutive_rejections: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Maximum consecutive policy rejections before forcing graceful abstention.",
    )


class AgentExecutionContext:
    """Isolated, thread-safe runtime container for a single AcademicAgent execution run.

    Guarantees:
    - Zero state sharing between concurrent agent runs.
    - Strict encapsulation of mutable state (steps, transitions, budget, FSM).
    - Monotonic latency tracking.
    - Construction of deeply immutable AgentTrace on completion.
    """

    def __init__(
        self,
        query: str,
        config: Optional[AgentRunConfig] = None,
        run_id: Optional[str] = None,
        cancellation_token: Optional[CancellationToken] = None,
    ) -> None:
        self.run_id: str = run_id or str(uuid.uuid4())
        self.query: str = query
        self.config: AgentRunConfig = config or AgentRunConfig()
        self.cancellation_token: CancellationToken = cancellation_token or CancellationToken()

        self.start_perf_time: float = time.perf_counter()
        self.created_at_iso: str = datetime.now(timezone.utc).isoformat()

        self.state_machine: AgentStateMachine = AgentStateMachine(initial_state=AgentExecutionState.INITIALIZED)
        self.budget_tracker: ExecutionBudgetTracker = ExecutionBudgetTracker(self.config.budget_config)

        self._steps: List[AgentStep] = []
        self._transitions: List[StateTransitionRecord] = []
        self.final_answer: Optional[AcademicAnswer] = None
        self.refusal_explanation: Optional[str] = None
        self.consecutive_rejections: int = 0

        self._lock = threading.RLock()

    @property
    def current_state(self) -> AgentState:
        """Current state in the execution state machine."""
        with self._lock:
            return self.state_machine.current_state

    @property
    def is_terminated(self) -> bool:
        """True if the state machine has reached any terminal state."""
        with self._lock:
            return self.state_machine.is_terminal

    @property
    def steps(self) -> List[AgentStep]:
        """Shallow copy of executed steps so far."""
        with self._lock:
            return list(self._steps)

    @property
    def transitions(self) -> List[StateTransitionRecord]:
        """Shallow copy of recorded state transitions."""
        with self._lock:
            return list(self._transitions)

    def add_step(self, step: AgentStep) -> None:
        """Append an executed step to the run history."""
        with self._lock:
            self._steps.append(step)

    def transition_to(self, target_state: AgentState, reason: Optional[str] = None) -> StateTransitionRecord:
        """Advance the execution state machine and record the transition."""
        with self._lock:
            record = self.state_machine.transition_to(target_state, reason=reason)
            self._transitions.append(record)
            return record

    def check_cancellation(self) -> bool:
        """Check if cancellation has been requested.

        Returns:
            True if cancelled; also sets refusal_explanation if not already set.
        """
        if self.cancellation_token.is_cancelled:
            with self._lock:
                if not self.refusal_explanation:
                    self.refusal_explanation = self.cancellation_token.reason or "Run cancelled by user."
            return True
        return False

    def build_trace(self, completed_at: Optional[str] = None) -> AgentTrace:
        """Construct the complete, frozen, and immutable AgentTrace."""
        with self._lock:
            total_latency_ms = (time.perf_counter() - self.start_perf_time) * 1000.0
            completed_iso = completed_at or datetime.now(timezone.utc).isoformat()
            consumption: BudgetConsumption = self.budget_tracker.get_consumption()

            return AgentTrace(
                run_id=self.run_id,
                query=self.query,
                terminal_state=self.state_machine.current_state.value,
                steps=tuple(self._steps),
                transitions=tuple(self._transitions),
                budget_consumed=consumption,
                final_answer=self.final_answer,
                refusal_explanation=self.refusal_explanation,
                total_latency_ms=round(total_latency_ms, 2),
                created_at=self.created_at_iso,
                completed_at=completed_iso,
            )
