"""Data models, state definitions, and telemetry structures for Phase 6 Agent Orchestration."""

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Union
from pydantic import BaseModel, Field, field_validator

from academic_agent.generation.synthesis.models import AcademicAnswer


class AgentExecutionState(str, Enum):
    """Active non-terminal states of the agent execution lifecycle."""

    INITIALIZED = "INITIALIZED"
    PLANNING = "PLANNING"
    ACTION_SELECTION = "ACTION_SELECTION"
    TOOL_INVOCATION = "TOOL_INVOCATION"
    OBSERVATION_PROCESSING = "OBSERVATION_PROCESSING"
    ANSWER_SYNTHESIS = "ANSWER_SYNTHESIS"


class AgentTerminalState(str, Enum):
    """Terminal final states representing bounded exit outcomes."""

    TERMINATED_SUCCESS = "TERMINATED_SUCCESS"
    TERMINATED_ABSTENTION = "TERMINATED_ABSTENTION"
    TERMINATED_BUDGET_EXCEEDED = "TERMINATED_BUDGET_EXCEEDED"
    TERMINATED_ERROR = "TERMINATED_ERROR"
    TERMINATED_CANCELLED = "TERMINATED_CANCELLED"


# Union / composite representation of all agent states
AgentState = Union[AgentExecutionState, AgentTerminalState]


class InvalidStateTransitionError(Exception):
    """Raised when an illegal state transition is attempted."""

    def __init__(self, from_state: AgentState, to_state: AgentState, message: Optional[str] = None):
        msg = message or f"Illegal state transition attempted: '{from_state}' -> '{to_state}'"
        super().__init__(msg)
        self.from_state = from_state
        self.to_state = to_state


# Explicit validated FSM transition matrix
VALID_STATE_TRANSITIONS: Dict[AgentState, Set[AgentState]] = {
    AgentExecutionState.INITIALIZED: {
        AgentExecutionState.PLANNING,
        AgentTerminalState.TERMINATED_CANCELLED,
        AgentTerminalState.TERMINATED_ERROR,
    },
    AgentExecutionState.PLANNING: {
        AgentExecutionState.ACTION_SELECTION,
        AgentExecutionState.ANSWER_SYNTHESIS,
        AgentTerminalState.TERMINATED_ABSTENTION,
        AgentTerminalState.TERMINATED_BUDGET_EXCEEDED,
        AgentTerminalState.TERMINATED_CANCELLED,
        AgentTerminalState.TERMINATED_ERROR,
    },
    AgentExecutionState.ACTION_SELECTION: {
        AgentExecutionState.TOOL_INVOCATION,
        AgentExecutionState.ANSWER_SYNTHESIS,
        AgentTerminalState.TERMINATED_ABSTENTION,
        AgentTerminalState.TERMINATED_BUDGET_EXCEEDED,
        AgentTerminalState.TERMINATED_CANCELLED,
        AgentTerminalState.TERMINATED_ERROR,
    },
    AgentExecutionState.TOOL_INVOCATION: {
        AgentExecutionState.OBSERVATION_PROCESSING,
        AgentTerminalState.TERMINATED_BUDGET_EXCEEDED,
        AgentTerminalState.TERMINATED_CANCELLED,
        AgentTerminalState.TERMINATED_ERROR,
    },
    AgentExecutionState.OBSERVATION_PROCESSING: {
        AgentExecutionState.PLANNING,
        AgentExecutionState.ANSWER_SYNTHESIS,
        AgentTerminalState.TERMINATED_BUDGET_EXCEEDED,
        AgentTerminalState.TERMINATED_CANCELLED,
        AgentTerminalState.TERMINATED_ERROR,
    },
    AgentExecutionState.ANSWER_SYNTHESIS: {
        AgentTerminalState.TERMINATED_SUCCESS,
        AgentTerminalState.TERMINATED_ABSTENTION,
        AgentTerminalState.TERMINATED_CANCELLED,
        AgentTerminalState.TERMINATED_ERROR,
    },
    # Terminal states have no permitted outgoing transitions
    AgentTerminalState.TERMINATED_SUCCESS: set(),
    AgentTerminalState.TERMINATED_ABSTENTION: set(),
    AgentTerminalState.TERMINATED_BUDGET_EXCEEDED: set(),
    AgentTerminalState.TERMINATED_ERROR: set(),
    AgentTerminalState.TERMINATED_CANCELLED: set(),
}


class ToolErrorCategory(str, Enum):
    """Classification of tool execution and validation failures."""

    VALIDATION_ERROR = "VALIDATION_ERROR"
    TIMEOUT = "TIMEOUT"
    EXECUTION_ERROR = "EXECUTION_ERROR"
    PERMISSION_DENIED = "PERMISSION_DENIED"
    NOT_FOUND = "NOT_FOUND"


class ToolSideEffectPolicy(BaseModel):
    """Safety and side-effect policy metadata governing a tool."""

    read_only: bool = Field(default=True, description="Whether tool makes no modifications to persistent state.")
    idempotent: bool = Field(default=True, description="Whether repeated identical invocations produce identical results.")
    mutating: bool = Field(default=False, description="Whether tool alters persistent state or external resources.")
    requires_confirmation: bool = Field(default=False, description="Whether user confirmation is required prior to execution.")
    timeout_seconds: float = Field(default=15.0, ge=0.1, description="Hard timeout ceiling for tool execution.")

    @field_validator("mutating")
    @classmethod
    def validate_mutation_conflict(cls, v: bool, info) -> bool:
        if v and info.data.get("read_only", True):
            raise ValueError("Tool cannot be both read_only and mutating.")
        return v


class ToolCall(BaseModel):
    """Structured tool invocation request proposed by planner or authorized by engine."""

    call_id: str = Field(description="Unique call identifier for correlation.")
    tool_name: str = Field(description="Registered tool name.")
    arguments: Dict[str, Any] = Field(default_factory=dict, description="Validated parameter dictionary.")


class ToolResult(BaseModel):
    """Structured outcome of a tool execution with error isolation."""

    call_id: str = Field(description="Call identifier correlating with ToolCall.")
    tool_name: str = Field(description="Name of executed tool.")
    success: bool = Field(description="True if tool executed successfully.")
    output: Any = Field(default=None, description="Tool return payload or observation data.")
    error_message: Optional[str] = Field(default=None, description="Human-readable error explanation if failed.")
    error_category: Optional[ToolErrorCategory] = Field(default=None, description="Categorized failure classification.")
    latency_ms: float = Field(default=0.0, ge=0.0, description="Measured execution duration in milliseconds.")


class TokenAccounting(BaseModel):
    """Cumulative token consumption metrics distinguishing exact provider counts from estimates."""

    prompt_tokens: int = Field(default=0, ge=0, description="Input prompt tokens consumed.")
    completion_tokens: int = Field(default=0, ge=0, description="Output tokens generated.")
    total_tokens: int = Field(default=0, ge=0, description="Sum of prompt and completion tokens.")
    is_exact: bool = Field(default=True, description="True if reported directly by provider; False if estimated.")

    def add(self, prompt: int, completion: int, is_exact: bool = True) -> "TokenAccounting":
        """Return a new TokenAccounting instance accumulating usage."""
        return TokenAccounting(
            prompt_tokens=self.prompt_tokens + prompt,
            completion_tokens=self.completion_tokens + completion,
            total_tokens=self.total_tokens + prompt + completion,
            is_exact=self.is_exact and is_exact,
        )


class StateTransitionRecord(BaseModel):
    """Immutable record of an agent state machine transition."""

    from_state: str = Field(description="Pre-transition state.")
    to_state: str = Field(description="Post-transition state.")
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="ISO 8601 UTC timestamp of transition.",
    )
    reason: Optional[str] = Field(default=None, description="Optional rationale or triggering condition.")


class ProposedAction(BaseModel):
    """Next action proposed by the LLM step planner before engine authorization."""

    thought: str = Field(description="Explicit reasoning rationale for this step.")
    action_type: str = Field(description="Action category: 'tool_call', 'direct_answer', or 'refusal'.")
    tool_call: Optional[ToolCall] = Field(default=None, description="Proposed tool invocation if action_type is 'tool_call'.")
    direct_answer: Optional[str] = Field(default=None, description="Draft answer if action_type is 'direct_answer'.")
    refusal_reason: Optional[str] = Field(default=None, description="Refusal rationale if action_type is 'refusal'.")


class AgentStep(BaseModel):
    """Immutable audit record of a single step in the agent reasoning and execution loop."""

    step_index: int = Field(ge=0, description="0-based sequence index.")
    state: str = Field(description="Agent state at start of step.")
    thought: str = Field(default="", description="Reasoning thought produced during planning.")
    proposed_action: Optional[ProposedAction] = Field(default=None, description="Action proposed by planner.")
    authorized: bool = Field(default=False, description="Whether engine execution guardrails authorized the action.")
    authorization_rejection_reason: Optional[str] = Field(default=None, description="Reason if action was rejected.")
    tool_call: Optional[ToolCall] = Field(default=None, description="Executed tool call if authorized.")
    tool_result: Optional[ToolResult] = Field(default=None, description="Tool execution outcome.")
    token_usage: TokenAccounting = Field(default_factory=TokenAccounting, description="Step token consumption.")
    latency_ms: float = Field(default=0.0, ge=0.0, description="Step execution time in milliseconds.")
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="ISO 8601 UTC step start timestamp.",
    )


class BudgetConfig(BaseModel):
    """Multi-dimensional execution budget limits constraining agent autonomy."""

    max_iterations: int = Field(default=5, ge=1, le=20, description="Maximum planning/tool cycles.")
    timeout_seconds: float = Field(default=60.0, ge=0.01, description="Hard wall-clock ceiling across entire run.")
    max_total_tokens: int = Field(default=10000, ge=100, description="Cumulative ceiling on prompt + output tokens.")
    tool_timeout_seconds: float = Field(default=15.0, ge=0.5, description="Per-tool timeout.")
    max_tool_invocations: int = Field(default=10, ge=1, description="Cumulative ceiling on tool calls.")
    detect_duplicate_calls: bool = Field(default=True, description="Whether to reject duplicate identical tool calls.")


class BudgetConsumption(BaseModel):
    """Audit report of consumed execution resources against BudgetConfig."""

    iterations: int = Field(default=0, ge=0, description="Planning iterations executed.")
    elapsed_seconds: float = Field(default=0.0, ge=0.0, description="Total wall-clock duration.")
    tool_invocations: int = Field(default=0, ge=0, description="Total tool calls dispatched.")
    token_usage: TokenAccounting = Field(default_factory=TokenAccounting, description="Total tokens consumed.")
    duplicate_calls_detected: int = Field(default=0, ge=0, description="Number of duplicate calls intercepted.")
    limit_exceeded: bool = Field(default=False, description="True if any budget ceiling was hit.")
    exhausted_resource: Optional[str] = Field(default=None, description="Name of resource that exhausted budget.")


class AgentTrace(BaseModel):
    """Complete, structured, and immutable audit trace of an agent execution run."""

    run_id: str = Field(description="Unique UUID identifying this execution session.")
    query: str = Field(description="Original student query.")
    terminal_state: Optional[str] = Field(default=None, description="Final terminal state of execution.")
    steps: List[AgentStep] = Field(default_factory=list, description="Ordered history of executed steps.")
    transitions: List[StateTransitionRecord] = Field(default_factory=list, description="State transition log.")
    budget_consumed: BudgetConsumption = Field(default_factory=BudgetConsumption, description="Resource consumption.")
    final_answer: Optional[AcademicAnswer] = Field(default=None, description="Verified AcademicAnswer if successful.")
    refusal_explanation: Optional[str] = Field(default=None, description="Refusal explanation if terminated in abstention.")
    total_latency_ms: float = Field(default=0.0, ge=0.0, description="Total run time in milliseconds.")
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="ISO 8601 UTC run start timestamp.",
    )
    completed_at: Optional[str] = Field(default=None, description="ISO 8601 UTC run finish timestamp.")
