"""Data models, state definitions, and telemetry structures for Phase 6 Agent Orchestration."""

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, Optional, Set, Tuple, Union
from pydantic import BaseModel, ConfigDict, Field, model_validator

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

    model_config = ConfigDict(frozen=True)

    read_only: bool = Field(default=True, description="Whether tool makes no modifications to persistent state.")
    idempotent: bool = Field(default=True, description="Whether repeated identical invocations produce identical results.")
    mutating: bool = Field(default=False, description="Whether tool alters persistent state or external resources.")
    requires_confirmation: bool = Field(default=False, description="Whether user confirmation is required prior to execution.")
    timeout_seconds: float = Field(default=15.0, ge=0.01, description="Hard timeout ceiling for tool execution.")

    @model_validator(mode="after")
    def validate_mutation_conflict(self) -> "ToolSideEffectPolicy":
        if self.mutating and self.read_only:
            raise ValueError("Tool cannot be both read_only and mutating.")
        return self


class ToolCall(BaseModel):
    """Structured tool invocation request proposed by planner or authorized by engine."""

    model_config = ConfigDict(frozen=True)

    call_id: str = Field(description="Unique call identifier for correlation.")
    tool_name: str = Field(description="Registered tool name.")
    arguments: Dict[str, Any] = Field(default_factory=dict, description="Validated parameter dictionary.")


class ToolResult(BaseModel):
    """Structured outcome of a tool execution with error isolation."""

    model_config = ConfigDict(frozen=True)

    call_id: str = Field(description="Call identifier correlating with ToolCall.")
    tool_name: str = Field(description="Name of executed tool.")
    success: bool = Field(description="True if tool executed successfully.")
    output: Any = Field(default=None, description="Tool return payload or observation data.")
    error_message: Optional[str] = Field(default=None, description="Human-readable error explanation if failed.")
    error: Optional[str] = Field(default=None, description="Alias for error_message.")
    error_category: Optional[ToolErrorCategory] = Field(default=None, description="Categorized failure classification.")
    latency_ms: float = Field(default=0.0, ge=0.0, description="Measured execution duration in milliseconds.")

    @model_validator(mode="before")
    @classmethod
    def resolve_error_alias(cls, data: Any) -> Any:
        if isinstance(data, dict):
            if "error" in data and ("error_message" not in data or not data["error_message"]):
                data["error_message"] = data["error"]
            elif "error_message" in data and ("error" not in data or not data["error"]):
                data["error"] = data["error_message"]
        return data


class TokenAccounting(BaseModel):
    """Cumulative token consumption metrics distinguishing exact provider counts from estimates."""

    model_config = ConfigDict(frozen=True)

    prompt_tokens: int = Field(default=0, ge=0, description="Input prompt tokens consumed.")
    completion_tokens: int = Field(default=0, ge=0, description="Output tokens generated.")
    total_tokens: int = Field(default=0, ge=0, description="Sum of prompt and completion tokens.")
    is_exact: bool = Field(default=True, description="True if reported directly by provider; False if estimated.")

    @model_validator(mode="after")
    def validate_token_sum(self) -> "TokenAccounting":
        if self.total_tokens != (self.prompt_tokens + self.completion_tokens):
            raise ValueError(
                f"total_tokens ({self.total_tokens}) must equal prompt_tokens ({self.prompt_tokens}) + "
                f"completion_tokens ({self.completion_tokens})"
            )
        return self

    def add(self, prompt: int = 0, completion: int = 0, is_exact: Optional[bool] = None) -> "TokenAccounting":
        """Return a new TokenAccounting instance accumulating usage."""
        if prompt < 0 or completion < 0:
            raise ValueError(f"Token increments must be non-negative (got prompt={prompt}, completion={completion})")
        new_p = self.prompt_tokens + prompt
        new_c = self.completion_tokens + completion
        new_exact = self.is_exact if is_exact is None else (self.is_exact and is_exact)
        return TokenAccounting(
            prompt_tokens=new_p,
            completion_tokens=new_c,
            total_tokens=new_p + new_c,
            is_exact=new_exact,
        )


class StateTransitionRecord(BaseModel):
    """Immutable record of an agent state machine transition."""

    model_config = ConfigDict(frozen=True)

    from_state: str = Field(description="Pre-transition state.")
    to_state: str = Field(description="Post-transition state.")
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="ISO 8601 UTC timestamp of transition.",
    )
    reason: Optional[str] = Field(default=None, description="Optional rationale or triggering condition.")


class ProposedAction(BaseModel):
    """Next action proposed by the LLM step planner before engine authorization."""

    model_config = ConfigDict(frozen=True)

    action_rationale: str = Field(default="", description="Concise rationale explaining why this action was selected.")
    action_type: str = Field(description="Action category: 'tool_call', 'direct_answer', or 'refusal'.")
    decision_summary: Optional[str] = Field(default=None, description="High-level summary of the decision.")
    tool_call: Optional[ToolCall] = Field(default=None, description="Proposed tool invocation if action_type is 'tool_call'.")
    direct_answer: Optional[str] = Field(default=None, description="Draft answer if action_type is 'direct_answer'.")
    refusal_reason: Optional[str] = Field(default=None, description="Refusal rationale if action_type is 'refusal'.")

    @model_validator(mode="before")
    @classmethod
    def resolve_action_rationale_alias(cls, data: Any) -> Any:
        if isinstance(data, dict):
            if "thought" in data and ("action_rationale" not in data or not data["action_rationale"]):
                data["action_rationale"] = data["thought"]
        return data

    @property
    def thought(self) -> str:
        """Backward compatibility alias for action_rationale."""
        return self.action_rationale


class AgentStep(BaseModel):
    """Immutable audit record of a single step in the agent reasoning and execution loop."""

    model_config = ConfigDict(frozen=True)

    step_index: int = Field(ge=0, description="0-based sequence index.")
    state: str = Field(description="Agent state at start of step.")
    action_rationale: str = Field(default="", description="Concise rationale for selected action.")
    decision_summary: str = Field(default="", description="High-level summary of action or decision taken.")
    observation_summary: Optional[str] = Field(default=None, description="Concise summary of tool observation.")
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

    @model_validator(mode="before")
    @classmethod
    def resolve_step_rationale_alias(cls, data: Any) -> Any:
        if isinstance(data, dict):
            if "thought" in data and ("action_rationale" not in data or not data["action_rationale"]):
                data["action_rationale"] = data["thought"]
        return data

    @property
    def thought(self) -> str:
        """Backward compatibility alias for action_rationale."""
        return self.action_rationale


class BudgetConfig(BaseModel):
    """Multi-dimensional execution budget limits constraining agent autonomy."""

    model_config = ConfigDict(frozen=True)

    max_iterations: int = Field(default=5, ge=1, le=20, description="Maximum planning/tool cycles.")
    timeout_seconds: float = Field(default=60.0, ge=0.01, description="Hard wall-clock ceiling across entire run.")
    max_total_tokens: int = Field(default=10000, ge=100, description="Cumulative ceiling on prompt + output tokens.")
    tool_timeout_seconds: float = Field(default=15.0, ge=0.01, description="Per-tool timeout.")
    max_tool_invocations: int = Field(default=10, ge=1, description="Cumulative ceiling on tool calls.")
    detect_duplicate_calls: bool = Field(default=True, description="Whether to reject duplicate identical tool calls.")


class BudgetConsumption(BaseModel):
    """Audit report of consumed execution resources against BudgetConfig."""

    model_config = ConfigDict(frozen=True)

    iterations: int = Field(default=0, ge=0, description="Planning iterations executed.")
    elapsed_seconds: float = Field(default=0.0, ge=0.0, description="Total wall-clock duration.")
    tool_invocations: int = Field(default=0, ge=0, description="Total tool calls dispatched.")
    token_usage: TokenAccounting = Field(default_factory=TokenAccounting, description="Total tokens consumed.")
    duplicate_calls_detected: int = Field(default=0, ge=0, description="Number of duplicate calls intercepted.")
    limit_exceeded: bool = Field(default=False, description="True if any budget ceiling was hit.")
    exhausted_resource: Optional[str] = Field(default=None, description="Name of resource that exhausted budget.")


class AgentTrace(BaseModel):
    """Complete, structured, and immutable audit trace of an agent execution run."""

    model_config = ConfigDict(frozen=True)

    run_id: str = Field(description="Unique UUID identifying this execution session.")
    query: str = Field(description="Original student query.")
    terminal_state: Optional[str] = Field(default=None, description="Final terminal state of execution.")
    steps: Tuple[AgentStep, ...] = Field(default_factory=tuple, description="Ordered immutable history of executed steps.")
    transitions: Tuple[StateTransitionRecord, ...] = Field(default_factory=tuple, description="State transition log.")
    budget_consumed: BudgetConsumption = Field(default_factory=BudgetConsumption, description="Resource consumption.")
    final_answer: Optional[AcademicAnswer] = Field(default=None, description="Verified AcademicAnswer if successful.")
    refusal_explanation: Optional[str] = Field(default=None, description="Refusal explanation if terminated in abstention.")
    total_latency_ms: float = Field(default=0.0, ge=0.0, description="Total run time in milliseconds.")
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="ISO 8601 UTC run start timestamp.",
    )
    completed_at: Optional[str] = Field(default=None, description="ISO 8601 UTC run finish timestamp.")
