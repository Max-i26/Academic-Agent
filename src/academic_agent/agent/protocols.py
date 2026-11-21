"""Protocol interfaces defining Phase 6 Agent and Tool contracts."""

from typing import Any, Dict, List, Optional, Protocol, Tuple, runtime_checkable

from academic_agent.agent.models import (
    AgentStep,
    AgentTrace,
    BudgetConsumption,
    ProposedAction,
    ToolCall,
    ToolResult,
    ToolSideEffectPolicy,
)


@runtime_checkable
class ToolProtocol(Protocol):
    """Protocol satisfied by all executable tools available to the academic agent."""

    @property
    def name(self) -> str:
        """Unique identifier of the tool."""
        ...

    @property
    def description(self) -> str:
        """Detailed academic and functional description of the tool's purpose."""
        ...

    @property
    def side_effect_policy(self) -> ToolSideEffectPolicy:
        """Purity, idempotency, and timeout safety metadata."""
        ...

    def get_schema(self) -> Dict[str, Any]:
        """Return the JSON Schema representation of the tool's input parameters."""
        ...

    def execute(self, arguments: Dict[str, Any]) -> ToolResult:
        """Execute the tool against validated arguments with error isolation."""
        ...


@runtime_checkable
class ToolRegistryProtocol(Protocol):
    """Protocol for tool registration, discovery, and schema cataloging."""

    def register(self, tool: ToolProtocol) -> None:
        """Register a tool instance, validating uniqueness and contract compliance."""
        ...

    def get(self, name: str) -> Optional[ToolProtocol]:
        """Retrieve registered tool by name (case-insensitive)."""
        ...

    def has(self, name: str) -> bool:
        """Check if tool name is registered."""
        ...

    def list_tools(self) -> List[ToolProtocol]:
        """List all currently registered tools."""
        ...

    def get_schemas(self) -> List[Dict[str, Any]]:
        """Return JSON Schema specifications for all registered tools."""
        ...


@runtime_checkable
class ToolDispatcherProtocol(Protocol):
    """Protocol for authorized, isolated, and bounded tool execution."""

    def dispatch(
        self,
        tool_call: ToolCall,
        budget_tracker: Optional["ExecutionBudgetTrackerProtocol"] = None,
    ) -> ToolResult:
        """Execute tool call with pre-authorization, hard timeout, and error containment."""
        ...


@runtime_checkable
class StepPlannerProtocol(Protocol):
    """Protocol for proposing next reasoning and action steps."""

    def plan_next_step(
        self,
        query: str,
        history: List[AgentStep],
        available_tools: List[ToolProtocol],
    ) -> ProposedAction:
        """Generate next reasoning thought and proposed action."""
        ...


@runtime_checkable
class ExecutionBudgetTrackerProtocol(Protocol):
    """Protocol for tracking multi-dimensional execution resources."""

    def record_iteration(self) -> None:
        """Record a completed planning iteration; raises BudgetExceededError if limit hit."""
        ...

    def record_tokens(self, prompt: int, completion: int, is_exact: bool = True) -> None:
        """Accumulate token consumption; raises BudgetExceededError if ceiling hit."""
        ...

    def record_tool_call(self, tool_call: ToolCall) -> Tuple[bool, Optional[str]]:
        """Authorize tool call against tool invocation limits and duplicate detection."""
        ...

    def is_exhausted(self) -> bool:
        """True if any budget ceiling has been exhausted."""
        ...

    def get_exhausted_resource(self) -> Optional[str]:
        """Name of resource that exhausted the budget, if any."""
        ...

    def get_consumption(self) -> BudgetConsumption:
        """Return structured report of resource consumption."""
        ...


@runtime_checkable
class AgentMemoryProtocol(Protocol):
    """Protocol for managing working scratchpad and step audit history."""

    def add_step(self, step: AgentStep) -> None:
        """Append an immutable execution step."""
        ...

    def get_steps(self) -> List[AgentStep]:
        """Retrieve full ordered list of steps executed so far."""
        ...

    def clear(self) -> None:
        """Reset working memory."""
        ...


@runtime_checkable
class AgentLifecycleObserverProtocol(Protocol):
    """Protocol for optional telemetry and execution observation hooks."""

    def on_run_started(self, run_id: str, query: str) -> None:
        """Invoked when agent run begins."""
        ...

    def on_step_completed(self, run_id: str, step: AgentStep) -> None:
        """Invoked when an agent execution step concludes."""
        ...

    def on_run_completed(self, run_id: str, trace: AgentTrace) -> None:
        """Invoked when run finishes and final AgentTrace is generated."""
        ...


@runtime_checkable
class AgentProtocol(Protocol):
    """High-level autonomous academic agent interface."""

    def run(
        self,
        query: str,
        cancellation_token: Optional[Any] = None,
    ) -> AgentTrace:
        """Execute bounded academic research and synthesis for user question."""
        ...
