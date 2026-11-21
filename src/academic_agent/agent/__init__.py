"""Academic-Agent Orchestration Subsystem (Phase 6)."""

from academic_agent.agent.models import (
    ActionType,
    AgentExecutionState,
    AgentTerminalState,
    AgentState,
    InvalidStateTransitionError,
    VALID_STATE_TRANSITIONS,
    ToolErrorCategory,
    ToolSideEffectPolicy,
    ToolCall,
    ToolResult,
    TokenAccounting,
    StateTransitionRecord,
    ProposedAction,
    AgentStep,
    BudgetConfig,
    BudgetConsumption,
    AgentTrace,
)
from academic_agent.agent.protocols import (
    ToolProtocol,
    ToolRegistryProtocol,
    ToolDispatcherProtocol,
    StepPlannerProtocol,
    ExecutionBudgetTrackerProtocol,
    AgentMemoryProtocol,
    AgentLifecycleObserverProtocol,
    AgentProtocol,
)
from academic_agent.agent.context import (
    CancellationToken,
    AgentRunConfig,
    AgentExecutionContext,
)
from academic_agent.agent.fsm import AgentStateMachine
from academic_agent.agent.budget import ExecutionBudgetTracker, BudgetExceededError
from academic_agent.agent.registry import ToolRegistry
from academic_agent.agent.dispatcher import ToolDispatcher
from academic_agent.agent.parser import StructuralValidationError, StructuredPlannerParser
from academic_agent.agent.prompts import (
    PLANNER_SYSTEM_PROMPT,
    REPAIR_SYSTEM_PROMPT,
    build_planner_prompt,
    build_repair_prompt,
)
from academic_agent.agent.scratchpad import (
    escape_untrusted_content,
    serialize_tool_result,
    wrap_untrusted_observation,
    format_scratchpad,
)
from academic_agent.agent.decision_policy import (
    PolicyRejectionCategory,
    PolicyDecision,
    DecisionPolicy,
)
from academic_agent.agent.planner import ReActStepPlanner
from academic_agent.agent.orchestrator import AcademicAgent
from academic_agent.agent.tools import (
    BaseAgentTool,
    AcademicSearchArgs,
    AcademicSearchTool,
    EvidenceFilterArgs,
    EvidenceFilterTool,
    AcademicSynthesizerArgs,
    AcademicSynthesizerTool,
    ClarificationPolicy,
    QueryAssessment,
    QueryAssessmentCategory,
    EvidenceAssessment,
    EvidenceAssessmentCategory,
)

__all__ = [
    # States & Transitions
    "AgentExecutionState",
    "AgentTerminalState",
    "AgentState",
    "InvalidStateTransitionError",
    "VALID_STATE_TRANSITIONS",
    "AgentStateMachine",
    # Data Models & Policies
    "ActionType",
    "ToolErrorCategory",
    "ToolSideEffectPolicy",
    "ToolCall",
    "ToolResult",
    "TokenAccounting",
    "StateTransitionRecord",
    "ProposedAction",
    "AgentStep",
    "BudgetConfig",
    "BudgetConsumption",
    "AgentTrace",
    # Protocols
    "ToolProtocol",
    "ToolRegistryProtocol",
    "ToolDispatcherProtocol",
    "StepPlannerProtocol",
    "ExecutionBudgetTrackerProtocol",
    "AgentMemoryProtocol",
    "AgentLifecycleObserverProtocol",
    "AgentProtocol",
    # Execution Context & Cancellation
    "CancellationToken",
    "AgentRunConfig",
    "AgentExecutionContext",
    # Budget Engine
    "ExecutionBudgetTracker",
    "BudgetExceededError",
    # Tool Registry & Dispatcher
    "ToolRegistry",
    "ToolDispatcher",
    # Structured Parser & Exceptions
    "StructuralValidationError",
    "StructuredPlannerParser",
    # Prompts
    "PLANNER_SYSTEM_PROMPT",
    "REPAIR_SYSTEM_PROMPT",
    "build_planner_prompt",
    "build_repair_prompt",
    # Scratchpad
    "escape_untrusted_content",
    "serialize_tool_result",
    "wrap_untrusted_observation",
    "format_scratchpad",
    # Decision Policy
    "PolicyRejectionCategory",
    "PolicyDecision",
    "DecisionPolicy",
    # ReAct Step Planner & Orchestrator
    "ReActStepPlanner",
    "AcademicAgent",
    # Academic Tools & Policies
    "BaseAgentTool",
    "AcademicSearchArgs",
    "AcademicSearchTool",
    "EvidenceFilterArgs",
    "EvidenceFilterTool",
    "AcademicSynthesizerArgs",
    "AcademicSynthesizerTool",
    "ClarificationPolicy",
    "QueryAssessment",
    "QueryAssessmentCategory",
    "EvidenceAssessment",
    "EvidenceAssessmentCategory",
]
