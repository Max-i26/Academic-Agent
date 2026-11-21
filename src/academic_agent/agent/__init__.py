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
    AgentProtocol,
)
from academic_agent.agent.fsm import AgentStateMachine
from academic_agent.agent.budget import ExecutionBudgetTracker, BudgetExceededError
from academic_agent.agent.registry import ToolRegistry
from academic_agent.agent.dispatcher import ToolDispatcher
from academic_agent.agent.parser import StructuralValidationError, StructuredPlannerParser
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
    "AgentProtocol",
    # Budget Engine
    "ExecutionBudgetTracker",
    "BudgetExceededError",
    # Tool Registry & Dispatcher
    "ToolRegistry",
    "ToolDispatcher",
    # Structured Parser & Exceptions
    "StructuralValidationError",
    "StructuredPlannerParser",
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
