"""Data transfer objects and serialization models for the Academic Agent service layer.

Strictly isolates internal engine types (FSM, traces, vector store chunks, AST nodes)
from API callers and client interfaces. All models are frozen and immutable.
"""

from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
from pydantic import BaseModel, ConfigDict, Field


class ServiceRunStatus(str, Enum):
    """Normalized high-level status of an agent execution run."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    ABSTENTION = "abstention"
    BUDGET_EXCEEDED = "budget_exceeded"
    ERROR = "error"
    CANCELLED = "cancelled"


class StreamEventType(str, Enum):
    """Categorized event types emitted during real-time streaming."""

    RUN_STARTED = "run_started"
    STEP_STARTED = "step_started"
    TOOL_DISPATCHED = "tool_dispatched"
    TOOL_COMPLETED = "tool_completed"
    SYNTHESIS_STARTED = "synthesis_started"
    STEP_COMPLETED = "step_completed"
    RUN_COMPLETED = "run_completed"
    RUN_FAILED = "run_failed"
    RUN_CANCELLED = "run_cancelled"


class QueryRequest(BaseModel):
    """Incoming student or client query request."""

    model_config = ConfigDict(frozen=True)

    query: str = Field(..., min_length=1, description="Student academic query or research prompt.")
    filters: Optional[Dict[str, Any]] = Field(default=None, description="Optional metadata filters (course, lecture, etc.).")
    max_steps: Optional[int] = Field(default=None, ge=1, le=50, description="Optional limit on reasoning iterations.")
    max_total_tokens: Optional[int] = Field(default=None, ge=100, description="Optional ceiling on cumulative token consumption.")
    timeout_seconds: Optional[float] = Field(default=None, gt=0.0, description="Optional global execution timeout.")
    session_id: Optional[str] = Field(default=None, description="Optional client session identifier.")
    run_id: Optional[str] = Field(default=None, description="Optional client-specified run ID (UUID generated if omitted).")


class CitationItem(BaseModel):
    """Grounded citation payload presented to the student with provenance."""

    model_config = ConfigDict(frozen=True)

    evidence_id: str = Field(..., description="Evidence identifier (e.g., E1).")
    citation_label: str = Field(..., description="Formatted bibliographic label (e.g., [Lecture 4, p. 18]).")
    source_file: str = Field(..., description="Source document file name.")
    source_pages: Tuple[int, ...] = Field(default_factory=tuple, description="Relevant source page numbers.")
    excerpt: str = Field(..., description="Exact textual excerpt from the source material.")
    chunk_id: Optional[str] = Field(default=None, description="Underlying storage chunk ID.")
    score: Optional[float] = Field(default=None, description="Relevance or retrieval score.")
    unit_ids: Tuple[str, ...] = Field(default_factory=tuple, description="Underlying content unit IDs.")


class EvidenceItem(BaseModel):
    """Retrieved evidence record packaged in the grounded context."""

    model_config = ConfigDict(frozen=True)

    evidence_id: str = Field(..., description="Evidence identifier (e.g., E1).")
    chunk_id: str = Field(..., description="Underlying storage chunk ID.")
    record_id: str = Field(..., description="Full record identifier.")
    source_file: str = Field(..., description="Source file name.")
    source_pages: Tuple[int, ...] = Field(default_factory=tuple, description="Source page numbers.")
    text: str = Field(..., description="Evidence content text.")
    score: Optional[float] = Field(default=None, description="Relevance or reranking score.")


class TraceStepSummary(BaseModel):
    """Sanitized summary of an agent reasoning and action step for client inspection."""

    model_config = ConfigDict(frozen=True)

    step_index: int = Field(..., description="Zero-based execution step index.")
    state: str = Field(..., description="Agent state during execution of this step.")
    action_type: str = Field(..., description="Proposed action type (SEARCH, FILTER, SYNTHESIZE, etc.).")
    action_rationale: Optional[str] = Field(default=None, description="Planner rationale for proposing this action.")
    decision_summary: Optional[str] = Field(default=None, description="Summary of action outcome or decision.")
    tool_name: Optional[str] = Field(default=None, description="Name of tool invoked, if any.")
    tool_arguments: Dict[str, Any] = Field(default_factory=dict, description="Arguments passed to tool invocation.")
    observation_snippet: Optional[str] = Field(default=None, description="Truncated observation or error output.")
    is_error: bool = Field(default=False, description="Whether this step resulted in tool error or policy rejection.")
    latency_ms: float = Field(default=0.0, description="Step duration in milliseconds.")


class QueryResponse(BaseModel):
    """Comprehensive, grounded response to an academic query."""

    model_config = ConfigDict(frozen=True)

    run_id: str = Field(..., description="Unique run identifier.")
    query: str = Field(..., description="Original academic query submitted.")
    status: ServiceRunStatus = Field(..., description="High-level execution status.")
    terminal_state: str = Field(..., description="Agent state machine terminal state.")
    terminal_reason: str = Field(..., description="Transition rationale explaining the terminal outcome.")
    answer_text: Optional[str] = Field(default=None, description="Resolved human-readable answer with formatted citations.")
    raw_answer_text: Optional[str] = Field(default=None, description="Raw synthesizer answer text containing [E1] markers.")
    citations: Tuple[CitationItem, ...] = Field(default_factory=tuple, description="Resolved citation references.")
    evidence_items: Tuple[EvidenceItem, ...] = Field(default_factory=tuple, description="Evidence items retained in context.")
    omitted_candidates_count: int = Field(default=0, description="Number of candidate evidence items omitted due to budget/overlap.")
    total_steps: int = Field(default=0, description="Total reasoning steps executed.")
    total_tokens_used: int = Field(default=0, description="Cumulative tokens consumed across planning and synthesis.")
    duration_ms: float = Field(default=0.0, description="Total execution wall-clock time in milliseconds.")
    step_summaries: Tuple[TraceStepSummary, ...] = Field(default_factory=tuple, description="Step audit trail.")


class StreamEvent(BaseModel):
    """Individual real-time event yielded during streaming execution."""

    model_config = ConfigDict(frozen=True)

    event_id: str = Field(..., description="Unique ID for this event.")
    event_type: StreamEventType = Field(..., description="Category of event.")
    run_id: str = Field(..., description="Associated run ID.")
    timestamp: float = Field(..., description="POSIX timestamp when event was emitted.")
    schema_version: str = Field(default="1.0", description="Event protocol schema version.")
    step_index: Optional[int] = Field(default=None, description="Associated step index, if step-scoped.")
    payload: Dict[str, Any] = Field(default_factory=dict, description="Event-specific metadata or payload.")


class RunSummary(BaseModel):
    """Compact summary of an execution run for dashboard/history inspection."""

    model_config = ConfigDict(frozen=True)

    run_id: str = Field(..., description="Unique run identifier.")
    query: str = Field(..., description="Submitted query text.")
    status: ServiceRunStatus = Field(..., description="Run status.")
    created_at_iso: str = Field(..., description="ISO 8601 start timestamp.")
    completed_at_iso: Optional[str] = Field(default=None, description="ISO 8601 completion timestamp.")
    duration_ms: Optional[float] = Field(default=None, description="Total duration in milliseconds.")
    terminal_state: Optional[str] = Field(default=None, description="Terminal agent state.")
    total_tokens: int = Field(default=0, description="Tokens consumed.")
    total_steps: int = Field(default=0, description="Steps executed.")
