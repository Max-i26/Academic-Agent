"""API response models and schemas for Academic-Agent REST and SSE serving layer."""

from typing import Any, List, Optional
from pydantic import BaseModel, ConfigDict, Field

from academic_agent.service.models import (
    CitationItem,
    EvidenceItem,
    QueryRequest,
    QueryResponse,
    RunSummary,
    ServiceRunStatus,
    StreamEvent,
    StreamEventType,
    TraceStepSummary,
)


class CancelResponse(BaseModel):
    """Response returned when requesting cancellation of an active query run."""

    model_config = ConfigDict(frozen=True)

    run_id: str = Field(description="Identifier of the targeted query run.")
    cancelled: bool = Field(description="True if an active execution was found and signalled.")
    message: str = Field(description="Human-readable cancellation status summary.")


class HealthResponse(BaseModel):
    """Health and readiness diagnostic payload."""

    model_config = ConfigDict(frozen=True)

    status: str = Field(default="ok", description="Service readiness indicator.")
    service: str = Field(default="academic-agent", description="Service identifier.")
    version: str = Field(default="0.1.0", description="Academic-Agent version string.")
    active_runs: Optional[int] = Field(default=None, description="Number of currently active execution runs.")
    registered_tools: Optional[List[str]] = Field(default=None, description="Registered tool identifiers.")


class ErrorResponse(BaseModel):
    """Standardized error envelope for API failures."""

    model_config = ConfigDict(frozen=True)

    error: str = Field(description="High-level error classification.")
    detail: Optional[Any] = Field(default=None, description="Detailed failure diagnostics or trace.")


__all__ = [
    "CancelResponse",
    "HealthResponse",
    "ErrorResponse",
    "QueryRequest",
    "QueryResponse",
    "RunSummary",
    "ServiceRunStatus",
    "StreamEvent",
    "StreamEventType",
    "TraceStepSummary",
    "CitationItem",
    "EvidenceItem",
]
