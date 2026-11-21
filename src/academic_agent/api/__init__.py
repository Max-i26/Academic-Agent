"""FastAPI REST and SSE serving layer for Academic-Agent."""

from academic_agent.api.app import create_app
from academic_agent.api.models import (
    CancelResponse,
    ErrorResponse,
    HealthResponse,
    QueryRequest,
    QueryResponse,
    RunSummary,
    ServiceRunStatus,
    StreamEvent,
    StreamEventType,
    TraceStepSummary,
)
from academic_agent.api.routes import router

__all__ = [
    "create_app",
    "router",
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
]
