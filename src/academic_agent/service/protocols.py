"""Service layer protocols and interface contracts."""

from typing import Iterator, List, Optional, Protocol, runtime_checkable

from academic_agent.service.models import (
    QueryRequest,
    QueryResponse,
    RunSummary,
    StreamEvent,
)


@runtime_checkable
class AgentServiceProtocol(Protocol):
    """Protocol satisfied by application service orchestrators managing AcademicAgent."""

    def run_query(self, request: QueryRequest) -> QueryResponse:
        """Execute a blocking query run and return the comprehensive grounded response."""
        ...

    def stream_query(self, request: QueryRequest) -> Iterator[StreamEvent]:
        """Execute query run and yield real-time streaming lifecycle events."""
        ...

    def cancel_run(self, run_id: str, reason: Optional[str] = None) -> bool:
        """Request cooperative cancellation for an active execution run."""
        ...

    def get_run(self, run_id: str) -> Optional[QueryResponse]:
        """Retrieve the response of a completed or active run by ID."""
        ...

    def list_runs(self, limit: int = 50, offset: int = 0) -> List[RunSummary]:
        """List summaries of recent execution runs ordered by recency."""
        ...
