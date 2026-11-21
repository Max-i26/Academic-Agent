"""REST and SSE route definitions for the Academic-Agent serving layer."""

import asyncio
import logging
from pathlib import Path
from typing import AsyncGenerator, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.responses import HTMLResponse
from sse_starlette.sse import EventSourceResponse, ServerSentEvent

from academic_agent.api.dependencies import get_agent_service
from academic_agent.api.models import (
    CancelResponse,
    HealthResponse,
    QueryRequest,
    QueryResponse,
    RunSummary,
)
from academic_agent.service.protocols import AgentServiceProtocol

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["academic-agent"])


async def _stream_event_publisher(
    service: AgentServiceProtocol,
    request: QueryRequest,
) -> AsyncGenerator[ServerSentEvent, None]:
    """Async generator yielding ServerSentEvents from the blocking service streaming queue.

    Offloads queue polling to worker threads via asyncio.to_thread to keep the event loop
    non-blocking. Guarantees generator closure on client disconnect.
    """
    gen = iter(service.stream_query(request))
    sentinel = object()
    try:
        while True:
            event = await asyncio.to_thread(next, gen, sentinel)
            if event is sentinel:
                break

            yield ServerSentEvent(
                id=event.event_id,
                event=event.event_type.value,
                data=event.model_dump_json(),
            )
    except asyncio.CancelledError:
        logger.info("SSE client disconnected for query run: %s", request.run_id)
        raise
    finally:
        await asyncio.to_thread(gen.close)


@router.post(
    "/query",
    response_model=QueryResponse,
    status_code=status.HTTP_200_OK,
    summary="Execute Academic Query",
    description="Synchronously executes academic research and synthesis, returning an immutable grounded response.",
)
async def execute_query(
    request: QueryRequest,
    service: AgentServiceProtocol = Depends(get_agent_service),
) -> QueryResponse:
    """Execute student question through the AcademicAgentService."""
    try:
        return await asyncio.to_thread(service.run_query, request)
    except Exception as exc:
        logger.exception("Unexpected error executing query: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Query execution failed: {exc}",
        ) from exc


@router.post(
    "/query/stream",
    response_class=EventSourceResponse,
    status_code=status.HTTP_200_OK,
    summary="Stream Academic Query Events",
    description="Streams real-time execution lifecycle events (search, filtering, synthesis) via Server-Sent Events (SSE).",
)
async def stream_query(
    request: QueryRequest,
    service: AgentServiceProtocol = Depends(get_agent_service),
) -> EventSourceResponse:
    """Stream execution progress for student question via SSE."""
    return EventSourceResponse(
        _stream_event_publisher(service, request),
        media_type="text/event-stream",
    )


@router.post(
    "/runs/{run_id}/cancel",
    response_model=CancelResponse,
    status_code=status.HTTP_200_OK,
    summary="Cancel Query Run",
    description="Cooperatively interrupts an active query execution.",
)
async def cancel_run(
    run_id: str,
    service: AgentServiceProtocol = Depends(get_agent_service),
) -> CancelResponse:
    """Trigger cooperative cancellation for the specified run ID."""
    cancelled = await asyncio.to_thread(service.cancel_run, run_id)
    msg = (
        "Cancellation signal successfully transmitted."
        if cancelled
        else "Run not found or already reached a terminal state."
    )
    return CancelResponse(run_id=run_id, cancelled=cancelled, message=msg)


@router.get(
    "/runs/{run_id}",
    response_model=QueryResponse,
    status_code=status.HTTP_200_OK,
    summary="Get Run Details",
    description="Retrieves the completed QueryResponse for a specific run ID.",
)
async def get_run(
    run_id: str,
    service: AgentServiceProtocol = Depends(get_agent_service),
) -> QueryResponse:
    """Fetch complete immutable query response from run history."""
    response = await asyncio.to_thread(service.get_run, run_id)
    if response is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Run '{run_id}' not found in service history.",
        )
    return response


@router.get(
    "/runs",
    response_model=List[RunSummary],
    status_code=status.HTTP_200_OK,
    summary="List Recent Runs",
    description="Returns a paginated list of recent query run summaries ordered by recency.",
)
async def list_runs(
    limit: int = Query(default=50, ge=1, le=100, description="Maximum number of runs to return."),
    offset: int = Query(default=0, ge=0, description="Offset for pagination."),
    service: AgentServiceProtocol = Depends(get_agent_service),
) -> List[RunSummary]:
    """List recent executions from service history."""
    return await asyncio.to_thread(service.list_runs, limit=limit, offset=offset)


@router.get(
    "/health",
    response_model=HealthResponse,
    status_code=status.HTTP_200_OK,
    summary="Health & Telemetry Check",
    description="Returns service operational status and version telemetry.",
)
async def health_check() -> HealthResponse:
    """Diagnostic readiness and health check."""
    return HealthResponse(status="ok", service="academic-agent", version="0.1.0")


_WORKSTATION_HTML_PATH = Path(__file__).parent.parent / "ui" / "workstation.html"


@router.get(
    "/workstation",
    response_class=HTMLResponse,
    status_code=status.HTTP_200_OK,
    summary="Student Research Workstation UI",
    description="Serves the non-AI-slop scholarly research workstation interface.",
)
async def serve_workstation() -> HTMLResponse:
    """Serve the standalone research workstation HTML."""
    if not _WORKSTATION_HTML_PATH.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Workstation UI template not found on server.",
        )
    return HTMLResponse(content=_WORKSTATION_HTML_PATH.read_text(encoding="utf-8"))


__all__ = ["router"]
