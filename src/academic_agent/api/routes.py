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
    """Async generator yielding ServerSentEvents using native async queue bridge.

    Completely eliminates worker thread generator advancement and closure races.
    When a client disconnects, asyncio cancellation triggers graceful cleanup
    in the service without raising ValueError: generator already executing.
    """
    if hasattr(service, "stream_query_async"):
        try:
            async for event in service.stream_query_async(request):
                yield ServerSentEvent(
                    id=event.event_id,
                    event=event.event_type.value,
                    data=event.model_dump_json(),
                )
        except asyncio.CancelledError:
            logger.info("SSE client disconnected for query run: %s", request.run_id)
            raise
    else:
        # Fallback for sync-only mock service implementations
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
            try:
                await asyncio.to_thread(gen.close)
            except Exception:
                pass


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
    if request.run_id:
        if hasattr(service, "is_active") and service.is_active(request.run_id):
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Run ID '{request.run_id}' is already actively executing.",
            )
        existing = await asyncio.to_thread(service.get_run, request.run_id)
        if existing is not None:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Run ID '{request.run_id}' has already been executed.",
            )
    try:
        return await asyncio.to_thread(service.run_query, request)
    except ValueError as exc:
        if "already" in str(exc):
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=str(exc),
            ) from exc
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        logger.exception("Unexpected error executing query: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An internal error occurred during query execution.",
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
    if request.run_id:
        if hasattr(service, "is_active") and service.is_active(request.run_id):
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Run ID '{request.run_id}' is already actively executing.",
            )
        existing = await asyncio.to_thread(service.get_run, request.run_id)
        if existing is not None:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Run ID '{request.run_id}' has already been executed.",
            )
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
async def health_check(
    service: AgentServiceProtocol = Depends(get_agent_service),
) -> HealthResponse:
    """Diagnostic readiness and health check verifying underlying agent service and tools."""
    try:
        active_count = None
        tools = None
        if hasattr(service, "get_active_count"):
            active_count = service.get_active_count()
        elif hasattr(service, "_active_tokens"):
            active_count = len(service._active_tokens)

        if hasattr(service, "agent"):
            agent = service.agent
            if hasattr(agent, "tool_registry") and hasattr(agent.tool_registry, "list_tools"):
                tools = [t.name for t in agent.tool_registry.list_tools()]
            elif hasattr(agent, "dispatcher") and hasattr(agent.dispatcher, "registry") and hasattr(agent.dispatcher.registry, "list_tools"):
                tools = [t.name for t in agent.dispatcher.registry.list_tools()]

        return HealthResponse(
            status="ok",
            service="academic-agent",
            version="0.1.0",
            active_runs=active_count,
            registered_tools=tools,
        )
    except Exception as exc:
        logger.exception("Health readiness check failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service readiness check failed.",
        ) from exc


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
