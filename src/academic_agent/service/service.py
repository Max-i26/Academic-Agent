"""Application service layer orchestrating AcademicAgent execution runs.

Provides clean separation of concerns:
- Thread-safe lifecycle tracking and run registry.
- Cooperative cancellation propagation via CancellationToken.
- Real-time streaming via event queue bridging AgentLifecycleObserverProtocol.
- Translates engine domain objects into deeply immutable service DTOs.
"""

from collections import OrderedDict
from datetime import datetime, timezone
import logging
import queue
import threading
import time
from typing import Any, Dict, Iterator, List, Optional, Sequence
import uuid

from academic_agent.agent.context import (
    AgentExecutionContext,
    AgentRunConfig,
    CancellationToken,
)
from academic_agent.agent.models import (
    AgentStep,
    AgentTerminalState,
    AgentTrace,
    BudgetConfig,
)
from academic_agent.agent.orchestrator import AcademicAgent
from academic_agent.agent.protocols import AgentLifecycleObserverProtocol
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
from academic_agent.service.protocols import AgentServiceProtocol

logger = logging.getLogger(__name__)


class _StreamingObserver(AgentLifecycleObserverProtocol):
    """Bridge observer routing agent lifecycle callbacks to a thread-safe streaming queue."""

    def __init__(self, event_queue: queue.Queue, run_id: str) -> None:
        self.queue = event_queue
        self.run_id = run_id
        self._step_counter = 0
        self.has_completed = False

    def on_run_started(self, run_id: str, query: str) -> None:
        event = StreamEvent(
            event_id=str(uuid.uuid4()),
            event_type=StreamEventType.RUN_STARTED,
            run_id=run_id,
            timestamp=time.time(),
            payload={"query": query, "started_at": datetime.now(timezone.utc).isoformat()},
        )
        self.queue.put(event)

    def on_step_completed(self, run_id: str, step: AgentStep) -> None:
        self._step_counter += 1
        now = time.time()

        # 1. Tool dispatch/completion notification if tool was called
        if step.tool_call is not None:
            tool_payload: Dict[str, Any] = {
                "tool_name": step.tool_call.tool_name,
                "arguments": dict(step.tool_call.arguments),
            }
            if step.tool_result is not None:
                tool_payload["success"] = step.tool_result.success
                if step.tool_result.success:
                    tool_payload["output_snippet"] = str(step.tool_result.output)[:200]
                else:
                    tool_payload["output_snippet"] = step.tool_result.error_message
                tool_payload["latency_ms"] = step.tool_result.latency_ms

            tool_event = StreamEvent(
                event_id=str(uuid.uuid4()),
                event_type=StreamEventType.TOOL_COMPLETED,
                run_id=run_id,
                timestamp=now,
                step_index=step.step_index,
                payload=tool_payload,
            )
            self.queue.put(tool_event)

        # 2. Synthesis event if this was a synthesis step
        if step.state == "ANSWER_SYNTHESIS":
            synth_event = StreamEvent(
                event_id=str(uuid.uuid4()),
                event_type=StreamEventType.SYNTHESIS_STARTED,
                run_id=run_id,
                timestamp=now,
                step_index=step.step_index,
                payload={"decision_summary": step.decision_summary},
            )
            self.queue.put(synth_event)

        # 3. Overall step completed event
        action_type_val = (
            step.proposed_action.action_type.value if step.proposed_action else "UNKNOWN"
        )
        step_event = StreamEvent(
            event_id=str(uuid.uuid4()),
            event_type=StreamEventType.STEP_COMPLETED,
            run_id=run_id,
            timestamp=now,
            step_index=step.step_index,
            payload={
                "state": step.state,
                "action_type": action_type_val,
                "action_rationale": step.action_rationale,
                "decision_summary": step.decision_summary,
                "latency_ms": step.latency_ms,
                "authorized": step.authorized,
            },
        )
        self.queue.put(step_event)

    def on_run_completed(self, run_id: str, trace: AgentTrace) -> None:
        now = time.time()
        is_cancelled = trace.terminal_state == AgentTerminalState.TERMINATED_CANCELLED.value
        is_error = trace.terminal_state == AgentTerminalState.TERMINATED_ERROR.value

        event_type = (
            StreamEventType.RUN_CANCELLED
            if is_cancelled
            else (StreamEventType.RUN_FAILED if is_error else StreamEventType.RUN_COMPLETED)
        )

        terminal_reason = (
            trace.transitions[-1].reason
            if trace.transitions
            else (trace.refusal_explanation or "Execution concluded.")
        )
        status_map = {
            AgentTerminalState.TERMINATED_SUCCESS.value: ServiceRunStatus.COMPLETED,
            AgentTerminalState.TERMINATED_ABSTENTION.value: ServiceRunStatus.ABSTENTION,
            AgentTerminalState.TERMINATED_BUDGET_EXCEEDED.value: ServiceRunStatus.BUDGET_EXCEEDED,
            AgentTerminalState.TERMINATED_CANCELLED.value: ServiceRunStatus.CANCELLED,
            AgentTerminalState.TERMINATED_ERROR.value: ServiceRunStatus.ERROR,
        }
        status_val = status_map.get(trace.terminal_state, ServiceRunStatus.ERROR).value

        payload: Dict[str, Any] = {
            "status": status_val,
            "terminal_state": trace.terminal_state,
            "terminal_reason": terminal_reason,
            "total_steps": len(trace.steps),
            "total_tokens": trace.budget_consumed.token_usage.total_tokens,
            "duration_ms": trace.total_latency_ms,
        }

        if trace.final_answer is not None:
            payload["answer_text"] = trace.final_answer.resolved_answer_text
            payload["citation_labels"] = [
                item.citation_label for item in trace.final_answer.evidence_context.items
            ]
        elif trace.refusal_explanation is not None:
            payload["refusal_explanation"] = trace.refusal_explanation

        completion_event = StreamEvent(
            event_id=str(uuid.uuid4()),
            event_type=event_type,
            run_id=run_id,
            timestamp=now,
            payload=payload,
        )
        self.has_completed = True
        self.queue.put(completion_event)
        self.queue.put(None)  # Sentinel ending queue iteration


class AcademicAgentService:
    """Production-grade application service coordinating AcademicAgent runs."""

    def __init__(
        self,
        agent: AcademicAgent,
        max_history: int = 500,
    ) -> None:
        self.agent = agent
        self.max_history = max_history

        self._active_tokens: Dict[str, CancellationToken] = {}
        self._active_runs: Dict[str, RunSummary] = {}
        self._completed_runs: OrderedDict[str, QueryResponse] = OrderedDict()
        self._run_summaries: OrderedDict[str, RunSummary] = OrderedDict()
        self._lock = threading.RLock()

    def run_query(
        self,
        request: QueryRequest,
        observers: Optional[Sequence[AgentLifecycleObserverProtocol]] = None,
    ) -> QueryResponse:
        """Execute a blocking query run and return the comprehensive grounded response."""
        run_id = request.run_id or str(uuid.uuid4())
        created_at_iso = datetime.now(timezone.utc).isoformat()
        token = CancellationToken()

        run_summary = RunSummary(
            run_id=run_id,
            query=request.query,
            status=ServiceRunStatus.RUNNING,
            created_at_iso=created_at_iso,
        )

        with self._lock:
            self._active_tokens[run_id] = token
            self._active_runs[run_id] = run_summary

        config = self._build_agent_config(request)

        try:
            trace = self.agent.run(
                query=request.query,
                cancellation_token=token,
                config=config,
                run_id=run_id,
                observers=observers,
            )
            response = self._build_query_response(trace)
        except Exception as exc:
            logger.exception("Service execution exception for run %s: %s", run_id, exc)
            response = QueryResponse(
                run_id=run_id,
                query=request.query,
                status=ServiceRunStatus.ERROR,
                terminal_state=AgentTerminalState.TERMINATED_ERROR.value,
                terminal_reason=f"Service error: {str(exc)}",
                answer_text=f"An unexpected error occurred during execution: {str(exc)}",
            )

        with self._lock:
            self._active_tokens.pop(run_id, None)
            self._active_runs.pop(run_id, None)

            # Record completed response & summary
            self._completed_runs[run_id] = response
            completed_summary = RunSummary(
                run_id=run_id,
                query=request.query,
                status=response.status,
                created_at_iso=created_at_iso,
                completed_at_iso=datetime.now(timezone.utc).isoformat(),
                duration_ms=response.duration_ms,
                terminal_state=response.terminal_state,
                total_tokens=response.total_tokens_used,
                total_steps=response.total_steps,
            )
            self._run_summaries[run_id] = completed_summary

            # Prune bounded history
            while len(self._completed_runs) > self.max_history:
                self._completed_runs.popitem(last=False)
            while len(self._run_summaries) > self.max_history:
                self._run_summaries.popitem(last=False)

        return response

    def stream_query(self, request: QueryRequest) -> Iterator[StreamEvent]:
        """Execute query run and yield real-time streaming lifecycle events."""
        run_id = request.run_id or str(uuid.uuid4())
        # Attach explicit run_id to request so child thread uses it
        req_with_id = request if request.run_id == run_id else request.model_copy(update={"run_id": run_id})

        event_queue: queue.Queue[Optional[StreamEvent]] = queue.Queue()
        observer = _StreamingObserver(event_queue=event_queue, run_id=run_id)

        worker_error: List[Exception] = []

        def _worker() -> None:
            try:
                self.run_query(req_with_id, observers=[observer])
            except Exception as exc:
                logger.exception("Background worker failure in stream_query: %s", exc)
                worker_error.append(exc)
                event_queue.put(
                    StreamEvent(
                        event_id=str(uuid.uuid4()),
                        event_type=StreamEventType.RUN_FAILED,
                        run_id=run_id,
                        timestamp=time.time(),
                        payload={"error": str(exc)},
                    )
                )
            finally:
                if not observer.has_completed:
                    event_queue.put(None)

        worker_thread = threading.Thread(target=_worker, daemon=True, name=f"agent-stream-{run_id[:8]}")
        worker_thread.start()

        try:
            while True:
                item = event_queue.get()
                if item is None:
                    break
                yield item
        except GeneratorExit:
            # Client disconnected or aborted consumption early -> cooperatively cancel run
            logger.info("Client aborted stream for run %s, requesting cooperative cancellation", run_id)
            self.cancel_run(run_id, reason="Client disconnected or closed stream.")
            raise

    def cancel_run(self, run_id: str, reason: Optional[str] = None) -> bool:
        """Request cooperative cancellation for an active execution run."""
        with self._lock:
            token = self._active_tokens.get(run_id)
            if token is not None and not token.is_cancelled:
                token.cancel(reason=reason or "Cancellation requested by service caller.")
                logger.info("Cancelled active run %s: %s", run_id, reason)
                return True
            return False

    def get_run(self, run_id: str) -> Optional[QueryResponse]:
        """Retrieve the response of a completed run by ID."""
        with self._lock:
            return self._completed_runs.get(run_id)

    def list_runs(self, limit: int = 50, offset: int = 0) -> List[RunSummary]:
        """List summaries of recent execution runs ordered by recency."""
        with self._lock:
            # Active runs first, then completed runs newest to oldest
            results = list(self._active_runs.values())
            completed = list(reversed(self._run_summaries.values()))
            results.extend(completed)
            return results[offset : offset + limit]

    def _build_agent_config(self, request: QueryRequest) -> AgentRunConfig:
        """Construct AgentRunConfig applying optional overrides from QueryRequest."""
        base_budget = self.agent.default_config.budget_config

        max_iterations = (
            request.max_steps if request.max_steps is not None else base_budget.max_iterations
        )
        max_total_tokens = (
            request.max_total_tokens
            if request.max_total_tokens is not None
            else base_budget.max_total_tokens
        )
        timeout_seconds = (
            request.timeout_seconds
            if request.timeout_seconds is not None
            else base_budget.timeout_seconds
        )

        budget_cfg = BudgetConfig(
            max_iterations=max_iterations,
            max_total_tokens=max_total_tokens,
            timeout_seconds=timeout_seconds,
            max_tool_invocations=base_budget.max_tool_invocations,
            tool_timeout_seconds=base_budget.tool_timeout_seconds,
            detect_duplicate_calls=base_budget.detect_duplicate_calls,
        )

        return AgentRunConfig(
            budget_config=budget_cfg,
            enable_query_screening=self.agent.default_config.enable_query_screening,
            max_consecutive_rejections=self.agent.default_config.max_consecutive_rejections,
        )

    def _build_query_response(self, trace: AgentTrace) -> QueryResponse:
        """Translate engine AgentTrace into client-facing QueryResponse DTO."""
        # 1. Map terminal state to ServiceRunStatus
        status_map = {
            AgentTerminalState.TERMINATED_SUCCESS.value: ServiceRunStatus.COMPLETED,
            AgentTerminalState.TERMINATED_ABSTENTION.value: ServiceRunStatus.ABSTENTION,
            AgentTerminalState.TERMINATED_BUDGET_EXCEEDED.value: ServiceRunStatus.BUDGET_EXCEEDED,
            AgentTerminalState.TERMINATED_CANCELLED.value: ServiceRunStatus.CANCELLED,
            AgentTerminalState.TERMINATED_ERROR.value: ServiceRunStatus.ERROR,
        }
        status = status_map.get(trace.terminal_state, ServiceRunStatus.ERROR)

        # 2. Extract grounded answer and citations
        answer_text: Optional[str] = None
        raw_answer_text: Optional[str] = None
        citation_items: List[CitationItem] = []
        evidence_items: List[EvidenceItem] = []
        omitted_count = 0

        if trace.final_answer is not None:
            answer_text = trace.final_answer.resolved_answer_text
            raw_answer_text = trace.final_answer.answer_text
            ctx = trace.final_answer.evidence_context

            for ge in ctx.items:
                citation_items.append(
                    CitationItem(
                        evidence_id=ge.evidence_id,
                        citation_label=ge.citation_label,
                        source_file=ge.evidence.source_file,
                        source_pages=tuple(ge.evidence.source_pages),
                        excerpt=ge.evidence.text,
                        unit_ids=tuple(ge.evidence.source_unit_ids),
                    )
                )
                evidence_items.append(
                    EvidenceItem(
                        evidence_id=ge.evidence_id,
                        chunk_id=ge.evidence.chunk_id,
                        record_id=ge.evidence.record_id,
                        source_file=ge.evidence.source_file,
                        source_pages=tuple(ge.evidence.source_pages),
                        text=ge.evidence.text,
                        score=ge.evidence.rerank_score or ge.evidence.dense_score,
                    )
                )
            omitted_count = ctx.dropped_candidates_count
        else:
            answer_text = trace.refusal_explanation or "No answer synthesized."

        # 3. Build step summaries
        step_summaries: List[TraceStepSummary] = []
        for step in trace.steps:
            action_type_val = (
                step.proposed_action.action_type.value if step.proposed_action else "UNKNOWN"
            )
            tool_name_val = step.tool_call.tool_name if step.tool_call else None
            tool_args_val = dict(step.tool_call.arguments) if step.tool_call else {}

            obs_snippet: Optional[str] = None
            is_error = False
            if step.tool_result is not None:
                if step.tool_result.success:
                    obs_snippet = str(step.tool_result.output)[:200]
                else:
                    obs_snippet = step.tool_result.error_message
                    is_error = True
            elif not step.authorized:
                is_error = True

            step_summaries.append(
                TraceStepSummary(
                    step_index=step.step_index,
                    state=step.state,
                    action_type=action_type_val,
                    action_rationale=step.action_rationale,
                    decision_summary=step.decision_summary,
                    tool_name=tool_name_val,
                    tool_arguments=tool_args_val,
                    observation_snippet=obs_snippet,
                    is_error=is_error,
                    latency_ms=step.latency_ms,
                )
            )

        terminal_reason = (
            trace.transitions[-1].reason
            if trace.transitions
            else (trace.refusal_explanation or "Execution concluded.")
        )

        return QueryResponse(
            run_id=trace.run_id,
            query=trace.query,
            status=status,
            terminal_state=trace.terminal_state or "UNKNOWN",
            terminal_reason=terminal_reason,
            answer_text=answer_text,
            raw_answer_text=raw_answer_text,
            citations=tuple(citation_items),
            evidence_items=tuple(evidence_items),
            omitted_candidates_count=omitted_count,
            total_steps=len(trace.steps),
            total_tokens_used=trace.budget_consumed.token_usage.total_tokens,
            duration_ms=trace.total_latency_ms,
            step_summaries=tuple(step_summaries),
        )
