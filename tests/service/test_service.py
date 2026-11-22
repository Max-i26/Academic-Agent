"""Comprehensive test suite for the Academic Agent application service layer.

Verifies:
- Service protocol conformance and deep immutability of DTO models.
- Synchronous grounded query execution, citation packaging, and step summaries.
- Pre-retrieval query screening abstention and refusal handling.
- Cooperative cancellation propagation and graceful cleanup.
- Real-time event streaming with ordered lifecycle hooks.
- Bounded run history tracking, retrieval, and concurrent execution isolation.
"""

from collections import OrderedDict
import json
import threading
import time
from typing import Any, Dict, List, Optional
import uuid

import pytest
from pydantic import ValidationError

from academic_agent.agent.context import AgentExecutionContext, AgentRunConfig, CancellationToken
from academic_agent.agent.decision_policy import DecisionPolicy
from academic_agent.agent.dispatcher import ToolDispatcher
from academic_agent.agent.models import (
    ActionType,
    AgentExecutionState,
    AgentTerminalState,
    AgentTrace,
    BudgetConfig,
    ProposedAction,
    ToolCall,
    ToolResult,
    ToolSideEffectPolicy,
)
from academic_agent.agent.orchestrator import AcademicAgent
from academic_agent.agent.planner import ReActStepPlanner
from academic_agent.agent.registry import ToolRegistry
from academic_agent.generation.models import (
    ChatMessage,
    LLMConfig,
    LLMResponse,
    LLMUsage,
    MessageRole,
)
from academic_agent.generation.synthesis.models import (
    AcademicAnswer,
    CitationOccurrence,
    EvidenceContext,
    GroundedEvidence,
    PolicyOutcome,
    PolicyValidationResult,
    ValidationDiagnostics,
)
from academic_agent.retrieval.models import Evidence
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
from academic_agent.service.service import AcademicAgentService


# ============================================================================
# Test Doubles & Fixtures
# ============================================================================

class MockLLMProvider:
    """Mock LLM provider returning scripted JSON ReAct responses."""

    def __init__(self, responses: List[str]) -> None:
        self.responses = responses
        self.call_count = 0
        self._lock = threading.Lock()

    @property
    def model_name(self) -> str:
        return "mock-service-llm"

    @property
    def context_window(self) -> int:
        return 4096

    @property
    def is_available(self) -> bool:
        return True

    def generate(self, messages: List[ChatMessage], config: Optional[LLMConfig] = None) -> LLMResponse:
        with self._lock:
            user_content = next((m.content for m in messages if m.role == MessageRole.USER), "")
            has_search = 'tool="academic_search"' in user_content or "=== Step 1 ===" in user_content
            if len(self.responses) == 2 and has_search:
                content = self.responses[1]
            elif len(self.responses) == 2 and not has_search:
                content = self.responses[0]
            else:
                if self.call_count < len(self.responses):
                    content = self.responses[self.call_count]
                else:
                    content = self.responses[-1]
                self.call_count += 1

        return LLMResponse(
            content=content,
            role=MessageRole.ASSISTANT,
            model_name=self.model_name,
            latency_ms=15.0,
            usage=LLMUsage(prompt_tokens=100, completion_tokens=50, total_tokens=150),
        )

    def load(self) -> None:
        pass

    def release(self) -> None:
        pass


class MockAcademicSearchTool:
    """Mock search tool returning grounded lecture candidates."""

    @property
    def name(self) -> str:
        return "academic_search"

    @property
    def description(self) -> str:
        return "Search academic materials."

    @property
    def side_effect_policy(self) -> ToolSideEffectPolicy:
        return ToolSideEffectPolicy(read_only=True, idempotent=True, timeout_seconds=5.0)

    def get_schema(self) -> dict:
        return {
            "name": "academic_search",
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        }

    def execute(self, arguments: dict) -> ToolResult:
        return ToolResult(
            call_id="call_search",
            tool_name="academic_search",
            success=True,
            output={
                "candidates": [
                    {
                        "candidate_id": "cand_0",
                        "source_file": "Lecture_4.pdf",
                        "page": 18,
                        "score": 0.92,
                        "content": "An AR(1) process is stationary if and only if |phi_1| < 1.",
                    }
                ],
                "count": 1,
            },
            output_snippet="Found 1 lecture candidate in Lecture_4.pdf",
            latency_ms=20.0,
        )


class MockAcademicSynthesizerTool:
    """Mock synthesizer tool providing an AcademicAnswer with valid citation integrity."""

    @property
    def name(self) -> str:
        return "academic_synthesizer"

    @property
    def description(self) -> str:
        return "Synthesizes grounded answers from retrieved evidence."

    @property
    def side_effect_policy(self) -> ToolSideEffectPolicy:
        return ToolSideEffectPolicy(read_only=True, idempotent=True, timeout_seconds=10.0)

    def get_schema(self) -> dict:
        return {
            "name": "academic_synthesizer",
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        }

    def execute(self, arguments: dict) -> ToolResult:
        ev = Evidence(
            record_id="rec_0",
            chunk_id="chunk_0",
            text="An AR(1) process is stationary if and only if |phi_1| < 1.",
            source_file="Lecture_4.pdf",
            source_pages=[18],
            source_unit_ids=["unit_1"],
            score=0.92,
            dense_score=0.92,
        )
        ge = GroundedEvidence(
            evidence_id="E1",
            evidence=ev,
            citation_label="[Lecture 4, p. 18]",
        )
        ctx = EvidenceContext(items=[ge])
        occ = CitationOccurrence(
            evidence_id="E1",
            raw_citation="[E1]",
            start_char=45,
            end_char=49,
            is_valid=True,
        )
        diag = ValidationDiagnostics(
            citations=(occ,),
            valid_cited_ids=("E1",),
            invalid_cited_ids=(),
        )
        policy_res = PolicyValidationResult(
            outcome=PolicyOutcome.ANSWERED,
            is_compliant=True,
            refusal_detected=False,
        )
        answer = AcademicAnswer(
            query=arguments.get("query", "What are the stationarity conditions for an AR(1) process?"),
            model_name="mock-synthesizer",
            latency_ms=30.0,
            answer_text="An AR(1) process is stationary if |phi_1| < 1 [E1].",
            resolved_answer_text="An AR(1) process is stationary if |phi_1| < 1 [Lecture 4, p. 18].",
            evidence_context=ctx,
            validation=diag,
            policy_result=policy_res,
            raw_response=LLMResponse(
                content="An AR(1) process is stationary if |phi_1| < 1 [E1].",
                role=MessageRole.ASSISTANT,
                model_name="mock-synthesizer",
                latency_ms=30.0,
            ),
        )

        return ToolResult(
            call_id="call_synth",
            tool_name="academic_synthesizer",
            success=True,
            output={
                "answer_text": answer.resolved_answer_text,
                "is_valid_academic_answer": True,
                "academic_answer": answer,
                "valid_cited_ids": ["E1"],
                "invalid_cited_ids": [],
            },
            output_snippet="Synthesized grounded answer [Lecture 4, p. 18]",
            latency_ms=35.0,
        )


@pytest.fixture
def mock_agent_service() -> AcademicAgentService:
    """Constructs a fully wired AcademicAgentService with search and synthesis tools."""
    registry = ToolRegistry()
    registry.register(MockAcademicSearchTool())
    registry.register(MockAcademicSynthesizerTool())

    search_step_json = json.dumps({
        "action_type": "SEARCH",
        "action_rationale": "Search lecture materials for AR(1) stationarity.",
        "decision_summary": "Querying slides for stationarity conditions.",
        "tool_call": {
            "call_id": "call_1",
            "tool_name": "academic_search",
            "arguments": {"query": "AR(1) stationarity condition"},
        },
    })
    synth_step_json = json.dumps({
        "action_type": "SYNTHESIZE",
        "action_rationale": "Evidence collected; synthesizing answer.",
        "decision_summary": "Invoking synthesizer with retrieved context.",
        "tool_call": {
            "call_id": "call_2",
            "tool_name": "academic_synthesizer",
            "arguments": {"query": "What are the stationarity conditions for AR(1)?"},
        },
    })

    provider = MockLLMProvider(responses=[search_step_json, synth_step_json])
    planner = ReActStepPlanner(llm_provider=provider)
    policy = DecisionPolicy(registry)
    dispatcher = ToolDispatcher(registry)

    agent = AcademicAgent(
        tool_registry=registry,
        planner=planner,
        dispatcher=dispatcher,
        decision_policy=policy,
        default_config=AgentRunConfig(
            budget_config=BudgetConfig(max_iterations=5, timeout_seconds=10.0),
            enable_query_screening=True,
        ),
    )

    return AcademicAgentService(agent=agent, max_history=10)


# ============================================================================
# Tests: DTO Data Models & Immutability
# ============================================================================

class TestServiceDataModels:
    """Verifies that all service DTO models enforce strict immutability and schema validation."""

    def test_query_request_validation_and_immutability(self):
        req = QueryRequest(query="Explain stationarity.", max_steps=5, timeout_seconds=15.0)
        assert req.query == "Explain stationarity."
        assert req.max_steps == 5

        with pytest.raises(ValidationError):
            # Empty query must be rejected
            QueryRequest(query="")

        with pytest.raises(ValidationError):
            # Model is frozen
            req.query = "New query"

    def test_query_response_immutability(self):
        cite = CitationItem(
            evidence_id="E1",
            citation_label="[Lecture 4, p. 18]",
            source_file="Lecture_04.pdf",
            source_pages=(18,),
            excerpt="Stationarity conditions.",
            unit_ids=("u1",),
        )
        resp = QueryResponse(
            run_id="run_123",
            query="Query",
            status=ServiceRunStatus.COMPLETED,
            terminal_state="TERMINATED_SUCCESS",
            terminal_reason="Done",
            answer_text="Stationary if |phi| < 1 [Lecture 4, p. 18].",
            citations=(cite,),
        )
        assert resp.citations[0].citation_label == "[Lecture 4, p. 18]"
        assert resp.citations[0].source_pages == (18,)

        with pytest.raises(ValidationError):
            resp.answer_text = "Mutated"

    def test_stream_event_immutability(self):
        event = StreamEvent(
            event_id="evt_1",
            event_type=StreamEventType.RUN_STARTED,
            run_id="run_123",
            timestamp=123456789.0,
            payload={"query": "Test"},
        )
        assert event.event_type == StreamEventType.RUN_STARTED

        with pytest.raises(ValidationError):
            event.event_type = StreamEventType.RUN_COMPLETED


# ============================================================================
# Tests: Service Protocol Conformance
# ============================================================================

class TestServiceProtocolConformance:
    """Verifies that AcademicAgentService conforms to AgentServiceProtocol."""

    def test_conforms_to_agent_service_protocol(self, mock_agent_service):
        assert isinstance(mock_agent_service, AgentServiceProtocol)


# ============================================================================
# Tests: Synchronous Query Execution
# ============================================================================

class TestSynchronousQueryExecution:
    """Verifies end-to-end blocking query execution via AcademicAgentService."""

    def test_run_query_successful_grounded_answer(self, mock_agent_service):
        request = QueryRequest(
            query="What are the stationarity conditions for an AR(1) process?",
            max_steps=5,
        )

        response = mock_agent_service.run_query(request)

        assert response.status == ServiceRunStatus.COMPLETED
        assert response.terminal_state == AgentTerminalState.TERMINATED_SUCCESS.value
        assert "TERMINATED_SUCCESS" in response.terminal_state
        assert response.answer_text is not None
        assert "[Lecture 4, p. 18]" in response.answer_text
        assert response.raw_answer_text is not None
        assert "[E1]" in response.raw_answer_text

        # Citations & Evidence
        assert len(response.citations) == 1
        citation = response.citations[0]
        assert citation.evidence_id == "E1"
        assert citation.citation_label == "[Lecture 4, p. 18]"
        assert citation.source_file == "Lecture_4.pdf"
        assert citation.source_pages == (18,)
        assert "stationary" in citation.excerpt

        assert len(response.evidence_items) == 1
        assert response.evidence_items[0].source_file == "Lecture_4.pdf"

        # Step Summaries
        assert len(response.step_summaries) == 2
        step0 = response.step_summaries[0]
        assert step0.tool_name == "academic_search"
        assert step0.is_error is False
        assert "Lecture_4.pdf" in (step0.observation_snippet or "")

        step1 = response.step_summaries[1]
        assert step1.tool_name == "academic_synthesizer"
        assert step1.is_error is False
        assert step1.state == "ANSWER_SYNTHESIS"

    def test_run_query_screening_rejection_abstains(self, mock_agent_service):
        request = QueryRequest(query="Can you give me a recipe for chocolate cake?")
        response = mock_agent_service.run_query(request)
        print("REASON:", response.terminal_reason)
        print("ANSWER:", response.answer_text)

        assert response.status == ServiceRunStatus.ABSTENTION
        assert response.terminal_state == AgentTerminalState.TERMINATED_ABSTENTION.value
        assert response.answer_text is not None
        assert "domain" in response.answer_text.lower() or "domain" in response.terminal_reason.lower()
        assert len(response.citations) == 0
        assert len(response.evidence_items) == 0
        assert response.total_steps == 0

    def test_run_query_custom_run_id_preserved(self, mock_agent_service):
        custom_id = "test-run-custom-uuid-42"
        request = QueryRequest(
            query="What are the stationarity conditions for an AR(1) process?",
            run_id=custom_id,
        )
        response = mock_agent_service.run_query(request)
        print("REASON:", response.terminal_reason)
        print("ANSWER:", response.answer_text)
        print("DEBUG RESPONSE STATUS:", response.status)
        print("DEBUG TERMINAL STATE:", response.terminal_state)
        print("DEBUG TERMINAL REASON:", response.terminal_reason)
        assert response.run_id == custom_id

    def test_run_query_exception_containment(self, mock_agent_service, monkeypatch):
        def failing_run(*args, **kwargs):
            raise RuntimeError("Engine catastrophic crash")

        monkeypatch.setattr(mock_agent_service.agent, "run", failing_run)

        request = QueryRequest(query="Stationarity conditions?")
        response = mock_agent_service.run_query(request)
        print("REASON:", response.terminal_reason)
        print("ANSWER:", response.answer_text)
        print("DEBUG RESPONSE STATUS:", response.status)
        print("DEBUG TERMINAL STATE:", response.terminal_state)
        print("DEBUG TERMINAL REASON:", response.terminal_reason)

        assert response.status == ServiceRunStatus.ERROR
        assert response.terminal_state == AgentTerminalState.TERMINATED_ERROR.value
        assert "Internal service execution error" in response.terminal_reason
        assert "Engine catastrophic crash" not in response.terminal_reason
        assert "Engine catastrophic crash" not in response.answer_text


# ============================================================================
# Tests: Cooperative Cancellation
# ============================================================================

class TestCooperativeCancellation:
    """Verifies cancellation token signaling and state transitions."""

    def test_cancel_non_existent_run_returns_false(self, mock_agent_service):
        assert mock_agent_service.cancel_run("non_existent_id") is False

    def test_cancel_active_run_cooperatively(self, mock_agent_service, monkeypatch):
        cancel_signal_received = threading.Event()

        def slow_run(query, cancellation_token, **kwargs):
            # Simulate a multi-step execution waiting for cancellation
            while not cancellation_token.is_cancelled:
                time.sleep(0.01)
            cancel_signal_received.set()
            return AgentTrace(
                run_id=kwargs.get("run_id", "test_run"),
                query=query,
                terminal_state=AgentTerminalState.TERMINATED_CANCELLED.value,
                refusal_explanation="Run was interrupted.",
                steps=(),
            )

        monkeypatch.setattr(mock_agent_service.agent, "run", slow_run)

        run_id = "cancellable_run_1"
        req = QueryRequest(query="Stationarity query", run_id=run_id)

        # Launch run in separate thread
        thread = threading.Thread(target=mock_agent_service.run_query, args=(req,))
        thread.start()

        # Allow run to register as active
        time.sleep(0.05)

        # Cancel the active run
        success = mock_agent_service.cancel_run(run_id, reason="User clicked stop.")
        assert success is True

        thread.join(timeout=2.0)
        assert not thread.is_alive()
        assert cancel_signal_received.is_set()

        # Verify recorded response is CANCELLED
        recorded = mock_agent_service.get_run(run_id)
        assert recorded is not None
        assert recorded.status == ServiceRunStatus.CANCELLED
        assert recorded.terminal_state == AgentTerminalState.TERMINATED_CANCELLED.value


# ============================================================================
# Tests: Real-Time Event Streaming
# ============================================================================

class TestRealTimeEventStreaming:
    """Verifies that stream_query yields structured events in chronological sequence."""

    def test_stream_query_full_event_sequence(self, mock_agent_service):
        request = QueryRequest(
            query="What are the stationarity conditions for an AR(1) process?",
            max_steps=5,
        )

        events: List[StreamEvent] = list(mock_agent_service.stream_query(request))

        assert len(events) >= 3

        # 1. First event is RUN_STARTED
        assert events[0].event_type == StreamEventType.RUN_STARTED
        assert events[0].payload["query"] == request.query

        # 2. Intermediate events contain TOOL_COMPLETED and STEP_COMPLETED
        event_types = [e.event_type for e in events]
        assert StreamEventType.TOOL_COMPLETED in event_types
        assert StreamEventType.STEP_COMPLETED in event_types

        # 3. Final event is RUN_COMPLETED
        assert events[-1].event_type == StreamEventType.RUN_COMPLETED
        assert events[-1].payload["terminal_state"] == AgentTerminalState.TERMINATED_SUCCESS.value
        assert "answer_text" in events[-1].payload
        assert "[Lecture 4, p. 18]" in events[-1].payload["answer_text"]

    def test_stream_query_early_generator_exit_triggers_cancellation(self, mock_agent_service, monkeypatch):
        token_cancelled = threading.Event()

        def slow_run(query, cancellation_token, **kwargs):
            observers = kwargs.get("observers") or []
            run_id = kwargs.get("run_id", "stream_run")
            for obs in observers:
                obs.on_run_started(run_id, query)
            for _ in range(50):
                if cancellation_token.is_cancelled:
                    token_cancelled.set()
                    break
                time.sleep(0.02)
            trace = AgentTrace(
                run_id=run_id,
                query=query,
                terminal_state=AgentTerminalState.TERMINATED_CANCELLED.value,
                refusal_explanation="Cancelled",
                steps=(),
            )
            for obs in observers:
                obs.on_run_completed(trace.run_id, trace)
            return trace

        monkeypatch.setattr(mock_agent_service.agent, "run", slow_run)

        req = QueryRequest(query="Streaming test", run_id="stream_exit_run")
        stream_gen = mock_agent_service.stream_query(req)

        # Read first event (RUN_STARTED)
        first_event = next(stream_gen)
        assert first_event.event_type == StreamEventType.RUN_STARTED

        # Abruptly close the stream generator (simulating client disconnection)
        stream_gen.close()

        # Verify that cooperative cancellation was signaled
        assert token_cancelled.wait(timeout=2.0) is True


# ============================================================================
# Tests: Run History, Retrieval & Pruning
# ============================================================================

class TestRunHistoryAndListing:
    """Verifies run caching, inspection, and historical bound enforcement."""

    def test_get_run_retrieval(self, mock_agent_service):
        req = QueryRequest(
            query="What are the stationarity conditions for an AR(1) process?",
            run_id="run_lookup_test",
        )
        assert mock_agent_service.get_run("run_lookup_test") is None

        mock_agent_service.run_query(req)

        retrieved = mock_agent_service.get_run("run_lookup_test")
        assert retrieved is not None
        assert retrieved.run_id == "run_lookup_test"
        assert retrieved.status == ServiceRunStatus.COMPLETED

    def test_list_runs_recency_ordering(self, mock_agent_service):
        for i in range(3):
            req = QueryRequest(
                query=f"Query {i}",
                run_id=f"run_{i}",
            )
            mock_agent_service.run_query(req)

        runs = mock_agent_service.list_runs(limit=2)
        assert len(runs) == 2
        # Most recent first
        assert runs[0].run_id == "run_2"
        assert runs[1].run_id == "run_1"

    def test_max_history_pruning_bounds(self, mock_agent_service):
        # Set small history capacity
        mock_agent_service.max_history = 3

        for i in range(5):
            req = QueryRequest(
                query=f"Query {i}",
                run_id=f"run_prune_{i}",
            )
            mock_agent_service.run_query(req)

        # Capacity strictly bounded at 3
        assert len(mock_agent_service._completed_runs) == 3
        assert len(mock_agent_service._run_summaries) == 3

        # Oldest runs (0, 1) pruned; (2, 3, 4) retained
        assert mock_agent_service.get_run("run_prune_0") is None
        assert mock_agent_service.get_run("run_prune_1") is None
        assert mock_agent_service.get_run("run_prune_4") is not None


# ============================================================================
# Tests: Concurrency & Thread Isolation
# ============================================================================

class TestServiceConcurrency:
    """Verifies that concurrent queries execute safely without state cross-contamination."""

    def test_concurrent_independent_queries(self, mock_agent_service):
        results: Dict[str, QueryResponse] = {}
        errors: List[Exception] = []

        def worker(idx: int):
            try:
                run_id = f"concurrent_run_{idx}"
                req = QueryRequest(
                    query="What are the stationarity conditions for an AR(1) process?",
                    run_id=run_id,
                )
                resp = mock_agent_service.run_query(req)
                results[run_id] = resp
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10.0)

        assert len(errors) == 0
        assert len(results) == 4
        for i in range(4):
            run_id = f"concurrent_run_{i}"
            assert run_id in results
            assert results[run_id].status == ServiceRunStatus.COMPLETED
            assert results[run_id].run_id == run_id


# ============================================================================
# Tests: Phase 7.4 Hardening & Gate C Safety
# ============================================================================


class TestServiceHardening:
    """Verifies atomic run ID reservation, cleanup guarantees, and filter propagation."""

    def test_duplicate_active_run_id_rejected(self, mock_agent_service):
        run_id = "test_dup_active_run"
        with mock_agent_service._lock:
            mock_agent_service._active_tokens[run_id] = CancellationToken()

        assert mock_agent_service.is_active(run_id) is True
        assert mock_agent_service.get_active_count() >= 1

        req = QueryRequest(query="What is AR(1)?", run_id=run_id)
        with pytest.raises(ValueError, match="already actively executing"):
            mock_agent_service.run_query(req)

        with mock_agent_service._lock:
            mock_agent_service._active_tokens.pop(run_id, None)

    def test_duplicate_completed_run_id_rejected(self, mock_agent_service):
        run_id = "test_dup_completed_run"
        req = QueryRequest(query="What is AR(1)?", run_id=run_id)
        resp = mock_agent_service.run_query(req)
        assert resp.run_id == run_id

        with pytest.raises(ValueError, match="has already been executed"):
            mock_agent_service.run_query(req)

    def test_post_registration_cleanup_on_failure(self, mock_agent_service):
        run_id = "test_cleanup_fail_run"
        orig_run = mock_agent_service.agent.run
        try:
            def broken_run(*args, **kwargs):
                raise RuntimeError("Simulated execution failure!")

            mock_agent_service.agent.run = broken_run

            req = QueryRequest(query="What is AR(1)?", run_id=run_id)
            resp = mock_agent_service.run_query(req)
            assert resp.status == ServiceRunStatus.ERROR
            assert "Internal service execution error" in resp.terminal_reason
            assert "Simulated execution failure!" not in resp.terminal_reason
            assert "Simulated execution failure!" not in resp.answer_text

            assert mock_agent_service.is_active(run_id) is False
            assert run_id not in mock_agent_service._active_runs
        finally:
            mock_agent_service.agent.run = orig_run

    def test_filter_propagation_to_tool_arguments(self, mock_agent_service):
        captured_tool_args = []
        original_dispatch = mock_agent_service.agent.dispatcher.dispatch

        def capturing_dispatch(tool_call, budget_tracker=None):
            if tool_call.tool_name == "academic_search":
                captured_tool_args.append(dict(tool_call.arguments))
            return original_dispatch(tool_call, budget_tracker=budget_tracker)

        mock_agent_service.agent.dispatcher.dispatch = capturing_dispatch
        try:
            req = QueryRequest(
                query="What are the stationarity conditions for an AR(1) process?",
                filters={"lecture_id": "Lecture_4", "course_id": "STAT601"},
            )
            resp = mock_agent_service.run_query(req)
            assert resp.status == ServiceRunStatus.COMPLETED
            assert len(captured_tool_args) > 0
            assert captured_tool_args[0].get("lecture_id") == "Lecture_4"
            assert captured_tool_args[0].get("course_id") == "STAT601"
        finally:
            mock_agent_service.agent.dispatcher.dispatch = original_dispatch

    def test_stream_query_async_execution(self, mock_agent_service):
        import asyncio

        async def _test():
            req = QueryRequest(
                query="What are the stationarity conditions for an AR(1) process?",
            )
            events = []
            async for ev in mock_agent_service.stream_query_async(req):
                events.append(ev)

            assert len(events) >= 3
            assert events[0].event_type == StreamEventType.RUN_STARTED
            assert events[-1].event_type == StreamEventType.RUN_COMPLETED
            assert events[-1].payload["status"] == ServiceRunStatus.COMPLETED.value

        asyncio.run(_test())


# ============================================================================
# Tests: Policy Leak Prevention (v0.1.1 Maintenance Freeze)
# ============================================================================

class TestPolicyLeakPrevention:
    """Verifies that ungrounded or policy-rejected synthesis answers are never leaked as accepted."""

    def test_unsupported_synthesis_answer_is_not_leaked_in_query_response(self, mock_agent_service):
        """When synthesizer generates an answer that policy rejects as UNSUPPORTED_ANSWER,
        QueryResponse must report ABSTENTION, contain no citations, and omit the ungrounded text."""
        orig_run = mock_agent_service.agent.run

        unsupported_answer = AcademicAnswer(
            query="What is Paxos consensus?",
            answer_text="Paxos is a distributed consensus algorithm invented by Leslie Lamport.",
            resolved_answer_text="Paxos is a distributed consensus algorithm invented by Leslie Lamport.",
            evidence_context=EvidenceContext(items=()),
            citation_occurrences=(),
            validation=ValidationDiagnostics(
                valid_cited_ids=(),
                invalid_cited_ids=(),
                malformed_citations=(),
                has_valid_citation_integrity=True,
            ),
            policy_result=PolicyValidationResult(
                outcome=PolicyOutcome.UNSUPPORTED_ANSWER,
                is_compliant=False,
                explanation="Unsupported answer: model made claims despite empty evidence context.",
            ),
            raw_response=LLMResponse(
                content="Paxos is a distributed consensus algorithm invented by Leslie Lamport.",
                role=MessageRole.ASSISTANT,
                model_name="mock-synthesizer",
                latency_ms=10.0,
            ),
            model_name="mock-synthesizer",
            latency_ms=10.0,
        )

        def mock_unsupported_run(query, **kwargs):
            return AgentTrace(
                run_id=kwargs.get("run_id", "test_unsupported_run"),
                query=query,
                terminal_state=AgentTerminalState.TERMINATED_ABSTENTION.value,
                refusal_explanation="Synthesizer generated claims unsupported by retrieved evidence.",
                final_answer=unsupported_answer,
                steps=(),
            )

        mock_agent_service.agent.run = mock_unsupported_run
        try:
            req = QueryRequest(query="What is Paxos consensus?")
            resp = mock_agent_service.run_query(req)

            assert resp.status == ServiceRunStatus.ABSTENTION
            assert resp.terminal_state == AgentTerminalState.TERMINATED_ABSTENTION.value
            # The ungrounded answer text must NOT appear as the accepted answer
            assert "Leslie Lamport" not in (resp.answer_text or "")
            assert resp.raw_answer_text is None
            assert len(resp.citations) == 0
            assert len(resp.evidence_items) == 0
            assert "unsupported" in (resp.answer_text or "").lower() or "unsupported" in resp.terminal_reason.lower()
        finally:
            mock_agent_service.agent.run = orig_run

    def test_unsupported_synthesis_answer_is_not_leaked_in_streaming_event(self, mock_agent_service):
        """When synthesizer generates an ungrounded answer, SSE RUN_COMPLETED event must
        set answer_text to None, provide refusal_explanation, and contain zero citations."""
        import asyncio

        unsupported_answer = AcademicAnswer(
            query="What is Paxos consensus?",
            answer_text="Paxos is a distributed consensus algorithm invented by Leslie Lamport.",
            resolved_answer_text="Paxos is a distributed consensus algorithm invented by Leslie Lamport.",
            evidence_context=EvidenceContext(items=()),
            citation_occurrences=(),
            validation=ValidationDiagnostics(
                valid_cited_ids=(),
                invalid_cited_ids=(),
                malformed_citations=(),
                has_valid_citation_integrity=True,
            ),
            policy_result=PolicyValidationResult(
                outcome=PolicyOutcome.UNSUPPORTED_ANSWER,
                is_compliant=False,
                explanation="Unsupported answer: model made claims despite empty evidence context.",
            ),
            raw_response=LLMResponse(
                content="Paxos is a distributed consensus algorithm invented by Leslie Lamport.",
                role=MessageRole.ASSISTANT,
                model_name="mock-synthesizer",
                latency_ms=10.0,
            ),
            model_name="mock-synthesizer",
            latency_ms=10.0,
        )

        def mock_unsupported_run(query, observers=None, **kwargs):
            trace = AgentTrace(
                run_id=kwargs.get("run_id", "test_stream_unsupported"),
                query=query,
                terminal_state=AgentTerminalState.TERMINATED_ABSTENTION.value,
                refusal_explanation="Synthesizer generated claims unsupported by retrieved evidence.",
                final_answer=unsupported_answer,
                steps=(),
            )
            if observers:
                for obs in observers:
                    obs.on_run_completed(trace.run_id, trace)
            return trace

        orig_run = mock_agent_service.agent.run
        mock_agent_service.agent.run = mock_unsupported_run
        try:
            async def _test():
                req = QueryRequest(query="What is Paxos consensus?")
                events = []
                async for ev in mock_agent_service.stream_query_async(req):
                    events.append(ev)

                assert len(events) >= 1
                completed_event = events[-1]
                assert completed_event.event_type == StreamEventType.RUN_COMPLETED
                payload = completed_event.payload
                assert payload["status"] == ServiceRunStatus.ABSTENTION.value
                assert payload.get("answer_text") is None
                assert "unsupported" in payload.get("refusal_explanation", "").lower()
                assert len(payload.get("citations", [])) == 0
                assert len(payload.get("citation_labels", [])) == 0

            asyncio.run(_test())
        finally:
            mock_agent_service.agent.run = orig_run

