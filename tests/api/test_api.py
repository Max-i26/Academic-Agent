"""Comprehensive test suite for Stage 7.2 FastAPI Serving Layer (REST & SSE)."""

from datetime import datetime, timezone
import json
import threading
from typing import Dict, Iterator, List, Optional
import uuid

from fastapi import status
from fastapi.testclient import TestClient
import pytest

from academic_agent.agent.context import AgentExecutionContext, AgentRunConfig
from academic_agent.agent.dispatcher import ToolDispatcher
from academic_agent.agent.models import (
    AgentExecutionState,
    AgentStep,
    AgentTerminalState,
    AgentTrace,
    BudgetConfig,
    BudgetConsumption,
    ProposedAction,
    ActionType,
    StateTransitionRecord,
    TokenAccounting,
    ToolCall,
    ToolResult,
    ToolSideEffectPolicy,
)
from academic_agent.agent.orchestrator import AcademicAgent
from academic_agent.agent.planner import ReActStepPlanner
from academic_agent.agent.decision_policy import DecisionPolicy
from academic_agent.agent.registry import ToolRegistry
from academic_agent.api.app import create_app
from academic_agent.api.dependencies import reset_dependencies
from academic_agent.api.models import (
    CancelResponse,
    CitationItem,
    EvidenceItem,
    HealthResponse,
    QueryRequest,
    QueryResponse,
    RunSummary,
    ServiceRunStatus,
    StreamEvent,
    StreamEventType,
    TraceStepSummary,
)
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
        return "mock-api-llm"

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
            latency_ms=10.0,
            usage=LLMUsage(prompt_tokens=80, completion_tokens=40, total_tokens=120),
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
                        "score": 0.95,
                        "content": "An AR(1) process is stationary if and only if |phi_1| < 1.",
                    }
                ],
                "count": 1,
            },
            latency_ms=15.0,
        )


class MockAcademicSynthesizerTool:
    """Mock synthesizer tool constructing a fully valid grounded AcademicAnswer."""

    @property
    def name(self) -> str:
        return "academic_synthesizer"

    @property
    def description(self) -> str:
        return "Synthesize grounded answer."

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
            score=0.95,
            dense_score=0.95,
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
            latency_ms=25.0,
            answer_text="An AR(1) process is stationary if |phi_1| < 1 [E1].",
            resolved_answer_text="An AR(1) process is stationary if |phi_1| < 1 [Lecture 4, p. 18].",
            evidence_context=ctx,
            validation=diag,
            policy_result=policy_res,
            raw_response=LLMResponse(
                content="An AR(1) process is stationary if |phi_1| < 1 [E1].",
                role=MessageRole.ASSISTANT,
                model_name="mock-synthesizer",
                latency_ms=25.0,
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
            latency_ms=25.0,
        )


@pytest.fixture(autouse=True)
def clean_dependencies():
    """Ensure global dependencies are reset between tests."""
    reset_dependencies()
    yield
    reset_dependencies()


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


@pytest.fixture
def client(mock_agent_service) -> TestClient:
    """TestClient wired to an application instance with mock_agent_service."""
    app = create_app(service=mock_agent_service)
    return TestClient(app)


# ============================================================================
# Tests: Health & OpenAPI Schemas
# ============================================================================

class TestHealthEndpoint:
    """Verifies service health checks and OpenAPI documentation generation."""

    def test_health_check_returns_ok(self, client):
        response = client.get("/api/v1/health")
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["status"] == "ok"
        assert data["service"] == "academic-agent"
        assert data["version"] == "0.1.0"

    def test_openapi_schema_generation(self, client):
        response = client.get("/openapi.json")
        assert response.status_code == status.HTTP_200_OK
        schema = response.json()
        assert "paths" in schema
        assert "/api/v1/query" in schema["paths"]
        assert "/api/v1/query/stream" in schema["paths"]
        assert "/api/v1/runs/{run_id}/cancel" in schema["paths"]
        assert "/api/v1/runs/{run_id}" in schema["paths"]
        assert "/api/v1/runs" in schema["paths"]
        assert "/api/v1/health" in schema["paths"]


# ============================================================================
# Tests: Synchronous Query Execution
# ============================================================================

class TestQueryEndpoints:
    """Verifies synchronous POST /api/v1/query execution."""

    def test_sync_query_successful_grounded_answer(self, client):
        payload = {
            "query": "What are the stationarity conditions for an AR(1) process?",
            "max_steps": 5,
        }
        response = client.post("/api/v1/query", json=payload)
        assert response.status_code == status.HTTP_200_OK

        data = response.json()
        assert data["status"] == ServiceRunStatus.COMPLETED.value
        assert "TERMINATED_SUCCESS" in data["terminal_state"]
        assert data["answer_text"] is not None
        assert "[Lecture 4, p. 18]" in data["answer_text"]
        assert data["raw_answer_text"] is not None
        assert "[E1]" in data["raw_answer_text"]

        # Citations
        assert len(data["citations"]) == 1
        cit = data["citations"][0]
        assert cit["evidence_id"] == "E1"
        assert cit["citation_label"] == "[Lecture 4, p. 18]"
        assert cit["source_file"] == "Lecture_4.pdf"
        assert cit["source_pages"] == [18]

        # Step summaries
        assert len(data["step_summaries"]) == 2
        assert data["step_summaries"][0]["tool_name"] == "academic_search"
        assert data["step_summaries"][1]["tool_name"] == "academic_synthesizer"

    def test_sync_query_empty_query_rejected_422(self, client):
        payload = {"query": ""}
        response = client.post("/api/v1/query", json=payload)
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_CONTENT
        data = response.json()
        assert "detail" in data

    def test_sync_query_screening_abstention(self, client):
        payload = {"query": "Can you give me a recipe for chocolate cake?"}
        response = client.post("/api/v1/query", json=payload)
        assert response.status_code == status.HTTP_200_OK

        data = response.json()
        assert data["status"] == ServiceRunStatus.ABSTENTION.value
        assert "TERMINATED_ABSTENTION" in data["terminal_state"]
        assert len(data["citations"]) == 0

    def test_sync_query_unhandled_error_yields_500(self):
        class BrokenService(AgentServiceProtocol):
            def run_query(self, request):
                raise RuntimeError("Catastrophic database crash!")

            def stream_query(self, request):
                raise NotImplementedError()

            def cancel_run(self, run_id):
                return False

            def get_run(self, run_id):
                return None

            def list_runs(self, limit=50, offset=0):
                return []

        broken_app = create_app(service=BrokenService())
        broken_client = TestClient(broken_app)

        response = broken_client.post(
            "/api/v1/query",
            json={"query": "Explain AR(1) models."},
        )
        assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
        data = response.json()
        assert data["error"] == "Query execution failed: Catastrophic database crash!"


# ============================================================================
# Tests: Server-Sent Events Streaming
# ============================================================================

class TestStreamQueryEndpoint:
    """Verifies Server-Sent Events streaming on POST /api/v1/query/stream."""

    def test_stream_query_sse_event_progression(self, client):
        payload = {
            "query": "What are the stationarity conditions for an AR(1) process?",
            "max_steps": 5,
        }

        with client.stream("POST", "/api/v1/query/stream", json=payload) as response:
            assert response.status_code == status.HTTP_200_OK
            assert "text/event-stream" in response.headers["content-type"]

            event_names = []
            event_payloads = []

            current_event_type = None
            for line in response.iter_lines():
                if line.startswith("event: "):
                    current_event_type = line[len("event: "):].strip()
                elif line.startswith("data: "):
                    data_str = line[len("data: "):].strip()
                    event_data = json.loads(data_str)
                    event_names.append(current_event_type)
                    event_payloads.append(event_data)
                    current_event_type = None

        assert len(event_names) >= 3
        # First event is run_started
        assert event_names[0] == StreamEventType.RUN_STARTED.value
        assert event_payloads[0]["event_type"] == StreamEventType.RUN_STARTED.value
        assert event_payloads[0]["payload"]["query"] == payload["query"]

        # Intermediate events include tool_completed
        assert StreamEventType.TOOL_COMPLETED.value in event_names

        # Last event is run_completed
        assert event_names[-1] == StreamEventType.RUN_COMPLETED.value
        assert event_payloads[-1]["payload"]["status"] == ServiceRunStatus.COMPLETED.value

    def test_stream_query_empty_query_rejected_422(self, client):
        payload = {"query": ""}
        response = client.post("/api/v1/query/stream", json=payload)
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_CONTENT


# ============================================================================
# Tests: Run Management & Cancellation
# ============================================================================

class TestRunManagementEndpoints:
    """Verifies run lookup, pagination, and cooperative cancellation."""

    def test_get_run_success_and_not_found_404(self, client):
        # 1. Non-existent run -> 404
        resp = client.get("/api/v1/runs/missing_run_id")
        assert resp.status_code == status.HTTP_404_NOT_FOUND
        assert "not found" in resp.json()["detail"].lower()

        # 2. Run query with specific ID
        run_id = f"test_run_{uuid.uuid4().hex[:8]}"
        q_resp = client.post(
            "/api/v1/query",
            json={"query": "Stationarity of AR(1)?", "run_id": run_id},
        )
        assert q_resp.status_code == status.HTTP_200_OK

        # 3. Lookup existing run -> 200
        get_resp = client.get(f"/api/v1/runs/{run_id}")
        assert get_resp.status_code == status.HTTP_200_OK
        data = get_resp.json()
        assert data["run_id"] == run_id
        assert data["status"] == ServiceRunStatus.COMPLETED.value

    def test_list_runs_pagination_and_ordering(self, client):
        # Populate runs
        for i in range(3):
            client.post(
                "/api/v1/query",
                json={
                    "query": f"Stationarity query index {i}?",
                    "run_id": f"list_run_{i}",
                },
            )

        resp = client.get("/api/v1/runs?limit=10&offset=0")
        assert resp.status_code == status.HTTP_200_OK
        runs = resp.json()
        assert len(runs) >= 3
        # Most recent first
        assert runs[0]["run_id"] == "list_run_2"
        assert runs[1]["run_id"] == "list_run_1"
        assert runs[2]["run_id"] == "list_run_0"

        # Offset pagination
        offset_resp = client.get("/api/v1/runs?limit=1&offset=1")
        assert offset_resp.status_code == status.HTTP_200_OK
        offset_runs = offset_resp.json()
        assert len(offset_runs) == 1
        assert offset_runs[0]["run_id"] == "list_run_1"

    def test_list_runs_invalid_pagination_parameters_422(self, client):
        # limit < 1 -> 422
        resp = client.get("/api/v1/runs?limit=0")
        assert resp.status_code == status.HTTP_422_UNPROCESSABLE_CONTENT

        # limit > 100 -> 422
        resp = client.get("/api/v1/runs?limit=101")
        assert resp.status_code == status.HTTP_422_UNPROCESSABLE_CONTENT

        # offset < 0 -> 422
        resp = client.get("/api/v1/runs?offset=-1")
        assert resp.status_code == status.HTTP_422_UNPROCESSABLE_CONTENT

    def test_cancel_non_existent_run_returns_false(self, client):
        resp = client.post("/api/v1/runs/unknown_run/cancel")
        assert resp.status_code == status.HTTP_200_OK
        data = resp.json()
        assert data["run_id"] == "unknown_run"
        assert data["cancelled"] is False

    def test_cancel_active_run_success(self, mock_agent_service):
        started_event = threading.Event()
        block_event = threading.Event()

        orig_run = mock_agent_service.agent.run

        def slow_run(query, cancellation_token=None, config=None, run_id=None, observers=None):
            started_event.set()
            block_event.wait(timeout=5.0)
            return orig_run(
                query,
                cancellation_token=cancellation_token,
                config=config,
                run_id=run_id,
                observers=observers,
            )

        mock_agent_service.agent.run = slow_run
        app = create_app(service=mock_agent_service)
        client = TestClient(app)

        target_run_id = "test_cancel_run"

        def bg_query():
            client.post(
                "/api/v1/query",
                json={"query": "Long running query?", "run_id": target_run_id},
            )

        t = threading.Thread(target=bg_query)
        t.start()

        started_event.wait(timeout=2.0)
        try:
            cancel_resp = client.post(f"/api/v1/runs/{target_run_id}/cancel")
            assert cancel_resp.status_code == status.HTTP_200_OK
            cancel_data = cancel_resp.json()
            assert cancel_data["run_id"] == target_run_id
            assert cancel_data["cancelled"] is True
        finally:
            block_event.set()
            t.join(timeout=5.0)


# ============================================================================
# Tests: CORS & Middleware
# ============================================================================

class TestCorsAndMiddleware:
    """Verifies CORS headers and client origin authorization."""

    def test_cors_headers_with_custom_origins(self, mock_agent_service):
        app = create_app(
            service=mock_agent_service,
            allowed_origins=["http://localhost:3000", "http://localhost:5173"],
        )
        client = TestClient(app)

        response = client.options(
            "/api/v1/health",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "GET",
            },
        )
        assert response.status_code == status.HTTP_200_OK
        assert response.headers["access-control-allow-origin"] == "http://localhost:3000"
        assert response.headers["access-control-allow-credentials"] == "true"

    def test_cors_wildcard_headers(self, mock_agent_service):
        app = create_app(service=mock_agent_service)
        client = TestClient(app)

        response = client.options(
            "/api/v1/health",
            headers={
                "Origin": "http://example.org",
                "Access-Control-Request-Method": "GET",
            },
        )
        assert response.status_code == status.HTTP_200_OK
        assert response.headers["access-control-allow-origin"] == "*"
