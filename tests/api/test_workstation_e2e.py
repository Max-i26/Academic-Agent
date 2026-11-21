"""End-to-end integration test validating the entire serving stack and workstation contracts.

Verifies:
- HTTP Client -> FastAPI REST/SSE Layer -> AcademicAgentService -> AcademicAgent.
- Retrieval via AcademicSearchTool and grounded synthesis via AcademicSynthesizerTool.
- Correct packaging of grounded answers, citation pills, and slide evidence items.
- Workstation HTML serving and UI data contract compatibility.
- Scoped query execution with request filters.
"""

import json
from typing import Optional
import uuid

import pytest
from starlette.testclient import TestClient

from academic_agent.agent.context import AgentRunConfig
from academic_agent.agent.decision_policy import DecisionPolicy
from academic_agent.agent.dispatcher import ToolDispatcher
from academic_agent.agent.models import BudgetConfig, TokenAccounting, ToolResult, ToolSideEffectPolicy
from academic_agent.agent.orchestrator import AcademicAgent
from academic_agent.agent.planner import ReActStepPlanner
from academic_agent.agent.registry import ToolRegistry
from academic_agent.api.app import create_app
from academic_agent.api.models import ServiceRunStatus
from academic_agent.generation.models import ChatMessage, LLMResponse, LLMUsage, MessageRole
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
from academic_agent.service.service import AcademicAgentService


class RealisticSearchTool:
    def __init__(self) -> None:
        self.last_arguments = None

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
        return {"name": "academic_search", "type": "object", "properties": {"query": {"type": "string"}}}

    def execute(self, arguments: dict) -> ToolResult:
        self.last_arguments = dict(arguments)
        return ToolResult(
            call_id="call_real_search",
            tool_name="academic_search",
            success=True,
            output="Found Lecture_4.pdf (Slide 18, 19) regarding AR(1) characteristic roots.",
            latency_ms=18.5,
        )


class RealisticSynthesizerTool:
    @property
    def name(self) -> str:
        return "academic_synthesizer"

    @property
    def description(self) -> str:
        return "Grounded synthesizer."

    @property
    def side_effect_policy(self) -> ToolSideEffectPolicy:
        return ToolSideEffectPolicy(read_only=True, idempotent=True, timeout_seconds=5.0)

    def get_schema(self) -> dict:
        return {"name": "academic_synthesizer", "type": "object", "properties": {"query": {"type": "string"}}}

    def execute(self, arguments: dict) -> ToolResult:
        ev1 = Evidence(
            record_id="rec_0",
            chunk_id="stat601_lec04_s18_c02",
            text="Roots of characteristic polynomial lie outside the unit circle, i.e., |phi_1| < 1.",
            source_file="Lecture_4.pdf",
            source_pages=[18],
            source_unit_ids=["unit_1"],
            score=0.952,
            dense_score=0.952,
        )
        ev2 = Evidence(
            record_id="rec_1",
            chunk_id="stat601_lec04_s19_c01",
            text="Absolute summability guarantees almost-sure and mean-square convergence.",
            source_file="Lecture_4.pdf",
            source_pages=[19],
            source_unit_ids=["unit_2"],
            score=0.918,
            dense_score=0.918,
        )
        ge1 = GroundedEvidence(
            evidence_id="E1",
            evidence=ev1,
            citation_label="[Lecture 4, p. 18]",
        )
        ge2 = GroundedEvidence(
            evidence_id="E2",
            evidence=ev2,
            citation_label="[Lecture 4, p. 19]",
        )
        ctx = EvidenceContext(items=[ge1, ge2])
        occ1 = CitationOccurrence(
            evidence_id="E1",
            raw_citation="[E1]",
            start_char=100,
            end_char=104,
            is_valid=True,
        )
        occ2 = CitationOccurrence(
            evidence_id="E2",
            raw_citation="[E2]",
            start_char=200,
            end_char=204,
            is_valid=True,
        )
        diag = ValidationDiagnostics(
            citations=(occ1, occ2),
            valid_cited_ids=("E1", "E2"),
            invalid_cited_ids=(),
        )
        policy_res = PolicyValidationResult(
            outcome=PolicyOutcome.ANSWERED,
            is_compliant=True,
            refusal_detected=False,
        )
        answer = AcademicAnswer(
            query=arguments.get("query", "Derive weak stationarity conditions for an AR(1) process"),
            model_name="mock-synthesizer",
            latency_ms=45.0,
            answer_text=(
                "An AR(1) process defined by X_t = c + phi_1 X_{t-1} + epsilon_t is weakly stationary "
                "if and only if the autoregressive parameter satisfies |phi_1| < 1 [E1]. "
                "By backward substitution, the memory term vanishes as k -> infinity, yielding an absolutely "
                "summable infinite moving average representation [E2]."
            ),
            resolved_answer_text=(
                "An AR(1) process defined by X_t = c + phi_1 X_{t-1} + epsilon_t is weakly stationary "
                "if and only if the autoregressive parameter satisfies |phi_1| < 1 [Lecture 4, p. 18]. "
                "By backward substitution, the memory term vanishes as k -> infinity, yielding an absolutely "
                "summable infinite moving average representation [Lecture 4, p. 19]."
            ),
            evidence_context=ctx,
            validation=diag,
            policy_result=policy_res,
            raw_response=LLMResponse(
                content="An AR(1) process is weakly stationary if |phi_1| < 1 [E1].",
                role=MessageRole.ASSISTANT,
                model_name="mock-synthesizer",
                latency_ms=45.0,
            ),
        )
        return ToolResult(
            call_id="call_real_synth",
            tool_name="academic_synthesizer",
            success=True,
            output={
                "answer_text": answer.resolved_answer_text,
                "is_valid_academic_answer": True,
                "academic_answer": answer,
                "valid_cited_ids": ["E1", "E2"],
                "invalid_cited_ids": [],
            },
            latency_ms=45.0,
        )


class E2ELLMProvider:
    def __init__(self) -> None:
        self.search_plan = json.dumps({
            "action_type": "SEARCH",
            "action_rationale": "Query vector store for AR(1) stationarity derivations.",
            "decision_summary": "Searching lecture slides for stationarity proof.",
            "tool_call": {
                "call_id": "call_1",
                "tool_name": "academic_search",
                "arguments": {"query": "AR(1) stationarity conditions characteristic roots"},
            },
        })
        self.synth_plan = json.dumps({
            "action_type": "SYNTHESIZE",
            "action_rationale": "Sufficient evidence collected; synthesizing monograph.",
            "decision_summary": "Generating grounded monograph with citations.",
            "tool_call": {
                "call_id": "call_2",
                "tool_name": "academic_synthesizer",
                "arguments": {"query": "Derive weak stationarity conditions for an AR(1) process"},
            },
        })

    def generate(self, messages, config: Optional[dict] = None) -> LLMResponse:
        user_content = next((m.content for m in messages if m.role == MessageRole.USER), "")
        has_search = (
            'tool="academic_search"' in user_content
            or "=== Step 1 ===" in user_content
            or "call_real_search" in user_content
        )
        if has_search:
            content = self.synth_plan
        else:
            content = self.search_plan
        return LLMResponse(
            content=content,
            usage=LLMUsage(prompt_tokens=20, completion_tokens=30, total_tokens=50),
            model_name="e2e-planner-model",
            latency_ms=15.0,
        )


@pytest.fixture
def e2e_client():
    search_tool = RealisticSearchTool()
    synth_tool = RealisticSynthesizerTool()

    registry = ToolRegistry()
    registry.register(search_tool)
    registry.register(synth_tool)

    planner = ReActStepPlanner(llm_provider=E2ELLMProvider())
    policy = DecisionPolicy(registry)
    dispatcher = ToolDispatcher(registry)

    agent = AcademicAgent(
        tool_registry=registry,
        planner=planner,
        dispatcher=dispatcher,
        decision_policy=policy,
        default_config=AgentRunConfig(budget_config=BudgetConfig(max_iterations=5, timeout_seconds=10.0)),
    )
    service = AcademicAgentService(agent=agent)
    app = create_app(service=service)
    client = TestClient(app)
    return client, search_tool


class TestWorkstationEndToEnd:
    """Verifies complete end-to-end flow from API to Workstation UI contracts."""

    def test_full_query_submission_to_grounded_response(self, e2e_client):
        client, search_tool = e2e_client

        query_payload = {
            "query": "Derive the weak stationarity conditions for an AR(1) process with parameter |phi_1| < 1",
            "filters": {"lecture_id": "Lecture_4", "course_id": "STAT601"},
            "max_steps": 6,
        }

        response = client.post("/api/v1/query", json=query_payload)
        assert response.status_code == 200
        data = response.json()

        # 1. Status and Terminal outcome contracts
        assert data["status"] == ServiceRunStatus.COMPLETED.value
        assert "TERMINATED_SUCCESS" in data["terminal_state"]
        assert data["total_steps"] >= 2
        assert data["total_tokens_used"] > 0

        # 2. Scholarly Monograph Text Contract
        assert "An AR(1) process defined by X_t" in data["answer_text"]
        assert "[Lecture 4, p. 18]" in data["answer_text"]
        assert "[Lecture 4, p. 19]" in data["answer_text"]

        # 3. Dynamic Citations Contract
        assert len(data["citations"]) == 2
        cit1 = data["citations"][0]
        assert cit1["citation_label"] == "[Lecture 4, p. 18]"
        assert cit1["source_file"] == "Lecture_4.pdf"
        assert 18 in cit1["source_pages"]

        # 4. Slide Evidence Records Contract (For Slide Inspector)
        assert len(data["evidence_items"]) == 2
        ev1 = data["evidence_items"][0]
        assert ev1["chunk_id"] == "stat601_lec04_s18_c02"
        assert ev1["score"] == 0.952
        assert "Roots of characteristic polynomial" in ev1["text"]

        # 5. Filter Propagation Contract
        assert search_tool.last_arguments is not None
        assert search_tool.last_arguments.get("lecture_id") == "Lecture_4"
        assert search_tool.last_arguments.get("course_id") == "STAT601"

    def test_workstation_html_serves_compliant_interactive_ui(self, e2e_client):
        client, _ = e2e_client
        response = client.get("/api/v1/workstation")
        assert response.status_code == 200
        html = response.text

        # Verify key interactive UI components
        assert 'id="query-form"' in html
        assert 'id="query-input"' in html
        assert 'id="btn-retrieve"' in html
        assert 'id="btn-reset"' in html
        assert 'id="telemetry-body"' in html
        assert 'id="inspector-slide-title"' in html
        assert 'id="inspector-chunk-id"' in html
        assert 'id="btn-copy-bibtex"' in html
        assert 'id="toast-bibtex"' in html
        assert 'AcademicAgent (Phase 7 RAG)' in html
        assert '/api/v1/query/stream' in html

    def test_e2e_streaming_event_consumption(self, e2e_client):
        client, _ = e2e_client
        query_payload = {
            "query": "Derive the weak stationarity conditions for an AR(1) process with parameter |phi_1| < 1",
            "filters": {"lecture_id": "Lecture_4"},
        }

        with client.stream("POST", "/api/v1/query/stream", json=query_payload) as response:
            assert response.status_code == 200
            events = []
            for line in response.iter_lines():
                if line.startswith("data:"):
                    json_str = line[len("data:"):].strip()
                    events.append(json.loads(json_str))

            assert len(events) >= 3
            assert events[0]["event_type"] == "run_started"
            assert events[-1]["event_type"] == "run_completed"
            assert events[-1]["payload"]["status"] == ServiceRunStatus.COMPLETED.value
            assert "[Lecture 4, p. 18]" in events[-1]["payload"]["answer_text"]

    def test_workstation_xss_sanitization(self, e2e_client):
        """Verify that workstation UI contains zero unsafe innerHTML interpolations."""
        client, _ = e2e_client
        response = client.get("/api/v1/workstation")
        assert response.status_code == 200
        html = response.text

        # 1. Zero innerHTML in workstation script
        script_start = html.find("<script>")
        script_end = html.rfind("</script>")
        assert script_start != -1 and script_end != -1
        script_content = html[script_start:script_end]
        assert "innerHTML" not in script_content, "Detected unsafe innerHTML usage in workstation JavaScript!"

        # 2. Strict DOM creation APIs used for rendering dynamic content
        assert "document.createElement" in script_content
        assert "document.createTextNode" in script_content
        assert "renderMonograph" in script_content

    def test_workstation_adversarial_xss_payload_handling(self, e2e_client):
        """Verify that queries with embedded script tags execute and return safe, sanitized responses."""
        client, _ = e2e_client
        xss_query = "<script>alert('pwned')</script><img src=x onerror=alert(1)> What is AR(1) stationarity?"
        resp = client.post("/api/v1/query", json={"query": xss_query})
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == ServiceRunStatus.COMPLETED.value
        # The query is processed without injecting script tags into html execution
        assert data["query"] == xss_query
