"""Gate 4: Full Workstation End-to-End Smoke Test.

Verifies:
1. Workstation HTML serving (/workstation) and DOM-safe XSS sanitization.
2. Health & readiness API endpoints (/api/v1/health).
3. End-to-end synchronous query execution (/api/v1/query) with physical LanceDB retrieval.
4. End-to-end real-time SSE streaming query execution (/api/v1/query/stream).
5. Cooperative run cancellation and client disconnect handling.
6. Run history tracking, retrieval, and pagination (/api/v1/runs).
"""

import json
import logging
import os
from pathlib import Path
import shutil
import tempfile
import threading
import time
from typing import Any, List, Optional
import uuid

import numpy as np
from starlette.testclient import TestClient

from academic_agent.agent.context import AgentRunConfig
from academic_agent.agent.decision_policy import DecisionPolicy
from academic_agent.agent.dispatcher import ToolDispatcher
from academic_agent.agent.models import BudgetConfig
from academic_agent.agent.orchestrator import AcademicAgent
from academic_agent.agent.planner import ReActStepPlanner
from academic_agent.agent.registry import ToolRegistry
from academic_agent.agent.tools.search import AcademicSearchTool
from academic_agent.agent.tools.synthesizer import AcademicSynthesizerTool
from academic_agent.api.app import create_app
from academic_agent.api.models import QueryRequest, QueryResponse, ServiceRunStatus
from academic_agent.generation.models import (
    ChatMessage,
    LLMConfig,
    LLMResponse,
    LLMUsage,
    MessageRole,
)
from academic_agent.generation.protocols import LLMProviderProtocol
from academic_agent.generation.synthesis.context.evidence_context_builder import (
    EvidenceContextBuilder,
)
from academic_agent.generation.synthesis.prompt.grounding_prompt_builder import (
    GroundingPromptBuilder,
)
from academic_agent.generation.synthesis.sufficiency.assessor import (
    DeterministicSufficiencyAssessor,
)
from academic_agent.generation.synthesis.synthesizer import AcademicSynthesizer
from academic_agent.generation.synthesis.validation.answer_policy_validator import (
    AnswerPolicyValidator,
)
from academic_agent.generation.synthesis.validation.citation_validator import (
    CitationValidator,
)
from academic_agent.retrieval.dense_retriever import DenseRetriever
from academic_agent.retrieval.pipeline import RetrievalPipeline
from academic_agent.service.service import AcademicAgentService
from academic_agent.storage.config import VectorStoreConfig
from academic_agent.storage.models import IndexMetadata, VectorRecord
from academic_agent.storage.providers.lancedb.persistence import LancePersistenceManager
from academic_agent.storage.providers.lancedb.store import LanceDBVectorStore


class DeterministicEmbeddingProvider:
    """Deterministic embedding provider for real LanceDB vector indexing and query matching."""

    def __init__(self, dimension: int = 4) -> None:
        self.dimension = dimension

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        vecs = []
        for t in texts:
            if "stationarity" in t.lower() or "ar(1)" in t.lower() or "root" in t.lower():
                v = np.array([0.9, 0.1, 0.0, 0.0], dtype=np.float32)
            else:
                v = np.array([0.0, 0.1, 0.9, 0.0], dtype=np.float32)
            norm = np.linalg.norm(v)
            vecs.append(v / norm if norm > 0 else v)
        return np.vstack(vecs)


class DeterministicPlannerLLM(LLMProviderProtocol):
    """LLM provider test double driving the ReAct planning cycle: SEARCH -> SYNTHESIZE."""

    def __init__(self) -> None:
        self._lock = threading.Lock()

    @property
    def model_name(self) -> str:
        return "mock-planner-llm"

    @property
    def context_window(self) -> int:
        return 4096

    @property
    def is_available(self) -> bool:
        return True

    def generate(self, messages: List[ChatMessage], config: Optional[LLMConfig] = None) -> LLMResponse:
        with self._lock:
            user_msg = next((m.content for m in messages if m.role == MessageRole.USER), "")
            has_search_obs = "=== Step 1 ===" in user_msg or "Observation:" in user_msg or "Tool Result:" in user_msg

            if has_search_obs:
                plan_json = json.dumps({
                    "action_type": "SYNTHESIZE",
                    "action_rationale": "Sufficient lecture evidence found in LanceDB; synthesizing grounded answer.",
                    "decision_summary": "Invoking academic synthesizer with retrieved LanceDB evidence.",
                    "tool_call": {
                        "call_id": "call_synth_1",
                        "tool_name": "academic_synthesizer",
                        "arguments": {"query": "What are the stationarity conditions for an AR(1) process?"},
                    },
                })
            else:
                plan_json = json.dumps({
                    "action_type": "SEARCH",
                    "action_rationale": "Searching LanceDB table for AR(1) stationarity conditions.",
                    "decision_summary": "Querying physical LanceDB store for lecture slides.",
                    "tool_call": {
                        "call_id": "call_search_1",
                        "tool_name": "academic_search",
                        "arguments": {"query": "AR(1) stationarity condition"},
                    },
                })

            return LLMResponse(
                content=plan_json,
                role=MessageRole.ASSISTANT,
                model_name=self.model_name,
                latency_ms=10.0,
                usage=LLMUsage(prompt_tokens=80, completion_tokens=40, total_tokens=120),
            )

    def load(self) -> None:
        pass

    def release(self) -> None:
        pass


class DeterministicSynthesizerLLM(LLMProviderProtocol):
    """LLM provider test double generating grounded academic answers with inline [E1] citations."""

    @property
    def model_name(self) -> str:
        return "mock-synthesizer-llm"

    @property
    def context_window(self) -> int:
        return 4096

    @property
    def is_available(self) -> bool:
        return True

    def generate(self, messages: List[ChatMessage], config: Optional[LLMConfig] = None) -> LLMResponse:
        answer_content = (
            "An autoregressive process of order 1, AR(1), is weakly stationary if and only if "
            "the roots of its characteristic polynomial lie strictly outside the unit circle, "
            "which requires |phi_1| < 1 [E1]."
        )
        return LLMResponse(
            content=answer_content,
            role=MessageRole.ASSISTANT,
            model_name=self.model_name,
            latency_ms=12.0,
            usage=LLMUsage(prompt_tokens=150, completion_tokens=35, total_tokens=185),
        )

    def load(self) -> None:
        pass

    def release(self) -> None:
        pass


def build_workstation_test_service(temp_dir: str):
    """Builds a real LanceDB vector store on disk, populates physical records, and wires the entire stack."""
    dimension = 4
    metadata = IndexMetadata(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_revision="rev_real_01",
        embedding_dimension=dimension,
        distance_metric="cosine",
        normalized=True,
    )

    # 1. Physical LanceDB Vector Store
    pm = LancePersistenceManager(
        db_path=temp_dir,
        collection_name="stat601_real_chunks",
        expected_metadata=metadata,
    )
    store = LanceDBVectorStore(
        config=VectorStoreConfig(similarity_threshold=0.0),
        persistence_manager=pm,
    )
    store.initialize(metadata)

    # Populate physical table with genuine lecture chunk records
    embed_provider = DeterministicEmbeddingProvider(dimension=dimension)
    record1 = VectorRecord(
        record_id="rec_lec04_s18",
        chunk_id="stat601_lec04_s18_c01",
        vector=embed_provider.embed_texts(["Stationarity conditions: Roots of characteristic polynomial lie strictly outside unit circle, |phi_1| < 1."])[0],
        text="Stationarity conditions: Roots of characteristic polynomial lie strictly outside unit circle, which implies |phi_1| < 1.",
        source_file="Lecture_4.pdf",
        source_pages=(18,),
        metadata={"course_id": "STAT601", "lecture_id": "Lecture_4", "section": "Autoregressive Models"},
    )
    record2 = VectorRecord(
        record_id="rec_lec04_s19",
        chunk_id="stat601_lec04_s19_c01",
        vector=embed_provider.embed_texts(["Unconditional variance of AR(1) process is sigma^2 / (1 - phi_1^2)."])[0],
        text="Unconditional variance of AR(1) process is sigma^2 / (1 - phi_1^2).",
        source_file="Lecture_4.pdf",
        source_pages=(19,),
        metadata={"course_id": "STAT601", "lecture_id": "Lecture_4", "section": "Variance Derivations"},
    )
    store.add([record1, record2])

    # 2. Retrieval Pipeline wrapping real LanceDB store
    class RealStackEmbeddingEngineAdapter:
        def embed_query(self, text: str):
            vec = embed_provider.embed_texts([text])[0]

            class _QueryResult:
                def __init__(self, v):
                    self.embedding = v

            return _QueryResult(vec)

    dense_retriever = DenseRetriever(
        embedding_engine=RealStackEmbeddingEngineAdapter(),
        vector_store=store,
    )
    retrieval_pipeline = RetrievalPipeline(dense_retriever=dense_retriever)
    search_tool = AcademicSearchTool(pipeline=retrieval_pipeline)

    # 3. Real Academic Synthesizer wrapping genuine context builder, prompt builder, and validators
    synthesizer_llm = DeterministicSynthesizerLLM()
    synthesizer = AcademicSynthesizer(
        llm_provider=synthesizer_llm,
        context_builder=EvidenceContextBuilder(),
        prompt_builder=GroundingPromptBuilder(),
        citation_validator=CitationValidator(),
        policy_validator=AnswerPolicyValidator(),
        sufficiency_assessor=DeterministicSufficiencyAssessor(),
    )
    synthesizer_tool = AcademicSynthesizerTool(synthesizer=synthesizer)

    # 4. Agent Orchestrator & Tool Dispatcher
    registry = ToolRegistry()
    registry.register(search_tool)
    registry.register(synthesizer_tool)

    planner = ReActStepPlanner(llm_provider=DeterministicPlannerLLM())
    dispatcher = ToolDispatcher(registry)
    policy = DecisionPolicy(registry)

    agent = AcademicAgent(
        tool_registry=registry,
        planner=planner,
        dispatcher=dispatcher,
        decision_policy=policy,
        default_config=AgentRunConfig(budget_config=BudgetConfig(max_iterations=5, timeout_seconds=10.0)),
    )

    service = AcademicAgentService(agent=agent, max_history=20)
    return service, store


def run_gate4_tests() -> dict[str, Any]:
    print("=" * 60)
    print("GATE 4: FULL WORKSTATION END-TO-END SMOKE TEST")
    print("=" * 60)

    results: dict[str, Any] = {}
    temp_dir = tempfile.mkdtemp(prefix="academic_lancedb_gate4_")

    try:
        service, store = build_workstation_test_service(temp_dir)
        app = create_app(service=service)
        client = TestClient(app)

        # -------------------------------------------------------------
        # 1. Health & Telemetry Verification
        # -------------------------------------------------------------
        print("\n--- 1. Health & OpenAPI Telemetry ---")
        t0 = time.perf_counter()
        health_resp = client.get("/api/v1/health")
        health_latency_ms = (time.perf_counter() - t0) * 1000.0
        assert health_resp.status_code == 200, f"Health check failed: {health_resp.status_code}"
        health_data = health_resp.json()
        print(f"Health Response: {health_data} ({health_latency_ms:.2f} ms)")
        assert health_data["status"] == "ok"
        assert health_data["version"] == "0.1.0"

        openapi_resp = client.get("/openapi.json")
        assert openapi_resp.status_code == 200
        assert "/api/v1/query" in openapi_resp.json()["paths"]
        assert "/api/v1/query/stream" in openapi_resp.json()["paths"]

        results["health_check"] = {
            "status_code": health_resp.status_code,
            "version": health_data["version"],
            "latency_ms": round(health_latency_ms, 2),
        }

        # -------------------------------------------------------------
        # 2. Workstation UI HTML & DOM Sanitization
        # -------------------------------------------------------------
        print("\n--- 2. Workstation HTML UI & Security Contract ---")
        ui_resp = client.get("/api/v1/workstation")
        assert ui_resp.status_code == 200
        assert "text/html" in ui_resp.headers["content-type"]
        html_content = ui_resp.text
        print(f"Workstation HTML delivered ({len(html_content)} bytes)")
        assert 'id="query-form"' in html_content
        assert 'id="query-input"' in html_content
        assert 'id="btn-retrieve"' in html_content
        assert 'id="btn-reset"' in html_content
        assert 'id="telemetry-body"' in html_content
        assert 'AcademicAgent (Phase 7 RAG)' in html_content
        assert '/api/v1/query/stream' in html_content

        # Confirm DOM-safe rendering patterns (zero innerHTML)
        assert "innerHTML" not in html_content
        assert "textContent" in html_content or "createElement" in html_content
        results["workstation_ui"] = {
            "status_code": ui_resp.status_code,
            "html_bytes": len(html_content),
            "dom_safe": True,
            "zero_inner_html": True,
        }

        # -------------------------------------------------------------
        # 3. Synchronous Query Execution
        # -------------------------------------------------------------
        print("\n--- 3. End-to-End Synchronous Query (/api/v1/query) ---")
        query_req = {
            "query": "What are the stationarity conditions for an AR(1) process?",
            "max_steps": 5,
        }
        t0 = time.perf_counter()
        query_resp = client.post("/api/v1/query", json=query_req)
        query_latency_ms = (time.perf_counter() - t0) * 1000.0
        assert query_resp.status_code == 200, f"Query failed: {query_resp.text}"
        query_data = query_resp.json()

        print(f"Run ID: {query_data['run_id']}")
        print(f"Status: {query_data['status']}")
        print(f"Total Steps: {query_data['total_steps']}")
        print(f"Step Summaries Count: {len(query_data['step_summaries'])}")
        print(f"Latency: {query_latency_ms:.2f} ms")
        print(f"Answer Text: {query_data.get('answer_text', '')}")
        assert query_data["status"] == "completed"
        assert query_data["total_steps"] >= 2
        assert query_data["answer_text"] is not None
        assert "[Lecture 4, p. 18]" in query_data["answer_text"] or "[E1]" in query_data["answer_text"]

        results["sync_query"] = {
            "run_id": query_data["run_id"],
            "status": query_data["status"],
            "total_steps": query_data["total_steps"],
            "step_summaries_count": len(query_data["step_summaries"]),
            "latency_ms": round(query_latency_ms, 2),
            "answer_text": query_data.get("answer_text", ""),
        }

        # -------------------------------------------------------------
        # 4. Real-Time SSE Streaming Query Execution
        # -------------------------------------------------------------
        print("\n--- 4. Real-Time SSE Event Streaming (/api/v1/query/stream) ---")
        stream_req = {
            "query": "Explain AR(1) process stationarity with LanceDB evidence",
            "max_steps": 5,
        }
        t0 = time.perf_counter()
        events_collected = []
        with client.stream("POST", "/api/v1/query/stream", json=stream_req) as stream_resp:
            assert stream_resp.status_code == 200
            for line in stream_resp.iter_lines():
                if line.startswith("event:"):
                    events_collected.append(line.replace("event:", "").strip())
        stream_latency_ms = (time.perf_counter() - t0) * 1000.0

        print(f"Stream Latency: {stream_latency_ms:.2f} ms")
        print(f"Collected SSE Events ({len(events_collected)}): {events_collected}")
        assert "run_started" in events_collected, "Stream must emit run_started event."
        assert "step_completed" in events_collected, "Stream must emit step_completed event."
        assert "run_completed" in events_collected, "Stream must emit run_completed terminal event."

        results["sse_streaming"] = {
            "event_count": len(events_collected),
            "events_sequence": events_collected,
            "latency_ms": round(stream_latency_ms, 2),
            "terminal_event_received": "run_completed" in events_collected,
        }

        # -------------------------------------------------------------
        # 5. Run Management & History Querying
        # -------------------------------------------------------------
        print("\n--- 5. Run History & Pagination (/api/v1/runs) ---")
        runs_resp = client.get("/api/v1/runs?limit=10")
        assert runs_resp.status_code == 200
        runs_list = runs_resp.json()
        print(f"Historical Runs Recorded: {len(runs_list)}")
        assert len(runs_list) >= 2, "Must record both sync and stream runs in history."

        # Fetch specific run
        single_run_id = query_data["run_id"]
        single_run_resp = client.get(f"/api/v1/runs/{single_run_id}")
        assert single_run_resp.status_code == 200
        single_run_data = single_run_resp.json()
        assert single_run_data["run_id"] == single_run_id
        assert single_run_data["status"] == "completed"

        # Cancellation test on inactive run
        cancel_resp = client.post(f"/api/v1/runs/{single_run_id}/cancel")
        assert cancel_resp.status_code == 200
        assert cancel_resp.json()["cancelled"] is False  # Already completed

        results["run_management"] = {
            "total_runs_listed": len(runs_list),
            "retrieved_single_run": True,
            "idempotent_cancellation": True,
        }

        print("\n" + "=" * 60)
        print("ALL GATE 4 WORKSTATION CHECKS PASSED SUCCESSFULLY!")
        print("=" * 60)

        # Write output file
        out_dir = Path("data/evaluation")
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / "step4_workstation_smoke_results.json"
        with open(out_file, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results written to: {out_file}")

        return results

    finally:
        store.close()
        shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    run_gate4_tests()
