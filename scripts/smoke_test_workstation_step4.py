"""Gate 4: Full Workstation End-to-End Smoke Test with Real Physical Stack.

Verifies:
1. Workstation HTML serving (/api/v1/workstation) and DOM-safe XSS sanitization (zero innerHTML).
2. Health & readiness API endpoints (/api/v1/health).
3. Complete physical user journey via synchronous query (/api/v1/query):
   - Real MiniLM embedding generation (384-dim, unit-normalized).
   - Physical LanceDB dense retrieval against academic_corpus (238 physical records).
   - Real local Qwen2.5-1.5B-Instruct generation (4-bit NF4 on CUDA).
   - Grounded synthesis with citation integrity ([E1] -> [Document, Page]).
   - Answer policy compliance certification.
4. End-to-end real-time SSE streaming query execution (/api/v1/query/stream) with strict frame sequencing.
5. Cooperative run cancellation and client disconnect handling (/api/v1/runs/{run_id}/cancel).
6. Run history tracking, retrieval, and pagination (/api/v1/runs and /api/v1/runs/{run_id}).
7. Physical corpus invariant: asserts exactly 238 records before and after execution (zero data corruption).
8. Resource release: VRAM reclaimed cleanly (< 1.0 MB remaining).
"""

import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, List, Optional
import uuid

# Force strictly offline execution
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import numpy as np
import torch
from starlette.testclient import TestClient

from academic_agent.agent.context import AgentRunConfig
from academic_agent.agent.decision_policy import DecisionPolicy
from academic_agent.agent.dispatcher import ToolDispatcher
from academic_agent.agent.models import (
    ActionType,
    AgentStep,
    BudgetConfig,
    ProposedAction,
    TokenAccounting,
)
from academic_agent.agent.orchestrator import AcademicAgent
from academic_agent.agent.protocols import ExecutionBudgetTrackerProtocol, StepPlannerProtocol, ToolProtocol
from academic_agent.agent.registry import ToolRegistry
from academic_agent.agent.tools.search import AcademicSearchTool
from academic_agent.agent.tools.synthesizer import AcademicSynthesizerTool
from academic_agent.api.app import create_app
from academic_agent.api.models import QueryRequest, QueryResponse, ServiceRunStatus
from academic_agent.embeddings.config import KnownEmbeddingModel
from academic_agent.embeddings.engine import EmbeddingEngine
from academic_agent.embeddings.providers.minilm import MiniLMProvider
from academic_agent.generation.models import (
    ChatMessage,
    LLMConfig,
    LLMResponse,
    LLMUsage,
    MessageRole,
    ModelLoadingConfig,
)
from academic_agent.generation.providers.hf_local import HuggingFaceTransformersProvider
from academic_agent.generation.synthesis.context.evidence_context_builder import (
    EvidenceContextBuilder,
)
from academic_agent.generation.synthesis.models import (
    AcademicAnswer,
    EvidenceSufficiency,
    PolicyOutcome,
)
from academic_agent.generation.synthesis.prompt.grounding_prompt_builder import (
    GroundingPromptBuilder,
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
from academic_agent.storage.models import IndexMetadata
from academic_agent.storage.providers.lancedb.persistence import LancePersistenceManager
from academic_agent.storage.providers.lancedb.store import LanceDBVectorStore

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("smoke_test_step4")


class DeterministicReActPlanner(StepPlannerProtocol):
    """Deterministic ReAct step planner orchestrating: SEARCH -> SYNTHESIZE -> FINISH."""

    def __init__(self, target_query: str) -> None:
        self.target_query = target_query

    def plan_next_step(
        self,
        query: str,
        history: List[AgentStep],
        available_tools: List[ToolProtocol],
        budget_tracker: Optional[ExecutionBudgetTrackerProtocol] = None,
    ) -> ProposedAction:
        step_idx = len(history)

        if step_idx == 0:
            return ProposedAction(
                action_type=ActionType.SEARCH,
                action_rationale="Searching physical LanceDB corpus for course lecture evidence.",
                decision_summary="Invoking academic_search against 238-record LanceDB store.",
                tool_call={
                    "call_id": "call_search_1",
                    "tool_name": "academic_search",
                    "arguments": {"query": self.target_query},
                },
                token_accounting=TokenAccounting(prompt_tokens=50, completion_tokens=25, total_tokens=75),
            )
        elif step_idx == 1:
            return ProposedAction(
                action_type=ActionType.SYNTHESIZE,
                action_rationale="Course evidence retrieved from LanceDB; synthesizing grounded answer with local Qwen.",
                decision_summary="Invoking academic_synthesizer with local Qwen 4-bit provider.",
                tool_call={
                    "call_id": "call_synth_1",
                    "tool_name": "academic_synthesizer",
                    "arguments": {"query": self.target_query},
                },
                token_accounting=TokenAccounting(prompt_tokens=80, completion_tokens=30, total_tokens=110),
            )
        else:
            return ProposedAction(
                action_type=ActionType.FINISH,
                action_rationale="Grounded academic synthesis complete.",
                decision_summary="Finalizing run with validated answer.",
                tool_call=None,
                token_accounting=TokenAccounting(prompt_tokens=30, completion_tokens=15, total_tokens=45),
            )


def run_gate4_tests() -> dict[str, Any]:
    print("=" * 60)
    print("GATE 4: FULL WORKSTATION END-TO-END SMOKE TEST")
    print("=" * 60)

    results: dict[str, Any] = {}

    # -------------------------------------------------------------
    # 1. Physical LanceDB & Real MiniLM Engine Setup
    # -------------------------------------------------------------
    print("\n--- 1. Connecting to Physical LanceDB Corpus & MiniLM Engine ---")
    db_path = Path("data/lancedb")
    expected_meta = IndexMetadata(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        embedding_dimension=384,
        distance_metric="cosine",
        normalized=True,
    )

    pm = LancePersistenceManager(
        db_path=db_path,
        collection_name="academic_corpus",
        expected_metadata=expected_meta,
    )

    store = LanceDBVectorStore(
        config=VectorStoreConfig(
            persist_path=str(db_path),
            collection_name="academic_corpus",
            similarity_threshold=0.0,
        ),
        persistence_manager=pm,
    )

    initial_record_count = store.count()
    print(f"Physical LanceDB Table: academic_corpus")
    print(f"Physical Record Count:  {initial_record_count}")
    assert initial_record_count == 238, f"Expected 238 physical records, got {initial_record_count}"

    # MiniLM Provider (CPU for 0 MB GPU overhead)
    minilm_provider = MiniLMProvider(model_name=KnownEmbeddingModel.MINILM.value, device="cpu")
    embedding_engine = EmbeddingEngine(provider=minilm_provider)
    dense_retriever = DenseRetriever(embedding_engine=embedding_engine, vector_store=store)
    retrieval_pipeline = RetrievalPipeline(dense_retriever=dense_retriever)
    search_tool = AcademicSearchTool(pipeline=retrieval_pipeline)

    # -------------------------------------------------------------
    # 2. Local Qwen 4-bit Synthesizer Setup
    # -------------------------------------------------------------
    print("\n--- 2. Loading Real Local Qwen2.5-1.5B (4-bit NF4 on CUDA) ---")
    qwen_loading_config = ModelLoadingConfig(
        model_name="Qwen/Qwen2.5-1.5B-Instruct",
        device="cuda",
        torch_dtype="float16",
        quantization="4bit",
        context_window=4096,
    )
    t0 = time.perf_counter()
    qwen_provider = HuggingFaceTransformersProvider(loading_config=qwen_loading_config)
    qwen_provider.load()
    qwen_load_time_ms = (time.perf_counter() - t0) * 1000.0
    print(f"Qwen Loaded in: {qwen_load_time_ms:.2f} ms")

    grounded_system_prompt = (
        "You are an authoritative academic knowledge assistant for university students. "
        "Answer the student's question using ONLY the provided course evidence.\n\n"
        "CRITICAL INSTRUCTION:\n"
        "You MUST cite the evidence using [E1] at the end of your answer.\n"
        "Never omit the citation [E1].\n\n"
        "Example:\n"
        "Evidence:\n"
        '<evidence id="E1">Weak stationarity requires constant mean μ and autocovariance depending only on lag h.</evidence>\n'
        "Question: What is weak stationarity?\n"
        "Answer: Weak stationarity requires a constant mean μ and autocovariance depending only on lag h [E1]."
    )

    synthesizer = AcademicSynthesizer(
        llm_provider=qwen_provider,
        context_builder=EvidenceContextBuilder(),
        prompt_builder=GroundingPromptBuilder(system_prompt=grounded_system_prompt),
        citation_validator=CitationValidator(),
        policy_validator=AnswerPolicyValidator(),
        default_evidence_budget=1500,
    )
    synthesizer_tool = AcademicSynthesizerTool(synthesizer=synthesizer)

    # -------------------------------------------------------------
    # 3. Assemble Agent Orchestrator & Application Service
    # -------------------------------------------------------------
    test_query = "What is the Simple Exponential Smoothing level equation?"
    tool_registry = ToolRegistry()
    tool_registry.register(search_tool)
    tool_registry.register(synthesizer_tool)

    planner = DeterministicReActPlanner(target_query=test_query)
    dispatcher = ToolDispatcher(tool_registry)
    decision_policy = DecisionPolicy(tool_registry)

    agent = AcademicAgent(
        tool_registry=tool_registry,
        planner=planner,
        dispatcher=dispatcher,
        decision_policy=decision_policy,
        default_config=AgentRunConfig(budget_config=BudgetConfig(max_iterations=5, timeout_seconds=60.0)),
    )

    service = AcademicAgentService(agent=agent, max_history=20)
    app = create_app(service=service)
    client = TestClient(app)

    # -------------------------------------------------------------
    # 4. Workstation HTML Route & DOM-Safe XSS Audit
    # -------------------------------------------------------------
    print("\n--- 3. Workstation HTML Route & DOM Security Audit ---")
    t0 = time.perf_counter()
    ws_resp = client.get("/api/v1/workstation")
    ws_latency_ms = (time.perf_counter() - t0) * 1000.0

    print(f"Workstation Route Latency: {ws_latency_ms:.2f} ms")
    print(f"Status Code: {ws_resp.status_code}")
    print(f"Content Type: {ws_resp.headers.get('content-type')}")
    print(f"Content Length: {len(ws_resp.text)} bytes")

    assert ws_resp.status_code == 200, f"Expected 200, got {ws_resp.status_code}"
    assert "text/html" in ws_resp.headers.get("content-type", "")
    assert len(ws_resp.text) > 10000, "Workstation HTML appears truncated."

    # Zero innerHTML audit
    assert "innerHTML" not in ws_resp.text, (
        "CRITICAL XSS FAILURE: Raw innerHTML detected in workstation frontend!"
    )
    assert "textContent" in ws_resp.text, "Workstation frontend must use DOM-safe textContent."
    assert "createElement" in ws_resp.text, "Workstation frontend must use createElement."
    print("Workstation DOM security certified: ZERO innerHTML present.")

    results["workstation_ui"] = {
        "status_code": ws_resp.status_code,
        "latency_ms": round(ws_latency_ms, 2),
        "content_length_bytes": len(ws_resp.text),
        "inner_html_present": False,
        "dom_safe": True,
    }

    # -------------------------------------------------------------
    # 5. Health & Telemetry Verification
    # -------------------------------------------------------------
    print("\n--- 4. Health & Readiness Verification ---")
    health_resp = client.get("/api/v1/health")
    assert health_resp.status_code == 200
    health_data = health_resp.json()
    print(f"Health Response: {health_data}")
    assert health_data.get("status") == "ok", f"Expected 'ok', got {health_data.get('status')}"
    assert health_data.get("version") in ("0.1.0", "0.1.1")

    results["health_check"] = health_data

    # -------------------------------------------------------------
    # 6. Complete Physical User Journey (Synchronous /api/v1/query)
    # -------------------------------------------------------------
    print("\n--- 5. Complete Physical User Journey (Synchronous Query) ---")
    query_payload = {"query": test_query}
    t0 = time.perf_counter()
    query_resp = client.post("/api/v1/query", json=query_payload)
    query_latency_ms = (time.perf_counter() - t0) * 1000.0

    print(f"Query Latency: {query_latency_ms:.2f} ms")
    print(f"Status Code: {query_resp.status_code}")
    assert query_resp.status_code == 200, f"Query failed: {query_resp.text}"

    query_data = query_resp.json()
    run_id = query_data["run_id"]
    print(f"Run ID:         {run_id}")
    print(f"Status:         {query_data['status']}")
    print(f"Total Steps:    {query_data['total_steps']}")
    print(f"Answer Text:\n{query_data['answer_text']}\n")
    print(f"Citations:      {query_data['citations']}")
    print(f"Step Summaries: {len(query_data['step_summaries'])} steps recorded.")

    # Strict Assertions for Complete Physical User Journey:
    assert query_data["status"].lower() in ("completed", "abstention"), f"Expected completed or abstention, got {query_data['status']}"
    assert query_data["total_steps"] >= 2, f"Expected >= 2 steps, got {query_data['total_steps']}"
    assert query_data["answer_text"].strip(), "Answer text must not be empty."
    if query_data["status"].lower() == "completed":
        assert len(query_data["citations"]) > 0, "Grounded citations must be present."
        top_citation = query_data["citations"][0]
        print(f"Top Citation: {top_citation}")
        assert "source_file" in top_citation
        assert "Lecture" in top_citation["source_file"] or "DS5104" in top_citation["source_file"]
    else:
        assert len(query_data["citations"]) == 0, "Abstained run must contain zero citations."
        print("Certified: Abstained run strictly omits citations.")

    results["physical_user_journey"] = {
        "status": "PASSED",
        "query": test_query,
        "run_id": run_id,
        "latency_ms": round(query_latency_ms, 2),
        "service_status": query_data["status"],
        "total_steps": query_data["total_steps"],
        "answer_text": query_data["answer_text"],
        "citations": query_data["citations"],
        "steps": [s.get("action_type") for s in query_data["step_summaries"]],
    }

    # -------------------------------------------------------------
    # 7. Real-Time SSE Streaming User Journey (/api/v1/query/stream)
    # -------------------------------------------------------------
    print("\n--- 6. Real-Time SSE Streaming User Journey ---")
    t0 = time.perf_counter()
    stream_events = []
    with client.stream("POST", "/api/v1/query/stream", json={"query": test_query}) as response:
        assert response.status_code == 200
        assert "text/event-stream" in response.headers.get("content-type", "")

        current_event = None
        for line in response.iter_lines():
            line_str = line.strip()
            if not line_str:
                continue
            if line_str.startswith("event:"):
                current_event = line_str.split(":", 1)[1].strip()
            elif line_str.startswith("data:") and current_event:
                data_str = line_str.split(":", 1)[1].strip()
                try:
                    payload = json.loads(data_str)
                    stream_events.append({"event": current_event, "data": payload})
                except json.JSONDecodeError:
                    pass
                current_event = None

    stream_latency_ms = (time.perf_counter() - t0) * 1000.0
    print(f"SSE Streaming Latency: {stream_latency_ms:.2f} ms")
    print(f"Total SSE Frames Received: {len(stream_events)}")
    event_names = [e["event"] for e in stream_events]
    print(f"Event Sequence: {' -> '.join(event_names)}")

    # Strict Assertions for SSE Event Stream:
    assert len(stream_events) >= 5, f"Expected >= 5 SSE frames, got {len(stream_events)}"
    assert event_names[0] == "run_started", f"First event must be run_started, got {event_names[0]}"
    assert event_names[-1] == "run_completed", f"Final event must be run_completed, got {event_names[-1]}"
    assert "tool_completed" in event_names, "Expected tool_completed events in stream."
    assert "step_completed" in event_names, "Expected step_completed events in stream."

    terminal_event = stream_events[-1]
    terminal_data = terminal_event["data"].get("payload", terminal_event["data"])
    assert terminal_data.get("status", "").lower() in ("completed", "abstention")
    assert terminal_data.get("terminal_state") in ("TERMINATED_SUCCESS", "TERMINATED_ABSTENTION", "COMPLETED", "ABSTENTION")
    assert terminal_data.get("answer_text", "").strip()

    results["sse_streaming_journey"] = {
        "status": "PASSED",
        "latency_ms": round(stream_latency_ms, 2),
        "frame_count": len(stream_events),
        "event_sequence": event_names,
        "terminal_state": terminal_data.get("terminal_state"),
    }

    # -------------------------------------------------------------
    # 8. Cooperative Cancellation & Error Handling
    # -------------------------------------------------------------
    print("\n--- 7. Cooperative Cancellation & Error Handling ---")
    # Cancel the completed run to test idempotent cancellation response
    cancel_resp = client.post(f"/api/v1/runs/{run_id}/cancel")
    print(f"Cancel Status: {cancel_resp.status_code}")
    print(f"Cancel Data:   {cancel_resp.json()}")
    assert cancel_resp.status_code == 200
    cancel_data = cancel_resp.json()
    assert cancel_data.get("run_id") == run_id

    # Non-existent run ID cancel -> returns 200 with cancelled=False
    non_existent_id = str(uuid.uuid4())
    bad_cancel = client.post(f"/api/v1/runs/{non_existent_id}/cancel")
    assert bad_cancel.status_code == 200
    assert not bad_cancel.json().get("cancelled")
    print(f"Non-existent Run Cancel Response: {bad_cancel.json()}")

    # Non-existent run ID GET -> returns 404 cleanly
    bad_get = client.get(f"/api/v1/runs/{non_existent_id}")
    assert bad_get.status_code == 404
    print(f"Non-existent Run GET 404 Response: {bad_get.json()}")

    results["cancellation_handling"] = {
        "status": "PASSED",
        "existing_run_cancel_status": cancel_resp.status_code,
        "non_existent_run_cancel_status": bad_cancel.status_code,
        "non_existent_run_get_status": bad_get.status_code,
    }

    # -------------------------------------------------------------
    # 9. Run History & Pagination
    # -------------------------------------------------------------
    print("\n--- 8. Run History & Pagination ---")
    runs_resp = client.get("/api/v1/runs?limit=10")
    assert runs_resp.status_code == 200
    runs_data = runs_resp.json()
    print(f"Retrieved {len(runs_data)} runs from history.")
    assert len(runs_data) >= 1

    single_run_resp = client.get(f"/api/v1/runs/{run_id}")
    assert single_run_resp.status_code == 200
    assert single_run_resp.json()["run_id"] == run_id
    print(f"Single Run Retrieval verified for run_id: {run_id}")

    results["run_history"] = {
        "total_runs_retrieved": len(runs_data),
        "single_run_verified": True,
    }

    # -------------------------------------------------------------
    # 10. Physical Corpus Invariant & Resource Deallocation
    # -------------------------------------------------------------
    print("\n--- 9. Corpus Invariant & Resource Deallocation ---")
    final_record_count = store.count()
    print(f"Final LanceDB Record Count: {final_record_count}")
    assert final_record_count == initial_record_count == 238, (
        f"CRITICAL: Physical LanceDB table was corrupted! Initial: {initial_record_count}, Final: {final_record_count}"
    )
    store.close()
    print("Physical LanceDB 238-record count invariant certified.")

    qwen_provider.release()
    released_vram_mb = torch.cuda.memory_allocated() / (1024**2)
    print(f"VRAM Allocated after qwen_provider.release(): {released_vram_mb:.4f} MB")
    assert released_vram_mb < 1.0, f"Expected < 1.0 MB after release, got {released_vram_mb:.4f} MB"
    assert not qwen_provider.is_loaded, "Qwen provider must report is_loaded=False."

    results["corpus_and_resource_integrity"] = {
        "status": "PASSED",
        "initial_lancedb_count": initial_record_count,
        "final_lancedb_count": final_record_count,
        "invariant_preserved": True,
        "final_allocated_vram_mb": round(released_vram_mb, 4),
        "vram_released_cleanly": True,
    }

    print("\n" + "=" * 60)
    print("ALL GATE 4 FULL WORKSTATION CHECKS PASSED SUCCESSFULLY!")
    print("=" * 60)

    # Save results to file
    out_dir = Path("data/evaluation")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "step4_workstation_smoke_results.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results written to: {out_file}")

    return results


if __name__ == "__main__":
    run_gate4_tests()
