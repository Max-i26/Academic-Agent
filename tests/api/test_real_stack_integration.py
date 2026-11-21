"""Real-stack integration test validating LanceDB vector retrieval and grounded synthesis.

Executes actual LanceDB physical table queries on disk and coordinates the genuine
Stage 5.2 grounding prompt builder, evidence context builder, citation validator,
and answer policy validator through the application service and REST/SSE API.
"""

import json
from pathlib import Path
import shutil
import tempfile
import threading
from typing import List, Optional
import uuid

import numpy as np
import pytest
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
from academic_agent.api.models import ServiceRunStatus
from academic_agent.embeddings.engine import EmbeddingEngine
from academic_agent.generation.models import ChatMessage, LLMConfig, LLMResponse, LLMUsage, MessageRole
from academic_agent.generation.protocols import LLMProviderProtocol
from academic_agent.generation.synthesis.context.evidence_context_builder import EvidenceContextBuilder
from academic_agent.generation.synthesis.prompt.grounding_prompt_builder import GroundingPromptBuilder
from academic_agent.generation.synthesis.sufficiency.assessor import DeterministicSufficiencyAssessor
from academic_agent.generation.synthesis.synthesizer import AcademicSynthesizer
from academic_agent.generation.synthesis.validation.answer_policy_validator import AnswerPolicyValidator
from academic_agent.generation.synthesis.validation.citation_validator import CitationValidator
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
        # Return deterministic normalized 4D vectors
        vecs = []
        for t in texts:
            if "stationarity" in t.lower() or "ar(1)" in t.lower() or "root" in t.lower():
                v = np.array([0.9, 0.1, 0.0, 0.0], dtype=np.float32)
            else:
                v = np.array([0.0, 0.1, 0.9, 0.0], dtype=np.float32)
            norm = np.linalg.norm(v)
            vecs.append(v / norm if norm > 0 else v)
        return np.vstack(vecs)


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


@pytest.fixture
def real_lancedb_stack():
    """Builds a real LanceDB vector store on disk, populates physical records, and wires the entire stack."""
    temp_dir = tempfile.mkdtemp(prefix="academic_lancedb_real_")
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
    app = create_app(service=service)
    client = TestClient(app)

    try:
        yield client, service, store
    finally:
        store.close()
        shutil.rmtree(temp_dir, ignore_errors=True)


class TestRealStackIntegration:
    """Verifies end-to-end execution against physical LanceDB storage and genuine synthesis verification."""

    def test_real_lancedb_vector_search_and_grounded_synthesis_sync(self, real_lancedb_stack):
        client, service, store = real_lancedb_stack

        # 1. Execute synchronous query
        response = client.post(
            "/api/v1/query",
            json={
                "query": "What are the stationarity conditions for an AR(1) process?",
                "filters": {"lecture_id": "Lecture_4.pdf"},
            },
        )
        assert response.status_code == 200
        data = response.json()

        # 2. Verify run status and terminal state
        assert data["status"] == ServiceRunStatus.COMPLETED.value
        assert "TERMINATED_SUCCESS" in data["terminal_state"]

        # 3. Verify grounded answer with resolved citations
        assert "Roots of characteristic polynomial" in data["answer_text"] or "|phi_1| < 1" in data["answer_text"]
        assert "[Lecture 4, p. 18]" in data["answer_text"]

        # 4. Verify citations preserve the genuine LanceDB chunk ID and score
        assert len(data["citations"]) >= 1
        cit = data["citations"][0]
        assert cit["chunk_id"] == "stat601_lec04_s18_c01"
        assert cit["source_file"] == "Lecture_4.pdf"
        assert cit["source_pages"] == [18]
        assert cit["score"] is not None
        assert cit["score"] > 0.0

        # 5. Verify physical evidence items from LanceDB table
        assert len(data["evidence_items"]) >= 1
        ev = data["evidence_items"][0]
        assert ev["chunk_id"] == "stat601_lec04_s18_c01"
        assert ev["record_id"] == "rec_lec04_s18"
        assert ev["source_file"] == "Lecture_4.pdf"

    def test_real_lancedb_vector_search_and_synthesis_sse_streaming(self, real_lancedb_stack):
        client, service, store = real_lancedb_stack

        with client.stream(
            "POST",
            "/api/v1/query/stream",
            json={
                "query": "What are the stationarity conditions for an AR(1) process?",
                "filters": {"lecture_id": "Lecture_4.pdf"},
            },
        ) as stream_resp:
            assert stream_resp.status_code == 200
            events = []
            for line in stream_resp.iter_lines():
                if line.startswith("data:"):
                    raw_data = line[len("data:"):].strip()
                    events.append(json.loads(raw_data))

            assert len(events) >= 3
            # Check schema version
            assert events[0].get("schema_version") == "1.0"
            assert events[-1].get("schema_version") == "1.0"

            # Check completion payload
            completed = events[-1]
            assert completed["event_type"] == "run_completed"
            p = completed["payload"]
            assert p["status"] == ServiceRunStatus.COMPLETED.value
            assert len(p["citations"]) >= 1
            assert p["citations"][0]["chunk_id"] == "stat601_lec04_s18_c01"
            assert p["citations"][0]["score"] > 0.0
