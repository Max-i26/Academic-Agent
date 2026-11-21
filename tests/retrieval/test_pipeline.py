"""Unit and integration tests for RetrievalPipeline orchestration."""

import pytest
import numpy as np

from academic_agent.storage.models import VectorRecord, FilterQuery
from academic_agent.retrieval.models import (
    RetrievalQuery,
    RetrievalCandidate,
    Evidence,
    RetrievalStrategy,
    RetrievalResponse,
)
from academic_agent.retrieval.protocols import (
    RetrievalPipelineProtocol,
    RetrieverProtocol,
    DenseRetrieverProtocol,
    SparseRetrieverProtocol,
    RerankerProtocol,
)
from academic_agent.retrieval.pipeline import RetrievalPipeline
from academic_agent.retrieval.fusion import ReciprocalRankFusion


class MockDenseRetriever:
    """Mock dense retriever returning predetermined candidates."""

    @property
    def name(self) -> str:
        return "dense"

    def retrieve(self, query: RetrievalQuery) -> list[RetrievalCandidate]:
        return [
            RetrievalCandidate(
                record_id="c1",
                chunk_id="c1",
                rank=1,
                retrieval_method=self.name,
                score=0.92,
                text="Dense passage on SES smoothing",
                source_file="Lecture4.pdf",
                source_pages=[7],
                source_unit_ids=["u10"],
                section_title="THE SES EQUATION",
            ),
            RetrievalCandidate(
                record_id="c2",
                chunk_id="c2",
                rank=2,
                retrieval_method=self.name,
                score=0.81,
                text="Dense passage on decomposition",
                source_file="Lecture5.pdf",
                source_pages=[8],
                source_unit_ids=["u20"],
                section_title="Decomposition",
            ),
        ]


class MockSparseRetriever:
    """Mock sparse retriever returning predetermined candidates."""

    @property
    def name(self) -> str:
        return "sparse"

    def index_records(self, records: list) -> None:
        pass

    def retrieve(self, query: RetrievalQuery) -> list[RetrievalCandidate]:
        return [
            RetrievalCandidate(
                record_id="c2",
                chunk_id="c2",
                rank=1,
                retrieval_method=self.name,
                score=16.4,
                text="Sparse passage on decomposition",
                source_file="Lecture5.pdf",
                source_pages=[8],
                source_unit_ids=["u20"],
                section_title="Decomposition",
            ),
            RetrievalCandidate(
                record_id="c3",
                chunk_id="c3",
                rank=2,
                retrieval_method=self.name,
                score=11.2,
                text="Sparse passage on ACF bounds",
                source_file="Lecture7.pdf",
                source_pages=[17],
                source_unit_ids=["u30"],
                section_title="ACF Bounds",
            ),
        ]


class MockReranker:
    """Mock reranker that reverses candidate order for test verification."""

    @property
    def name(self) -> str:
        return "mock_reranker"

    def rerank(
        self,
        query: RetrievalQuery,
        candidates: list[RetrievalCandidate],
        top_k: int,
    ) -> list[RetrievalCandidate]:
        reversed_cands = list(reversed(candidates))
        # Update 1-based ranks
        res = []
        for idx, c in enumerate(reversed_cands[:top_k], start=1):
            res.append(c.model_copy(update={"rank": idx, "retrieval_method": "rerank"}))
        return res


@pytest.fixture
def pipeline_setup():
    dense = MockDenseRetriever()
    sparse = MockSparseRetriever()
    fusion = ReciprocalRankFusion(k=60)
    pipeline = RetrievalPipeline(
        dense_retriever=dense,
        sparse_retriever=sparse,
        fusion_strategy=fusion,
    )
    return pipeline, dense, sparse, fusion


def test_pipeline_satisfies_protocol(pipeline_setup):
    pipeline, _, _, _ = pipeline_setup
    assert isinstance(pipeline, RetrievalPipelineProtocol)


def test_raw_string_query_auto_conversion(pipeline_setup):
    pipeline, _, _, _ = pipeline_setup
    response = pipeline.search("What is simple exponential smoothing?")

    assert isinstance(response, RetrievalResponse)
    assert response.query.text == "What is simple exponential smoothing?"
    assert response.total_evidence > 0
    assert response.strategy_used == RetrievalStrategy.HYBRID


def test_dense_only_strategy(pipeline_setup):
    pipeline, _, _, _ = pipeline_setup
    query = RetrievalQuery(text="SES equation", strategy=RetrievalStrategy.DENSE, top_k=2)
    response = pipeline.search(query)

    assert response.strategy_used == RetrievalStrategy.DENSE
    assert response.total_evidence == 2
    assert response.evidence[0].record_id == "c1"
    assert response.evidence[0].retrieval_methods == ["dense"]
    assert response.diagnostics.dense_latency_ms is not None
    assert response.diagnostics.sparse_latency_ms is None
    assert response.diagnostics.fusion_latency_ms is None


def test_sparse_only_strategy(pipeline_setup):
    pipeline, _, _, _ = pipeline_setup
    query = RetrievalQuery(text="ACF bounds", strategy=RetrievalStrategy.SPARSE, top_k=2)
    response = pipeline.search(query)

    assert response.strategy_used == RetrievalStrategy.SPARSE
    assert response.total_evidence == 2
    assert response.evidence[0].record_id == "c2"
    assert response.evidence[0].retrieval_methods == ["sparse"]
    assert response.diagnostics.sparse_latency_ms is not None
    assert response.diagnostics.dense_latency_ms is None
    assert response.diagnostics.fusion_latency_ms is None


def test_hybrid_strategy_execution_and_diagnostics(pipeline_setup):
    pipeline, _, _, _ = pipeline_setup
    query = RetrievalQuery(text="Decomposition formula", strategy=RetrievalStrategy.HYBRID, top_k=3)
    response = pipeline.search(query)

    assert response.strategy_used == RetrievalStrategy.HYBRID
    assert response.total_evidence == 3

    # In our mock setup:
    # c2 is Rank 2 in dense, Rank 1 in sparse -> Dual agreement -> Score = 1/62 + 1/61 = 0.03252 -> Rank 1!
    top_ev = response.evidence[0]
    assert top_ev.record_id == "c2"
    assert set(top_ev.retrieval_methods) == {"dense", "sparse"}
    assert top_ev.method_scores == {"dense": 0.81, "sparse": 16.4}

    # Verify diagnostics
    diag = response.diagnostics
    assert diag.dense_latency_ms is not None
    assert diag.sparse_latency_ms is not None
    assert diag.fusion_latency_ms is not None
    assert diag.total_latency_ms >= 0.0
    assert diag.candidate_counts_by_method == {"dense": 2, "sparse": 2}
    assert diag.method_agreement_count == 1  # c2 has dual agreement
    assert diag.additional_info["pool_agreement_count"] == 1


def test_evidence_provenance_and_llm_context(pipeline_setup):
    pipeline, _, _, _ = pipeline_setup
    response = pipeline.search("What is time series decomposition?")

    ev = response.evidence[0]
    assert ev.format_citation() == "[Lecture 5, p. 8]"
    assert ev.source_unit_ids == ["u20"]

    context = response.to_llm_context()
    assert "### [Lecture 5, p. 8] — Decomposition" in context
    assert "Decomposition" in context


def test_optional_reranker_hook(pipeline_setup):
    _, dense, sparse, fusion = pipeline_setup
    reranker = MockReranker()
    pipeline = RetrievalPipeline(
        dense_retriever=dense,
        sparse_retriever=sparse,
        fusion_strategy=fusion,
        reranker=reranker,
    )

    query = RetrievalQuery(
        text="Test reranker",
        strategy=RetrievalStrategy.HYBRID_RERANK,
        top_k=2,
    )
    response = pipeline.search(query)

    assert response.strategy_used == RetrievalStrategy.HYBRID_RERANK
    assert response.diagnostics.rerank_latency_ms is not None
    assert response.diagnostics.rerank_latency_ms >= 0.0


def test_missing_retriever_fallbacks():
    dense = MockDenseRetriever()
    sparse = MockSparseRetriever()

    # 1. Sparse missing -> hybrid falls back to dense-only
    p_dense_only = RetrievalPipeline(dense_retriever=dense, sparse_retriever=None)
    resp1 = p_dense_only.search(RetrievalQuery(text="test", strategy=RetrievalStrategy.HYBRID))
    assert resp1.strategy_used == RetrievalStrategy.DENSE

    # 2. Dense missing -> hybrid falls back to sparse-only
    p_sparse_only = RetrievalPipeline(dense_retriever=None, sparse_retriever=sparse)
    resp2 = p_sparse_only.search(RetrievalQuery(text="test", strategy=RetrievalStrategy.HYBRID))
    assert resp2.strategy_used == RetrievalStrategy.SPARSE

    # 3. Neither configured -> raises ValueError
    p_empty = RetrievalPipeline(dense_retriever=None, sparse_retriever=None)
    with pytest.raises(ValueError, match="Neither DenseRetriever nor SparseRetriever is configured"):
        p_empty.search("test")
