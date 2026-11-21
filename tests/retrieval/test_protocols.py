"""Unit tests verifying protocol compliance for retrievers, fusion strategies, rerankers, and pipelines."""

from typing import Any
import pytest

from academic_agent.retrieval.models import (
    RetrievalQuery,
    RetrievalCandidate,
    RetrievalResponse,
    Evidence,
)
from academic_agent.retrieval.protocols import (
    RetrieverProtocol,
    DenseRetrieverProtocol,
    SparseRetrieverProtocol,
    FusionStrategyProtocol,
    RerankerProtocol,
    RetrievalPipelineProtocol,
)


class MockRetriever:
    """Mock implementation satisfying RetrieverProtocol and DenseRetrieverProtocol."""

    @property
    def name(self) -> str:
        return "mock_dense"

    def retrieve(self, query: RetrievalQuery) -> list[RetrievalCandidate]:
        return [
            RetrievalCandidate(
                record_id="c1",
                chunk_id="c1",
                rank=1,
                retrieval_method=self.name,
                score=0.9,
                text="Mock passage",
                source_file="Lecture1.pdf",
                source_pages=[1],
            )
        ]


class MockSparseRetriever:
    """Mock implementation satisfying SparseRetrieverProtocol."""

    @property
    def name(self) -> str:
        return "mock_sparse"

    def index_records(self, records: list) -> None:
        pass

    def retrieve(self, query: RetrievalQuery) -> list[RetrievalCandidate]:
        return [
            RetrievalCandidate(
                record_id="c2",
                chunk_id="c2",
                rank=1,
                retrieval_method=self.name,
                score=12.5,
                text="Mock sparse passage",
                source_file="Lecture2.pdf",
                source_pages=[5],
            )
        ]


class MockFusionStrategy:
    """Mock implementation satisfying FusionStrategyProtocol."""

    @property
    def name(self) -> str:
        return "mock_rrf"

    def fuse(
        self,
        query: RetrievalQuery,
        candidates_by_method: dict[str, list[RetrievalCandidate]],
        top_k: int,
    ) -> list[RetrievalCandidate]:
        all_cands: list[RetrievalCandidate] = []
        for cands in candidates_by_method.values():
            all_cands.extend(cands)
        return all_cands[:top_k]


class MockReranker:
    """Mock implementation satisfying RerankerProtocol."""

    @property
    def name(self) -> str:
        return "mock_reranker"

    def rerank(
        self,
        query: RetrievalQuery,
        candidates: list[RetrievalCandidate],
        top_k: int,
    ) -> list[RetrievalCandidate]:
        return sorted(candidates, key=lambda c: c.score, reverse=True)[:top_k]


class MockRetrievalPipeline:
    """Mock implementation satisfying RetrievalPipelineProtocol."""

    def search(self, query: RetrievalQuery | str) -> RetrievalResponse:
        q_obj = query if isinstance(query, RetrievalQuery) else RetrievalQuery(text=query)
        evidence = [
            Evidence(
                record_id="c1",
                chunk_id="c1",
                text="Pipeline evidence",
                source_file="Lecture1.pdf",
                source_pages=[1],
                score=0.95,
            )
        ]
        return RetrievalResponse(query=q_obj, evidence=evidence)


def test_protocol_runtime_checks():
    dense = MockRetriever()
    assert isinstance(dense, RetrieverProtocol)
    assert isinstance(dense, DenseRetrieverProtocol)

    sparse = MockSparseRetriever()
    assert isinstance(sparse, RetrieverProtocol)
    assert isinstance(sparse, SparseRetrieverProtocol)

    fusion = MockFusionStrategy()
    assert isinstance(fusion, FusionStrategyProtocol)

    reranker = MockReranker()
    assert isinstance(reranker, RerankerProtocol)

    pipeline = MockRetrievalPipeline()
    assert isinstance(pipeline, RetrievalPipelineProtocol)


def test_protocol_execution_flow():
    query = RetrievalQuery(text="How does differencing achieve stationarity?", top_k=2)

    dense = MockRetriever()
    sparse = MockSparseRetriever()
    fusion = MockFusionStrategy()

    dense_candidates = dense.retrieve(query)
    sparse_candidates = sparse.retrieve(query)

    assert len(dense_candidates) == 1
    assert dense_candidates[0].retrieval_method == "mock_dense"

    assert len(sparse_candidates) == 1
    assert sparse_candidates[0].retrieval_method == "mock_sparse"

    fused = fusion.fuse(
        query=query,
        candidates_by_method={"dense": dense_candidates, "sparse": sparse_candidates},
        top_k=2,
    )
    assert len(fused) == 2

    pipeline = MockRetrievalPipeline()
    response = pipeline.search(query)
    assert response.total_evidence == 1
    assert response.evidence[0].record_id == "c1"
