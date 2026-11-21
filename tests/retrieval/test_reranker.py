"""Unit and integration tests for CrossEncoderReranker (Stage 4.5)."""

from typing import Any
import pytest

from academic_agent.retrieval.models import (
    RetrievalQuery,
    RetrievalCandidate,
    RetrievalStrategy,
    RetrievalResponse,
)
from academic_agent.retrieval.protocols import RerankerProtocol
from academic_agent.retrieval.reranker import CrossEncoderReranker
from academic_agent.retrieval.pipeline import RetrievalPipeline
from academic_agent.retrieval.fusion import ReciprocalRankFusion


def _make_candidate(
    record_id: str,
    text: str,
    rank: int = 1,
    score: float = 0.5,
    method: str = "dense",
    chunk_id: str | None = None,
    source_file: str = "test_lecture.pdf",
    source_pages: list[int] | None = None,
    source_unit_ids: list[str] | None = None,
    section_title: str | None = "Test Section",
    dense_score: float | None = None,
    sparse_score: float | None = None,
    rrf_score: float | None = None,
) -> RetrievalCandidate:
    """Helper to construct a populated RetrievalCandidate."""
    return RetrievalCandidate(
        record_id=record_id,
        chunk_id=chunk_id or record_id.split("#")[0],
        rank=rank,
        retrieval_method=method,
        score=score,
        text=text,
        source_file=source_file,
        source_pages=source_pages or [1],
        source_unit_ids=source_unit_ids or [f"unit_{record_id}"],
        section_title=section_title,
        content_type="PARAGRAPH",
        fragment_index=0,
        total_fragments=1,
        contributing_methods=[method],
        method_scores={method: score},
        method_ranks={method: rank},
        dense_score=dense_score or (score if method == "dense" else None),
        sparse_score=sparse_score or (score if method == "sparse" else None),
        rrf_score=rrf_score,
        metadata={"custom": "tag"},
    )


class TestCrossEncoderReranker:
    """Test suite for CrossEncoderReranker protocol compliance, scoring, and safeguards."""

    def test_protocol_conformance(self):
        """CrossEncoderReranker must satisfy RerankerProtocol at runtime."""
        reranker = CrossEncoderReranker(lazy=True)
        assert isinstance(reranker, RerankerProtocol)
        assert reranker.name == "cross-encoder/ms-marco-MiniLM-L-6-v2"
        assert reranker.device == "cpu"
        assert reranker.max_length == 512

    def test_empty_candidates_or_zero_top_k(self):
        """Empty candidates or non-positive top_k must safely return empty list."""
        reranker = CrossEncoderReranker(lazy=True)
        query = RetrievalQuery(text="test query")

        # Empty candidates
        assert reranker.rerank(query, [], top_k=5) == []

        # Zero or negative top_k
        cands = [_make_candidate("c1", "Sample text")]
        assert reranker.rerank(query, cands, top_k=0) == []
        assert reranker.rerank(query, cands, top_k=-1) == []

    def test_deterministic_tie_breaking(self):
        """Candidates with identical scores must break ties via original score, then record_id."""
        reranker = CrossEncoderReranker(lazy=True)

        class MockModel:
            def predict(self, pairs, **kwargs):
                # Return identical scores for all pairs to force tie-breaking
                return [2.5] * len(pairs)

        reranker._model = MockModel()

        c1 = _make_candidate("rec_b", "text b", rank=2, score=0.4, rrf_score=0.4)
        c2 = _make_candidate("rec_a", "text a", rank=1, score=0.9, rrf_score=0.9)  # Higher pre-rerank score
        c3 = _make_candidate("rec_c", "text c", rank=3, score=0.4, rrf_score=0.4)  # Identical pre-score to b, but 'rec_b' < 'rec_c'

        results = reranker.rerank(RetrievalQuery(text="query"), [c1, c2, c3], top_k=3)

        assert len(results) == 3
        # First should be rec_a (highest pre-rerank score: 0.9)
        assert results[0].record_id == "rec_a"
        assert results[0].rank == 1

        # Second should be rec_b (pre-score 0.4, alphabetical 'rec_b' < 'rec_c')
        assert results[1].record_id == "rec_b"
        assert results[1].rank == 2

        # Third should be rec_c
        assert results[2].record_id == "rec_c"
        assert results[2].rank == 3

    def test_preserves_provenance_and_independent_scores(self):
        """Reranker must preserve fragment attributes, provenance, and original branch scores."""
        reranker = CrossEncoderReranker(lazy=True)

        class MockModel:
            def predict(self, pairs, **kwargs):
                return [3.14]

        reranker._model = MockModel()

        c = _make_candidate(
            record_id="frag_123#frag_0",
            chunk_id="frag_123",
            text="Academic content",
            source_file="Lecture5.pdf",
            source_pages=[14, 15],
            source_unit_ids=["u1", "u2"],
            section_title="CMA Workspace",
            dense_score=0.88,
            sparse_score=14.5,
            rrf_score=0.032,
            method="hybrid",
        )
        c.contributing_methods = ["dense", "sparse"]
        c.method_scores = {"dense": 0.88, "sparse": 14.5, "rrf": 0.032}

        reranked = reranker.rerank(RetrievalQuery(text="What is CMA?"), [c], top_k=1)
        res = reranked[0]

        # Verify provenance
        assert res.record_id == "frag_123#frag_0"
        assert res.chunk_id == "frag_123"
        assert res.source_file == "Lecture5.pdf"
        assert res.source_pages == [14, 15]
        assert res.source_unit_ids == ["u1", "u2"]
        assert res.section_title == "CMA Workspace"

        # Verify score independence
        assert res.rerank_score == pytest.approx(3.14)
        assert res.score == pytest.approx(3.14)
        assert res.dense_score == pytest.approx(0.88)
        assert res.sparse_score == pytest.approx(14.5)
        assert res.rrf_score == pytest.approx(0.032)
        assert res.method_scores["rerank"] == pytest.approx(3.14)
        assert res.method_scores["dense"] == pytest.approx(0.88)
        assert res.method_scores["sparse"] == pytest.approx(14.5)
        assert res.method_scores["rrf"] == pytest.approx(0.032)
        assert "rerank" in res.contributing_methods

        # Verify to_evidence conversion maintains all scores
        ev = res.to_evidence()
        assert ev.rerank_score == pytest.approx(3.14)
        assert ev.dense_score == pytest.approx(0.88)
        assert ev.sparse_score == pytest.approx(14.5)
        assert ev.rrf_score == pytest.approx(0.032)

    def test_truncation_diagnostics(self):
        """Passages that exceed token/char budget must trigger diagnostic counter and warnings."""
        reranker = CrossEncoderReranker(lazy=True, max_length=64)

        class MockModel:
            def predict(self, pairs, **kwargs):
                return [1.0]

        reranker._model = MockModel()

        long_text = "word " * 300  # ~1500 chars, well over 64 tokens * 4
        c = _make_candidate("c_long", long_text)

        initial_truncations = reranker.truncation_count
        _ = reranker.rerank(RetrievalQuery(text="short query"), [c], top_k=1)

        assert reranker.truncation_count == initial_truncations + 1

    @pytest.mark.slow
    def test_live_cross_encoder_semantic_discrimination(self):
        """Live CrossEncoder must clearly separate target passage from distractor."""
        reranker = CrossEncoderReranker(device="cpu", lazy=False)

        q = RetrievalQuery(text="What is the formula for Mean Absolute Error (MAE)?")
        target_cand = _make_candidate(
            "target",
            "Mean Absolute Error (MAE) measures average error magnitude: MAE = mean(|e_t|).",
            score=0.1,  # Lower pre-rerank score to test promotion
        )
        distractor_cand = _make_candidate(
            "distractor",
            "Centered moving average is calculated by averaging two consecutive four-period averages.",
            score=0.9,  # Higher pre-rerank score
        )

        # Rerank both candidates
        results = reranker.rerank(q, [distractor_cand, target_cand], top_k=2)

        assert len(results) == 2
        # Target must be promoted to rank 1
        assert results[0].record_id == "target"
        assert results[0].rank == 1
        assert results[0].rerank_score > results[1].rerank_score
        # Check significant margin
        assert results[0].rerank_score - results[1].rerank_score > 5.0

    @pytest.mark.slow
    def test_pipeline_integration_with_hybrid_rerank(self):
        """RetrievalPipeline with CrossEncoderReranker must execute HYBRID_RERANK end-to-end."""
        class MockRetriever:
            def __init__(self, name, cands):
                self._name = name
                self._cands = cands

            def retrieve(self, query):
                return self._cands

        d_cand = _make_candidate("d1", "Mean Absolute Error formula MAE", method="dense", score=0.85)
        s_cand = _make_candidate("s1", "Random distractor text without relevance", method="sparse", score=12.0)

        pipeline = RetrievalPipeline(
            dense_retriever=MockRetriever("dense", [d_cand]),
            sparse_retriever=MockRetriever("sparse", [s_cand]),
            fusion_strategy=ReciprocalRankFusion(k=60),
            reranker=CrossEncoderReranker(device="cpu", lazy=False),
        )

        query = RetrievalQuery(
            text="What is the formula for MAE?",
            strategy=RetrievalStrategy.HYBRID_RERANK,
            top_k=2,
            candidate_pool_size=5,
        )

        response = pipeline.search(query)

        assert isinstance(response, RetrievalResponse)
        assert response.strategy_used == RetrievalStrategy.HYBRID_RERANK
        assert len(response.evidence) == 2
        assert response.evidence[0].record_id == "d1"
        assert response.evidence[0].rerank_score is not None
        assert response.diagnostics.rerank_latency_ms is not None
        assert response.diagnostics.rerank_latency_ms > 0
