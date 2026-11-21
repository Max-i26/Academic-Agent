"""Unit tests for retrieval models: RetrievalQuery, RetrievalCandidate, Evidence, and RetrievalResponse."""

import pytest
from pydantic import ValidationError

from academic_agent.storage.models import FilterQuery
from academic_agent.retrieval.models import (
    RetrievalStrategy,
    RetrievalQuery,
    RetrievalCandidate,
    Evidence,
    RetrievalDiagnostics,
    RetrievalResponse,
)


class TestRetrievalQuery:
    """Tests for RetrievalQuery construction, validation, and normalization."""

    def test_valid_query_defaults(self):
        query = RetrievalQuery(text="What is the MAE formula?")
        assert query.text == "What is the MAE formula?"
        assert query.top_k == 5
        assert query.candidate_pool_size == 20
        assert query.strategy == RetrievalStrategy.HYBRID
        assert query.filters is None
        assert query.normalized_text == "what is the mae formula?"

    def test_whitespace_stripping(self):
        query = RetrievalQuery(text="   What is SES?   \n")
        assert query.text == "What is SES?"
        assert query.normalized_text == "what is ses?"

    def test_empty_or_whitespace_text_raises_validation_error(self):
        with pytest.raises(ValidationError):
            RetrievalQuery(text="")
        with pytest.raises(ValidationError):
            RetrievalQuery(text="    \t\n  ")

    def test_invalid_top_k_raises(self):
        with pytest.raises(ValidationError):
            RetrievalQuery(text="valid text", top_k=0)
        with pytest.raises(ValidationError):
            RetrievalQuery(text="valid text", top_k=-5)

    def test_candidate_pool_size_automatically_adjusts_if_less_than_top_k(self):
        query = RetrievalQuery(text="test query", top_k=15, candidate_pool_size=5)
        # Model validator raises candidate_pool_size up to top_k
        assert query.candidate_pool_size == 15

    def test_with_filter_query(self):
        filt = FilterQuery(source_file="Lecture4.pdf", source_pages=[7])
        query = RetrievalQuery(
            text="Explain alpha parameter",
            top_k=3,
            filters=filt,
            strategy=RetrievalStrategy.DENSE,
        )
        assert query.filters is not None
        assert query.filters.source_file == "Lecture4.pdf"
        assert query.strategy == RetrievalStrategy.DENSE


class TestRetrievalCandidate:
    """Tests for RetrievalCandidate intermediate representations."""

    def test_valid_candidate(self):
        cand = RetrievalCandidate(
            record_id="c1#frag_0",
            chunk_id="c1",
            rank=1,
            retrieval_method="dense",
            score=0.884,
            normalized_score=0.92,
            text="Simple Exponential Smoothing formula",
            source_file="4. DS5104 Lecture4 Simple_Exponential_Smoothing.pdf",
            source_pages=[7],
            section_title="THE SES EQUATION",
            content_type="FORMULA",
            fragment_index=0,
            total_fragments=3,
        )
        assert cand.record_id == "c1#frag_0"
        assert cand.rank == 1
        assert cand.score == 0.884
        assert cand.is_fragment is True

    def test_unfragmented_candidate(self):
        cand = RetrievalCandidate(
            record_id="c2",
            chunk_id="c2",
            rank=2,
            retrieval_method="sparse",
            score=14.5,
            total_fragments=1,
        )
        assert cand.is_fragment is False

    def test_invalid_rank_raises(self):
        with pytest.raises(ValidationError):
            RetrievalCandidate(
                record_id="c1",
                chunk_id="c1",
                rank=0,  # Rank must be >= 1
                retrieval_method="dense",
                score=0.5,
            )

    def test_negative_score_allowed(self):
        # Cosine similarity can be negative for opposing vectors
        cand = RetrievalCandidate(
            record_id="c1",
            chunk_id="c1",
            rank=10,
            retrieval_method="dense",
            score=-0.15,
        )
        assert cand.score == -0.15

    def test_to_evidence_conversion(self):
        cand = RetrievalCandidate(
            record_id="frag_99",
            chunk_id="parent_99",
            rank=1,
            retrieval_method="dense",
            score=0.75,
            text="Exponential smoothing weight decaying",
            source_file="Lecture4.pdf",
            source_pages=[8],
            section_title="Weight Decay",
            content_type="PARAGRAPH",
            fragment_index=1,
            total_fragments=2,
            metadata={"token_count": 120},
        )
        ev = cand.to_evidence()
        assert isinstance(ev, Evidence)
        assert ev.record_id == "frag_99"
        assert ev.chunk_id == "parent_99"
        assert ev.text == "Exponential smoothing weight decaying"
        assert ev.retrieval_methods == ["dense"]
        assert ev.method_scores == {"dense": 0.75}
        assert ev.is_fragment is True
        assert ev.metadata["token_count"] == 120


class TestEvidence:
    """Tests for academic citation formatting, provenance, and prompt formatting."""

    def test_citation_formatting_single_page(self):
        ev = Evidence(
            record_id="c1",
            chunk_id="c1",
            text="MAE definition",
            source_file="4. DS5104 Lecture4 Simple_Exponential_Smoothing.pdf",
            source_pages=[18],
            score=0.85,
        )
        assert ev.format_citation() == "[Lecture 4, p. 18]"

    def test_citation_formatting_contiguous_pages(self):
        ev = Evidence(
            record_id="c2",
            chunk_id="c2",
            text="Trend seasonal decomposition",
            source_file="5. DS5104 Lecture5 Trend_Seasonal_Decomposition.pdf",
            source_pages=[12, 13, 14],
            score=0.80,
        )
        assert ev.format_citation() == "[Lecture 5, pp. 12-14]"

    def test_citation_formatting_non_contiguous_pages(self):
        ev = Evidence(
            record_id="c3",
            chunk_id="c3",
            text="Autocorrelation ACF bounds",
            source_file="7. DS5104 Lecture7 Autocorrelation ACF.pdf",
            source_pages=[4, 7, 11],
            score=0.78,
        )
        assert ev.format_citation() == "[Lecture 7, pp. 4, 7, 11]"

    def test_citation_formatting_fallback_filename(self):
        ev = Evidence(
            record_id="c4",
            chunk_id="c4",
            text="Some supplement",
            source_file="additional_notes_math.pdf",
            source_pages=[2],
            score=0.70,
        )
        assert ev.format_citation() == "[additional_notes_math.pdf, p. 2]"

    def test_citation_formatting_no_pages(self):
        ev = Evidence(
            record_id="c5",
            chunk_id="c5",
            text="Overview",
            source_file="4. DS5104 Lecture4 Simple_Exponential_Smoothing.pdf",
            source_pages=[],
            score=0.65,
        )
        assert ev.format_citation() == "[Lecture 4]"

    def test_to_prompt_context_formatting(self):
        ev = Evidence(
            record_id="c1#frag_0",
            chunk_id="c1",
            text="S_t = alpha * Y_t + (1 - alpha) * S_{t-1}",
            source_file="4. DS5104 Lecture4 Simple_Exponential_Smoothing.pdf",
            source_pages=[7],
            section_title="THE SES EQUATION",
            content_type="FORMULA",
            fragment_index=0,
            total_fragments=3,
            score=0.9123,
        )
        prompt_str = ev.to_prompt_context(include_score=True)
        assert "### [Lecture 4, p. 7] — THE SES EQUATION" in prompt_str
        assert "Type: FORMULA" in prompt_str
        assert "Part 1/3" in prompt_str
        assert "Relevance: 0.9123" in prompt_str
        assert "S_t = alpha * Y_t + (1 - alpha) * S_{t-1}" in prompt_str


class TestRetrievalResponse:
    """Tests for complete retrieval response packaging and multi-evidence LLM context."""

    def test_response_construction_and_source_files(self):
        q = RetrievalQuery(text="What is ACF?")
        ev1 = Evidence(
            record_id="c1",
            chunk_id="c1",
            text="ACF definition",
            source_file="Lecture7.pdf",
            source_pages=[2],
            score=0.85,
        )
        ev2 = Evidence(
            record_id="c2",
            chunk_id="c2",
            text="Stationarity",
            source_file="Lecture6.pdf",
            source_pages=[4],
            score=0.75,
        )
        ev3 = Evidence(
            record_id="c3",
            chunk_id="c3",
            text="ACF bounds",
            source_file="Lecture7.pdf",
            source_pages=[17],
            score=0.70,
        )

        resp = RetrievalResponse(
            query=q,
            evidence=[ev1, ev2, ev3],
            strategy_used=RetrievalStrategy.HYBRID,
            diagnostics=RetrievalDiagnostics(
                dense_latency_ms=12.4,
                sparse_latency_ms=3.1,
                fusion_latency_ms=0.5,
                total_latency_ms=16.0,
                method_agreement_count=2,
            ),
        )

        assert resp.total_evidence == 3
        # Unique source files in order of appearance
        assert resp.source_files_represented == ["Lecture7.pdf", "Lecture6.pdf"]
        assert resp.diagnostics.total_latency_ms == 16.0
        assert resp.diagnostics.method_agreement_count == 2

    def test_to_llm_context(self):
        q = RetrievalQuery(text="What is ACF?")
        ev1 = Evidence(
            record_id="c1",
            chunk_id="c1",
            text="Passage 1 content",
            source_file="4. DS5104 Lecture4 Simple_Exponential_Smoothing.pdf",
            source_pages=[7],
            score=0.85,
        )
        ev2 = Evidence(
            record_id="c2",
            chunk_id="c2",
            text="Passage 2 content",
            source_file="7. DS5104 Lecture7 Autocorrelation ACF.pdf",
            source_pages=[17],
            score=0.75,
        )
        resp = RetrievalResponse(query=q, evidence=[ev1, ev2])

        context = resp.to_llm_context()
        assert "Passage 1 content" in context
        assert "Passage 2 content" in context
        assert "\n\n---\n\n" in context

        # Test truncation with max_items
        context_top1 = resp.to_llm_context(max_items=1)
        assert "Passage 1 content" in context_top1
        assert "Passage 2 content" not in context_top1

    def test_empty_evidence_llm_context(self):
        q = RetrievalQuery(text="Unrelated query")
        resp = RetrievalResponse(query=q, evidence=[])
        assert resp.to_llm_context() == "No relevant academic evidence found."
