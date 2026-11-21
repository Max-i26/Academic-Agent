"""Unit tests for Stage 5.2B EvidenceContextBuilder."""

import pytest

from academic_agent.generation.synthesis.context.evidence_context_builder import EvidenceContextBuilder
from academic_agent.generation.synthesis.context.evidence_formatter import EvidenceFormatter
from academic_agent.generation.synthesis.models import (
    EvidenceContext,
    OmissionReason,
    OmittedCandidate,
)
from academic_agent.generation.synthesis.protocols import EvidenceContextBuilderProtocol
from academic_agent.retrieval.models import Evidence, RetrievalQuery, RetrievalResponse


@pytest.fixture
def default_query() -> RetrievalQuery:
    return RetrievalQuery(text="Explain stationarity and AR(1) models in time series.")


@pytest.fixture
def sample_evidence_list() -> list[Evidence]:
    return [
        Evidence(
            record_id="rec_1#0",
            chunk_id="chunk_1",
            text="Weak stationarity requires constant mean and autocovariance depending only on lag.",
            source_file="Lecture_3.pdf",
            source_pages=[12],
            score=0.95,
        ),
        Evidence(
            record_id="rec_2#0",
            chunk_id="chunk_2",
            text="An AR(1) process X_t = phi_1 X_{t-1} + epsilon_t is stationary if |phi_1| < 1.",
            source_file="Lecture_4.pdf",
            source_pages=[5],
            score=0.90,
        ),
        Evidence(
            record_id="rec_3#0",
            chunk_id="chunk_3",
            text="The PACF of an AR(p) process cuts off sharply after lag p.",
            source_file="Lecture_4.pdf",
            source_pages=[18],
            score=0.85,
        ),
    ]


class TestEvidenceContextBuilderBasics:
    """Verify protocol compliance, parameter validation, and empty cases."""

    def test_protocol_conformance(self):
        builder = EvidenceContextBuilder()
        assert isinstance(builder, EvidenceContextBuilderProtocol)

    @pytest.mark.parametrize("invalid_threshold", [-0.1, 1.1])
    def test_invalid_overlap_threshold(self, invalid_threshold):
        with pytest.raises(ValueError, match="overlap_threshold"):
            EvidenceContextBuilder(overlap_threshold=invalid_threshold)

    def test_invalid_min_overlap_words(self):
        with pytest.raises(ValueError, match="min_overlap_words"):
            EvidenceContextBuilder(min_overlap_words=0)

    def test_invalid_shingle_size(self):
        with pytest.raises(ValueError, match="shingle_size"):
            EvidenceContextBuilder(shingle_size=0)

    def test_negative_allocated_budget(self, default_query):
        builder = EvidenceContextBuilder()
        response = RetrievalResponse(query=default_query, evidence=[])
        with pytest.raises(ValueError, match="allocated_evidence_tokens"):
            builder.build_context(response, allocated_evidence_tokens=-1)

    def test_empty_retrieval_response(self, default_query):
        builder = EvidenceContextBuilder()
        response = RetrievalResponse(query=default_query, evidence=[])
        ctx = builder.build_context(response, allocated_evidence_tokens=1000)

        assert len(ctx.items) == 0
        assert ctx.total_tokens == 0
        assert ctx.budget_allocated == 1000
        assert ctx.has_omissions is False
        assert ctx.dropped_candidates_count == 0
        assert ctx.is_exact_token_count is False

    def test_zero_allocated_budget(self, default_query, sample_evidence_list):
        builder = EvidenceContextBuilder()
        response = RetrievalResponse(query=default_query, evidence=sample_evidence_list)
        ctx = builder.build_context(response, allocated_evidence_tokens=0)

        assert len(ctx.items) == 0
        assert ctx.total_tokens == 0
        assert ctx.has_omissions is True
        assert ctx.dropped_candidates_count == 3
        for omitted in ctx.omitted_candidates:
            assert omitted.reason == OmissionReason.BUDGET_EXCEEDED
            assert "0 tokens" in omitted.details


class TestDeduplicationAndProvenance:
    """Verify mathematical case sensitivity, provenance merging, and collision handling."""

    def test_case_sensitive_mathematical_deduplication(self, default_query):
        """Upper and lower case math symbols (e.g. X_t vs x_t, ACF vs acf) must not be collapsed."""
        builder = EvidenceContextBuilder(tokenizer=lambda s: len(s.split()))

        ev1 = Evidence(
            record_id="rec_x1#0",
            chunk_id="chunk_x1",
            text="The time series X_t has unconditional variance sigma^2 / (1 - phi_1^2).",
            source_file="Lecture_4.pdf",
            source_pages=[10],
            score=0.90,
        )
        ev2 = Evidence(
            record_id="rec_x2#0",
            chunk_id="chunk_x2",
            text="The time series x_t has unconditional variance sigma^2 / (1 - phi_1^2).",
            source_file="Lecture_4.pdf",
            source_pages=[11],
            score=0.85,
        )

        response = RetrievalResponse(query=default_query, evidence=[ev1, ev2])
        ctx = builder.build_context(response, allocated_evidence_tokens=1000)

        assert len(ctx.items) == 2
        assert ctx.items[0].evidence_id == "E1"
        assert ctx.items[1].evidence_id == "E2"
        assert "X_t" in ctx.items[0].text
        assert "x_t" in ctx.items[1].text
        assert ctx.has_omissions is False

    def test_same_text_different_source_files_retained(self, default_query):
        """Identical text from different lectures must be retained as separate items with distinct provenance."""
        builder = EvidenceContextBuilder(tokenizer=lambda s: len(s.split()))
        text = "Weak stationarity requires constant mean and finite autocovariance for all lags."

        ev1 = Evidence(
            record_id="rec_l3#0",
            chunk_id="chunk_l3",
            text=text,
            source_file="Lecture_3.pdf",
            source_pages=[12],
            score=0.92,
        )
        ev2 = Evidence(
            record_id="rec_l5#0",
            chunk_id="chunk_l5",
            text=text,
            source_file="Lecture_5.pdf",
            source_pages=[8],
            score=0.88,
        )

        response = RetrievalResponse(query=default_query, evidence=[ev1, ev2])
        ctx = builder.build_context(response, allocated_evidence_tokens=1000)

        assert len(ctx.items) == 2
        assert ctx.items[0].evidence_id == "E1"
        assert ctx.items[0].citation_label == "[Lecture 3, p. 12]"
        assert ctx.items[1].evidence_id == "E2"
        assert ctx.items[1].citation_label == "[Lecture 5, p. 8]"
        assert ctx.has_omissions is False

    def test_same_record_id_identical_content_deduplicated_with_provenance_merge(self, default_query):
        """Matching record_id from same source merges pages without creating separate evidence items."""
        builder = EvidenceContextBuilder(tokenizer=lambda s: len(s.split()))
        text = "Stationarity requires constant mean and autocovariance."

        ev1 = Evidence(
            record_id="rec_dup#0",
            chunk_id="chunk_dup",
            text=text,
            source_file="Lecture_3.pdf",
            source_pages=[12],
            source_unit_ids=["u1"],
            score=0.92,
        )
        ev2 = Evidence(
            record_id="rec_dup#0",
            chunk_id="chunk_dup",
            text=text,
            source_file="Lecture_3.pdf",
            source_pages=[15],
            source_unit_ids=["u2"],
            score=0.80,
        )

        response = RetrievalResponse(query=default_query, evidence=[ev1, ev2])
        ctx = builder.build_context(response, allocated_evidence_tokens=1000)

        assert len(ctx.items) == 1
        item = ctx.items[0]
        assert item.evidence_id == "E1"
        assert item.source_pages == [12, 15]
        assert item.evidence.source_unit_ids == ["u1", "u2"]
        assert item.citation_label == "[Lecture 3, pp. 12, 15]"
        assert ctx.has_omissions is True
        assert ctx.dropped_candidates_count == 1
        assert ctx.omitted_candidates[0].reason == OmissionReason.DUPLICATE
        assert ctx.omitted_candidates[0].rank == 2

    def test_same_record_id_conflicting_content_flagged_for_integrity(self, default_query):
        """Matching record_id with differing source file or text must be flagged and omitted."""
        builder = EvidenceContextBuilder(tokenizer=lambda s: len(s.split()))

        ev1 = Evidence(
            record_id="rec_conflict#0",
            chunk_id="chunk_conflict",
            text="Weak stationarity requires constant mean.",
            source_file="Lecture_3.pdf",
            source_pages=[12],
            score=0.92,
        )
        ev2 = Evidence(
            record_id="rec_conflict#0",
            chunk_id="chunk_conflict",
            text="Completely different content under the same record ID.",
            source_file="Lecture_4.pdf",
            source_pages=[5],
            score=0.80,
        )

        response = RetrievalResponse(query=default_query, evidence=[ev1, ev2])
        ctx = builder.build_context(response, allocated_evidence_tokens=1000)

        assert len(ctx.items) == 1
        assert ctx.items[0].evidence_id == "E1"
        assert ctx.items[0].source_pages == [12]  # Provenance untouched
        assert ctx.dropped_candidates_count == 1
        omitted = ctx.omitted_candidates[0]
        assert omitted.reason == OmissionReason.INVALID_CANDIDATE
        assert "Conflicting record_id" in omitted.details


class TestSequenceAwareOverlap:
    """Verify n-gram word shingle deduplication vs scrambled words and short formulas."""

    def test_sequence_aware_shingles_distinguishes_scrambled_words(self, default_query):
        """Passages with identical vocabulary in different grammatical order must not be collapsed."""
        builder = EvidenceContextBuilder(
            tokenizer=lambda s: len(s.split()),
            overlap_threshold=0.85,
            min_overlap_words=15,
            shingle_size=5,
        )

        # 18 words in sentence order
        text_ordered = (
            "In time series analysis weak stationarity requires constant mean "
            "and finite autocovariance for all integer lags h."
        )
        # Exact same words scrambled
        text_scrambled = (
            "Integer lags h for all finite autocovariance and constant mean "
            "requires weak stationarity analysis time series in."
        )

        ev1 = Evidence(
            record_id="rec_ord#0",
            chunk_id="chunk_ord",
            text=text_ordered,
            source_file="Lecture_3.pdf",
            source_pages=[10],
            score=0.90,
        )
        ev2 = Evidence(
            record_id="rec_scram#0",
            chunk_id="chunk_scram",
            text=text_scrambled,
            source_file="Lecture_3.pdf",
            source_pages=[11],
            score=0.85,
        )

        response = RetrievalResponse(query=default_query, evidence=[ev1, ev2])
        ctx = builder.build_context(response, allocated_evidence_tokens=1000)

        # Both accepted because contiguous 5-grams mismatch!
        assert len(ctx.items) == 2
        assert ctx.items[0].evidence_id == "E1"
        assert ctx.items[1].evidence_id == "E2"
        assert ctx.has_omissions is False

    def test_sequence_aware_shingles_detects_near_duplicate_subsequence(self, default_query):
        """High contiguous shingle overlap (subsequence) from same source must merge pages and omit."""
        builder = EvidenceContextBuilder(
            tokenizer=lambda s: len(s.split()),
            overlap_threshold=0.80,
            min_overlap_words=15,
            shingle_size=5,
        )

        text_base = (
            "In time series analysis weak stationarity requires constant mean "
            "and finite autocovariance for all integer lags h in the domain."
        )
        # High overlap subsequence (> 80% of 5-grams match)
        text_sub = (
            "In time series analysis weak stationarity requires constant mean "
            "and finite autocovariance for all integer lags h."
        )

        ev1 = Evidence(
            record_id="rec_base#0",
            chunk_id="chunk_base",
            text=text_base,
            source_file="Lecture_3.pdf",
            source_pages=[10],
            score=0.90,
        )
        ev2 = Evidence(
            record_id="rec_sub#0",
            chunk_id="chunk_sub",
            text=text_sub,
            source_file="Lecture_3.pdf",
            source_pages=[12],
            score=0.85,
        )

        response = RetrievalResponse(query=default_query, evidence=[ev1, ev2])
        ctx = builder.build_context(response, allocated_evidence_tokens=1000)

        assert len(ctx.items) == 1
        item = ctx.items[0]
        assert item.evidence_id == "E1"
        assert item.source_pages == [10, 12]  # Pages merged
        assert ctx.dropped_candidates_count == 1
        assert ctx.omitted_candidates[0].reason == OmissionReason.OVERLAPPING
        assert "Sequence containment" in ctx.omitted_candidates[0].details

    def test_short_formulas_below_min_overlap_words_preserved(self, default_query):
        """Short mathematical formulas (< min_overlap_words) must never be dropped as overlapping."""
        builder = EvidenceContextBuilder(
            tokenizer=lambda s: len(s.split()),
            overlap_threshold=0.70,
            min_overlap_words=15,
        )

        ev1 = Evidence(
            record_id="form_1#0",
            chunk_id="chunk_f1",
            text="Y_t = phi_1 Y_{t-1} + epsilon_t",  # 6 words
            source_file="Lecture_4.pdf",
            source_pages=[6],
            score=0.95,
        )
        ev2 = Evidence(
            record_id="form_2#0",
            chunk_id="chunk_f2",
            text="Y_t = phi_1 Y_{t-1} + phi_2 Y_{t-2} + epsilon_t",  # 8 words
            source_file="Lecture_4.pdf",
            source_pages=[8],
            score=0.90,
        )

        response = RetrievalResponse(query=default_query, evidence=[ev1, ev2])
        ctx = builder.build_context(response, allocated_evidence_tokens=1000)

        assert len(ctx.items) == 2
        assert ctx.items[0].evidence_id == "E1"
        assert ctx.items[1].evidence_id == "E2"
        assert ctx.has_omissions is False


class TestBudgetAndOmissionAuditing:
    """Verify rank-monotonic packing, package-level token accounting, and audit logging."""

    def test_complete_xml_package_token_accounting(self, default_query, sample_evidence_list):
        """Context total_tokens must match the formatted XML package, not bare passage text."""
        # Custom mock tokenizer that returns character length
        builder = EvidenceContextBuilder(tokenizer=len)
        response = RetrievalResponse(query=default_query, evidence=[sample_evidence_list[0]])

        ctx = builder.build_context(response, allocated_evidence_tokens=5000)
        formatter = EvidenceFormatter()
        expected_xml = formatter.format_evidence_block(ctx)

        assert ctx.total_tokens == len(expected_xml)
        assert ctx.is_exact_token_count is True

    def test_rank_monotonic_budget_packing_halts_and_audits_subsequent(self, default_query):
        """When a higher-ranked item overflows budget, packing halts and all subsequent are audited."""
        # Tokenizer where each word is 10 tokens
        builder = EvidenceContextBuilder(tokenizer=lambda s: len(s.split()) * 10)

        ev1 = Evidence(
            record_id="rec_1#0",
            chunk_id="c1",
            text="First evidence text fits in budget.",  # 6 words -> ~60 tokens
            source_file="Lecture_1.pdf",
            source_pages=[1],
            score=0.95,
        )
        ev2 = Evidence(
            record_id="rec_2#0",
            chunk_id="c2",
            text="Second evidence is very long and has lots and lots of detailed mathematical equations and words.",
            source_file="Lecture_2.pdf",
            source_pages=[2],
            score=0.90,
        )
        ev3 = Evidence(
            record_id="rec_3#0",
            chunk_id="c3",
            text="Short text.",  # Tiny candidate that could physically fit, but is rank 3
            source_file="Lecture_3.pdf",
            source_pages=[3],
            score=0.85,
        )
        ev4 = Evidence(
            record_id="rec_4#0",
            chunk_id="c4",
            text="Another short text.",
            source_file="Lecture_4.pdf",
            source_pages=[4],
            score=0.80,
        )

        response = RetrievalResponse(query=default_query, evidence=[ev1, ev2, ev3, ev4])
        # Package tokens for ev1 will be around 140 (including XML wrapper words * 10)
        # Set budget to 200: ev1 fits, ev1 + ev2 overflows!
        ctx = builder.build_context(response, allocated_evidence_tokens=200)

        # Monotonicity check: only E1 accepted
        assert len(ctx.items) == 1
        assert ctx.items[0].evidence_id == "E1"
        assert ctx.has_omissions is True
        assert ctx.dropped_candidates_count == 3

        # ev2 caused the overflow
        assert ctx.omitted_candidates[0].record_id == "rec_2#0"
        assert ctx.omitted_candidates[0].rank == 2
        assert ctx.omitted_candidates[0].reason == OmissionReason.BUDGET_EXCEEDED
        assert "exceeds allocated budget" in ctx.omitted_candidates[0].details

        # ev3 and ev4 omitted to preserve rank monotonicity
        assert ctx.omitted_candidates[1].record_id == "rec_3#0"
        assert ctx.omitted_candidates[1].rank == 3
        assert ctx.omitted_candidates[1].reason == OmissionReason.BUDGET_EXCEEDED
        assert "exhausted by higher-ranked candidates" in ctx.omitted_candidates[1].details

        assert ctx.omitted_candidates[2].record_id == "rec_4#0"
        assert ctx.omitted_candidates[2].rank == 4
        assert ctx.omitted_candidates[2].reason == OmissionReason.BUDGET_EXCEEDED
        assert "exhausted by higher-ranked candidates" in ctx.omitted_candidates[2].details

    def test_first_candidate_exceeds_budget(self, default_query, sample_evidence_list):
        """If candidate 1 cannot fit, returns empty items with omission record."""
        builder = EvidenceContextBuilder(tokenizer=lambda s: len(s))
        response = RetrievalResponse(query=default_query, evidence=sample_evidence_list)

        # Budget of only 10 chars is too small even for single formatted evidence XML
        ctx = builder.build_context(response, allocated_evidence_tokens=10)

        assert len(ctx.items) == 0
        assert ctx.total_tokens == 0
        assert ctx.has_omissions is True
        assert ctx.dropped_candidates_count == 3
        assert ctx.omitted_candidates[0].rank == 1
        assert ctx.omitted_candidates[0].reason == OmissionReason.BUDGET_EXCEEDED
        assert "exceeds allocated budget" in ctx.omitted_candidates[0].details

    def test_fallback_heuristic_token_accounting(self, default_query, sample_evidence_list):
        """When tokenizer is None, builder uses heuristic and marks is_exact_token_count=False."""
        builder = EvidenceContextBuilder(tokenizer=None)
        response = RetrievalResponse(query=default_query, evidence=sample_evidence_list)

        ctx = builder.build_context(response, allocated_evidence_tokens=2000)

        assert len(ctx.items) == 3
        assert ctx.is_exact_token_count is False
        assert ctx.total_tokens > 0

    def test_deterministic_ordering_and_ids(self, default_query, sample_evidence_list):
        """Repeated runs on identical input produce identical evidence IDs and audit logs."""
        builder = EvidenceContextBuilder(tokenizer=lambda s: len(s.split()))
        response = RetrievalResponse(query=default_query, evidence=sample_evidence_list)

        ctx1 = builder.build_context(response, allocated_evidence_tokens=2000)
        ctx2 = builder.build_context(response, allocated_evidence_tokens=2000)

        assert [it.evidence_id for it in ctx1.items] == [it.evidence_id for it in ctx2.items]
        assert [it.citation_label for it in ctx1.items] == [it.citation_label for it in ctx2.items]
        assert ctx1.total_tokens == ctx2.total_tokens
        assert ctx1.dropped_candidates_count == ctx2.dropped_candidates_count

    def test_empty_or_whitespace_candidate_omitted(self, default_query):
        """Candidates with blank or whitespace-only text are omitted with INVALID_CANDIDATE."""
        builder = EvidenceContextBuilder(tokenizer=lambda s: len(s.split()))

        ev_valid = Evidence(
            record_id="rec_val#0",
            chunk_id="c_val",
            text="Valid evidence passage.",
            source_file="Lecture_1.pdf",
            source_pages=[1],
            score=0.90,
        )
        ev_blank = Evidence(
            record_id="rec_blank#0",
            chunk_id="c_blank",
            text="   \n\t  ",
            source_file="Lecture_1.pdf",
            source_pages=[2],
            score=0.80,
        )

        response = RetrievalResponse(query=default_query, evidence=[ev_valid, ev_blank])
        ctx = builder.build_context(response, allocated_evidence_tokens=1000)

        assert len(ctx.items) == 1
        assert ctx.items[0].evidence_id == "E1"
        assert ctx.dropped_candidates_count == 1
        assert ctx.omitted_candidates[0].record_id == "rec_blank#0"
        assert ctx.omitted_candidates[0].reason == OmissionReason.INVALID_CANDIDATE
