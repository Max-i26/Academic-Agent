"""Unit tests for Stage 5.2A Evidence Grounding and Synthesis data models and contracts."""

import pytest
from pydantic import ValidationError

from academic_agent.generation.models import LLMResponse, LLMUsage, MessageRole
from academic_agent.generation.synthesis.models import (
    AcademicAnswer,
    CitationOccurrence,
    EvidenceContext,
    EvidenceSufficiency,
    GroundedEvidence,
    ValidationDiagnostics,
)
from academic_agent.generation.synthesis.protocols import (
    AcademicSynthesizerProtocol,
    CitationValidatorProtocol,
    EvidenceContextBuilderProtocol,
    EvidenceFormatterProtocol,
    EvidenceSufficiencyAssessorProtocol,
    GroundingPromptBuilderProtocol,
)
from academic_agent.retrieval.models import Evidence


@pytest.fixture
def sample_evidence_1() -> Evidence:
    return Evidence(
        record_id="chunk_1#0",
        chunk_id="chunk_1",
        text="Weak stationarity requires constant mean and autocovariance.",
        source_file="Lecture_3.pdf",
        source_pages=[12],
        score=0.92,
    )


@pytest.fixture
def sample_evidence_2() -> Evidence:
    return Evidence(
        record_id="chunk_2#0",
        chunk_id="chunk_2",
        text="An AR(1) process has theoretical variance sigma^2 / (1 - phi_1^2).",
        source_file="Lecture_5.pdf",
        source_pages=[8, 9],
        score=0.88,
    )


class TestEvidenceSufficiency:
    """Verify EvidenceSufficiency enum values and baseline defaults."""

    def test_enum_values(self):
        assert EvidenceSufficiency.UNKNOWN == "unknown"
        assert EvidenceSufficiency.SUFFICIENT == "sufficient"
        assert EvidenceSufficiency.PARTIALLY_SUFFICIENT == "partial"
        assert EvidenceSufficiency.INSUFFICIENT == "insufficient"

    def test_default_is_unknown(self):
        diag = ValidationDiagnostics()
        assert diag.evidence_sufficiency == EvidenceSufficiency.UNKNOWN

        ans = AcademicAnswer(
            answer_text="Sample answer",
            resolved_answer_text="Sample resolved",
            query="Sample query",
            evidence_context=EvidenceContext(),
            validation=diag,
            raw_response=LLMResponse(
                content="Sample",
                model_name="mock",
                latency_ms=10.0,
                usage=LLMUsage(),
            ),
            model_name="mock",
            latency_ms=10.0,
        )
        assert ans.evidence_sufficiency == EvidenceSufficiency.UNKNOWN


class TestGroundedEvidence:
    """Verify strict ID validation, provenance citation derivation, and properties."""

    def test_valid_grounded_evidence(self, sample_evidence_1):
        ge = GroundedEvidence(evidence_id="E1", evidence=sample_evidence_1)
        assert ge.evidence_id == "E1"
        assert ge.tag == "[E1]"
        assert ge.citation_label == "[Lecture 3, p. 12]"
        assert ge.source_file == "Lecture_3.pdf"
        assert ge.source_pages == [12]
        assert ge.text == sample_evidence_1.text
        assert ge.token_count == 0
        assert ge.is_truncated is False

    def test_multi_page_citation_derivation(self, sample_evidence_2):
        ge = GroundedEvidence(evidence_id="E2", evidence=sample_evidence_2)
        assert ge.citation_label == "[Lecture 5, pp. 8-9]"

    @pytest.mark.parametrize("invalid_id", ["E0", "e1", "1", "E", "E-1", "E_1", "ID1"])
    def test_reject_invalid_id_formats(self, sample_evidence_1, invalid_id):
        with pytest.raises(ValidationError):
            GroundedEvidence(evidence_id=invalid_id, evidence=sample_evidence_1)

    def test_citation_label_matching_provenance_passes(self, sample_evidence_1):
        ge = GroundedEvidence(
            evidence_id="E1",
            evidence=sample_evidence_1,
            citation_label="[Lecture 3, p. 12]",
        )
        assert ge.citation_label == "[Lecture 3, p. 12]"

    def test_reject_conflicting_citation_label(self, sample_evidence_1):
        with pytest.raises(ValidationError, match="does not match underlying evidence provenance"):
            GroundedEvidence(
                evidence_id="E1",
                evidence=sample_evidence_1,
                citation_label="[Lecture 99, p. 999]",
            )


class TestEvidenceContext:
    """Verify EvidenceContext canonical items, uniqueness, and lookup derivation."""

    def test_canonical_items_and_dynamic_lookup(self, sample_evidence_1, sample_evidence_2):
        ge1 = GroundedEvidence(evidence_id="E1", evidence=sample_evidence_1)
        ge2 = GroundedEvidence(evidence_id="E2", evidence=sample_evidence_2)

        ctx = EvidenceContext(items=[ge1, ge2], budget_allocated=1200)
        assert len(ctx.items) == 2
        assert ctx.evidence_lookup == {"E1": ge1, "E2": ge2}
        assert ctx.get("E1") is ge1
        assert ctx.get("[E1]") is ge1
        assert ctx.get("E2") is ge2
        assert ctx.get("E3") is None
        assert ctx.has_omissions is False
        assert ctx.dropped_candidates_count == 0
        assert ctx.is_exact_token_count is True

    def test_reject_duplicate_evidence_ids(self, sample_evidence_1, sample_evidence_2):
        ge1 = GroundedEvidence(evidence_id="E1", evidence=sample_evidence_1)
        ge2_duplicate = GroundedEvidence(evidence_id="E1", evidence=sample_evidence_2)

        with pytest.raises(ValidationError, match="Duplicate evidence_id 'E1'"):
            EvidenceContext(items=[ge1, ge2_duplicate])

    def test_has_omissions_and_dropped_candidates(self, sample_evidence_1):
        ge1 = GroundedEvidence(evidence_id="E1", evidence=sample_evidence_1)
        ctx = EvidenceContext(
            items=[ge1],
            budget_allocated=500,
            has_omissions=True,
            dropped_candidates_count=3,
            is_exact_token_count=False,
        )
        assert ctx.has_omissions is True
        assert ctx.dropped_candidates_count == 3
        assert ctx.is_exact_token_count is False


class TestCitationOccurrence:
    """Verify CitationOccurrence span offsets and validity constraints."""

    def test_valid_occurrence(self):
        occ = CitationOccurrence(
            evidence_id="E1",
            raw_citation="[E1]",
            start_char=10,
            end_char=14,
            is_valid=True,
        )
        assert occ.evidence_id == "E1"
        assert occ.start_char == 10
        assert occ.end_char == 14
        assert occ.is_valid is True

    def test_reject_invalid_offsets(self):
        with pytest.raises(ValidationError, match="must be strictly greater than start_char"):
            CitationOccurrence(
                evidence_id="E1",
                raw_citation="[E1]",
                start_char=10,
                end_char=10,
            )

        with pytest.raises(ValidationError, match="must be strictly greater than start_char"):
            CitationOccurrence(
                evidence_id="E1",
                raw_citation="[E1]",
                start_char=15,
                end_char=10,
            )


class TestValidationDiagnostics:
    """Verify citation diagnostics tracking, spans, and uniqueness ordering."""

    def test_empty_diagnostics_defaults(self):
        diag = ValidationDiagnostics()
        assert diag.citations == []
        assert diag.valid_cited_ids == []
        assert diag.invalid_cited_ids == []
        assert diag.uncited_evidence_ids == []
        assert diag.malformed_citations == []
        assert diag.refusal_detected is False
        assert diag.evidence_sufficiency == EvidenceSufficiency.UNKNOWN
        assert diag.cited_ids == []

    def test_ordered_unique_cited_ids(self):
        c1 = CitationOccurrence(evidence_id="E1", raw_citation="[E1]", start_char=0, end_char=4, is_valid=True)
        c2 = CitationOccurrence(evidence_id="E2", raw_citation="[E2]", start_char=10, end_char=14, is_valid=True)
        c3 = CitationOccurrence(evidence_id="E1", raw_citation="[E1]", start_char=20, end_char=24, is_valid=True)

        diag = ValidationDiagnostics(citations=[c1, c2, c3], valid_cited_ids=["E1", "E2"])
        assert diag.cited_ids == ["E1", "E2"]


class TestAcademicAnswer:
    """Verify decoupled citation integrity, refusal, and abstention properties."""

    @pytest.fixture
    def base_answer_kwargs(self, sample_evidence_1):
        ge = GroundedEvidence(evidence_id="E1", evidence=sample_evidence_1)
        ctx = EvidenceContext(items=[ge])
        return {
            "answer_text": "Answer text [E1]",
            "resolved_answer_text": "Answer text [Lecture 3, p. 12]",
            "query": "What is stationarity?",
            "evidence_context": ctx,
            "raw_response": LLMResponse(content="Answer text [E1]", model_name="mock", latency_ms=15.0, usage=LLMUsage()),
            "model_name": "mock-model",
            "latency_ms": 15.0,
        }

    def test_has_valid_citation_integrity_true(self, base_answer_kwargs):
        diag = ValidationDiagnostics(valid_cited_ids=["E1"], invalid_cited_ids=[])
        ans = AcademicAnswer(validation=diag, **base_answer_kwargs)
        assert ans.has_valid_citation_integrity is True

    def test_has_valid_citation_integrity_false_on_invalid_citations(self, base_answer_kwargs):
        diag = ValidationDiagnostics(valid_cited_ids=["E1"], invalid_cited_ids=["E99"])
        ans = AcademicAnswer(validation=diag, **base_answer_kwargs)
        assert ans.has_valid_citation_integrity is False

    def test_has_valid_citation_integrity_false_when_no_citations(self, base_answer_kwargs):
        diag = ValidationDiagnostics(valid_cited_ids=[], invalid_cited_ids=[])
        ans = AcademicAnswer(validation=diag, **base_answer_kwargs)
        assert ans.has_valid_citation_integrity is False

    def test_refusal_and_abstention_separation(self, base_answer_kwargs):
        # Case 1: Refusal detected, but evidence was sufficient (unjustified refusal)
        diag1 = ValidationDiagnostics(refusal_detected=True, evidence_sufficiency=EvidenceSufficiency.SUFFICIENT)
        ans1 = AcademicAnswer(validation=diag1, evidence_sufficiency=EvidenceSufficiency.SUFFICIENT, **base_answer_kwargs)
        assert ans1.refusal_detected is True
        assert ans1.is_abstention_required is False
        assert ans1.is_compliant_abstention is False

        # Case 2: Abstention required, but model failed to refuse (hallucination)
        diag2 = ValidationDiagnostics(refusal_detected=False, evidence_sufficiency=EvidenceSufficiency.INSUFFICIENT)
        ans2 = AcademicAnswer(validation=diag2, evidence_sufficiency=EvidenceSufficiency.INSUFFICIENT, **base_answer_kwargs)
        assert ans2.refusal_detected is False
        assert ans2.is_abstention_required is True
        assert ans2.is_compliant_abstention is False

        # Case 3: Abstention required AND model cleanly refused (compliant abstention)
        diag3 = ValidationDiagnostics(refusal_detected=True, evidence_sufficiency=EvidenceSufficiency.INSUFFICIENT)
        ans3 = AcademicAnswer(validation=diag3, evidence_sufficiency=EvidenceSufficiency.INSUFFICIENT, **base_answer_kwargs)
        assert ans3.refusal_detected is True
        assert ans3.is_abstention_required is True
        assert ans3.is_compliant_abstention is True


class TestProtocolsRuntimeCheckable:
    """Verify that all six Stage 5.2 protocols are runtime-checkable."""

    def test_protocols(self):
        class DummyFormatter:
            def format_evidence_block(self, context): ...

        class DummyContextBuilder:
            def build_context(self, retrieval_response, allocated_evidence_tokens): ...

        class DummyPromptBuilder:
            def build_messages(self, query, context): ...

        class DummyCitationValidator:
            def validate(self, answer_text, context): ...
            def resolve_citations(self, answer_text, context): ...

        class DummySufficiencyAssessor:
            def assess(self, query, context): ...

        class DummySynthesizer:
            def synthesize(self, query, retrieval_response, config=None): ...

        assert isinstance(DummyFormatter(), EvidenceFormatterProtocol)
        assert isinstance(DummyContextBuilder(), EvidenceContextBuilderProtocol)
        assert isinstance(DummyPromptBuilder(), GroundingPromptBuilderProtocol)
        assert isinstance(DummyCitationValidator(), CitationValidatorProtocol)
        assert isinstance(DummySufficiencyAssessor(), EvidenceSufficiencyAssessorProtocol)
        assert isinstance(DummySynthesizer(), AcademicSynthesizerProtocol)
