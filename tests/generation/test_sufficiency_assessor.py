"""Tests for DeterministicSufficiencyAssessor and evidence sufficiency contracts."""

import pytest
from academic_agent.generation.synthesis.models import (
    EvidenceContext,
    EvidenceSufficiency,
    GroundedEvidence,
)
from academic_agent.generation.synthesis.protocols import EvidenceSufficiencyAssessorProtocol
from academic_agent.generation.synthesis.sufficiency.assessor import DeterministicSufficiencyAssessor
from academic_agent.retrieval.models import Evidence


@pytest.fixture
def sample_stationarity_evidence() -> EvidenceContext:
    ev1 = Evidence(
        record_id="rec_1#0",
        chunk_id="c_1",
        text="Weak stationarity requires a constant mean and autocovariance function depending only on lag h.",
        source_file="Lecture_1.pdf",
        source_pages=[10],
        score=0.95,
    )
    ev2 = Evidence(
        record_id="rec_2#0",
        chunk_id="c_2",
        text="The autoregressive model AR(1) satisfies X_t = phi_1 X_{t-1} + w_t where |phi_1| < 1 ensures causality.",
        source_file="Lecture_2.pdf",
        source_pages=[15],
        score=0.90,
    )
    items = [
        GroundedEvidence(
            evidence_id="E1",
            evidence=ev1,
            citation_label=ev1.format_citation(),
            token_count=25,
        ),
        GroundedEvidence(
            evidence_id="E2",
            evidence=ev2,
            citation_label=ev2.format_citation(),
            token_count=30,
        ),
    ]
    return EvidenceContext(items=items)


class TestDeterministicSufficiencyAssessor:
    """Test suite verifying the deterministic baseline sufficiency assessor."""

    def test_protocol_conformance(self):
        """Assessor must satisfy EvidenceSufficiencyAssessorProtocol."""
        assessor = DeterministicSufficiencyAssessor()
        assert isinstance(assessor, EvidenceSufficiencyAssessorProtocol)

    def test_invalid_thresholds(self):
        """Invalid threshold parameters must raise ValueError."""
        with pytest.raises(ValueError, match="Thresholds must satisfy"):
            DeterministicSufficiencyAssessor(min_coverage_ratio=0.2, partial_coverage_ratio=0.5)

        with pytest.raises(ValueError, match="Thresholds must satisfy"):
            DeterministicSufficiencyAssessor(min_coverage_ratio=1.2, partial_coverage_ratio=0.5)

    def test_empty_or_whitespace_evidence_is_insufficient(self):
        """Empty context or blank evidence items must be classified as INSUFFICIENT."""
        assessor = DeterministicSufficiencyAssessor()
        empty_ctx = EvidenceContext(items=[])
        assert assessor.assess("Explain weak stationarity", empty_ctx) == EvidenceSufficiency.INSUFFICIENT

    def test_fully_covered_single_clause_query(self, sample_stationarity_evidence):
        """Query whose concepts are thoroughly present in evidence must be SUFFICIENT."""
        assessor = DeterministicSufficiencyAssessor()
        query = "Explain weak stationarity and constant autocovariance function."
        result = assessor.assess(query, sample_stationarity_evidence)
        assert result == EvidenceSufficiency.SUFFICIENT

    def test_multi_clause_partial_coverage(self, sample_stationarity_evidence):
        """Multi-part question where one concept is covered and the other is absent is PARTIALLY_SUFFICIENT."""
        assessor = DeterministicSufficiencyAssessor()
        # AR(1) is covered in E2, but GARCH / volatility clustering is absent from all items
        query = "Explain AR(1) causality and compare with GARCH volatility clustering."
        result = assessor.assess(query, sample_stationarity_evidence)
        assert result == EvidenceSufficiency.PARTIALLY_SUFFICIENT

    def test_completely_irrelevant_query(self, sample_stationarity_evidence):
        """Query with zero conceptual overlap in evidence is INSUFFICIENT."""
        assessor = DeterministicSufficiencyAssessor()
        query = "What are the stages of biological cellular mitosis in eukaryotic organisms?"
        result = assessor.assess(query, sample_stationarity_evidence)
        assert result == EvidenceSufficiency.INSUFFICIENT

    def test_query_with_no_substantive_concepts_returns_unknown(self, sample_stationarity_evidence):
        """Query consisting only of stopwords returns UNKNOWN sufficiency."""
        assessor = DeterministicSufficiencyAssessor()
        query = "what is it and how do you do it?"
        result = assessor.assess(query, sample_stationarity_evidence)
        assert result == EvidenceSufficiency.UNKNOWN

    def test_substring_word_boundary_prevents_false_positive(self):
        """Assessor must not match concept 'ar' inside unrelated words like 'particular' or 'linear'."""
        ev = Evidence(
            record_id="rec_unrelated#0",
            chunk_id="c_unrelated",
            text="In this particular section, we consider a linear function of time.",
            source_file="Lecture_3.pdf",
            source_pages=[5],
            score=0.85,
        )
        ctx = EvidenceContext(items=[
            GroundedEvidence(
                evidence_id="E1",
                evidence=ev,
                citation_label=ev.format_citation(),
                token_count=20,
            )
        ])

        assessor = DeterministicSufficiencyAssessor()
        # Query concept 'ar' (e.g. from AR model) should NOT match 'particular' or 'linear'
        query = "Explain AR process properties."
        result = assessor.assess(query, ctx)
        assert result == EvidenceSufficiency.INSUFFICIENT
