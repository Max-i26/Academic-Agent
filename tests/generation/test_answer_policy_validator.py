"""Unit tests for Stage 5.2C.3 AnswerPolicyValidator."""

import pytest

from academic_agent.generation.synthesis.models import (
    EvidenceContext,
    EvidenceSufficiency,
    GroundedEvidence,
    PolicyOutcome,
    PolicyValidationResult,
    ValidationDiagnostics,
)
from academic_agent.generation.synthesis.protocols import AnswerPolicyValidatorProtocol
from academic_agent.generation.synthesis.validation.answer_policy_validator import AnswerPolicyValidator
from academic_agent.generation.synthesis.validation.citation_validator import CitationValidator
from academic_agent.retrieval.models import Evidence


@pytest.fixture
def sample_evidence_e1() -> GroundedEvidence:
    ev = Evidence(
        record_id="rec_1#0",
        chunk_id="c1",
        text="Weak stationarity requires constant mean and autocovariance.",
        source_file="Lecture_4.pdf",
        source_pages=[18],
        score=0.92,
    )
    return GroundedEvidence(evidence_id="E1", evidence=ev)


@pytest.fixture
def sample_context(sample_evidence_e1) -> EvidenceContext:
    return EvidenceContext(
        items=[sample_evidence_e1],
        budget_allocated=1000,
    )


@pytest.fixture
def citation_validator() -> CitationValidator:
    return CitationValidator()


class TestAnswerPolicyValidatorProtocolAndBasics:
    """Verify protocol conformance and policy outcome enumeration."""

    def test_protocol_conformance(self):
        validator = AnswerPolicyValidator()
        assert isinstance(validator, AnswerPolicyValidatorProtocol)

    def test_policy_outcomes_enumeration(self):
        assert PolicyOutcome.ANSWERED == "answered"
        assert PolicyOutcome.COMPLIANT_ABSTENTION == "compliant_abstention"
        assert PolicyOutcome.PARTIAL_ANSWER == "partial_answer"
        assert PolicyOutcome.CONTRADICTORY_REFUSAL == "contradictory_refusal"
        assert PolicyOutcome.UNSUPPORTED_ANSWER == "unsupported_answer"
        assert PolicyOutcome.INDETERMINATE == "indeterminate"


class TestPolicyOutcomes:
    """Verify all 14 required answering and abstention policy scenarios."""

    def test_clear_compliant_abstention(self, sample_context, citation_validator):
        """Case 1: Model explicitly refuses under insufficient evidence without making unsupported claims."""
        validator = AnswerPolicyValidator()
        text = (
            "The provided lectures do not contain enough information to establish the answer. "
            "I cannot determine this reliably from the available evidence."
        )
        diag = citation_validator.validate(text, sample_context)

        result = validator.validate_policy(
            answer_text=text,
            context=sample_context,
            diagnostics=diag,
            sufficiency=EvidenceSufficiency.INSUFFICIENT,
        )

        assert result.outcome == PolicyOutcome.COMPLIANT_ABSTENTION
        assert result.is_compliant is True
        assert result.refusal_detected is True
        assert result.has_substantive_continuation is False
        assert len(result.refusal_phrases) >= 1

    def test_refusal_followed_by_substantive_claim(self, sample_context, citation_validator):
        """Case 2: Model refuses, but then proceeds to make substantive domain claims."""
        validator = AnswerPolicyValidator()
        text = "I don't have enough evidence to answer this. However, the correct model is definitely ARIMA(2,1,1)."
        diag = citation_validator.validate(text, sample_context)

        result = validator.validate_policy(
            answer_text=text,
            context=sample_context,
            diagnostics=diag,
            sufficiency=EvidenceSufficiency.INSUFFICIENT,
        )

        assert result.outcome == PolicyOutcome.CONTRADICTORY_REFUSAL
        assert result.is_compliant is False
        assert result.refusal_detected is True
        assert result.has_substantive_continuation is True

    def test_insufficient_evidence_with_unsupported_answer(self, sample_context, citation_validator):
        """Case 3: Evidence is insufficient, but model gives an answer without abstaining."""
        validator = AnswerPolicyValidator()
        text = "An AR(1) process is stationary when the autoregressive coefficient is strictly less than one."
        diag = citation_validator.validate(text, sample_context)

        result = validator.validate_policy(
            answer_text=text,
            context=sample_context,
            diagnostics=diag,
            sufficiency=EvidenceSufficiency.INSUFFICIENT,
        )

        assert result.outcome == PolicyOutcome.UNSUPPORTED_ANSWER
        assert result.is_compliant is False
        assert result.refusal_detected is False

    def test_valid_partial_answer(self, sample_context, citation_validator):
        """Case 4: Supported portions are cited, and unsupported portions are explicitly identified."""
        validator = AnswerPolicyValidator()
        text = (
            "The ACF measures correlation between observations at different lags [E1]. "
            "However, the provided material does not establish the appropriate model order."
        )
        diag = citation_validator.validate(text, sample_context)

        result = validator.validate_policy(
            answer_text=text,
            context=sample_context,
            diagnostics=diag,
            sufficiency=EvidenceSufficiency.PARTIALLY_SUFFICIENT,
        )

        assert result.outcome == PolicyOutcome.PARTIAL_ANSWER
        assert result.is_compliant is True
        assert result.has_gap_statement is True

    def test_partial_answer_without_identifying_unsupported_portions(self, sample_context, citation_validator):
        """Case 5: Partial evidence available, but model answers without acknowledging missing parts."""
        validator = AnswerPolicyValidator()
        text = (
            "The ACF measures correlation between observations at different lags [E1]. "
            "The model order is definitely three."
        )
        diag = citation_validator.validate(text, sample_context)

        result = validator.validate_policy(
            answer_text=text,
            context=sample_context,
            diagnostics=diag,
            sufficiency=EvidenceSufficiency.PARTIALLY_SUFFICIENT,
        )

        assert result.outcome == PolicyOutcome.UNSUPPORTED_ANSWER
        assert result.is_compliant is False
        assert result.has_gap_statement is False

    def test_empty_evidence_context_compliant_and_unsupported(self, citation_validator):
        """Case 6: Empty context correctly yields compliant abstention or unsupported answer."""
        validator = AnswerPolicyValidator()
        empty_ctx = EvidenceContext()

        # Case 6a: Compliant abstention on empty context
        text_abstain = "No information was provided in the evidence to answer this question."
        diag_a = citation_validator.validate(text_abstain, empty_ctx)
        res_a = validator.validate_policy(text_abstain, empty_ctx, diag_a, EvidenceSufficiency.INSUFFICIENT)
        assert res_a.outcome == PolicyOutcome.COMPLIANT_ABSTENTION
        assert res_a.is_compliant is True

        # Case 6b: Unsupported claims on empty context
        text_claims = "Weak stationarity requires constant mean and constant variance."
        diag_b = citation_validator.validate(text_claims, empty_ctx)
        res_b = validator.validate_policy(text_claims, empty_ctx, diag_b, EvidenceSufficiency.INSUFFICIENT)
        assert res_b.outcome == PolicyOutcome.UNSUPPORTED_ANSWER
        assert res_b.is_compliant is False

    def test_valid_citations_with_insufficient_evidence_never_answered(self, sample_context, citation_validator):
        """Case 7: INSUFFICIENT evidence must never produce ANSWERED, even with valid citations."""
        validator = AnswerPolicyValidator()
        text = "The parameter is positive [E1]."
        diag = citation_validator.validate(text, sample_context)

        result = validator.validate_policy(
            answer_text=text,
            context=sample_context,
            diagnostics=diag,
            sufficiency=EvidenceSufficiency.INSUFFICIENT,
        )

        assert result.outcome != PolicyOutcome.ANSWERED
        assert result.outcome == PolicyOutcome.UNSUPPORTED_ANSWER
        assert result.is_compliant is False

    def test_invalid_citations_with_otherwise_plausible_answer(self, sample_context, citation_validator):
        """Case 8: Fabricated citation IDs [E99] under SUFFICIENT evidence violate policy."""
        validator = AnswerPolicyValidator()
        text = "Weak stationarity requires constant mean and autocovariance [E99]."
        diag = citation_validator.validate(text, sample_context)

        result = validator.validate_policy(
            answer_text=text,
            context=sample_context,
            diagnostics=diag,
            sufficiency=EvidenceSufficiency.SUFFICIENT,
        )

        assert result.outcome == PolicyOutcome.UNSUPPORTED_ANSWER
        assert result.is_compliant is False
        assert "E99" in result.explanation

    def test_refusal_language_in_non_refusal_context(self, sample_context, citation_validator):
        """Case 9: Phrase mentioning hypothesis testing of insufficient evidence does not trigger false refusal."""
        validator = AnswerPolicyValidator()
        text = "In formal time series testing, we reject the hypothesis that evidence is insufficient [E1]."
        diag = citation_validator.validate(text, sample_context)

        result = validator.validate_policy(
            answer_text=text,
            context=sample_context,
            diagnostics=diag,
            sufficiency=EvidenceSufficiency.SUFFICIENT,
        )

        assert result.refusal_detected is False
        assert result.outcome == PolicyOutcome.ANSWERED
        assert result.is_compliant is True

    def test_multiple_refusal_statements(self, sample_context, citation_validator):
        """Case 10: Multiple refusal statements in text are audited cleanly."""
        validator = AnswerPolicyValidator()
        text = (
            "I cannot determine the answer from the lecture notes. "
            "Insufficient evidence is provided to verify this model. "
            "No information was provided regarding the parameters."
        )
        diag = citation_validator.validate(text, sample_context)

        result = validator.validate_policy(
            answer_text=text,
            context=sample_context,
            diagnostics=diag,
            sufficiency=EvidenceSufficiency.INSUFFICIENT,
        )

        assert result.outcome == PolicyOutcome.COMPLIANT_ABSTENTION
        assert result.refusal_detected is True
        assert len(result.refusal_phrases) >= 2

    def test_mathematical_derivation_following_refusal(self, sample_context, citation_validator):
        """Case 11: Equations or LaTeX formulas following a refusal trigger CONTRADICTORY_REFUSAL."""
        validator = AnswerPolicyValidator()
        text = "I cannot find sufficient evidence. However, let $\\sigma^2 = \\frac{\\gamma_0}{1 - \\phi_1^2}$."
        diag = citation_validator.validate(text, sample_context)

        result = validator.validate_policy(
            answer_text=text,
            context=sample_context,
            diagnostics=diag,
            sufficiency=EvidenceSufficiency.INSUFFICIENT,
        )

        assert result.outcome == PolicyOutcome.CONTRADICTORY_REFUSAL
        assert result.is_compliant is False
        assert result.has_substantive_continuation is True

    def test_unknown_evidence_sufficiency(self, sample_context, citation_validator):
        """Case 12: UNKNOWN sufficiency without explicit violations produces INDETERMINATE."""
        validator = AnswerPolicyValidator()
        text = "General discussion without citations."
        diag = citation_validator.validate(text, sample_context)

        result = validator.validate_policy(
            answer_text=text,
            context=sample_context,
            diagnostics=diag,
            sufficiency=EvidenceSufficiency.UNKNOWN,
        )

        assert result.outcome == PolicyOutcome.INDETERMINATE
        assert result.is_compliant is False

    def test_no_refusal_and_sufficient_evidence(self, sample_context, citation_validator):
        """Case 13: Sufficient evidence with valid citations produces ANSWERED."""
        validator = AnswerPolicyValidator()
        text = "Weak stationarity requires constant mean and autocovariance depending only on lag [E1]."
        diag = citation_validator.validate(text, sample_context)

        result = validator.validate_policy(
            answer_text=text,
            context=sample_context,
            diagnostics=diag,
            sufficiency=EvidenceSufficiency.SUFFICIENT,
        )

        assert result.outcome == PolicyOutcome.ANSWERED
        assert result.is_compliant is True
        assert result.refusal_detected is False

    def test_contradictory_statements_across_multiple_paragraphs(self, sample_context, citation_validator):
        """Case 14: Refusal in paragraph 1, followed by substantive claims in paragraph 2."""
        validator = AnswerPolicyValidator()
        text = (
            "The provided lectures do not contain enough information to establish the answer.\n\n"
            "The optimal lag length for this series is $p=3$ and the model parameters are $\\phi_1=0.5, \\phi_2=-0.2$."
        )
        diag = citation_validator.validate(text, sample_context)

        result = validator.validate_policy(
            answer_text=text,
            context=sample_context,
            diagnostics=diag,
            sufficiency=EvidenceSufficiency.INSUFFICIENT,
        )

        assert result.outcome == PolicyOutcome.CONTRADICTORY_REFUSAL
        assert result.is_compliant is False
        assert result.has_substantive_continuation is True

    def test_partial_answer_with_fabricated_citation_rejected(self, sample_context, citation_validator):
        """Adversarial H1/R1: Fabricated citation in partial answer must be rejected as UNSUPPORTED_ANSWER."""
        validator = AnswerPolicyValidator()
        text = (
            "Weak stationarity requires constant mean [E1]. "
            "However, the lectures do not mention ARCH effects [E99]."
        )
        diag = citation_validator.validate(text, sample_context)
        assert "E99" in diag.invalid_cited_ids

        result = validator.validate_policy(
            answer_text=text,
            context=sample_context,
            diagnostics=diag,
            sufficiency=EvidenceSufficiency.PARTIALLY_SUFFICIENT,
        )

        assert result.outcome == PolicyOutcome.UNSUPPORTED_ANSWER
        assert result.is_compliant is False
        assert "invalid or fabricated citation IDs" in result.explanation

    def test_clean_refusal_without_gap_statement_under_partial_sufficiency(self, sample_context, citation_validator):
        """Case C refinement: Clean refusal with no valid citations under partial sufficiency is COMPLIANT_ABSTENTION."""
        validator = AnswerPolicyValidator()
        text = "I do not have sufficient information in the provided lectures to answer this question."
        diag = citation_validator.validate(text, sample_context)

        result = validator.validate_policy(
            answer_text=text,
            context=sample_context,
            diagnostics=diag,
            sufficiency=EvidenceSufficiency.PARTIALLY_SUFFICIENT,
        )

        assert result.outcome == PolicyOutcome.COMPLIANT_ABSTENTION
        assert result.is_compliant is True
        assert result.refusal_detected is True
