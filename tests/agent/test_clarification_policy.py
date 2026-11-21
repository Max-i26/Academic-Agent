"""Unit tests for Stage 6.2 ClarificationPolicy and refusal helpers."""

import pytest
from academic_agent.agent.tools.policy import (
    ClarificationPolicy,
    EvidenceAssessmentCategory,
    QueryAssessmentCategory,
)


class TestClarificationPolicy:
    """Test suite validating query assessment, out-of-domain refusal, and evidence sufficiency evaluation."""

    @pytest.mark.parametrize(
        "query",
        [
            "Explain the conditions under which an AR(1) process is stationary.",
            "Derive the autocorrelation function of an MA(1) model.",
            "What is the difference between AIC and BIC in model selection?",
            "How does Dickey-Fuller test determine unit roots in time series?",
        ],
    )
    def test_valid_academic_queries(self, query: str):
        """Standard academic and course-related queries are categorized as VALID_ACADEMIC and actionable."""
        assessment = ClarificationPolicy.assess_query(query)
        assert assessment.category == QueryAssessmentCategory.VALID_ACADEMIC
        assert assessment.is_actionable is True
        assert assessment.suggested_action == "academic_search"

    @pytest.mark.parametrize(
        "query",
        [
            "",
            "   ",
            "??",
            "a",
        ],
    )
    def test_malformed_queries(self, query: str):
        """Empty, whitespace, or degenerate queries are classified as MALFORMED."""
        assessment = ClarificationPolicy.assess_query(query)
        assert assessment.category == QueryAssessmentCategory.MALFORMED
        assert assessment.is_actionable is False
        assert assessment.suggested_action == "refusal"

    @pytest.mark.parametrize(
        "query",
        [
            "what?",
            "explain",
            "tell me",
            "what is it?",
            "how does it work",
        ],
    )
    def test_underspecified_queries(self, query: str):
        """Vague, highly underspecified queries trigger clarification_request."""
        assessment = ClarificationPolicy.assess_query(query)
        assert assessment.category == QueryAssessmentCategory.UNDERSPECIFIED
        assert assessment.is_actionable is False
        assert assessment.suggested_action == "clarification_request"

    @pytest.mark.parametrize(
        "query",
        [
            "Can you give me a recipe for chocolate cake?",
            "What is the weather forecast for tomorrow?",
            "Tell me a joke about dogs.",
            "What is my daily horoscope for Aries?",
            "Who won the Super Bowl last year?",
        ],
    )
    def test_out_of_domain_queries(self, query: str):
        """Non-academic, lifestyle, or chit-chat queries are classified as OUT_OF_DOMAIN."""
        assessment = ClarificationPolicy.assess_query(query)
        assert assessment.category == QueryAssessmentCategory.OUT_OF_DOMAIN
        assert assessment.is_actionable is False
        assert assessment.suggested_action == "refusal"

    def test_evidence_assessment_empty_candidates(self):
        """Empty evidence list recommends abstention with EMPTY category."""
        assessment = ClarificationPolicy.assess_evidence("AR(1) proof", candidates=[])
        assert assessment.category == EvidenceAssessmentCategory.EMPTY
        assert assessment.can_synthesize is False
        assert assessment.suggested_action == "abstention"

    def test_evidence_assessment_low_confidence_score(self):
        """Evidence candidates with scores below confidence threshold recommend abstention."""
        candidates = [
            {"record_id": "c1", "score": 0.12},
            {"record_id": "c2", "score": 0.18},
        ]
        assessment = ClarificationPolicy.assess_evidence(
            "GARCH volatility",
            candidates=candidates,
            min_confidence_score=0.25,
        )
        assert assessment.category == EvidenceAssessmentCategory.LOW_CONFIDENCE
        assert assessment.can_synthesize is False
        assert assessment.suggested_action == "abstention"

    def test_evidence_assessment_sufficient_evidence(self):
        """High-scoring evidence candidates allow proceeding to academic_synthesizer."""
        candidates = [
            {"record_id": "c1", "score": 0.89},
            {"record_id": "c2", "score": 0.65},
        ]
        assessment = ClarificationPolicy.assess_evidence(
            "Stationarity condition",
            candidates=candidates,
            min_confidence_score=0.25,
        )
        assert assessment.category == EvidenceAssessmentCategory.SUFFICIENT
        assert assessment.can_synthesize is True
        assert assessment.suggested_action == "academic_synthesizer"
