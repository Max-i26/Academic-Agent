"""Unit tests for Stage 6.2 AcademicSynthesizerTool wrapping Phase 5 answer synthesis."""

import pytest
from academic_agent.agent.models import ToolErrorCategory
from academic_agent.agent.protocols import ToolProtocol
from academic_agent.agent.tools.synthesizer import AcademicSynthesizerTool
from academic_agent.generation.models import LLMResponse
from academic_agent.generation.synthesis.models import (
    AcademicAnswer,
    CitationOccurrence,
    EvidenceContext,
    EvidenceSufficiency,
    PolicyOutcome,
    PolicyValidationResult,
    ValidationDiagnostics,
)
from academic_agent.generation.synthesis.protocols import AcademicSynthesizerProtocol
from academic_agent.retrieval.models import RetrievalResponse


class MockSynthesizer(AcademicSynthesizerProtocol):
    """Mock synthesizer returning a verified AcademicAnswer."""

    def __init__(self, outcome: PolicyOutcome = PolicyOutcome.ANSWERED) -> None:
        self.outcome = outcome
        self.last_query: str | None = None
        self.last_response: RetrievalResponse | None = None

    def synthesize(
        self,
        query: str,
        retrieval_response: RetrievalResponse,
        config=None,
        allocated_evidence_tokens=None,
        sufficiency=None,
    ) -> AcademicAnswer:
        self.last_query = query
        self.last_response = retrieval_response

        citation = CitationOccurrence(
            evidence_id="E1",
            raw_citation="[E1]",
            start_char=40,
            end_char=44,
            is_valid=True,
            resolution_label="[Lecture 2, p. 14]",
        )
        return AcademicAnswer(
            answer_text="An AR(1) process is stationary if |phi| < 1 [E1].",
            resolved_answer_text="An AR(1) process is stationary if |phi| < 1 [Lecture 2, p. 14].",
            query=query,
            validation=ValidationDiagnostics(
                citations=[citation],
                valid_cited_ids=["E1"],
                invalid_cited_ids=[],
                uncited_evidence_ids=[],
                evidence_sufficiency=EvidenceSufficiency.SUFFICIENT,
            ),
            policy_result=PolicyValidationResult(
                outcome=self.outcome,
                is_compliant=True,
                explanation="Verified answer.",
            ),
            evidence_context=EvidenceContext(
                items=[],
                allocated_budget=500,
                consumed_tokens=100,
                sufficiency=EvidenceSufficiency.SUFFICIENT,
            ),
            raw_response=LLMResponse(
                content="An AR(1) process is stationary if |phi| < 1 [E1].",
                prompt_tokens=80,
                completion_tokens=20,
                total_tokens=100,
                model_name="mock_llm",
                latency_ms=15.0,
            ),
            model_name="mock_llm",
            latency_ms=45.2,
        )


class TestAcademicSynthesizerTool:
    """Test suite validating AcademicSynthesizerTool interface, input mapping, and output extraction."""

    def test_protocol_conformance_and_metadata(self):
        """AcademicSynthesizerTool satisfies ToolProtocol with read-only idempotent policy."""
        mock_syn = MockSynthesizer()
        tool = AcademicSynthesizerTool(synthesizer=mock_syn)

        assert isinstance(tool, ToolProtocol)
        assert tool.name == "academic_synthesizer"
        assert tool.side_effect_policy.read_only is True
        assert tool.side_effect_policy.idempotent is True
        assert tool.side_effect_policy.timeout_seconds == 30.0

        schema = tool.get_schema()
        assert schema["type"] == "object"
        assert "query" in schema["required"]
        assert "evidence_candidates" in schema["properties"]

    def test_successful_synthesis_execution(self):
        """Candidate dictionaries are converted to Evidence and synthesized into an audit observation."""
        mock_syn = MockSynthesizer()
        tool = AcademicSynthesizerTool(synthesizer=mock_syn)

        candidates = [
            {
                "record_id": "rec_1",
                "chunk_id": "chunk_1",
                "text": "Condition for stationarity in AR(1).",
                "source_file": "Lecture_2.pdf",
                "source_pages": [14],
                "score": 0.95,
            }
        ]

        result = tool.execute({
            "query": "When is an AR(1) process stationary?",
            "evidence_candidates": candidates,
            "allocated_evidence_tokens": 600,
        })

        assert result.success is True
        assert result.tool_name == "academic_synthesizer"

        out = result.output
        assert "stationary if |phi| < 1 [Lecture 2, p. 14]" in out["answer_text"]
        assert out["raw_answer"] == "An AR(1) process is stationary if |phi| < 1 [E1]."
        assert out["policy_outcome"] == "answered"
        assert out["is_valid_academic_answer"] is True
        assert out["citations_count"] == 1
        assert out["citations"][0]["raw_citation"] == "[E1]"
        assert out["valid_cited_ids"] == ["E1"]
        assert out["latency_ms"] == 45.2

        # Verify underlying call received converted evidence
        assert mock_syn.last_query == "When is an AR(1) process stationary?"
        assert len(mock_syn.last_response.evidence) == 1
        assert mock_syn.last_response.evidence[0].record_id == "rec_1"

    def test_empty_query_validation_rejection(self):
        """Empty query fails schema validation with VALIDATION_ERROR."""
        mock_syn = MockSynthesizer()
        tool = AcademicSynthesizerTool(synthesizer=mock_syn)

        result = tool.execute({"query": "", "evidence_candidates": []})
        assert result.success is False
        assert result.error_category == ToolErrorCategory.VALIDATION_ERROR
