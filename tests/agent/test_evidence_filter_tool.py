"""Unit tests for Stage 6.2 EvidenceFilterTool verifying strict omission auditing and provenance preservation."""

import pytest
from academic_agent.agent.models import ToolErrorCategory
from academic_agent.agent.protocols import ToolProtocol
from academic_agent.agent.tools.filter import EvidenceFilterTool


class TestEvidenceFilterTool:
    """Test suite ensuring EvidenceFilterTool prioritizes, filters, and records structured omission reasons."""

    @pytest.fixture
    def fixture_candidates(self):
        return [
            {
                "record_id": "c1",
                "chunk_id": "ch1",
                "text": "First passage on stationarity condition.",  # ~6 words -> ~8 tokens
                "source_file": "Lecture_1.pdf",
                "score": 0.95,
                "metadata": {"course_id": "MATH_201"},
            },
            {
                "record_id": "c2",
                "chunk_id": "ch2",
                "text": "Second passage on autocorrelation decay.",  # ~5 words -> ~7 tokens
                "source_file": "Lecture_1.pdf",
                "score": 0.85,
                "metadata": {"course_id": "MATH_201"},
            },
            {
                "record_id": "c3",
                "chunk_id": "ch3",
                "text": "Third passage from unrelated lecture.",
                "source_file": "Lecture_9.pdf",
                "score": 0.70,
                "metadata": {"course_id": "MATH_201"},
            },
            {
                "record_id": "c4",
                "chunk_id": "ch4",
                "text": "Fourth passage with very low relevance score.",
                "source_file": "Lecture_1.pdf",
                "score": 0.20,
                "metadata": {"course_id": "MATH_201"},
            },
        ]

    def test_protocol_conformance_and_metadata(self):
        """EvidenceFilterTool satisfies ToolProtocol with read-only idempotent policy."""
        tool = EvidenceFilterTool()
        assert isinstance(tool, ToolProtocol)
        assert tool.name == "evidence_filter"
        assert tool.side_effect_policy.read_only is True
        assert tool.side_effect_policy.idempotent is True
        assert tool.side_effect_policy.timeout_seconds == 5.0

    def test_filter_by_min_score_records_omissions(self, fixture_candidates):
        """Candidates below min_score are omitted with structured omission_reason."""
        tool = EvidenceFilterTool()
        result = tool.execute({
            "candidates": fixture_candidates,
            "min_score": 0.75,
        })

        assert result.success is True
        out = result.output
        assert out["summary"]["retained_count"] == 2
        assert out["summary"]["omitted_count"] == 2

        retained_ids = [c["record_id"] for c in out["retained_candidates"]]
        assert retained_ids == ["c1", "c2"]

        omitted = out["omitted_candidates"]
        omitted_ids = [o["record_id"] for o in omitted]
        assert "c3" in omitted_ids and "c4" in omitted_ids

        # Verify omission rationale is explicitly preserved
        for o in omitted:
            assert o["omission_category"] == "below_score_threshold"
            assert "is below minimum threshold" in o["omission_reason"]

    def test_filter_by_max_candidates(self, fixture_candidates):
        """Ceiling on candidate count retains top candidates and records excess as omitted."""
        tool = EvidenceFilterTool()
        result = tool.execute({
            "candidates": fixture_candidates,
            "max_candidates": 2,
        })

        assert result.success is True
        out = result.output
        assert len(out["retained_candidates"]) == 2
        assert out["retained_candidates"][0]["record_id"] == "c1"
        assert out["retained_candidates"][1]["record_id"] == "c2"

        omitted = out["omitted_candidates"]
        assert len(omitted) == 2
        assert omitted[0]["omission_category"] == "max_candidates_exceeded"
        assert "ceiling reached" in omitted[0]["omission_reason"]

    def test_filter_by_source_file_boundary(self, fixture_candidates):
        """Candidates from non-matching lecture files are omitted with source_mismatch reason."""
        tool = EvidenceFilterTool()
        result = tool.execute({
            "candidates": fixture_candidates,
            "required_source_file": "Lecture_1.pdf",
        })

        assert result.success is True
        out = result.output
        # c3 is from Lecture_9.pdf
        retained_ids = [c["record_id"] for c in out["retained_candidates"]]
        assert "c3" not in retained_ids

        omitted_c3 = next(o for o in out["omitted_candidates"] if o["record_id"] == "c3")
        assert omitted_c3["omission_category"] == "source_mismatch"
        assert "does not match required" in omitted_c3["omission_reason"]

    def test_filter_by_token_budget_ceiling(self, fixture_candidates):
        """Cumulative token budget halts retention and audits all subsequent candidates as omitted."""
        tool = EvidenceFilterTool()
        # c1 has ~8 tokens, c2 has ~7 tokens. Budget of 12 tokens should retain only c1
        result = tool.execute({
            "candidates": fixture_candidates,
            "max_tokens": 12,
        })

        assert result.success is True
        out = result.output
        assert len(out["retained_candidates"]) == 1
        assert out["retained_candidates"][0]["record_id"] == "c1"

        # c2 was omitted because it would exceed 12 tokens
        omitted_c2 = next(o for o in out["omitted_candidates"] if o["record_id"] == "c2")
        assert omitted_c2["omission_category"] == "token_budget_exceeded"
        assert "Token budget exceeded" in omitted_c2["omission_reason"]

    def test_empty_candidates_handling(self):
        """Empty input candidates cleanly produces empty retained and omitted lists."""
        tool = EvidenceFilterTool()
        result = tool.execute({"candidates": []})

        assert result.success is True
        assert result.output["summary"]["total_evaluated"] == 0
        assert result.output["summary"]["retained_count"] == 0
        assert result.output["summary"]["omitted_count"] == 0
