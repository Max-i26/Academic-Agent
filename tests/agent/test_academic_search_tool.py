"""Unit tests for Stage 6.2 AcademicSearchTool wrapping Phase 4 hybrid retrieval."""

import pytest
from academic_agent.agent.models import ToolErrorCategory
from academic_agent.agent.protocols import ToolProtocol
from academic_agent.agent.tools.search import AcademicSearchTool
from academic_agent.retrieval.models import (
    Evidence,
    RetrievalDiagnostics,
    RetrievalQuery,
    RetrievalResponse,
)


class MockRetrievalPipeline:
    """Mock Phase 4 retrieval pipeline recording queries and returning fixture evidence."""

    def __init__(self) -> None:
        self.last_query: RetrievalQuery | None = None

    def retrieve(self, query: RetrievalQuery) -> RetrievalResponse:
        self.last_query = query
        ev1 = Evidence(
            record_id="rec_1",
            chunk_id="chunk_1",
            text="AR(1) model equation is X_t = phi * X_{t-1} + epsilon_t.",
            source_file="Lecture_2.pdf",
            source_pages=[14, 15],
            score=0.92,
            metadata={"course_id": "TSA_101"},
        )
        ev2 = Evidence(
            record_id="rec_2",
            chunk_id="chunk_2",
            text="Stationarity condition requires |phi| < 1.",
            source_file="Lecture_2.pdf",
            source_pages=[16],
            score=0.88,
            metadata={"course_id": "TSA_101"},
        )
        return RetrievalResponse(
            query=query,
            evidence=[ev1, ev2],
            diagnostics=RetrievalDiagnostics(retrieval_strategy="hybrid_mock"),
        )


class TestAcademicSearchTool:
    """Test suite validating AcademicSearchTool contract, parameter propagation, and formatting."""

    def test_protocol_conformance_and_metadata(self):
        """AcademicSearchTool satisfies ToolProtocol with read-only idempotent policy."""
        pipeline = MockRetrievalPipeline()
        tool = AcademicSearchTool(pipeline=pipeline)

        assert isinstance(tool, ToolProtocol)
        assert tool.name == "academic_search"
        assert tool.side_effect_policy.read_only is True
        assert tool.side_effect_policy.idempotent is True
        assert tool.side_effect_policy.mutating is False
        assert tool.side_effect_policy.timeout_seconds == 15.0

        schema = tool.get_schema()
        assert schema["type"] == "object"
        assert "query" in schema["required"]
        assert "top_k" in schema["properties"]

    def test_successful_search_execution(self):
        """Tool executes retrieval pipeline and formats clean JSON-serializable candidates with citations."""
        pipeline = MockRetrievalPipeline()
        tool = AcademicSearchTool(pipeline=pipeline)

        result = tool.execute({"query": "AR(1) stationarity", "top_k": 2})

        assert result.success is True
        assert result.tool_name == "academic_search"
        assert result.output["query"] == "AR(1) stationarity"
        assert result.output["total_found"] == 2

        candidates = result.output["candidates"]
        assert len(candidates) == 2
        assert candidates[0]["chunk_id"] == "chunk_1"
        assert candidates[0]["score"] == 0.92
        assert list(candidates[0]["source_pages"]) == [14, 15]
        assert "Lecture 2" in candidates[0]["citation"]

        # Verify pipeline received query
        assert pipeline.last_query is not None
        assert pipeline.last_query.text == "AR(1) stationarity"
        assert pipeline.last_query.top_k == 2

    def test_search_with_course_and_lecture_filters(self):
        """Optional course_id and lecture_id parameters are passed to pipeline filters."""
        pipeline = MockRetrievalPipeline()
        tool = AcademicSearchTool(pipeline=pipeline)

        result = tool.execute({
            "query": "autocorrelation",
            "course_id": "TSA_101",
            "lecture_id": "Lecture_3.pdf",
        })

        assert result.success is True
        assert pipeline.last_query.filters.custom_filters["course_id"] == "TSA_101"
        assert pipeline.last_query.filters.source_file == "Lecture_3.pdf"

    def test_schema_validation_rejection(self):
        """Empty query text fails schema validation with VALIDATION_ERROR."""
        pipeline = MockRetrievalPipeline()
        tool = AcademicSearchTool(pipeline=pipeline)

        result = tool.execute({"query": ""})
        assert result.success is False
        assert result.error_category == ToolErrorCategory.VALIDATION_ERROR
        assert "Validation failed" in result.error_message
