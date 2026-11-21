"""Data models and schemas for retrieval benchmarking and evaluation metrics."""

from datetime import datetime, timezone
from enum import Enum
from typing import Any
from pydantic import BaseModel, Field, field_validator


class QueryCategory(str, Enum):
    """Categorization of benchmark questions for stratified retrieval evaluation."""

    FORMULAS = "formulas_and_notation"
    DEFINITIONS = "definitions_and_terminology"
    EXPLANATIONS = "conceptual_explanations"
    CALCULATIONS = "calculations_and_workspaces"
    CROSS_LECTURE = "cross_lecture_and_multi_page"
    AMBIGUOUS_VARIANTS = "ambiguous_and_variants"


class BenchmarkSourceReference(BaseModel):
    """Document citation provenance for expected evidence."""

    source_file: str = Field(description="PDF filename.")
    pages: list[int] = Field(default_factory=list, description="Target page numbers.")
    section_title: str | None = Field(default=None, description="Section or slide header.")


class BenchmarkQuery(BaseModel):
    """Single benchmark evaluation query with graded relevance ground truth."""

    query_id: str = Field(description="Unique query identifier (e.g. q01).")
    query: str = Field(description="Natural-language question string.")
    category: QueryCategory = Field(description="Category of question.")
    difficulty: str = Field(default="medium", description="Difficulty: easy, medium, or hard.")
    relevant_records: dict[str, int] = Field(
        description="Mapping from record_id (or chunk_id) to graded relevance (1=partial, 2=relevant, 3=exact)."
    )
    canonical_chunk_ids: list[str] = Field(
        default_factory=list,
        description="Canonical parent chunk IDs for fragment-tolerant evaluation.",
    )
    expected_sources: list[BenchmarkSourceReference] = Field(
        default_factory=list,
        description="Expected document provenance.",
    )
    notes: str = Field(default="", description="Domain notes, nuances, or acceptable alternate evidence.")

    @field_validator("relevant_records")
    @classmethod
    def validate_relevance_grades(cls, v: dict[str, int]) -> dict[str, int]:
        if not v:
            raise ValueError("BenchmarkQuery must have at least one relevant record.")
        for rid, grade in v.items():
            if not isinstance(grade, int) or grade < 1 or grade > 3:
                raise ValueError(f"Relevance grade for '{rid}' must be an integer between 1 and 3, got {grade}")
        return v


class QueryEvaluationResult(BaseModel):
    """Evaluation metrics for a single query under a specific retrieval strategy."""

    query_id: str
    query: str
    category: str
    strategy: str
    ranked_records: list[str] = Field(default_factory=list)
    hit_at_1: float
    hit_at_3: float
    hit_at_5: float
    mrr: float
    map_score: float
    recall_at_5: float
    ndcg_at_5: float
    latency_ms: float
    first_relevant_rank: int | None = None
    target_found_by_methods: list[str] = Field(default_factory=list)
    error_message: str | None = None


class EvaluationSummary(BaseModel):
    """Aggregate metrics and diagnostic report across an entire benchmark dataset."""

    dataset_name: str
    dataset_version: str
    strategy: str
    total_queries: int
    evaluated_queries: int
    failed_queries_count: int
    hit_at_1: float
    hit_at_3: float
    hit_at_5: float
    mrr: float
    map_score: float
    recall_at_5: float
    ndcg_at_5: float
    avg_latency_ms: float
    category_metrics: dict[str, dict[str, float]] = Field(default_factory=dict)
    per_query_results: list[QueryEvaluationResult] = Field(default_factory=list)
    failed_queries: list[dict[str, Any]] = Field(default_factory=list)
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
