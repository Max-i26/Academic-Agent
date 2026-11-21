"""Data models for retrieval queries, candidates, academic evidence, and responses."""

from enum import Enum
import re
from typing import Any
from pydantic import BaseModel, Field, field_validator, model_validator

from academic_agent.storage.models import FilterQuery


class RetrievalStrategy(str, Enum):
    """Available retrieval execution strategies."""

    DENSE = "dense"
    SPARSE = "sparse"
    HYBRID = "hybrid"
    HYBRID_RERANK = "hybrid_rerank"


class RetrievalQuery(BaseModel):
    """Structured student or agent query for academic evidence retrieval."""

    text: str = Field(description="Raw query question string.")
    top_k: int = Field(default=5, ge=1, description="Number of final evidence passages requested.")
    filters: FilterQuery | None = Field(default=None, description="Optional metadata filter constraints.")
    strategy: RetrievalStrategy = Field(
        default=RetrievalStrategy.HYBRID,
        description="Retrieval strategy: 'dense', 'sparse', 'hybrid', or 'hybrid_rerank'.",
    )
    candidate_pool_size: int = Field(
        default=20,
        ge=1,
        description="Number of candidate records fetched by each component retriever before fusion/reranking.",
    )
    metadata: dict[str, Any] = Field(default_factory=dict, description="Contextual query tags or parameters.")

    @field_validator("text")
    @classmethod
    def validate_non_empty_text(cls, v: str) -> str:
        cleaned = v.strip()
        if not cleaned:
            raise ValueError("Query text cannot be empty or pure whitespace.")
        return cleaned

    @model_validator(mode="after")
    def validate_pool_size_gte_top_k(self) -> "RetrievalQuery":
        if self.candidate_pool_size < self.top_k:
            self.candidate_pool_size = self.top_k
        return self

    @property
    def normalized_text(self) -> str:
        """Lowercase stripped query text."""
        return self.text.lower().strip()


class RetrievalCandidate(BaseModel):
    """Intermediate candidate passage produced by a single retrieval mechanism."""

    record_id: str = Field(description="Unique primary key in storage (chunk_id or chunk_id#frag_idx).")
    chunk_id: str = Field(description="Canonical parent Phase 2 ChunkRecord ID.")
    rank: int = Field(ge=1, description="1-based ranking within this retrieval method's output.")
    retrieval_method: str = Field(
        description="Retrieval method that generated this candidate (e.g. 'dense', 'sparse', 'hybrid', 'rerank')."
    )
    score: float = Field(
        description="Relevance or similarity score (e.g. cosine similarity, BM25 score, RRF score)."
    )
    normalized_score: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Optional calibrated score scaled to [0.0, 1.0].",
    )
    text: str = Field(default="", description="Verbatim academic text.")
    source_file: str = Field(default="", description="Source lecture PDF filename.")
    source_pages: list[int] = Field(default_factory=list, description="Associated lecture page numbers.")
    source_unit_ids: list[str] = Field(default_factory=list, description="Associated source layout unit IDs.")
    section_title: str | None = Field(default=None, description="Slide or section heading.")
    content_type: str = Field(default="PARAGRAPH", description="Content classification.")
    fragment_index: int = Field(default=0, ge=0, description="0-based sequence index.")
    total_fragments: int = Field(default=1, ge=1, description="Total fragments produced for parent chunk.")
    contributing_methods: list[str] = Field(
        default_factory=list,
        description="All retrieval methods that surfaced this candidate (e.g. ['dense', 'sparse']).",
    )
    method_scores: dict[str, float] = Field(
        default_factory=dict,
        description="Original raw scores from each contributing retrieval method.",
    )
    method_ranks: dict[str, int] = Field(
        default_factory=dict,
        description="Original 1-based ranks within each contributing retrieval method.",
    )
    dense_score: float | None = Field(default=None, description="Raw dense cosine similarity score.")
    sparse_score: float | None = Field(default=None, description="Raw sparse BM25 score.")
    rrf_score: float | None = Field(default=None, description="Reciprocal Rank Fusion score.")
    rerank_score: float | None = Field(default=None, description="Cross-encoder reranking relevance score.")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Record and retrieval diagnostics metadata.")

    @property
    def is_fragment(self) -> bool:
        """True if candidate represents a decomposed fragment of an oversized chunk."""
        return self.total_fragments > 1

    def to_evidence(self) -> "Evidence":
        """Convert this candidate directly into an Evidence object."""
        methods = list(self.contributing_methods) if self.contributing_methods else [self.retrieval_method]
        scores = dict(self.method_scores) if self.method_scores else {self.retrieval_method: self.score}
        return Evidence(
            record_id=self.record_id,
            chunk_id=self.chunk_id,
            text=self.text,
            source_file=self.source_file,
            source_pages=self.source_pages,
            source_unit_ids=list(self.source_unit_ids),
            section_title=self.section_title,
            content_type=self.content_type,
            fragment_index=self.fragment_index,
            total_fragments=self.total_fragments,
            score=self.score,
            dense_score=self.dense_score or self.method_scores.get("dense"),
            sparse_score=self.sparse_score or self.method_scores.get("sparse"),
            rrf_score=self.rrf_score or self.method_scores.get("rrf"),
            rerank_score=self.rerank_score or self.method_scores.get("rerank"),
            retrieval_methods=methods,
            method_scores=scores,
            metadata=dict(self.metadata),
        )


class Evidence(BaseModel):
    """Grounded academic evidence passage formatted for attribution, citation, and LLM context."""

    record_id: str = Field(description="Unique storage record ID.")
    chunk_id: str = Field(description="Parent Phase 2 ChunkRecord ID.")
    text: str = Field(description="Evidence text to ground the academic answer.")
    source_file: str = Field(description="Source lecture PDF filename.")
    source_pages: list[int] = Field(description="Contributing page numbers.")
    source_unit_ids: list[str] = Field(default_factory=list, description="Associated source layout unit IDs.")
    section_title: str | None = Field(default=None, description="Section heading context.")
    content_type: str = Field(default="PARAGRAPH", description="Content classification.")
    fragment_index: int = Field(default=0, ge=0, description="0-based fragment index.")
    total_fragments: int = Field(default=1, ge=1, description="Total fragments for parent chunk.")
    score: float = Field(description="Final unified or reranked relevance score.")
    dense_score: float | None = Field(default=None, description="Raw dense cosine similarity score.")
    sparse_score: float | None = Field(default=None, description="Raw sparse BM25 score.")
    rrf_score: float | None = Field(default=None, description="Reciprocal Rank Fusion score.")
    rerank_score: float | None = Field(default=None, description="Cross-encoder reranking relevance score.")
    retrieval_methods: list[str] = Field(
        default_factory=list,
        description="All retrieval methods that surfaced this candidate (e.g. ['dense', 'sparse']).",
    )
    method_scores: dict[str, float] = Field(
        default_factory=dict,
        description="Individual scores from contributing retrieval methods (e.g. {'dense': 0.85, 'sparse': 12.3}).",
    )
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional provenance attributes.")

    @property
    def is_fragment(self) -> bool:
        """True if evidence represents a decomposed fragment of an oversized chunk."""
        return self.total_fragments > 1

    def format_citation(self) -> str:
        """Generate a clean, standardized academic citation label.

        Examples:
            '[Lecture 4, p. 18]'
            '[Lecture 5, pp. 12-14]'
            '[Lecture 7, pp. 4, 7]'
        """
        # Extract lecture number if present, otherwise use source_file basename
        match = re.search(r"Lecture[\s_]*(\d+)", self.source_file, re.IGNORECASE)
        if match:
            doc_label = f"Lecture {match.group(1)}"
        else:
            doc_label = self.source_file.split("/")[-1].split("\\")[-1]

        # Format page numbers
        sorted_pages = sorted(set(self.source_pages))
        if not sorted_pages:
            pages_label = ""
        elif len(sorted_pages) == 1:
            pages_label = f", p. {sorted_pages[0]}"
        else:
            # Check for contiguous range
            if sorted_pages[-1] - sorted_pages[0] == len(sorted_pages) - 1:
                pages_label = f", pp. {sorted_pages[0]}-{sorted_pages[-1]}"
            else:
                pages_label = f", pp. {', '.join(str(p) for p in sorted_pages)}"

        return f"[{doc_label}{pages_label}]"

    def to_prompt_context(self, include_score: bool = False) -> str:
        """Format this evidence item into a markdown block for LLM prompt context injection."""
        citation = self.format_citation()
        header = f"### {citation}"
        if self.section_title:
            header += f" — {self.section_title}"

        details = []
        if self.content_type:
            details.append(f"Type: {self.content_type}")
        if self.is_fragment:
            details.append(f"Part {self.fragment_index + 1}/{self.total_fragments}")
        if include_score:
            details.append(f"Relevance: {self.score:.4f}")

        details_str = f" *({', '.join(details)})*" if details else ""

        return f"{header}{details_str}\n{self.text.strip()}"


class RetrievalDiagnostics(BaseModel):
    """Runtime execution metrics, latency diagnostics, and candidate pool telemetry."""

    dense_latency_ms: float | None = Field(default=None, description="Dense retrieval latency in milliseconds.")
    sparse_latency_ms: float | None = Field(default=None, description="Sparse retrieval latency in milliseconds.")
    fusion_latency_ms: float | None = Field(default=None, description="Score fusion latency in milliseconds.")
    rerank_latency_ms: float | None = Field(default=None, description="Reranker latency in milliseconds.")
    total_latency_ms: float = Field(default=0.0, description="Total pipeline latency in milliseconds.")
    candidate_counts_by_method: dict[str, int] = Field(
        default_factory=dict,
        description="Number of candidates retrieved by each method.",
    )
    method_agreement_count: int = Field(
        default=0,
        description="Number of unique candidates returned by both dense and sparse retrieval.",
    )
    additional_info: dict[str, Any] = Field(default_factory=dict, description="Custom debugging data.")


class RetrievalResponse(BaseModel):
    """Complete retrieval result containing final evidence passages, diagnostics, and telemetry."""

    query: RetrievalQuery = Field(description="The input query that triggered this retrieval.")
    evidence: list[Evidence] = Field(
        default_factory=list,
        description="Final ranked, deduplicated evidence passages ready for answer generation.",
    )
    candidates: list[RetrievalCandidate] = Field(
        default_factory=list,
        description="Raw intermediate candidate pool across all retrieval branches.",
    )
    diagnostics: RetrievalDiagnostics = Field(
        default_factory=RetrievalDiagnostics,
        description="Execution timing and candidate telemetry.",
    )
    strategy_used: RetrievalStrategy = Field(
        default=RetrievalStrategy.HYBRID,
        description="Strategy actually executed.",
    )

    @property
    def total_evidence(self) -> int:
        """Count of returned evidence items."""
        return len(self.evidence)

    @property
    def source_files_represented(self) -> list[str]:
        """Unique list of lecture source files contributing to the retrieved evidence."""
        seen = []
        for e in self.evidence:
            if e.source_file not in seen:
                seen.append(e.source_file)
        return seen

    def to_llm_context(self, max_items: int | None = None) -> str:
        """Combine all evidence blocks into a structured markdown context string."""
        items = self.evidence if max_items is None else self.evidence[:max_items]
        if not items:
            return "No relevant academic evidence found."

        blocks = [e.to_prompt_context() for e in items]
        return "\n\n---\n\n".join(blocks)
