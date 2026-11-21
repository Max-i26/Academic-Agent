"""Data models and record schemas for vector storage and retrieval."""

from datetime import datetime, timezone
from typing import Any
import numpy as np
from pydantic import BaseModel, Field, field_validator


class IndexMetadata(BaseModel):
    """Metadata defining index provenance and ensuring embedding-space isolation.

    Prevents disastrous accidental cross-model similarity comparisons (e.g., comparing
    MiniLM vectors with BGE vectors, even if both are 384-dimensional).
    """

    model_name: str = Field(description="Name or HF identifier of the embedding model.")
    model_revision: str | None = Field(default=None, description="Exact commit/revision hash of the model.")
    embedding_dimension: int = Field(description="Output vector dimensionality (e.g. 384).")
    distance_metric: str = Field(default="cosine", description="Distance metric: 'cosine', 'l2', or 'ip'.")
    normalized: bool = Field(default=True, description="Whether indexed vectors are unit L2 normalized.")
    total_records: int = Field(default=0, description="Number of active records in the index.")
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="ISO 8601 UTC creation timestamp.",
    )
    updated_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="ISO 8601 UTC last update timestamp.",
    )

    def is_compatible_with(self, other: "IndexMetadata") -> bool:
        """Verify that another index or query vector shares the exact same embedding space."""
        return (
            self.model_name == other.model_name
            and self.embedding_dimension == other.embedding_dimension
            and self.distance_metric == other.distance_metric
            and self.normalized == other.normalized
        )


class VectorRecord(BaseModel):
    """Canonical representation of an indexed embedding record in storage.

    Binds the dense vector directly to its textual payload and deep academic provenance.
    """

    record_id: str = Field(description="Unique primary key in the index (chunk_id or chunk_id#frag_idx).")
    chunk_id: str = Field(description="Parent Phase 2 ChunkRecord ID.")
    vector: list[float] = Field(description="Dense floating-point vector.")
    text: str = Field(description="Text content embedded and stored for retrieval evidence.")
    source_file: str = Field(description="Originating PDF filename.")
    source_pages: list[int] = Field(description="Sorted list of page numbers.")
    source_unit_ids: list[str] = Field(default_factory=list, description="Raw layout unit references.")
    section_title: str | None = Field(default=None, description="Section heading context.")
    content_type: str = Field(default="PARAGRAPH", description="Content classification.")
    fragment_index: int = Field(default=0, description="Fragment index (0 if unfragmented).")
    total_fragments: int = Field(default=1, description="Total fragments for parent chunk.")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Arbitrary searchable metadata.")
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="ISO 8601 UTC insertion timestamp.",
    )

    @field_validator("vector")
    @classmethod
    def validate_non_empty_vector(cls, v: list[float]) -> list[float]:
        if not v:
            raise ValueError("Vector cannot be empty")
        return v

    def as_numpy(self) -> np.ndarray:
        """Return the vector as a float32 numpy array."""
        return np.array(self.vector, dtype=np.float32)


class SearchResult(BaseModel):
    """Ranked evidence returned from vector similarity search to the Academic Agent."""

    record_id: str = Field(description="Primary key of the retrieved record.")
    chunk_id: str = Field(description="Parent Phase 2 ChunkRecord ID.")
    score: float = Field(description="Similarity score (e.g. cosine similarity in [-1.0, 1.0]).")
    text: str = Field(description="Evidence text to be provided to the LLM context.")
    source_file: str = Field(description="Originating PDF lecture.")
    source_pages: list[int] = Field(description="Page numbers for citation provenance.")
    section_title: str | None = Field(default=None, description="Section heading context.")
    content_type: str = Field(default="PARAGRAPH", description="Content classification.")
    fragment_index: int = Field(default=0, description="Fragment index.")
    total_fragments: int = Field(default=1, description="Total fragments.")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata attributes.")


class FilterQuery(BaseModel):
    """Structured query filter for metadata-scoped retrieval."""

    source_file: str | None = Field(default=None, description="Filter by specific PDF filename.")
    source_pages: list[int] | None = Field(default=None, description="Filter by overlapping pages.")
    section_title: str | None = Field(default=None, description="Filter by section title substring or match.")
    content_type: str | None = Field(default=None, description="Filter by content classification.")
    chunk_id: str | None = Field(default=None, description="Filter by parent chunk ID.")
    custom_filters: dict[str, Any] | None = Field(default=None, description="Key-value filters.")

    def matches(self, record: VectorRecord) -> bool:
        """Evaluate whether a VectorRecord satisfies this filter."""
        if self.source_file and record.source_file != self.source_file:
            return False
        if self.source_pages:
            if not set(self.source_pages).intersection(record.source_pages):
                return False
        if self.section_title:
            if not record.section_title or self.section_title.lower() not in record.section_title.lower():
                return False
        if self.content_type and record.content_type != self.content_type:
            return False
        if self.chunk_id and record.chunk_id != self.chunk_id:
            return False
        if self.custom_filters:
            for k, v in self.custom_filters.items():
                if record.metadata.get(k) != v:
                    return False
        return True
