"""Typed data models and records for the Embedding Engine."""

from datetime import datetime, timezone
from typing import Any
import numpy as np
from pydantic import BaseModel, Field


class EmbeddingMetadata(BaseModel):
    """Metadata recorded with every generated embedding for auditability and provenance."""

    model_name: str = Field(description="Name or identifier of the embedding model.")
    dimension: int = Field(description="Vector dimensionality of the embedding.")
    max_sequence_length: int = Field(description="Maximum token sequence length of the model.")
    normalized: bool = Field(default=True, description="Whether the vector is unit L2-normalized.")
    device_used: str = Field(description="Device where inference was computed ('cuda' or 'cpu').")
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="ISO 8601 UTC timestamp of creation.",
    )


class DocumentEmbeddingRecord(BaseModel):
    """Embedding representation of a chunk or document ready for retrieval indexing."""

    chunk_id: str = Field(description="Unique, deterministic chunk identifier.")
    embedding: list[float] = Field(description="Dense floating-point embedding vector.")
    token_count: int = Field(description="Exact sub-token count according to the model tokenizer.")
    metadata: EmbeddingMetadata = Field(description="Provenance and model configuration metadata.")

    def as_numpy(self) -> np.ndarray:
        """Return the embedding as a float32 1D numpy array."""
        return np.array(self.embedding, dtype=np.float32)


class QueryEmbeddingRecord(BaseModel):
    """Embedding representation of a user search query."""

    query_text: str = Field(description="Raw query text provided by the user or agent.")
    embedding: list[float] = Field(description="Dense floating-point embedding vector.")
    token_count: int = Field(description="Exact token count of the query.")
    metadata: EmbeddingMetadata = Field(description="Provenance and model configuration metadata.")

    def as_numpy(self) -> np.ndarray:
        """Return the embedding as a float32 1D numpy array."""
        return np.array(self.embedding, dtype=np.float32)


class BatchEmbeddingResult(BaseModel):
    """Container for the output of a batch chunk embedding job."""

    records: list[DocumentEmbeddingRecord] = Field(default_factory=list)
    total_chunks: int = Field(description="Number of chunks embedded.")
    elapsed_seconds: float = Field(description="Total batch processing latency in seconds.")
    chunks_per_second: float = Field(description="Processing throughput.")
    device_used: str = Field(description="Device where inference took place.")

    def to_numpy_matrix(self) -> np.ndarray:
        """Stack all chunk embeddings into a 2D numpy array of shape (N, dimension)."""
        if not self.records:
            return np.empty((0, 0), dtype=np.float32)
        return np.vstack([rec.as_numpy() for rec in self.records])
