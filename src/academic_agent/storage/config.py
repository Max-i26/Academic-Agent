"""Configuration schema for vector stores."""

from pydantic import BaseModel, Field


class VectorStoreConfig(BaseModel):
    """Configuration for vector store backend and embedding compatibility."""

    backend: str = Field(
        default="memory",
        description="Vector store backend implementation: 'memory', 'faiss', 'chroma', or 'lance'.",
    )
    collection_name: str = Field(
        default="academic_agent_ds5104",
        description="Name of the collection or table.",
    )
    persist_path: str | None = Field(
        default=None,
        description="Local directory for on-disk persistence. None indicates purely in-memory.",
    )
    expected_model_name: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        description="Target embedding model name for strict space isolation.",
    )
    expected_dimension: int = Field(
        default=384,
        description="Expected vector dimension.",
    )
    distance_metric: str = Field(
        default="cosine",
        description="Distance metric for similarity ranking ('cosine', 'l2', or 'ip').",
    )
    allow_overwrite: bool = Field(
        default=True,
        description="Whether inserting an existing record_id updates/replaces the previous record.",
    )
