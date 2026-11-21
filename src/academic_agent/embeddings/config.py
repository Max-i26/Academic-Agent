"""Configuration models for the Embedding Engine."""

from enum import Enum
from pydantic import BaseModel, Field, field_validator


class DeviceType(str, Enum):
    """Device target for tensor computation."""
    AUTO = "auto"
    CUDA = "cuda"
    CPU = "cpu"


class KnownEmbeddingModel(str, Enum):
    """Supported first-class embedding models."""
    MINILM = "sentence-transformers/all-MiniLM-L6-v2"
    BGE_SMALL = "BAAI/bge-small-en-v1.5"


class EmbeddingConfig(BaseModel):
    """Configuration settings for the embedding subsystem."""

    model_name: str = Field(
        default=KnownEmbeddingModel.MINILM.value,
        description="HuggingFace model ID or known embedding model alias.",
    )
    device: DeviceType = Field(
        default=DeviceType.AUTO,
        description="Target compute device: 'auto' (CUDA if available, else CPU), 'cuda', or 'cpu'.",
    )
    batch_size: int = Field(
        default=32,
        ge=1,
        le=512,
        description="Inference batch size for document encoding.",
    )
    normalize_embeddings: bool = Field(
        default=True,
        description="Whether to L2-normalize output embedding vectors for cosine similarity.",
    )
    strict_token_limit: bool = Field(
        default=True,
        description="When True, raises TokenLimitExceededError if text exceeds max_sequence_length. "
                    "When False, issues a warning but allows provider truncation.",
    )
    expected_dimension: int = Field(
        default=384,
        ge=64,
        le=4096,
        description="Expected vector dimensionality for runtime shape validation.",
    )

    @field_validator("batch_size")
    @classmethod
    def validate_batch_size(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("batch_size must be positive")
        return v
