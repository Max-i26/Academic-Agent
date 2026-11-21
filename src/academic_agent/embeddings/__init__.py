"""Academic-Agent Embedding Subsystem (Phase 3)."""

from academic_agent.embeddings.protocols import EmbeddingProvider
from academic_agent.embeddings.config import (
    EmbeddingConfig,
    DeviceType,
    KnownEmbeddingModel,
)
from academic_agent.embeddings.models import (
    EmbeddingMetadata,
    DocumentEmbeddingRecord,
    QueryEmbeddingRecord,
    BatchEmbeddingResult,
)
from academic_agent.embeddings.validator import (
    EmbeddingValidator,
    EmbeddingError,
    TokenLimitExceededError,
    DimensionMismatchError,
    InvalidEmbeddingValueError,
)
from academic_agent.embeddings.tokenizer import (
    TokenCounter,
    HuggingFaceTokenCounter,
    WhitespaceTokenCounter,
)
from academic_agent.embeddings.providers.minilm import MiniLMProvider
from academic_agent.embeddings.providers.bge import BGEProvider
from academic_agent.embeddings.engine import EmbeddingEngine

__all__ = [
    "EmbeddingEngine",
    "EmbeddingConfig",
    "DeviceType",
    "KnownEmbeddingModel",
    "EmbeddingProvider",
    "EmbeddingMetadata",
    "DocumentEmbeddingRecord",
    "QueryEmbeddingRecord",
    "BatchEmbeddingResult",
    "EmbeddingValidator",
    "EmbeddingError",
    "TokenLimitExceededError",
    "DimensionMismatchError",
    "InvalidEmbeddingValueError",
    "TokenCounter",
    "HuggingFaceTokenCounter",
    "WhitespaceTokenCounter",
    "MiniLMProvider",
    "BGEProvider",
]
