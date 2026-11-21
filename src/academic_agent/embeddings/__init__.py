"""Academic-Agent Embedding Subsystem (Phase 3)."""

from typing import Any

from academic_agent.embeddings.config import (
    DeviceType,
    EmbeddingConfig,
    KnownEmbeddingModel,
)
from academic_agent.embeddings.models import (
    BatchEmbeddingResult,
    DocumentEmbeddingRecord,
    EmbeddingMetadata,
    QueryEmbeddingRecord,
)
from academic_agent.embeddings.protocols import EmbeddingProvider
from academic_agent.embeddings.tokenizer import (
    TokenCounter,
    WhitespaceTokenCounter,
)
from academic_agent.embeddings.validator import (
    DimensionMismatchError,
    EmbeddingError,
    EmbeddingValidator,
    InvalidEmbeddingValueError,
    TokenLimitExceededError,
)


def __getattr__(name: str) -> Any:
    """PEP 562 lazy import of heavyweight or hardware-dependent embedding providers."""
    if name == "MiniLMProvider":
        from academic_agent.embeddings.providers.minilm import MiniLMProvider
        return MiniLMProvider
    if name == "BGEProvider":
        from academic_agent.embeddings.providers.bge import BGEProvider
        return BGEProvider
    if name == "EmbeddingEngine":
        from academic_agent.embeddings.engine import EmbeddingEngine
        return EmbeddingEngine
    if name == "HuggingFaceTokenCounter":
        from academic_agent.embeddings.tokenizer import HuggingFaceTokenCounter
        return HuggingFaceTokenCounter
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


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
