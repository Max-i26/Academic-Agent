"""Embedding provider protocol and interface contracts."""

from typing import Protocol, runtime_checkable
import numpy as np


@runtime_checkable
class EmbeddingProvider(Protocol):
    """Contract for pluggable embedding model providers.

    All model-specific logic (e.g. query prefixes for asymmetric models like BGE,
    symmetric encoding for MiniLM, device placement, tokenizer calls)
    must reside inside the provider implementation, keeping the EmbeddingEngine
    strictly model-agnostic.
    """

    @property
    def name(self) -> str:
        """Name or HuggingFace repo identifier of the model."""
        ...

    @property
    def dimension(self) -> int:
        """Output embedding vector dimension (e.g., 384)."""
        ...

    @property
    def max_sequence_length(self) -> int:
        """Maximum allowable sequence length in tokens (e.g., 256 or 512)."""
        ...

    @property
    def device(self) -> str:
        """Currently active device ('cuda' or 'cpu')."""
        ...

    def count_tokens(self, text: str) -> int:
        """Count the exact number of sub-tokens in text using the provider's tokenizer."""
        ...

    def encode_queries(
        self,
        queries: list[str],
        batch_size: int = 32,
        normalize: bool = True,
    ) -> np.ndarray:
        """Encode search queries into normalized embedding vectors.

        For asymmetric models (e.g. BGE), this method applies the appropriate
        query prefix/instruction. For symmetric models, it encodes directly.
        Returns:
            np.ndarray of shape (len(queries), dimension), dtype float32.
        """
        ...

    def encode_documents(
        self,
        documents: list[str],
        batch_size: int = 32,
        normalize: bool = True,
    ) -> np.ndarray:
        """Encode passage/chunk documents into embedding vectors without query prefixes.

        Returns:
            np.ndarray of shape (len(documents), dimension), dtype float32.
        """
        ...

    def to(self, device: str) -> None:
        """Transfer model to the specified device ('cuda' or 'cpu')."""
        ...
