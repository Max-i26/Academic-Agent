"""Validation rules, safety checks, and domain exceptions for embeddings."""

import logging
import numpy as np

logger = logging.getLogger(__name__)


class EmbeddingError(Exception):
    """Base exception for all embedding-related errors."""
    pass


class TokenLimitExceededError(EmbeddingError):
    """Raised when an input sequence exceeds the maximum sequence length of the model."""

    def __init__(self, token_count: int, max_tokens: int, chunk_id: str | None = None):
        self.token_count = token_count
        self.max_tokens = max_tokens
        self.chunk_id = chunk_id
        chunk_info = f" in chunk '{chunk_id}'" if chunk_id else ""
        super().__init__(
            f"Input sequence{chunk_info} contains {token_count} tokens, exceeding the "
            f"model limit of {max_tokens} tokens without truncation."
        )


class DimensionMismatchError(EmbeddingError):
    """Raised when output vectors do not match the expected dimensionality."""

    def __init__(self, actual_dim: int, expected_dim: int):
        self.actual_dim = actual_dim
        self.expected_dim = expected_dim
        super().__init__(
            f"Embedding vector dimension {actual_dim} does not match expected dimension {expected_dim}."
        )


class InvalidEmbeddingValueError(EmbeddingError):
    """Raised when embedding vectors contain NaN or Inf values."""
    pass


class EmbeddingValidator:
    """Runtime validator for token limits, vector shapes, and numerical integrity."""

    @staticmethod
    def validate_token_count(
        token_count: int,
        max_tokens: int,
        chunk_id: str | None = None,
        strict: bool = True,
    ) -> None:
        """Verify that token count does not exceed model maximum.

        Args:
            token_count: Exact token count of the text.
            max_tokens: Model maximum sequence length limit.
            chunk_id: Identifier of the chunk being evaluated.
            strict: If True, raises TokenLimitExceededError. If False, logs warning.
        """
        if token_count > max_tokens:
            if strict:
                raise TokenLimitExceededError(token_count, max_tokens, chunk_id)
            logger.warning(
                "Chunk '%s' token count (%d) exceeds max sequence length (%d); "
                "content will be truncated by provider.",
                chunk_id or "unnamed",
                token_count,
                max_tokens,
            )

    @staticmethod
    def validate_vectors(
        vectors: np.ndarray,
        expected_dimension: int,
        expected_count: int | None = None,
    ) -> None:
        """Validate shape, dimensionality, and numeric integrity of output vectors."""
        if not isinstance(vectors, np.ndarray):
            raise InvalidEmbeddingValueError(f"Expected np.ndarray, got {type(vectors).__name__}")

        if vectors.ndim != 2:
            raise DimensionMismatchError(
                actual_dim=vectors.ndim,
                expected_dim=2,
            )

        if vectors.shape[1] != expected_dimension:
            raise DimensionMismatchError(
                actual_dim=vectors.shape[1],
                expected_dim=expected_dimension,
            )

        if expected_count is not None and vectors.shape[0] != expected_count:
            raise EmbeddingError(
                f"Vector count mismatch: got {vectors.shape[0]}, expected {expected_count}."
            )

        if not np.all(np.isfinite(vectors)):
            nan_count = int(np.isnan(vectors).sum())
            inf_count = int(np.isinf(vectors).sum())
            raise InvalidEmbeddingValueError(
                f"Embedding vectors contain non-finite numbers: {nan_count} NaNs, {inf_count} Infs."
            )

    @staticmethod
    def validate_normalized(vectors: np.ndarray, tolerance: float = 1e-2) -> None:
        """Verify that vectors are unit L2-normalized within numerical tolerance."""
        norms = np.linalg.norm(vectors, axis=1)
        deviations = np.abs(norms - 1.0)
        max_deviation = float(np.max(deviations))
        if max_deviation > tolerance:
            raise InvalidEmbeddingValueError(
                f"Embedding vectors are not unit normalized. Maximum norm deviation: {max_deviation:.4f}"
            )
