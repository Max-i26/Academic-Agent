"""Tests for EmbeddingProvider protocol and EmbeddingValidator."""

import numpy as np
import pytest

from academic_agent.embeddings.protocols import EmbeddingProvider
from academic_agent.embeddings.validator import (
    EmbeddingValidator,
    TokenLimitExceededError,
    DimensionMismatchError,
    InvalidEmbeddingValueError,
)


class DummyMockProvider:
    """Minimal provider satisfying the EmbeddingProvider protocol."""

    @property
    def name(self) -> str:
        return "mock-provider"

    @property
    def dimension(self) -> int:
        return 384

    @property
    def max_sequence_length(self) -> int:
        return 256

    @property
    def device(self) -> str:
        return "cpu"

    def count_tokens(self, text: str) -> int:
        return len(text.split())

    def encode_queries(self, queries: list[str], batch_size: int = 32, normalize: bool = True) -> np.ndarray:
        vecs = np.ones((len(queries), 384), dtype=np.float32)
        if normalize:
            vecs = vecs / np.linalg.norm(vecs, axis=1, keepdims=True)
        return vecs

    def encode_documents(self, documents: list[str], batch_size: int = 32, normalize: bool = True) -> np.ndarray:
        vecs = np.ones((len(documents), 384), dtype=np.float32)
        if normalize:
            vecs = vecs / np.linalg.norm(vecs, axis=1, keepdims=True)
        return vecs

    def to(self, device: str) -> None:
        pass


def test_protocol_runtime_checkable():
    provider = DummyMockProvider()
    assert isinstance(provider, EmbeddingProvider)


def test_validator_token_limit_strict():
    validator = EmbeddingValidator()
    # Within limit: should not raise
    validator.validate_token_count(token_count=100, max_tokens=256, chunk_id="c1", strict=True)

    # Exceeding limit: raises TokenLimitExceededError
    with pytest.raises(TokenLimitExceededError) as exc_info:
        validator.validate_token_count(token_count=322, max_tokens=256, chunk_id="test_chunk", strict=True)

    assert "test_chunk" in str(exc_info.value)
    assert "322" in str(exc_info.value)
    assert "256" in str(exc_info.value)


def test_validator_token_limit_non_strict():
    validator = EmbeddingValidator()
    # In non-strict mode, does not raise
    validator.validate_token_count(token_count=300, max_tokens=256, chunk_id="c2", strict=False)


def test_validator_dimension_check():
    validator = EmbeddingValidator()
    valid_vec = np.ones((2, 384), dtype=np.float32)
    validator.validate_vectors(valid_vec, expected_dimension=384, expected_count=2)

    wrong_dim = np.ones((2, 512), dtype=np.float32)
    with pytest.raises(DimensionMismatchError):
        validator.validate_vectors(wrong_dim, expected_dimension=384)

    wrong_ndim = np.ones((2, 384, 1), dtype=np.float32)
    with pytest.raises(DimensionMismatchError):
        validator.validate_vectors(wrong_ndim, expected_dimension=384)


def test_validator_non_finite_rejection():
    validator = EmbeddingValidator()
    nan_vec = np.ones((2, 384), dtype=np.float32)
    nan_vec[0, 5] = np.nan
    with pytest.raises(InvalidEmbeddingValueError) as exc:
        validator.validate_vectors(nan_vec, expected_dimension=384)
    assert "NaNs" in str(exc.value)

    inf_vec = np.ones((2, 384), dtype=np.float32)
    inf_vec[1, 10] = np.inf
    with pytest.raises(InvalidEmbeddingValueError) as exc:
        validator.validate_vectors(inf_vec, expected_dimension=384)
    assert "Infs" in str(exc.value)


def test_validator_unit_normalization():
    validator = EmbeddingValidator()
    # Properly normalized vector
    vec = np.random.randn(3, 384).astype(np.float32)
    norm_vec = vec / np.linalg.norm(vec, axis=1, keepdims=True)
    validator.validate_normalized(norm_vec)

    # Unnormalized vector
    unnorm_vec = np.ones((3, 384), dtype=np.float32)
    with pytest.raises(InvalidEmbeddingValueError) as exc:
        validator.validate_normalized(unnorm_vec)
    assert "not unit normalized" in str(exc.value)
