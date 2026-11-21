"""Unit tests for EmbeddingConfig and options."""

import pytest
from academic_agent.embeddings.config import (
    EmbeddingConfig,
    DeviceType,
    KnownEmbeddingModel,
)


def test_default_config():
    config = EmbeddingConfig()
    assert config.model_name == KnownEmbeddingModel.MINILM.value
    assert config.device == DeviceType.AUTO
    assert config.batch_size == 32
    assert config.normalize_embeddings is True
    assert config.strict_token_limit is True
    assert config.expected_dimension == 384


def test_custom_config():
    config = EmbeddingConfig(
        model_name=KnownEmbeddingModel.BGE_SMALL.value,
        device=DeviceType.CPU,
        batch_size=64,
        strict_token_limit=False,
    )
    assert config.model_name == "BAAI/bge-small-en-v1.5"
    assert config.device == DeviceType.CPU
    assert config.batch_size == 64
    assert config.strict_token_limit is False


def test_invalid_batch_size():
    with pytest.raises(ValueError):
        EmbeddingConfig(batch_size=0)
