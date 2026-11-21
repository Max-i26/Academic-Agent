"""Integration tests for MiniLMProvider and BGEProvider."""

import numpy as np
import pytest
import torch

from academic_agent.embeddings.config import EmbeddingConfig, KnownEmbeddingModel
from academic_agent.embeddings.engine import EmbeddingEngine
from academic_agent.embeddings.providers.minilm import MiniLMProvider
from academic_agent.embeddings.providers.bge import BGEProvider


@pytest.fixture(scope="module")
def minilm_provider():
    # Use CPU for deterministic, low-overhead testing
    return MiniLMProvider(device="cpu")


@pytest.fixture(scope="module")
def bge_provider():
    return BGEProvider(device="cpu")


def test_minilm_provider_properties(minilm_provider):
    assert minilm_provider.dimension == 384
    assert minilm_provider.max_sequence_length == 256
    assert minilm_provider.device == "cpu"


def test_minilm_token_counting(minilm_provider):
    text = "Exponential smoothing is a forecasting method."
    tokens = minilm_provider.count_tokens(text)
    # [CLS], Exponential, smoothing, is, a, forecasting, method, ., [SEP]
    assert tokens >= 8


def test_minilm_encode_shapes_and_norm(minilm_provider):
    queries = ["What is MAE?", "How to difference a series?"]
    vecs = minilm_provider.encode_queries(queries, normalize=True)
    assert vecs.shape == (2, 384)
    assert vecs.dtype == np.float32

    # Check L2 unit normalization
    norms = np.linalg.norm(vecs, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-3)


def test_bge_provider_properties(bge_provider):
    assert bge_provider.dimension == 384
    assert bge_provider.max_sequence_length == 512
    assert bge_provider.device == "cpu"
    assert "Represent this sentence" in bge_provider.query_instruction


def test_bge_asymmetric_encoding(bge_provider):
    text = "What is stationarity?"
    # Encode as query (with instruction prefix)
    q_vec = bge_provider.encode_queries([text], normalize=True)
    # Encode as document (without instruction prefix)
    d_vec = bge_provider.encode_documents([text], normalize=True)

    assert q_vec.shape == (1, 384)
    assert d_vec.shape == (1, 384)
    # Because of the query instruction prefix, q_vec and d_vec should be close but not identical
    similarity = float(np.dot(q_vec[0], d_vec[0]))
    assert similarity < 0.99, "Asymmetric instruction must produce distinct representation"


def test_engine_factory_from_config():
    cfg_minilm = EmbeddingConfig(model_name="all-MiniLM-L6-v2", device="cpu")
    engine_minilm = EmbeddingEngine.from_config(cfg_minilm)
    assert isinstance(engine_minilm.provider, MiniLMProvider)

    cfg_bge = EmbeddingConfig(model_name="bge-small-en-v1.5", device="cpu")
    engine_bge = EmbeddingEngine.from_config(cfg_bge)
    assert isinstance(engine_bge.provider, BGEProvider)
