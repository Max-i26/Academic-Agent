"""Unit tests for EmbeddingEngine using mock providers."""

import numpy as np
import pytest

from academic_agent.embeddings.config import EmbeddingConfig, DeviceType
from academic_agent.embeddings.engine import EmbeddingEngine
from academic_agent.embeddings.models import (
    DocumentEmbeddingRecord,
    QueryEmbeddingRecord,
    BatchEmbeddingResult,
)
from academic_agent.embeddings.validator import TokenLimitExceededError
from academic_agent.processing.chunking import ChunkRecord, ContentType


class MockProvider:
    """Configurable mock provider for fast offline unit tests."""

    def __init__(self, name: str = "mock-minilm", dim: int = 384, max_seq: int = 256, device: str = "cpu"):
        self._name = name
        self._dim = dim
        self._max_seq = max_seq
        self._device = device
        self.encoded_queries: list[str] = []
        self.encoded_documents: list[str] = []

    @property
    def name(self) -> str:
        return self._name

    @property
    def dimension(self) -> int:
        return self._dim

    @property
    def max_sequence_length(self) -> int:
        return self._max_seq

    @property
    def device(self) -> str:
        return self._device

    def count_tokens(self, text: str) -> int:
        # 1 word = 1 token for predictable testing
        return len(text.split())

    def encode_queries(self, queries: list[str], batch_size: int = 32, normalize: bool = True) -> np.ndarray:
        self.encoded_queries.extend(queries)
        vecs = np.random.randn(len(queries), self._dim).astype(np.float32)
        if normalize:
            vecs = vecs / np.linalg.norm(vecs, axis=1, keepdims=True)
        return vecs

    def encode_documents(self, documents: list[str], batch_size: int = 32, normalize: bool = True) -> np.ndarray:
        self.encoded_documents.extend(documents)
        vecs = np.random.randn(len(documents), self._dim).astype(np.float32)
        if normalize:
            vecs = vecs / np.linalg.norm(vecs, axis=1, keepdims=True)
        return vecs

    def to(self, device: str) -> None:
        self._device = device


def test_embed_query_success():
    provider = MockProvider()
    config = EmbeddingConfig(expected_dimension=384, normalize_embeddings=True)
    engine = EmbeddingEngine(provider=provider, config=config)

    res = engine.embed_query("What is exponential smoothing?")
    assert isinstance(res, QueryEmbeddingRecord)
    assert res.query_text == "What is exponential smoothing?"
    assert len(res.embedding) == 384
    assert res.token_count == 4
    assert res.metadata.model_name == "mock-minilm"
    assert res.metadata.dimension == 384
    assert res.metadata.normalized is True

    arr = res.as_numpy()
    assert isinstance(arr, np.ndarray)
    assert arr.shape == (384,)
    assert np.isclose(np.linalg.norm(arr), 1.0, atol=1e-3)


def test_embed_query_empty_raises():
    provider = MockProvider()
    engine = EmbeddingEngine(provider=provider)

    with pytest.raises(ValueError, match="cannot be empty"):
        engine.embed_query("   ")


def test_embed_query_oversized_raises():
    provider = MockProvider(max_seq=5)
    config = EmbeddingConfig(strict_token_limit=True)
    engine = EmbeddingEngine(provider=provider, config=config)

    # 6 words > 5 max tokens
    with pytest.raises(TokenLimitExceededError):
        engine.embed_query("one two three four five six")


def test_embed_documents_success():
    provider = MockProvider()
    config = EmbeddingConfig(expected_dimension=384)
    engine = EmbeddingEngine(provider=provider, config=config)

    docs = ["First document text", "Second document text here"]
    cids = ["chunk_01", "chunk_02"]
    records = engine.embed_documents(texts=docs, chunk_ids=cids)

    assert len(records) == 2
    assert records[0].chunk_id == "chunk_01"
    assert records[1].chunk_id == "chunk_02"
    assert len(records[0].embedding) == 384
    assert records[0].token_count == 3
    assert records[1].token_count == 4


def test_embed_documents_oversized_rejection():
    # Demonstrates NO SILENT TRUNCATION
    provider = MockProvider(max_seq=10)
    config = EmbeddingConfig(strict_token_limit=True)
    engine = EmbeddingEngine(provider=provider, config=config)

    # 12 words exceeds 10 tokens
    long_doc = "word " * 12
    with pytest.raises(TokenLimitExceededError) as exc_info:
        engine.embed_documents(texts=[long_doc], chunk_ids=["oversized_chunk"])

    assert "oversized_chunk" in str(exc_info.value)
    assert exc_info.value.token_count == 12
    assert exc_info.value.max_tokens == 10


def test_embed_chunks_from_phase2():
    provider = MockProvider()
    engine = EmbeddingEngine(provider=provider)

    # Create dummy ChunkRecord objects as produced by ChunkingEngine
    chunk1 = ChunkRecord(
        chunk_id="chk_001",
        document_id="doc_1",
        source_file="lecture1.pdf",
        chunk_index=0,
        text="Introduction to time series analysis",
        source_pages=[1],
        source_unit_ids=["u1", "u2"],
        word_count=5,
        char_count=36,
        estimated_token_count=7,
        overlap_word_count=0,
        section_title="Introduction",
        content_type=ContentType.PARAGRAPH,
    )
    chunk2 = ChunkRecord(
        chunk_id="chk_002",
        document_id="doc_1",
        source_file="lecture1.pdf",
        chunk_index=1,
        text="Stationarity requires constant mean and variance",
        source_pages=[2],
        source_unit_ids=["u3"],
        word_count=6,
        char_count=48,
        estimated_token_count=8,
        overlap_word_count=0,
        section_title="Stationarity",
        content_type=ContentType.PARAGRAPH,
    )

    batch_result = engine.embed_chunks([chunk1, chunk2])

    assert isinstance(batch_result, BatchEmbeddingResult)
    assert batch_result.total_chunks == 2
    assert len(batch_result.records) == 2
    assert batch_result.records[0].chunk_id == "chk_001"
    assert batch_result.records[1].chunk_id == "chk_002"
    assert batch_result.elapsed_seconds >= 0.0
    assert batch_result.chunks_per_second > 0.0

    mat = batch_result.to_numpy_matrix()
    assert mat.shape == (2, 384)
    assert mat.dtype == np.float32


def test_embed_chunks_empty():
    provider = MockProvider()
    engine = EmbeddingEngine(provider=provider)

    batch_result = engine.embed_chunks([])
    assert batch_result.total_chunks == 0
    assert len(batch_result.records) == 0
    assert batch_result.to_numpy_matrix().shape == (0, 0)
