"""EmbeddingEngine: Central coordinator for embedding generation and validation."""

import time
import logging
from typing import Any
import numpy as np

from academic_agent.embeddings.protocols import EmbeddingProvider
from academic_agent.embeddings.config import EmbeddingConfig, KnownEmbeddingModel
from academic_agent.embeddings.models import (
    EmbeddingMetadata,
    DocumentEmbeddingRecord,
    QueryEmbeddingRecord,
    BatchEmbeddingResult,
)
from academic_agent.embeddings.validator import EmbeddingValidator

logger = logging.getLogger(__name__)


class EmbeddingEngine:
    """Coordinates embedding generation, batching, validation, and metadata tracking.

    Strictly decoupled from model-specific architectures through the
    EmbeddingProvider protocol.
    """

    def __init__(
        self,
        provider: EmbeddingProvider,
        config: EmbeddingConfig | None = None,
        validator: EmbeddingValidator | None = None,
    ):
        self._provider = provider
        self._config = config or EmbeddingConfig()
        self._validator = validator or EmbeddingValidator()

    @classmethod
    def from_config(cls, config: EmbeddingConfig) -> "EmbeddingEngine":
        """Factory method to instantiate the engine with the appropriate provider."""
        from academic_agent.embeddings.providers.minilm import MiniLMProvider
        from academic_agent.embeddings.providers.bge import BGEProvider

        model_name = config.model_name
        device_str = config.device.value

        if model_name in (KnownEmbeddingModel.MINILM.value, "minilm", "all-MiniLM-L6-v2"):
            provider = MiniLMProvider(model_name=KnownEmbeddingModel.MINILM.value, device=device_str)
        elif model_name in (KnownEmbeddingModel.BGE_SMALL.value, "bge", "bge-small-en-v1.5"):
            provider = BGEProvider(model_name=KnownEmbeddingModel.BGE_SMALL.value, device=device_str)
        else:
            # Generic fallback to MiniLM or custom sentence-transformer
            logger.info("Using MiniLMProvider for custom model: %s", model_name)
            provider = MiniLMProvider(model_name=model_name, device=device_str)

        return cls(provider=provider, config=config)

    @property
    def provider(self) -> EmbeddingProvider:
        """The underlying embedding provider."""
        return self._provider

    @property
    def config(self) -> EmbeddingConfig:
        """Active configuration."""
        return self._config

    def _build_metadata(self) -> EmbeddingMetadata:
        return EmbeddingMetadata(
            model_name=self._provider.name,
            dimension=self._provider.dimension,
            max_sequence_length=self._provider.max_sequence_length,
            normalized=self._config.normalize_embeddings,
            device_used=self._provider.device,
        )

    def embed_query(self, query: str) -> QueryEmbeddingRecord:
        """Encode a search query into a verified QueryEmbeddingRecord.

        Args:
            query: User or agent search query string.

        Returns:
            QueryEmbeddingRecord containing normalized embedding and provenance.
        """
        if not query or not query.strip():
            raise ValueError("Query string cannot be empty")

        token_count = self._provider.count_tokens(query)
        self._validator.validate_token_count(
            token_count=token_count,
            max_tokens=self._provider.max_sequence_length,
            strict=self._config.strict_token_limit,
        )

        vectors = self._provider.encode_queries(
            [query],
            batch_size=1,
            normalize=self._config.normalize_embeddings,
        )

        self._validator.validate_vectors(
            vectors,
            expected_dimension=self._config.expected_dimension,
            expected_count=1,
        )
        if self._config.normalize_embeddings:
            self._validator.validate_normalized(vectors)

        metadata = self._build_metadata()
        return QueryEmbeddingRecord(
            query_text=query,
            embedding=vectors[0].tolist(),
            token_count=token_count,
            metadata=metadata,
        )

    def embed_documents(
        self,
        texts: list[str],
        chunk_ids: list[str] | None = None,
    ) -> list[DocumentEmbeddingRecord]:
        """Encode a list of document strings into DocumentEmbeddingRecords with validation.

        Args:
            texts: List of text passages/chunks.
            chunk_ids: Optional corresponding chunk identifiers.

        Returns:
            List of DocumentEmbeddingRecord instances.
        """
        if not texts:
            return []

        ids = chunk_ids if chunk_ids is not None else [f"doc_{i}" for i in range(len(texts))]
        if len(ids) != len(texts):
            raise ValueError("Length of chunk_ids must match length of texts")

        # 1. Exact token audit across all inputs (NO SILENT TRUNCATION)
        token_counts: list[int] = []
        for text, cid in zip(texts, ids):
            count = self._provider.count_tokens(text)
            self._validator.validate_token_count(
                token_count=count,
                max_tokens=self._provider.max_sequence_length,
                chunk_id=cid,
                strict=self._config.strict_token_limit,
            )
            token_counts.append(count)

        # 2. Batch encoding
        vectors = self._provider.encode_documents(
            texts,
            batch_size=self._config.batch_size,
            normalize=self._config.normalize_embeddings,
        )

        # 3. Vector validation
        self._validator.validate_vectors(
            vectors,
            expected_dimension=self._config.expected_dimension,
            expected_count=len(texts),
        )
        if self._config.normalize_embeddings:
            self._validator.validate_normalized(vectors)

        metadata = self._build_metadata()
        records = [
            DocumentEmbeddingRecord(
                chunk_id=cid,
                embedding=vectors[i].tolist(),
                token_count=token_counts[i],
                metadata=metadata,
            )
            for i, cid in enumerate(ids)
        ]
        return records

    def embed_chunks(self, chunks: list[Any]) -> BatchEmbeddingResult:
        """Embed a list of ChunkRecord instances from Phase 2 chunking.

        Args:
            chunks: List of ChunkRecord objects with .chunk_id and .text attributes.

        Returns:
            BatchEmbeddingResult containing all records, throughput, and elapsed time.
        """
        if not chunks:
            return BatchEmbeddingResult(
                records=[],
                total_chunks=0,
                elapsed_seconds=0.0,
                chunks_per_second=0.0,
                device_used=self._provider.device,
            )

        texts = [c.text for c in chunks]
        ids = [c.chunk_id for c in chunks]

        t0 = time.perf_counter()
        records = self.embed_documents(texts=texts, chunk_ids=ids)
        elapsed = time.perf_counter() - t0

        throughput = len(chunks) / elapsed if elapsed > 0 else float("inf")

        return BatchEmbeddingResult(
            records=records,
            total_chunks=len(records),
            elapsed_seconds=elapsed,
            chunks_per_second=throughput,
            device_used=self._provider.device,
        )
