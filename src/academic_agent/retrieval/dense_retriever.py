"""Dense vector retriever adapter connecting EmbeddingEngine with VectorStoreProtocol."""

import logging
import time
from typing import Any
import numpy as np

from academic_agent.embeddings.engine import EmbeddingEngine
from academic_agent.storage.protocols import VectorStoreProtocol
from academic_agent.storage.models import SearchResult
from academic_agent.retrieval.models import RetrievalQuery, RetrievalCandidate

logger = logging.getLogger(__name__)


class DenseRetriever:
    """Retrieves academic evidence candidates using dense vector similarity search.

    Connects EmbeddingEngine (for query tokenization and vector generation) with
    VectorStoreProtocol (LanceDB or InMemoryVectorStore) to retrieve a pool of
    ranked candidates while preserving complete provenance.
    """

    def __init__(
        self,
        embedding_engine: EmbeddingEngine,
        vector_store: VectorStoreProtocol,
    ):
        """Initialize DenseRetriever with required embedding and storage dependencies.

        Args:
            embedding_engine: Initialized EmbeddingEngine coordinator.
            vector_store: Initialized vector store satisfying VectorStoreProtocol.
        """
        self._embedding_engine = embedding_engine
        self._vector_store = vector_store
        self._last_diagnostics: dict[str, float] = {
            "embed_latency_ms": 0.0,
            "search_latency_ms": 0.0,
        }

    @property
    def name(self) -> str:
        """Identifier of this retrieval mechanism."""
        return "dense"

    @property
    def embedding_engine(self) -> EmbeddingEngine:
        """Underlying embedding engine."""
        return self._embedding_engine

    @property
    def vector_store(self) -> VectorStoreProtocol:
        """Underlying vector store."""
        return self._vector_store

    @property
    def last_diagnostics(self) -> dict[str, float]:
        """Timing diagnostics from the most recent retrieve() call."""
        return dict(self._last_diagnostics)

    def retrieve(self, query: RetrievalQuery) -> list[RetrievalCandidate]:
        """Fetch dense similarity candidates for a RetrievalQuery.

        Args:
            query: Structured retrieval query containing text, filters, and pool size.

        Returns:
            List of RetrievalCandidate objects ordered by descending cosine similarity.
        """
        if not isinstance(query, RetrievalQuery):
            raise TypeError(f"Expected RetrievalQuery instance, got {type(query).__name__}")

        # 1. Query Encoding (provider handles prefixes if asymmetric, e.g. BGE)
        t0 = time.perf_counter()
        query_record = self._embedding_engine.embed_query(query.text)
        embed_latency_ms = (time.perf_counter() - t0) * 1000.0

        query_vector = query_record.embedding

        # 2. Vector Validation against Store Expectations
        store_meta = self._vector_store.get_index_metadata()
        if store_meta is not None:
            if len(query_vector) != store_meta.embedding_dimension:
                raise ValueError(
                    f"Query vector dimension {len(query_vector)} does not match store index dimension "
                    f"{store_meta.embedding_dimension}"
                )

        # 3. Vector Similarity Search
        # Uses candidate_pool_size (not top_k) to feed downstream fusion and reranking
        t1 = time.perf_counter()
        search_results: list[SearchResult] = self._vector_store.search(
            query_vector=query_vector,
            top_k=query.candidate_pool_size,
            filters=query.filters,
        )
        search_latency_ms = (time.perf_counter() - t1) * 1000.0

        self._last_diagnostics = {
            "embed_latency_ms": embed_latency_ms,
            "search_latency_ms": search_latency_ms,
        }

        # 4. Map SearchResult to RetrievalCandidate
        # Cosine similarity is preserved directly without min-max scaling or negative clipping
        candidates: list[RetrievalCandidate] = []
        for rank_idx, res in enumerate(search_results, start=1):
            cand = RetrievalCandidate(
                record_id=res.record_id,
                chunk_id=res.chunk_id,
                rank=rank_idx,
                retrieval_method=self.name,
                score=res.score,
                normalized_score=None,
                text=res.text,
                source_file=res.source_file,
                source_pages=res.source_pages,
                source_unit_ids=res.source_unit_ids,
                section_title=res.section_title,
                content_type=res.content_type,
                fragment_index=res.fragment_index,
                total_fragments=res.total_fragments,
                metadata={
                    **res.metadata,
                    "dense_similarity": res.score,
                },
            )
            candidates.append(cand)

        logger.debug(
            "DenseRetriever retrieved %d candidates for '%s' (embed=%.2fms, search=%.2fms)",
            len(candidates),
            query.text[:40],
            embed_latency_ms,
            search_latency_ms,
        )

        return candidates
