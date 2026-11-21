"""End-to-End Retrieval Pipeline orchestrating Dense, Sparse, Fusion, and Reranking."""

import logging
import time
from typing import Any

from academic_agent.retrieval.models import (
    RetrievalQuery,
    RetrievalCandidate,
    Evidence,
    RetrievalStrategy,
    RetrievalDiagnostics,
    RetrievalResponse,
)
from academic_agent.retrieval.protocols import (
    RetrieverProtocol,
    DenseRetrieverProtocol,
    SparseRetrieverProtocol,
    FusionStrategyProtocol,
    RerankerProtocol,
    RetrievalPipelineProtocol,
)
from academic_agent.retrieval.fusion import ReciprocalRankFusion

logger = logging.getLogger(__name__)


class RetrievalPipeline:
    """Orchestrates multi-stage retrieval, routing queries to dense, sparse, or hybrid pipelines.

    Satisfies RetrievalPipelineProtocol:
    - Normalizes raw string queries into RetrievalQuery objects.
    - Executes dense semantic search and/or sparse BM25 search.
    - Fuses candidate pools using ReciprocalRankFusion with candidate agreement attribution.
    - Optionally reranks top candidates through a pluggable RerankerProtocol.
    - Packages final results into grounded Evidence objects with clean academic citations.
    - Captures per-stage latency telemetry and candidate diagnostic counts.
    """

    def __init__(
        self,
        dense_retriever: DenseRetrieverProtocol | None = None,
        sparse_retriever: SparseRetrieverProtocol | None = None,
        fusion_strategy: FusionStrategyProtocol | None = None,
        reranker: RerankerProtocol | None = None,
    ):
        """Initialize the pipeline with component retrievers, fusion strategy, and optional reranker.

        Args:
            dense_retriever: Dense retriever implementing DenseRetrieverProtocol.
            sparse_retriever: Sparse retriever implementing SparseRetrieverProtocol.
            fusion_strategy: Fusion strategy (defaults to ReciprocalRankFusion(k=60)).
            reranker: Optional post-retrieval reranker implementing RerankerProtocol.
        """
        self.dense_retriever = dense_retriever
        self.sparse_retriever = sparse_retriever
        self.fusion_strategy = fusion_strategy or ReciprocalRankFusion(k=60)
        self.reranker = reranker

    def search(self, query: RetrievalQuery | str) -> RetrievalResponse:
        """Execute end-to-end retrieval, fusion, deduplication, and evidence packaging.

        Args:
            query: Either a raw query string or a structured RetrievalQuery.

        Returns:
            Complete RetrievalResponse containing ranked Evidence, telemetry, and diagnostics.
        """
        # Normalize string inputs to structured RetrievalQuery
        if isinstance(query, str):
            query_obj = RetrievalQuery(text=query)
        elif isinstance(query, RetrievalQuery):
            query_obj = query
        else:
            raise TypeError(f"Expected RetrievalQuery or str, got {type(query).__name__}")

        t_pipeline_start = time.perf_counter()
        strategy = query_obj.strategy

        # Route execution based on strategy
        if strategy == RetrievalStrategy.DENSE:
            return self._search_dense_only(query_obj, t_pipeline_start)
        elif strategy == RetrievalStrategy.SPARSE:
            return self._search_sparse_only(query_obj, t_pipeline_start)
        elif strategy in (RetrievalStrategy.HYBRID, RetrievalStrategy.HYBRID_RERANK):
            return self._search_hybrid(query_obj, t_pipeline_start, apply_rerank=(strategy == RetrievalStrategy.HYBRID_RERANK))
        else:
            raise ValueError(f"Unsupported retrieval strategy: '{strategy}'")

    def _search_dense_only(self, query: RetrievalQuery, t_start: float) -> RetrievalResponse:
        """Execute dense-only retrieval pipeline."""
        if self.dense_retriever is None:
            raise ValueError("DenseRetriever is not configured in this RetrievalPipeline.")

        t0 = time.perf_counter()
        candidates = self.dense_retriever.retrieve(query)
        dense_latency_ms = (time.perf_counter() - t0) * 1000.0

        # Truncate candidates to top_k
        top_candidates = candidates[: query.top_k]
        evidence = [c.to_evidence() for c in top_candidates]

        total_latency_ms = (time.perf_counter() - t_start) * 1000.0
        diagnostics = RetrievalDiagnostics(
            dense_latency_ms=dense_latency_ms,
            total_latency_ms=total_latency_ms,
            candidate_counts_by_method={"dense": len(candidates)},
            method_agreement_count=0,
        )

        return RetrievalResponse(
            query=query,
            evidence=evidence,
            candidates=candidates,
            diagnostics=diagnostics,
            strategy_used=RetrievalStrategy.DENSE,
        )

    def _search_sparse_only(self, query: RetrievalQuery, t_start: float) -> RetrievalResponse:
        """Execute sparse-only (BM25) retrieval pipeline."""
        if self.sparse_retriever is None:
            raise ValueError("SparseRetriever is not configured in this RetrievalPipeline.")

        t0 = time.perf_counter()
        candidates = self.sparse_retriever.retrieve(query)
        sparse_latency_ms = (time.perf_counter() - t0) * 1000.0

        # Truncate candidates to top_k
        top_candidates = candidates[: query.top_k]
        evidence = [c.to_evidence() for c in top_candidates]

        total_latency_ms = (time.perf_counter() - t_start) * 1000.0
        diagnostics = RetrievalDiagnostics(
            sparse_latency_ms=sparse_latency_ms,
            total_latency_ms=total_latency_ms,
            candidate_counts_by_method={"sparse": len(candidates)},
            method_agreement_count=0,
        )

        return RetrievalResponse(
            query=query,
            evidence=evidence,
            candidates=candidates,
            diagnostics=diagnostics,
            strategy_used=RetrievalStrategy.SPARSE,
        )

    def _search_hybrid(
        self,
        query: RetrievalQuery,
        t_start: float,
        apply_rerank: bool = False,
    ) -> RetrievalResponse:
        """Execute hybrid dense + sparse retrieval with Reciprocal Rank Fusion."""
        # Handle fallback if one retriever is not configured
        if self.dense_retriever is None and self.sparse_retriever is not None:
            logger.warning("DenseRetriever not configured for hybrid query. Falling back to sparse-only.")
            return self._search_sparse_only(query, t_start)
        elif self.sparse_retriever is None and self.dense_retriever is not None:
            logger.warning("SparseRetriever not configured for hybrid query. Falling back to dense-only.")
            return self._search_dense_only(query, t_start)
        elif self.dense_retriever is None and self.sparse_retriever is None:
            raise ValueError("Neither DenseRetriever nor SparseRetriever is configured in this pipeline.")

        # 1. Dense retrieval
        t_d0 = time.perf_counter()
        dense_candidates = self.dense_retriever.retrieve(query)  # type: ignore
        dense_latency_ms = (time.perf_counter() - t_d0) * 1000.0

        # 2. Sparse retrieval
        t_s0 = time.perf_counter()
        sparse_candidates = self.sparse_retriever.retrieve(query)  # type: ignore
        sparse_latency_ms = (time.perf_counter() - t_s0) * 1000.0

        # 3. Reciprocal Rank Fusion
        t_f0 = time.perf_counter()
        fused_candidates = self.fusion_strategy.fuse(
            query=query,
            candidates_by_method={
                "dense": dense_candidates,
                "sparse": sparse_candidates,
            },
            top_k=query.top_k,
        )
        fusion_latency_ms = (time.perf_counter() - t_f0) * 1000.0

        # 4. Optional Reranking (Stage 4.5 hook)
        rerank_latency_ms: float | None = None
        final_candidates = fused_candidates
        if apply_rerank:
            if self.reranker is not None:
                t_r0 = time.perf_counter()
                final_candidates = self.reranker.rerank(
                    query=query,
                    candidates=fused_candidates,
                    top_k=query.top_k,
                )
                rerank_latency_ms = (time.perf_counter() - t_r0) * 1000.0
            else:
                logger.warning("Reranker requested (HYBRID_RERANK) but none configured. Using RRF ranking.")

        # 5. Evidence packaging
        evidence = [c.to_evidence() for c in final_candidates]

        # Calculate consensus diagnostics
        top_agreement_count = sum(1 for e in evidence if len(e.retrieval_methods) > 1)
        dense_ids = {c.record_id for c in dense_candidates}
        sparse_ids = {c.record_id for c in sparse_candidates}
        pool_agreement_count = len(dense_ids.intersection(sparse_ids))

        total_latency_ms = (time.perf_counter() - t_start) * 1000.0

        diagnostics = RetrievalDiagnostics(
            dense_latency_ms=dense_latency_ms,
            sparse_latency_ms=sparse_latency_ms,
            fusion_latency_ms=fusion_latency_ms,
            rerank_latency_ms=rerank_latency_ms,
            total_latency_ms=total_latency_ms,
            candidate_counts_by_method={
                "dense": len(dense_candidates),
                "sparse": len(sparse_candidates),
            },
            method_agreement_count=top_agreement_count,
            additional_info={
                "pool_agreement_count": pool_agreement_count,
            },
        )

        all_candidates = dense_candidates + sparse_candidates
        strategy_used = (
            RetrievalStrategy.HYBRID_RERANK
            if (apply_rerank and self.reranker is not None)
            else RetrievalStrategy.HYBRID
        )

        return RetrievalResponse(
            query=query,
            evidence=evidence,
            candidates=all_candidates,
            diagnostics=diagnostics,
            strategy_used=strategy_used,
        )
