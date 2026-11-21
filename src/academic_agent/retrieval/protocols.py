"""Protocols defining contracts for retrievers, fusion strategies, rerankers, and pipelines."""

from typing import Protocol, runtime_checkable
from academic_agent.retrieval.models import (
    RetrievalQuery,
    RetrievalCandidate,
    RetrievalResponse,
)


@runtime_checkable
class RetrieverProtocol(Protocol):
    """Base interface for any candidate retrieval component (dense, sparse, or external)."""

    @property
    def name(self) -> str:
        """Identifier of this retrieval mechanism (e.g. 'dense', 'bm25')."""
        ...

    def retrieve(self, query: RetrievalQuery) -> list[RetrievalCandidate]:
        """Fetch candidate records satisfying the query and filter constraints.

        Args:
            query: The structured retrieval request containing text, top_k, and filters.

        Returns:
            List of RetrievalCandidate objects ranked by score descending.
        """
        ...


@runtime_checkable
class DenseRetrieverProtocol(RetrieverProtocol, Protocol):
    """Protocol for dense vector similarity retrieval."""

    ...


@runtime_checkable
class SparseRetrieverProtocol(RetrieverProtocol, Protocol):
    """Protocol for sparse keyword/lexical retrieval (e.g. BM25)."""

    def index_records(self, records: list) -> None:
        """Index a collection of text records for sparse keyword search."""
        ...


@runtime_checkable
class FusionStrategyProtocol(Protocol):
    """Protocol for combining multi-method candidate lists into a unified ranked list."""

    @property
    def name(self) -> str:
        """Name of the fusion algorithm (e.g. 'rrf', 'linear_combination')."""
        ...

    def fuse(
        self,
        query: RetrievalQuery,
        candidates_by_method: dict[str, list[RetrievalCandidate]],
        top_k: int,
    ) -> list[RetrievalCandidate]:
        """Merge candidate lists from different retrieval methods.

        Args:
            query: Originating retrieval query.
            candidates_by_method: Mapping from method name ('dense', 'sparse') to its candidate list.
            top_k: Maximum number of fused candidates to return.

        Returns:
            Unified, deduplicated, and ranked list of RetrievalCandidates.
        """
        ...


@runtime_checkable
class RerankerProtocol(Protocol):
    """Protocol for post-retrieval candidate relevance refinement and rescoring."""

    @property
    def name(self) -> str:
        """Name of the reranking algorithm or model."""
        ...

    def rerank(
        self,
        query: RetrievalQuery,
        candidates: list[RetrievalCandidate],
        top_k: int,
    ) -> list[RetrievalCandidate]:
        """Rescore and reorder a list of candidate passages against the query.

        Args:
            query: Originating retrieval query.
            candidates: Input candidate passages to be refined.
            top_k: Maximum candidates to keep.

        Returns:
            Reranked list of candidates.
        """
        ...


@runtime_checkable
class RetrievalPipelineProtocol(Protocol):
    """Protocol for end-to-end multi-stage retrieval orchestration."""

    def search(self, query: RetrievalQuery | str) -> RetrievalResponse:
        """Execute end-to-end retrieval, fusion, deduplication, and packaging into a RetrievalResponse.

        Args:
            query: Either a raw query string or a configured RetrievalQuery.

        Returns:
            Complete RetrievalResponse ready for downstream answer synthesis.
        """
        ...
