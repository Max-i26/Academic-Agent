"""VectorStore protocol and interface contracts."""

from typing import Any, Protocol, runtime_checkable
import numpy as np

from academic_agent.storage.models import (
    VectorRecord,
    SearchResult,
    IndexMetadata,
    FilterQuery,
)


@runtime_checkable
class VectorStoreProtocol(Protocol):
    """Abstract interface contract for all vector storage backends.

    Ensures that retrieval, filtering, provenance preservation, and
    embedding-space isolation operate uniformly regardless of whether
    the backend is in-memory, FAISS, ChromaDB, or LanceDB.
    """

    def initialize(self, metadata: IndexMetadata) -> None:
        """Initialize or mount the vector store with binding IndexMetadata."""
        ...

    def get_index_metadata(self) -> IndexMetadata | None:
        """Return the metadata describing the active index and embedding model."""
        ...

    def verify_model_compatibility(self, model_name: str, dimension: int) -> None:
        """Assert that an incoming operation matches the index's embedding space.

        Raises:
            IncompatibleModelError: If model_name or dimension do not match.
        """
        ...

    def add(self, records: list[VectorRecord]) -> None:
        """Insert or update batch of VectorRecords into the store."""
        ...

    def get(self, record_ids: list[str]) -> list[VectorRecord | None]:
        """Fetch records by their primary record_ids."""
        ...

    def search(
        self,
        query_vector: list[float] | np.ndarray,
        top_k: int = 5,
        filters: FilterQuery | dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """Perform similarity search and return top-K matching SearchResult items.

        Args:
            query_vector: Dense query vector.
            top_k: Number of highest-ranking results to return.
            filters: Optional FilterQuery or dict to restrict search scope.

        Returns:
            List of SearchResult items sorted by score descending.
        """
        ...

    def delete(self, record_ids: list[str]) -> int:
        """Delete records by ID. Returns count of deleted records."""
        ...

    def delete_by_filter(self, filters: FilterQuery | dict[str, Any]) -> int:
        """Delete all records matching the filter. Returns count of deleted records."""
        ...

    def count(self) -> int:
        """Return total number of active records in the index."""
        ...

    def close(self) -> None:
        """Flush buffers and cleanly release resources."""
        ...
