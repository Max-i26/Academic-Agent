"""In-memory reference implementation of VectorStoreProtocol for testing and development."""

from typing import Any
import numpy as np

from academic_agent.storage.models import (
    VectorRecord,
    SearchResult,
    IndexMetadata,
    FilterQuery,
)
from academic_agent.storage.protocols import VectorStoreProtocol
from academic_agent.storage.exceptions import (
    IncompatibleModelError,
    DuplicateRecordError,
    StoreNotInitializedError,
)


class InMemoryVectorStore:
    """Pure Python/NumPy in-memory vector store satisfying VectorStoreProtocol."""

    def __init__(self, allow_overwrite: bool = True):
        self.allow_overwrite = allow_overwrite
        self._metadata: IndexMetadata | None = None
        self._records: dict[str, VectorRecord] = {}

    def initialize(self, metadata: IndexMetadata) -> None:
        """Initialize store with index metadata."""
        if self._metadata is not None and not self._metadata.is_compatible_with(metadata):
            raise IncompatibleModelError(
                expected_model=self._metadata.model_name,
                actual_model=metadata.model_name,
                detail=f"Store already bound to dimension {self._metadata.embedding_dimension}",
            )
        self._metadata = metadata

    def get_index_metadata(self) -> IndexMetadata | None:
        return self._metadata

    def verify_model_compatibility(self, model_name: str, dimension: int) -> None:
        if self._metadata is None:
            raise StoreNotInitializedError("Store has not been initialized with IndexMetadata")
        if self._metadata.model_name != model_name:
            raise IncompatibleModelError(
                expected_model=self._metadata.model_name,
                actual_model=model_name,
            )
        if self._metadata.embedding_dimension != dimension:
            raise IncompatibleModelError(
                expected_model=f"dim={self._metadata.embedding_dimension}",
                actual_model=f"dim={dimension}",
            )

    def add(self, records: list[VectorRecord]) -> None:
        if not records:
            return
        if self._metadata is None:
            # Auto-initialize with default metadata if not explicitly initialized
            self._metadata = IndexMetadata(
                model_name=records[0].metadata.get("model_name", "unknown"),
                embedding_dimension=len(records[0].vector),
            )

        for rec in records:
            if len(rec.vector) != self._metadata.embedding_dimension:
                raise ValueError(
                    f"Vector dimension {len(rec.vector)} does not match index dimension {self._metadata.embedding_dimension}"
                )
            if rec.record_id in self._records and not self.allow_overwrite:
                raise DuplicateRecordError(f"Record '{rec.record_id}' already exists")
            self._records[rec.record_id] = rec

        self._metadata.total_records = len(self._records)

    def get(self, record_ids: list[str]) -> list[VectorRecord | None]:
        return [self._records.get(rid) for rid in record_ids]

    def search(
        self,
        query_vector: list[float] | np.ndarray,
        top_k: int = 5,
        filters: FilterQuery | dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        if not self._records:
            return []

        q_vec = np.array(query_vector, dtype=np.float32)
        if self._metadata and len(q_vec) != self._metadata.embedding_dimension:
            raise ValueError(
                f"Query vector dimension {len(q_vec)} does not match index {self._metadata.embedding_dimension}"
            )

        # Normalize query if not unit norm
        norm = np.linalg.norm(q_vec)
        if norm > 0:
            q_vec = q_vec / norm

        # Parse filter
        filter_obj: FilterQuery | None = None
        if isinstance(filters, FilterQuery):
            filter_obj = filters
        elif isinstance(filters, dict):
            filter_obj = FilterQuery(custom_filters=filters)

        candidates: list[VectorRecord] = []
        for rec in self._records.values():
            if filter_obj is not None and not filter_obj.matches(rec):
                continue
            candidates.append(rec)

        if not candidates:
            return []

        doc_matrix = np.vstack([c.as_numpy() for c in candidates])
        # Normalize candidates
        doc_norms = np.linalg.norm(doc_matrix, axis=1, keepdims=True)
        doc_norms[doc_norms == 0] = 1.0
        doc_matrix = doc_matrix / doc_norms

        sims = np.dot(doc_matrix, q_vec)
        ranked_indices = np.argsort(-sims)[:top_k]

        results: list[SearchResult] = []
        for idx in ranked_indices:
            cand = candidates[idx]
            results.append(
                SearchResult(
                    record_id=cand.record_id,
                    chunk_id=cand.chunk_id,
                    score=float(sims[idx]),
                    text=cand.text,
                    source_file=cand.source_file,
                    source_pages=cand.source_pages,
                    section_title=cand.section_title,
                    content_type=cand.content_type,
                    fragment_index=cand.fragment_index,
                    total_fragments=cand.total_fragments,
                    metadata=cand.metadata,
                )
            )
        return results

    def delete(self, record_ids: list[str]) -> int:
        deleted = 0
        for rid in record_ids:
            if rid in self._records:
                del self._records[rid]
                deleted += 1
        if self._metadata:
            self._metadata.total_records = len(self._records)
        return deleted

    def delete_by_filter(self, filters: FilterQuery | dict[str, Any]) -> int:
        filter_obj = filters if isinstance(filters, FilterQuery) else FilterQuery(custom_filters=filters)
        to_delete = [rid for rid, rec in self._records.items() if filter_obj.matches(rec)]
        return self.delete(to_delete)

    def count(self) -> int:
        return len(self._records)

    def close(self) -> None:
        pass
