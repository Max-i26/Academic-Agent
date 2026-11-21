"""LanceDB implementation of VectorStoreProtocol."""

import json
import logging
from pathlib import Path
from typing import Any
import numpy as np

from academic_agent.storage.models import (
    VectorRecord,
    SearchResult,
    IndexMetadata,
    FilterQuery,
)
from academic_agent.storage.config import VectorStoreConfig
from academic_agent.storage.protocols import VectorStoreProtocol
from academic_agent.storage.exceptions import (
    StorageError,
    IncompatibleModelError,
    StoreNotInitializedError,
)
from academic_agent.storage.providers.lancedb.persistence import LancePersistenceManager
from academic_agent.storage.providers.lancedb.schema import (
    validate_record_vector,
    records_to_arrow_table,
)
from academic_agent.storage.providers.lancedb.filters import (
    build_lance_filter,
    _escape_sql_string,
)

logger = logging.getLogger(__name__)


class LanceDBVectorStore:
    """Production vector store implementation backed by LanceDB columnar format.

    Satisfies VectorStoreProtocol with native PyArrow zero-copy storage,
    idempotent upserts, and strict embedding-space isolation.
    """

    def __init__(
        self,
        config: VectorStoreConfig | None = None,
        persistence_manager: LancePersistenceManager | None = None,
        metadata: IndexMetadata | None = None,
    ):
        self._config = config or VectorStoreConfig()
        self._metadata = metadata or IndexMetadata(
            model_name=self._config.expected_model_name,
            embedding_dimension=self._config.expected_dimension,
            distance_metric=self._config.distance_metric,
        )

        if persistence_manager is not None:
            self._pm = persistence_manager
        else:
            persist_dir = self._config.persist_path or "data/storage/lancedb"
            self._pm = LancePersistenceManager(
                db_path=persist_dir,
                collection_name=self._config.collection_name,
                expected_metadata=self._metadata,
            )

        self._table: Any | None = None
        self._is_closed = False

    def _ensure_active(self) -> Any:
        """Ensure store is open and table handle is initialized."""
        if self._is_closed:
            raise StoreNotInitializedError("LanceDBVectorStore is closed. Call reopen() or instantiate a new store.")
        if self._table is None:
            self._table = self._pm.get_or_create_table()
        return self._table

    def initialize(self, metadata: IndexMetadata) -> None:
        """Initialize or mount store with binding IndexMetadata."""
        if self._is_closed:
            self._is_closed = False
        self._metadata = metadata
        self._pm._expected_metadata = metadata
        self._table = self._pm.get_or_create_table()

    def get_index_metadata(self) -> IndexMetadata | None:
        """Return persisted IndexMetadata."""
        if self._is_closed:
            return None
        return self._pm.get_index_metadata()

    def verify_model_compatibility(self, model_name: str, dimension: int) -> None:
        """Assert that an incoming operation matches the index's embedding space."""
        if self._is_closed:
            raise StoreNotInitializedError("Cannot verify compatibility on a closed store.")
        meta = self.get_index_metadata() or self._metadata
        if meta.model_name != model_name:
            raise IncompatibleModelError(
                expected_model=meta.model_name,
                actual_model=model_name,
            )
        if meta.embedding_dimension != dimension:
            raise IncompatibleModelError(
                expected_model=f"dim={meta.embedding_dimension}",
                actual_model=f"dim={dimension}",
            )

    def add(self, records: list[VectorRecord]) -> None:
        """Insert or update batch of VectorRecords into LanceDB via idempotent upsert."""
        tbl = self._ensure_active()
        if not records:
            return

        # Deduplicate records by record_id within the incoming batch (latest wins)
        deduped: dict[str, VectorRecord] = {}
        for r in records:
            deduped[r.record_id] = r
        clean_records = list(deduped.values())

        # Validate all records before writing (all-or-nothing guarantee)
        expected_dim = self._metadata.embedding_dimension
        for r in clean_records:
            validate_record_vector(
                record=r,
                expected_dimension=expected_dim,
                check_normalized=self._metadata.normalized,
            )

        # Convert to strongly-typed PyArrow table
        arrow_table = records_to_arrow_table(clean_records, schema=tbl.schema)

        # Execute idempotent merge_insert on record_id
        tbl.merge_insert("record_id").when_matched_update_all().when_not_matched_insert_all().execute(arrow_table)

    def get(self, record_ids: list[str]) -> list[VectorRecord | None]:
        """Fetch records by primary record_ids, preserving requested order and returning None for misses."""
        if self._is_closed:
            raise StoreNotInitializedError("Cannot get records from a closed store.")
        if not record_ids:
            return []

        tbl = self._ensure_active()
        if len(tbl) == 0:
            return [None] * len(record_ids)

        escaped = [_escape_sql_string(rid) for rid in record_ids]
        sql = f"record_id IN ({', '.join(escaped)})"
        rows = tbl.search().where(sql).limit(len(record_ids)).to_list()

        by_id = {row["record_id"]: row for row in rows}

        results: list[VectorRecord | None] = []
        for rid in record_ids:
            row = by_id.get(rid)
            if row is not None:
                results.append(self._row_to_record(row))
            else:
                results.append(None)
        return results

    def _row_to_record(self, row: dict[str, Any]) -> VectorRecord:
        """Convert a LanceDB row dict into a typed VectorRecord."""
        meta_dict: dict[str, Any] = {}
        if row.get("metadata_json"):
            try:
                meta_dict = json.loads(row["metadata_json"])
            except Exception:
                meta_dict = {}

        return VectorRecord(
            record_id=row["record_id"],
            chunk_id=row["chunk_id"],
            vector=list(row["vector"]),
            text=row["text"],
            source_file=row["source_file"],
            source_pages=list(row["source_pages"]),
            source_unit_ids=list(row["source_unit_ids"]),
            section_title=row.get("section_title"),
            content_type=row.get("content_type", "PARAGRAPH"),
            fragment_index=row.get("fragment_index", 0),
            total_fragments=row.get("total_fragments", 1),
            metadata=meta_dict,
            created_at=row.get("created_at", ""),
        )

    def search(
        self,
        query_vector: list[float] | np.ndarray,
        top_k: int = 5,
        filters: FilterQuery | dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """Perform vector similarity search with optional metadata filtering."""
        if self._is_closed:
            raise StoreNotInitializedError("Cannot search on a closed store.")

        # Validate and format query vector before checking count
        q_arr = np.array(query_vector, dtype=np.float32)
        if len(q_arr) != self._metadata.embedding_dimension:
            raise ValueError(
                f"Query vector dimension {len(q_arr)} does not match index {self._metadata.embedding_dimension}"
            )
        if not np.all(np.isfinite(q_arr)):
            raise ValueError("Query vector contains NaN or Inf values")

        if self._metadata.normalized:
            norm = float(np.linalg.norm(q_arr))
            if norm == 0 or abs(norm - 1.0) > 1e-2:
                raise ValueError(f"Query vector is not unit normalized (norm={norm:.4f})")

        if top_k <= 0:
            return []

        tbl = self._ensure_active()
        if len(tbl) == 0:
            return []

        sql_filter = build_lance_filter(filters)

        query = tbl.search(q_arr.tolist()).metric("cosine").limit(top_k)
        if sql_filter:
            query = query.where(sql_filter)

        rows = query.to_list()
        results: list[SearchResult] = []
        for r in rows:
            dist = r.get("_distance", 0.0)
            sim = float(1.0 - dist)
            meta_dict = {}
            if r.get("metadata_json"):
                try:
                    meta_dict = json.loads(r["metadata_json"])
                except Exception:
                    pass

            results.append(
                SearchResult(
                    record_id=r["record_id"],
                    chunk_id=r["chunk_id"],
                    score=sim,
                    text=r["text"],
                    source_file=r["source_file"],
                    source_pages=list(r["source_pages"]),
                    section_title=r.get("section_title"),
                    content_type=r.get("content_type", "PARAGRAPH"),
                    fragment_index=r.get("fragment_index", 0),
                    total_fragments=r.get("total_fragments", 1),
                    metadata=meta_dict,
                )
            )
        return results

    def delete(self, record_ids: list[str]) -> int:
        """Delete records by record_ids. Returns count of deleted records."""
        if self._is_closed:
            raise StoreNotInitializedError("Cannot delete from a closed store.")
        if not record_ids:
            return 0

        tbl = self._ensure_active()
        cnt_before = len(tbl)
        if cnt_before == 0:
            return 0

        escaped = [_escape_sql_string(rid) for rid in record_ids]
        sql = f"record_id IN ({', '.join(escaped)})"
        tbl.delete(sql)
        cnt_after = len(tbl)
        return cnt_before - cnt_after

    def delete_by_filter(self, filters: FilterQuery | dict[str, Any]) -> int:
        """Delete records matching filter. Returns count of deleted records."""
        if self._is_closed:
            raise StoreNotInitializedError("Cannot delete from a closed store.")
        tbl = self._ensure_active()
        cnt_before = len(tbl)
        if cnt_before == 0:
            return 0

        sql_filter = build_lance_filter(filters)
        if sql_filter is None:
            return 0

        tbl.delete(sql_filter)
        cnt_after = len(tbl)
        return cnt_before - cnt_after

    def count(self) -> int:
        """Return count of active records."""
        if self._is_closed:
            raise StoreNotInitializedError("Cannot count records on a closed store.")
        tbl = self._ensure_active()
        return len(tbl)

    def close(self) -> None:
        """Safely close table and persistence manager."""
        if self._is_closed:
            return
        self._table = None
        self._pm.close()
        self._is_closed = True
