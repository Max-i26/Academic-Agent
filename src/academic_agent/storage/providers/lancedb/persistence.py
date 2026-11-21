"""LanceDB connection lifecycle, path validation, table initialization, and reopen management."""

import json
import logging
from pathlib import Path
from typing import Any
import lancedb

from academic_agent.storage.models import IndexMetadata
from academic_agent.storage.exceptions import (
    StorageError,
    IncompatibleModelError,
    StoreNotInitializedError,
)
from academic_agent.storage.providers.lancedb.schema import (
    create_lance_schema,
    extract_index_metadata,
    validate_schema_compatibility,
)

logger = logging.getLogger(__name__)


class LancePersistenceManager:
    """Manages connection lifecycle, table initialization, and metadata integrity for LanceDB.

    Ensures that existing databases are reopened with strict compatibility validation
    and never silently overwritten.
    """

    def __init__(
        self,
        db_path: str | Path,
        collection_name: str,
        expected_metadata: IndexMetadata,
    ):
        if not db_path or not str(db_path).strip():
            raise ValueError("db_path cannot be empty or whitespace")
        if not collection_name or not collection_name.strip():
            raise ValueError("collection_name cannot be empty or whitespace")

        self._db_path = Path(db_path).resolve()
        self._collection_name = collection_name.strip()
        self._expected_metadata = expected_metadata
        self._connection: Any | None = None
        self._table: Any | None = None
        self._is_closed = False

    @property
    def db_path(self) -> Path:
        return self._db_path

    @property
    def collection_name(self) -> str:
        return self._collection_name

    @property
    def expected_metadata(self) -> IndexMetadata:
        return self._expected_metadata

    @property
    def is_connected(self) -> bool:
        return self._connection is not None and not self._is_closed

    @property
    def is_closed(self) -> bool:
        return self._is_closed

    def _resolve_and_validate_path(self) -> None:
        """Validate path safety and ensure directory exists."""
        if self._db_path.is_file():
            raise ValueError(
                f"db_path '{self._db_path}' points to an existing file, expected a directory."
            )
        try:
            self._db_path.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            raise StorageError(f"Failed to create or access database directory '{self._db_path}': {e}") from e

    def _get_connection(self) -> Any:
        """Get active connection or establish a new connection."""
        if self._connection is None or self._is_closed:
            self._resolve_and_validate_path()
            self._connection = lancedb.connect(str(self._db_path))
            self._is_closed = False
        return self._connection

    def _list_table_names(self, conn: Any) -> list[str]:
        """Safely list table names handling both modern list_tables() and table_names()."""
        if hasattr(conn, "list_tables"):
            return list(conn.list_tables())
        return list(conn.table_names())

    def get_or_create_table(self) -> Any:
        """Initialize or reopen table with strict embedding-space isolation.

        Returns:
            lancedb Table handle.

        Raises:
            IncompatibleModelError: If the existing table metadata does not match expected_metadata.
            StorageError: If the existing table metadata is missing or corrupted.
        """
        if self._table is not None and not self._is_closed:
            return self._table

        conn = self._get_connection()
        existing_tables = self._list_table_names(conn)

        if self._collection_name not in existing_tables:
            logger.info(
                "Creating new LanceDB table '%s' in '%s' (model=%s, dim=%d)",
                self._collection_name,
                self._db_path,
                self._expected_metadata.model_name,
                self._expected_metadata.embedding_dimension,
            )
            schema = create_lance_schema(
                dimension=self._expected_metadata.embedding_dimension,
                metadata=self._expected_metadata,
            )
            # Create without overwriting existing unrelated data
            self._table = conn.create_table(self._collection_name, schema=schema)
            # Persist companion JSON sidecar for human inspection and recovery
            self._write_metadata_sidecar()
        else:
            logger.info("Opening existing LanceDB table '%s' in '%s'", self._collection_name, self._db_path)
            tbl = conn.open_table(self._collection_name)
            self._validate_existing_table(tbl)
            self._table = tbl

        return self._table

    def _write_metadata_sidecar(self) -> None:
        """Write human-readable JSON metadata sidecar alongside table."""
        sidecar_path = self._db_path / f"{self._collection_name}.meta.json"
        try:
            with open(sidecar_path, "w", encoding="utf-8") as f:
                f.write(self._expected_metadata.model_dump_json(indent=2))
        except OSError as e:
            logger.warning("Could not write metadata sidecar to %s: %s", sidecar_path, e)

    def _validate_existing_table(self, tbl: Any) -> None:
        """Verify that an existing table's schema and metadata match expected_metadata."""
        schema = tbl.schema
        actual_meta = extract_index_metadata(schema)
        sidecar_path = self._db_path / f"{self._collection_name}.meta.json"

        if actual_meta is None:
            # Fallback when Arrow metadata is completely absent
            if sidecar_path.is_file():
                try:
                    with open(sidecar_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    actual_meta = IndexMetadata(**data)
                except Exception as e:
                    raise StorageError(
                        f"Existing table '{self._collection_name}' has missing Arrow metadata "
                        f"and corrupted sidecar JSON: {e}"
                    ) from e
            else:
                raise StorageError(
                    f"Existing table '{self._collection_name}' has no embedded IndexMetadata "
                    f"or sidecar metadata. Refusing to operate on an unverified index."
                )
        else:
            # Integrity cross-check: if sidecar also exists, they must not disagree
            if sidecar_path.is_file():
                try:
                    with open(sidecar_path, "r", encoding="utf-8") as f:
                        sidecar_data = json.load(f)
                    sidecar_meta = IndexMetadata(**sidecar_data)
                    if not actual_meta.is_compatible_with(sidecar_meta):
                        raise StorageError(
                            f"Metadata integrity conflict in '{self._collection_name}': "
                            f"Arrow schema metadata ({actual_meta.model_name}, dim={actual_meta.embedding_dimension}) "
                            f"disagrees with sidecar JSON ({sidecar_meta.model_name}, dim={sidecar_meta.embedding_dimension})."
                        )
                except StorageError:
                    raise
                except Exception as e:
                    logger.warning("Could not read sidecar JSON for integrity cross-check: %s", e)

        # Validate complete compatibility with active expected_metadata
        validate_schema_compatibility(schema, self._expected_metadata)

    def get_index_metadata(self) -> IndexMetadata | None:
        """Return the persisted IndexMetadata if table exists, or expected_metadata if initialized."""
        if self._is_closed:
            return None
        conn = self._get_connection()
        existing = self._list_table_names(conn)
        if self._collection_name not in existing:
            return None
        tbl = conn.open_table(self._collection_name)
        meta = extract_index_metadata(tbl.schema)
        if meta:
            return meta
        # Fallback to sidecar
        sidecar_path = self._db_path / f"{self._collection_name}.meta.json"
        if sidecar_path.is_file():
            with open(sidecar_path, "r", encoding="utf-8") as f:
                return IndexMetadata(**json.load(f))
        return None

    def close(self) -> None:
        """Release table handle and close connection."""
        self._table = None
        self._connection = None
        self._is_closed = True
        logger.debug("Closed LancePersistenceManager connection to '%s'", self._db_path)

    def reopen(self) -> Any:
        """Reopen a previously closed connection and return validated table."""
        self._is_closed = False
        return self.get_or_create_table()
