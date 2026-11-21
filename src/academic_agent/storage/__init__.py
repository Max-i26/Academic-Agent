"""Academic-Agent Storage Subsystem."""

from typing import Any

from academic_agent.storage.config import VectorStoreConfig
from academic_agent.storage.exceptions import (
    DuplicateRecordError,
    IncompatibleModelError,
    RecordNotFoundError,
    StorageError,
    StoreNotInitializedError,
)
from academic_agent.storage.mock_store import InMemoryVectorStore
from academic_agent.storage.models import (
    FilterQuery,
    IndexMetadata,
    SearchResult,
    VectorRecord,
)
from academic_agent.storage.protocols import VectorStoreProtocol


def __getattr__(name: str) -> Any:
    """PEP 562 lazy import of heavyweight database providers."""
    if name == "LanceDBVectorStore":
        from academic_agent.storage.providers.lancedb.store import LanceDBVectorStore
        return LanceDBVectorStore
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


__all__ = [
    "VectorRecord",
    "SearchResult",
    "IndexMetadata",
    "FilterQuery",
    "VectorStoreProtocol",
    "VectorStoreConfig",
    "StorageError",
    "IncompatibleModelError",
    "RecordNotFoundError",
    "DuplicateRecordError",
    "StoreNotInitializedError",
    "InMemoryVectorStore",
    "LanceDBVectorStore",
]
