"""Domain exceptions for vector storage and retrieval."""


class StorageError(Exception):
    """Base exception for vector storage operations."""
    pass


class IncompatibleModelError(StorageError):
    """Raised when an operation attempts to mix vectors from different embedding spaces."""

    def __init__(self, expected_model: str, actual_model: str, detail: str = ""):
        self.expected_model = expected_model
        self.actual_model = actual_model
        msg = (
            f"Embedding space mismatch: store expects vectors from '{expected_model}', "
            f"but received vectors from '{actual_model}'. Mixing different embedding spaces "
            f"corrupts similarity search."
        )
        if detail:
            msg += f" ({detail})"
        super().__init__(msg)


class RecordNotFoundError(StorageError):
    """Raised when a requested record ID does not exist in storage."""
    pass


class DuplicateRecordError(StorageError):
    """Raised when inserting a record whose ID already exists and overwrite is disallowed."""
    pass


class StoreNotInitializedError(StorageError):
    """Raised when querying a vector store that has not been initialized."""
    pass
