"""PyArrow schema definition, metadata serialization, and validation for LanceDB."""

import json
import logging
from typing import Any
import numpy as np
import pyarrow as pa

from academic_agent.storage.models import IndexMetadata, VectorRecord
from academic_agent.storage.exceptions import IncompatibleModelError

logger = logging.getLogger(__name__)


def create_lance_schema(
    dimension: int = 384,
    metadata: IndexMetadata | None = None,
) -> pa.Schema:
    """Create a strictly typed PyArrow schema for the LanceDB academic vector table.

    Embeds IndexMetadata into the schema metadata as byte key-value pairs
    for persistent embedding-space isolation across application restarts.
    """
    arrow_metadata: dict[bytes, bytes] = {}
    if metadata is not None:
        arrow_metadata = {
            b"model_name": metadata.model_name.encode("utf-8"),
            b"model_revision": (metadata.model_revision or "").encode("utf-8"),
            b"embedding_dimension": str(metadata.embedding_dimension).encode("utf-8"),
            b"distance_metric": metadata.distance_metric.encode("utf-8"),
            b"normalized": str(metadata.normalized).lower().encode("utf-8"),
            b"created_at": metadata.created_at.encode("utf-8"),
            b"updated_at": metadata.updated_at.encode("utf-8"),
        }

    return pa.schema(
        [
            pa.field("record_id", pa.string(), nullable=False),
            pa.field("chunk_id", pa.string(), nullable=False),
            pa.field("vector", pa.list_(pa.float32(), dimension), nullable=False),
            pa.field("text", pa.string(), nullable=False),
            pa.field("source_file", pa.string(), nullable=False),
            pa.field("source_pages", pa.list_(pa.int32()), nullable=False),
            pa.field("source_unit_ids", pa.list_(pa.string()), nullable=False),
            pa.field("section_title", pa.string(), nullable=True),
            pa.field("content_type", pa.string(), nullable=False),
            pa.field("fragment_index", pa.int32(), nullable=False),
            pa.field("total_fragments", pa.int32(), nullable=False),
            pa.field("metadata_json", pa.string(), nullable=False),
            pa.field("created_at", pa.string(), nullable=False),
        ],
        metadata=arrow_metadata if arrow_metadata else None,
    )


def extract_index_metadata(schema: pa.Schema) -> IndexMetadata | None:
    """Extract and reconstruct IndexMetadata from PyArrow schema metadata."""
    if not schema.metadata:
        return None

    raw = schema.metadata

    def get_str(key: bytes) -> str | None:
        val = raw.get(key)
        return val.decode("utf-8") if val is not None else None

    model_name = get_str(b"model_name")
    if not model_name:
        return None

    dim_str = get_str(b"embedding_dimension")
    dim = int(dim_str) if dim_str else 384
    metric = get_str(b"distance_metric") or "cosine"
    norm_str = get_str(b"normalized")
    normalized = norm_str.lower() == "true" if norm_str else True
    rev = get_str(b"model_revision") or None

    return IndexMetadata(
        model_name=model_name,
        model_revision=rev if rev else None,
        embedding_dimension=dim,
        distance_metric=metric,
        normalized=normalized,
        created_at=get_str(b"created_at") or "",
        updated_at=get_str(b"updated_at") or "",
    )


def validate_schema_compatibility(
    schema: pa.Schema,
    expected_metadata: IndexMetadata,
) -> None:
    """Verify that an existing LanceDB table schema is compatible with the active embedding model.

    Raises:
        IncompatibleModelError: If model_name, dimension, metric, or normalization mismatch.
    """
    actual_metadata = extract_index_metadata(schema)
    if actual_metadata is None:
        logger.warning(
            "Existing LanceDB table schema has no embedded IndexMetadata. "
            "Proceeding with dimension check only."
        )
        # Fallback: check vector field dimension
        vec_field = schema.field("vector")
        vec_type = vec_field.type
        actual_dim = getattr(vec_type, "list_size", None)
        if actual_dim is not None and actual_dim != expected_metadata.embedding_dimension:
            raise IncompatibleModelError(
                expected_model=f"{expected_metadata.model_name} (dim={expected_metadata.embedding_dimension})",
                actual_model=f"unknown (dim={actual_dim})",
                detail=f"Vector dimension mismatch: expected {expected_metadata.embedding_dimension}, found {actual_dim}",
            )
        return

    # Check model name
    if actual_metadata.model_name != expected_metadata.model_name:
        raise IncompatibleModelError(
            expected_model=expected_metadata.model_name,
            actual_model=actual_metadata.model_name,
            detail="Embedding model mismatch",
        )

    # Check dimension
    if actual_metadata.embedding_dimension != expected_metadata.embedding_dimension:
        raise IncompatibleModelError(
            expected_model=f"{expected_metadata.model_name} (dim={expected_metadata.embedding_dimension})",
            actual_model=f"{actual_metadata.model_name} (dim={actual_metadata.embedding_dimension})",
            detail="Vector dimensionality mismatch",
        )

    # Check distance metric
    if actual_metadata.distance_metric != expected_metadata.distance_metric:
        raise IncompatibleModelError(
            expected_model=f"metric={expected_metadata.distance_metric}",
            actual_model=f"metric={actual_metadata.distance_metric}",
            detail="Distance metric mismatch",
        )

    # Check normalization
    if actual_metadata.normalized != expected_metadata.normalized:
        raise IncompatibleModelError(
            expected_model=f"normalized={expected_metadata.normalized}",
            actual_model=f"normalized={actual_metadata.normalized}",
            detail="Normalization configuration mismatch",
        )

    # Check revision if expected specifies one
    if (
        expected_metadata.model_revision
        and actual_metadata.model_revision
        and expected_metadata.model_revision != actual_metadata.model_revision
    ):
        raise IncompatibleModelError(
            expected_model=f"rev={expected_metadata.model_revision}",
            actual_model=f"rev={actual_metadata.model_revision}",
            detail="Model revision mismatch",
        )


def validate_record_vector(
    record: VectorRecord,
    expected_dimension: int,
    check_normalized: bool = True,
    tolerance: float = 1e-2,
) -> None:
    """Validate vector length, finite values, and unit normalization before writing to LanceDB."""
    vec = record.vector
    if len(vec) != expected_dimension:
        raise ValueError(
            f"Record '{record.record_id}' vector length ({len(vec)}) does not match "
            f"expected dimension ({expected_dimension})."
        )

    arr = np.array(vec, dtype=np.float32)
    if not np.all(np.isfinite(arr)):
        nan_count = int(np.isnan(arr).sum())
        inf_count = int(np.isinf(arr).sum())
        raise ValueError(
            f"Record '{record.record_id}' contains non-finite numbers ({nan_count} NaNs, {inf_count} Infs)."
        )

    if check_normalized:
        norm = float(np.linalg.norm(arr))
        if abs(norm - 1.0) > tolerance:
            raise ValueError(
                f"Record '{record.record_id}' vector is not unit normalized (norm={norm:.4f})."
            )


def records_to_arrow_table(records: list[VectorRecord], schema: pa.Schema) -> pa.Table:
    """Convert a list of VectorRecord instances into a PyArrow Table adhering to schema."""
    if not records:
        return schema.empty_table()

    data: dict[str, list[Any]] = {
        "record_id": [],
        "chunk_id": [],
        "vector": [],
        "text": [],
        "source_file": [],
        "source_pages": [],
        "source_unit_ids": [],
        "section_title": [],
        "content_type": [],
        "fragment_index": [],
        "total_fragments": [],
        "metadata_json": [],
        "created_at": [],
    }

    for r in records:
        data["record_id"].append(r.record_id)
        data["chunk_id"].append(r.chunk_id)
        data["vector"].append(r.vector)
        data["text"].append(r.text)
        data["source_file"].append(r.source_file)
        data["source_pages"].append(r.source_pages)
        data["source_unit_ids"].append(r.source_unit_ids)
        data["section_title"].append(r.section_title)
        data["content_type"].append(r.content_type)
        data["fragment_index"].append(r.fragment_index)
        data["total_fragments"].append(r.total_fragments)
        data["metadata_json"].append(json.dumps(r.metadata))
        data["created_at"].append(r.created_at)

    return pa.Table.from_pydict(data, schema=schema)
