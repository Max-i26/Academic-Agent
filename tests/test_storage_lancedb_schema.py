"""Unit tests for LanceDB PyArrow schema, metadata serialization, and compatibility validation."""

import json
import numpy as np
import pyarrow as pa
import pytest

from academic_agent.storage.models import IndexMetadata, VectorRecord
from academic_agent.storage.exceptions import IncompatibleModelError
from academic_agent.storage.providers.lancedb.schema import (
    create_lance_schema,
    extract_index_metadata,
    validate_schema_compatibility,
    validate_record_vector,
    records_to_arrow_table,
)


@pytest.fixture
def sample_metadata():
    return IndexMetadata(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_revision="main_rev_123",
        embedding_dimension=384,
        distance_metric="cosine",
        normalized=True,
    )


def test_create_lance_schema_types(sample_metadata):
    schema = create_lance_schema(dimension=384, metadata=sample_metadata)

    assert isinstance(schema, pa.Schema)
    assert schema.field("record_id").type == pa.string()
    assert schema.field("chunk_id").type == pa.string()
    assert schema.field("vector").type == pa.list_(pa.float32(), 384)
    assert schema.field("source_pages").type == pa.list_(pa.int32())
    assert schema.field("source_unit_ids").type == pa.list_(pa.string())
    assert schema.field("section_title").nullable is True
    assert schema.field("metadata_json").type == pa.string()


def test_schema_metadata_roundtrip(sample_metadata):
    schema = create_lance_schema(dimension=384, metadata=sample_metadata)
    extracted = extract_index_metadata(schema)

    assert extracted is not None
    assert extracted.model_name == sample_metadata.model_name
    assert extracted.model_revision == sample_metadata.model_revision
    assert extracted.embedding_dimension == 384
    assert extracted.distance_metric == "cosine"
    assert extracted.normalized is True


def test_validate_schema_compatibility_success(sample_metadata):
    schema = create_lance_schema(dimension=384, metadata=sample_metadata)
    # Identical expected metadata must pass without error
    validate_schema_compatibility(schema, sample_metadata)


def test_validate_schema_compatibility_model_mismatch(sample_metadata):
    schema = create_lance_schema(dimension=384, metadata=sample_metadata)

    bge_meta = IndexMetadata(
        model_name="BAAI/bge-small-en-v1.5",
        embedding_dimension=384,
        distance_metric="cosine",
        normalized=True,
    )
    with pytest.raises(IncompatibleModelError) as exc_info:
        validate_schema_compatibility(schema, bge_meta)

    assert "BAAI/bge-small-en-v1.5" in str(exc_info.value)
    assert "sentence-transformers/all-MiniLM-L6-v2" in str(exc_info.value)
    assert "Embedding model mismatch" in str(exc_info.value)


def test_validate_schema_compatibility_dimension_mismatch(sample_metadata):
    schema = create_lance_schema(dimension=384, metadata=sample_metadata)

    wrong_dim_meta = IndexMetadata(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        embedding_dimension=512,
        distance_metric="cosine",
        normalized=True,
    )
    with pytest.raises(IncompatibleModelError) as exc_info:
        validate_schema_compatibility(schema, wrong_dim_meta)

    assert "Vector dimensionality mismatch" in str(exc_info.value)


def test_validate_schema_compatibility_metric_mismatch(sample_metadata):
    schema = create_lance_schema(dimension=384, metadata=sample_metadata)

    wrong_metric_meta = IndexMetadata(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        embedding_dimension=384,
        distance_metric="l2",
        normalized=True,
    )
    with pytest.raises(IncompatibleModelError) as exc_info:
        validate_schema_compatibility(schema, wrong_metric_meta)

    assert "Distance metric mismatch" in str(exc_info.value)


def test_validate_record_vector_valid():
    norm_vec = [1.0, 0.0, 0.0]
    rec = VectorRecord(
        record_id="r1",
        chunk_id="c1",
        vector=norm_vec,
        text="Sample text",
        source_file="lec1.pdf",
        source_pages=[1],
    )
    validate_record_vector(rec, expected_dimension=3, check_normalized=True)


def test_validate_record_vector_wrong_dim():
    rec = VectorRecord(
        record_id="r1",
        chunk_id="c1",
        vector=[1.0, 0.0],
        text="Sample text",
        source_file="lec1.pdf",
        source_pages=[1],
    )
    with pytest.raises(ValueError, match="does not match expected dimension"):
        validate_record_vector(rec, expected_dimension=3)


def test_validate_record_vector_non_finite():
    rec = VectorRecord(
        record_id="r1",
        chunk_id="c1",
        vector=[1.0, float("nan"), 0.0],
        text="Sample text",
        source_file="lec1.pdf",
        source_pages=[1],
    )
    with pytest.raises(ValueError, match="contains non-finite numbers"):
        validate_record_vector(rec, expected_dimension=3)


def test_validate_record_vector_unnormalized():
    rec = VectorRecord(
        record_id="r1",
        chunk_id="c1",
        vector=[2.0, 2.0, 2.0],
        text="Sample text",
        source_file="lec1.pdf",
        source_pages=[1],
    )
    with pytest.raises(ValueError, match="vector is not unit normalized"):
        validate_record_vector(rec, expected_dimension=3, check_normalized=True)


def test_records_to_arrow_table(sample_metadata):
    schema = create_lance_schema(dimension=3, metadata=sample_metadata)
    r1 = VectorRecord(
        record_id="r1",
        chunk_id="c1",
        vector=[1.0, 0.0, 0.0],
        text="First passage",
        source_file="lec1.pdf",
        source_pages=[1, 2],
        source_unit_ids=["u1", "u2"],
        section_title="Intro",
        content_type="PARAGRAPH",
        metadata={"priority": "high"},
    )
    r2 = VectorRecord(
        record_id="r2",
        chunk_id="c2",
        vector=[0.0, 1.0, 0.0],
        text="Second passage",
        source_file="lec2.pdf",
        source_pages=[3],
        source_unit_ids=["u3"],
        section_title="Methods",
        content_type="TABLE",
        metadata={"priority": "low"},
    )

    table = records_to_arrow_table([r1, r2], schema=schema)
    assert isinstance(table, pa.Table)
    assert table.num_rows == 2
    assert table.column("record_id").to_pylist() == ["r1", "r2"]
    assert table.column("source_pages").to_pylist() == [[1, 2], [3]]
    parsed_meta = json.loads(table.column("metadata_json").to_pylist()[0])
    assert parsed_meta == {"priority": "high"}
