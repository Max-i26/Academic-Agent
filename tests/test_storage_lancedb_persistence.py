"""Unit and integration tests for LancePersistenceManager."""

import shutil
import tempfile
from pathlib import Path
import lancedb
import pyarrow as pa
import pytest

from academic_agent.storage.models import IndexMetadata
from academic_agent.storage.exceptions import (
    StorageError,
    IncompatibleModelError,
)
from academic_agent.storage.providers.lancedb.persistence import LancePersistenceManager
from academic_agent.storage.providers.lancedb.schema import create_lance_schema


@pytest.fixture
def default_metadata():
    return IndexMetadata(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_revision="rev_101",
        embedding_dimension=384,
        distance_metric="cosine",
        normalized=True,
    )


def test_path_safety_validation(default_metadata):
    # Empty path
    with pytest.raises(ValueError, match="db_path cannot be empty"):
        LancePersistenceManager(db_path="", collection_name="test", expected_metadata=default_metadata)

    # Empty collection name
    with pytest.raises(ValueError, match="collection_name cannot be empty"):
        LancePersistenceManager(db_path="/tmp", collection_name="  ", expected_metadata=default_metadata)

    # Path pointing to regular file
    temp_file = tempfile.NamedTemporaryFile(delete=False)
    temp_file.close()
    try:
        mgr = LancePersistenceManager(db_path=temp_file.name, collection_name="test", expected_metadata=default_metadata)
        with pytest.raises(ValueError, match="points to an existing file"):
            mgr.get_or_create_table()
    finally:
        Path(temp_file.name).unlink(missing_ok=True)


def test_new_directory_and_table_initialization(default_metadata):
    temp_base = tempfile.mkdtemp()
    db_path = Path(temp_base) / "nested" / "sub_dir"
    try:
        mgr = LancePersistenceManager(
            db_path=db_path,
            collection_name="academic_chunks",
            expected_metadata=default_metadata,
        )
        assert not db_path.exists()
        tbl = mgr.get_or_create_table()
        assert db_path.exists()
        assert tbl.name == "academic_chunks"
        assert mgr.is_connected is True
        assert mgr.is_closed is False

        # Metadata sidecar was written
        sidecar = db_path / "academic_chunks.meta.json"
        assert sidecar.is_file()

        # Check metadata query
        meta = mgr.get_index_metadata()
        assert meta is not None
        assert meta.model_name == default_metadata.model_name
        assert meta.embedding_dimension == 384
    finally:
        shutil.rmtree(temp_base, ignore_errors=True)


def test_reopen_compatible_table(default_metadata):
    temp_dir = tempfile.mkdtemp()
    try:
        # Session 1: Create
        mgr1 = LancePersistenceManager(
            db_path=temp_dir,
            collection_name="col1",
            expected_metadata=default_metadata,
        )
        tbl1 = mgr1.get_or_create_table()
        tbl1.add([
            {
                "record_id": "r1",
                "chunk_id": "c1",
                "vector": [1.0] * 384,
                "text": "sample text",
                "source_file": "lec1.pdf",
                "source_pages": [1],
                "source_unit_ids": ["u1"],
                "section_title": "title",
                "content_type": "PARAGRAPH",
                "fragment_index": 0,
                "total_fragments": 1,
                "metadata_json": "{}",
                "created_at": "2026-11-21T00:00:00Z",
            }
        ])
        mgr1.close()
        assert mgr1.is_closed is True

        # Session 2: Reopen with same metadata
        mgr2 = LancePersistenceManager(
            db_path=temp_dir,
            collection_name="col1",
            expected_metadata=default_metadata,
        )
        tbl2 = mgr2.get_or_create_table()
        assert len(tbl2) == 1
        meta2 = mgr2.get_index_metadata()
        assert meta2.model_name == default_metadata.model_name
        mgr2.close()
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_reopen_rejects_incompatible_model(default_metadata):
    temp_dir = tempfile.mkdtemp()
    try:
        # Create with MiniLM
        mgr1 = LancePersistenceManager(
            db_path=temp_dir,
            collection_name="col1",
            expected_metadata=default_metadata,
        )
        mgr1.get_or_create_table()
        mgr1.close()

        # Reopen with BGE -> must raise IncompatibleModelError!
        bge_meta = IndexMetadata(
            model_name="BAAI/bge-small-en-v1.5",
            embedding_dimension=384,
            distance_metric="cosine",
            normalized=True,
        )
        mgr2 = LancePersistenceManager(
            db_path=temp_dir,
            collection_name="col1",
            expected_metadata=bge_meta,
        )
        with pytest.raises(IncompatibleModelError) as exc_info:
            mgr2.get_or_create_table()

        assert "BAAI/bge-small-en-v1.5" in str(exc_info.value)
        assert "sentence-transformers/all-MiniLM-L6-v2" in str(exc_info.value)
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_reopen_rejects_incompatible_dimension(default_metadata):
    temp_dir = tempfile.mkdtemp()
    try:
        mgr1 = LancePersistenceManager(
            db_path=temp_dir,
            collection_name="col1",
            expected_metadata=default_metadata,
        )
        mgr1.get_or_create_table()
        mgr1.close()

        dim_mismatch_meta = IndexMetadata(
            model_name=default_metadata.model_name,
            embedding_dimension=512,  # wrong
            distance_metric="cosine",
            normalized=True,
        )
        mgr2 = LancePersistenceManager(
            db_path=temp_dir,
            collection_name="col1",
            expected_metadata=dim_mismatch_meta,
        )
        with pytest.raises(IncompatibleModelError) as exc_info:
            mgr2.get_or_create_table()

        assert "dim=512" in str(exc_info.value)
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_reopen_rejects_incompatible_metric(default_metadata):
    temp_dir = tempfile.mkdtemp()
    try:
        mgr1 = LancePersistenceManager(
            db_path=temp_dir,
            collection_name="col1",
            expected_metadata=default_metadata,
        )
        mgr1.get_or_create_table()
        mgr1.close()

        metric_mismatch_meta = IndexMetadata(
            model_name=default_metadata.model_name,
            embedding_dimension=384,
            distance_metric="l2",  # wrong
            normalized=True,
        )
        mgr2 = LancePersistenceManager(
            db_path=temp_dir,
            collection_name="col1",
            expected_metadata=metric_mismatch_meta,
        )
        with pytest.raises(IncompatibleModelError) as exc_info:
            mgr2.get_or_create_table()

        assert "metric=l2" in str(exc_info.value)
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_reopen_rejects_corrupted_or_missing_metadata(default_metadata):
    temp_dir = tempfile.mkdtemp()
    try:
        # Manually create table without IndexMetadata in schema or sidecar
        db = lancedb.connect(temp_dir)
        raw_schema = pa.schema([
            pa.field("id", pa.string()),
            pa.field("vector", pa.list_(pa.float32(), 384)),
        ])
        db.create_table("col_corrupted", schema=raw_schema)
        del db

        mgr = LancePersistenceManager(
            db_path=temp_dir,
            collection_name="col_corrupted",
            expected_metadata=default_metadata,
        )
        with pytest.raises(StorageError, match="no embedded IndexMetadata"):
            mgr.get_or_create_table()
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_close_and_reopen_cycle(default_metadata):
    temp_dir = tempfile.mkdtemp()
    try:
        mgr = LancePersistenceManager(
            db_path=temp_dir,
            collection_name="test_cycle",
            expected_metadata=default_metadata,
        )
        tbl1 = mgr.get_or_create_table()
        assert mgr.is_connected is True
        assert mgr.is_closed is False

        mgr.close()
        assert mgr.is_closed is True
        assert mgr.is_connected is False

        tbl2 = mgr.reopen()
        assert mgr.is_connected is True
        assert mgr.is_closed is False
        assert tbl2.name == "test_cycle"
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
