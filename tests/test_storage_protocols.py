"""Tests for VectorStore protocols, models, and in-memory mock store."""

import numpy as np
import pytest

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
from academic_agent.storage.mock_store import InMemoryVectorStore


def test_vector_store_protocol_compliance():
    store = InMemoryVectorStore()
    assert isinstance(store, VectorStoreProtocol)


def test_index_metadata_compatibility():
    meta1 = IndexMetadata(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        embedding_dimension=384,
        distance_metric="cosine",
    )
    meta2 = IndexMetadata(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        embedding_dimension=384,
        distance_metric="cosine",
    )
    meta3 = IndexMetadata(
        model_name="BAAI/bge-small-en-v1.5",
        embedding_dimension=384,
        distance_metric="cosine",
    )

    assert meta1.is_compatible_with(meta2) is True
    assert meta1.is_compatible_with(meta3) is False


def test_store_embedding_space_isolation():
    store = InMemoryVectorStore()
    meta_minilm = IndexMetadata(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        embedding_dimension=384,
    )
    store.initialize(meta_minilm)

    # MiniLM check passes
    store.verify_model_compatibility("sentence-transformers/all-MiniLM-L6-v2", 384)

    # BGE check fails with IncompatibleModelError even though dimension is 384!
    with pytest.raises(IncompatibleModelError) as exc:
        store.verify_model_compatibility("BAAI/bge-small-en-v1.5", 384)
    assert "Embedding space mismatch" in str(exc.value)


def test_store_crud_and_search():
    store = InMemoryVectorStore()
    meta = IndexMetadata(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        embedding_dimension=3,
    )
    store.initialize(meta)

    # Add records
    rec1 = VectorRecord(
        record_id="c1",
        chunk_id="c1",
        vector=[1.0, 0.0, 0.0],
        text="Exponential smoothing equation",
        source_file="Lecture4.pdf",
        source_pages=[7],
        section_title="THE SES EQUATION",
    )
    rec2 = VectorRecord(
        record_id="c2",
        chunk_id="c2",
        vector=[0.0, 1.0, 0.0],
        text="Autocorrelation ACF bounds",
        source_file="Lecture7.pdf",
        source_pages=[17],
        section_title="Significance Bounds in ACF",
    )
    rec3 = VectorRecord(
        record_id="c3",
        chunk_id="c3",
        vector=[0.7071, 0.7071, 0.0],
        text="Stationarity and differencing",
        source_file="Lecture6.pdf",
        source_pages=[4],
        section_title="WHY Do We Need Stationarity",
    )

    store.add([rec1, rec2, rec3])
    assert store.count() == 3

    # Get by ID
    fetched = store.get(["c1", "nonexistent"])
    assert fetched[0] is not None
    assert fetched[0].chunk_id == "c1"
    assert fetched[1] is None

    # Search top 2 for query aligned with [1.0, 0.0, 0.0]
    results = store.search(query_vector=[1.0, 0.0, 0.0], top_k=2)
    assert len(results) == 2
    assert results[0].record_id == "c1"
    assert np.isclose(results[0].score, 1.0, atol=1e-3)
    assert results[1].record_id == "c3"

    # Search with metadata filter (only Lecture 7)
    filtered_results = store.search(
        query_vector=[1.0, 1.0, 0.0],
        top_k=5,
        filters=FilterQuery(source_file="Lecture7.pdf"),
    )
    assert len(filtered_results) == 1
    assert filtered_results[0].record_id == "c2"

    # Delete by lecture filter (re-index scenario)
    deleted_count = store.delete_by_filter(FilterQuery(source_file="Lecture4.pdf"))
    assert deleted_count == 1
    assert store.count() == 2
    assert store.get(["c1"])[0] is None
