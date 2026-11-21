"""Shared contract test suite executed identically against InMemoryVectorStore and LanceDBVectorStore."""

import shutil
import tempfile
from pathlib import Path
import numpy as np
import pytest

from academic_agent.storage.models import (
    VectorRecord,
    SearchResult,
    IndexMetadata,
    FilterQuery,
)
from academic_agent.storage.config import VectorStoreConfig
from academic_agent.storage.exceptions import (
    IncompatibleModelError,
    StoreNotInitializedError,
)
from academic_agent.storage.mock_store import InMemoryVectorStore
from academic_agent.storage.providers.lancedb.store import LanceDBVectorStore
from academic_agent.storage.providers.lancedb.persistence import LancePersistenceManager


@pytest.fixture(params=["memory", "lancedb"])
def vector_store(request):
    """Parameterized fixture yielding both InMemoryVectorStore and LanceDBVectorStore."""
    dim = 3
    metadata = IndexMetadata(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_revision="rev_contract_01",
        embedding_dimension=dim,
        distance_metric="cosine",
        normalized=True,
    )

    if request.param == "memory":
        store = InMemoryVectorStore(allow_overwrite=True)
        store.initialize(metadata)
        yield store
        store.close()
    elif request.param == "lancedb":
        temp_dir = tempfile.mkdtemp(prefix="lance_contract_")
        pm = LancePersistenceManager(
            db_path=temp_dir,
            collection_name="contract_chunks",
            expected_metadata=metadata,
        )
        store = LanceDBVectorStore(
            config=VectorStoreConfig(
                persist_path=temp_dir,
                collection_name="contract_chunks",
                expected_dimension=dim,
            ),
            persistence_manager=pm,
            metadata=metadata,
        )
        store.initialize(metadata)
        yield store
        store.close()
        shutil.rmtree(temp_dir, ignore_errors=True)


# --- 1. Initialization and Metadata Compatibility ---

def test_initialization_and_metadata_contract(vector_store):
    meta = vector_store.get_index_metadata()
    assert meta is not None
    assert meta.model_name == "sentence-transformers/all-MiniLM-L6-v2"
    assert meta.embedding_dimension == 3

    # Matching check passes
    vector_store.verify_model_compatibility("sentence-transformers/all-MiniLM-L6-v2", 3)

    # Incompatible model rejected
    with pytest.raises(IncompatibleModelError):
        vector_store.verify_model_compatibility("BAAI/bge-small-en-v1.5", 3)

    # Incompatible dimension rejected
    with pytest.raises(IncompatibleModelError):
        vector_store.verify_model_compatibility("sentence-transformers/all-MiniLM-L6-v2", 512)


# --- 2. Insert and Stable Upsert ---

def test_insert_and_stable_upsert_contract(vector_store):
    r1 = VectorRecord(
        record_id="c1",
        chunk_id="c1",
        vector=[1.0, 0.0, 0.0],
        text="Initial text for c1",
        source_file="Lecture4.pdf",
        source_pages=[7],
        section_title="SES",
    )
    r2 = VectorRecord(
        record_id="c2",
        chunk_id="c2",
        vector=[0.0, 1.0, 0.0],
        text="Text for c2",
        source_file="Lecture5.pdf",
        source_pages=[13],
        section_title="CMA",
    )

    vector_store.add([r1, r2])
    assert vector_store.count() == 2

    # Re-adding c1 with updated text must update in place without duplicating IDs
    r1_updated = VectorRecord(
        record_id="c1",
        chunk_id="c1",
        vector=[0.7071, 0.7071, 0.0],
        text="Updated text for c1",
        source_file="Lecture4.pdf",
        source_pages=[7, 8],
        section_title="SES Updated",
    )
    vector_store.add([r1_updated])
    assert vector_store.count() == 2, "Re-inserting existing record_id must not duplicate"

    fetched = vector_store.get(["c1"])[0]
    assert fetched is not None
    assert fetched.text == "Updated text for c1"
    assert fetched.source_pages == [7, 8]

    # In-batch duplicate: latest record wins
    r_batch_a = VectorRecord(
        record_id="c3",
        chunk_id="c3",
        vector=[0.0, 0.0, 1.0],
        text="Version 1",
        source_file="Lecture6.pdf",
        source_pages=[1],
    )
    r_batch_b = VectorRecord(
        record_id="c3",
        chunk_id="c3",
        vector=[0.0, 0.0, 1.0],
        text="Version 2",
        source_file="Lecture6.pdf",
        source_pages=[1],
    )
    vector_store.add([r_batch_a, r_batch_b])
    assert vector_store.count() == 3
    assert vector_store.get(["c3"])[0].text == "Version 2"

    # Empty add([]) is a safe no-op
    vector_store.add([])
    assert vector_store.count() == 3


# --- 3. Retrieval by ID ---

def test_get_by_id_contract(vector_store):
    records = [
        VectorRecord(record_id=f"r_{i}", chunk_id=f"c_{i}", vector=[1.0, 0.0, 0.0], text=f"Doc {i}", source_file="lec.pdf", source_pages=[i])
        for i in range(4)
    ]
    vector_store.add(records)

    # Missing ID returns None preserving exact order
    fetched = vector_store.get(["r_2", "missing_id", "r_0", "missing_2"])
    assert len(fetched) == 4
    assert fetched[0] is not None and fetched[0].record_id == "r_2"
    assert fetched[1] is None
    assert fetched[2] is not None and fetched[2].record_id == "r_0"
    assert fetched[3] is None

    # Empty get([]) returns empty list
    assert vector_store.get([]) == []


# --- 4. Cosine Similarity Ranking and Top-K ---

def test_search_ranking_and_top_k_contract(vector_store):
    # Empty store search
    assert vector_store.search([1.0, 0.0, 0.0], top_k=5) == []

    # Insert 3 orthogonal unit vectors
    r1 = VectorRecord(record_id="x", chunk_id="x", vector=[1.0, 0.0, 0.0], text="X axis", source_file="axis.pdf", source_pages=[1])
    r2 = VectorRecord(record_id="y", chunk_id="y", vector=[0.0, 1.0, 0.0], text="Y axis", source_file="axis.pdf", source_pages=[2])
    r3 = VectorRecord(record_id="xy", chunk_id="xy", vector=[0.7071, 0.7071, 0.0], text="Diagonal", source_file="axis.pdf", source_pages=[3])
    vector_store.add([r1, r2, r3])

    # Search with query aligned with [1.0, 0.0, 0.0]
    results = vector_store.search([1.0, 0.0, 0.0], top_k=2)
    assert len(results) == 2
    assert results[0].record_id == "x"
    assert np.isclose(results[0].score, 1.0, atol=1e-3)
    assert results[1].record_id == "xy"
    assert np.isclose(results[1].score, 0.7071, atol=1e-2)

    # top_k=0 and negative top_k return empty
    assert vector_store.search([1.0, 0.0, 0.0], top_k=0) == []
    assert vector_store.search([1.0, 0.0, 0.0], top_k=-1) == []


# --- 5. Metadata Filtering ---

def test_metadata_filtering_contract(vector_store):
    r1 = VectorRecord(record_id="r1", chunk_id="c1", vector=[1.0, 0.0, 0.0], text="Table", source_file="Lecture4.pdf", source_pages=[10, 11], content_type="TABLE")
    r2 = VectorRecord(record_id="r2", chunk_id="c2", vector=[0.0, 1.0, 0.0], text="Prose", source_file="Lecture4.pdf", source_pages=[12], content_type="PARAGRAPH")
    r3 = VectorRecord(record_id="r3", chunk_id="c3", vector=[0.0, 0.0, 1.0], text="Apostrophe file", source_file="Lecture's Notes.pdf", source_pages=[11, 20], content_type="TABLE")
    vector_store.add([r1, r2, r3])

    # 1. Filter by source_file
    res_file = vector_store.search([1.0, 0.0, 0.0], top_k=5, filters=FilterQuery(source_file="Lecture4.pdf"))
    assert len(res_file) == 2
    assert {r.record_id for r in res_file} == {"r1", "r2"}

    # 2. Filter with apostrophe in source_file
    res_apos = vector_store.search([1.0, 0.0, 0.0], top_k=5, filters=FilterQuery(source_file="Lecture's Notes.pdf"))
    assert len(res_apos) == 1
    assert res_apos[0].record_id == "r3"

    # 3. Filter by page array membership (match 11 -> matches r1 and r3)
    res_page = vector_store.search([1.0, 0.0, 0.0], top_k=5, filters=FilterQuery(source_pages=[11]))
    assert len(res_page) == 2
    assert {r.record_id for r in res_page} == {"r1", "r3"}

    # 4. Multi-field AND filter
    res_multi = vector_store.search([1.0, 0.0, 0.0], top_k=5, filters=FilterQuery(source_file="Lecture4.pdf", content_type="TABLE"))
    assert len(res_multi) == 1
    assert res_multi[0].record_id == "r1"

    # 5. Empty filter object returns unfiltered results
    res_empty_filter = vector_store.search([1.0, 0.0, 0.0], top_k=5, filters=FilterQuery())
    assert len(res_empty_filter) == 3


# --- 6. Deletion by ID and Filter ---

def test_deletion_contract(vector_store):
    records = [
        VectorRecord(record_id=f"del_{i}", chunk_id=f"c_{i}", vector=[1.0, 0.0, 0.0], text=f"D{i}", source_file="LecA.pdf" if i < 2 else "LecB.pdf", source_pages=[i])
        for i in range(4)
    ]
    vector_store.add(records)
    assert vector_store.count() == 4

    # Delete single ID
    deleted = vector_store.delete(["del_0"])
    assert deleted == 1
    assert vector_store.count() == 3
    assert vector_store.get(["del_0"])[0] is None

    # Delete nonexistent ID
    assert vector_store.delete(["nonexistent"]) == 0
    assert vector_store.count() == 3

    # Empty delete([])
    assert vector_store.delete([]) == 0
    assert vector_store.count() == 3

    # Delete by filter (delete all LecB.pdf records)
    deleted_filter = vector_store.delete_by_filter(FilterQuery(source_file="LecB.pdf"))
    assert deleted_filter == 2
    assert vector_store.count() == 1
    assert vector_store.get(["del_1"])[0] is not None


# --- 7. Provenance and Fragment Preservation ---

def test_provenance_and_fragment_preservation(vector_store):
    # Simulate the 3 fragments of the 322-token chunk e9285418782c3cf7
    parent_id = "e9285418782c3cf7"
    frag0 = VectorRecord(
        record_id=f"{parent_id}#frag_0",
        chunk_id=parent_id,
        vector=[1.0, 0.0, 0.0],
        text="Activity 2: Centered Moving Average (Table)",
        source_file="3. DS5104 Lecture3.pdf",
        source_pages=[30, 31, 32],
        source_unit_ids=["u_p30_1"],
        section_title="CMA Activity",
        content_type="MIXED",
        fragment_index=0,
        total_fragments=3,
    )
    frag1 = VectorRecord(
        record_id=f"{parent_id}#frag_1",
        chunk_id=parent_id,
        vector=[0.7071, 0.7071, 0.0],
        text="CMA Workspace Step 1",
        source_file="3. DS5104 Lecture3.pdf",
        source_pages=[30, 31, 32],
        source_unit_ids=["u_p31_1"],
        section_title="CMA Activity",
        content_type="MIXED",
        fragment_index=1,
        total_fragments=3,
    )
    frag2 = VectorRecord(
        record_id=f"{parent_id}#frag_2",
        chunk_id=parent_id,
        vector=[0.0, 1.0, 0.0],
        text="CMA Workspace Step 2",
        source_file="3. DS5104 Lecture3.pdf",
        source_pages=[30, 31, 32],
        source_unit_ids=["u_p32_1"],
        section_title="CMA Activity",
        content_type="MIXED",
        fragment_index=2,
        total_fragments=3,
    )

    vector_store.add([frag0, frag1, frag2])
    assert vector_store.count() == 3

    # Retrieve all 3 fragments by querying parent chunk ID
    res = vector_store.search([1.0, 0.0, 0.0], top_k=5, filters=FilterQuery(chunk_id=parent_id))
    assert len(res) == 3
    for r in res:
        assert r.chunk_id == parent_id
        assert r.total_fragments == 3
        assert r.source_pages == [30, 31, 32]
        assert "#frag_" in r.record_id


# --- 8. Invalid Vector and Record Rejection ---

def test_invalid_vector_rejection(vector_store):
    # Vector length mismatch
    with pytest.raises(ValueError, match="does not match expected dimension"):
        vector_store.add([
            VectorRecord(record_id="bad_dim", chunk_id="c", vector=[1.0, 0.0], text="t", source_file="f", source_pages=[1])
        ])

    # NaN in vector
    with pytest.raises(ValueError, match="non-finite numbers"):
        vector_store.add([
            VectorRecord(record_id="bad_nan", chunk_id="c", vector=[1.0, float("nan"), 0.0], text="t", source_file="f", source_pages=[1])
        ])

    # Unnormalized vector in search query
    with pytest.raises(ValueError, match="unit normalized"):
        vector_store.search([10.0, 10.0, 10.0], top_k=1)


# --- 9. Lifecycle Safety ---

def test_lifecycle_safety(vector_store):
    # Repeated close() must be safe and idempotent
    vector_store.close()
    vector_store.close()

    # Operations after close must raise StoreNotInitializedError
    with pytest.raises(StoreNotInitializedError):
        vector_store.count()

    with pytest.raises(StoreNotInitializedError):
        vector_store.search([1.0, 0.0, 0.0])

    with pytest.raises(StoreNotInitializedError):
        vector_store.add([])
