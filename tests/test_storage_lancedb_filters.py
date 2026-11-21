"""Unit and integration tests for LanceDB filter builder."""

import shutil
import tempfile
import lancedb
import pyarrow as pa
import pytest

from academic_agent.storage.models import FilterQuery
from academic_agent.storage.providers.lancedb.filters import (
    build_lance_filter,
    ALLOWED_FILTER_FIELDS,
)
from academic_agent.storage.providers.lancedb.schema import create_lance_schema


def test_empty_filter():
    assert build_lance_filter(None) is None
    assert build_lance_filter(FilterQuery()) is None
    assert build_lance_filter({}) is None


def test_single_field_filter():
    fq = FilterQuery(source_file="4. DS5104 Lecture4.pdf")
    pred = build_lance_filter(fq)
    assert pred == "(source_file = '4. DS5104 Lecture4.pdf')"


def test_multiple_fields_and_composition():
    fq = FilterQuery(
        source_file="Lecture4.pdf",
        content_type="TABLE",
        section_title="RMSE",
    )
    pred = build_lance_filter(fq)
    assert "(source_file = 'Lecture4.pdf')" in pred
    assert "(content_type = 'TABLE')" in pred
    assert "(section_title = 'RMSE')" in pred
    assert " AND " in pred


def test_apostrophe_escaping():
    fq = FilterQuery(source_file="Dr. O'Reilly's Time Series.pdf")
    pred = build_lance_filter(fq)
    assert pred == "(source_file = 'Dr. O''Reilly''s Time Series.pdf')"


def test_unicode_preservation():
    fq = FilterQuery(section_title="Formula for parameter α and variance σ²")
    pred = build_lance_filter(fq)
    assert "parameter α and variance σ²" in pred


def test_page_list_membership():
    fq = FilterQuery(source_pages=[18, 7, 7, 12])
    pred = build_lance_filter(fq)
    assert pred == "(array_has_any(source_pages, [7, 12, 18]))"


def test_empty_page_list():
    fq = FilterQuery(source_pages=[])
    pred = build_lance_filter(fq)
    assert pred is None


def test_unsupported_field_rejection():
    with pytest.raises(ValueError, match="Unsupported filter field"):
        build_lance_filter({"malicious_column": "value"})

    with pytest.raises(ValueError, match="Unsupported filter field"):
        build_lance_filter(FilterQuery(custom_filters={"unregistered_col": 123}))


def test_invalid_types_rejection():
    # source_pages must be list of ints, not booleans in dict input
    with pytest.raises(TypeError, match="must be an integer"):
        build_lance_filter({"source_pages": [True, False]})

    # source_pages non-int string in dict input
    with pytest.raises(TypeError, match="must be an integer"):
        build_lance_filter({"source_pages": ["page_1"]})

    # invalid filter container
    with pytest.raises(TypeError, match="Expected FilterQuery or dict"):
        build_lance_filter(12345)


def test_sql_injection_resistance():
    # SQL injection attempt via quotation breakout
    injection = "test' OR 1=1 --"
    fq = FilterQuery(source_file=injection)
    pred = build_lance_filter(fq)
    # Must be safely escaped into a literal string without breaking syntax
    assert pred == "(source_file = 'test'' OR 1=1 --')"

    # Null byte injection rejection
    with pytest.raises(ValueError, match="Null bytes are not permitted"):
        build_lance_filter(FilterQuery(source_file="test\0injected.pdf"))


def test_lancedb_live_table_filtering():
    """Verify generated filters execute correctly and accurately against an actual LanceDB table."""
    temp_dir = tempfile.mkdtemp(prefix="lance_filter_test_")
    try:
        db = lancedb.connect(temp_dir)
        schema = create_lance_schema(dimension=2)
        tbl = db.create_table("test_table", schema=schema)

        records = [
            {
                "record_id": "r1",
                "chunk_id": "c1",
                "vector": [1.0, 0.0],
                "text": "Text from lecture 4",
                "source_file": "Lecture4.pdf",
                "source_pages": [10, 11],
                "source_unit_ids": ["u1"],
                "section_title": "RMSE",
                "content_type": "TABLE",
                "fragment_index": 0,
                "total_fragments": 1,
                "metadata_json": "{}",
                "created_at": "2026-11-21T00:00:00Z",
            },
            {
                "record_id": "r2",
                "chunk_id": "c2",
                "vector": [0.0, 1.0],
                "text": "Text from lecture 5",
                "source_file": "Lecture's 5.pdf",
                "source_pages": [15],
                "source_unit_ids": ["u2"],
                "section_title": "Decomposition",
                "content_type": "PARAGRAPH",
                "fragment_index": 0,
                "total_fragments": 1,
                "metadata_json": "{}",
                "created_at": "2026-11-21T00:00:00Z",
            },
            {
                "record_id": "r3",
                "chunk_id": "c3",
                "vector": [0.7071, 0.7071],
                "text": "Text from lecture 7",
                "source_file": "Lecture7.pdf",
                "source_pages": [11, 25],
                "source_unit_ids": ["u3"],
                "section_title": "ACF bounds",
                "content_type": "PARAGRAPH",
                "fragment_index": 0,
                "total_fragments": 1,
                "metadata_json": "{}",
                "created_at": "2026-11-21T00:00:00Z",
            },
        ]
        tbl.add(records)

        # 1. Single-field filter
        f1 = build_lance_filter(FilterQuery(source_file="Lecture4.pdf"))
        r1 = tbl.search([1.0, 0.0]).where(f1).to_list()
        assert len(r1) == 1 and r1[0]["record_id"] == "r1"

        # 2. Apostrophe escaping filter on real LanceDB
        f2 = build_lance_filter(FilterQuery(source_file="Lecture's 5.pdf"))
        r2 = tbl.search([0.0, 1.0]).where(f2).to_list()
        assert len(r2) == 1 and r2[0]["record_id"] == "r2"

        # 3. Array page membership filter (pages 11, 20 -> should match r1 and r3)
        f3 = build_lance_filter(FilterQuery(source_pages=[11, 20]))
        r3 = tbl.search([1.0, 0.0]).where(f3).to_list()
        ids = sorted([item["record_id"] for item in r3])
        assert ids == ["r1", "r3"]

        # 4. Multi-field AND filter (source_file and content_type)
        f4 = build_lance_filter(FilterQuery(source_file="Lecture4.pdf", content_type="TABLE"))
        r4 = tbl.search([1.0, 0.0]).where(f4).to_list()
        assert len(r4) == 1 and r4[0]["record_id"] == "r1"

        # 5. Multi-field AND filter that matches nothing
        f5 = build_lance_filter(FilterQuery(source_file="Lecture4.pdf", content_type="EQUATION"))
        r5 = tbl.search([1.0, 0.0]).where(f5).to_list()
        assert len(r5) == 0

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
