import pytest

from academic_agent.processing.chunking.models import ContentType
from academic_agent.processing.chunking.provenance import ProvenanceTracker
from academic_agent.processing.chunking.size_controller import DraftChunk


class TestProvenanceTracker:
    def test_finalize_empty_list(self):
        tracker = ProvenanceTracker()
        assert tracker.finalize_chunks([]) == []

    def test_finalize_sequential_chunks(self):
        tracker = ProvenanceTracker()
        d1 = DraftChunk(
            document_id="doc1",
            source_file="doc1.pdf",
            section_title="Intro",
            text="First chunk text.",
            source_pages=[1, 2],
            source_unit_ids=["u1", "u2", "u1"],  # Has duplicate to test deduplication
            content_type=ContentType.PARAGRAPH,
            word_count=3,
            char_count=17,
            token_count=3,
        )
        d2 = DraftChunk(
            document_id="doc1",
            source_file="doc1.pdf",
            section_title="Intro",
            text="Second chunk text.",
            source_pages=[2, 3],
            source_unit_ids=["u3"],
            content_type=ContentType.PARAGRAPH,
            word_count=3,
            char_count=18,
            token_count=3,
            metadata={"overlap_word_count": 2},
        )

        records = tracker.finalize_chunks([d1, d2])
        assert len(records) == 2

        assert records[0].chunk_index == 0
        assert records[0].document_id == "doc1"
        assert len(records[0].chunk_id) == 16
        assert records[0].source_pages == [1, 2]
        assert records[0].source_unit_ids == ["u1", "u2"]
        assert records[0].overlap_word_count == 0

        assert records[1].chunk_index == 1
        assert records[1].document_id == "doc1"
        assert len(records[1].chunk_id) == 16
        assert records[1].chunk_id != records[0].chunk_id
        assert records[1].source_pages == [2, 3]
        assert records[1].source_unit_ids == ["u3"]
        assert records[1].overlap_word_count == 2
