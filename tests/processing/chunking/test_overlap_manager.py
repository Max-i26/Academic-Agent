import pytest

from academic_agent.processing.chunking.models import (
    ChunkingConfig,
    ContentType,
)
from academic_agent.processing.chunking.overlap_manager import OverlapManager
from academic_agent.processing.chunking.size_controller import DraftChunk


class TestOverlapManager:
    def test_apply_overlap_empty_list(self):
        manager = OverlapManager()
        assert manager.apply_overlap([]) == []

    def test_apply_overlap_disabled_by_config(self):
        cfg = ChunkingConfig(overlap_words=0)
        manager = OverlapManager(cfg)
        chunk = DraftChunk(
            document_id="doc1",
            source_file="doc1.pdf",
            section_title="Intro",
            text="First sentence. Second sentence.",
            source_pages=[1],
            source_unit_ids=["u1"],
            content_type=ContentType.PARAGRAPH,
            word_count=4,
            char_count=32,
            token_count=4,
        )
        assert manager.apply_overlap([chunk, chunk]) == [chunk, chunk]

    def test_apply_overlap_same_section(self):
        cfg = ChunkingConfig(overlap_words=10, max_overlap_words=20)
        manager = OverlapManager(cfg)

        c1 = DraftChunk(
            document_id="doc1",
            source_file="doc1.pdf",
            section_title="Smoothing",
            text="First sentence explains moving averages. Second sentence introduces exponential smoothing.",
            source_pages=[1],
            source_unit_ids=["u1"],
            content_type=ContentType.PARAGRAPH,
            word_count=10,
            char_count=91,
            token_count=10,
        )
        c2 = DraftChunk(
            document_id="doc1",
            source_file="doc1.pdf",
            section_title="Smoothing",
            text="Third sentence defines alpha. Fourth sentence provides the equation.",
            source_pages=[2],
            source_unit_ids=["u2"],
            content_type=ContentType.PARAGRAPH,
            word_count=9,
            char_count=69,
            token_count=9,
        )

        result = manager.apply_overlap([c1, c2])
        assert len(result) == 2

        # First chunk remains clean
        assert result[0].text == c1.text
        assert "has_overlap" not in result[0].metadata

        # Second chunk prepends the second sentence
        assert "Second sentence introduces exponential smoothing." in result[1].text
        assert "Third sentence defines alpha." in result[1].text
        assert result[1].metadata.get("has_overlap") is True
        assert result[1].metadata.get("overlap_word_count") > 0
        assert result[1].source_pages == [1, 2]

    def test_no_overlap_across_different_sections(self):
        cfg = ChunkingConfig(overlap_words=10, max_overlap_words=20)
        manager = OverlapManager(cfg)

        c1 = DraftChunk(
            document_id="doc1",
            source_file="doc1.pdf",
            section_title="Section 1",
            text="Sentence one. Sentence two.",
            source_pages=[1],
            source_unit_ids=["u1"],
            content_type=ContentType.PARAGRAPH,
            word_count=4,
            char_count=27,
            token_count=4,
        )
        c2 = DraftChunk(
            document_id="doc1",
            source_file="doc1.pdf",
            section_title="Section 2",
            text="Sentence three. Sentence four.",
            source_pages=[2],
            source_unit_ids=["u2"],
            content_type=ContentType.PARAGRAPH,
            word_count=4,
            char_count=30,
            token_count=4,
        )

        result = manager.apply_overlap([c1, c2])
        assert len(result) == 2
        assert "has_overlap" not in result[1].metadata
        assert result[1].text == c2.text

    def test_no_overlap_across_different_documents(self):
        cfg = ChunkingConfig(overlap_words=10, max_overlap_words=20)
        manager = OverlapManager(cfg)

        c1 = DraftChunk(
            document_id="docA",
            source_file="docA.pdf",
            section_title="Intro",
            text="Sentence one. Sentence two.",
            source_pages=[1],
            source_unit_ids=["u1"],
            content_type=ContentType.PARAGRAPH,
            word_count=4,
            char_count=27,
            token_count=4,
        )
        c2 = DraftChunk(
            document_id="docB",
            source_file="docB.pdf",
            section_title="Intro",
            text="Sentence three. Sentence four.",
            source_pages=[1],
            source_unit_ids=["u2"],
            content_type=ContentType.PARAGRAPH,
            word_count=4,
            char_count=30,
            token_count=4,
        )

        result = manager.apply_overlap([c1, c2])
        assert len(result) == 2
        assert "has_overlap" not in result[1].metadata
        assert result[1].text == c2.text

    def test_no_overlap_with_table(self):
        cfg = ChunkingConfig(overlap_words=10, max_overlap_words=20)
        manager = OverlapManager(cfg)

        c1 = DraftChunk(
            document_id="doc1",
            source_file="doc1.pdf",
            section_title="Tables",
            text="Sentence one. Sentence two.",
            source_pages=[1],
            source_unit_ids=["u1"],
            content_type=ContentType.PARAGRAPH,
            word_count=4,
            char_count=27,
            token_count=4,
        )
        c2 = DraftChunk(
            document_id="doc1",
            source_file="doc1.pdf",
            section_title="Tables",
            text="| Col1 | Col2 |\n| --- | --- |\n| A | B |",
            source_pages=[1],
            source_unit_ids=["u2"],
            content_type=ContentType.TABLE,
            word_count=11,
            char_count=40,
            token_count=11,
        )

        result = manager.apply_overlap([c1, c2])
        assert len(result) == 2
        assert "has_overlap" not in result[1].metadata
