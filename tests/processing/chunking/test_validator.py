import pytest

from academic_agent.processing.chunking.models import (
    ChunkingConfig,
    ChunkRecord,
    ContentType,
)
from academic_agent.processing.chunking.validator import (
    ChunkValidationError,
    ChunkValidator,
)


class TestChunkValidator:
    def test_validate_empty_list(self):
        validator = ChunkValidator()
        report = validator.validate([])
        assert report.total_chunks == 0
        assert report.total_words == 0

    def test_validate_valid_chunks(self):
        validator = ChunkValidator()
        text = "Sample chunk text for testing."
        chunk_id = ChunkRecord.generate_chunk_id("doc1", 0, text)
        chunk = ChunkRecord(
            chunk_id=chunk_id,
            document_id="doc1",
            source_file="doc1.pdf",
            chunk_index=0,
            text=text,
            source_pages=[1],
            source_unit_ids=["u1"],
            word_count=len(text.split()),
            char_count=len(text),
            estimated_token_count=6,
            overlap_word_count=0,
            content_type=ContentType.PARAGRAPH,
        )

        report = validator.validate([chunk])
        assert report.total_chunks == 1
        assert report.total_words == 5
        assert report.document_count == 1
        assert report.overlap_ratio == 0.0

    def test_duplicate_chunk_id_raises(self):
        validator = ChunkValidator()
        text = "Sample text."
        chunk_id = ChunkRecord.generate_chunk_id("doc1", 0, text)
        c1 = ChunkRecord(
            chunk_id=chunk_id,
            document_id="doc1",
            source_file="doc1.pdf",
            chunk_index=0,
            text=text,
            source_pages=[1],
            word_count=2,
            char_count=len(text),
            estimated_token_count=2,
            content_type=ContentType.PARAGRAPH,
        )
        c2 = ChunkRecord(
            chunk_id=chunk_id,  # Same chunk_id!
            document_id="doc1",
            source_file="doc1.pdf",
            chunk_index=1,
            text=text,
            source_pages=[1],
            word_count=2,
            char_count=len(text),
            estimated_token_count=2,
            content_type=ContentType.PARAGRAPH,
        )
        with pytest.raises(ChunkValidationError, match="Duplicate chunk_id"):
            validator.validate([c1, c2])

    def test_non_sequential_chunk_index_raises(self):
        validator = ChunkValidator()
        text1 = "First chunk."
        text2 = "Second chunk."
        c1 = ChunkRecord(
            chunk_id=ChunkRecord.generate_chunk_id("doc1", 0, text1),
            document_id="doc1",
            source_file="doc1.pdf",
            chunk_index=0,
            text=text1,
            source_pages=[1],
            word_count=2,
            char_count=len(text1),
            estimated_token_count=2,
            content_type=ContentType.PARAGRAPH,
        )
        c2 = ChunkRecord(
            chunk_id=ChunkRecord.generate_chunk_id("doc1", 5, text2),
            document_id="doc1",
            source_file="doc1.pdf",
            chunk_index=5,  # Skipped from 0 to 5
            text=text2,
            source_pages=[1],
            word_count=2,
            char_count=len(text2),
            estimated_token_count=2,
            content_type=ContentType.PARAGRAPH,
        )
        with pytest.raises(ChunkValidationError, match="Non-sequential chunk_index"):
            validator.validate([c1, c2])

    def test_chunk_exceeding_max_words_raises(self):
        validator = ChunkValidator()
        cfg = ChunkingConfig(target_words=4, max_words=5, min_words=2, overlap_words=0)
        text = "This chunk contains definitely more than five words."
        c1 = ChunkRecord(
            chunk_id=ChunkRecord.generate_chunk_id("doc1", 0, text),
            document_id="doc1",
            source_file="doc1.pdf",
            chunk_index=0,
            text=text,
            source_pages=[1],
            word_count=len(text.split()),
            char_count=len(text),
            estimated_token_count=10,
            content_type=ContentType.PARAGRAPH,
        )
        with pytest.raises(ChunkValidationError, match="exceeds maximum word limit"):
            validator.validate([c1], config=cfg)
