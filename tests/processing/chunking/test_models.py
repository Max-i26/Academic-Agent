import pytest
from pydantic import ValidationError

from academic_agent.processing.chunking.models import (
    ChunkingConfig,
    ChunkRecord,
    ContentType,
    ContentUnit,
)
from academic_agent.processing.chunking.tokenizer import (
    EstimateTokenizer,
    TokenizerProtocol,
    WhitespaceTokenizer,
)


class TestContentType:
    def test_content_type_values(self):
        assert ContentType.PARAGRAPH.value == "paragraph"
        assert ContentType.HEADING.value == "heading"
        assert ContentType.EQUATION.value == "equation"
        assert ContentType.TABLE.value == "table"
        assert ContentType.LIST.value == "list"
        assert ContentType.MIXED.value == "mixed"
        assert ContentType.UNKNOWN.value == "unknown"


class TestContentUnit:
    def test_valid_content_unit(self):
        unit = ContentUnit(
            unit_id="doc1_p1_u001",
            document_id="doc1",
            source_file="doc1.pdf",
            page_number=1,
            content_type=ContentType.HEADING,
            text="Introduction to Forecasting",
            block_number=0,
            section_title="Introduction",
            metadata={"font_size": 18.0},
        )
        assert unit.unit_id == "doc1_p1_u001"
        assert unit.page_number == 1
        assert unit.content_type == ContentType.HEADING
        assert unit.text == "Introduction to Forecasting"
        assert unit.metadata["font_size"] == 18.0

    def test_empty_unit_id_raises(self):
        with pytest.raises(ValidationError):
            ContentUnit(
                unit_id="   ",
                document_id="doc1",
                source_file="doc1.pdf",
                page_number=1,
                content_type=ContentType.PARAGRAPH,
                text="Some text",
            )

    def test_invalid_page_number_raises(self):
        with pytest.raises(ValidationError):
            ContentUnit(
                unit_id="u1",
                document_id="doc1",
                source_file="doc1.pdf",
                page_number=0,
                text="Some text",
            )


class TestChunkRecord:
    def test_valid_chunk_record(self):
        text = "Exponential smoothing assigns weights to past observations."
        chunk_id = ChunkRecord.generate_chunk_id("doc1", 0, text)
        record = ChunkRecord(
            chunk_id=chunk_id,
            document_id="doc1",
            source_file="doc1.pdf",
            chunk_index=0,
            text=text,
            source_pages=[1, 2],
            source_unit_ids=["u1", "u2"],
            word_count=len(text.split()),
            char_count=len(text),
            estimated_token_count=10,
            overlap_word_count=0,
            section_title="Smoothing Methods",
            content_type=ContentType.PARAGRAPH,
        )
        assert record.chunk_id == chunk_id
        assert record.source_pages == [1, 2]
        assert record.char_count == len(text)
        assert record.word_count == 7

    def test_empty_text_raises(self):
        with pytest.raises(ValidationError):
            ChunkRecord(
                chunk_id="chk1",
                document_id="doc1",
                source_file="doc1.pdf",
                chunk_index=0,
                text="   ",
                source_pages=[1],
                word_count=0,
                char_count=0,
                estimated_token_count=0,
                content_type=ContentType.PARAGRAPH,
            )

    def test_char_count_mismatch_raises(self):
        text = "Hello world"
        with pytest.raises(ValidationError, match="char_count"):
            ChunkRecord(
                chunk_id="chk1",
                document_id="doc1",
                source_file="doc1.pdf",
                chunk_index=0,
                text=text,
                source_pages=[1],
                word_count=2,
                char_count=999,  # Mismatch
                estimated_token_count=2,
                content_type=ContentType.PARAGRAPH,
            )

    def test_invalid_source_pages_ordering_or_duplicates_raises(self):
        text = "Valid text content"
        # Duplicate pages
        with pytest.raises(ValidationError):
            ChunkRecord(
                chunk_id="chk1",
                document_id="doc1",
                source_file="doc1.pdf",
                chunk_index=0,
                text=text,
                source_pages=[1, 1],
                word_count=3,
                char_count=len(text),
                estimated_token_count=3,
                content_type=ContentType.PARAGRAPH,
            )

        # Unsorted pages
        with pytest.raises(ValidationError):
            ChunkRecord(
                chunk_id="chk1",
                document_id="doc1",
                source_file="doc1.pdf",
                chunk_index=0,
                text=text,
                source_pages=[3, 1],
                word_count=3,
                char_count=len(text),
                estimated_token_count=3,
                content_type=ContentType.PARAGRAPH,
            )

    def test_overlap_word_count_exceeding_word_count_raises(self):
        text = "Short text"
        with pytest.raises(ValidationError, match="overlap_word_count"):
            ChunkRecord(
                chunk_id="chk1",
                document_id="doc1",
                source_file="doc1.pdf",
                chunk_index=0,
                text=text,
                source_pages=[1],
                word_count=2,
                char_count=len(text),
                estimated_token_count=2,
                overlap_word_count=5,  # Exceeds word_count
                content_type=ContentType.PARAGRAPH,
            )

    def test_deterministic_chunk_id(self):
        id1 = ChunkRecord.generate_chunk_id("docA", 2, "Sample academic text")
        id2 = ChunkRecord.generate_chunk_id("docA", 2, "Sample academic text")
        id3 = ChunkRecord.generate_chunk_id("docA", 2, "Different text")
        id4 = ChunkRecord.generate_chunk_id("docB", 2, "Sample academic text")
        assert id1 == id2
        assert len(id1) == 16
        assert id1 != id3
        assert id1 != id4


class TestChunkingConfig:
    def test_default_config(self):
        cfg = ChunkingConfig()
        assert cfg.target_words == 350
        assert cfg.max_words == 500
        assert cfg.min_words == 80
        assert cfg.overlap_words == 50
        assert cfg.max_overlap_words == 75
        assert cfg.preserve_headings is True
        assert cfg.preserve_equations is True
        assert cfg.strategy == "recursive"

    def test_invalid_max_words_less_than_target(self):
        with pytest.raises(ValidationError, match="max_words"):
            ChunkingConfig(target_words=400, max_words=300)

    def test_invalid_min_words_greater_than_target(self):
        with pytest.raises(ValidationError, match="min_words"):
            ChunkingConfig(target_words=200, min_words=250, max_words=400)

    def test_invalid_overlap_words_greater_than_max(self):
        with pytest.raises(ValidationError, match="overlap_words"):
            ChunkingConfig(target_words=300, max_words=350, overlap_words=400)

    def test_invalid_max_overlap_less_than_overlap(self):
        with pytest.raises(ValidationError, match="max_overlap_words"):
            ChunkingConfig(overlap_words=60, max_overlap_words=50)

    def test_token_constraints_validation(self):
        with pytest.raises(ValidationError, match="max_tokens"):
            ChunkingConfig(target_tokens=256, max_tokens=200)

        valid_cfg = ChunkingConfig(
            target_tokens=256,
            max_tokens=300,
            min_tokens=64,
            overlap_tokens=32,
            max_overlap_tokens=48,
        )
        assert valid_cfg.target_tokens == 256


class TestTokenizerProtocol:
    def test_whitespace_tokenizer(self):
        tokenizer = WhitespaceTokenizer()
        assert isinstance(tokenizer, TokenizerProtocol)
        text = "Academic Agent for University Lectures"
        assert tokenizer.count_tokens(text) == 5
        assert tokenizer.tokenize(text) == ["Academic", "Agent", "for", "University", "Lectures"]
        assert tokenizer.count_tokens("") == 0
        assert tokenizer.tokenize("") == []

    def test_estimate_tokenizer(self):
        tokenizer = EstimateTokenizer(factor=1.33)
        assert isinstance(tokenizer, TokenizerProtocol)
        text = "Academic Agent for University Lectures"
        # 5 words * 1.33 = 6.65 -> 6
        assert tokenizer.count_tokens(text) == 6
        assert tokenizer.tokenize(text) == ["Academic", "Agent", "for", "University", "Lectures"]

    def test_estimate_tokenizer_invalid_factor(self):
        with pytest.raises(ValueError):
            EstimateTokenizer(factor=-0.5)
