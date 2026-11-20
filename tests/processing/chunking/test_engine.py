import pytest

from academic_agent.ingestion.layout_extractor import (
    PageLayoutRecord,
    TableRecord,
    TextBlockRecord,
    TextLineRecord,
    TextSpanRecord,
)
from academic_agent.ingestion.pdf_extractor import PageRecord
from academic_agent.processing.chunking.engine import ChunkingEngine
from academic_agent.processing.chunking.models import (
    ChunkingConfig,
    ContentType,
)


class TestChunkingEngine:
    def test_chunk_empty_input(self):
        engine = ChunkingEngine()
        assert engine.chunk([]) == []

    def test_chunk_single_page_record(self):
        engine = ChunkingEngine()
        record = PageRecord(
            document_id="lec01",
            source_file="lec01.pdf",
            page_number=1,
            page_count=5,
            text=(
                "MEASURING ACCURACY\n\n"
                "MAE = mean(|e_t|)\n\n"
                "MAE is simple to interpret as it measures the average magnitude of errors.\n\n"
                "MSE = mean(e_t^2)\n\n"
                "MSE penalizes larger errors more heavily."
            ),
            char_count=180,
            has_text=True,
        )

        chunks = engine.chunk(record)
        assert len(chunks) >= 1
        assert chunks[0].document_id == "lec01"
        assert chunks[0].chunk_index == 0
        assert chunks[0].source_pages == [1]
        assert len(chunks[0].chunk_id) == 16
        assert "MAE = mean(|e_t|)" in chunks[0].text

        report = engine.get_validation_report(chunks)
        assert report.total_chunks == len(chunks)
        assert report.document_count == 1
        assert report.total_words > 0

    def test_chunk_multi_page_layout_record(self):
        engine = ChunkingEngine(ChunkingConfig(target_words=20, max_words=35, min_words=5, overlap_words=0))

        # Page 1: Heading and Paragraph
        span1 = TextSpanRecord(
            text="1. Exponential Smoothing",
            font="Helvetica-Bold",
            size=18.0,
            bbox=(50.0, 50.0, 300.0, 75.0),
            origin=(50.0, 70.0),
        )
        line1 = TextLineRecord(text="1. Exponential Smoothing", bbox=(50.0, 50.0, 300.0, 75.0), spans=[span1])
        block1 = TextBlockRecord(block_number=0, text="1. Exponential Smoothing\n", bbox=(50.0, 50.0, 300.0, 75.0), lines=[line1])

        span2 = TextSpanRecord(
            text="Exponential smoothing is a popular forecasting method for univariate time series data.",
            font="Helvetica",
            size=11.0,
            bbox=(50.0, 100.0, 450.0, 120.0),
            origin=(50.0, 115.0),
        )
        line2 = TextLineRecord(text="Exponential smoothing is a popular forecasting method for univariate time series data.", bbox=(50.0, 100.0, 450.0, 120.0), spans=[span2])
        block2 = TextBlockRecord(block_number=1, text="Exponential smoothing is a popular forecasting method for univariate time series data.\n", bbox=(50.0, 100.0, 450.0, 120.0), lines=[line2])

        p1 = PageLayoutRecord(
            document_id="lec02",
            source_file="lec02.pdf",
            page_number=1,
            page_count=2,
            page_width=595.0,
            page_height=842.0,
            text_blocks=[block1, block2],
        )

        # Page 2: Table
        table = TableRecord(
            table_number=0,
            bbox=(50.0, 100.0, 400.0, 250.0),
            row_count=3,
            column_count=2,
            cells=[
                ["Parameter", "Description"],
                ["Alpha", "Level smoothing weight"],
                ["Beta", "Trend smoothing weight"],
            ],
        )
        p2 = PageLayoutRecord(
            document_id="lec02",
            source_file="lec02.pdf",
            page_number=2,
            page_count=2,
            page_width=595.0,
            page_height=842.0,
            tables=[table],
        )

        chunks = engine.chunk([p1, p2])
        assert len(chunks) >= 2

        # Verify page references
        pages_seen = set()
        for c in chunks:
            pages_seen.update(c.source_pages)
        assert 1 in pages_seen
        assert 2 in pages_seen

        # Verify table representation
        table_chunks = [c for c in chunks if c.content_type == ContentType.TABLE]
        assert len(table_chunks) >= 1
        assert "| Parameter | Description |" in table_chunks[0].text

    def test_chunking_deterministic_id_reproducibility(self):
        engine = ChunkingEngine()
        record = PageRecord(
            document_id="lec03",
            source_file="lec03.pdf",
            page_number=1,
            page_count=1,
            text="Deterministic testing chunk content for Academic-Agent.",
            char_count=55,
            has_text=True,
        )

        run1 = engine.chunk(record)
        run2 = engine.chunk(record)

        assert len(run1) == len(run2)
        for c1, c2 in zip(run1, run2):
            assert c1.chunk_id == c2.chunk_id
            assert c1.chunk_index == c2.chunk_index
            assert c1.text == c2.text
