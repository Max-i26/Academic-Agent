import pytest

from academic_agent.ingestion.layout_extractor import (
    PageLayoutRecord,
    TableRecord,
    TextBlockRecord,
    TextLineRecord,
    TextSpanRecord,
)
from academic_agent.ingestion.pdf_extractor import PageRecord
from academic_agent.processing.chunking.models import ContentType
from academic_agent.processing.chunking.normalizer import (
    DocumentNormalizer,
    NormalizationError,
)


class TestDocumentNormalizer:
    def test_normalize_empty_sequence(self):
        normalizer = DocumentNormalizer()
        assert normalizer.normalize([]) == []

    def test_normalize_unsupported_type_raises(self):
        normalizer = DocumentNormalizer()
        with pytest.raises(NormalizationError, match="Unsupported record type"):
            normalizer.normalize(["invalid_string"])  # type: ignore

    def test_normalize_page_record(self):
        normalizer = DocumentNormalizer()
        record = PageRecord(
            document_id="lecture01",
            source_file="lecture01.pdf",
            page_number=1,
            page_count=10,
            text="First paragraph of lecture.\n\nSecond paragraph explaining models.",
            char_count=65,
            has_text=True,
        )

        units = normalizer.normalize([record])
        assert len(units) == 2
        assert units[0].unit_id == "lecture01_p1_u0001"
        assert units[0].document_id == "lecture01"
        assert units[0].page_number == 1
        assert units[0].content_type == ContentType.UNKNOWN
        assert units[0].text == "First paragraph of lecture."
        assert units[1].text == "Second paragraph explaining models."

    def test_normalize_empty_page_record(self):
        normalizer = DocumentNormalizer()
        record = PageRecord(
            document_id="lecture01",
            source_file="lecture01.pdf",
            page_number=2,
            page_count=10,
            text="   \n\n  ",
            char_count=6,
            has_text=False,
        )
        assert normalizer.normalize([record]) == []

    def test_normalize_layout_record_with_blocks_and_tables(self):
        normalizer = DocumentNormalizer()

        span1 = TextSpanRecord(
            text="Measuring Accuracy",
            font="Helvetica-Bold",
            size=18.0,
            bbox=(50.0, 50.0, 200.0, 70.0),
            origin=(50.0, 68.0),
        )
        line1 = TextLineRecord(text="Measuring Accuracy", bbox=(50.0, 50.0, 200.0, 70.0), spans=[span1])
        block1 = TextBlockRecord(
            block_number=0,
            text="Measuring Accuracy\n",
            bbox=(50.0, 50.0, 200.0, 70.0),
            lines=[line1],
        )

        table = TableRecord(
            table_number=0,
            bbox=(50.0, 100.0, 400.0, 250.0),
            row_count=3,
            column_count=2,
            cells=[
                ["Metric", "Formula"],
                ["MAE", "mean(|e_t|)"],
                ["MSE", "mean(e_t^2)"],
            ],
        )

        span2 = TextSpanRecord(
            text="Lower values indicate better performance.",
            font="Helvetica",
            size=11.0,
            bbox=(50.0, 300.0, 350.0, 315.0),
            origin=(50.0, 312.0),
        )
        line2 = TextLineRecord(
            text="Lower values indicate better performance.",
            bbox=(50.0, 300.0, 350.0, 315.0),
            spans=[span2],
        )
        block2 = TextBlockRecord(
            block_number=1,
            text="Lower values indicate better performance.\n",
            bbox=(50.0, 300.0, 350.0, 315.0),
            lines=[line2],
        )

        # Deliberately unordered in the input
        layout = PageLayoutRecord(
            document_id="lecture04",
            source_file="lecture04.pdf",
            page_number=18,
            page_count=30,
            page_width=595.0,
            page_height=842.0,
            text_blocks=[block2, block1],
            tables=[table],
        )

        units = normalizer.normalize([layout])
        assert len(units) == 3

        # Must be ordered top-to-bottom: block1 (y=50), table (y=100), block2 (y=300)
        assert units[0].text == "Measuring Accuracy"
        assert units[0].metadata["font_size"] == 18.0
        assert units[0].metadata["fonts"] == ["Helvetica-Bold"]

        assert units[1].content_type == ContentType.TABLE
        assert "| Metric | Formula |" in units[1].text
        assert "| MAE | mean(|e_t|) |" in units[1].text
        assert units[1].metadata["row_count"] == 3

        assert units[2].text == "Lower values indicate better performance."
        assert units[2].metadata["font_size"] == 11.0
