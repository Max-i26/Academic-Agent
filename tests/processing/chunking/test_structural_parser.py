import pytest

from academic_agent.processing.chunking.models import ContentType, ContentUnit
from academic_agent.processing.chunking.structural_parser import StructuralParser


class TestStructuralParser:
    def test_parse_empty_list(self):
        parser = StructuralParser()
        assert parser.parse([]) == []

    def test_detect_headings_and_propagate_section_title(self):
        parser = StructuralParser()

        units = [
            ContentUnit(
                unit_id="u1",
                document_id="doc1",
                source_file="doc1.pdf",
                page_number=1,
                content_type=ContentType.UNKNOWN,
                text="1.1 Moving Averages",
            ),
            ContentUnit(
                unit_id="u2",
                document_id="doc1",
                source_file="doc1.pdf",
                page_number=1,
                content_type=ContentType.UNKNOWN,
                text="A moving average calculates the mean over a rolling window.",
            ),
            ContentUnit(
                unit_id="u3",
                document_id="doc1",
                source_file="doc1.pdf",
                page_number=2,
                content_type=ContentType.UNKNOWN,
                text="EXPONENTIAL SMOOTHING",
            ),
            ContentUnit(
                unit_id="u4",
                document_id="doc1",
                source_file="doc1.pdf",
                page_number=2,
                content_type=ContentType.UNKNOWN,
                text="Alpha controls the influence of the most recent observation.",
            ),
        ]

        parsed = parser.parse(units)

        assert parsed[0].content_type == ContentType.HEADING
        assert parsed[0].section_title == "1.1 Moving Averages"

        assert parsed[1].content_type == ContentType.PARAGRAPH
        assert parsed[1].section_title == "1.1 Moving Averages"

        assert parsed[2].content_type == ContentType.HEADING
        assert parsed[2].section_title == "EXPONENTIAL SMOOTHING"

        assert parsed[3].content_type == ContentType.PARAGRAPH
        assert parsed[3].section_title == "EXPONENTIAL SMOOTHING"

    def test_detect_headings_via_font_size(self):
        parser = StructuralParser()
        units = [
            ContentUnit(
                unit_id="u1",
                document_id="doc1",
                source_file="doc1.pdf",
                page_number=1,
                content_type=ContentType.UNKNOWN,
                text="Introduction to Forecasting Methods",
                metadata={"font_size": 20.0},
            ),
            ContentUnit(
                unit_id="u2",
                document_id="doc1",
                source_file="doc1.pdf",
                page_number=1,
                content_type=ContentType.UNKNOWN,
                text="Forecasting is essential for planning future operations.",
                metadata={"font_size": 11.0},
            ),
            ContentUnit(
                unit_id="u3",
                document_id="doc1",
                source_file="doc1.pdf",
                page_number=1,
                content_type=ContentType.UNKNOWN,
                text="Models can be qualitative or quantitative.",
                metadata={"font_size": 11.0},
            ),
        ]

        parsed = parser.parse(units)
        assert parsed[0].content_type == ContentType.HEADING
        assert parsed[1].content_type == ContentType.PARAGRAPH
        assert parsed[2].content_type == ContentType.PARAGRAPH

    def test_detect_equations(self):
        parser = StructuralParser()
        units = [
            ContentUnit(
                unit_id="u1",
                document_id="doc1",
                source_file="doc1.pdf",
                page_number=1,
                content_type=ContentType.UNKNOWN,
                text="MAE = mean(|e_t|)",
            ),
            ContentUnit(
                unit_id="u2",
                document_id="doc1",
                source_file="doc1.pdf",
                page_number=1,
                content_type=ContentType.UNKNOWN,
                text="$$ F_{t+1} = \\alpha Y_t + (1-\\alpha)F_t $$",
            ),
        ]

        parsed = parser.parse(units)
        assert parsed[0].content_type == ContentType.EQUATION
        assert parsed[1].content_type == ContentType.EQUATION

    def test_detect_bullet_lists(self):
        parser = StructuralParser()
        units = [
            ContentUnit(
                unit_id="u1",
                document_id="doc1",
                source_file="doc1.pdf",
                page_number=1,
                content_type=ContentType.UNKNOWN,
                text="• Sales trend\n• Seasonal variation\n• Cyclical movements",
            ),
        ]

        parsed = parser.parse(units)
        assert parsed[0].content_type == ContentType.LIST

    def test_detect_tables(self):
        parser = StructuralParser()
        units = [
            ContentUnit(
                unit_id="u1",
                document_id="doc1",
                source_file="doc1.pdf",
                page_number=1,
                content_type=ContentType.UNKNOWN,
                text="| Component | Meaning |\n| --- | --- |\n| T | Trend |",
            ),
        ]

        parsed = parser.parse(units)
        assert parsed[0].content_type == ContentType.TABLE
