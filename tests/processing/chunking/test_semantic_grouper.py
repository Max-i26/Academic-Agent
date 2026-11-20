import pytest

from academic_agent.processing.chunking.models import ContentType, ContentUnit
from academic_agent.processing.chunking.semantic_grouper import (
    SemanticGroup,
    SemanticGrouper,
)


class TestSemanticGrouper:
    def test_group_empty_list(self):
        grouper = SemanticGrouper()
        assert grouper.group([]) == []

    def test_group_heading_with_paragraphs_and_equation(self):
        grouper = SemanticGrouper()
        units = [
            ContentUnit(
                unit_id="u1",
                document_id="doc1",
                source_file="doc1.pdf",
                page_number=18,
                content_type=ContentType.HEADING,
                text="Simple Exponential Smoothing",
                section_title="Simple Exponential Smoothing",
            ),
            ContentUnit(
                unit_id="u2",
                document_id="doc1",
                source_file="doc1.pdf",
                page_number=18,
                content_type=ContentType.PARAGRAPH,
                text="Exponential smoothing assigns greater weight to recent observations.",
                section_title="Simple Exponential Smoothing",
            ),
            ContentUnit(
                unit_id="u3",
                document_id="doc1",
                source_file="doc1.pdf",
                page_number=19,
                content_type=ContentType.EQUATION,
                text="F_{t+1} = \\alpha Y_t + (1-\\alpha)F_t",
                section_title="Simple Exponential Smoothing",
            ),
            ContentUnit(
                unit_id="u4",
                document_id="doc1",
                source_file="doc1.pdf",
                page_number=19,
                content_type=ContentType.PARAGRAPH,
                text="Where alpha is the smoothing constant between 0 and 1.",
                section_title="Simple Exponential Smoothing",
            ),
        ]

        groups = grouper.group(units)
        assert len(groups) == 1
        group = groups[0]
        assert group.group_id == "doc1_g0001"
        assert group.section_title == "Simple Exponential Smoothing"
        assert group.source_pages == [18, 19]
        assert group.source_unit_ids == ["u1", "u2", "u3", "u4"]
        assert len(group.units) == 4
        assert "Simple Exponential Smoothing" in group.text
        assert "F_{t+1} = \\alpha Y_t" in group.text
        assert group.word_count > 0

    def test_stacked_headings_kept_together(self):
        grouper = SemanticGrouper()
        units = [
            ContentUnit(
                unit_id="u1",
                document_id="doc1",
                source_file="doc1.pdf",
                page_number=1,
                content_type=ContentType.HEADING,
                text="PART I: TIME SERIES",
                section_title="PART I: TIME SERIES",
            ),
            ContentUnit(
                unit_id="u2",
                document_id="doc1",
                source_file="doc1.pdf",
                page_number=1,
                content_type=ContentType.HEADING,
                text="1.1 Introduction",
                section_title="1.1 Introduction",
            ),
            ContentUnit(
                unit_id="u3",
                document_id="doc1",
                source_file="doc1.pdf",
                page_number=1,
                content_type=ContentType.PARAGRAPH,
                text="Time series data consists of observations taken sequentially over time.",
                section_title="1.1 Introduction",
            ),
        ]

        groups = grouper.group(units)
        assert len(groups) == 1
        assert len(groups[0].units) == 3
        assert groups[0].units[0].text == "PART I: TIME SERIES"
        assert groups[0].units[1].text == "1.1 Introduction"

    def test_new_section_creates_new_group(self):
        grouper = SemanticGrouper()
        units = [
            ContentUnit(
                unit_id="u1",
                document_id="doc1",
                source_file="doc1.pdf",
                page_number=1,
                content_type=ContentType.HEADING,
                text="Section 1",
                section_title="Section 1",
            ),
            ContentUnit(
                unit_id="u2",
                document_id="doc1",
                source_file="doc1.pdf",
                page_number=1,
                content_type=ContentType.PARAGRAPH,
                text="Body text for section 1.",
                section_title="Section 1",
            ),
            ContentUnit(
                unit_id="u3",
                document_id="doc1",
                source_file="doc1.pdf",
                page_number=2,
                content_type=ContentType.HEADING,
                text="Section 2",
                section_title="Section 2",
            ),
            ContentUnit(
                unit_id="u4",
                document_id="doc1",
                source_file="doc1.pdf",
                page_number=2,
                content_type=ContentType.PARAGRAPH,
                text="Body text for section 2.",
                section_title="Section 2",
            ),
        ]

        groups = grouper.group(units)
        assert len(groups) == 2
        assert groups[0].section_title == "Section 1"
        assert groups[1].section_title == "Section 2"

    def test_document_boundary_splits_group(self):
        grouper = SemanticGrouper()
        units = [
            ContentUnit(
                unit_id="u1",
                document_id="docA",
                source_file="docA.pdf",
                page_number=1,
                content_type=ContentType.PARAGRAPH,
                text="Text from doc A.",
            ),
            ContentUnit(
                unit_id="u2",
                document_id="docB",
                source_file="docB.pdf",
                page_number=1,
                content_type=ContentType.PARAGRAPH,
                text="Text from doc B.",
            ),
        ]

        groups = grouper.group(units)
        assert len(groups) == 2
        assert groups[0].document_id == "docA"
        assert groups[1].document_id == "docB"
