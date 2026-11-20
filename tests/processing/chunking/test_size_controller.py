import pytest

from academic_agent.processing.chunking.models import (
    ChunkingConfig,
    ContentType,
    ContentUnit,
)
from academic_agent.processing.chunking.semantic_grouper import SemanticGroup
from academic_agent.processing.chunking.size_controller import SizeController
from academic_agent.processing.chunking.tokenizer import WhitespaceTokenizer


class TestSizeController:
    def test_control_size_empty_list(self):
        controller = SizeController()
        assert controller.control_size([]) == []

    def test_single_group_within_target_size(self):
        controller = SizeController(ChunkingConfig(target_words=50, max_words=100, min_words=10))
        unit = ContentUnit(
            unit_id="u1",
            document_id="doc1",
            source_file="doc1.pdf",
            page_number=1,
            content_type=ContentType.PARAGRAPH,
            text="This is a short paragraph well within the target word budget.",
        )
        group = SemanticGroup(
            group_id="g1",
            document_id="doc1",
            source_file="doc1.pdf",
            units=[unit],
        )

        drafts = controller.control_size([group])
        assert len(drafts) == 1
        assert drafts[0].document_id == "doc1"
        assert drafts[0].word_count == 11
        assert drafts[0].source_pages == [1]
        assert drafts[0].source_unit_ids == ["u1"]

    def test_merge_small_groups_under_same_section(self):
        controller = SizeController(ChunkingConfig(target_words=30, max_words=50, min_words=10, overlap_words=10))
        u1 = ContentUnit(
            unit_id="u1",
            document_id="doc1",
            source_file="doc1.pdf",
            page_number=1,
            content_type=ContentType.PARAGRAPH,
            text="First short paragraph about forecasting methods.",
            section_title="Intro",
        )
        u2 = ContentUnit(
            unit_id="u2",
            document_id="doc1",
            source_file="doc1.pdf",
            page_number=2,
            content_type=ContentType.PARAGRAPH,
            text="Second short paragraph explaining why data matters.",
            section_title="Intro",
        )

        g1 = SemanticGroup(group_id="g1", document_id="doc1", source_file="doc1.pdf", section_title="Intro", units=[u1])
        g2 = SemanticGroup(group_id="g2", document_id="doc1", source_file="doc1.pdf", section_title="Intro", units=[u2])

        drafts = controller.control_size([g1, g2])
        assert len(drafts) == 1
        assert drafts[0].source_pages == [1, 2]
        assert drafts[0].source_unit_ids == ["u1", "u2"]
        assert drafts[0].content_type == ContentType.PARAGRAPH
        assert "First short paragraph" in drafts[0].text
        assert "Second short paragraph" in drafts[0].text

    def test_do_not_merge_across_different_sections(self):
        controller = SizeController(ChunkingConfig(target_words=50, max_words=100, min_words=10))
        u1 = ContentUnit(
            unit_id="u1",
            document_id="doc1",
            source_file="doc1.pdf",
            page_number=1,
            content_type=ContentType.PARAGRAPH,
            text="Text for section A.",
            section_title="Section A",
        )
        u2 = ContentUnit(
            unit_id="u2",
            document_id="doc1",
            source_file="doc1.pdf",
            page_number=1,
            content_type=ContentType.PARAGRAPH,
            text="Text for section B.",
            section_title="Section B",
        )

        g1 = SemanticGroup(group_id="g1", document_id="doc1", source_file="doc1.pdf", section_title="Section A", units=[u1])
        g2 = SemanticGroup(group_id="g2", document_id="doc1", source_file="doc1.pdf", section_title="Section B", units=[u2])

        drafts = controller.control_size([g1, g2])
        assert len(drafts) == 2
        assert drafts[0].section_title == "Section A"
        assert drafts[1].section_title == "Section B"

    def test_split_oversized_paragraph_at_sentences(self):
        # target=15 words, max=25 words, min=5 words, overlap=0
        controller = SizeController(ChunkingConfig(target_words=15, max_words=25, min_words=5, overlap_words=0))
        long_text = (
            "Sentence one discusses exponential smoothing. "
            "Sentence two explains the alpha parameter value. "
            "Sentence three highlights the importance of initial values. "
            "Sentence four concludes the lecture topic."
        )
        unit = ContentUnit(
            unit_id="u1",
            document_id="doc1",
            source_file="doc1.pdf",
            page_number=5,
            content_type=ContentType.PARAGRAPH,
            text=long_text,
        )
        group = SemanticGroup(group_id="g1", document_id="doc1", source_file="doc1.pdf", units=[unit])

        drafts = controller.control_size([group])
        assert len(drafts) > 1
        for d in drafts:
            assert d.word_count <= 25
            assert d.source_pages == [5]
            assert d.source_unit_ids == ["u1"]

    def test_split_oversized_table_with_repeated_headers(self):
        # Each row has a few words; set small max_words so it splits across rows
        controller = SizeController(ChunkingConfig(target_words=10, max_words=16, min_words=4, overlap_words=0))
        table_md = (
            "| Model | Accuracy |\n"
            "| --- | --- |\n"
            "| Simple Exp | 85 percent |\n"
            "| Holt Winters | 92 percent |\n"
            "| ARIMA Model | 95 percent |"
        )
        unit = ContentUnit(
            unit_id="t1",
            document_id="doc1",
            source_file="doc1.pdf",
            page_number=10,
            content_type=ContentType.TABLE,
            text=table_md,
        )
        group = SemanticGroup(group_id="g1", document_id="doc1", source_file="doc1.pdf", units=[unit])

        drafts = controller.control_size([group])
        assert len(drafts) >= 2
        for d in drafts:
            assert d.content_type == ContentType.TABLE
            assert "| Model | Accuracy |" in d.text
            assert "| --- | --- |" in d.text
            assert d.metadata.get("forced_split") is True

    def test_sentence_split_preserves_abbreviations(self):
        controller = SizeController()
        text = "Methods like e.g. exponential smoothing work well. Dr. Smith proved it in vol. 4 of the journal."
        sentences = controller.split_sentences(text)
        assert len(sentences) == 2
        assert "e.g." in sentences[0]
        assert "Dr. Smith" in sentences[1]

    def test_small_chunk_reason_tagging(self):
        controller = SizeController(ChunkingConfig(target_words=50, max_words=100, min_words=20))
        unit = ContentUnit(
            unit_id="eq1",
            document_id="doc1",
            source_file="doc1.pdf",
            page_number=1,
            content_type=ContentType.EQUATION,
            text="y = mx + b",
        )
        group = SemanticGroup(group_id="g1", document_id="doc1", source_file="doc1.pdf", units=[unit])
        drafts = controller.control_size([group])
        assert len(drafts) == 1
        assert drafts[0].word_count < 20
        assert drafts[0].metadata.get("small_chunk_reason") == "atomic_equation"
