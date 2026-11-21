"""Tests for StructureAwareFragmenter at embedding time."""

import pytest
from academic_agent.embeddings.fragmenter import StructureAwareFragmenter, EmbeddingFragment
from academic_agent.embeddings.tokenizer import WhitespaceTokenCounter, HuggingFaceTokenCounter
from academic_agent.processing.chunking import ChunkRecord, ContentType


def test_fragmenter_under_limit_unchanged():
    counter = WhitespaceTokenCounter(multiplier=1.0)
    fragmenter = StructureAwareFragmenter(token_counter=counter)

    chunk = ChunkRecord(
        chunk_id="chunk_short",
        document_id="doc_1",
        source_file="lec1.pdf",
        chunk_index=0,
        text="A brief explanation of mean absolute error.",
        source_pages=[10],
        source_unit_ids=["u1"],
        word_count=7,
        char_count=43,
        estimated_token_count=7,
        overlap_word_count=0,
        section_title="MAE",
        content_type=ContentType.PARAGRAPH,
    )

    frags = fragmenter.fragment_chunk(chunk, max_tokens=50)
    assert len(frags) == 1
    assert frags[0].fragment_id == "chunk_short#frag_0"
    assert frags[0].parent_chunk_id == "chunk_short"
    assert frags[0].fragment_index == 0
    assert frags[0].total_fragments == 1
    assert frags[0].metadata["is_fragment"] is False


def test_fragmenter_multi_step_workspace():
    counter = WhitespaceTokenCounter(multiplier=1.0)
    fragmenter = StructureAwareFragmenter(token_counter=counter)

    long_text = (
        "Activity 2: Moving Average Exercise\n\n"
        "Here is the table of sales:\n"
        "| Day | Sales |\n| Mon | 100 |\n\n"
        "CMA Workspace Step 1\n"
        "Calculate 4-MA1 = (100+120)/2 = 110\n\n"
        "CMA Workspace Step 2\n"
        "Now center the averages = 110"
    )

    chunk = ChunkRecord(
        chunk_id="chk_long",
        document_id="doc_1",
        source_file="lec3.pdf",
        chunk_index=5,
        text=long_text,
        source_pages=[30, 31, 32],
        source_unit_ids=["u1", "u2", "u3"],
        word_count=len(long_text.split()),
        char_count=len(long_text),
        estimated_token_count=len(long_text.split()),
        overlap_word_count=0,
        section_title="CMA Exercise",
        content_type=ContentType.MIXED,
    )

    # Set token limit low enough to require splitting
    frags = fragmenter.fragment_chunk(chunk, max_tokens=18)
    assert len(frags) >= 2
    for idx, f in enumerate(frags):
        assert f.parent_chunk_id == "chk_long"
        assert f.fragment_index == idx
        assert f.total_fragments == len(frags)
        assert f.source_pages == [30, 31, 32]
        assert f.metadata["is_fragment"] is True
        assert counter.count_tokens(f.text) <= 25  # bounded
