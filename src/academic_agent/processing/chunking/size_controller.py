from __future__ import annotations

import re
from typing import Any, Sequence

from pydantic import BaseModel, Field

from academic_agent.processing.chunking.models import (
    ChunkingConfig,
    ContentType,
    ContentUnit,
)
from academic_agent.processing.chunking.semantic_grouper import SemanticGroup
from academic_agent.processing.chunking.tokenizer import (
    TokenizerProtocol,
    WhitespaceTokenizer,
)


class DraftChunk(BaseModel):
    """
    Intermediate chunk structure after size controlling and before overlap resolution.
    """

    document_id: str
    source_file: str
    section_title: str | None = None
    text: str
    source_pages: list[int]
    source_unit_ids: list[str]
    content_type: ContentType
    word_count: int = Field(ge=0)
    char_count: int = Field(ge=0)
    token_count: int = Field(ge=0)
    metadata: dict[str, Any] = Field(default_factory=dict)


class SizeController:
    """
    Controls and manages chunk sizes using structure-aware recursive splitting.

    Priorities:
    1. Section boundaries
    2. Paragraph / ContentUnit boundaries
    3. Sentence boundaries (for oversized paragraphs)
    4. Safe word boundaries (only if a single sentence exceeds maximum limit)
    5. Table row-level splitting with repeated column headers
    """

    ABBREVIATIONS = (
        r"\be\.g\.",
        r"\bi\.e\.",
        r"\bet al\.",
        r"\bfig\.",
        r"\bfigs\.",
        r"\beq\.",
        r"\beqs\.",
        r"\btab\.",
        r"\btabs\.",
        r"\bvol\.",
        r"\bno\.",
        r"\bpp\.",
        r"\bvs\.",
        r"\bdr\.",
        r"\bprof\.",
        r"\bdept\.",
    )

    ABBR_COMBINED_PATTERN = re.compile(
        "|".join(ABBREVIATIONS),
        re.IGNORECASE,
    )

    def __init__(
        self,
        config: ChunkingConfig | None = None,
        tokenizer: TokenizerProtocol | None = None,
    ) -> None:
        self.config = config or ChunkingConfig()
        self.tokenizer = tokenizer or WhitespaceTokenizer()

    def _get_size(self, text: str) -> int:
        """Measure size according to configured word or token budget."""
        if self.config.target_tokens is not None:
            return self.tokenizer.count_tokens(text)
        return len(text.split())

    @property
    def target_size(self) -> int:
        return self.config.target_tokens or self.config.target_words

    @property
    def max_size(self) -> int:
        return self.config.max_tokens or self.config.max_words

    @property
    def min_size(self) -> int:
        return self.config.min_tokens or self.config.min_words

    def control_size(self, groups: Sequence[SemanticGroup]) -> list[DraftChunk]:
        """Convert SemanticGroups into size-constrained DraftChunks."""
        if not groups:
            return []

        draft_chunks: list[DraftChunk] = []

        current_units: list[ContentUnit] = []
        current_section: str | None = None
        current_doc: str | None = None
        current_file: str | None = None

        def _flush_current() -> None:
            nonlocal current_units
            if not current_units:
                return

            chunk = self._build_draft_chunk(
                units=current_units,
                doc_id=current_doc or "",
                source_file=current_file or "",
                section_title=current_section,
            )
            draft_chunks.append(chunk)
            current_units = []

        for group in groups:
            # Document or section boundary forces flush of pending accumulated units
            if (
                current_doc is not None
                and (group.document_id != current_doc or group.section_title != current_section)
            ):
                _flush_current()

            current_doc = group.document_id
            current_file = group.source_file
            current_section = group.section_title

            group_size = self._get_size(group.text)

            # Case 1: Entire group fits comfortably
            if group_size <= self.max_size:
                tentative_text = "\n\n".join(
                    u.text for u in current_units + group.units if u.text.strip()
                )
                tentative_size = self._get_size(tentative_text)

                # Merge with current pending units if within target size
                if current_units and tentative_size <= self.target_size:
                    current_units.extend(group.units)
                else:
                    _flush_current()
                    current_units = list(group.units)
                continue

            # Case 2: Group exceeds max_size, process units individually
            _flush_current()
            for unit in group.units:
                unit_size = self._get_size(unit.text)

                if unit_size <= self.max_size:
                    tentative_text = "\n\n".join(
                        u.text for u in current_units + [unit] if u.text.strip()
                    )
                    tentative_size = self._get_size(tentative_text)

                    if current_units and tentative_size > self.target_size:
                        _flush_current()

                    current_units.append(unit)
                else:
                    # Unit exceeds max_size by itself: split recursively
                    _flush_current()
                    sub_chunks = self._split_oversized_unit(unit)
                    draft_chunks.extend(sub_chunks)

        _flush_current()
        return draft_chunks

    def _split_oversized_unit(self, unit: ContentUnit) -> list[DraftChunk]:
        """Split a single oversized unit (table or paragraph) safely."""
        if unit.content_type == ContentType.TABLE:
            return self._split_oversized_table(unit)
        return self._split_oversized_text(unit)

    def _split_oversized_table(self, unit: ContentUnit) -> list[DraftChunk]:
        """Split a large table at row boundaries, repeating header lines."""
        lines = [line.strip() for line in unit.text.splitlines() if line.strip()]
        if len(lines) <= 2:
            # Cannot meaningfully split header only
            return [self._build_single_unit_chunk(unit, forced_split=True)]

        header_lines = lines[:2]
        header_text = "\n".join(header_lines)
        header_size = self._get_size(header_text)
        data_rows = lines[2:]

        table_chunks: list[DraftChunk] = []
        current_rows: list[str] = []

        for row in data_rows:
            test_text = "\n".join(header_lines + current_rows + [row])
            if current_rows and self._get_size(test_text) > self.max_size:
                chunk_text = "\n".join(header_lines + current_rows)
                table_chunks.append(
                    self._create_split_chunk(
                        unit=unit,
                        text=chunk_text,
                        content_type=ContentType.TABLE,
                        forced_split=True,
                    )
                )
                current_rows = [row]
            else:
                current_rows.append(row)

        if current_rows:
            chunk_text = "\n".join(header_lines + current_rows)
            table_chunks.append(
                self._create_split_chunk(
                    unit=unit,
                    text=chunk_text,
                    content_type=ContentType.TABLE,
                    forced_split=True,
                )
            )

        return table_chunks

    def _split_oversized_text(self, unit: ContentUnit) -> list[DraftChunk]:
        """Split an oversized paragraph at sentence boundaries with fallback to words."""
        sentences = self.split_sentences(unit.text)
        if not sentences:
            return [self._build_single_unit_chunk(unit, forced_split=True)]

        chunks: list[DraftChunk] = []
        current_sentences: list[str] = []

        for sentence in sentences:
            sentence_size = self._get_size(sentence)

            # If a single sentence exceeds max_size, split at words
            if sentence_size > self.max_size:
                if current_sentences:
                    chunk_text = " ".join(current_sentences)
                    chunks.append(
                        self._create_split_chunk(
                            unit=unit,
                            text=chunk_text,
                            content_type=unit.content_type,
                            forced_split=True,
                        )
                    )
                    current_sentences = []

                word_chunks = self._split_sentence_by_words(sentence, unit)
                chunks.extend(word_chunks)
                continue

            test_text = " ".join(current_sentences + [sentence])
            if current_sentences and self._get_size(test_text) > self.target_size:
                chunk_text = " ".join(current_sentences)
                chunks.append(
                    self._create_split_chunk(
                        unit=unit,
                        text=chunk_text,
                        content_type=unit.content_type,
                        forced_split=True,
                    )
                )
                current_sentences = [sentence]
            else:
                current_sentences.append(sentence)

        if current_sentences:
            chunk_text = " ".join(current_sentences)
            chunks.append(
                self._create_split_chunk(
                    unit=unit,
                    text=chunk_text,
                    content_type=unit.content_type,
                    forced_split=True,
                )
            )

        return chunks

    def _split_sentence_by_words(self, sentence: str, unit: ContentUnit) -> list[DraftChunk]:
        """Emergency fallback: split an excessively long sentence by word budget."""
        words = sentence.split()
        chunks: list[DraftChunk] = []
        current_words: list[str] = []

        for word in words:
            if current_words and len(current_words) >= self.target_size:
                chunk_text = " ".join(current_words)
                chunks.append(
                    self._create_split_chunk(
                        unit=unit,
                        text=chunk_text,
                        content_type=unit.content_type,
                        forced_split=True,
                    )
                )
                current_words = [word]
            else:
                current_words.append(word)

        if current_words:
            chunk_text = " ".join(current_words)
            chunks.append(
                self._create_split_chunk(
                    unit=unit,
                    text=chunk_text,
                    content_type=unit.content_type,
                    forced_split=True,
                )
            )

        return chunks

    def split_sentences(self, text: str) -> list[str]:
        """Split text into sentences while respecting common academic abbreviations."""
        normalized = text.strip()
        if not normalized:
            return []

        # Replace dots inside abbreviations with private-use unicode character \uE000
        temp = self.ABBR_COMBINED_PATTERN.sub(
            lambda m: m.group(0).replace(".", "\uE000"),
            normalized,
        )
        # Protect decimal numbers like 3.14
        temp = re.sub(r"(\d+)\.(\d+)", lambda m: f"{m.group(1)}\uE001{m.group(2)}", temp)

        raw_parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])", temp)

        restored: list[str] = []
        for part in raw_parts:
            s_clean = part.replace("\uE000", ".").replace("\uE001", ".").strip()
            if s_clean:
                restored.append(s_clean)

        return restored

    def _build_draft_chunk(
        self,
        units: list[ContentUnit],
        doc_id: str,
        source_file: str,
        section_title: str | None,
    ) -> DraftChunk:
        """Construct a DraftChunk from accumulated ContentUnits."""
        text = "\n\n".join(u.text.strip() for u in units if u.text.strip())
        source_pages = sorted({u.page_number for u in units if u.page_number >= 1})
        source_unit_ids = [u.unit_id for u in units]

        # Determine dominant content type
        types = {u.content_type for u in units}
        if len(types) == 1:
            dominant_type = next(iter(types))
        else:
            dominant_type = ContentType.MIXED

        word_count = len(text.split())
        char_count = len(text)
        token_count = self.tokenizer.count_tokens(text)

        return DraftChunk(
            document_id=doc_id,
            source_file=source_file,
            section_title=section_title,
            text=text,
            source_pages=source_pages,
            source_unit_ids=source_unit_ids,
            content_type=dominant_type,
            word_count=word_count,
            char_count=char_count,
            token_count=token_count,
            metadata={"units_count": len(units)},
        )

    def _create_split_chunk(
        self,
        unit: ContentUnit,
        text: str,
        content_type: ContentType,
        forced_split: bool,
    ) -> DraftChunk:
        """Create a DraftChunk for a fragment resulting from recursive splitting."""
        word_count = len(text.split())
        char_count = len(text)
        token_count = self.tokenizer.count_tokens(text)

        metadata = dict(unit.metadata)
        metadata["forced_split"] = forced_split

        return DraftChunk(
            document_id=unit.document_id,
            source_file=unit.source_file,
            section_title=unit.section_title,
            text=text,
            source_pages=[unit.page_number],
            source_unit_ids=[unit.unit_id],
            content_type=content_type,
            word_count=word_count,
            char_count=char_count,
            token_count=token_count,
            metadata=metadata,
        )

    def _build_single_unit_chunk(self, unit: ContentUnit, forced_split: bool = False) -> DraftChunk:
        """Wrap an individual unit into a DraftChunk."""
        return self._create_split_chunk(
            unit=unit,
            text=unit.text,
            content_type=unit.content_type,
            forced_split=forced_split,
        )
