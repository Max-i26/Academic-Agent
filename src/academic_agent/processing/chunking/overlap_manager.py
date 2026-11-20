from __future__ import annotations

from typing import Sequence

from academic_agent.processing.chunking.models import (
    ChunkingConfig,
    ContentType,
)
from academic_agent.processing.chunking.size_controller import (
    DraftChunk,
    SizeController,
)
from academic_agent.processing.chunking.tokenizer import (
    TokenizerProtocol,
    WhitespaceTokenizer,
)


class OverlapManager:
    """
    Applies sentence-bounded, section-aware contextual overlap to adjacent DraftChunks.

    Rules:
    - Only overlap within the same logical section and document.
    - Prefer complete sentences, never slicing words or incomplete thoughts.
    - Respect configured overlap word / token budgets.
    - Avoid duplicating entire tables or mixing table syntax into prose.
    - Explicitly tracks overlap_word_count and overlap metadata.
    """

    def __init__(
        self,
        config: ChunkingConfig | None = None,
        tokenizer: TokenizerProtocol | None = None,
        size_controller: SizeController | None = None,
    ) -> None:
        self.config = config or ChunkingConfig()
        self.tokenizer = tokenizer or WhitespaceTokenizer()
        self.size_controller = size_controller or SizeController(self.config, self.tokenizer)

    @property
    def max_budget(self) -> int:
        return self.config.max_tokens or self.config.max_words

    @property
    def target_overlap(self) -> int:
        return self.config.overlap_tokens or self.config.overlap_words

    @property
    def max_overlap(self) -> int:
        return self.config.max_overlap_tokens or self.config.max_overlap_words

    def _get_size(self, text: str) -> int:
        if self.config.target_tokens is not None:
            return self.tokenizer.count_tokens(text)
        return len(text.split())

    def apply_overlap(self, draft_chunks: Sequence[DraftChunk]) -> list[DraftChunk]:
        """Apply contextual overlap to a sequence of DraftChunks."""
        if not draft_chunks:
            return []

        if self.target_overlap <= 0:
            return list(draft_chunks)

        overlapped_chunks: list[DraftChunk] = []

        for idx, current in enumerate(draft_chunks):
            if idx == 0:
                overlapped_chunks.append(current)
                continue

            previous = draft_chunks[idx - 1]

            # 1. Do not overlap across different documents or sections
            if (
                current.document_id != previous.document_id
                or current.section_title != previous.section_title
            ):
                overlapped_chunks.append(current)
                continue

            # 2. Do not overlap tables
            if (
                current.content_type == ContentType.TABLE
                or previous.content_type == ContentType.TABLE
            ):
                overlapped_chunks.append(current)
                continue

            # 3. Extract sentence-bounded overlap from previous chunk
            overlap_text = self._extract_tail_overlap(previous.text)
            if not overlap_text:
                overlapped_chunks.append(current)
                continue

            # 4. Enforce authoritative token/size ceiling: overlap must fit within remaining budget headroom
            current_size = self._get_size(current.text)
            remaining_headroom = self.max_budget - current_size
            if remaining_headroom <= 0:
                # Chunk already fills or exceeds the size budget; omit overlap to prevent budget breach
                overlapped_chunks.append(current)
                continue

            # If overlap exceeds available headroom, trim sentences from the front
            overlap_sentences = self.size_controller.split_sentences(overlap_text)
            fitted_sentences: list[str] = []
            accumulated_overlap = 0

            for s in reversed(overlap_sentences):
                s_size = self._get_size(s)
                if (
                    accumulated_overlap + s_size <= remaining_headroom
                    and accumulated_overlap + s_size <= self.max_overlap
                ):
                    fitted_sentences.insert(0, s)
                    accumulated_overlap += s_size
                else:
                    break

            if not fitted_sentences:
                overlapped_chunks.append(current)
                continue

            overlap_text = " ".join(fitted_sentences)

            # 5. Combine overlap with current chunk text
            new_text = f"{overlap_text}\n\n{current.text}"
            overlap_words = len(overlap_text.split())
            new_word_count = len(new_text.split())
            new_char_count = len(new_text)
            new_token_count = self.tokenizer.count_tokens(new_text)

            # Merge and sort source pages
            merged_pages = sorted(set(current.source_pages + previous.source_pages[-1:]))

            metadata = dict(current.metadata)
            metadata["has_overlap"] = True
            metadata["overlap_word_count"] = overlap_words
            metadata["overlap_text"] = overlap_text

            overlapped = DraftChunk(
                document_id=current.document_id,
                source_file=current.source_file,
                section_title=current.section_title,
                text=new_text,
                source_pages=merged_pages,
                source_unit_ids=current.source_unit_ids,
                content_type=current.content_type,
                word_count=new_word_count,
                char_count=new_char_count,
                token_count=new_token_count,
                metadata=metadata,
            )
            overlapped_chunks.append(overlapped)

        return overlapped_chunks

    def _extract_tail_overlap(self, text: str) -> str:
        """
        Extract complete sentences from the tail of text up to the overlap budget.
        """
        sentences = self.size_controller.split_sentences(text)
        if len(sentences) <= 1:
            # Avoid duplicating an entire single-sentence chunk
            return ""

        chosen_sentences: list[str] = []
        accumulated_size = 0

        # Walk backwards from last sentence
        for s in reversed(sentences):
            s_size = self._get_size(s)
            if chosen_sentences and (accumulated_size + s_size > self.max_overlap):
                break

            if not chosen_sentences and s_size > self.max_overlap:
                # Even single sentence exceeds max overlap, skip to avoid excessive duplication
                break

            chosen_sentences.insert(0, s)
            accumulated_size += s_size

            if accumulated_size >= self.target_overlap:
                break

        # If chosen sentences constitute the entire previous text, avoid complete duplication
        if len(chosen_sentences) == len(sentences):
            chosen_sentences = chosen_sentences[1:]

        return " ".join(chosen_sentences)
