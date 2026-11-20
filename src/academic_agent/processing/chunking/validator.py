from __future__ import annotations

from typing import Sequence

from pydantic import BaseModel, Field

from academic_agent.processing.chunking.models import (
    ChunkingConfig,
    ChunkRecord,
)


class ChunkValidationError(Exception):
    """Raised when one or more ChunkRecords fail integrity validation."""


class ValidationReport(BaseModel):
    """Integrity and distribution summary for a collection of ChunkRecords."""

    total_chunks: int = Field(ge=0)
    total_words: int = Field(ge=0)
    total_chars: int = Field(ge=0)
    avg_words_per_chunk: float = Field(ge=0.0)
    min_words: int = Field(ge=0)
    max_words: int = Field(ge=0)
    total_overlap_words: int = Field(ge=0)
    overlap_ratio: float = Field(ge=0.0, le=1.0)
    document_count: int = Field(ge=0)
    forced_splits_count: int = Field(ge=0)


class ChunkValidator:
    """
    Validates structural, academic, and content integrity of generated ChunkRecords.
    """

    def validate(
        self,
        chunks: Sequence[ChunkRecord],
        config: ChunkingConfig | None = None,
    ) -> ValidationReport:
        """
        Validate ChunkRecords against integrity rules and budget limits.
        
        Raises ChunkValidationError upon detecting invalid records.
        Returns a ValidationReport detailing distribution metrics.
        """
        if not chunks:
            return ValidationReport(
                total_chunks=0,
                total_words=0,
                total_chars=0,
                avg_words_per_chunk=0.0,
                min_words=0,
                max_words=0,
                total_overlap_words=0,
                overlap_ratio=0.0,
                document_count=0,
                forced_splits_count=0,
            )

        seen_chunk_ids: set[str] = set()
        doc_indices: dict[str, int] = {}
        total_words = 0
        total_chars = 0
        total_overlap_words = 0
        min_words = float("inf")
        max_words = 0
        forced_splits = 0

        for idx, chunk in enumerate(chunks):
            # 1. Unique chunk identifier check
            if chunk.chunk_id in seen_chunk_ids:
                raise ChunkValidationError(
                    f"Duplicate chunk_id detected: {chunk.chunk_id} at index {idx}"
                )
            seen_chunk_ids.add(chunk.chunk_id)

            # 2. Sequential chunk_index per document
            expected_index = doc_indices.get(chunk.document_id, 0)
            if chunk.chunk_index != expected_index:
                raise ChunkValidationError(
                    f"Non-sequential chunk_index for doc {chunk.document_id}: "
                    f"expected {expected_index}, got {chunk.chunk_index}"
                )
            doc_indices[chunk.document_id] = expected_index + 1

            # 3. Non-empty text
            if not chunk.text or not chunk.text.strip():
                raise ChunkValidationError(f"Empty chunk text at index {idx} ({chunk.chunk_id})")

            # 4. Strict count accuracy
            actual_chars = len(chunk.text)
            if chunk.char_count != actual_chars:
                raise ChunkValidationError(
                    f"Character count mismatch in {chunk.chunk_id}: "
                    f"stored {chunk.char_count}, actual {actual_chars}"
                )

            actual_words = len(chunk.text.split())
            if chunk.word_count != actual_words:
                raise ChunkValidationError(
                    f"Word count mismatch in {chunk.chunk_id}: "
                    f"stored {chunk.word_count}, actual {actual_words}"
                )

            # 5. Strictly ascending unique pages >= 1
            if not chunk.source_pages:
                raise ChunkValidationError(f"Missing source_pages in chunk {chunk.chunk_id}")
            if any(p < 1 for p in chunk.source_pages):
                raise ChunkValidationError(f"Invalid non-positive page number in {chunk.chunk_id}")
            if chunk.source_pages != sorted(set(chunk.source_pages)):
                raise ChunkValidationError(f"Unsorted or duplicate source_pages in {chunk.chunk_id}")

            # 6. Budget boundary adherence
            if config is not None:
                is_forced = chunk.metadata.get("forced_split", False)
                if not is_forced and chunk.word_count > config.max_words:
                    raise ChunkValidationError(
                        f"Chunk {chunk.chunk_id} exceeds maximum word limit: "
                        f"{chunk.word_count} > {config.max_words}"
                    )

            if chunk.metadata.get("forced_split"):
                forced_splits += 1

            total_words += actual_words
            total_chars += actual_chars
            total_overlap_words += chunk.overlap_word_count
            min_words = min(min_words, actual_words)
            max_words = max(max_words, actual_words)

        overlap_ratio = (total_overlap_words / total_words) if total_words > 0 else 0.0

        return ValidationReport(
            total_chunks=len(chunks),
            total_words=total_words,
            total_chars=total_chars,
            avg_words_per_chunk=round(total_words / len(chunks), 2),
            min_words=int(min_words) if min_words != float("inf") else 0,
            max_words=int(max_words),
            total_overlap_words=total_overlap_words,
            overlap_ratio=round(overlap_ratio, 4),
            document_count=len(doc_indices),
            forced_splits_count=forced_splits,
        )
