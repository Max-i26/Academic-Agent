from __future__ import annotations

from enum import Enum
import hashlib
from typing import Any, Self

from pydantic import BaseModel, Field, field_validator, model_validator


class ContentType(str, Enum):
    """Classification of academic document content units."""

    PARAGRAPH = "paragraph"
    HEADING = "heading"
    EQUATION = "equation"
    TABLE = "table"
    LIST = "list"
    MIXED = "mixed"
    UNKNOWN = "unknown"


class ContentUnit(BaseModel):
    """
    Normalized atomic unit of academic content.

    Serves as the standardized internal representation between extraction
    and structural chunking.
    """

    unit_id: str
    document_id: str
    source_file: str
    page_number: int = Field(ge=1)
    content_type: ContentType = ContentType.UNKNOWN
    text: str
    block_number: int | None = None
    section_title: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("unit_id")
    @classmethod
    def validate_unit_id(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("unit_id must not be empty")
        return v.strip()

    @field_validator("text")
    @classmethod
    def validate_text(cls, v: str) -> str:
        if not isinstance(v, str):
            raise TypeError("text must be a string")
        return v


class ChunkRecord(BaseModel):
    """
    A structured, validated academic knowledge chunk ready for retrieval.

    Preserves content, provenance, boundaries, and metadata across
    academic document sources.
    """

    chunk_id: str
    document_id: str
    source_file: str
    chunk_index: int = Field(ge=0)
    text: str
    source_pages: list[int]
    source_unit_ids: list[str] = Field(default_factory=list)
    word_count: int = Field(ge=0)
    char_count: int = Field(ge=0)
    estimated_token_count: int = Field(ge=0)
    overlap_word_count: int = Field(default=0, ge=0)
    section_title: str | None = None
    content_type: ContentType
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("text")
    @classmethod
    def validate_non_empty_text(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Chunk text cannot be empty or whitespace only")
        return v

    @field_validator("source_pages")
    @classmethod
    def validate_source_pages(cls, pages: list[int]) -> list[int]:
        if not pages:
            raise ValueError("source_pages cannot be empty")
        for p in pages:
            if p < 1:
                raise ValueError(f"All page numbers must be >= 1, got {p}")
        if pages != sorted(set(pages)):
            raise ValueError("source_pages must be sorted in strictly ascending order without duplicates")
        return pages

    @model_validator(mode="after")
    def validate_counts_and_boundaries(self) -> Self:
        expected_chars = len(self.text)
        if self.char_count != expected_chars:
            raise ValueError(f"char_count ({self.char_count}) does not match len(text) ({expected_chars})")
        if self.overlap_word_count > self.word_count:
            raise ValueError(
                f"overlap_word_count ({self.overlap_word_count}) cannot exceed word_count ({self.word_count})"
            )
        return self

    @staticmethod
    def generate_chunk_id(document_id: str, chunk_index: int, text: str) -> str:
        """
        Generate a deterministic 16-character SHA-256 chunk identifier.
        """
        normalized_text = text.strip()
        payload = f"{document_id}:{chunk_index}:{normalized_text}"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


class ChunkingConfig(BaseModel):
    """
    Configuration budget and rules for the Chunking Engine.
    """

    target_words: int = Field(default=350, gt=0)
    max_words: int = Field(default=500, gt=0)
    min_words: int = Field(default=80, gt=0)
    overlap_words: int = Field(default=50, ge=0)
    max_overlap_words: int = Field(default=75, ge=0)

    # Optional token-level limits when specific tokenizer is configured
    target_tokens: int | None = Field(default=None, gt=0)
    max_tokens: int | None = Field(default=None, gt=0)
    min_tokens: int | None = Field(default=None, gt=0)
    overlap_tokens: int | None = Field(default=None, ge=0)
    max_overlap_tokens: int | None = Field(default=None, ge=0)

    preserve_headings: bool = True
    preserve_equations: bool = True
    preserve_bullet_lists: bool = True
    preserve_tables: bool = True
    split_long_paragraphs: bool = True
    strategy: str = "recursive"

    @model_validator(mode="after")
    def validate_word_budget(self) -> Self:
        if self.max_words < self.target_words:
            raise ValueError(f"max_words ({self.max_words}) must be >= target_words ({self.target_words})")
        if self.min_words > self.target_words:
            raise ValueError(f"min_words ({self.min_words}) must be <= target_words ({self.target_words})")
        if self.overlap_words >= self.max_words:
            raise ValueError(f"overlap_words ({self.overlap_words}) must be < max_words ({self.max_words})")
        if self.max_overlap_words < self.overlap_words:
            raise ValueError(
                f"max_overlap_words ({self.max_overlap_words}) must be >= overlap_words ({self.overlap_words})"
            )

        # Validate token constraints if provided
        if self.max_tokens is not None and self.target_tokens is not None:
            if self.max_tokens < self.target_tokens:
                raise ValueError(f"max_tokens ({self.max_tokens}) must be >= target_tokens ({self.target_tokens})")
        if self.min_tokens is not None and self.target_tokens is not None:
            if self.min_tokens > self.target_tokens:
                raise ValueError(f"min_tokens ({self.min_tokens}) must be <= target_tokens ({self.target_tokens})")
        if self.overlap_tokens is not None and self.max_tokens is not None:
            if self.overlap_tokens >= self.max_tokens:
                raise ValueError(f"overlap_tokens ({self.overlap_tokens}) must be < max_tokens ({self.max_tokens})")
        if self.max_overlap_tokens is not None and self.overlap_tokens is not None:
            if self.max_overlap_tokens < self.overlap_tokens:
                raise ValueError(
                    f"max_overlap_tokens ({self.max_overlap_tokens}) must be >= overlap_tokens ({self.overlap_tokens})"
                )

        return self
