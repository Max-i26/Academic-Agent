"""Structure-aware embedding fragmenter for handling oversized chunks at embedding time."""

import re
from typing import Any
from pydantic import BaseModel, Field

from academic_agent.embeddings.tokenizer import TokenCounter
from academic_agent.processing.chunking.models import ChunkRecord


class EmbeddingFragment(BaseModel):
    """An atomic embedding-ready text fragment derived from a ChunkRecord.

    Preserves parent chunk provenance, context headers, and structural integrity.
    """

    fragment_id: str = Field(description="Unique fragment identifier, e.g. <parent_chunk_id>#frag_0.")
    parent_chunk_id: str = Field(description="Canonical Phase 2 ChunkRecord ID.")
    fragment_index: int = Field(description="0-based sequence index of this fragment.")
    total_fragments: int = Field(description="Total number of fragments produced for this parent chunk.")
    text: str = Field(description="Self-contained text ready for embedding.")
    source_file: str = Field(description="Originating PDF filename.")
    source_pages: list[int] = Field(description="Original page numbers contributing to this fragment.")
    source_unit_ids: list[str] = Field(default_factory=list, description="Associated source layout unit IDs.")
    section_title: str | None = Field(default=None, description="Section heading context.")
    content_type: str = Field(description="Content classification (e.g. TABLE, PARAGRAPH, MIXED).")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional provenance attributes.")


class StructureAwareFragmenter:
    """Decomposes oversized ChunkRecords into valid sequence-bounded fragments at embedding time.

    Preserves parent provenance, headers, and does not alter the frozen Phase 2 ChunkRecord.
    """

    def __init__(self, token_counter: TokenCounter):
        self.token_counter = token_counter

    def fragment_chunk(self, chunk: ChunkRecord, max_tokens: int) -> list[EmbeddingFragment]:
        """Convert a ChunkRecord into one or more EmbeddingFragments satisfying max_tokens.

        If the chunk already fits within max_tokens, returns a single fragment.
        Otherwise, performs a structure-aware split at natural exercise/workspace/table boundaries.
        """
        token_count = self.token_counter.count_tokens(chunk.text)
        if token_count <= max_tokens:
            return [
                EmbeddingFragment(
                    fragment_id=f"{chunk.chunk_id}#frag_0",
                    parent_chunk_id=chunk.chunk_id,
                    fragment_index=0,
                    total_fragments=1,
                    text=chunk.text,
                    source_file=chunk.source_file,
                    source_pages=chunk.source_pages,
                    source_unit_ids=chunk.source_unit_ids,
                    section_title=chunk.section_title,
                    content_type=chunk.content_type.value if hasattr(chunk.content_type, "value") else str(chunk.content_type),
                    metadata={
                        **chunk.metadata,
                        "token_count": token_count,
                        "is_fragment": False,
                    },
                )
            ]

        # Decompose along meaningful sub-structure boundaries
        # For multi-slide activities: split between presentation/table and calculation workspaces
        text = chunk.text
        split_markers = [
            r"\n(?=CMA Workspace Step \d)",
            r"\n(?=Workspace Step \d)",
            r"\n(?=Activity \d+ Workspace)",
            r"\n(?=Step \d+:)",
            r"\n\n(?=\| [^|]+ \|)",  # Table boundary
        ]

        raw_parts = [text]
        for pattern in split_markers:
            refined_parts = []
            for part in raw_parts:
                if self.token_counter.count_tokens(part) > max_tokens:
                    splits = re.split(pattern, part)
                    splits = [s.strip() for s in splits if s.strip()]
                    if len(splits) > 1:
                        refined_parts.extend(splits)
                    else:
                        refined_parts.append(part)
                else:
                    refined_parts.append(part)
            raw_parts = refined_parts

        # If boundary splitting didn't divide enough, split at double newlines preserving context
        if len(raw_parts) == 1 and self.token_counter.count_tokens(raw_parts[0]) > max_tokens:
            paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
            current_group: list[str] = []
            grouped_parts: list[str] = []
            for para in paragraphs:
                candidate = "\n\n".join(current_group + [para])
                if current_group and self.token_counter.count_tokens(candidate) > max_tokens:
                    grouped_parts.append("\n\n".join(current_group))
                    current_group = [para]
                else:
                    current_group.append(para)
            if current_group:
                grouped_parts.append("\n\n".join(current_group))
            raw_parts = grouped_parts

        total_frags = len(raw_parts)
        fragments: list[EmbeddingFragment] = []

        # Context header to prepend if missing from subsequent fragments
        context_header = ""
        first_line = text.split("\n")[0].strip()
        if first_line.startswith(("Activity", "Lecture", "Section", "Topic")):
            context_header = first_line

        for idx, part in enumerate(raw_parts):
            fragment_text = part
            if idx > 0 and context_header and not fragment_text.startswith(context_header):
                fragment_text = f"{context_header} (continued)\n\n{fragment_text}"

            frag_tokens = self.token_counter.count_tokens(fragment_text)
            fragments.append(
                EmbeddingFragment(
                    fragment_id=f"{chunk.chunk_id}#frag_{idx}",
                    parent_chunk_id=chunk.chunk_id,
                    fragment_index=idx,
                    total_fragments=total_frags,
                    text=fragment_text,
                    source_file=chunk.source_file,
                    source_pages=chunk.source_pages,
                    source_unit_ids=chunk.source_unit_ids,
                    section_title=chunk.section_title,
                    content_type=chunk.content_type.value if hasattr(chunk.content_type, "value") else str(chunk.content_type),
                    metadata={
                        **chunk.metadata,
                        "token_count": frag_tokens,
                        "is_fragment": True,
                        "parent_chunk_id": chunk.chunk_id,
                    },
                )
            )

        return fragments
