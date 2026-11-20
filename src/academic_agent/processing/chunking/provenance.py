from __future__ import annotations

from typing import Sequence

from academic_agent.processing.chunking.models import (
    ChunkRecord,
)
from academic_agent.processing.chunking.size_controller import DraftChunk
from academic_agent.processing.chunking.tokenizer import (
    EstimateTokenizer,
    TokenizerProtocol,
)


class ProvenanceTracker:
    """
    Transforms DraftChunks into finalized, validated ChunkRecords with exact academic provenance.
    """

    def __init__(self, tokenizer: TokenizerProtocol | None = None) -> None:
        self.tokenizer = tokenizer or EstimateTokenizer()

    def finalize_chunks(
        self,
        draft_chunks: Sequence[DraftChunk],
    ) -> list[ChunkRecord]:
        """Convert DraftChunks into sequential, indexed ChunkRecords with stable identifiers."""
        if not draft_chunks:
            return []

        final_records: list[ChunkRecord] = []
        document_indices: dict[str, int] = {}

        for draft in draft_chunks:
            doc_id = draft.document_id
            chunk_index = document_indices.get(doc_id, 0)
            document_indices[doc_id] = chunk_index + 1

            clean_text = draft.text.strip()
            chunk_id = ChunkRecord.generate_chunk_id(
                document_id=doc_id,
                chunk_index=chunk_index,
                text=clean_text,
            )

            word_count = len(clean_text.split())
            char_count = len(clean_text)
            estimated_tokens = self.tokenizer.count_tokens(clean_text)
            overlap_words = int(draft.metadata.get("overlap_word_count", 0))

            # Strictly sorted, unique, positive page numbers
            source_pages = sorted({p for p in draft.source_pages if p >= 1})
            if not source_pages:
                source_pages = [1]

            # Deduplicate source unit IDs preserving order
            seen_units: set[str] = set()
            ordered_units: list[str] = []
            for uid in draft.source_unit_ids:
                if uid not in seen_units:
                    seen_units.add(uid)
                    ordered_units.append(uid)

            record = ChunkRecord(
                chunk_id=chunk_id,
                document_id=doc_id,
                source_file=draft.source_file,
                chunk_index=chunk_index,
                text=clean_text,
                source_pages=source_pages,
                source_unit_ids=ordered_units,
                word_count=word_count,
                char_count=char_count,
                estimated_token_count=estimated_tokens,
                overlap_word_count=overlap_words,
                section_title=draft.section_title,
                content_type=draft.content_type,
                metadata=dict(draft.metadata),
            )
            final_records.append(record)

        return final_records
