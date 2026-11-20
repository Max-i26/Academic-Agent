from __future__ import annotations

from typing import Sequence

from academic_agent.ingestion.layout_extractor import PageLayoutRecord
from academic_agent.ingestion.pdf_extractor import PageRecord
from academic_agent.processing.chunking.models import (
    ChunkingConfig,
    ChunkRecord,
)
from academic_agent.processing.chunking.normalizer import DocumentNormalizer
from academic_agent.processing.chunking.overlap_manager import OverlapManager
from academic_agent.processing.chunking.provenance import ProvenanceTracker
from academic_agent.processing.chunking.semantic_grouper import SemanticGrouper
from academic_agent.processing.chunking.size_controller import SizeController
from academic_agent.processing.chunking.structural_parser import StructuralParser
from academic_agent.processing.chunking.tokenizer import (
    TokenizerProtocol,
    WhitespaceTokenizer,
)
from academic_agent.processing.chunking.validator import (
    ChunkValidator,
    ValidationReport,
)


class ChunkingEngine:
    """
    Public facade orchestrating the complete Academic-Agent Chunking Engine pipeline (Stages A-G).

    Pipeline:
    1. Document Normalizer: Convert PageRecord/PageLayoutRecord into standardized ContentUnits.
    2. Structural Parser: Identify academic headings, equations, lists, tables, and paragraphs.
    3. Semantic Grouper: Contextually bundle structurally related units.
    4. Size Controller: Enforce size budgets and handle safe recursive splitting.
    5. Overlap Manager: Inject sentence-bounded, section-aware contextual overlap.
    6. Provenance Tracker: Assign deterministic SHA-256 chunk IDs, source pages, and source unit IDs.
    7. Validator: Validate output data integrity and produce a ValidationReport.
    """

    def __init__(
        self,
        config: ChunkingConfig | None = None,
        tokenizer: TokenizerProtocol | None = None,
        normalizer: DocumentNormalizer | None = None,
        parser: StructuralParser | None = None,
        grouper: SemanticGrouper | None = None,
        size_controller: SizeController | None = None,
        overlap_manager: OverlapManager | None = None,
        provenance_tracker: ProvenanceTracker | None = None,
        validator: ChunkValidator | None = None,
    ) -> None:
        self.config = config or ChunkingConfig()
        self.tokenizer = tokenizer or WhitespaceTokenizer()
        self.normalizer = normalizer or DocumentNormalizer()
        self.parser = parser or StructuralParser()
        self.grouper = grouper or SemanticGrouper()
        self.size_controller = size_controller or SizeController(self.config, self.tokenizer)
        self.overlap_manager = overlap_manager or OverlapManager(
            self.config, self.tokenizer, self.size_controller
        )
        self.provenance_tracker = provenance_tracker or ProvenanceTracker(self.tokenizer)
        self.validator = validator or ChunkValidator()

    def chunk(
        self,
        records: Sequence[PageRecord | PageLayoutRecord] | PageRecord | PageLayoutRecord,
    ) -> list[ChunkRecord]:
        """
        Transform extracted document page records into validated academic ChunkRecords.
        """
        # Support passing a single record or a sequence
        if isinstance(records, (PageRecord, PageLayoutRecord)):
            records_seq: Sequence[PageRecord | PageLayoutRecord] = [records]
        else:
            records_seq = records

        if not records_seq:
            return []

        # Stage A: Document Normalization
        raw_units = self.normalizer.normalize(records_seq)
        if not raw_units:
            return []

        # Stage B: Structural Unit Detection & Classification
        structured_units = self.parser.parse(raw_units)

        # Stage C: Semantic Grouping
        semantic_groups = self.grouper.group(structured_units)

        # Stage D: Chunk Size Management & Recursive Splitting
        draft_chunks = self.size_controller.control_size(semantic_groups)

        # Stage E: Overlap Management
        overlapped_drafts = self.overlap_manager.apply_overlap(draft_chunks)

        # Stage F: Provenance & Final ChunkRecord Construction
        final_chunks = self.provenance_tracker.finalize_chunks(overlapped_drafts)

        # Stage G: Output Validation
        self.validator.validate(final_chunks, self.config)

        return final_chunks

    def get_validation_report(self, chunks: Sequence[ChunkRecord]) -> ValidationReport:
        """Inspect and summarize distribution metrics for a collection of ChunkRecords."""
        return self.validator.validate(chunks, self.config)
