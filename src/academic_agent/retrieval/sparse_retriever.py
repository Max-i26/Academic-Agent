"""Sparse BM25 retriever adapter satisfying SparseRetrieverProtocol."""

import logging
import time
from typing import Any, Sequence

from academic_agent.storage.models import VectorRecord, FilterQuery
from academic_agent.storage.exceptions import StoreNotInitializedError
from academic_agent.retrieval.models import RetrievalQuery, RetrievalCandidate
from academic_agent.retrieval.sparse.tokenizer import AcademicTokenizer
from academic_agent.retrieval.sparse.bm25 import BM25Index

logger = logging.getLogger(__name__)


class SparseRetriever:
    """Retrieves academic evidence candidates using lexical BM25 matching.

    Specialized for exact acronyms, formula variables, Greek parameters,
    and named theorems across canonical lecture chunks and fragments.
    """

    def __init__(
        self,
        index: BM25Index | None = None,
        tokenizer: AcademicTokenizer | None = None,
        k1: float = 1.5,
        b: float = 0.75,
    ):
        """Initialize SparseRetriever with BM25 parameters and tokenizer.

        Args:
            index: Pre-configured BM25Index instance, or None to construct with defaults.
            tokenizer: AcademicTokenizer instance.
            k1: Term frequency saturation parameter (default: 1.5).
            b: Document length normalization parameter (default: 0.75).
        """
        self._tokenizer = tokenizer or AcademicTokenizer()
        self._index = index or BM25Index(tokenizer=self._tokenizer, k1=k1, b=b)
        self._records: dict[str, Any] = {}
        self._is_initialized: bool = False
        self._last_diagnostics: dict[str, float] = {
            "sparse_latency_ms": 0.0,
        }

    @property
    def name(self) -> str:
        """Identifier of this retrieval mechanism."""
        return "sparse"

    @property
    def is_initialized(self) -> bool:
        """True if records have been indexed."""
        return self._is_initialized

    @property
    def corpus_size(self) -> int:
        """Number of indexed records."""
        return self._index.corpus_size

    @property
    def last_diagnostics(self) -> dict[str, float]:
        """Timing diagnostics from the most recent retrieve() call."""
        return dict(self._last_diagnostics)

    def index_records(self, records: Sequence[Any]) -> None:
        """Build the sparse BM25 index over the provided records.

        Clears and completely rebuilds the index for deterministic execution.

        Args:
            records: List of VectorRecord or chunk-like objects with text and provenance.
        """
        self._records.clear()
        for rec in records:
            if isinstance(rec, dict):
                rec_id = rec.get("record_id")
            else:
                rec_id = getattr(rec, "record_id", None)
            if not rec_id:
                raise ValueError("Record must possess a valid 'record_id'")
            self._records[rec_id] = rec

        self._index.index(records)
        self._is_initialized = True
        logger.info("SparseRetriever indexed %d academic records.", len(self._records))

    def retrieve(self, query: RetrievalQuery) -> list[RetrievalCandidate]:
        """Fetch sparse lexical candidates matching the RetrievalQuery.

        Args:
            query: Structured query containing text, filters, and pool size.

        Returns:
            List of RetrievalCandidate objects ordered by descending BM25 score.
        """
        if not isinstance(query, RetrievalQuery):
            raise TypeError(f"Expected RetrievalQuery instance, got {type(query).__name__}")

        if not self._is_initialized:
            raise StoreNotInitializedError(
                "SparseRetriever has not been initialized with records. Call index_records() first."
            )

        t0 = time.perf_counter()

        # 1. Evaluate metadata filters to identify allowed record IDs
        allowed_doc_ids: set[str] | None = None
        if query.filters is not None:
            allowed_doc_ids = set()
            for rid, rec in self._records.items():
                if isinstance(rec, VectorRecord):
                    if query.filters.matches(rec):
                        allowed_doc_ids.add(rid)
                else:
                    # Generic object matching for source_file, etc.
                    s_file = getattr(rec, "source_file", None) or (rec.get("source_file") if isinstance(rec, dict) else None)
                    if query.filters.source_file and s_file != query.filters.source_file:
                        continue
                    allowed_doc_ids.add(rid)

        # 2. Execute BM25 search
        # Governed by candidate_pool_size (not top_k)
        ranked_scores = self._index.search(
            query=query.text,
            top_k=query.candidate_pool_size,
            allowed_doc_ids=allowed_doc_ids,
        )

        sparse_latency_ms = (time.perf_counter() - t0) * 1000.0
        self._last_diagnostics = {
            "sparse_latency_ms": sparse_latency_ms,
        }

        # 3. Map BM25 results into RetrievalCandidate objects
        candidates: list[RetrievalCandidate] = []
        for rank_idx, (doc_id, score) in enumerate(ranked_scores, start=1):
            rec = self._records[doc_id]

            if isinstance(rec, dict):
                rec_id = rec.get("record_id", doc_id)
                chunk_id = rec.get("chunk_id", rec_id)
                text = rec.get("text", "")
                source_file = rec.get("source_file", "")
                source_pages = rec.get("source_pages", [])
                source_unit_ids = rec.get("source_unit_ids", [])
                section_title = rec.get("section_title")
                content_type = rec.get("content_type", "PARAGRAPH")
                fragment_index = rec.get("fragment_index", 0)
                total_fragments = rec.get("total_fragments", 1)
                raw_meta = rec.get("metadata", {})
            else:
                rec_id = getattr(rec, "record_id", doc_id)
                chunk_id = getattr(rec, "chunk_id", rec_id)
                text = getattr(rec, "text", "")
                source_file = getattr(rec, "source_file", "")
                source_pages = getattr(rec, "source_pages", [])
                source_unit_ids = getattr(rec, "source_unit_ids", [])
                section_title = getattr(rec, "section_title", None)
                content_type = getattr(rec, "content_type", "PARAGRAPH")
                fragment_index = getattr(rec, "fragment_index", 0)
                total_fragments = getattr(rec, "total_fragments", 1)
                raw_meta = getattr(rec, "metadata", {})

            cand = RetrievalCandidate(
                record_id=rec_id,
                chunk_id=chunk_id,
                rank=rank_idx,
                retrieval_method=self.name,
                score=score,
                normalized_score=None,
                text=text,
                source_file=source_file,
                source_pages=list(source_pages),
                source_unit_ids=list(source_unit_ids),
                section_title=section_title,
                content_type=content_type,
                fragment_index=fragment_index,
                total_fragments=total_fragments,
                metadata={
                    **raw_meta,
                    "bm25_score": score,
                },
            )
            candidates.append(cand)

        logger.debug(
            "SparseRetriever retrieved %d candidates for '%s' (latency=%.2fms)",
            len(candidates),
            query.text[:40],
            sparse_latency_ms,
        )

        return candidates
