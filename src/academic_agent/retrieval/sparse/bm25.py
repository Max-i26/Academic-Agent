"""Pure-Python BM25 Index with Robertson-Spärck Jones non-negative IDF."""

from collections import Counter, defaultdict
import math
from typing import Any, Sequence

from academic_agent.retrieval.sparse.tokenizer import AcademicTokenizer


class BM25Index:
    """In-memory inverted index and scoring engine implementing BM25Okapi."""

    def __init__(
        self,
        tokenizer: AcademicTokenizer | None = None,
        k1: float = 1.5,
        b: float = 0.75,
    ):
        """Initialize BM25 index with tuning parameters and tokenizer.

        Args:
            tokenizer: AcademicTokenizer instance for domain-aware tokenization.
            k1: Term frequency saturation parameter (default: 1.5).
            b: Document length normalization parameter (default: 0.75).
        """
        self.tokenizer = tokenizer or AcademicTokenizer()
        self.k1 = k1
        self.b = b

        # Index state
        self._corpus_size: int = 0
        self._avg_doc_length: float = 0.0
        self._doc_lengths: dict[str, int] = {}
        # Inverted index: term -> {doc_id: term_frequency}
        self._inverted_index: dict[str, dict[str, int]] = defaultdict(dict)
        # Stored records: doc_id -> record
        self._records: dict[str, Any] = {}
        # Precomputed IDF weights
        self._idf: dict[str, float] = {}

    @property
    def corpus_size(self) -> int:
        """Total number of indexed documents."""
        return self._corpus_size

    @property
    def avg_doc_length(self) -> float:
        """Average document length across the indexed corpus."""
        return self._avg_doc_length

    def index(self, records: Sequence[Any]) -> None:
        """Build the BM25 inverted index from a sequence of records.

        Clears any existing index state before rebuilding.

        Args:
            records: Collection of objects with record_id and text attributes (e.g. VectorRecord).
        """
        # Reset state for deterministic re-indexing
        self._corpus_size = len(records)
        self._doc_lengths.clear()
        self._inverted_index.clear()
        self._records.clear()
        self._idf.clear()

        if not records:
            self._avg_doc_length = 0.0
            return

        total_length = 0
        for rec in records:
            if isinstance(rec, dict):
                rec_id = rec.get("record_id")
                text = rec.get("text", "")
            else:
                rec_id = getattr(rec, "record_id", None)
                text = getattr(rec, "text", "")
            if not rec_id:
                raise ValueError("Record must have a valid non-empty 'record_id'")

            self._records[rec_id] = rec

            tokens = self.tokenizer.tokenize(text)
            doc_len = len(tokens)
            self._doc_lengths[rec_id] = doc_len
            total_length += doc_len

            term_counts = Counter(tokens)
            for term, count in term_counts.items():
                self._inverted_index[term][rec_id] = count

        self._avg_doc_length = total_length / self._corpus_size if self._corpus_size > 0 else 0.0

        # Precompute non-negative IDF for all indexed terms
        # IDF(t) = ln(1 + (N - n(t) + 0.5) / (n(t) + 0.5))
        for term, postings in self._inverted_index.items():
            df = len(postings)
            idf = math.log(1.0 + (self._corpus_size - df + 0.5) / (df + 0.5))
            self._idf[term] = idf

    def get_record(self, record_id: str) -> Any | None:
        """Retrieve original record by record_id."""
        return self._records.get(record_id)

    def search(
        self,
        query: str | list[str],
        top_k: int | None = None,
        allowed_doc_ids: set[str] | None = None,
    ) -> list[tuple[str, float]]:
        """Score indexed documents against query tokens using BM25Okapi.

        Args:
            query: Query string or pre-tokenized token list.
            top_k: Maximum candidate tuples (doc_id, score) to return.
            allowed_doc_ids: Optional set of permitted record IDs for pre-filtering.

        Returns:
            List of (doc_id, score) tuples ranked in descending score order.
        """
        if self._corpus_size == 0:
            return []

        if isinstance(query, str):
            query_tokens = self.tokenizer.tokenize(query)
        else:
            query_tokens = query

        if not query_tokens:
            return []

        scores: dict[str, float] = defaultdict(float)

        for term in query_tokens:
            if term not in self._inverted_index:
                continue

            idf = self._idf.get(term, 0.0)
            if idf <= 0.0:
                continue

            postings = self._inverted_index[term]
            for doc_id, tf in postings.items():
                if allowed_doc_ids is not None and doc_id not in allowed_doc_ids:
                    continue

                doc_len = self._doc_lengths[doc_id]
                denom = tf + self.k1 * (1.0 - self.b + self.b * (doc_len / self._avg_doc_length))
                term_score = idf * (tf * (self.k1 + 1.0)) / denom
                scores[doc_id] += term_score

        if not scores:
            return []

        # Sort documents by BM25 score descending
        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        if top_k is not None and top_k > 0:
            return ranked[:top_k]
        return ranked
