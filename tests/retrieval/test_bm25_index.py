"""Unit tests for BM25Index scoring, inverted index structures, and ranking."""

import pytest
from academic_agent.retrieval.sparse.bm25 import BM25Index


class TestBM25Index:
    """Tests covering BM25Okapi scoring mechanics and inverted index indexing."""

    @pytest.fixture
    def sample_records(self):
        return [
            {
                "record_id": "doc1",
                "text": "Simple exponential smoothing uses smoothing parameter alpha to weight current observations.",
            },
            {
                "record_id": "doc2",
                "text": "Autocorrelation function ACF measures correlation between Y_t and Y_{t-k} at lag k.",
            },
            {
                "record_id": "doc3",
                "text": "Centered moving average CMA for quarterly data centers even period moving averages.",
            },
            {
                "record_id": "doc4",
                "text": "ACF bounds are approximately plus minus 1.96 divided by sqrt of n for white noise.",
            },
        ]

    def test_indexing_state(self, sample_records):
        index = BM25Index(k1=1.5, b=0.75)
        index.index(sample_records)

        assert index.corpus_size == 4
        assert index.avg_doc_length > 0
        assert index.get_record("doc1") is not None
        assert index.get_record("nonexistent") is None

    def test_exact_term_match_ranking(self, sample_records):
        index = BM25Index()
        index.index(sample_records)

        # Query for CMA
        results = index.search("CMA centered moving average", top_k=2)
        assert len(results) > 0
        top_doc_id, top_score = results[0]
        assert top_doc_id == "doc3"
        assert top_score > 0.0

    def test_acronym_and_greek_query(self, sample_records):
        index = BM25Index()
        index.index(sample_records)

        # Query using Greek symbol α (doc1 has 'alpha')
        results = index.search("smoothing parameter α", top_k=2)
        assert len(results) > 0
        assert results[0][0] == "doc1"

    def test_subscript_and_formula_query(self, sample_records):
        index = BM25Index()
        index.index(sample_records)

        # Query for Y_t and lag k
        results = index.search("Y_t at lag k", top_k=2)
        assert len(results) > 0
        assert results[0][0] == "doc2"

    def test_allowed_doc_ids_prefiltering(self, sample_records):
        index = BM25Index()
        index.index(sample_records)

        # Query for ACF: both doc2 and doc4 match, but allow only doc4
        results = index.search("ACF", allowed_doc_ids={"doc4"})
        assert len(results) == 1
        assert results[0][0] == "doc4"

    def test_empty_or_unmatched_queries(self, sample_records):
        index = BM25Index()
        index.index(sample_records)

        assert index.search("") == []
        assert index.search("   ") == []
        assert index.search("completely unmentioned quantum mechanics") == []

    def test_reindexing_clears_old_state(self, sample_records):
        index = BM25Index()
        index.index(sample_records)
        assert index.corpus_size == 4

        # Re-index with a single document
        index.index([{"record_id": "new_doc", "text": "Brand new document"}])
        assert index.corpus_size == 1
        assert index.get_record("doc1") is None
        assert index.get_record("new_doc") is not None
