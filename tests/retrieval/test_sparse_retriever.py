"""Unit and integration tests for SparseRetriever."""

import pytest

from academic_agent.storage.models import VectorRecord, FilterQuery
from academic_agent.storage.exceptions import StoreNotInitializedError
from academic_agent.retrieval.models import (
    RetrievalQuery,
    RetrievalCandidate,
    Evidence,
)
from academic_agent.retrieval.protocols import (
    RetrieverProtocol,
    SparseRetrieverProtocol,
)
from academic_agent.retrieval.sparse_retriever import SparseRetriever


@pytest.fixture
def sample_academic_records():
    """Sample realistic academic records mimicking Phase 2 and 3 chunks and fragments."""
    r1 = VectorRecord(
        record_id="chunk_ses_1",
        chunk_id="chunk_ses_1",
        vector=[0.1] * 64,
        text="THE SES EQUATION: Sₜ = αYₜ + (1−α)Sₜ₋₁. α is the smoothing parameter between 0 and 1.",
        source_file="4. DS5104 Lecture4 Simple_Exponential_Smoothing.pdf",
        source_pages=[7],
        source_unit_ids=["u10", "u11"],
        section_title="THE SES EQUATION",
        content_type="FORMULA",
        fragment_index=0,
        total_fragments=1,
    )
    r2 = VectorRecord(
        record_id="chunk_mae_1",
        chunk_id="chunk_mae_1",
        vector=[0.2] * 64,
        text="Accuracy metrics: MAE = mean(|eₜ|), MSE = mean(eₜ²), RMSE = √MSE. Easy to interpret.",
        source_file="4. DS5104 Lecture4 Simple_Exponential_Smoothing.pdf",
        source_pages=[18],
        source_unit_ids=["u25"],
        section_title="RMSE and MAE",
        content_type="TABLE",
        fragment_index=0,
        total_fragments=1,
    )
    r3 = VectorRecord(
        record_id="chunk_cma_frag_1",
        chunk_id="chunk_cma",
        vector=[0.3] * 64,
        text="Centered Moving Average: For quarterly data m = 4, a 4-period MA falls between periods. CMA centers the trend.",
        source_file="5. DS5104 Lecture5 Trend_Seasonal_Decomposition.pdf",
        source_pages=[13],
        source_unit_ids=["u40", "u41"],
        section_title="Centered Moving Average: Why Center?",
        content_type="PARAGRAPH",
        fragment_index=1,
        total_fragments=2,
    )
    r4 = VectorRecord(
        record_id="chunk_acf_1",
        chunk_id="chunk_acf_1",
        vector=[0.4] * 64,
        text="Significance Bounds in ACF: Approximate bounds ± 1.96 / √n. Bars outside bounds indicate significant autocorrelation.",
        source_file="7. DS5104 Lecture7 Autocorrelation ACF.pdf",
        source_pages=[17],
        source_unit_ids=["u77"],
        section_title="Significance Bounds in ACF",
        content_type="FORMULA",
        fragment_index=0,
        total_fragments=1,
    )
    r5 = VectorRecord(
        record_id="chunk_acf_lag2",
        chunk_id="chunk_acf_lag2",
        vector=[0.5] * 64,
        text="ACF at lag 2 measures direct correlation between observations two periods apart: r₂ = Cov(Yₜ, Yₜ₋₂) / Var(Yₜ).",
        source_file="7. DS5104 Lecture7 Autocorrelation ACF.pdf",
        source_pages=[11],
        source_unit_ids=["u65"],
        section_title="Lag k Autocorrelation",
        content_type="FORMULA",
        fragment_index=0,
        total_fragments=1,
    )
    return [r1, r2, r3, r4, r5]


def test_sparse_retriever_satisfies_protocols():
    retriever = SparseRetriever()
    assert isinstance(retriever, RetrieverProtocol)
    assert isinstance(retriever, SparseRetrieverProtocol)
    assert retriever.name == "sparse"


def test_uninitialized_retriever_raises():
    retriever = SparseRetriever()
    query = RetrievalQuery(text="test query")
    with pytest.raises(StoreNotInitializedError):
        retriever.retrieve(query)


def test_exact_formula_and_symbol_query(sample_academic_records):
    retriever = SparseRetriever()
    retriever.index_records(sample_academic_records)

    # Query for alpha and SES: should rank r1 first
    query = RetrievalQuery(text="smoothing parameter alpha in SES", top_k=2)
    candidates = retriever.retrieve(query)

    assert len(candidates) > 0
    assert candidates[0].record_id == "chunk_ses_1"
    assert candidates[0].rank == 1
    assert candidates[0].retrieval_method == "sparse"
    assert candidates[0].score > 0.0


def test_acf_at_lag_2_exact_specialization(sample_academic_records):
    retriever = SparseRetriever()
    retriever.index_records(sample_academic_records)

    # Query: "ACF at lag 2"
    # Should retrieve chunk_acf_lag2 with top rank, not generic significance bounds!
    query = RetrievalQuery(text="ACF at lag 2", top_k=3)
    candidates = retriever.retrieve(query)

    assert len(candidates) > 0
    assert candidates[0].record_id == "chunk_acf_lag2"
    assert candidates[0].source_pages == [11]


def test_filter_query_forwarding(sample_academic_records):
    retriever = SparseRetriever()
    retriever.index_records(sample_academic_records)

    # Query for MAE, but filter strictly for Lecture 5 (which does not have MAE)
    query_blocked = RetrievalQuery(
        text="MAE accuracy metric",
        top_k=5,
        filters=FilterQuery(source_file="5. DS5104 Lecture5 Trend_Seasonal_Decomposition.pdf"),
    )
    candidates_blocked = retriever.retrieve(query_blocked)
    assert len(candidates_blocked) == 0

    # Query with correct filter for Lecture 4
    query_ok = RetrievalQuery(
        text="MAE accuracy metric",
        top_k=5,
        filters=FilterQuery(source_file="4. DS5104 Lecture4 Simple_Exponential_Smoothing.pdf"),
    )
    candidates_ok = retriever.retrieve(query_ok)
    assert len(candidates_ok) > 0
    assert candidates_ok[0].record_id == "chunk_mae_1"
    assert candidates_ok[0].source_file == "4. DS5104 Lecture4 Simple_Exponential_Smoothing.pdf"


def test_candidate_provenance_and_fragment_metadata(sample_academic_records):
    retriever = SparseRetriever()
    retriever.index_records(sample_academic_records)

    # Query for CMA
    query = RetrievalQuery(text="centered moving average", top_k=1)
    candidates = retriever.retrieve(query)

    assert len(candidates) == 1
    cand = candidates[0]
    assert cand.record_id == "chunk_cma_frag_1"
    assert cand.chunk_id == "chunk_cma"
    assert cand.rank == 1
    assert cand.source_pages == [13]
    assert cand.source_unit_ids == ["u40", "u41"]
    assert cand.section_title == "Centered Moving Average: Why Center?"
    assert cand.fragment_index == 1
    assert cand.total_fragments == 2
    assert cand.is_fragment is True

    # Evidence conversion
    ev = cand.to_evidence()
    assert isinstance(ev, Evidence)
    assert ev.format_citation() == "[Lecture 5, p. 13]"
    assert ev.source_unit_ids == ["u40", "u41"]


def test_latency_diagnostics(sample_academic_records):
    retriever = SparseRetriever()
    retriever.index_records(sample_academic_records)

    query = RetrievalQuery(text="MAE formula", top_k=2)
    retriever.retrieve(query)

    diag = retriever.last_diagnostics
    assert "sparse_latency_ms" in diag
    assert diag["sparse_latency_ms"] >= 0.0


def test_reindexing_and_empty_query(sample_academic_records):
    retriever = SparseRetriever()
    retriever.index_records(sample_academic_records)
    assert retriever.corpus_size == 5

    # Re-index with smaller subset
    retriever.index_records([sample_academic_records[0]])
    assert retriever.corpus_size == 1

    # Empty text or unmentioned terms return empty list
    query_unmentioned = RetrievalQuery(text="quantum entanglement thermodynamics")
    assert retriever.retrieve(query_unmentioned) == []
