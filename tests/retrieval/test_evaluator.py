"""Unit tests for RetrievalEvaluator coordinator, diagnostics, and category reporting."""

import pytest
import numpy as np

from academic_agent.retrieval.models import (
    RetrievalQuery,
    RetrievalCandidate,
    Evidence,
    RetrievalStrategy,
    RetrievalResponse,
)
from academic_agent.retrieval.protocols import RetrievalPipelineProtocol
from academic_agent.retrieval.evaluation_models import (
    BenchmarkQuery,
    QueryCategory,
    BenchmarkSourceReference,
)
from academic_agent.retrieval.evaluator import RetrievalEvaluator


class MockPipelineForEvaluation:
    """Mock pipeline returning tailored evidence for testing evaluator aggregations."""

    def __init__(self, rankings_by_qid: dict[str, list[str]]):
        self.rankings = rankings_by_qid

    def search(self, query: RetrievalQuery | str) -> RetrievalResponse:
        q_obj = query if isinstance(query, RetrievalQuery) else RetrievalQuery(text=query)
        # Search by query text matching
        ranked_ids = self.rankings.get(q_obj.text, ["default_d1", "default_d2"])

        evidence = []
        for rank, rid in enumerate(ranked_ids, start=1):
            evidence.append(
                Evidence(
                    record_id=rid,
                    chunk_id=rid.split("#")[0],
                    text=f"Evidence text for {rid}",
                    source_file="Lecture1.pdf",
                    source_pages=[1],
                    score=1.0 - rank * 0.1,
                    retrieval_methods=["dense", "sparse"] if rank == 1 else ["dense"],
                )
            )

        return RetrievalResponse(
            query=q_obj,
            evidence=evidence,
            strategy_used=q_obj.strategy,
        )


class FailingMockPipeline:
    """Mock pipeline that raises an exception on a specific query to test failure visibility."""

    def search(self, query: RetrievalQuery | str) -> RetrievalResponse:
        q_obj = query if isinstance(query, RetrievalQuery) else RetrievalQuery(text=query)
        if "fail" in q_obj.text:
            raise RuntimeError("Database connection timed out")
        return RetrievalResponse(
            query=q_obj,
            evidence=[Evidence(record_id="doc1", chunk_id="doc1", text="text", source_file="L.pdf", source_pages=[1], score=0.9)],
        )


def test_evaluator_single_query():
    rankings = {"What is MAE?": ["c_mae", "c_mse"]}
    pipeline = MockPipelineForEvaluation(rankings)
    evaluator = RetrievalEvaluator(pipeline=pipeline)

    query = BenchmarkQuery(
        query_id="q01",
        query="What is MAE?",
        category=QueryCategory.FORMULAS,
        relevant_records={"c_mae": 3},
        expected_sources=[BenchmarkSourceReference(source_file="Lecture4.pdf", pages=[18])],
    )

    res = evaluator.evaluate_query(query, strategy=RetrievalStrategy.HYBRID, top_k=5)

    assert res.query_id == "q01"
    assert res.hit_at_1 == 1.0
    assert res.hit_at_3 == 1.0
    assert res.mrr == 1.0
    assert res.map_score == 1.0
    assert res.first_relevant_rank == 1
    assert set(res.target_found_by_methods) == {"dense", "sparse"}
    assert res.error_message is None


def test_evaluator_dataset_macro_averages_and_categories():
    rankings = {
        "Q_formula": ["c_rel", "c_dist"],       # Hit@1 = 1.0, MRR = 1.0
        "Q_concept": ["c_dist", "c_rel_concept"], # Hit@1 = 0.0, MRR = 0.5
    }
    pipeline = MockPipelineForEvaluation(rankings)
    evaluator = RetrievalEvaluator(pipeline=pipeline)

    q1 = BenchmarkQuery(
        query_id="q1",
        query="Q_formula",
        category=QueryCategory.FORMULAS,
        relevant_records={"c_rel": 3},
    )
    q2 = BenchmarkQuery(
        query_id="q2",
        query="Q_concept",
        category=QueryCategory.EXPLANATIONS,
        relevant_records={"c_rel_concept": 3},
    )

    summary = evaluator.evaluate_dataset([q1, q2], strategy=RetrievalStrategy.HYBRID, top_k=5)

    assert summary.total_queries == 2
    assert summary.evaluated_queries == 2
    assert summary.failed_queries_count == 0

    # Macro averages: (1.0 + 0.0)/2 = 0.5 for Hit@1; (1.0 + 0.5)/2 = 0.75 for MRR
    assert np.isclose(summary.hit_at_1, 0.5)
    assert np.isclose(summary.mrr, 0.75)

    # Category breakdown
    assert QueryCategory.FORMULAS.value in summary.category_metrics
    assert summary.category_metrics[QueryCategory.FORMULAS.value]["hit_at_1"] == 1.0

    assert QueryCategory.EXPLANATIONS.value in summary.category_metrics
    assert summary.category_metrics[QueryCategory.EXPLANATIONS.value]["hit_at_1"] == 0.0
    assert summary.category_metrics[QueryCategory.EXPLANATIONS.value]["mrr"] == 0.5


def test_evaluator_failure_visibility():
    """Verifies that failed queries are recorded explicitly and never silently excluded from aggregates."""
    pipeline = FailingMockPipeline()
    evaluator = RetrievalEvaluator(pipeline=pipeline)

    q_ok = BenchmarkQuery(
        query_id="q_ok",
        query="normal query",
        category=QueryCategory.DEFINITIONS,
        relevant_records={"doc1": 3},
    )
    q_fail = BenchmarkQuery(
        query_id="q_fail",
        query="trigger fail",
        category=QueryCategory.DEFINITIONS,
        relevant_records={"doc1": 3},
    )

    summary = evaluator.evaluate_dataset([q_ok, q_fail], strategy=RetrievalStrategy.HYBRID)

    assert summary.total_queries == 2
    assert summary.failed_queries_count == 1
    assert len(summary.failed_queries) == 1
    assert summary.failed_queries[0]["query_id"] == "q_fail"
    assert "Database connection timed out" in summary.failed_queries[0]["error"]

    # The failed query receives 0.0, so macro Hit@1 is (1.0 + 0.0) / 2 = 0.5
    assert np.isclose(summary.hit_at_1, 0.5)
