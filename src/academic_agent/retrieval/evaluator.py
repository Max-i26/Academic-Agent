"""Comprehensive Retrieval Evaluator implementing Hit@K, MRR, MAP, Recall@K, and nDCG@K."""

import json
import logging
import math
import time
from typing import Any, Sequence

from academic_agent.retrieval.models import RetrievalQuery, RetrievalStrategy, RetrievalCandidate
from academic_agent.retrieval.protocols import RetrievalPipelineProtocol
from academic_agent.retrieval.evaluation_models import (
    BenchmarkQuery,
    QueryEvaluationResult,
    EvaluationSummary,
)

logger = logging.getLogger(__name__)


def get_record_relevance(
    record_id: str,
    relevant_records: dict[str, int],
    canonical_chunk_ids: list[str] | None = None,
) -> int:
    """Resolve graded relevance (0-3) for a record_id, with fragment parent tolerance."""
    if record_id in relevant_records:
        return relevant_records[record_id]

    # Fragment ID check (e.g. 'chunk_id#frag_0')
    if "#" in record_id:
        parent_id = record_id.split("#")[0]
        if parent_id in relevant_records:
            return relevant_records[parent_id]
        if canonical_chunk_ids and parent_id in canonical_chunk_ids:
            return 2  # Default to strongly relevant if in canonical list

    # Canonical parent chunk check
    if canonical_chunk_ids and record_id in canonical_chunk_ids:
        return 2

    return 0


def compute_hit_at_k(
    ranked_record_ids: Sequence[str],
    relevant_records: dict[str, int],
    k: int,
    min_grade: int = 1,
    canonical_chunk_ids: list[str] | None = None,
) -> float:
    """Compute Hit@K: 1.0 if at least one relevant record appears in top K, else 0.0."""
    if k <= 0 or not ranked_record_ids:
        return 0.0

    for rid in ranked_record_ids[:k]:
        grade = get_record_relevance(rid, relevant_records, canonical_chunk_ids)
        if grade >= min_grade:
            return 1.0
    return 0.0


def compute_reciprocal_rank(
    ranked_record_ids: Sequence[str],
    relevant_records: dict[str, int],
    k: int | None = None,
    min_grade: int = 1,
    canonical_chunk_ids: list[str] | None = None,
) -> float:
    """Compute Reciprocal Rank: 1 / rank of first relevant record, else 0.0."""
    items = ranked_record_ids[:k] if k is not None and k > 0 else ranked_record_ids
    for rank_idx, rid in enumerate(items, start=1):
        grade = get_record_relevance(rid, relevant_records, canonical_chunk_ids)
        if grade >= min_grade:
            return 1.0 / rank_idx
    return 0.0


def compute_average_precision(
    ranked_record_ids: Sequence[str],
    relevant_records: dict[str, int],
    k: int | None = None,
    min_grade: int = 1,
    canonical_chunk_ids: list[str] | None = None,
) -> float:
    """Compute Average Precision (AP) using standard TREC formulation."""
    total_relevant = sum(1 for g in relevant_records.values() if g >= min_grade)
    if total_relevant == 0:
        return 0.0

    items = ranked_record_ids[:k] if k is not None and k > 0 else ranked_record_ids
    if not items:
        return 0.0

    hits = 0
    precisions_sum = 0.0
    seen_relevant: set[str] = set()

    for rank_idx, rid in enumerate(items, start=1):
        grade = get_record_relevance(rid, relevant_records, canonical_chunk_ids)
        if grade >= min_grade:
            # Map fragment to canonical identity to prevent duplicate counting
            canonical_key = rid.split("#")[0] if "#" in rid else rid
            if canonical_key not in seen_relevant:
                hits += 1
                seen_relevant.add(canonical_key)
                precisions_sum += hits / rank_idx

    return precisions_sum / total_relevant


def compute_recall_at_k(
    ranked_record_ids: Sequence[str],
    relevant_records: dict[str, int],
    k: int,
    min_grade: int = 1,
    canonical_chunk_ids: list[str] | None = None,
) -> float:
    """Compute Recall@K: proportion of all known relevant records retrieved in top K."""
    total_relevant = sum(1 for g in relevant_records.values() if g >= min_grade)
    if total_relevant == 0:
        return 0.0

    if k <= 0 or not ranked_record_ids:
        return 0.0

    retrieved_relevant: set[str] = set()
    for rid in ranked_record_ids[:k]:
        grade = get_record_relevance(rid, relevant_records, canonical_chunk_ids)
        if grade >= min_grade:
            canonical_key = rid.split("#")[0] if "#" in rid else rid
            retrieved_relevant.add(canonical_key)

    return len(retrieved_relevant) / total_relevant


def compute_ndcg_at_k(
    ranked_record_ids: Sequence[str],
    relevant_records: dict[str, int],
    k: int,
    canonical_chunk_ids: list[str] | None = None,
) -> float:
    """Compute normalized Discounted Cumulative Gain (nDCG@K) using graded relevance (0-3)."""
    if k <= 0 or not ranked_record_ids:
        return 0.0

    # 1. Calculate DCG@K
    dcg = 0.0
    items = ranked_record_ids[:k]
    for rank_idx, rid in enumerate(items, start=1):
        rel = get_record_relevance(rid, relevant_records, canonical_chunk_ids)
        if rel > 0:
            gain = (2.0 ** rel) - 1.0
            discount = math.log2(rank_idx + 1)
            dcg += gain / discount

    # 2. Calculate Ideal DCG@K (IDCG@K)
    sorted_grades = sorted(relevant_records.values(), reverse=True)
    idcg = 0.0
    for rank_idx, rel in enumerate(sorted_grades[:k], start=1):
        if rel > 0:
            gain = (2.0 ** rel) - 1.0
            discount = math.log2(rank_idx + 1)
            idcg += gain / discount

    if idcg == 0.0:
        return 0.0

    return dcg / idcg


class RetrievalEvaluator:
    """Evaluation coordinator evaluating a RetrievalPipeline across benchmark queries."""

    def __init__(
        self,
        pipeline: RetrievalPipelineProtocol,
        dataset_name: str = "Academic-Agent-Retrieval-Benchmark",
        dataset_version: str = "v1.0",
    ):
        """Initialize evaluator with target retrieval pipeline."""
        self.pipeline = pipeline
        self.dataset_name = dataset_name
        self.dataset_version = dataset_version

    def evaluate_query(
        self,
        benchmark_query: BenchmarkQuery,
        strategy: RetrievalStrategy = RetrievalStrategy.HYBRID,
        top_k: int = 10,
    ) -> QueryEvaluationResult:
        """Run single query evaluation through the pipeline and compute all metrics."""
        q_obj = RetrievalQuery(
            text=benchmark_query.query,
            top_k=top_k,
            strategy=strategy,
            candidate_pool_size=max(20, top_k * 2),
        )

        t0 = time.perf_counter()
        error_msg = None
        ranked_records: list[str] = []
        target_methods: list[str] = []

        try:
            response = self.pipeline.search(q_obj)
            latency_ms = response.diagnostics.total_latency_ms or (time.perf_counter() - t0) * 1000.0
            # Extract ranked record IDs from evidence
            ranked_records = [e.record_id for e in response.evidence]

            # Find methods that surfaced the first relevant result
            for e in response.evidence:
                rel = get_record_relevance(e.record_id, benchmark_query.relevant_records, benchmark_query.canonical_chunk_ids)
                if rel >= 1:
                    target_methods = list(e.retrieval_methods)
                    break

        except Exception as exc:
            latency_ms = (time.perf_counter() - t0) * 1000.0
            error_msg = str(exc)
            logger.error("Query '%s' failed in evaluation: %s", benchmark_query.query_id, exc)

        # Compute all information retrieval metrics
        rel_records = benchmark_query.relevant_records
        canon_ids = benchmark_query.canonical_chunk_ids

        hit_1 = compute_hit_at_k(ranked_records, rel_records, k=1, canonical_chunk_ids=canon_ids)
        hit_3 = compute_hit_at_k(ranked_records, rel_records, k=3, canonical_chunk_ids=canon_ids)
        hit_5 = compute_hit_at_k(ranked_records, rel_records, k=5, canonical_chunk_ids=canon_ids)
        mrr = compute_reciprocal_rank(ranked_records, rel_records, k=top_k, canonical_chunk_ids=canon_ids)
        map_s = compute_average_precision(ranked_records, rel_records, k=top_k, canonical_chunk_ids=canon_ids)
        recall_5 = compute_recall_at_k(ranked_records, rel_records, k=5, canonical_chunk_ids=canon_ids)
        ndcg_5 = compute_ndcg_at_k(ranked_records, rel_records, k=5, canonical_chunk_ids=canon_ids)

        # 1-based first relevant rank
        first_rel_rank = int(round(1.0 / mrr)) if mrr > 0 else None

        return QueryEvaluationResult(
            query_id=benchmark_query.query_id,
            query=benchmark_query.query,
            category=benchmark_query.category.value if hasattr(benchmark_query.category, "value") else str(benchmark_query.category),
            strategy=strategy.value if hasattr(strategy, "value") else str(strategy),
            ranked_records=ranked_records,
            hit_at_1=hit_1,
            hit_at_3=hit_3,
            hit_at_5=hit_5,
            mrr=mrr,
            map_score=map_s,
            recall_at_5=recall_5,
            ndcg_at_5=ndcg_5,
            latency_ms=latency_ms,
            first_relevant_rank=first_rel_rank,
            target_found_by_methods=target_methods,
            error_message=error_msg,
        )

    def evaluate_dataset(
        self,
        queries: Sequence[BenchmarkQuery],
        strategy: RetrievalStrategy = RetrievalStrategy.HYBRID,
        top_k: int = 10,
    ) -> EvaluationSummary:
        """Run full evaluation suite across all benchmark queries for a given strategy."""
        per_query_results: list[QueryEvaluationResult] = []
        failed_queries: list[dict[str, Any]] = []

        for b_query in queries:
            res = self.evaluate_query(b_query, strategy=strategy, top_k=top_k)
            per_query_results.append(res)
            if res.error_message:
                failed_queries.append({
                    "query_id": b_query.query_id,
                    "query": b_query.query,
                    "error": res.error_message,
                })

        total = len(queries)
        eval_count = len(per_query_results)
        if eval_count == 0:
            raise ValueError("No queries evaluated.")

        # Aggregate macro averages across all queries (failed queries count as 0.0)
        hit_1_macro = sum(r.hit_at_1 for r in per_query_results) / total
        hit_3_macro = sum(r.hit_at_3 for r in per_query_results) / total
        hit_5_macro = sum(r.hit_at_5 for r in per_query_results) / total
        mrr_macro = sum(r.mrr for r in per_query_results) / total
        map_macro = sum(r.map_score for r in per_query_results) / total
        recall_5_macro = sum(r.recall_at_5 for r in per_query_results) / total
        ndcg_5_macro = sum(r.ndcg_at_5 for r in per_query_results) / total
        avg_lat = sum(r.latency_ms for r in per_query_results) / total

        # Category breakdown
        category_metrics: dict[str, dict[str, float]] = {}
        by_category: dict[str, list[QueryEvaluationResult]] = {}
        for r in per_query_results:
            by_category.setdefault(r.category, []).append(r)

        for cat, results in by_category.items():
            n = len(results)
            category_metrics[cat] = {
                "count": n,
                "hit_at_1": sum(r.hit_at_1 for r in results) / n,
                "hit_at_3": sum(r.hit_at_3 for r in results) / n,
                "hit_at_5": sum(r.hit_at_5 for r in results) / n,
                "mrr": sum(r.mrr for r in results) / n,
                "ndcg_at_5": sum(r.ndcg_at_5 for r in results) / n,
            }

        return EvaluationSummary(
            dataset_name=self.dataset_name,
            dataset_version=self.dataset_version,
            strategy=strategy.value if hasattr(strategy, "value") else str(strategy),
            total_queries=total,
            evaluated_queries=eval_count,
            failed_queries_count=len(failed_queries),
            hit_at_1=hit_1_macro,
            hit_at_3=hit_3_macro,
            hit_at_5=hit_5_macro,
            mrr=mrr_macro,
            map_score=map_macro,
            recall_at_5=recall_5_macro,
            ndcg_at_5=ndcg_5_macro,
            avg_latency_ms=avg_lat,
            category_metrics=category_metrics,
            per_query_results=per_query_results,
            failed_queries=failed_queries,
        )
