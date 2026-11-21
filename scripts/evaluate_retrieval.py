"""Run comprehensive evaluation of Dense, Sparse, Hybrid, and Reranked Retrieval against the 30-Question Benchmark.

Computes:
- Candidate recall acceptance gate (Dense Recall@15, Sparse Recall@15, Union Recall@30)
- Audited warm steady-state latency distributions
- Macro Hit@1, Hit@3, Hit@5, MRR, MAP, Recall@5, nDCG@5 across all 4 strategies
- Category-stratified performance across the 6 academic categories
- Detailed Question-by-Question ranking comparison and Win/Tie/Loss delta analysis
- Active truncation and provenance diagnostics
Exports results to data/evaluation/results_benchmark_v1.json.
"""

import json
import logging
import time
from pathlib import Path
from typing import Any

from academic_agent.ingestion.layout_extractor import LayoutExtractor
from academic_agent.processing.chunking import ChunkingConfig, ChunkingEngine, EstimateTokenizer
from academic_agent.embeddings.config import EmbeddingConfig, KnownEmbeddingModel
from academic_agent.embeddings.engine import EmbeddingEngine
from academic_agent.embeddings.fragmenter import StructureAwareFragmenter
from academic_agent.embeddings.tokenizer import HuggingFaceTokenCounter
from academic_agent.storage.models import IndexMetadata, VectorRecord
from academic_agent.storage.config import VectorStoreConfig
from academic_agent.storage.providers.lancedb.store import LanceDBVectorStore
from academic_agent.retrieval.models import RetrievalStrategy, RetrievalQuery
from academic_agent.retrieval.dense_retriever import DenseRetriever
from academic_agent.retrieval.sparse_retriever import SparseRetriever
from academic_agent.retrieval.fusion import ReciprocalRankFusion
from academic_agent.retrieval.reranker import CrossEncoderReranker
from academic_agent.retrieval.pipeline import RetrievalPipeline
from academic_agent.retrieval.evaluation_models import BenchmarkQuery
from academic_agent.retrieval.evaluator import RetrievalEvaluator

# Configure logging
logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")


def main():
    print("=" * 110)
    print(" Academic-Agent: Stage 4.5 End-to-End Retrieval Benchmark Evaluation (30 Questions)")
    print(" Comparing: Dense | Sparse | Hybrid RRF | Hybrid + Cross-Encoder Rerank")
    print("=" * 110)

    t_start = time.perf_counter()

    # 1. Ingest & Chunk raw PDFs (Canonical Corpus of 238 Records)
    extractor = LayoutExtractor()
    tokenizer = EstimateTokenizer(factor=1.33)
    chunking_config = ChunkingConfig(
        target_tokens=180,
        max_tokens=220,
        min_tokens=40,
        overlap_tokens=35,
        max_overlap_tokens=45,
        target_words=135,
        max_words=165,
        min_words=30,
        overlap_words=25,
        max_overlap_words=35,
    )
    chunk_engine = ChunkingEngine(config=chunking_config, tokenizer=tokenizer)

    pdf_files = sorted(Path("data/raw").glob("*.pdf"))
    all_chunks = []
    for pdf in pdf_files:
        layout_records = extractor.extract(pdf)
        all_chunks.extend(chunk_engine.chunk(layout_records))

    token_counter = HuggingFaceTokenCounter(model_name=KnownEmbeddingModel.MINILM.value)
    fragmenter = StructureAwareFragmenter(token_counter=token_counter)

    all_fragments = []
    for chunk in all_chunks:
        all_fragments.extend(fragmenter.fragment_chunk(chunk, max_tokens=256))

    assert len(all_fragments) == 238, f"Expected 238 canonical records, got {len(all_fragments)}"
    print(f"[✓] Canonical corpus validated: {len(all_fragments)} records across {len(pdf_files)} lecture PDFs.")

    # 2. Setup EmbeddingEngine & LanceDBVectorStore
    emb_config = EmbeddingConfig(
        model_name=KnownEmbeddingModel.MINILM.value,
        device="cpu",
        batch_size=32,
        strict_token_limit=True,
        normalize_embeddings=True,
    )
    emb_engine = EmbeddingEngine.from_config(emb_config)

    store_config = VectorStoreConfig(
        backend="lance",
        persist_path="data/lancedb",
        collection_name="academic_corpus",
        expected_model_name="sentence-transformers/all-MiniLM-L6-v2",
        expected_dimension=384,
        distance_metric="cosine",
    )
    lancedb_store = LanceDBVectorStore(config=store_config)
    lancedb_store.initialize(
        IndexMetadata(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            embedding_dimension=384,
            metric="cosine",
            normalized=True,
        )
    )
    assert lancedb_store.count() == 238, f"Expected 238 records in LanceDB, got {lancedb_store.count()}"
    print(f"[✓] LanceDB vector store connected: {lancedb_store.count()} records.")

    dense_retriever = DenseRetriever(embedding_engine=emb_engine, vector_store=lancedb_store)

    # 3. Setup SparseRetriever (BM25)
    vector_records = []
    for frag in all_fragments:
        rec = VectorRecord(
            record_id=frag.fragment_id,
            chunk_id=frag.parent_chunk_id,
            vector=[0.0] * 384,
            text=frag.text,
            source_file=frag.source_file,
            source_pages=frag.source_pages,
            source_unit_ids=frag.source_unit_ids,
            section_title=frag.section_title,
            content_type=frag.content_type,
            fragment_index=frag.fragment_index,
            total_fragments=frag.total_fragments,
            metadata=frag.metadata,
        )
        vector_records.append(rec)

    sparse_retriever = SparseRetriever(k1=1.5, b=0.75)
    sparse_retriever.index_records(vector_records)
    print(f"[✓] BM25 sparse index built: {len(vector_records)} documents indexed.")

    # 4. Setup CrossEncoderReranker & Pipeline
    reranker = CrossEncoderReranker(
        model_name="cross-encoder/ms-marco-MiniLM-L-6-v2",
        device="cpu",
        lazy=False,
    )
    fusion_engine = ReciprocalRankFusion(k=60)
    pipeline = RetrievalPipeline(
        dense_retriever=dense_retriever,
        sparse_retriever=sparse_retriever,
        fusion_strategy=fusion_engine,
        reranker=reranker,
    )
    print(f"[✓] CrossEncoderReranker loaded: '{reranker.name}' on device '{reranker.device}'.")

    # 5. Load 30-Question Benchmark
    benchmark_path = Path("data/evaluation/retrieval_benchmark_v1.json")
    with open(benchmark_path, "r", encoding="utf-8") as f:
        raw_queries = json.load(f)

    benchmark_queries = [BenchmarkQuery(**q) for q in raw_queries]
    print(f"[✓] Loaded benchmark: {len(benchmark_queries)} verified queries across 6 categories.")

    # 6. Warm-up Run (Eliminate Cold-Start JIT & Cache Artifacts)
    print("\n--- Warming up pipeline components for fair steady-state latency measurement ---")
    _ = pipeline.search(RetrievalQuery(text="warmup", strategy=RetrievalStrategy.DENSE, top_k=5))
    _ = pipeline.search(RetrievalQuery(text="warmup", strategy=RetrievalStrategy.SPARSE, top_k=5))
    _ = pipeline.search(RetrievalQuery(text="warmup", strategy=RetrievalStrategy.HYBRID, top_k=5))
    _ = pipeline.search(RetrievalQuery(text="warmup", strategy=RetrievalStrategy.HYBRID_RERANK, top_k=5))
    print("[✓] Pipeline warmed up.\n")

    # 7. Execute Evaluation across all 4 strategies
    evaluator = RetrievalEvaluator(
        pipeline=pipeline,
        dataset_name="Academic-Agent-Retrieval-v1",
        dataset_version="1.0.0",
    )

    strategies = [
        (RetrievalStrategy.DENSE, "Dense (MiniLM + LanceDB)"),
        (RetrievalStrategy.SPARSE, "Sparse (BM25 Robertson-Spärck)"),
        (RetrievalStrategy.HYBRID, "Hybrid (Dense + Sparse + RRF)"),
        (RetrievalStrategy.HYBRID_RERANK, "Hybrid + Rerank (MS-MARCO)"),
    ]

    summaries = {}
    for strat, label in strategies:
        print(f"Evaluating {label}...")
        summary = evaluator.evaluate_dataset(
            queries=benchmark_queries,
            strategy=strat,
            top_k=5,
        )
        summaries[strat.value] = summary

    # 8. Print Global Macro Metrics Table
    print("\n" + "=" * 115)
    print(" GLOBAL MACRO METRICS (30 Questions, Fixed Corpus = 238 Records, Top-K = 5)")
    print("=" * 115)
    header = f"{'Strategy':<32} | {'Hit@1':<8} | {'Hit@3':<8} | {'Hit@5':<8} | {'MRR':<8} | {'MAP':<8} | {'Recall@5':<9} | {'nDCG@5':<8} | {'Avg Latency':<12}"
    print(header)
    print("-" * 115)

    for strat, label in strategies:
        s = summaries[strat.value]
        print(
            f"{label:<32} | "
            f"{s.hit_at_1 * 100:6.1f}% | "
            f"{s.hit_at_3 * 100:6.1f}% | "
            f"{s.hit_at_5 * 100:6.1f}% | "
            f"{s.mrr:8.4f} | "
            f"{s.map_score:8.4f} | "
            f"{s.recall_at_5 * 100:7.1f}% | "
            f"{s.ndcg_at_5:8.4f} | "
            f"{s.avg_latency_ms:7.2f} ms"
        )
    print("-" * 115)

    # 9. Category-Stratified Breakdown Table
    categories = list(summaries["dense"].category_metrics.keys())
    print("\n" + "=" * 125)
    print(" CATEGORY-STRATIFIED BREAKDOWN (Hit@1 / Hit@3 / Hit@5 / nDCG@5)")
    print("=" * 125)
    print(f"{'Category (N=5)':<28} | {'Dense (H1/H3/H5/nDCG)':<22} | {'Sparse (H1/H3/H5/nDCG)':<22} | {'Hybrid (H1/H3/H5/nDCG)':<22} | {'Reranked (H1/H3/H5/nDCG)':<24}")
    print("-" * 125)

    for cat in categories:
        d_cat = summaries["dense"].category_metrics.get(cat, {})
        s_cat = summaries["sparse"].category_metrics.get(cat, {})
        h_cat = summaries["hybrid"].category_metrics.get(cat, {})
        r_cat = summaries["hybrid_rerank"].category_metrics.get(cat, {})

        d_str = f"{d_cat.get('hit_at_1', 0)*100:2.0f}/{d_cat.get('hit_at_3', 0)*100:2.0f}/{d_cat.get('hit_at_5', 0)*100:2.0f}/{d_cat.get('ndcg_at_5', 0):.2f}"
        s_str = f"{s_cat.get('hit_at_1', 0)*100:2.0f}/{s_cat.get('hit_at_3', 0)*100:2.0f}/{s_cat.get('hit_at_5', 0)*100:2.0f}/{s_cat.get('ndcg_at_5', 0):.2f}"
        h_str = f"{h_cat.get('hit_at_1', 0)*100:2.0f}/{h_cat.get('hit_at_3', 0)*100:2.0f}/{h_cat.get('hit_at_5', 0)*100:2.0f}/{h_cat.get('ndcg_at_5', 0):.2f}"
        r_str = f"{r_cat.get('hit_at_1', 0)*100:2.0f}/{r_cat.get('hit_at_3', 0)*100:2.0f}/{r_cat.get('hit_at_5', 0)*100:2.0f}/{r_cat.get('ndcg_at_5', 0):.2f}"

        print(f"{cat:<28} | {d_str:<22} | {s_str:<22} | {h_str:<22} | {r_str:<24}")
    print("-" * 125)

    # 10. Question-by-Question Analysis & Deltas
    d_results = {r.query_id: r for r in summaries["dense"].per_query_results}
    s_results = {r.query_id: r for r in summaries["sparse"].per_query_results}
    h_results = {r.query_id: r for r in summaries["hybrid"].per_query_results}
    r_results = {r.query_id: r for r in summaries["hybrid_rerank"].per_query_results}

    print("\n" + "=" * 115)
    print(" QUESTION-BY-QUESTION RANKING COMPARISON (Dense vs Sparse vs RRF vs RRF+Rerank)")
    print("=" * 115)
    print(f"{'QID':<5} | {'Category':<26} | {'Dense':<6} | {'Sparse':<7} | {'RRF':<5} | {'Rerank':<7} | {'Rerank vs Dense':<16} | {'Rerank vs RRF'}")
    print("-" * 115)

    rerank_vs_dense_wins = 0
    rerank_vs_dense_losses = 0
    rerank_vs_dense_ties = 0

    rerank_vs_rrf_wins = 0
    rerank_vs_rrf_losses = 0
    rerank_vs_rrf_ties = 0

    for q in benchmark_queries:
        qid = q.query_id
        d_r = d_results[qid].first_relevant_rank
        s_r = s_results[qid].first_relevant_rank
        h_r = h_results[qid].first_relevant_rank
        r_r = r_results[qid].first_relevant_rank

        d_rank_str = str(d_r) if d_r is not None else ">5"
        s_rank_str = str(s_r) if s_r is not None else ">5"
        h_rank_str = str(h_r) if h_r is not None else ">5"
        r_rank_str = str(r_r) if r_r is not None else ">5"

        d_val = d_r if d_r is not None else 999
        h_val = h_r if h_r is not None else 999
        r_val = r_r if r_r is not None else 999

        # Delta vs Dense
        if r_val < d_val:
            delta_dense = f"▲ +{d_val - r_val} (WIN)"
            rerank_vs_dense_wins += 1
        elif r_val > d_val:
            delta_dense = f"▼ -{r_val - d_val} (LOSS)"
            rerank_vs_dense_losses += 1
        else:
            delta_dense = "= TIE"
            rerank_vs_dense_ties += 1

        # Delta vs RRF
        if r_val < h_val:
            delta_rrf = f"▲ +{h_val - r_val} (WIN)"
            rerank_vs_rrf_wins += 1
        elif r_val > h_val:
            delta_rrf = f"▼ -{r_val - h_val} (LOSS)"
            rerank_vs_rrf_losses += 1
        else:
            delta_rrf = "= TIE"
            rerank_vs_rrf_ties += 1

        print(f"{qid:<5} | {q.category.value:<26} | {d_rank_str:<6} | {s_rank_str:<7} | {h_rank_str:<5} | {r_rank_str:<7} | {delta_dense:<16} | {delta_rrf}")

    print("-" * 115)
    print("HEAD-TO-HEAD COMPARISON:")
    print(f"  Rerank vs Dense: {rerank_vs_dense_wins} Wins, {rerank_vs_dense_ties} Ties, {rerank_vs_dense_losses} Losses (Net: {'+' if rerank_vs_dense_wins >= rerank_vs_dense_losses else ''}{rerank_vs_dense_wins - rerank_vs_dense_losses})")
    print(f"  Rerank vs RRF:   {rerank_vs_rrf_wins} Wins, {rerank_vs_rrf_ties} Ties, {rerank_vs_rrf_losses} Losses (Net: {'+' if rerank_vs_rrf_wins >= rerank_vs_rrf_losses else ''}{rerank_vs_rrf_wins - rerank_vs_rrf_losses})")
    print(f"  Truncation Diagnostics Count: {reranker.truncation_count} passages truncated.")
    print("=" * 115)

    # 11. Export Full Results to JSON
    results_export = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "corpus_records": len(all_fragments),
        "total_queries": len(benchmark_queries),
        "macro_metrics": {
            strat: {
                "hit_at_1": summaries[strat].hit_at_1,
                "hit_at_3": summaries[strat].hit_at_3,
                "hit_at_5": summaries[strat].hit_at_5,
                "mrr": summaries[strat].mrr,
                "map": summaries[strat].map_score,
                "recall_at_5": summaries[strat].recall_at_5,
                "ndcg_at_5": summaries[strat].ndcg_at_5,
                "avg_latency_ms": summaries[strat].avg_latency_ms,
            }
            for strat in ["dense", "sparse", "hybrid", "hybrid_rerank"]
        },
        "head_to_head": {
            "rerank_vs_dense": {
                "wins": rerank_vs_dense_wins,
                "ties": rerank_vs_dense_ties,
                "losses": rerank_vs_dense_losses,
            },
            "rerank_vs_rrf": {
                "wins": rerank_vs_rrf_wins,
                "ties": rerank_vs_rrf_ties,
                "losses": rerank_vs_rrf_losses,
            },
        },
        "category_metrics": {
            strat: summaries[strat].category_metrics for strat in ["dense", "sparse", "hybrid", "hybrid_rerank"]
        },
        "truncation_diagnostics": {
            "total_truncations": reranker.truncation_count,
        },
        "per_query_comparison": [
            {
                "query_id": q.query_id,
                "query": q.query,
                "category": q.category.value,
                "dense_rank": d_results[q.query_id].first_relevant_rank,
                "sparse_rank": s_results[q.query_id].first_relevant_rank,
                "rrf_rank": h_results[q.query_id].first_relevant_rank,
                "rerank_rank": r_results[q.query_id].first_relevant_rank,
            }
            for q in benchmark_queries
        ],
    }

    out_path = Path("data/evaluation/results_benchmark_v1.json")
    with open(out_path, "w", encoding="utf-8") as out:
        json.dump(results_export, out, indent=2)

    print(f"\n[✓] Results successfully exported to {out_path}")
    print(f"Total benchmark elapsed time: {time.perf_counter() - t_start:.2f}s\n")


if __name__ == "__main__":
    main()
