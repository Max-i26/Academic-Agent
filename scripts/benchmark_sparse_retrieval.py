"""Benchmark SparseRetriever (BM25) against the 8 Academic Domain Questions.

Evaluates lexical retrieval across the 238 canonical lecture records:
- Hit@1, Hit@3, MRR
- Detailed question-by-question breakdown
- Comparison with dense retrieval baseline
"""

import json
import time
from pathlib import Path
import numpy as np

from academic_agent.ingestion.layout_extractor import LayoutExtractor
from academic_agent.processing.chunking import ChunkingConfig, ChunkingEngine, EstimateTokenizer
from academic_agent.embeddings.config import KnownEmbeddingModel
from academic_agent.embeddings.fragmenter import StructureAwareFragmenter
from academic_agent.embeddings.tokenizer import HuggingFaceTokenCounter
from academic_agent.storage.models import VectorRecord
from academic_agent.retrieval.models import RetrievalQuery
from academic_agent.retrieval.sparse_retriever import SparseRetriever


def main():
    print("=" * 80)
    print(" Academic-Agent: Stage 4.3 Sparse Retrieval (BM25) Benchmark")
    print("=" * 80)

    # 1. Ingest & Chunk raw PDFs
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

    # 2. Structure-Aware Fragmentation
    token_counter = HuggingFaceTokenCounter(model_name=KnownEmbeddingModel.MINILM.value)
    fragmenter = StructureAwareFragmenter(token_counter=token_counter)

    all_fragments = []
    for chunk in all_chunks:
        all_fragments.extend(fragmenter.fragment_chunk(chunk, max_tokens=256))

    assert len(all_fragments) == 238, f"Expected 238 items, got {len(all_fragments)}"
    print(f"Loaded {len(all_fragments)} canonical records across {len(pdf_files)} PDF lectures.")

    # Convert to VectorRecords (vector field dummy for sparse testing)
    records = []
    for frag in all_fragments:
        rec = VectorRecord(
            record_id=frag.fragment_id,
            chunk_id=frag.parent_chunk_id,
            vector=[0.0] * 64,
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
        records.append(rec)

    # 3. Index into SparseRetriever
    t0 = time.perf_counter()
    retriever = SparseRetriever(k1=1.5, b=0.75)
    retriever.index_records(records)
    index_elapsed = (time.perf_counter() - t0) * 1000.0
    print(f"Indexed {retriever.corpus_size} records in {index_elapsed:.2f} ms.")

    # 4. Load benchmark questions & ground truth
    with open("benchmarks/questions.json") as f:
        questions = json.load(f)
    with open("benchmarks/ground_truth.json") as f:
        ground_truth = json.load(f)

    gt_by_qid = {item["question_id"]: item for item in ground_truth}

    hits_at_1 = 0
    hits_at_3 = 0
    hits_at_5 = 0
    reciprocal_ranks = []
    latencies = []

    print("\n" + "-" * 80)
    print(f"{'QID':<5} | {'Hit@1':<6} | {'Hit@3':<6} | {'Rank':<5} | {'BM25 Score':<11} | {'Time (ms)':<9} | Topic")
    print("-" * 80)

    for q in questions:
        qid = q["id"]
        gt = gt_by_qid[qid]
        target_chunk_id = gt["target_chunk_id"]

        q_obj = RetrievalQuery(text=q["question"], top_k=10, candidate_pool_size=20)
        t_start = time.perf_counter()
        candidates = retriever.retrieve(q_obj)
        latency_ms = (time.perf_counter() - t_start) * 1000.0
        latencies.append(latency_ms)

        target_rank = None
        target_score = None
        for cand in candidates:
            if cand.chunk_id == target_chunk_id:
                target_rank = cand.rank
                target_score = cand.score
                break

        if target_rank is not None:
            rr = 1.0 / target_rank
            reciprocal_ranks.append(rr)
            h1 = 1 if target_rank == 1 else 0
            h3 = 1 if target_rank <= 3 else 0
            h5 = 1 if target_rank <= 5 else 0
            hits_at_1 += h1
            hits_at_3 += h3
            hits_at_5 += h5
            print(f"{qid:<5} | {str(bool(h1)):<6} | {str(bool(h3)):<6} | {target_rank:<5} | {target_score:<11.4f} | {latency_ms:<9.2f} | {q['topic']}")
        else:
            reciprocal_ranks.append(0.0)
            print(f"{qid:<5} | False  | False  | >20   | {'N/A':<11} | {latency_ms:<9.2f} | {q['topic']}")

    hit_rate_1 = hits_at_1 / len(questions)
    hit_rate_3 = hits_at_3 / len(questions)
    hit_rate_5 = hits_at_5 / len(questions)
    mrr = float(np.mean(reciprocal_ranks))
    avg_latency = float(np.mean(latencies))

    print("-" * 80)
    print(f"Sparse (BM25) Metrics: Hit@1 = {hit_rate_1*100:.1f}% | Hit@3 = {hit_rate_3*100:.1f}% | Hit@5 = {hit_rate_5*100:.1f}% | MRR = {mrr:.4f} | Avg Latency = {avg_latency:.2f} ms")
    print("=" * 80)


if __name__ == "__main__":
    main()
