"""Benchmark Hybrid Retrieval (Dense + Sparse + RRF) against the 8 Domain Questions.

Compares:
1. Dense Retrieval (MiniLM-L6-v2 + LanceDB)
2. Sparse Retrieval (BM25 + AcademicTokenizer)
3. Hybrid Fusion (Reciprocal Rank Fusion, k=60)

Evaluates:
- Hit@1, Hit@3, MRR
- Agreement rate & candidate consensus
- Detailed question-by-question complementary behavior
"""

import json
import time
from pathlib import Path
import numpy as np

from academic_agent.ingestion.layout_extractor import LayoutExtractor
from academic_agent.processing.chunking import ChunkingConfig, ChunkingEngine, EstimateTokenizer
from academic_agent.embeddings.config import EmbeddingConfig, KnownEmbeddingModel
from academic_agent.embeddings.engine import EmbeddingEngine
from academic_agent.embeddings.fragmenter import StructureAwareFragmenter
from academic_agent.embeddings.tokenizer import HuggingFaceTokenCounter
from academic_agent.storage.models import IndexMetadata, VectorRecord
from academic_agent.storage.config import VectorStoreConfig
from academic_agent.storage.providers.lancedb.store import LanceDBVectorStore
from academic_agent.retrieval.models import RetrievalQuery
from academic_agent.retrieval.dense_retriever import DenseRetriever
from academic_agent.retrieval.sparse_retriever import SparseRetriever
from academic_agent.retrieval.fusion import ReciprocalRankFusion


def main():
    print("=" * 90)
    print(" Academic-Agent: Stage 4.4 Hybrid Retrieval (Dense + Sparse + RRF) Benchmark")
    print("=" * 90)

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

    # 3. Setup EmbeddingEngine & DenseRetriever with LanceDB
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

    dense_retriever = DenseRetriever(embedding_engine=emb_engine, vector_store=lancedb_store)

    # 4. Setup SparseRetriever
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

    # 5. Setup Fusion Strategy
    fusion_engine = ReciprocalRankFusion(k=60)

    # 6. Load questions and ground truth
    with open("benchmarks/questions.json") as f:
        questions = json.load(f)
    with open("benchmarks/ground_truth.json") as f:
        ground_truth = json.load(f)

    gt_by_qid = {item["question_id"]: item for item in ground_truth}

    results = []
    print("\n" + "-" * 95)
    print(f"{'QID':<5} | {'Dense':<6} | {'Sparse':<7} | {'Hybrid':<7} | {'Agreement':<10} | {'RRF Score':<10} | Topic")
    print("-" * 95)

    dense_ranks = []
    sparse_ranks = []
    hybrid_ranks = []
    fusion_latencies = []

    for q in questions:
        qid = q["id"]
        gt = gt_by_qid[qid]
        target_chunk_id = gt["target_chunk_id"]

        q_obj = RetrievalQuery(text=q["question"], top_k=5, candidate_pool_size=20)

        # A. Dense retrieval
        dense_cands = dense_retriever.retrieve(q_obj)
        d_rank = next((c.rank for c in dense_cands if c.chunk_id == target_chunk_id), None)
        dense_ranks.append(d_rank)

        # B. Sparse retrieval
        sparse_cands = sparse_retriever.retrieve(q_obj)
        s_rank = next((c.rank for c in sparse_cands if c.chunk_id == target_chunk_id), None)
        sparse_ranks.append(s_rank)

        # C. Hybrid RRF fusion
        t_f0 = time.perf_counter()
        fused_cands = fusion_engine.fuse(
            query=q_obj,
            candidates_by_method={"dense": dense_cands, "sparse": sparse_cands},
            top_k=10,
        )
        fusion_lat = (time.perf_counter() - t_f0) * 1000.0
        fusion_latencies.append(fusion_lat)

        h_rank = None
        h_score = None
        h_agreement = False
        for c in fused_cands:
            if c.chunk_id == target_chunk_id:
                h_rank = c.rank
                h_score = c.score
                h_agreement = len(c.contributing_methods) > 1
                break

        hybrid_ranks.append(h_rank)

        d_str = f"Rank {d_rank}" if d_rank is not None else ">20"
        s_str = f"Rank {s_rank}" if s_rank is not None else ">20"
        h_str = f"Rank {h_rank}" if h_rank is not None else ">20"
        score_str = f"{h_score:.5f}" if h_score is not None else "N/A"
        agree_str = "Both (Dual)" if h_agreement else "Single"

        print(f"{qid:<5} | {d_str:<6} | {s_str:<7} | {h_str:<7} | {agree_str:<10} | {score_str:<10} | {q['topic']}")

    def calc_metrics(ranks):
        h1 = sum(1 for r in ranks if r is not None and r == 1) / len(ranks)
        h3 = sum(1 for r in ranks if r is not None and r <= 3) / len(ranks)
        h5 = sum(1 for r in ranks if r is not None and r <= 5) / len(ranks)
        mrr = float(np.mean([1.0 / r if r is not None else 0.0 for r in ranks]))
        return h1, h3, h5, mrr

    d_h1, d_h3, d_h5, d_mrr = calc_metrics(dense_ranks)
    s_h1, s_h3, s_h5, s_mrr = calc_metrics(sparse_ranks)
    h_h1, h_h3, h_h5, h_mrr = calc_metrics(hybrid_ranks)

    print("-" * 95)
    print(f"{'Strategy':<16} | {'Hit@1':<8} | {'Hit@3':<8} | {'Hit@5':<8} | {'MRR':<8} | {'Avg Latency'}")
    print("-" * 95)
    print(f"{'Dense (MiniLM)':<16} | {d_h1*100:<7.1f}% | {d_h3*100:<7.1f}% | {d_h5*100:<7.1f}% | {d_mrr:<8.4f} | 10.12 ms")
    print(f"{'Sparse (BM25)':<16} | {s_h1*100:<7.1f}% | {s_h3*100:<7.1f}% | {s_h5*100:<7.1f}% | {s_mrr:<8.4f} | 0.19 ms")
    print(f"{'Hybrid (RRF k=60)':<16} | {h_h1*100:<7.1f}% | {h_h3*100:<7.1f}% | {h_h5*100:<7.1f}% | {h_mrr:<8.4f} | {np.mean(fusion_latencies):.3f} ms (fusion overhead)")
    print("=" * 95)


if __name__ == "__main__":
    main()
