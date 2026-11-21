"""End-to-End Academic Corpus Indexing & LanceDB Vector Store Verification.

Stage 3.3 Verification Script:
1. Ingests and chunks the 7 academic lecture PDFs from data/raw/.
2. Applies structure-aware fragmentation for oversized chunks (> 256 tokens).
3. Embeds all 238 items using sentence-transformers/all-MiniLM-L6-v2 via EmbeddingEngine.
4. Initializes and populates LanceDBVectorStore (data/lancedb, collection: academic_corpus).
5. Verifies:
   - Total indexed records = 238 (235 intact chunks + 3 fragments from Chunk e9285418782c3cf7).
   - Idempotent upserts (re-adding identical records does not inflate count).
   - Query retrieval on the 8 benchmark questions matches or exceeds benchmark results.
   - Metadata filtering (source_file, source_pages).
   - Cold restart / persistence verification (reopening existing LanceDB table and verifying sidecar metadata).
   - Embedding-space isolation (rejects incompatible models).
"""

import json
import logging
import shutil
import sys
import time
from pathlib import Path
import numpy as np

from academic_agent.ingestion.layout_extractor import LayoutExtractor
from academic_agent.processing.chunking import ChunkingConfig, ChunkingEngine, EstimateTokenizer
from academic_agent.embeddings.config import EmbeddingConfig, KnownEmbeddingModel
from academic_agent.embeddings.engine import EmbeddingEngine
from academic_agent.embeddings.fragmenter import StructureAwareFragmenter
from academic_agent.embeddings.tokenizer import HuggingFaceTokenCounter
from academic_agent.storage.models import IndexMetadata, VectorRecord, FilterQuery
from academic_agent.storage.config import VectorStoreConfig
from academic_agent.storage.providers.lancedb.store import LanceDBVectorStore
from academic_agent.storage.exceptions import IncompatibleModelError

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("IndexCorpus")


def main():
    print("=" * 80)
    print(" Academic-Agent: Stage 3.3 End-to-End Corpus Indexing & LanceDB Verification")
    print("=" * 80)

    # -------------------------------------------------------------------------
    # 1. Ingestion & Chunking
    # -------------------------------------------------------------------------
    print("\n[Step 1/6] Ingesting and Chunking raw PDFs...")
    data_dir = Path("data/raw")
    pdf_files = sorted(data_dir.glob("*.pdf"))
    if not pdf_files:
        print(f"ERROR: No PDFs found in {data_dir.resolve()}")
        sys.exit(1)

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

    all_chunks = []
    for pdf in pdf_files:
        layout_records = extractor.extract(pdf)
        chunks = chunk_engine.chunk(layout_records)
        all_chunks.extend(chunks)

    print(f"  Extracted {len(all_chunks)} chunks across {len(pdf_files)} PDF lectures.")
    assert len(all_chunks) == 236, f"Expected 236 chunks, got {len(all_chunks)}"

    # -------------------------------------------------------------------------
    # 2. Embedding Preparation & Structure-Aware Fragmentation
    # -------------------------------------------------------------------------
    print("\n[Step 2/6] Structure-Aware Fragmentation for sequence limits (max_tokens=256)...")
    token_counter = HuggingFaceTokenCounter(model_name=KnownEmbeddingModel.MINILM.value)
    fragmenter = StructureAwareFragmenter(token_counter=token_counter)

    all_fragments = []
    fragmented_chunk_ids = []
    for chunk in all_chunks:
        frags = fragmenter.fragment_chunk(chunk, max_tokens=256)
        all_fragments.extend(frags)
        if len(frags) > 1:
            fragmented_chunk_ids.append((chunk.chunk_id, len(frags)))

    print(f"  Total items prepared for embedding: {len(all_fragments)}")
    print(f"  Multi-fragment parent chunks: {fragmented_chunk_ids}")
    assert len(all_fragments) == 238, f"Expected 238 total fragments, got {len(all_fragments)}"
    assert len(fragmented_chunk_ids) == 1, "Expected exactly 1 multi-fragment chunk"
    assert fragmented_chunk_ids[0][0] == "e9285418782c3cf7"
    assert fragmented_chunk_ids[0][1] == 3

    # -------------------------------------------------------------------------
    # 3. Embedding Generation via EmbeddingEngine
    # -------------------------------------------------------------------------
    print("\n[Step 3/6] Generating embeddings with EmbeddingEngine (all-MiniLM-L6-v2)...")
    emb_config = EmbeddingConfig(
        model_name=KnownEmbeddingModel.MINILM.value,
        device="cpu",  # CPU for reproducible, deterministic benchmark
        batch_size=32,
        strict_token_limit=True,
        normalize_embeddings=True,
    )
    emb_engine = EmbeddingEngine.from_config(emb_config)

    texts_to_embed = [f.text for f in all_fragments]
    ids_to_embed = [f.fragment_id for f in all_fragments]

    t0 = time.perf_counter()
    embedded_docs = emb_engine.embed_documents(texts=texts_to_embed, chunk_ids=ids_to_embed)
    embed_elapsed = time.perf_counter() - t0
    print(f"  Embedded {len(embedded_docs)} passages in {embed_elapsed:.2f}s ({(len(embedded_docs)/embed_elapsed):.1f} docs/sec).")

    # Map to VectorRecords
    vector_records: list[VectorRecord] = []
    for frag, emb_doc in zip(all_fragments, embedded_docs):
        rec = VectorRecord(
            record_id=frag.fragment_id,
            chunk_id=frag.parent_chunk_id,
            vector=emb_doc.embedding,
            text=frag.text,
            source_file=frag.source_file,
            source_pages=frag.source_pages,
            section_title=frag.section_title,
            content_type=frag.content_type,
            fragment_index=frag.fragment_index,
            total_fragments=frag.total_fragments,
            metadata=frag.metadata,
        )
        vector_records.append(rec)

    # -------------------------------------------------------------------------
    # 4. LanceDB Vector Store Initialization & Upsert
    # -------------------------------------------------------------------------
    print("\n[Step 4/6] Initializing LanceDBVectorStore and indexing records...")
    db_path = Path("data/lancedb")
    # Clean previous test artifacts in db_path if present
    if db_path.exists():
        shutil.rmtree(db_path)

    store_config = VectorStoreConfig(
        backend="lance",
        persist_path=str(db_path),
        collection_name="academic_corpus",
        expected_model_name="sentence-transformers/all-MiniLM-L6-v2",
        expected_dimension=384,
        distance_metric="cosine",
        allow_overwrite=True,
    )
    store = LanceDBVectorStore(config=store_config)
    index_meta = IndexMetadata(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        embedding_dimension=384,
        metric="cosine",
        normalized=True,
    )
    store.initialize(index_meta)

    # Insert records
    store.add(vector_records)
    initial_count = store.count()
    print(f"  Indexed records count: {initial_count}")
    assert initial_count == 238, f"Expected 238 records, got {initial_count}"

    # Verify idempotency
    print("  Testing idempotency (upserting same 238 records)...")
    store.add(vector_records)
    idempotent_count = store.count()
    print(f"  Count after re-upsert: {idempotent_count}")
    assert idempotent_count == 238, f"Expected 238 after upsert, got {idempotent_count}"

    # -------------------------------------------------------------------------
    # 5. Query Retrieval & Benchmark Quality Audit
    # -------------------------------------------------------------------------
    print("\n[Step 5/6] Running Retrieval Benchmark across 8 Domain Questions...")
    with open("benchmarks/questions.json") as f:
        questions = json.load(f)
    with open("benchmarks/ground_truth.json") as f:
        ground_truth = json.load(f)

    gt_by_qid = {item["question_id"]: item for item in ground_truth}

    hits_at_1 = 0
    hits_at_3 = 0
    reciprocal_ranks = []
    separations = []

    print("-" * 80)
    print(f"{'QID':<5} | {'Hit@1':<6} | {'Hit@3':<6} | {'Rank':<5} | {'Score':<7} | {'Distr':<7} | {'Diff':<7} | Topic")
    print("-" * 80)

    for q in questions:
        qid = q["id"]
        gt = gt_by_qid[qid]
        target_chunk_id = gt["target_chunk_id"]

        q_emb = emb_engine.embed_query(q["question"])
        results = store.search(query_vector=q_emb.embedding, top_k=10)

        # Check rankings
        target_rank = None
        target_score = None
        highest_distractor_score = None

        for rank, res in enumerate(results, start=1):
            if res.chunk_id == target_chunk_id:
                if target_rank is None:
                    target_rank = rank
                    target_score = res.score
            else:
                if highest_distractor_score is None:
                    highest_distractor_score = res.score

        if target_rank is not None:
            rr = 1.0 / target_rank
            reciprocal_ranks.append(rr)
            h1 = 1 if target_rank == 1 else 0
            h3 = 1 if target_rank <= 3 else 0
            hits_at_1 += h1
            hits_at_3 += h3
            sep = (target_score - (highest_distractor_score or 0.0))
            separations.append(sep)
            print(f"{qid:<5} | {str(bool(h1)):<6} | {str(bool(h3)):<6} | {target_rank:<5} | {target_score:.4f}  | {(highest_distractor_score or 0.0):.4f}  | {sep:+.4f} | {q['topic']}")
        else:
            reciprocal_ranks.append(0.0)
            separations.append(0.0)
            print(f"{qid:<5} | False  | False  | >10   | N/A     | {(highest_distractor_score or 0.0):.4f}  | N/A     | {q['topic']}")

    hit_rate_1 = hits_at_1 / len(questions)
    hit_rate_3 = hits_at_3 / len(questions)
    mrr = float(np.mean(reciprocal_ranks))
    avg_sep = float(np.mean(separations))

    print("-" * 80)
    print(f"Overall Metrics: Hit@1 = {hit_rate_1*100:.1f}% | Hit@3 = {hit_rate_3*100:.1f}% | MRR = {mrr:.4f} | Avg Separation = {avg_sep:+.4f}")
    assert hit_rate_1 >= 0.50, f"Hit@1 {hit_rate_1} fell below benchmark baseline of 50%"
    assert hit_rate_3 >= 0.75, f"Hit@3 {hit_rate_3} fell below benchmark baseline of 75%"
    assert mrr >= 0.64, f"MRR {mrr} fell below benchmark baseline of 0.6493"

    # -------------------------------------------------------------------------
    # 6. Persistence, Filtering & Embedding-Space Isolation Tests
    # -------------------------------------------------------------------------
    print("\n[Step 6/6] Verifying Metadata Filtering, Cold Reopen & Embedding Isolation...")

    # A. Metadata filtering
    print("  -> Testing metadata filtering on source_file...")
    filtered_lec7 = store.search(
        query_vector=q_emb.embedding,
        top_k=5,
        filters=FilterQuery(source_file="7. DS5104 Lecture7 Autocorrelation ACF.pdf"),
    )
    assert len(filtered_lec7) > 0
    assert all(r.source_file == "7. DS5104 Lecture7 Autocorrelation ACF.pdf" for r in filtered_lec7)
    print(f"     Filtered Lecture 7 search returned {len(filtered_lec7)} records, all strictly from Lecture 7.")

    # B. Cold reopen
    print("  -> Testing cold restart & persistence reopening...")
    store.close()

    reopened_store = LanceDBVectorStore(config=store_config)
    # Reopen with compatible metadata
    reopened_store.initialize(index_meta)
    reopened_meta = reopened_store.get_index_metadata()
    assert reopened_meta is not None
    assert reopened_meta.model_name == "sentence-transformers/all-MiniLM-L6-v2"
    assert reopened_meta.embedding_dimension == 384
    assert reopened_store.count() == 238
    print(f"     Reopened store successfully with verified metadata: {reopened_meta.model_name} ({reopened_store.count()} records).")

    # C. Embedding space isolation
    print("  -> Testing embedding-space isolation (attempting to reopen with BGE)...")
    bge_meta = IndexMetadata(
        model_name="BAAI/bge-small-en-v1.5",
        embedding_dimension=384,
        metric="cosine",
        normalized=True,
    )
    bge_config = VectorStoreConfig(
        backend="lance",
        persist_path=str(db_path),
        collection_name="academic_corpus",
        expected_model_name="BAAI/bge-small-en-v1.5",
        expected_dimension=384,
    )
    isolated_store = LanceDBVectorStore(config=bge_config)
    try:
        isolated_store.initialize(bge_meta)
        print("ERROR: Should have rejected BGE model on MiniLM index!")
        sys.exit(1)
    except IncompatibleModelError as e:
        print(f"     SUCCESS: IncompatibleModelError correctly raised: {e}")

    reopened_store.close()
    print("\n" + "=" * 80)
    print(" ALL STAGE 3.3 SYSTEM CHECKS & VERIFICATIONS PASSED SUCCESSFULLY!")
    print("=" * 80)


if __name__ == "__main__":
    main()
