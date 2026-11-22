"""Step 2B Smoke Test: Real MiniLM Embedding Generation and Physical LanceDB Retrieval."""

import os
import sys
import time
import json
from pathlib import Path
import numpy as np
import torch

# Force strictly offline execution
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from academic_agent.embeddings.config import EmbeddingConfig, KnownEmbeddingModel
from academic_agent.embeddings.engine import EmbeddingEngine
from academic_agent.embeddings.providers.minilm import MiniLMProvider
from academic_agent.storage.models import IndexMetadata
from academic_agent.storage.config import VectorStoreConfig
from academic_agent.storage.providers.lancedb.persistence import LancePersistenceManager
from academic_agent.storage.providers.lancedb.store import LanceDBVectorStore
from academic_agent.retrieval.dense_retriever import DenseRetriever
from academic_agent.retrieval.pipeline import RetrievalPipeline
from academic_agent.retrieval.models import RetrievalQuery, RetrievalStrategy

results = {}

print("=" * 60)
print("STEP 2B: REAL MINILM EMBEDDING AND PHYSICAL RETRIEVAL SMOKE TEST")
print("=" * 60)

# 1. OFFLINE MODEL LOADING
print("\n--- 1. Offline Model Loading ---")
if torch.cuda.is_available():
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()

t0 = time.perf_counter()
cuda_provider = MiniLMProvider(model_name=KnownEmbeddingModel.MINILM.value, device="cuda")
cuda_load_time_ms = (time.perf_counter() - t0) * 1000.0

cuda_engine = EmbeddingEngine(provider=cuda_provider)
device_used = cuda_provider.device
peak_gpu_mem_load_mb = torch.cuda.max_memory_allocated() / (1024 ** 2) if torch.cuda.is_available() else 0.0

print(f"CUDA MiniLM Provider Loaded in: {cuda_load_time_ms:.2f} ms")
print(f"Selected Device: {device_used}")
print(f"Peak GPU Memory after Model Load: {peak_gpu_mem_load_mb:.2f} MB")

results["load_time_ms"] = round(cuda_load_time_ms, 2)
results["device_used"] = device_used
results["peak_gpu_mem_load_mb"] = round(peak_gpu_mem_load_mb, 2)

# Also load CPU provider to test CPU execution and fallback
t0_cpu = time.perf_counter()
cpu_provider = MiniLMProvider(model_name=KnownEmbeddingModel.MINILM.value, device="cpu")
cpu_load_time_ms = (time.perf_counter() - t0_cpu) * 1000.0
cpu_engine = EmbeddingEngine(provider=cpu_provider)
print(f"CPU MiniLM Provider Loaded in: {cpu_load_time_ms:.2f} ms")

# 2. ACTUAL EMBEDDING GENERATION
print("\n--- 2. Actual Embedding Generation (CUDA vs CPU) ---")
test_queries = [
    "What are the stationarity conditions for an AR(1) process?",
    "How does simple exponential smoothing compute level updates with alpha?",
    "Explain the sample autocorrelation function ACF and correlogram.",
    "Discuss classical additive and multiplicative trend seasonal decomposition.",
    "Difference between stationary and non-stationary time series with unit roots.",
]

# Reset peak memory stats before embedding
if torch.cuda.is_available():
    torch.cuda.reset_peak_memory_stats()

t_embed_0 = time.perf_counter()
cuda_records = [cuda_engine.embed_query(q) for q in test_queries]
cuda_latency_ms = (time.perf_counter() - t_embed_0) * 1000.0
cuda_vecs = np.vstack([r.embedding for r in cuda_records])
peak_gpu_mem_embed_mb = torch.cuda.max_memory_allocated() / (1024 ** 2) if torch.cuda.is_available() else 0.0

t_embed_cpu_0 = time.perf_counter()
cpu_records = [cpu_engine.embed_query(q) for q in test_queries]
cpu_latency_ms = (time.perf_counter() - t_embed_cpu_0) * 1000.0
cpu_vecs = np.vstack([r.embedding for r in cpu_records])

print(f"Batch Size: {len(test_queries)} queries")
print(f"CUDA Embedding Latency (total): {cuda_latency_ms:.2f} ms ({cuda_latency_ms / len(test_queries):.2f} ms per query)")
print(f"CPU Embedding Latency (total):  {cpu_latency_ms:.2f} ms ({cpu_latency_ms / len(test_queries):.2f} ms per query)")
print(f"Peak GPU Memory during Embedding: {peak_gpu_mem_embed_mb:.2f} MB")

# Shape and Norm Validations
assert cuda_vecs.shape == (5, 384), f"Unexpected shape: {cuda_vecs.shape}"
assert np.all(np.isfinite(cuda_vecs)), "Non-finite values detected in CUDA embeddings!"

cuda_norms = np.linalg.norm(cuda_vecs, axis=1)
norm_min = float(np.min(cuda_norms))
norm_max = float(np.max(cuda_norms))
print(f"Vector Shape: {cuda_vecs.shape}")
print(f"Vector L2 Norms min: {norm_min:.6f}, max: {norm_max:.6f}")
assert abs(norm_min - 1.0) < 1e-4 and abs(norm_max - 1.0) < 1e-4, "Vectors are not unit normalized!"

# CUDA vs CPU Numerical Comparison
max_diff = float(np.max(np.abs(cuda_vecs - cpu_vecs)))
mean_diff = float(np.mean(np.abs(cuda_vecs - cpu_vecs)))
cos_sims = [float(np.dot(c, p) / (np.linalg.norm(c) * np.linalg.norm(p))) for c, p in zip(cuda_vecs, cpu_vecs)]
min_cos_sim = min(cos_sims)
print(f"CUDA vs CPU Max Absolute Difference: {max_diff:.6e}")
print(f"CUDA vs CPU Mean Absolute Difference: {mean_diff:.6e}")
print(f"CUDA vs CPU Minimum Cosine Similarity: {min_cos_sim:.6f}")
assert min_cos_sim > 0.9999, f"CUDA and CPU embeddings deviate significantly! Min cosine: {min_cos_sim}"

# Determinism Check (re-running on CUDA produces identical vectors)
re_cuda_vecs = np.vstack([cuda_engine.embed_query(q).embedding for q in test_queries])
determinism_diff = float(np.max(np.abs(cuda_vecs - re_cuda_vecs)))
print(f"CUDA Determinism Max Difference: {determinism_diff:.6e}")
assert determinism_diff < 1e-6, "CUDA embedding generation is non-deterministic!"

results["embedding_shape"] = list(cuda_vecs.shape)
results["cuda_latency_ms"] = round(cuda_latency_ms, 2)
results["cpu_latency_ms"] = round(cpu_latency_ms, 2)
results["peak_gpu_mem_embed_mb"] = round(peak_gpu_mem_embed_mb, 2)
results["norm_min"] = round(norm_min, 6)
results["norm_max"] = round(norm_max, 6)
results["cuda_cpu_max_diff"] = max_diff
results["cuda_cpu_min_cos_sim"] = min_cos_sim
results["determinism_diff"] = determinism_diff

# 3. PHYSICAL RETRIEVAL FROM LANCEDB CORPUS
print("\n--- 3. Physical Retrieval Against Persisted LanceDB Corpus ---")
db_path = Path("data/lancedb")
expected_meta = IndexMetadata(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    embedding_dimension=384,
    distance_metric="cosine",
    normalized=True,
)

pm = LancePersistenceManager(
    db_path=db_path,
    collection_name="academic_corpus",
    expected_metadata=expected_meta,
)

store = LanceDBVectorStore(
    config=VectorStoreConfig(
        persist_path=str(db_path),
        collection_name="academic_corpus",
        similarity_threshold=0.0,
    ),
    persistence_manager=pm,
)

count_before = store.count()
print(f"Persisted Table: academic_corpus")
print(f"Initial Record Count: {count_before}")
assert count_before == 238, f"Expected 238 records, found {count_before}"

dense_retriever = DenseRetriever(
    embedding_engine=cuda_engine,
    vector_store=store,
)
retrieval_pipeline = RetrievalPipeline(dense_retriever=dense_retriever)

retrieval_cases = [
    {
        "query": "Simple exponential smoothing parameter alpha and level equation",
        "expected_topic": "Smoothing",
    },
    {
        "query": "Autocorrelation function ACF and sample correlogram r_k",
        "expected_topic": "Autocorrelation",
    },
    {
        "query": "Additive and multiplicative trend seasonal decomposition",
        "expected_topic": "Decomposition",
    },
]

retrieval_outputs = []
for idx, item in enumerate(retrieval_cases, 1):
    q_str = item["query"]
    rq = RetrievalQuery(text=q_str, top_k=3, strategy=RetrievalStrategy.DENSE)
    resp = retrieval_pipeline.search(rq)
    
    assert len(resp.evidence) >= 1, f"No evidence retrieved for query: {q_str}"
    top_ev = resp.evidence[0]
    
    # Assert structural provenance integrity
    assert top_ev.chunk_id, "Top evidence missing chunk_id!"
    assert top_ev.source_file, "Top evidence missing source_file!"
    assert top_ev.source_pages, "Top evidence missing source_pages!"
    assert top_ev.score is not None and top_ev.score > 0.0, "Top evidence missing valid similarity score!"
    assert len(top_ev.text.strip()) > 20, "Top evidence text too short or empty!"
    
    q_out = {
        "case": idx,
        "query": q_str,
        "top_chunk_id": top_ev.chunk_id,
        "top_source_file": top_ev.source_file,
        "top_source_pages": list(top_ev.source_pages),
        "top_similarity_score": round(float(top_ev.score), 4),
        "citation": top_ev.format_citation() if hasattr(top_ev, "format_citation") else f"[{top_ev.source_file}]",
        "snippet": top_ev.text[:120].replace("\n", " ") + "...",
    }
    retrieval_outputs.append(q_out)
    print(f"\n[Query {idx}]: {q_str}")
    print(f"  Top Match: {q_out['top_chunk_id']} ({q_out['top_source_file']}, pages {q_out['top_source_pages']})")
    print(f"  Similarity Score: {q_out['top_similarity_score']:.4f}")
    print(f"  Snippet: {q_out['snippet']}")

# Confirm read-only persistence guarantee
count_after = store.count()
print(f"\nFinal Record Count after Retrieval: {count_after}")
assert count_after == count_before == 238, "CRITICAL FAILURE: Physical table was modified during retrieval!"
store.close()

results["retrieval_verified"] = True
results["retrieval_cases"] = retrieval_outputs
results["table_count_verified"] = count_after

# 4. FAILURE HANDLING CONTRACT VERIFICATION
print("\n--- 4. Failure Handling Contract Verification ---")

# 4.1 Empty input
try:
    cuda_engine.embed_query("")
    assert False, "Should have raised ValueError on empty query"
except ValueError as e:
    print(f"  [PASS] Empty query rejected: {e}")

# 4.2 Whitespace-only input
try:
    cuda_engine.embed_query("    \t\n  ")
    assert False, "Should have raised ValueError on whitespace query"
except ValueError as e:
    print(f"  [PASS] Whitespace query rejected: {e}")

# 4.3 Dimension mismatch rejection in LanceDB store
bad_dim_vector = np.ones(128, dtype=np.float32)
bad_dim_vector /= np.linalg.norm(bad_dim_vector)
pm_test = LancePersistenceManager(db_path=db_path, collection_name="academic_corpus", expected_metadata=expected_meta)
store_test = LanceDBVectorStore(config=VectorStoreConfig(similarity_threshold=0.0), persistence_manager=pm_test)
try:
    store_test.search(query_vector=bad_dim_vector, top_k=3)
    assert False, "Should have raised ValueError on mismatched vector dimension"
except ValueError as e:
    print(f"  [PASS] Dimension mismatch (128 vs 384) rejected: {e}")
store_test.close()

# 4.4 Non-normalized vector rejection
unnorm_vector = np.ones(384, dtype=np.float32) * 5.0
store_test2 = LanceDBVectorStore(config=VectorStoreConfig(similarity_threshold=0.0), persistence_manager=pm_test)
try:
    store_test2.search(query_vector=unnorm_vector, top_k=3)
    assert False, "Should have raised ValueError on unnormalized query vector"
except ValueError as e:
    print(f"  [PASS] Unnormalized query vector rejected: {e}")
store_test2.close()

# 4.5 Offline model loading failure for non-cached model
try:
    MiniLMProvider(model_name="sentence-transformers/nonexistent-model-xyz", device="cpu")
    assert False, "Should have failed loading non-cached model in offline mode"
except Exception as e:
    print(f"  [PASS] Offline non-cached model loading failed gracefully: {type(e).__name__}")

print("\n" + "=" * 60)
print("ALL STEP 2B CHECKS PASSED SUCCESSFULLY!")
print("=" * 60)

with open("data/evaluation/step2b_minilm_smoke_results.json", "w") as f:
    json.dump(results, f, indent=2)
