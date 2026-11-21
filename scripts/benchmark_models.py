import json
import time
from pathlib import Path
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from academic_agent.ingestion.layout_extractor import LayoutExtractor
from academic_agent.processing.chunking import ChunkingConfig, ChunkingEngine, EstimateTokenizer

def main():
    print("=" * 70)
    print(" Academic-Agent: Phase 3 Embedding Benchmark (Stage 3.1)")
    print("=" * 70)

    # 1. Extract and Chunk all raw PDFs
    extractor = LayoutExtractor()
    tokenizer = EstimateTokenizer(factor=1.33)
    config = ChunkingConfig(
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
    engine = ChunkingEngine(config=config, tokenizer=tokenizer)

    data_dir = Path("data/raw")
    pdf_files = sorted(data_dir.glob("*.pdf"))
    all_chunks = []
    for pdf_path in pdf_files:
        records = extractor.extract(pdf_path)
        chunks = engine.chunk(records)
        all_chunks.extend(chunks)

    print(f"Loaded {len(all_chunks)} chunks across {len(pdf_files)} PDF lectures.")
    chunk_texts = [c.text for c in all_chunks]
    chunk_ids = [c.chunk_id for c in all_chunks]
    chunk_id_to_idx = {cid: idx for idx, cid in enumerate(chunk_ids)}

    # 2. Load Questions and Ground Truth
    with open("benchmarks/questions.json") as f:
        questions = json.load(f)
    with open("benchmarks/ground_truth.json") as f:
        ground_truth = json.load(f)

    gt_by_qid = {item["question_id"]: item for item in ground_truth}

    models_to_test = [
        {
            "name": "all-MiniLM-L6-v2",
            "repo_id": "sentence-transformers/all-MiniLM-L6-v2",
            "max_seq_length": 256,
            "dim": 384,
            "query_prefix": "",
        },
        {
            "name": "bge-small-en-v1.5",
            "repo_id": "BAAI/bge-small-en-v1.5",
            "max_seq_length": 512,
            "dim": 384,
            "query_prefix": "Represent this sentence for searching relevant passages: ",
        },
    ]

    has_cuda = torch.cuda.is_available()
    device_name = torch.cuda.get_device_name(0) if has_cuda else "None"
    print(f"CUDA Available: {has_cuda} ({device_name})")

    results = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "total_chunks": len(all_chunks),
        "total_questions": len(questions),
        "gpu_device": device_name,
        "models": {},
    }

    results_dir = Path("benchmarks/results")
    results_dir.mkdir(parents=True, exist_ok=True)

    for m_info in models_to_test:
        m_name = m_info["name"]
        repo_id = m_info["repo_id"]
        q_prefix = m_info["query_prefix"]
        max_len = m_info["max_seq_length"]
        print(f"\nEvaluating Model: {m_name} ({repo_id})...")

        # Load model on CPU first
        model_cpu = SentenceTransformer(repo_id, device="cpu")
        hf_tokenizer = model_cpu.tokenizer

        # A. Exact Tokenization Audit
        print("  -> Running exact tokenization audit on all 236 chunks...")
        token_counts = []
        for text in chunk_texts:
            tokens = hf_tokenizer.encode(text, add_special_tokens=True)
            token_counts.append(len(tokens))

        avg_tokens = float(np.mean(token_counts))
        max_tokens = int(np.max(token_counts))
        min_tokens = int(np.min(token_counts))
        over_256 = sum(1 for count in token_counts if count > 256)
        over_512 = sum(1 for count in token_counts if count > 512)

        # Audit queries with prefix
        query_token_counts = []
        for q in questions:
            q_text = q_prefix + q["question"]
            tokens = hf_tokenizer.encode(q_text, add_special_tokens=True)
            query_token_counts.append(len(tokens))

        print(f"     Avg tokens: {avg_tokens:.2f} | Max tokens: {max_tokens} | Min tokens: {min_tokens}")
        print(f"     > 256 tokens: {over_256} ({over_256/len(chunk_texts)*100:.1f}%) | > 512 tokens: {over_512}")
        print(f"     Query token length avg: {np.mean(query_token_counts):.1f} | max: {np.max(query_token_counts)}")

        # B. Latency and Throughput on CPU
        print("  -> Benchmarking CPU inference...")
        # Warmup
        _ = model_cpu.encode(chunk_texts[:16], batch_size=16, normalize_embeddings=True)
        t0 = time.perf_counter()
        cpu_embeddings = model_cpu.encode(chunk_texts, batch_size=32, normalize_embeddings=True, show_progress_bar=False)
        cpu_elapsed = time.perf_counter() - t0
        cpu_latency_per_chunk_ms = (cpu_elapsed / len(chunk_texts)) * 1000.0
        cpu_throughput = len(chunk_texts) / cpu_elapsed
        print(f"     CPU Latency: {cpu_latency_per_chunk_ms:.2f} ms/chunk | Throughput: {cpu_throughput:.2f} chunks/sec (Total: {cpu_elapsed:.2f}s)")

        # C. Latency, Throughput and VRAM on GPU (if available)
        gpu_latency_per_chunk_ms = None
        gpu_throughput = None
        peak_allocated_mb = None
        peak_reserved_mb = None

        if has_cuda:
            print("  -> Benchmarking GPU inference...")
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            model_gpu = SentenceTransformer(repo_id, device="cuda")

            # Warmup
            _ = model_gpu.encode(chunk_texts[:16], batch_size=16, normalize_embeddings=True)
            torch.cuda.synchronize()

            t0 = time.perf_counter()
            gpu_embeddings = model_gpu.encode(chunk_texts, batch_size=32, normalize_embeddings=True, show_progress_bar=False)
            torch.cuda.synchronize()
            gpu_elapsed = time.perf_counter() - t0

            gpu_latency_per_chunk_ms = (gpu_elapsed / len(chunk_texts)) * 1000.0
            gpu_throughput = len(chunk_texts) / gpu_elapsed
            peak_allocated_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
            peak_reserved_mb = torch.cuda.max_memory_reserved() / (1024 ** 2)

            print(f"     GPU Latency: {gpu_latency_per_chunk_ms:.2f} ms/chunk | Throughput: {gpu_throughput:.2f} chunks/sec (Total: {gpu_elapsed:.2f}s)")
            print(f"     VRAM Allocated: {peak_allocated_mb:.2f} MB | VRAM Reserved: {peak_reserved_mb:.2f} MB")
            chunk_embeddings = gpu_embeddings
            del model_gpu
            torch.cuda.empty_cache()
        else:
            chunk_embeddings = cpu_embeddings

        del model_cpu

        # D. Retrieval Quality on Ground Truth Questions
        print("  -> Evaluating retrieval metrics on 8 benchmark questions...")
        query_texts = [q_prefix + q["question"] for q in questions]

        # Use CPU model or sentence-transformer encoder for query embeddings
        eval_model = SentenceTransformer(repo_id, device="cuda" if has_cuda else "cpu")
        query_embeddings = eval_model.encode(query_texts, batch_size=len(query_texts), normalize_embeddings=True)

        question_evals = []
        hits_at_1 = []
        hits_at_3 = []
        reciprocal_ranks = []
        score_separations = []

        for i, q in enumerate(questions):
            qid = q["id"]
            gt = gt_by_qid[qid]
            target_cid = gt["target_chunk_id"]
            target_idx = chunk_id_to_idx[target_cid]

            q_vec = query_embeddings[i]
            # Cosine similarity is dot product of normalized embeddings
            sims = np.dot(chunk_embeddings, q_vec)

            # Sort descending
            ranked_indices = np.argsort(-sims)
            rank = int(np.where(ranked_indices == target_idx)[0][0]) + 1

            target_score = float(sims[target_idx])
            top_distractor_idx = ranked_indices[0] if ranked_indices[0] != target_idx else ranked_indices[1]
            top_distractor_score = float(sims[top_distractor_idx])
            separation = target_score - top_distractor_score

            hit_1 = 1 if rank == 1 else 0
            hit_3 = 1 if rank <= 3 else 0
            rr = 1.0 / rank

            hits_at_1.append(hit_1)
            hits_at_3.append(hit_3)
            reciprocal_ranks.append(rr)
            score_separations.append(separation)

            top_distractor_cid = chunk_ids[top_distractor_idx]
            distractor_chunk = all_chunks[top_distractor_idx]

            q_eval = {
                "question_id": qid,
                "question": q["question"],
                "target_chunk_id": target_cid,
                "target_rank": rank,
                "hit@1": hit_1,
                "hit@3": hit_3,
                "reciprocal_rank": rr,
                "target_similarity": target_score,
                "top_distractor_chunk_id": top_distractor_cid,
                "top_distractor_similarity": top_distractor_score,
                "score_separation": separation,
                "top_distractor_preview": distractor_chunk.text[:120].replace("\n", " "),
            }
            question_evals.append(q_eval)

            status = "HIT@1" if rank == 1 else ("HIT@3" if rank <= 3 else f"RANK #{rank}")
            print(f"     [{qid}] Rank: #{rank} ({status}) | Target Sim: {target_score:.4f} | Distractor: {top_distractor_score:.4f} | Sep: {separation:+.4f}")

        del eval_model
        if has_cuda:
            torch.cuda.empty_cache()

        mean_hit_1 = float(np.mean(hits_at_1))
        mean_hit_3 = float(np.mean(hits_at_3))
        mrr = float(np.mean(reciprocal_ranks))
        mean_separation = float(np.mean(score_separations))

        print(f"\n  Summary for {m_name}:")
        print(f"    Hit@1: {mean_hit_1 * 100:.1f}%")
        print(f"    Hit@3: {mean_hit_3 * 100:.1f}%")
        print(f"    MRR:   {mrr:.4f}")
        print(f"    Avg Score Separation: {mean_separation:+.4f}")

        results["models"][m_name] = {
            "repo_id": repo_id,
            "dimension": m_info["dim"],
            "max_sequence_length": max_len,
            "tokenization": {
                "avg_tokens": avg_tokens,
                "max_tokens": max_tokens,
                "min_tokens": min_tokens,
                "chunks_exceeding_256": over_256,
                "chunks_exceeding_512": over_512,
                "query_token_avg": float(np.mean(query_token_counts)),
                "query_token_max": int(np.max(query_token_counts)),
            },
            "latency": {
                "cpu_ms_per_chunk": cpu_latency_per_chunk_ms,
                "cpu_chunks_per_sec": cpu_throughput,
                "gpu_ms_per_chunk": gpu_latency_per_chunk_ms,
                "gpu_chunks_per_sec": gpu_throughput,
            },
            "memory": {
                "gpu_peak_allocated_mb": peak_allocated_mb,
                "gpu_peak_reserved_mb": peak_reserved_mb,
            },
            "retrieval": {
                "hit_at_1": mean_hit_1,
                "hit_at_3": mean_hit_3,
                "mrr": mrr,
                "mean_separation": mean_separation,
                "question_breakdown": question_evals,
            },
        }

    # Save results to JSON
    results_path = results_dir / "benchmark_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved benchmark results to {results_path}")

    # Print Comparison Table
    print("\n" + "=" * 80)
    print("FINAL BENCHMARK COMPARISON TABLE")
    print("=" * 80)
    header = f"{'Metric':<32} | {'all-MiniLM-L6-v2':<20} | {'bge-small-en-v1.5':<20}"
    print(header)
    print("-" * 80)
    m1 = results["models"]["all-MiniLM-L6-v2"]
    m2 = results["models"]["bge-small-en-v1.5"]

    rows = [
        ("Vector Dimension", f"{m1['dimension']}", f"{m2['dimension']}"),
        ("Max Sequence Limit", f"{m1['max_sequence_length']} tokens", f"{m2['max_sequence_length']} tokens"),
        ("Avg Chunk Tokens (Actual)", f"{m1['tokenization']['avg_tokens']:.1f}", f"{m2['tokenization']['avg_tokens']:.1f}"),
        ("Max Chunk Tokens (Actual)", f"{m1['tokenization']['max_tokens']}", f"{m2['tokenization']['max_tokens']}"),
        ("Chunks Truncated (> Max)", f"{m1['tokenization']['chunks_exceeding_256']}", f"{m2['tokenization']['chunks_exceeding_512']}"),
        ("CPU Latency (ms/chunk)", f"{m1['latency']['cpu_ms_per_chunk']:.2f} ms", f"{m2['latency']['cpu_ms_per_chunk']:.2f} ms"),
        ("CPU Throughput (chunks/s)", f"{m1['latency']['cpu_chunks_per_sec']:.1f} doc/s", f"{m2['latency']['cpu_chunks_per_sec']:.1f} doc/s"),
        ("GPU Latency (ms/chunk)", f"{m1['latency']['gpu_ms_per_chunk']:.2f} ms", f"{m2['latency']['gpu_ms_per_chunk']:.2f} ms"),
        ("GPU Throughput (chunks/s)", f"{m1['latency']['gpu_chunks_per_sec']:.1f} doc/s", f"{m2['latency']['gpu_chunks_per_sec']:.1f} doc/s"),
        ("GPU VRAM Peak Allocated", f"{m1['memory']['gpu_peak_allocated_mb']:.1f} MB", f"{m2['memory']['gpu_peak_allocated_mb']:.1f} MB"),
        ("GPU VRAM Peak Reserved", f"{m1['memory']['gpu_peak_reserved_mb']:.1f} MB", f"{m2['memory']['gpu_peak_reserved_mb']:.1f} MB"),
        ("Hit Rate @ 1", f"{m1['retrieval']['hit_at_1']*100:.1f}%", f"{m2['retrieval']['hit_at_1']*100:.1f}%"),
        ("Hit Rate @ 3", f"{m1['retrieval']['hit_at_3']*100:.1f}%", f"{m2['retrieval']['hit_at_3']*100:.1f}%"),
        ("MRR (Mean Reciprocal Rank)", f"{m1['retrieval']['mrr']:.4f}", f"{m2['retrieval']['mrr']:.4f}"),
        ("Avg Score Separation", f"{m1['retrieval']['mean_separation']:+.4f}", f"{m2['retrieval']['mean_separation']:+.4f}"),
    ]
    for row in rows:
        print(f"{row[0]:<32} | {row[1]:<20} | {row[2]:<20}")
    print("=" * 80)

if __name__ == "__main__":
    main()
