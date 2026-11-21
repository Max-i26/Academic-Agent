"""Benchmark script for evaluating local LLM candidates on RTX 2050 (4GB VRAM).

Measures:
- Model loading latency and memory footprint.
- VRAM peak consumption (MB) under 2K context window.
- Generation throughput (tokens/second) and first-token latency.
- Academic quality across 5 targeted prompts:
  1. Formula and notation preservation.
  2. Evidence citation adherence ([E1]).
  3. Mathematical reasoning & step-by-step derivation.
  4. Unsupported claim refusal (hallucination defense).
  5. Academic definitions and formal criteria.
"""

import argparse
import json
import logging
import os
import sys
import time
from typing import Any

import psutil
import torch

from academic_agent.generation.models import (
    ChatMessage,
    ContextLengthExceededError,
    CudaOutOfMemoryError,
    GenerationError,
    LLMConfig,
    MessageRole,
    ModelLoadingConfig,
)
from academic_agent.generation.providers.hf_local import HuggingFaceTransformersProvider

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("benchmark")

BENCHMARK_PROMPTS = [
    {
        "id": "P1_formula_preservation",
        "category": "Exact Formula & Notation Preservation",
        "question": (
            "Given the lecture excerpt:\n"
            "\"The multiplicative time series model is $Y_t = T_t \\times S_t \\times I_t$, "
            "whereas additive is $Y_t = T_t + S_t + I_t$.\"\n"
            "Explain when the multiplicative decomposition should be chosen over additive, "
            "and preserve the exact mathematical notation in your explanation."
        ),
        "expected_elements": ["Y_t", "T_t", "S_t", "I_t"],
    },
    {
        "id": "P2_citation_adherence",
        "category": "Evidence Citation Adherence",
        "question": (
            "Answer the question using strictly the provided citations:\n"
            "[E1] A stochastic process is weakly stationary if its mean $\\mu$ is constant over time "
            "and its autocovariance $\\gamma(t, t+k)$ depends only on the lag $k$, not on time $t$.\n"
            "[E2] Non-stationary time series often exhibit trend or seasonal components that vary with time.\n\n"
            "Question: What are the two core conditions of weak stationarity? "
            "You must cite [E1] for every factual statement."
        ),
        "expected_elements": ["[E1]"],
    },
    {
        "id": "P3_step_by_step_derivation",
        "category": "Mathematical Derivation & Calculation",
        "question": (
            "Consider a stationary AR(1) process $Y_t = \\phi_1 Y_{t-1} + \\epsilon_t$, where $|\\phi_1| < 1$ "
            "and $\\epsilon_t \\sim WN(0, \\sigma^2)$ is white noise with $\\text{Cov}(Y_{t-1}, \\epsilon_t) = 0$.\n"
            "Derive the theoretical variance $\\text{Var}(Y_t) = \\frac{\\sigma^2}{1 - \\phi_1^2}$ step by step."
        ),
        "expected_elements": ["1 - \\phi_1^2", "\\sigma^2"],
    },
    {
        "id": "P4_unsupported_claim_refusal",
        "category": "Refusal & Hallucination Resistance",
        "question": (
            "Context:\n"
            "[E1] The autocorrelation function (ACF) for a moving average process of order 1, MA(1), "
            "cuts off abruptly after lag 1 (i.e. $\\rho(k) = 0$ for $|k| > 1$).\n\n"
            "Question: Based solely on the provided context [E1], what is the exact algebraic expression "
            "for the Partial Autocorrelation Function (PACF) of an MA(2) model?"
        ),
        "refusal_expected": True,
        "expected_elements": ["not provided", "cannot", "does not contain", "no information", "not mentioned"],
    },
    {
        "id": "P5_academic_definition",
        "category": "Academic Definition & Formal Criteria",
        "question": (
            "Provide a formal academic definition of a Gaussian White Noise process $W_t$, "
            "and explicitly state its three mathematical properties."
        ),
        "expected_elements": ["mean", "variance", "uncorrelated", "normal", "0"],
    },
]


def get_vram_mb() -> float:
    if torch.cuda.is_available():
        return round(torch.cuda.memory_allocated() / (1024 * 1024), 2)
    return 0.0


def get_peak_vram_mb() -> float:
    if torch.cuda.is_available():
        return round(torch.cuda.max_memory_allocated() / (1024 * 1024), 2)
    return 0.0


def get_ram_mb() -> float:
    process = psutil.Process(os.getpid())
    return round(process.memory_info().rss / (1024 * 1024), 2)


def run_benchmark(
    model_name: str,
    quantization: str | None,
    context_window: int = 2048,
    device: str = "auto",
) -> dict[str, Any]:
    logger.info("=" * 70)
    logger.info("Starting Benchmark: model='%s', quant='%s', device='%s'", model_name, quantization, device)
    logger.info("=" * 70)

    # Initial system state
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()

    initial_vram = get_vram_mb()
    initial_ram = get_ram_mb()

    cfg = ModelLoadingConfig(
        model_name=model_name,
        device=device,
        quantization=quantization,
        context_window=context_window,
        torch_dtype="float16",
    )

    provider = HuggingFaceTransformersProvider(loading_config=cfg)

    # 1. Measure Loading Time & Memory
    logger.info("Loading model weights into memory...")
    t_load_start = time.perf_counter()
    try:
        provider.load()
    except CudaOutOfMemoryError as e:
        logger.error("CUDA OOM during model load: %s", e)
        return {"model_name": model_name, "quantization": quantization, "status": "OOM_LOAD", "error": str(e)}
    except Exception as e:
        logger.error("Error loading model: %s", e)
        return {"model_name": model_name, "quantization": quantization, "status": "ERROR_LOAD", "error": str(e)}

    load_time_sec = round(time.perf_counter() - t_load_start, 2)
    post_load_vram = get_vram_mb()
    weights_vram_mb = round(post_load_vram - initial_vram, 2)
    post_load_ram = get_ram_mb()
    logger.info(
        "Model loaded in %.2f s | Weights VRAM: %.2f MB | System RAM: %.2f MB",
        load_time_sec,
        weights_vram_mb,
        post_load_ram,
    )

    # 2. Run Benchmark Prompts
    prompt_results = []
    total_tokens_gen = 0
    total_gen_time_sec = 0.0

    gen_cfg = LLMConfig(
        temperature=0.1,
        top_p=0.95,
        max_new_tokens=384,
        timeout_seconds=60.0,
    )

    for item in BENCHMARK_PROMPTS:
        p_id = item["id"]
        category = item["category"]
        question = item["question"]
        logger.info("\n--- Prompt: %s (%s) ---", p_id, category)

        messages = [
            ChatMessage(
                role=MessageRole.SYSTEM,
                content="You are an expert academic assistant in time series analysis and econometrics. Answer rigorously.",
            ),
            ChatMessage(role=MessageRole.USER, content=question),
        ]

        # Reset peak stats before inference
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        t0 = time.perf_counter()
        try:
            resp = provider.generate(messages, config=gen_cfg)
            elapsed_sec = time.perf_counter() - t0
        except CudaOutOfMemoryError as oom_e:
            logger.error("CUDA OOM on prompt %s: %s", p_id, oom_e)
            prompt_results.append({
                "id": p_id,
                "status": "OOM",
                "error": str(oom_e),
            })
            break
        except Exception as gen_e:
            logger.error("Generation error on %s: %s", p_id, gen_e)
            prompt_results.append({
                "id": p_id,
                "status": "ERROR",
                "error": str(gen_e),
            })
            continue

        peak_vram = get_peak_vram_mb()
        comp_tokens = resp.usage.completion_tokens
        prompt_tokens = resp.usage.prompt_tokens
        tokens_per_sec = round(comp_tokens / elapsed_sec, 2) if elapsed_sec > 0 else 0.0

        total_tokens_gen += comp_tokens
        total_gen_time_sec += elapsed_sec

        # Quality adherence check
        adherence_passed = False
        content_lower = resp.content.lower()

        if item.get("refusal_expected", False):
            # Check if any refusal phrase matches
            refusal_matches = [e for e in item["expected_elements"] if e.lower() in content_lower]
            adherence_passed = len(refusal_matches) > 0
        else:
            # Check for presence of required notation / citations
            matches = [e for e in item["expected_elements"] if e in resp.content or e.lower() in content_lower]
            adherence_passed = len(matches) >= len(item["expected_elements"]) // 2 + 1

        logger.info(
            "Prompt tokens: %d | Comp tokens: %d | Time: %.2f s | Speed: %.1f tok/s | Peak VRAM: %.1f MB",
            prompt_tokens,
            comp_tokens,
            elapsed_sec,
            tokens_per_sec,
            peak_vram,
        )
        logger.info("Adherence check: %s", "PASS" if adherence_passed else "REVIEW")

        prompt_results.append({
            "id": p_id,
            "category": category,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": comp_tokens,
            "latency_ms": round(elapsed_sec * 1000.0, 1),
            "tokens_per_sec": tokens_per_sec,
            "peak_vram_mb": peak_vram,
            "finish_reason": resp.finish_reason,
            "adherence_passed": adherence_passed,
            "response_snippet": resp.content[:200].replace("\n", " "),
            "full_response": resp.content,
        })

    # 3. Test Memory Release
    logger.info("\nReleasing provider resources...")
    provider.release()
    post_release_vram = get_vram_mb()
    logger.info("Post-release VRAM: %.2f MB (Clean reclaim: %s)", post_release_vram, post_release_vram <= 50.0)

    avg_tok_sec = round(total_tokens_gen / total_gen_time_sec, 2) if total_gen_time_sec > 0 else 0.0
    overall_peak_vram = max([p.get("peak_vram_mb", 0.0) for p in prompt_results] or [post_load_vram])

    return {
        "model_name": model_name,
        "quantization": quantization or "None (FP16)",
        "context_window": context_window,
        "load_time_sec": load_time_sec,
        "weights_vram_mb": weights_vram_mb,
        "overall_peak_vram_mb": overall_peak_vram,
        "post_release_vram_mb": post_release_vram,
        "avg_tokens_per_sec": avg_tok_sec,
        "total_generated_tokens": total_tokens_gen,
        "total_gen_time_sec": round(total_gen_time_sec, 2),
        "prompts": prompt_results,
    }


def main():
    parser = argparse.ArgumentParser(description="Academic-Agent Local Model Benchmark")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--quant", type=str, default="4bit", choices=["4bit", "none", "both"])
    parser.add_argument("--out", type=str, default="benchmark_results.json")
    args = parser.parse_args()

    results = []
    quants_to_test = ["4bit"] if args.quant == "4bit" else [None] if args.quant == "none" else ["4bit", None]

    for q in quants_to_test:
        res = run_benchmark(
            model_name=args.model,
            quantization=q,
            context_window=2048,
        )
        results.append(res)

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    logger.info("Benchmark complete. Results written to %s", args.out)


if __name__ == "__main__":
    main()
