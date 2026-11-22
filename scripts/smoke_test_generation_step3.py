"""Gate 3: Real Local LLM Generation & Grounded Synthesis Smoke Test.

Verifies:
1. Offline loading of Qwen/Qwen2.5-1.5B-Instruct with 4-bit NF4 quantization on CUDA.
2. Grounded synthesis through AcademicSynthesizer with real course evidence.
3. Citation validation, integrity checking, and non-destructive resolution.
4. Policy outcome evaluation and hallucination detection on real model output.
5. Abstention policy enforcement when evidence is empty / insufficient.
6. Strict token bounding (max_new_tokens ceiling respected).
7. Resource footprint and full GPU memory reclamation on provider release.
"""

import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("smoke_test_step3")

# Enforce offline mode
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import torch

from academic_agent.generation.models import (
    ChatMessage,
    LLMConfig,
    MessageRole,
    ModelLoadingConfig,
)
from academic_agent.generation.providers.hf_local import HuggingFaceTransformersProvider
from academic_agent.generation.synthesis.context.evidence_context_builder import (
    EvidenceContextBuilder,
)
from academic_agent.generation.synthesis.models import (
    AcademicAnswer,
    EvidenceSufficiency,
    PolicyOutcome,
    PolicyValidationResult,
    ValidationDiagnostics,
)
from academic_agent.generation.synthesis.prompt.grounding_prompt_builder import (
    GroundingPromptBuilder,
)
from academic_agent.generation.synthesis.synthesizer import AcademicSynthesizer
from academic_agent.generation.synthesis.validation.answer_policy_validator import (
    AnswerPolicyValidator,
)
from academic_agent.generation.synthesis.validation.citation_validator import (
    CitationValidator,
)
from academic_agent.retrieval.models import (
    Evidence,
    RetrievalQuery,
    RetrievalResponse,
    RetrievalStrategy,
)


def run_gate3_tests() -> dict[str, Any]:
    print("=" * 60)
    print("GATE 3: REAL LOCAL LLM GENERATION & GROUNDED SYNTHESIS SMOKE TEST")
    print("=" * 60)

    results: dict[str, Any] = {}

    # -------------------------------------------------------------
    # 1. Environment & Device Pre-Check
    # -------------------------------------------------------------
    assert torch.cuda.is_available(), "CUDA must be available for Gate 3 test."
    device_name = torch.cuda.get_device_name(0)
    initial_vram_mb = torch.cuda.memory_allocated() / (1024**2)
    print(f"Device: {device_name}")
    print(f"Initial VRAM Allocated: {initial_vram_mb:.2f} MB")
    results["device"] = device_name
    results["initial_vram_mb"] = initial_vram_mb

    # -------------------------------------------------------------
    # 2. Offline Model Loading with 4-bit NF4 Quantization
    # -------------------------------------------------------------
    print("\n--- 1. Offline Model Loading (Qwen2.5-1.5B-Instruct, 4-bit NF4) ---")
    model_name = "Qwen/Qwen2.5-1.5B-Instruct"
    loading_config = ModelLoadingConfig(
        model_name=model_name,
        device="cuda",
        torch_dtype="float16",
        quantization="4bit",
        context_window=4096,
    )

    t0 = time.perf_counter()
    provider = HuggingFaceTransformersProvider(loading_config=loading_config)
    provider.load()
    load_time_ms = (time.perf_counter() - t0) * 1000.0

    post_load_vram_mb = torch.cuda.memory_allocated() / (1024**2)
    post_load_reserved_mb = torch.cuda.memory_reserved() / (1024**2)

    print(f"Model Loaded in: {load_time_ms:.2f} ms")
    print(f"VRAM Allocated: {post_load_vram_mb:.2f} MB")
    print(f"VRAM Reserved:  {post_load_reserved_mb:.2f} MB")
    assert provider.is_loaded, "Provider must be loaded."
    assert provider.device == "cuda", "Provider must be on cuda."
    assert post_load_vram_mb < 2000.0, f"Expected 4-bit allocation < 2GB, got {post_load_vram_mb:.2f} MB"

    results["model_name"] = model_name
    results["load_time_ms"] = round(load_time_ms, 2)
    results["post_load_vram_mb"] = round(post_load_vram_mb, 2)
    results["post_load_reserved_mb"] = round(post_load_reserved_mb, 2)

    # -------------------------------------------------------------
    # 3. Assemble Academic Synthesizer Pipeline
    # -------------------------------------------------------------
    synthesizer = AcademicSynthesizer(
        llm_provider=provider,
        context_builder=EvidenceContextBuilder(),
        prompt_builder=GroundingPromptBuilder(),
        citation_validator=CitationValidator(),
        policy_validator=AnswerPolicyValidator(),
        default_evidence_budget=1500,
    )

    # -------------------------------------------------------------
    # 4. Grounded Synthesis Test with Real Lecture Evidence
    # -------------------------------------------------------------
    print("\n--- 2. Grounded Synthesis with Real Lecture Evidence ---")
    query_text = "What is the Simple Exponential Smoothing (SES) level equation and what does the smoothing parameter alpha represent?"

    # Real evidence from DS5104 Lecture 4
    ev1 = Evidence(
        record_id="rec_ses_1#0",
        chunk_id="1a71a9327db35525",
        text=(
            "THE SES EQUATION\n"
            "The new level is a weighted compromise between the current observation and previous level:\n"
            "S_t = αY_t + (1 - α)S_{t-1}\n"
            "α = smoothing parameter, 0 < α < 1\n"
            "Y_t = current observation\n"
            "S_{t-1} = previous level estimate\n"
            "S_t = new smoothed level"
        ),
        source_file="4. DS5104 Lecture4 Simple_Exponential_Smoothing.pdf",
        source_pages=[7],
        score=0.92,
    )
    retrieval_resp = RetrievalResponse(
        query=RetrievalQuery(text=query_text),
        evidence=[ev1],
        candidates=[],
        strategy=RetrievalStrategy.HYBRID,
    )

    t0 = time.perf_counter()
    answer: AcademicAnswer = synthesizer.synthesize(
        query=query_text,
        retrieval_response=retrieval_resp,
        config=LLMConfig(max_new_tokens=150, temperature=0.1),
    )
    synth_latency_ms = (time.perf_counter() - t0) * 1000.0

    print(f"Synthesis Latency: {synth_latency_ms:.2f} ms")
    print(f"Evidence Sufficiency: {answer.evidence_sufficiency}")
    print(f"Policy Outcome: {answer.policy_outcome}")
    print(f"Is Compliant: {answer.is_policy_compliant}")
    print(f"Valid Cited IDs: {answer.validation.valid_cited_ids}")
    print(f"Invalid Cited IDs: {answer.validation.invalid_cited_ids}")
    print(f"Malformed Citations: {answer.validation.malformed_citations}")
    print(f"Has Valid Citation Integrity: {answer.has_valid_citation_integrity}")
    print(f"Policy Explanation: {answer.policy_result.explanation if answer.policy_result else ''}")
    print(f"Raw Answer Text:\n{answer.answer_text.strip()}\n")
    print(f"Resolved Answer Text:\n{answer.resolved_answer_text.strip()}\n")

    # Assertions for Grounded Synthesis Pipeline
    assert answer.answer_text.strip(), "Answer text must not be empty."
    assert answer.raw_response.usage.completion_tokens > 0, "Must report generated tokens."
    assert isinstance(answer.validation, ValidationDiagnostics), "Validation diagnostics must be populated."
    assert isinstance(answer.policy_result, PolicyValidationResult), "Policy result must be populated."
    # Check that the answer text contains the formula or mathematical variables from evidence
    assert "S_t" in answer.answer_text or "alpha" in answer.answer_text or "α" in answer.answer_text, (
        f"Answer text must reference SES variables: {answer.answer_text}"
    )

    results["grounded_synthesis"] = {
        "query": query_text,
        "latency_ms": round(synth_latency_ms, 2),
        "evidence_sufficiency": str(answer.evidence_sufficiency),
        "policy_outcome": str(answer.policy_outcome),
        "is_compliant": answer.is_policy_compliant,
        "valid_cited_ids": list(answer.validation.valid_cited_ids),
        "invalid_cited_ids": list(answer.validation.invalid_cited_ids),
        "has_valid_citation_integrity": answer.has_valid_citation_integrity,
        "policy_explanation": answer.policy_result.explanation if answer.policy_result else "",
        "raw_answer": answer.answer_text.strip(),
        "resolved_answer": answer.resolved_answer_text.strip(),
        "completion_tokens": answer.raw_response.usage.completion_tokens,
    }

    # -------------------------------------------------------------
    # 5. Abstention & Refusal on Insufficient Evidence
    # -------------------------------------------------------------
    print("\n--- 3. Abstention & Policy Enforcement on Empty Evidence ---")
    out_of_domain_query = "What is the protocol for Paxos consensus leader election in distributed systems?"
    empty_retrieval_resp = RetrievalResponse(
        query=RetrievalQuery(text=out_of_domain_query),
        evidence=[],
        candidates=[],
        strategy=RetrievalStrategy.HYBRID,
    )

    t0 = time.perf_counter()
    abstention_answer: AcademicAnswer = synthesizer.synthesize(
        query=out_of_domain_query,
        retrieval_response=empty_retrieval_resp,
        config=LLMConfig(max_new_tokens=60, temperature=0.1),
    )
    abstention_latency_ms = (time.perf_counter() - t0) * 1000.0

    print(f"Abstention Latency: {abstention_latency_ms:.2f} ms")
    print(f"Sufficiency: {abstention_answer.evidence_sufficiency}")
    print(f"Policy Outcome: {abstention_answer.policy_outcome}")
    print(f"Refusal Detected: {abstention_answer.refusal_detected}")
    print(f"Is Compliant Abstention: {abstention_answer.is_compliant_abstention}")
    print(f"Policy Explanation: {abstention_answer.policy_result.explanation if abstention_answer.policy_result else ''}")
    print(f"Raw Answer Text:\n{abstention_answer.answer_text.strip()}\n")

    # Assertions for Abstention & Policy Enforcement:
    # 1. Sufficiency must be INSUFFICIENT when evidence package is empty
    assert abstention_answer.evidence_sufficiency == EvidenceSufficiency.INSUFFICIENT, (
        f"Expected INSUFFICIENT, got {abstention_answer.evidence_sufficiency}"
    )
    # 2. When empty evidence is passed, the policy engine MUST either detect compliant abstention
    # OR flag ungrounded parametric hallucination as UNSUPPORTED_ANSWER
    assert abstention_answer.policy_outcome in (
        PolicyOutcome.COMPLIANT_ABSTENTION,
        PolicyOutcome.UNSUPPORTED_ANSWER,
    ), f"Unexpected policy outcome on empty evidence: {abstention_answer.policy_outcome}"

    # If the model refused, compliant abstention is certified.
    # If the model hallucinated from parametric memory without evidence, policy engine correctly marks it uncompliant!
    if abstention_answer.refusal_detected:
        assert abstention_answer.policy_outcome == PolicyOutcome.COMPLIANT_ABSTENTION
        assert abstention_answer.is_compliant_abstention
        print("Model compliant refusal certified.")
    else:
        assert abstention_answer.policy_outcome == PolicyOutcome.UNSUPPORTED_ANSWER
        assert not abstention_answer.is_policy_compliant
        print("Parametric hallucination correctly caught and rejected by AnswerPolicyValidator.")

    results["abstention_test"] = {
        "query": out_of_domain_query,
        "latency_ms": round(abstention_latency_ms, 2),
        "evidence_sufficiency": str(abstention_answer.evidence_sufficiency),
        "policy_outcome": str(abstention_answer.policy_outcome),
        "refusal_detected": abstention_answer.refusal_detected,
        "is_compliant_abstention": abstention_answer.is_compliant_abstention,
        "is_policy_compliant": abstention_answer.is_policy_compliant,
        "policy_explanation": abstention_answer.policy_result.explanation if abstention_answer.policy_result else "",
        "raw_text": abstention_answer.answer_text.strip(),
    }

    # -------------------------------------------------------------
    # 6. Token Bounding Contract Verification
    # -------------------------------------------------------------
    print("\n--- 4. Token Bounding Contract Verification ---")
    bound_test_tokens = 14
    messages = [
        ChatMessage(role=MessageRole.SYSTEM, content="You are a helpful assistant."),
        ChatMessage(role=MessageRole.USER, content="Explain the history of the universe in great detail."),
    ]
    t0 = time.perf_counter()
    bounded_resp = provider.generate(
        messages=messages,
        config=LLMConfig(max_new_tokens=bound_test_tokens, temperature=0.7),
    )
    bound_latency_ms = (time.perf_counter() - t0) * 1000.0
    print(f"Requested max_new_tokens: {bound_test_tokens}")
    print(f"Generated completion_tokens: {bounded_resp.usage.completion_tokens}")
    print(f"Bounding Latency: {bound_latency_ms:.2f} ms")
    assert bounded_resp.usage.completion_tokens <= bound_test_tokens, (
        f"Tokens generated {bounded_resp.usage.completion_tokens} exceeded ceiling {bound_test_tokens}"
    )

    results["token_bounding"] = {
        "requested_ceiling": bound_test_tokens,
        "generated_tokens": bounded_resp.usage.completion_tokens,
        "latency_ms": round(bound_latency_ms, 2),
        "bound_respected": bounded_resp.usage.completion_tokens <= bound_test_tokens,
    }

    # -------------------------------------------------------------
    # 7. Memory Tracking & Lifecycle Release Verification
    # -------------------------------------------------------------
    print("\n--- 5. Memory Tracking & Lifecycle Release Verification ---")
    peak_vram_mb = torch.cuda.max_memory_allocated() / (1024**2)
    print(f"Peak VRAM during Gate 3: {peak_vram_mb:.2f} MB")

    provider.release()
    released_vram_mb = torch.cuda.memory_allocated() / (1024**2)
    print(f"VRAM Allocated after release(): {released_vram_mb:.4f} MB")
    assert released_vram_mb < 1.0, f"Expected < 1.0 MB after release, got {released_vram_mb:.4f} MB"
    assert not provider.is_loaded, "Provider must report is_loaded=False after release."

    results["memory_telemetry"] = {
        "peak_vram_mb": round(peak_vram_mb, 2),
        "final_allocated_vram_mb": round(released_vram_mb, 4),
        "released_cleanly": bool(released_vram_mb < 1.0),
    }

    print("\n" + "=" * 60)
    print("ALL GATE 3 CHECKS PASSED SUCCESSFULLY!")
    print("=" * 60)

    # Save results to file
    out_dir = Path("data/evaluation")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "step3_generation_smoke_results.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results written to: {out_file}")

    return results


if __name__ == "__main__":
    run_gate3_tests()
