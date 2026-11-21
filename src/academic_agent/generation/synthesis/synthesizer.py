"""Academic answer synthesizer coordinating evidence budgeting, prompt generation, LLM generation, and validation."""

import time
from typing import Any

from academic_agent.generation.models import (
    ChatMessage,
    LLMConfig,
    LLMResponse,
    ProviderUnavailableError,
)
from academic_agent.generation.protocols import LLMProviderProtocol
from academic_agent.generation.synthesis.context.evidence_context_builder import (
    EvidenceContextBuilder,
)
from academic_agent.generation.synthesis.models import (
    AcademicAnswer,
    EvidenceContext,
    EvidenceSufficiency,
    PolicyOutcome,
    PolicyValidationResult,
    ValidationDiagnostics,
)
from academic_agent.generation.synthesis.prompt.grounding_prompt_builder import (
    GroundingPromptBuilder,
)
from academic_agent.generation.synthesis.protocols import (
    AcademicSynthesizerProtocol,
    AnswerPolicyValidatorProtocol,
    CitationValidatorProtocol,
    EvidenceContextBuilderProtocol,
    EvidenceSufficiencyAssessorProtocol,
    GroundingPromptBuilderProtocol,
)
from academic_agent.generation.synthesis.validation.answer_policy_validator import (
    AnswerPolicyValidator,
)
from academic_agent.generation.synthesis.validation.citation_validator import (
    CitationValidator,
)
from academic_agent.generation.synthesis.sufficiency.assessor import (
    DeterministicSufficiencyAssessor,
)
from academic_agent.retrieval.models import RetrievalResponse


class AcademicSynthesizer:
    """Production implementation of AcademicSynthesizerProtocol.

    Coordinates the complete Stage 5.2 generation and verification pipeline:
    1. RetrievalResponse -> EvidenceContextBuilder -> EvidenceContext (budgeted & deduplicated)
    2. EvidenceContext + Query -> GroundingPromptBuilder -> ChatMessages (role-separated, untrusted XML)
    3. ChatMessages -> LLMProvider.generate -> LLMResponse
    4. LLMResponse.content -> CitationValidator.validate & resolve_citations
    5. LLMResponse.content + Context + Diagnostics -> AnswerPolicyValidator.validate_policy
    6. Assembly into verified AcademicAnswer with provenance, diagnostics, and telemetry.
    """

    def __init__(
        self,
        llm_provider: LLMProviderProtocol,
        context_builder: EvidenceContextBuilderProtocol | None = None,
        prompt_builder: GroundingPromptBuilderProtocol | None = None,
        citation_validator: CitationValidatorProtocol | None = None,
        policy_validator: AnswerPolicyValidatorProtocol | None = None,
        sufficiency_assessor: EvidenceSufficiencyAssessorProtocol | None = None,
        default_evidence_budget: int = 1500,
    ) -> None:
        """Initialize the AcademicSynthesizer orchestrator with pipeline components.

        Args:
            llm_provider: Language model provider implementing LLMProviderProtocol.
            context_builder: Evidence context builder (defaults to EvidenceContextBuilder).
            prompt_builder: Grounding prompt builder (defaults to GroundingPromptBuilder).
            citation_validator: Citation validator (defaults to CitationValidator).
            policy_validator: Answer policy validator (defaults to AnswerPolicyValidator).
            sufficiency_assessor: Optional assessor for automated sufficiency classification.
            default_evidence_budget: Default token allocation for evidence passages.
        """
        self.llm_provider = llm_provider
        self.context_builder = context_builder or EvidenceContextBuilder()
        self.prompt_builder = prompt_builder or GroundingPromptBuilder()
        self.citation_validator = citation_validator or CitationValidator()
        self.policy_validator = policy_validator or AnswerPolicyValidator()
        self.sufficiency_assessor = (
            sufficiency_assessor
            if sufficiency_assessor is not None
            else DeterministicSufficiencyAssessor()
        )
        self.default_evidence_budget = default_evidence_budget

    def synthesize(
        self,
        query: str,
        retrieval_response: RetrievalResponse,
        config: LLMConfig | None = None,
        allocated_evidence_tokens: int | None = None,
        sufficiency: EvidenceSufficiency | None = None,
    ) -> AcademicAnswer:
        """Execute end-to-end grounded synthesis.

        Args:
            query: Academic question.
            retrieval_response: Phase 4 retrieval candidates and evidence.
            config: Optional per-request LLM generation parameters.
            allocated_evidence_tokens: Optional override for evidence token budget.
            sufficiency: Optional explicit sufficiency override. If None and sufficiency_assessor
                is provided, assessor is called. If empty evidence, defaults to INSUFFICIENT.
                Otherwise defaults to UNKNOWN.

        Returns:
            AcademicAnswer containing verified citations, resolved text, policy outcome, and telemetry.
        """
        start_time = time.perf_counter()

        # 1. Validate query input
        if not isinstance(query, str) or not query.strip():
            raise ValueError("Query text cannot be empty or whitespace only.")

        # 2. Check provider availability early
        if not self.llm_provider.is_available:
            raise ProviderUnavailableError(
                f"LLM provider '{self.llm_provider.model_name}' is currently unavailable."
            )

        # 3. Assemble budgeted evidence context
        provider_ctx = self.llm_provider.context_window
        max_gen_tokens = config.max_new_tokens if (config and config.max_new_tokens is not None) else 512
        safety_margin = 64

        if allocated_evidence_tokens is not None:
            budget = allocated_evidence_tokens
        else:
            sys_prompt_text = getattr(self.prompt_builder, "system_prompt", "")
            sys_tokens, _ = self.prompt_builder.estimate_tokens(sys_prompt_text)
            q_tokens, _ = self.prompt_builder.estimate_tokens(query)
            # Dynamic available evidence budget:
            # B_e = max(0, min(B_max, C - T_s - T_q - T_g - M))
            computed_budget = max(
                0,
                min(
                    self.default_evidence_budget,
                    provider_ctx - sys_tokens - q_tokens - max_gen_tokens - safety_margin,
                ),
            )
            budget = computed_budget

        evidence_context = self.context_builder.build_context(
            retrieval_response=retrieval_response,
            allocated_evidence_tokens=budget,
        )

        # 4. Determine evidence sufficiency
        if sufficiency is not None:
            assessed_sufficiency = sufficiency
        elif evidence_context.count == 0:
            assessed_sufficiency = EvidenceSufficiency.INSUFFICIENT
        elif self.sufficiency_assessor is not None:
            assessed_sufficiency = self.sufficiency_assessor.assess(query, evidence_context)
        else:
            assessed_sufficiency = EvidenceSufficiency.UNKNOWN

        # 5. Build chat messages passing per-call ceiling without mutating shared builder state
        max_prompt_budget = max(0, provider_ctx - max_gen_tokens)
        try:
            messages = self.prompt_builder.build_messages(
                query=query,
                context=evidence_context,
                max_prompt_tokens=max_prompt_budget,
            )
        except TypeError:
            # Fallback for custom prompt_builder protocols lacking max_prompt_tokens argument
            messages = self.prompt_builder.build_messages(
                query=query,
                context=evidence_context,
            )

        # 6. Generate LLM completion (propagates provider exceptions directly)
        raw_response = self.llm_provider.generate(
            messages=messages,
            config=config,
        )
        raw_text = raw_response.content

        # 7. Validate citations & resolve human-readable labels
        diagnostics = self.citation_validator.validate(raw_text, evidence_context)
        diagnostics = diagnostics.model_copy(update={"evidence_sufficiency": assessed_sufficiency})

        resolved_answer_text = self.citation_validator.resolve_citations(
            raw_text,
            evidence_context,
        )

        # 8. Evaluate answer policy
        policy_result = self.policy_validator.validate_policy(
            answer_text=raw_text,
            context=evidence_context,
            diagnostics=diagnostics,
            sufficiency=assessed_sufficiency,
        )
        diagnostics = diagnostics.model_copy(update={"refusal_detected": policy_result.refusal_detected})

        total_latency_ms = (time.perf_counter() - start_time) * 1000.0

        # 9. Return structured AcademicAnswer preserving raw_response intact
        return AcademicAnswer(
            answer_text=raw_text,
            resolved_answer_text=resolved_answer_text,
            query=query,
            evidence_context=evidence_context,
            validation=diagnostics,
            evidence_sufficiency=assessed_sufficiency,
            policy_result=policy_result,
            raw_response=raw_response,
            model_name=self.llm_provider.model_name,
            latency_ms=total_latency_ms,
        )
