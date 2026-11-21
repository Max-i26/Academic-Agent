"""Academic-Agent Generation Subsystem."""

from typing import Any

from academic_agent.generation.models import (
    ChatMessage,
    ContextLengthExceededError,
    CudaOutOfMemoryError,
    GenerationError,
    LLMConfig,
    LLMResponse,
    LLMUsage,
    MessageRole,
    ModelLoadingConfig,
    ModelNotFoundError,
    ProviderUnavailableError,
)
from academic_agent.generation.protocols import (
    LLMProviderProtocol,
)
from academic_agent.generation.providers.base import BaseLLMProvider
from academic_agent.generation.providers.mock import MockLLMProvider
from academic_agent.generation.synthesis import (
    AcademicAnswer,
    AcademicSynthesizer,
    AcademicSynthesizerProtocol,
    AnswerPolicyValidator,
    AnswerPolicyValidatorProtocol,
    CitationOccurrence,
    CitationValidator,
    CitationValidatorProtocol,
    DEFAULT_SYSTEM_PROMPT,
    DeterministicSufficiencyAssessor,
    EvidenceContext,
    EvidenceContextBuilder,
    EvidenceContextBuilderProtocol,
    EvidenceFormatter,
    EvidenceFormatterProtocol,
    EvidenceSufficiency,
    EvidenceSufficiencyAssessorProtocol,
    GroundedEvidence,
    GroundingPromptBuilder,
    GroundingPromptBuilderProtocol,
    OmissionReason,
    OmittedCandidate,
    PolicyOutcome,
    PolicyValidationResult,
    ValidationDiagnostics,
)


def __getattr__(name: str) -> Any:
    """PEP 562 lazy import of heavyweight or hardware-dependent LLM providers."""
    if name == "HuggingFaceTransformersProvider":
        from academic_agent.generation.providers.hf_local import HuggingFaceTransformersProvider
        return HuggingFaceTransformersProvider
    if name == "OllamaProvider":
        from academic_agent.generation.providers.ollama import OllamaProvider
        return OllamaProvider
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


__all__ = [
    "MessageRole",
    "ChatMessage",
    "LLMConfig",
    "ModelLoadingConfig",
    "LLMUsage",
    "LLMResponse",
    "GenerationError",
    "ContextLengthExceededError",
    "ModelNotFoundError",
    "CudaOutOfMemoryError",
    "ProviderUnavailableError",
    "LLMProviderProtocol",
    "BaseLLMProvider",
    "MockLLMProvider",
    "HuggingFaceTransformersProvider",
    "OllamaProvider",
    "EvidenceSufficiency",
    "GroundedEvidence",
    "EvidenceContext",
    "CitationOccurrence",
    "ValidationDiagnostics",
    "AcademicAnswer",
    "EvidenceFormatterProtocol",
    "EvidenceContextBuilderProtocol",
    "GroundingPromptBuilderProtocol",
    "CitationValidatorProtocol",
    "AnswerPolicyValidatorProtocol",
    "EvidenceSufficiencyAssessorProtocol",
    "AcademicSynthesizerProtocol",
    "EvidenceFormatter",
    "EvidenceContextBuilder",
    "GroundingPromptBuilder",
    "DEFAULT_SYSTEM_PROMPT",
    "CitationValidator",
    "AnswerPolicyValidator",
    "AcademicSynthesizer",
    "DeterministicSufficiencyAssessor",
    "OmissionReason",
    "OmittedCandidate",
    "PolicyOutcome",
    "PolicyValidationResult",
]
