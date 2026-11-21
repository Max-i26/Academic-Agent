"""Academic-Agent Generation Subsystem."""

from academic_agent.generation.models import (
    MessageRole,
    ChatMessage,
    LLMConfig,
    ModelLoadingConfig,
    LLMUsage,
    LLMResponse,
    GenerationError,
    ContextLengthExceededError,
    ModelNotFoundError,
    CudaOutOfMemoryError,
    ProviderUnavailableError,
)
from academic_agent.generation.protocols import (
    LLMProviderProtocol,
)
from academic_agent.generation.providers.base import BaseLLMProvider
from academic_agent.generation.providers.mock import MockLLMProvider
from academic_agent.generation.providers.hf_local import HuggingFaceTransformersProvider
from academic_agent.generation.providers.ollama import OllamaProvider
from academic_agent.generation.synthesis import (
    EvidenceSufficiency,
    GroundedEvidence,
    EvidenceContext,
    CitationOccurrence,
    ValidationDiagnostics,
    AcademicAnswer,
    EvidenceFormatterProtocol,
    EvidenceContextBuilderProtocol,
    GroundingPromptBuilderProtocol,
    CitationValidatorProtocol,
    EvidenceSufficiencyAssessorProtocol,
    AcademicSynthesizerProtocol,
    EvidenceFormatter,
)

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
    "EvidenceSufficiencyAssessorProtocol",
    "AcademicSynthesizerProtocol",
    "EvidenceFormatter",
]
