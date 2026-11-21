"""Stage 5.2 Evidence Grounding and Synthesis subsystem."""

from academic_agent.generation.synthesis.models import (
    AcademicAnswer,
    CitationOccurrence,
    EvidenceContext,
    EvidenceSufficiency,
    GroundedEvidence,
    OmissionReason,
    OmittedCandidate,
    PolicyOutcome,
    PolicyValidationResult,
    ValidationDiagnostics,
)
from academic_agent.generation.synthesis.protocols import (
    AcademicSynthesizerProtocol,
    AnswerPolicyValidatorProtocol,
    CitationValidatorProtocol,
    EvidenceContextBuilderProtocol,
    EvidenceFormatterProtocol,
    EvidenceSufficiencyAssessorProtocol,
    GroundingPromptBuilderProtocol,
)
from academic_agent.generation.synthesis.context.evidence_context_builder import EvidenceContextBuilder
from academic_agent.generation.synthesis.context.evidence_formatter import EvidenceFormatter
from academic_agent.generation.synthesis.prompt.grounding_prompt_builder import (
    DEFAULT_SYSTEM_PROMPT,
    GroundingPromptBuilder,
)
from academic_agent.generation.synthesis.validation.answer_policy_validator import AnswerPolicyValidator
from academic_agent.generation.synthesis.validation.citation_validator import CitationValidator
from academic_agent.generation.synthesis.synthesizer import AcademicSynthesizer

__all__ = [
    "EvidenceSufficiency",
    "OmissionReason",
    "OmittedCandidate",
    "PolicyOutcome",
    "PolicyValidationResult",
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
]
