"""Stage 5.2 Evidence Grounding and Synthesis subsystem."""

from academic_agent.generation.synthesis.models import (
    AcademicAnswer,
    CitationOccurrence,
    EvidenceContext,
    EvidenceSufficiency,
    GroundedEvidence,
    OmissionReason,
    OmittedCandidate,
    ValidationDiagnostics,
)
from academic_agent.generation.synthesis.protocols import (
    AcademicSynthesizerProtocol,
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

__all__ = [
    "EvidenceSufficiency",
    "OmissionReason",
    "OmittedCandidate",
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
    "EvidenceContextBuilder",
    "GroundingPromptBuilder",
    "DEFAULT_SYSTEM_PROMPT",
]
