"""Stage 5.2 Evidence Grounding and Synthesis subsystem."""

from academic_agent.generation.synthesis.models import (
    EvidenceSufficiency,
    GroundedEvidence,
    EvidenceContext,
    CitationOccurrence,
    ValidationDiagnostics,
    AcademicAnswer,
)
from academic_agent.generation.synthesis.protocols import (
    EvidenceFormatterProtocol,
    EvidenceContextBuilderProtocol,
    GroundingPromptBuilderProtocol,
    CitationValidatorProtocol,
    EvidenceSufficiencyAssessorProtocol,
    AcademicSynthesizerProtocol,
)
from academic_agent.generation.synthesis.context.evidence_formatter import EvidenceFormatter

__all__ = [
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
