"""Protocols defining contracts for Stage 5.2 Evidence Grounding and Synthesis."""

from typing import Protocol, runtime_checkable

from academic_agent.generation.models import ChatMessage, LLMConfig
from academic_agent.generation.synthesis.models import (
    AcademicAnswer,
    EvidenceContext,
    EvidenceSufficiency,
    ValidationDiagnostics,
)
from academic_agent.retrieval.models import RetrievalResponse


@runtime_checkable
class EvidenceFormatterProtocol(Protocol):
    """Protocol for safely formatting untrusted evidence text with XML escaping."""

    def format_evidence_block(self, context: EvidenceContext) -> str:
        """Format evidence into safely escaped XML blocks.

        Args:
            context: EvidenceContext containing canonical grounded evidence items.

        Returns:
            XML string representing escaped evidence package.
        """
        ...


@runtime_checkable
class EvidenceContextBuilderProtocol(Protocol):
    """Protocol for deduplicating, ordering, and assembling token-budgeted evidence packages."""

    def build_context(
        self,
        retrieval_response: RetrievalResponse,
        allocated_evidence_tokens: int,
    ) -> EvidenceContext:
        """Assemble a budgeted evidence context from retrieval results.

        Args:
            retrieval_response: Results surfaced by Phase 4 Retrieval Engine.
            allocated_evidence_tokens: Exact token ceiling allocated to evidence.

        Returns:
            EvidenceContext respecting the budget allocation.
        """
        ...


@runtime_checkable
class GroundingPromptBuilderProtocol(Protocol):
    """Protocol for constructing instruction-grounded chat dialogue messages."""

    def build_messages(
        self,
        query: str,
        context: EvidenceContext,
    ) -> list[ChatMessage]:
        """Construct grounded chat messages separating instructions from untrusted evidence.

        Args:
            query: The academic question asked by the user.
            context: Budgeted evidence context to inject into prompt.

        Returns:
            List of ChatMessage objects formatted for LLM consumption.
        """
        ...


@runtime_checkable
class CitationValidatorProtocol(Protocol):
    """Protocol for parsing, validating integrity, and resolving evidence citations."""

    def validate(
        self,
        answer_text: str,
        context: EvidenceContext,
    ) -> ValidationDiagnostics:
        """Extract citations, check context presence, and record span offsets.

        Args:
            answer_text: Generated model answer with compact IDs (e.g. '[E1]').
            context: EvidenceContext provided during generation.

        Returns:
            ValidationDiagnostics containing occurrence spans and integrity flags.
        """
        ...

    def resolve_citations(
        self,
        answer_text: str,
        context: EvidenceContext,
    ) -> str:
        """Replace valid IDs like '[E1]' with human-readable provenance (e.g. '[Lecture 4, p. 18]').

        Unrecognized or invalid IDs (e.g. '[E99]') must remain unreplaced and flagged.

        Args:
            answer_text: Text containing citation tags.
            context: EvidenceContext containing valid mapping.

        Returns:
            Text with recognized tags safely resolved.
        """
        ...


@runtime_checkable
class EvidenceSufficiencyAssessorProtocol(Protocol):
    """Protocol for evaluating whether retrieved evidence adequately supports the query."""

    def assess(
        self,
        query: str,
        context: EvidenceContext,
    ) -> EvidenceSufficiency:
        """Evaluate evidence coverage against user question.

        Args:
            query: User's academic question.
            context: Assembled evidence context.

        Returns:
            EvidenceSufficiency classification (SUFFICIENT, PARTIALLY_SUFFICIENT, INSUFFICIENT).
        """
        ...


@runtime_checkable
class AcademicSynthesizerProtocol(Protocol):
    """Top-level orchestrator coordinating budgeting, prompt generation, LLM inference, and validation."""

    def synthesize(
        self,
        query: str,
        retrieval_response: RetrievalResponse,
        config: LLMConfig | None = None,
    ) -> AcademicAnswer:
        """Execute end-to-end grounded synthesis.

        Args:
            query: Academic question.
            retrieval_response: Phase 4 retrieval candidates.
            config: Optional request-level generation configuration.

        Returns:
            AcademicAnswer with verified citations, provenance, and sufficiency diagnostics.
        """
        ...
