"""Data contracts and validation models for Stage 5.2 Evidence Grounding & Synthesis."""

import re
from enum import Enum
from typing import Any
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from academic_agent.generation.models import LLMResponse
from academic_agent.retrieval.models import Evidence


class EvidenceSufficiency(str, Enum):
    """Categorical classification of retrieved evidence adequacy for a query.

    Must be explicitly evaluated; defaults to UNKNOWN until an assessment component
    analyzes the query against the evidence context.
    """

    UNKNOWN = "unknown"
    SUFFICIENT = "sufficient"
    PARTIALLY_SUFFICIENT = "partial"
    INSUFFICIENT = "insufficient"


class OmissionReason(str, Enum):
    """Categorical reason why a candidate evidence item was omitted during context assembly."""

    DUPLICATE = "duplicate"
    OVERLAPPING = "overlapping"
    BUDGET_EXCEEDED = "budget_exceeded"
    INVALID_CANDIDATE = "invalid_candidate"


class OmittedCandidate(BaseModel):
    """Structured audit record of an omitted candidate during context assembly."""

    record_id: str = Field(description="Record ID of the candidate.")
    chunk_id: str = Field(description="Chunk ID or unit ID of the candidate.")
    source_file: str = Field(description="Source lecture PDF or document filename.")
    source_pages: list[int] = Field(default_factory=list, description="Contributing source pages.")
    reason: OmissionReason = Field(description="Categorical reason why candidate was omitted.")
    details: str = Field(default="", description="Detailed explanation or threshold metrics for omission.")
    rank: int = Field(ge=1, description="Original retrieval rank (1-based).")

    @property
    def omission_reason(self) -> OmissionReason:
        """Alias for reason conforming to audit interface."""
        return self.reason


class GroundedEvidence(BaseModel):
    """An individual retrieved evidence item with application-assigned ID and verified provenance."""

    evidence_id: str = Field(description="Strict synthesis identifier matching '^E[1-9]\\d*$' (e.g. 'E1', 'E2').")
    evidence: Evidence = Field(description="Underlying Phase 4 Evidence record with full provenance.")
    citation_label: str = Field(
        default="",
        description="Human-readable citation matching evidence provenance (e.g. '[Lecture 4, p. 18]').",
    )
    token_count: int = Field(
        default=0,
        ge=0,
        description="Token consumption of this passage in its formatted representation.",
    )
    is_truncated: bool = Field(
        default=False,
        description="True if this specific passage text was truncated to fit context budget.",
    )

    @field_validator("evidence_id")
    @classmethod
    def validate_id_format(cls, v: str) -> str:
        if not re.match(r"^E[1-9]\d*$", v):
            raise ValueError(f"evidence_id must match format 'E<int>' starting from 1 (e.g. 'E1', 'E2'), got '{v}'.")
        return v

    @model_validator(mode="after")
    def validate_or_derive_citation_label(self) -> "GroundedEvidence":
        expected_citation = self.evidence.format_citation()
        if not self.citation_label:
            self.citation_label = expected_citation
        elif self.citation_label != expected_citation:
            raise ValueError(
                f"citation_label '{self.citation_label}' does not match underlying evidence provenance '{expected_citation}'."
            )
        return self

    @property
    def tag(self) -> str:
        """Formatted citation tag expected from the LLM (e.g. '[E1]')."""
        return f"[{self.evidence_id}]"

    @property
    def source_file(self) -> str:
        return self.evidence.source_file

    @property
    def source_pages(self) -> list[int]:
        return self.evidence.source_pages

    @property
    def text(self) -> str:
        return self.evidence.text


class EvidenceContext(BaseModel):
    """Structured evidence package assembled for prompt injection within a token budget.

    'items' is the canonical source of truth. Lookup maps and omission metrics are dynamically derived.
    """

    items: list[GroundedEvidence] = Field(
        default_factory=list,
        description="Canonical ordered grounded evidence items.",
    )
    total_tokens: int = Field(
        default=0,
        ge=0,
        description="Total tokens consumed by formatted evidence block.",
    )
    budget_allocated: int = Field(
        default=0,
        ge=0,
        description="Maximum token budget allocated for evidence.",
    )
    omitted_candidates: list[OmittedCandidate] = Field(
        default_factory=list,
        description="Audit record of candidate items omitted during context assembly.",
    )
    is_exact_token_count: bool = Field(
        default=True,
        description="False if heuristic fallback was used for token counting.",
    )

    @model_validator(mode="before")
    @classmethod
    def _handle_legacy_omissions(cls, data: Any) -> Any:
        if isinstance(data, dict):
            # If omitted_candidates is not provided but legacy fields are:
            if "omitted_candidates" not in data:
                count = data.pop("dropped_candidates_count", 0)
                has_om = data.pop("has_omissions", False)
                if count > 0 or has_om:
                    synth_count = max(count, 1 if has_om else 0)
                    data["omitted_candidates"] = [
                        OmittedCandidate(
                            record_id=f"legacy_dropped_{i+1}",
                            chunk_id=f"legacy_{i+1}",
                            source_file="unknown",
                            source_pages=[],
                            reason=OmissionReason.BUDGET_EXCEEDED,
                            details="Legacy dropped candidate placeholder",
                            rank=i + 1,
                        )
                        for i in range(synth_count)
                    ]
            else:
                data.pop("dropped_candidates_count", None)
                data.pop("has_omissions", None)
        return data

    @model_validator(mode="after")
    def validate_unique_ids(self) -> "EvidenceContext":
        seen_ids = set()
        for item in self.items:
            if item.evidence_id in seen_ids:
                raise ValueError(f"Duplicate evidence_id '{item.evidence_id}' found in EvidenceContext items.")
            seen_ids.add(item.evidence_id)
        return self

    @property
    def count(self) -> int:
        """Count of canonical evidence items assembled in this context."""
        return len(self.items)

    @property
    def evidence_ids(self) -> list[str]:
        """Ordered list of valid evidence IDs present in this context."""
        return [item.evidence_id for item in self.items]

    @property
    def has_omissions(self) -> bool:
        """True if any candidate was omitted due to budget constraints or deduplication."""
        return len(self.omitted_candidates) > 0

    @property
    def dropped_candidates_count(self) -> int:
        """Count of candidate items omitted during context assembly."""
        return len(self.omitted_candidates)

    @property
    def evidence_lookup(self) -> dict[str, GroundedEvidence]:
        """Dynamic lookup mapping 'E1' -> GroundedEvidence derived strictly from canonical items."""
        return {item.evidence_id: item for item in self.items}

    def get(self, evidence_id: str) -> GroundedEvidence | None:
        """Fetch item by normalized ID (accepts 'E1' or '[E1]')."""
        clean_id = evidence_id.strip("[]").strip()
        return self.evidence_lookup.get(clean_id)


class CitationOccurrence(BaseModel):
    """A citation marker extracted from generated text with exact character offsets.

    Distinguishes the outer citation group span from the individual evidence reference token.
    """

    model_config = ConfigDict(frozen=True)

    evidence_id: str = Field(description="Parsed identifier without brackets (e.g. 'E1').")
    raw_citation: str = Field(description="Raw citation substring (e.g. '[E1]' or '[E1, E2]').")
    start_char: int = Field(ge=0, description="Start character offset in generated text.")
    end_char: int = Field(ge=0, description="End character offset in generated text.")
    is_valid: bool = Field(default=False, description="True if evidence_id exists in the context.")
    resolution_label: str = Field(
        default="",
        description="Resolved human-readable citation label when valid (e.g. '[Lecture 4, p. 18]').",
    )
    is_grouped: bool = Field(
        default=False,
        description="True if this occurrence represents a reference within a multi-ID group (e.g. '[E1, E2]').",
    )
    group_span: tuple[int, int] | None = Field(
        default=None,
        description="The enclosing (start, end) span of the entire bracketed citation group.",
    )
    sub_span: tuple[int, int] | None = Field(
        default=None,
        description="Exact (start, end) span of this individual evidence_id token within answer text.",
    )

    @model_validator(mode="after")
    def validate_offsets(self) -> "CitationOccurrence":
        if self.end_char <= self.start_char:
            raise ValueError(
                f"end_char ({self.end_char}) must be strictly greater than start_char ({self.start_char})."
            )
        return self


class PolicyOutcome(str, Enum):
    """Categorical classification of generated answer compliance with academic answering policy."""

    ANSWERED = "answered"
    COMPLIANT_ABSTENTION = "compliant_abstention"
    PARTIAL_ANSWER = "partial_answer"
    CONTRADICTORY_REFUSAL = "contradictory_refusal"
    UNSUPPORTED_ANSWER = "unsupported_answer"
    INDETERMINATE = "indeterminate"


class PolicyValidationResult(BaseModel):
    """Evaluation result assessing answer compliance against academic answering policy."""

    model_config = ConfigDict(frozen=True)

    outcome: PolicyOutcome = Field(description="Categorical policy outcome.")
    refusal_detected: bool = Field(default=False, description="Whether refusal language was found.")
    refusal_phrases: list[str] = Field(default_factory=list, description="Extracted refusal substrings.")
    has_substantive_continuation: bool = Field(
        default=False,
        description="Whether substantive claims followed a refusal statement.",
    )
    has_gap_statement: bool = Field(
        default=False,
        description="Whether unsupported gaps were explicitly identified in a partial answer.",
    )
    is_compliant: bool = Field(
        description="True if answer complies with academic answering guidelines (ANSWERED, COMPLIANT_ABSTENTION, PARTIAL_ANSWER).",
    )
    explanation: str = Field(default="", description="Detailed rationale for the policy classification.")
    warnings: list[str] = Field(default_factory=list, description="Policy anomalies or warnings.")


class ValidationDiagnostics(BaseModel):
    """Detailed audit metrics and anomalies detected during answer validation."""

    model_config = ConfigDict(frozen=True)

    citations: list[CitationOccurrence] = Field(
        default_factory=list,
        description="All citation occurrences with exact character spans.",
    )
    valid_cited_ids: list[str] = Field(
        default_factory=list,
        description="Unique cited IDs verified to exist in context.",
    )
    invalid_cited_ids: list[str] = Field(
        default_factory=list,
        description="Fabricated IDs cited by the LLM (e.g. [E99]).",
    )
    uncited_evidence_ids: list[str] = Field(
        default_factory=list,
        description="Context evidence IDs never cited in answer.",
    )
    malformed_citations: list[str] = Field(
        default_factory=list,
        description="Malformed or ambiguous citation strings.",
    )
    refusal_detected: bool = Field(
        default=False,
        description="Whether LLM emitted an explicit refusal statement.",
    )
    evidence_sufficiency: EvidenceSufficiency = Field(
        default=EvidenceSufficiency.UNKNOWN,
        description="Assessed evidence sufficiency for this query/answer pair.",
    )
    warnings: list[str] = Field(
        default_factory=list,
        description="Non-fatal anomalies or warnings.",
    )

    @property
    def cited_ids(self) -> list[str]:
        """All unique cited IDs preserving first-occurrence order."""
        seen = set()
        out = []
        for c in self.citations:
            if c.evidence_id not in seen:
                seen.add(c.evidence_id)
                out.append(c.evidence_id)
        return out


class AcademicAnswer(BaseModel):
    """Final grounded academic answer with verified citations, provenance, and diagnostics."""

    model_config = ConfigDict(frozen=True)

    answer_text: str = Field(description="Generated answer with original compact citation tags (e.g. [E1]).")
    resolved_answer_text: str = Field(
        description="Answer with IDs replaced by human-readable citations (e.g. [Lecture 4, p. 18])."
    )
    query: str = Field(description="Original user query.")
    evidence_context: EvidenceContext = Field(description="Evidence package provided to the LLM.")
    validation: ValidationDiagnostics = Field(description="Citation integrity and grounding diagnostics.")
    evidence_sufficiency: EvidenceSufficiency = Field(
        default=EvidenceSufficiency.UNKNOWN,
        description="Assessed sufficiency classification.",
    )
    policy_result: PolicyValidationResult | None = Field(
        default=None,
        description="Assessed answering policy compliance result.",
    )
    raw_response: LLMResponse = Field(description="Underlying raw response from the LLM provider.")
    model_name: str = Field(description="Model identifier that produced this completion.")
    latency_ms: float = Field(ge=0.0, description="Total end-to-end synthesis latency in milliseconds.")

    @property
    def policy_outcome(self) -> PolicyOutcome | None:
        """Categorical policy outcome, if policy validation was executed."""
        return self.policy_result.outcome if self.policy_result else None

    @property
    def is_policy_compliant(self) -> bool:
        """True if policy validation passed and answer complies with academic answering policy."""
        return self.policy_result.is_compliant if self.policy_result else False

    @property
    def has_valid_citation_integrity(self) -> bool:
        """True if answer contains valid citations and ZERO invalid/fabricated citation IDs.

        Note: Verifies citation integrity only; does not independently prove factual claim support.
        """
        return len(self.validation.valid_cited_ids) > 0 and len(self.validation.invalid_cited_ids) == 0

    @property
    def refusal_detected(self) -> bool:
        """True if the LLM output contained an explicit refusal/abstention phrase."""
        if self.policy_result is not None:
            return self.policy_result.refusal_detected
        return self.validation.refusal_detected

    @property
    def is_abstention_required(self) -> bool:
        """True if the system's sufficiency assessment determined evidence is insufficient."""
        return self.evidence_sufficiency == EvidenceSufficiency.INSUFFICIENT

    @property
    def is_compliant_abstention(self) -> bool:
        """Preliminary syntactic check: abstention was required AND refusal phrase was detected.

        Note: A preliminary indicator only; an answer-policy validator will confirm that no
        substantive unsupported claims followed the refusal.
        """
        return self.is_abstention_required and self.refusal_detected
