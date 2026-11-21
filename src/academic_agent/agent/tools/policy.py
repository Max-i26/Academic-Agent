"""Deterministic clarification and refusal policy helpers for bounded agent autonomy."""

from enum import Enum
import re
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class QueryAssessmentCategory(str, Enum):
    """Categorization of initial student queries prior to retrieval."""

    VALID_ACADEMIC = "VALID_ACADEMIC"
    OUT_OF_DOMAIN = "OUT_OF_DOMAIN"
    UNDERSPECIFIED = "UNDERSPECIFIED"
    MALFORMED = "MALFORMED"


class EvidenceAssessmentCategory(str, Enum):
    """Assessment of retrieved evidence candidates before invoking answer synthesis."""

    SUFFICIENT = "SUFFICIENT"
    EMPTY = "EMPTY"
    LOW_CONFIDENCE = "LOW_CONFIDENCE"


class QueryAssessment(BaseModel):
    """Diagnostic assessment of student query feasibility."""

    category: QueryAssessmentCategory
    is_actionable: bool
    rationale: str
    suggested_action: str


class EvidenceAssessment(BaseModel):
    """Diagnostic assessment of evidence candidate adequacy."""

    category: EvidenceAssessmentCategory
    can_synthesize: bool
    rationale: str
    suggested_action: str


class ClarificationPolicy:
    """Deterministic rules governing query clarification, out-of-domain refusals, and abstention."""

    OUT_OF_DOMAIN_PATTERNS = [
        re.compile(r"\b(recipe|bake|cook|ingredients)\b", re.IGNORECASE),
        re.compile(r"\b(weather forecast|rain today|temperature outside)\b", re.IGNORECASE),
        re.compile(r"\b(tell me a joke|write a poem|sing a song)\b", re.IGNORECASE),
        re.compile(r"\b(horoscope|astrology|zodiac)\b", re.IGNORECASE),
        re.compile(r"\b(who won the (game|match|super bowl|world cup))\b", re.IGNORECASE),
    ]

    UNDERSPECIFIED_PATTERNS = [
        re.compile(r"^(what|why|how|explain|define|tell me)\??$", re.IGNORECASE),
        re.compile(r"^(what is it|explain it|how does it work)\??$", re.IGNORECASE),
    ]

    @classmethod
    def assess_query(cls, query: str) -> QueryAssessment:
        """Evaluate student query feasibility and detect out-of-domain or underspecified inputs."""
        cleaned = query.strip() if query else ""

        if not cleaned or len(cleaned) < 3:
            return QueryAssessment(
                category=QueryAssessmentCategory.MALFORMED,
                is_actionable=False,
                rationale="Query is empty or too short to contain academic intent.",
                suggested_action="refusal",
            )

        for pat in cls.UNDERSPECIFIED_PATTERNS:
            if pat.match(cleaned):
                return QueryAssessment(
                    category=QueryAssessmentCategory.UNDERSPECIFIED,
                    is_actionable=False,
                    rationale="Query is ambiguous or lacks specific academic concepts or course topics.",
                    suggested_action="clarification_request",
                )

        for pat in cls.OUT_OF_DOMAIN_PATTERNS:
            if pat.search(cleaned):
                return QueryAssessment(
                    category=QueryAssessmentCategory.OUT_OF_DOMAIN,
                    is_actionable=False,
                    rationale="Query falls outside academic course domains (e.g. personal, chit-chat, or lifestyle).",
                    suggested_action="refusal",
                )

        return QueryAssessment(
            category=QueryAssessmentCategory.VALID_ACADEMIC,
            is_actionable=True,
            rationale="Query contains actionable academic terms suitable for lecture retrieval.",
            suggested_action="academic_search",
        )

    @classmethod
    def assess_evidence(
        cls,
        query: str,
        candidates: List[Dict[str, Any]],
        min_confidence_score: float = 0.25,
    ) -> EvidenceAssessment:
        """Evaluate whether retrieved evidence is adequate for grounded academic synthesis."""
        if not candidates:
            return EvidenceAssessment(
                category=EvidenceAssessmentCategory.EMPTY,
                can_synthesize=False,
                rationale="No relevant lecture evidence candidates were surfaced by retrieval.",
                suggested_action="abstention",
            )

        top_score = max(float(c.get("score", 0.0)) for c in candidates)
        if top_score < min_confidence_score:
            return EvidenceAssessment(
                category=EvidenceAssessmentCategory.LOW_CONFIDENCE,
                can_synthesize=False,
                rationale=f"Highest evidence score ({top_score:.3f}) falls below confidence threshold ({min_confidence_score:.3f}).",
                suggested_action="abstention",
            )

        return EvidenceAssessment(
            category=EvidenceAssessmentCategory.SUFFICIENT,
            can_synthesize=True,
            rationale=f"Surfaced {len(candidates)} candidates with top confidence score {top_score:.3f}.",
            suggested_action="academic_synthesizer",
        )
