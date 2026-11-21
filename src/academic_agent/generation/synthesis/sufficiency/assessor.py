"""Deterministic baseline evidence sufficiency assessor evaluating query coverage across retrieved evidence."""

import re
from typing import Set

from academic_agent.generation.synthesis.models import EvidenceContext, EvidenceSufficiency
from academic_agent.generation.synthesis.protocols import EvidenceSufficiencyAssessorProtocol

STOPWORDS: Set[str] = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are",
    "aren't", "as", "at", "be", "because", "been", "before", "being", "below", "between", "both",
    "but", "by", "can", "can't", "cannot", "could", "couldn't", "did", "didn't", "do", "does",
    "doesn't", "doing", "don't", "down", "during", "each", "explain", "describe", "define", "discuss",
    "few", "for", "from", "further", "had", "hadn't", "has", "hasn't", "have", "haven't", "having",
    "he", "her", "here", "hers", "herself", "him", "himself", "his", "how", "i", "if", "in", "into",
    "is", "isn't", "it", "it's", "its", "itself", "let's", "me", "more", "most", "mustn't", "my",
    "myself", "no", "nor", "not", "of", "off", "on", "once", "only", "or", "other", "ought", "our",
    "ours", "ourselves", "out", "over", "own", "same", "shan't", "she", "should", "shouldn't", "so",
    "some", "such", "than", "that", "the", "their", "theirs", "them", "themselves", "then", "there",
    "these", "they", "this", "those", "through", "to", "too", "under", "until", "up", "very", "was",
    "wasn't", "we", "were", "weren't", "what", "when", "where", "which", "while", "who", "whom",
    "why", "with", "won't", "would", "wouldn't", "you", "your", "yours", "yourself", "yourselves",
    "please", "tell", "give", "show", "what's", "whats",
}


class DeterministicSufficiencyAssessor(EvidenceSufficiencyAssessorProtocol):
    """Deterministic rule-based assessor for evidence sufficiency.

    Evaluates whether the retrieved evidence contains sufficient terminology, concepts,
    and sub-clause coverage to answer the user query.
    """

    def __init__(
        self,
        min_coverage_ratio: float = 0.55,
        partial_coverage_ratio: float = 0.25,
    ) -> None:
        """Initialize assessor with configurable coverage thresholds.

        Args:
            min_coverage_ratio: Fraction of query concept terms required in evidence for SUFFICIENT.
            partial_coverage_ratio: Fraction required for PARTIALLY_SUFFICIENT. Below this is INSUFFICIENT.
        """
        if not (0.0 < partial_coverage_ratio <= min_coverage_ratio <= 1.0):
            raise ValueError("Thresholds must satisfy 0 < partial_coverage_ratio <= min_coverage_ratio <= 1.0")
        self.min_coverage_ratio = min_coverage_ratio
        self.partial_coverage_ratio = partial_coverage_ratio

    @staticmethod
    def _extract_concepts(text: str) -> list[str]:
        """Extract alphanumeric concepts, formulas, and domain terms from text."""
        raw_tokens = re.findall(r"[a-zA-Z0-9_()$\\]+", text.lower())
        concepts = []
        strip_chars = "()$\\"
        for t in raw_tokens:
            cleaned = t.strip(strip_chars)
            if len(cleaned) >= 2 and cleaned not in STOPWORDS:
                concepts.append(cleaned)
        return concepts

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
        if not context or context.count == 0 or not context.items:
            return EvidenceSufficiency.INSUFFICIENT

        # Aggregate evidence text
        evidence_text = " ".join(item.text.lower() for item in context.items)
        if not evidence_text.strip():
            return EvidenceSufficiency.INSUFFICIENT

        # Check for multi-clause questions (e.g. joined by "and", "compare ... with", "vs")
        clauses = re.split(r"\b(?:and|compare|versus|vs|difference between)\b", query, flags=re.IGNORECASE)
        clause_concepts = [self._extract_concepts(c) for c in clauses if c.strip()]
        clause_concepts = [c for c in clause_concepts if len(c) > 0]

        if len(clause_concepts) >= 2:
            clause_coverage = []
            for concepts in clause_concepts:
                hits = sum(1 for c in concepts if bool(re.search(r"\b" + re.escape(c) + r"\b", evidence_text)))
                ratio = hits / len(concepts) if concepts else 0.0
                clause_coverage.append(ratio)

            fully_covered = sum(1 for r in clause_coverage if r >= self.min_coverage_ratio)
            partially_covered = sum(1 for r in clause_coverage if r >= self.partial_coverage_ratio)

            if fully_covered == len(clause_concepts):
                return EvidenceSufficiency.SUFFICIENT
            elif partially_covered >= 1 and fully_covered < len(clause_concepts):
                return EvidenceSufficiency.PARTIALLY_SUFFICIENT
            elif partially_covered == 0:
                return EvidenceSufficiency.INSUFFICIENT
            else:
                return EvidenceSufficiency.PARTIALLY_SUFFICIENT

        # Single-clause inquiry: overall concept coverage
        query_concepts = self._extract_concepts(query)
        if not query_concepts:
            return EvidenceSufficiency.UNKNOWN

        hits = sum(1 for c in query_concepts if bool(re.search(r"\b" + re.escape(c) + r"\b", evidence_text)))
        overall_ratio = hits / len(query_concepts)

        if overall_ratio >= self.min_coverage_ratio:
            return EvidenceSufficiency.SUFFICIENT
        elif overall_ratio >= self.partial_coverage_ratio:
            return EvidenceSufficiency.PARTIALLY_SUFFICIENT
        else:
            return EvidenceSufficiency.INSUFFICIENT
