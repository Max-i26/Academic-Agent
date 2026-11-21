"""Answer policy validator assessing compliance, refusal signals, and contradictory claims."""

import logging
import re

from academic_agent.generation.synthesis.models import (
    EvidenceContext,
    EvidenceSufficiency,
    PolicyOutcome,
    PolicyValidationResult,
    ValidationDiagnostics,
)
from academic_agent.generation.synthesis.protocols import AnswerPolicyValidatorProtocol

logger = logging.getLogger(__name__)

# Refusal statements asserting insufficiency of evidence
REFUSAL_PATTERNS = [
    re.compile(r"\b(?:do|does|did)\s+not\s+(?:contain|provide|have|include)\s+(?:(?:enough|sufficient|any)\s+)?(?:evidence|information|details)\b", re.IGNORECASE),
    re.compile(r"\b(?:cannot|can\s+not|could\s+not)\s+(?:reliably\s+)?(?:find|determine|establish|ascertain|answer|verify|say)\b", re.IGNORECASE),
    re.compile(r"\binsufficient\s+(?:evidence|information|materials?|data|reference|context)\b", re.IGNORECASE),
    re.compile(r"\bnot\s+enough\s+(?:evidence|information|materials?|data|context)\b", re.IGNORECASE),
    re.compile(r"\bnot\s+(?:mentioned|stated|covered|provided|found|specified|discussed)\s+in\s+the\s+(?:provided|retrieved|given)?\s*(?:evidence|materials?|lectures?|context|text|notes)\b", re.IGNORECASE),
    re.compile(r"\bno\s+information\s+.*?\b(?:provided|found|given)\b", re.IGNORECASE),
    re.compile(r"\bunable\s+to\s+(?:answer|determine|verify|ascertain)\b", re.IGNORECASE),
    re.compile(r"\bcannot\s+be\s+determined\s+from\s+the\s+(?:provided|retrieved)?\s*(?:evidence|materials?|lectures?|context)\b", re.IGNORECASE),
    re.compile(r"\bnot\s+possible\s+to\s+(?:determine|answer|know)\s+from\s+the\s+(?:provided|retrieved)?\s*(?:evidence|materials?)\b", re.IGNORECASE),
    re.compile(r"\bI\s+(?:don't|do\s+not)\s+have\s+enough\s+evidence\b", re.IGNORECASE),
    re.compile(r"\bI\s+(?:cannot|can\s+not)\s+find\s+sufficient\s+evidence\b", re.IGNORECASE),
]

# Non-refusal context patterns (negated refusal language or hypothesis testing)
NON_REFUSAL_CONTEXTS = [
    re.compile(r"\b(?:reject|test|accept)\s+the\s+hypothesis\s+that\b", re.IGNORECASE),
    re.compile(r"\bassume\s+that\s+evidence\s+is\b", re.IGNORECASE),
    re.compile(r"\bdoes\s+not\s+(?:imply|mean)\s+(?:that\s+)?evidence\s+is\b", re.IGNORECASE),
]

# Explicit gap statements for partial answers
GAP_PATTERNS = [
    re.compile(r"\bhowever,\s*(?:the\s+)?(?:provided|retrieved)?\s*(?:lecture\s+)?(?:materials?|evidence|notes?)\s+(?:do|does)\s+not\s+(?:establish|specify|state|mention|contain|provide)\b", re.IGNORECASE),
    re.compile(r"\bthe\s+(?:provided\s+)?(?:lecture\s+)?(?:evidence|materials?|notes?)\s+(?:do|does)\s+not\s+(?:establish|specify|state|mention|contain|provide)\b", re.IGNORECASE),
    re.compile(r"\bcannot\s+be\s+determined\s+from\s+the\s+(?:provided|available)?\s*(?:evidence|materials?|lectures?)\b", re.IGNORECASE),
    re.compile(r"\b(?:does|do)\s+not\s+establish\s+the\s+(?:appropriate\s+)?(?:model\s+order|order|parameter|value)\b", re.IGNORECASE),
    re.compile(r"\bthe\s+evidence\s+does\s+not\s+mention\b", re.IGNORECASE),
    re.compile(r"\bdoes\s+not\s+mention\s+whether\b", re.IGNORECASE),
    re.compile(r"\bno\s+details\s+are\s+provided\s+(?:about|regarding|for)\b", re.IGNORECASE),
    re.compile(r"\bportion\s+is\s+unsupported\b", re.IGNORECASE),
    re.compile(r"\bunsupported\s+parts?\b", re.IGNORECASE),
]

# Substantive continuation indicators
SUBSTANTIVE_LEADS = [
    re.compile(r"\b(?:however|nevertheless|nonetheless|but|yet|that\s+said|having\s+said\s+that|although)\b", re.IGNORECASE),
    re.compile(r"\b(?:the\s+)?(?:correct|actual|true|exact)\s+(?:model|answer|order|value|formula|parameter)\s+is\b", re.IGNORECASE),
    re.compile(r"\b(?:is\s+definitely|is\s+given\s+by|is\s+defined\s+as)\b", re.IGNORECASE),
    re.compile(r"\bARIMA\s*\(\s*\d+\s*,\s*\d+\s*,\s*\d+\s*\)", re.IGNORECASE),
    re.compile(r"\bAR\s*\(\s*\d+\s*\)", re.IGNORECASE),
    re.compile(r"\bMA\s*\(\s*\d+\s*\)", re.IGNORECASE),
    re.compile(r"\$[^$]+\$"),  # LaTeX equations
    re.compile(r"\\\[[\s\S]+?\\\]"),  # Display equations
    re.compile(r"\b\\(?:sigma|phi|theta|mu|gamma)\b"),  # LaTeX symbols
]


class AnswerPolicyValidator(AnswerPolicyValidatorProtocol):
    """Assesses whether an answer adheres to academic answering and abstention policy.

    Evaluates:
    - Policy outcomes: ANSWERED, COMPLIANT_ABSTENTION, PARTIAL_ANSWER, CONTRADICTORY_REFUSAL,
      UNSUPPORTED_ANSWER, INDETERMINATE.
    - Detection of contradictory refusal (refusal followed by ungrounded claims/derivations).
    - Preservation of legitimate partial answers.
    - Distinction between observable policy signals and complete semantic factual verification.
    """

    def _detect_refusals(self, text: str) -> list[tuple[str, int, int]]:
        """Find valid refusal phrases while excluding non-refusal contexts."""
        matches: list[tuple[str, int, int]] = []
        for pattern in REFUSAL_PATTERNS:
            for m in pattern.finditer(text):
                start, end = m.start(), m.end()
                # Check for non-refusal context surrounding match
                context_window = text[max(0, start - 50) : min(len(text), end + 50)]
                is_non_refusal = any(nr.search(context_window) for nr in NON_REFUSAL_CONTEXTS)
                if not is_non_refusal:
                    matches.append((m.group(0), start, end))

        # Sort matches by start position
        matches.sort(key=lambda x: x[1])
        return matches

    def _has_substantive_continuation(self, text: str, refusal_matches: list[tuple[str, int, int]]) -> bool:
        """Check if substantive factual claims or equations follow or contradict a refusal statement."""
        if not refusal_matches:
            return False

        # Split into sentences or lines
        # First check intra-sentence contradiction (e.g. "I refuse, but the answer is X")
        for phrase, start, end in refusal_matches:
            # Find the sentence containing this refusal
            sent_start = text.rfind("\n", 0, start)
            if sent_start == -1:
                sent_start = 0
            else:
                sent_start += 1

            sent_end = text.find("\n", end)
            if sent_end == -1:
                sent_end = len(text)

            sentence = text[sent_start:sent_end]
            post_refusal_in_sent = text[end:sent_end].strip()

            # If post-refusal text in the same sentence has contrastive transition + claim
            for pat in SUBSTANTIVE_LEADS:
                if pat.search(post_refusal_in_sent):
                    return True

        # Now check subsequent sentences / paragraphs after the refusal
        last_refusal_end = max(m[2] for m in refusal_matches)
        remaining_text = text[last_refusal_end:].strip()
        if not remaining_text:
            return False

        # Find any text after the end of the sentence containing the last refusal
        period_pos = text.find(".", last_refusal_end)
        newline_pos = text.find("\n", last_refusal_end)
        cutoff_candidates = [pos for pos in (period_pos, newline_pos) if pos != -1]
        post_sent_start = min(cutoff_candidates) + 1 if cutoff_candidates else len(text)

        subsequent_text = text[post_sent_start:].strip()
        if not subsequent_text:
            return False

        # Check if subsequent text contains substantive leads or equations
        for pat in SUBSTANTIVE_LEADS:
            if pat.search(subsequent_text):
                return True

        if re.search(r"\[E[1-9]\d*\]", subsequent_text):
            return True

        # Check if subsequent sentences are substantive and not additional refusals
        sub_sentences = [s.strip() for s in re.split(r"[.!?]\s+", subsequent_text) if s.strip()]
        for s in sub_sentences:
            is_refusal = any(rp.search(s) for rp in REFUSAL_PATTERNS)
            if not is_refusal and len(s) > 15:
                return True

        return False

    def _detect_gap_statements(self, text: str) -> bool:
        """Check if the text explicitly states unsupported gaps for partial answers."""
        return any(pat.search(text) for pat in GAP_PATTERNS)

    def validate_policy(
        self,
        answer_text: str,
        context: EvidenceContext,
        diagnostics: ValidationDiagnostics,
        sufficiency: EvidenceSufficiency = EvidenceSufficiency.UNKNOWN,
    ) -> PolicyValidationResult:
        """Assess whether generated answer adheres to academic answering and abstention policy.

        Args:
            answer_text: Raw completion generated by the LLM.
            context: EvidenceContext assembled for the prompt.
            diagnostics: Citation integrity and occurrence diagnostics.
            sufficiency: Assessed or declared evidence sufficiency.

        Returns:
            PolicyValidationResult with categorical outcome, compliance flag, and rationale.
        """
        raw_text = (answer_text or "").strip()
        warnings: list[str] = list(diagnostics.warnings)

        # 1. Detect refusal statements
        refusal_matches = self._detect_refusals(raw_text)
        refusal_detected = len(refusal_matches) > 0
        refusal_phrases = [m[0] for m in refusal_matches]

        # Check substantive continuation after refusal statements
        has_substantive_continuation = False
        if refusal_detected:
            has_substantive_continuation = self._has_substantive_continuation(
                raw_text, refusal_matches
            )

        # 2. Detect gap statement for partial answers
        has_gap_statement = self._detect_gap_statements(raw_text)

        # 3. Decision Logic across Sufficiency and Observable Signals

        # Case A: Empty Context
        if len(context.items) == 0:
            if refusal_detected and not has_substantive_continuation:
                return PolicyValidationResult(
                    outcome=PolicyOutcome.COMPLIANT_ABSTENTION,
                    refusal_detected=True,
                    refusal_phrases=refusal_phrases,
                    has_substantive_continuation=False,
                    has_gap_statement=has_gap_statement,
                    is_compliant=True,
                    explanation="Compliant abstention: empty evidence context and model refused without ungrounded continuation.",
                    warnings=warnings,
                )
            elif refusal_detected and has_substantive_continuation:
                return PolicyValidationResult(
                    outcome=PolicyOutcome.CONTRADICTORY_REFUSAL,
                    refusal_detected=True,
                    refusal_phrases=refusal_phrases,
                    has_substantive_continuation=True,
                    has_gap_statement=has_gap_statement,
                    is_compliant=False,
                    explanation="Contradictory refusal: model refused but proceeded to make substantive ungrounded claims with zero evidence.",
                    warnings=warnings,
                )
            else:
                return PolicyValidationResult(
                    outcome=PolicyOutcome.UNSUPPORTED_ANSWER,
                    refusal_detected=False,
                    refusal_phrases=[],
                    has_substantive_continuation=False,
                    has_gap_statement=has_gap_statement,
                    is_compliant=False,
                    explanation="Unsupported answer: model made claims despite empty evidence context.",
                    warnings=warnings,
                )

        # Case B: Refusal detected with substantive continuation
        if refusal_detected and has_substantive_continuation:
            return PolicyValidationResult(
                outcome=PolicyOutcome.CONTRADICTORY_REFUSAL,
                refusal_detected=True,
                refusal_phrases=refusal_phrases,
                has_substantive_continuation=True,
                has_gap_statement=has_gap_statement,
                is_compliant=False,
                explanation="Contradictory refusal: model emitted an abstention phrase but continued with substantive claims or derivations.",
                warnings=warnings,
            )

        # Case C: Clean refusal detected without substantive continuation
        if refusal_detected and not has_substantive_continuation:
            # Rejection guard: fabricated citation IDs always invalidate the response
            if len(diagnostics.invalid_cited_ids) > 0:
                return PolicyValidationResult(
                    outcome=PolicyOutcome.UNSUPPORTED_ANSWER,
                    refusal_detected=True,
                    refusal_phrases=refusal_phrases,
                    has_substantive_continuation=False,
                    has_gap_statement=has_gap_statement,
                    is_compliant=False,
                    explanation=f"Unsupported answer: model cited invalid or fabricated citation IDs {diagnostics.invalid_cited_ids}.",
                    warnings=warnings,
                )

            if (
                sufficiency == EvidenceSufficiency.PARTIALLY_SUFFICIENT
                and has_gap_statement
                and len(diagnostics.valid_cited_ids) > 0
                and len(diagnostics.invalid_cited_ids) == 0
            ):
                return PolicyValidationResult(
                    outcome=PolicyOutcome.PARTIAL_ANSWER,
                    refusal_detected=True,
                    refusal_phrases=refusal_phrases,
                    has_substantive_continuation=False,
                    has_gap_statement=True,
                    is_compliant=True,
                    explanation="Partial answer: supported portions are grounded with citations, and unsupported gaps are explicitly identified.",
                    warnings=warnings,
                )

            if len(diagnostics.valid_cited_ids) == 0:
                if sufficiency == EvidenceSufficiency.SUFFICIENT:
                    warnings.append("Model refused to answer despite evidence being assessed as sufficient.")
                return PolicyValidationResult(
                    outcome=PolicyOutcome.COMPLIANT_ABSTENTION,
                    refusal_detected=True,
                    refusal_phrases=refusal_phrases,
                    has_substantive_continuation=False,
                    has_gap_statement=has_gap_statement,
                    is_compliant=True,
                    explanation="Compliant abstention: model explicitly refused without making unsupported continuation claims.",
                    warnings=warnings,
                )

            # If model has valid citations under partial sufficiency but lacks an explicit gap statement
            return PolicyValidationResult(
                outcome=PolicyOutcome.UNSUPPORTED_ANSWER,
                refusal_detected=True,
                refusal_phrases=refusal_phrases,
                has_substantive_continuation=False,
                has_gap_statement=False,
                is_compliant=False,
                explanation="Unsupported answer: partial evidence cited, but model failed to provide a compliant gap statement.",
                warnings=warnings,
            )

        # Case D: No refusal detected, but evidence is INSUFFICIENT
        if sufficiency == EvidenceSufficiency.INSUFFICIENT:
            # Model emitted claims without abstaining when evidence is insufficient
            return PolicyValidationResult(
                outcome=PolicyOutcome.UNSUPPORTED_ANSWER,
                refusal_detected=False,
                refusal_phrases=[],
                has_substantive_continuation=False,
                has_gap_statement=has_gap_statement,
                is_compliant=False,
                explanation="Unsupported answer: evidence was insufficient, but model provided an ungrounded answer without abstaining.",
                warnings=warnings,
            )

        # Case E: Evidence is PARTIALLY_SUFFICIENT
        if sufficiency == EvidenceSufficiency.PARTIALLY_SUFFICIENT:
            # Strict citation integrity guard: fabricated citation IDs always invalidate partial answers
            if len(diagnostics.invalid_cited_ids) > 0:
                return PolicyValidationResult(
                    outcome=PolicyOutcome.UNSUPPORTED_ANSWER,
                    refusal_detected=refusal_detected,
                    refusal_phrases=refusal_phrases,
                    has_substantive_continuation=False,
                    has_gap_statement=has_gap_statement,
                    is_compliant=False,
                    explanation=f"Unsupported answer: model cited invalid or fabricated citation IDs {diagnostics.invalid_cited_ids} in partial answer.",
                    warnings=warnings,
                )

            if (
                has_gap_statement
                and len(diagnostics.valid_cited_ids) > 0
                and len(diagnostics.invalid_cited_ids) == 0
            ):
                return PolicyValidationResult(
                    outcome=PolicyOutcome.PARTIAL_ANSWER,
                    refusal_detected=refusal_detected,
                    refusal_phrases=refusal_phrases,
                    has_substantive_continuation=False,
                    has_gap_statement=True,
                    is_compliant=True,
                    explanation="Partial answer: supported portions are grounded with citations, and unsupported gaps are explicitly identified.",
                    warnings=warnings,
                )
            else:
                return PolicyValidationResult(
                    outcome=PolicyOutcome.UNSUPPORTED_ANSWER,
                    refusal_detected=refusal_detected,
                    refusal_phrases=refusal_phrases,
                    has_substantive_continuation=False,
                    has_gap_statement=has_gap_statement,
                    is_compliant=False,
                    explanation="Unsupported answer: partial evidence was available, but model failed to explicitly identify the unsupported gaps with valid citations.",
                    warnings=warnings,
                )

        # Case F: Evidence is SUFFICIENT
        if sufficiency == EvidenceSufficiency.SUFFICIENT:
            # Check citation integrity
            if len(diagnostics.invalid_cited_ids) > 0:
                return PolicyValidationResult(
                    outcome=PolicyOutcome.UNSUPPORTED_ANSWER,
                    refusal_detected=False,
                    refusal_phrases=[],
                    has_substantive_continuation=False,
                    has_gap_statement=has_gap_statement,
                    is_compliant=False,
                    explanation=f"Unsupported answer: model cited invalid or fabricated citation IDs {diagnostics.invalid_cited_ids}.",
                    warnings=warnings,
                )

            if len(diagnostics.valid_cited_ids) > 0:
                return PolicyValidationResult(
                    outcome=PolicyOutcome.ANSWERED,
                    refusal_detected=False,
                    refusal_phrases=[],
                    has_substantive_continuation=False,
                    has_gap_statement=has_gap_statement,
                    is_compliant=True,
                    explanation="Answered: model provided an answer grounded with valid citations under sufficient evidence.",
                    warnings=warnings,
                )
            else:
                # No citations provided under sufficient evidence
                warnings.append("Answer lacks citation markers despite available sufficient evidence.")
                return PolicyValidationResult(
                    outcome=PolicyOutcome.INDETERMINATE,
                    refusal_detected=False,
                    refusal_phrases=[],
                    has_substantive_continuation=False,
                    has_gap_statement=has_gap_statement,
                    is_compliant=False,
                    explanation="Indeterminate: answer lacks citations, preventing observable grounding verification.",
                    warnings=warnings,
                )

        # Case G: Sufficiency is UNKNOWN
        return PolicyValidationResult(
            outcome=PolicyOutcome.INDETERMINATE,
            refusal_detected=refusal_detected,
            refusal_phrases=refusal_phrases,
            has_substantive_continuation=has_substantive_continuation,
            has_gap_statement=has_gap_statement,
            is_compliant=False,
            explanation="Indeterminate: evidence sufficiency is UNKNOWN and no decisive policy violation was triggered.",
            warnings=warnings,
        )
