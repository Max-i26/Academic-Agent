"""Citation validator extracting, checking integrity of, and non-destructively resolving citations."""

import logging
import re
from typing import NamedTuple

from academic_agent.generation.synthesis.models import (
    CitationOccurrence,
    EvidenceContext,
    ValidationDiagnostics,
)
from academic_agent.generation.synthesis.protocols import CitationValidatorProtocol

logger = logging.getLogger(__name__)

# Single valid citation syntax: e.g. E1, E2, E99
SINGLE_EID_PATTERN = re.compile(r"^E[1-9]\d*$")

# Grouped valid citation syntax: e.g. E1, E2 or E1, E2, E3
GROUPED_EID_PATTERN = re.compile(r"^E[1-9]\d*(?:\s*,\s*E[1-9]\d*)+$")

# Malformed pattern heuristics (citation attempts that violate strict syntax)
MALFORMED_INDICATORS = re.compile(
    r"(?:\bE\d*\b|\be\d*\b|\bLecture\b|\bLect\b|\bpp?\.?\b|\bSlide\b|\bPage\b)",
    re.IGNORECASE,
)


class _SpanMatch(NamedTuple):
    raw: str
    inner: str
    start: int
    end: int


class CitationValidator(CitationValidatorProtocol):
    """Parses, validates integrity, and non-destructively resolves evidence citations.

    Enforces:
    - Half-open character ranges: start_char inclusive, end_char exclusive.
    - Code block and inline code masking: ignores citations inside fenced or inline code.
    - Preserves mathematical expressions while detecting legitimate citations.
    - Syntax validation: distinguishes valid single [E1], adjacent [E1][E2], grouped [E1, E2],
      and flags malformed citations (e.g. [Lecture 4], [E0], [E1,]).
    - Non-destructive resolution: original answer remains completely untouched.
    """

    @staticmethod
    def _extract_code_spans(text: str) -> list[tuple[int, int]]:
        """Identify character intervals belonging to fenced code blocks and inline code spans."""
        spans: list[tuple[int, int]] = []

        # 1. Fenced code blocks ```...```
        for m in re.finditer(r"```[\s\S]*?```", text):
            spans.append((m.start(), m.end()))

        # 2. Inline code spans `...` (only on non-fenced regions)
        for m in re.finditer(r"`[^`\n]+`", text):
            spans.append((m.start(), m.end()))

        return spans

    @staticmethod
    def _is_in_code(start: int, end: int, code_spans: list[tuple[int, int]]) -> bool:
        """Check if a candidate span falls within any code span."""
        for c_start, c_end in code_spans:
            if start >= c_start and end <= c_end:
                return True
        return False

    @staticmethod
    def _is_escaped(pos: int, text: str) -> bool:
        """Check if character at pos is preceded by an odd number of backslashes."""
        slashes = 0
        p = pos - 1
        while p >= 0 and text[p] == "\\":
            slashes += 1
            p -= 1
        return slashes % 2 == 1

    def _find_bracket_candidates(self, text: str) -> list[_SpanMatch]:
        """Find all non-code, unescaped bracketed tokens in text."""
        if not text:
            return []

        code_spans = self._extract_code_spans(text)
        candidates: list[_SpanMatch] = []

        # Find single-line bracketed expressions
        for m in re.finditer(r"\[([^\]\n]+)\]", text):
            start = m.start()
            end = m.end()

            if self._is_in_code(start, end, code_spans):
                continue
            if self._is_escaped(start, text):
                continue

            candidates.append(
                _SpanMatch(
                    raw=m.group(0),
                    inner=m.group(1).strip(),
                    start=start,
                    end=end,
                )
            )

        return candidates

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
        raw_text = answer_text or ""
        citations: list[CitationOccurrence] = []
        valid_cited_ids: list[str] = []
        invalid_cited_ids: list[str] = []
        malformed_citations: list[str] = []

        candidates = self._find_bracket_candidates(raw_text)

        for cand in candidates:
            inner = cand.inner

            # 1. Single valid citation ID: [E1], [E99]
            if SINGLE_EID_PATTERN.match(inner):
                eid = inner
                is_valid = eid in context.evidence_lookup
                res_label = context.evidence_lookup[eid].citation_label if is_valid else ""

                if is_valid and eid not in valid_cited_ids:
                    valid_cited_ids.append(eid)
                elif not is_valid and eid not in invalid_cited_ids:
                    invalid_cited_ids.append(eid)

                rel_pos = cand.raw.find(eid)
                sub_start = cand.start + rel_pos
                sub_end = sub_start + len(eid)

                citations.append(
                    CitationOccurrence(
                        evidence_id=eid,
                        raw_citation=cand.raw,
                        start_char=cand.start,
                        end_char=cand.end,
                        is_valid=is_valid,
                        resolution_label=res_label,
                        is_grouped=False,
                        group_span=(cand.start, cand.end),
                        sub_span=(sub_start, sub_end),
                    )
                )

            # 2. Grouped valid citation IDs: [E1, E2]
            elif GROUPED_EID_PATTERN.match(inner):
                # Match each individual token with its position inside cand.raw
                search_offset = 0
                for match_id in re.finditer(r"\bE[1-9]\d*\b", cand.raw):
                    eid = match_id.group(0)
                    is_valid = eid in context.evidence_lookup
                    res_label = context.evidence_lookup[eid].citation_label if is_valid else ""

                    if is_valid and eid not in valid_cited_ids:
                        valid_cited_ids.append(eid)
                    elif not is_valid and eid not in invalid_cited_ids:
                        invalid_cited_ids.append(eid)

                    sub_start = cand.start + match_id.start()
                    sub_end = cand.start + match_id.end()

                    citations.append(
                        CitationOccurrence(
                            evidence_id=eid,
                            raw_citation=cand.raw,
                            start_char=cand.start,
                            end_char=cand.end,
                            is_valid=is_valid,
                            resolution_label=res_label,
                            is_grouped=True,
                            group_span=(cand.start, cand.end),
                            sub_span=(sub_start, sub_end),
                        )
                    )

            # 3. Check for malformed citation attempts
            elif MALFORMED_INDICATORS.search(inner) or inner.startswith("E") or inner.startswith("e"):
                # Candidate has citation keywords/prefixes but violated strict syntax
                malformed_citations.append(cand.raw)

            # 4. Otherwise: ordinary bracketed math or prose (e.g. [0, 1], [X_t]) -> ignored

        # Compute context IDs never cited
        all_context_ids = [item.evidence_id for item in context.items]
        uncited_evidence_ids = [eid for eid in all_context_ids if eid not in valid_cited_ids]

        return ValidationDiagnostics(
            citations=citations,
            valid_cited_ids=valid_cited_ids,
            invalid_cited_ids=invalid_cited_ids,
            uncited_evidence_ids=uncited_evidence_ids,
            malformed_citations=malformed_citations,
            refusal_detected=False,
        )

    def resolve_citations(
        self,
        answer_text: str,
        context: EvidenceContext,
    ) -> str:
        """Replace valid IDs like '[E1]' with human-readable provenance (e.g. '[Lecture 4, p. 18]').

        Unrecognized or invalid IDs (e.g. '[E99]') remain unreplaced and flagged.
        Citations inside code blocks or inline code are never modified.
        Original answer_text is never mutated.

        Args:
            answer_text: Text containing citation tags.
            context: EvidenceContext containing valid mapping.

        Returns:
            Text with recognized tags safely resolved.
        """
        if not answer_text:
            return ""

        candidates = self._find_bracket_candidates(answer_text)
        if not candidates:
            return answer_text

        # Process candidates in reverse position order so string replacements do not affect earlier offsets
        resolved_text = answer_text
        for cand in reversed(candidates):
            inner = cand.inner

            # 1. Single valid citation
            if SINGLE_EID_PATTERN.match(inner):
                eid = inner
                if eid in context.evidence_lookup:
                    rep = context.evidence_lookup[eid].citation_label
                    resolved_text = (
                        resolved_text[: cand.start] + rep + resolved_text[cand.end :]
                    )

            # 2. Grouped citation: e.g. [E1, E2] or [E1, E99]
            elif GROUPED_EID_PATTERN.match(inner):
                sub_eids = [x.strip() for x in inner.split(",") if x.strip()]
                # Check if at least one ID is valid in context
                resolved_parts: list[str] = []
                has_any_valid = False

                for eid in sub_eids:
                    if eid in context.evidence_lookup:
                        # Extract inner label without brackets for clean grouping
                        label = context.evidence_lookup[eid].citation_label.strip("[]")
                        resolved_parts.append(label)
                        has_any_valid = True
                    else:
                        resolved_parts.append(eid)

                if has_any_valid:
                    combined_label = f"[{'; '.join(resolved_parts)}]"
                    resolved_text = (
                        resolved_text[: cand.start]
                        + combined_label
                        + resolved_text[cand.end :]
                    )

        return resolved_text
