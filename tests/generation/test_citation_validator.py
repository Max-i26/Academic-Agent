"""Unit tests for Stage 5.2C.2 CitationValidator."""

import pytest

from academic_agent.generation.synthesis.models import (
    CitationOccurrence,
    EvidenceContext,
    GroundedEvidence,
    ValidationDiagnostics,
)
from academic_agent.generation.synthesis.protocols import CitationValidatorProtocol
from academic_agent.generation.synthesis.validation.citation_validator import CitationValidator
from academic_agent.retrieval.models import Evidence


@pytest.fixture
def sample_evidence_e1() -> GroundedEvidence:
    ev = Evidence(
        record_id="rec_1#0",
        chunk_id="c1",
        text="Weak stationarity requires constant mean and autocovariance.",
        source_file="Lecture_4.pdf",
        source_pages=[18],
        score=0.92,
    )
    return GroundedEvidence(evidence_id="E1", evidence=ev)


@pytest.fixture
def sample_evidence_e2() -> GroundedEvidence:
    ev = Evidence(
        record_id="rec_2#0",
        chunk_id="c2",
        text="The PACF measures the relationship after controlling for intermediate lags.",
        source_file="Lecture_5.pdf",
        source_pages=[8, 9],
        score=0.88,
    )
    return GroundedEvidence(evidence_id="E2", evidence=ev)


@pytest.fixture
def sample_context(sample_evidence_e1, sample_evidence_e2) -> EvidenceContext:
    return EvidenceContext(
        items=[sample_evidence_e1, sample_evidence_e2],
        budget_allocated=1000,
    )


class TestCitationValidatorBasics:
    """Verify protocol conformance, empty cases, and character range validity."""

    def test_protocol_conformance(self):
        validator = CitationValidator()
        assert isinstance(validator, CitationValidatorProtocol)

    def test_empty_answer_and_empty_context(self):
        validator = CitationValidator()
        empty_ctx = EvidenceContext()

        diag = validator.validate("", empty_ctx)
        assert len(diag.citations) == 0
        assert len(diag.valid_cited_ids) == 0
        assert len(diag.invalid_cited_ids) == 0
        assert len(diag.malformed_citations) == 0

        res = validator.resolve_citations("", empty_ctx)
        assert res == ""

    def test_citation_at_character_zero(self, sample_context):
        validator = CitationValidator()
        text = "[E1] defines the condition for weak stationarity."

        diag = validator.validate(text, sample_context)
        assert len(diag.citations) == 1
        occ = diag.citations[0]
        assert occ.evidence_id == "E1"
        assert occ.start_char == 0
        assert occ.end_char == 4
        assert text[occ.start_char : occ.end_char] == occ.raw_citation
        assert occ.is_valid is True

        resolved = validator.resolve_citations(text, sample_context)
        assert resolved.startswith("[Lecture 4, p. 18] defines the condition")

    def test_citation_at_end_of_answer(self, sample_context):
        validator = CitationValidator()
        text = "This result is proven in [E1]"

        diag = validator.validate(text, sample_context)
        assert len(diag.citations) == 1
        occ = diag.citations[0]
        assert occ.end_char == len(text)
        assert text[occ.start_char : occ.end_char] == "[E1]"

        resolved = validator.resolve_citations(text, sample_context)
        assert resolved == "This result is proven in [Lecture 4, p. 18]"

    def test_exact_offsets_with_unicode_preceding_text(self, sample_context):
        validator = CitationValidator()
        text = "L'analyse d'autocorrélation et la stationnarité faible sont établies [E1]."

        diag = validator.validate(text, sample_context)
        assert len(diag.citations) == 1
        occ = diag.citations[0]
        assert text[occ.start_char : occ.end_char] == "[E1]"
        assert occ.is_valid is True


class TestCitationSyntaxVariants:
    """Verify single, adjacent, grouped, and repeated citations."""

    def test_adjacent_citations(self, sample_context):
        validator = CitationValidator()
        text = "The properties are discussed in [E1][E2]."

        diag = validator.validate(text, sample_context)
        assert len(diag.citations) == 2
        occ1, occ2 = diag.citations

        assert occ1.evidence_id == "E1"
        assert occ1.raw_citation == "[E1]"
        assert text[occ1.start_char : occ1.end_char] == "[E1]"

        assert occ2.evidence_id == "E2"
        assert occ2.raw_citation == "[E2]"
        assert text[occ2.start_char : occ2.end_char] == "[E2]"
        assert occ1.end_char == occ2.start_char

        resolved = validator.resolve_citations(text, sample_context)
        assert resolved == "The properties are discussed in [Lecture 4, p. 18][Lecture 5, pp. 8-9]."

    def test_repeated_citations_preserved_at_occurrence_level(self, sample_context):
        validator = CitationValidator()
        text = "Weak stationarity [E1] requires constant mean. PACF [E2] cuts off. Recall [E1] again."

        diag = validator.validate(text, sample_context)
        # 3 distinct occurrences preserved
        assert len(diag.citations) == 3
        assert diag.citations[0].evidence_id == "E1"
        assert diag.citations[1].evidence_id == "E2"
        assert diag.citations[2].evidence_id == "E1"

        # Unique cited IDs preserved in first-occurrence order
        assert diag.valid_cited_ids == ["E1", "E2"]

        # All 3 offsets exact
        for occ in diag.citations:
            assert text[occ.start_char : occ.end_char] == occ.raw_citation

        resolved = validator.resolve_citations(text, sample_context)
        assert resolved.count("[Lecture 4, p. 18]") == 2
        assert resolved.count("[Lecture 5, pp. 8-9]") == 1

    def test_grouped_citations(self, sample_context):
        validator = CitationValidator()
        text = "Both models are evaluated together [E1, E2]."

        diag = validator.validate(text, sample_context)
        assert len(diag.citations) == 2
        occ1, occ2 = diag.citations

        # Group span check
        assert occ1.is_grouped is True
        assert occ2.is_grouped is True
        assert occ1.group_span == (occ1.start_char, occ1.end_char)
        assert occ2.group_span == (occ2.start_char, occ2.end_char)
        assert text[occ1.start_char : occ1.end_char] == "[E1, E2]"
        assert text[occ2.start_char : occ2.end_char] == "[E1, E2]"

        # Individual evidence reference sub_span check
        assert occ1.sub_span is not None
        assert text[occ1.sub_span[0] : occ1.sub_span[1]] == "E1"
        assert occ2.sub_span is not None
        assert text[occ2.sub_span[0] : occ2.sub_span[1]] == "E2"

        # Resolution replaces group exactly once without duplication
        resolved = validator.resolve_citations(text, sample_context)
        assert resolved == "Both models are evaluated together [Lecture 4, p. 18; Lecture 5, pp. 8-9]."

    def test_unknown_id_mixed_with_valid_id_in_group(self, sample_context):
        validator = CitationValidator()
        text = "The analysis combines established results [E1, E99]."

        diag = validator.validate(text, sample_context)
        assert len(diag.citations) == 2
        assert diag.citations[0].evidence_id == "E1"
        assert diag.citations[0].is_valid is True
        assert diag.citations[1].evidence_id == "E99"
        assert diag.citations[1].is_valid is False

        assert diag.valid_cited_ids == ["E1"]
        assert diag.invalid_cited_ids == ["E99"]

        # Non-destructive resolution keeps invalid E99 intact
        resolved = validator.resolve_citations(text, sample_context)
        assert resolved == "The analysis combines established results [Lecture 4, p. 18; E99]."

    def test_mixed_grouped_citations_resolved_exactly_once(self, sample_context):
        """Verify each grouped citation is resolved exactly once without overlapping replacement corruption."""
        validator = CitationValidator()
        text = "First group [E1, E2], second group [E1, E99], third single [E2]."

        resolved = validator.resolve_citations(text, sample_context)
        expected = (
            "First group [Lecture 4, p. 18; Lecture 5, pp. 8-9], "
            "second group [Lecture 4, p. 18; E99], "
            "third single [Lecture 5, pp. 8-9]."
        )
        assert resolved == expected

    def test_uncited_evidence_ids_tracked(self, sample_context):
        validator = CitationValidator()
        text = "Only the first evidence item is referenced [E1]."

        diag = validator.validate(text, sample_context)
        assert diag.valid_cited_ids == ["E1"]
        assert diag.uncited_evidence_ids == ["E2"]


class TestCodeMaskingAndMathPreservation:
    """Verify that citations in code are ignored while math expressions are preserved."""

    def test_citation_inside_fenced_code_block_ignored(self, sample_context):
        validator = CitationValidator()
        text = (
            "Here is a code example:\n"
            "```python\n"
            "# Reference [E1] inside code\n"
            "citation = '[E2]'\n"
            "```\n"
            "And here is real explanation [E1]."
        )

        diag = validator.validate(text, sample_context)
        # Only the citation outside code must be extracted!
        assert len(diag.citations) == 1
        assert diag.citations[0].evidence_id == "E1"
        assert text[diag.citations[0].start_char : diag.citations[0].end_char] == "[E1]"

        resolved = validator.resolve_citations(text, sample_context)
        assert "citation = '[E2]'" in resolved  # Untouched inside code!
        assert "# Reference [E1]" in resolved  # Untouched inside code!
        assert "And here is real explanation [Lecture 4, p. 18]." in resolved

    def test_citation_inside_inline_code_span_ignored(self, sample_context):
        validator = CitationValidator()
        text = "Run the command `load_evidence('[E1]')` to inspect evidence [E2]."

        diag = validator.validate(text, sample_context)
        assert len(diag.citations) == 1
        assert diag.citations[0].evidence_id == "E2"

        resolved = validator.resolve_citations(text, sample_context)
        assert "`load_evidence('[E1]')`" in resolved
        assert "[Lecture 5, pp. 8-9]" in resolved

    def test_citation_in_mathematical_derivation_extracted(self, sample_context):
        validator = CitationValidator()
        text = "The variance is $\\mathrm{Var}(X_t) = \\frac{\\sigma^2}{1 - \\phi_1^2}$ [E1]."

        diag = validator.validate(text, sample_context)
        assert len(diag.citations) == 1
        assert diag.citations[0].evidence_id == "E1"
        assert diag.citations[0].is_valid is True

        resolved = validator.resolve_citations(text, sample_context)
        assert "$\\mathrm{Var}(X_t) = \\frac{\\sigma^2}{1 - \\phi_1^2}$ [Lecture 4, p. 18]." in resolved

    def test_escaped_brackets_not_treated_as_citation(self, sample_context):
        validator = CitationValidator()
        text = "The notation \\[E1\\] denotes an interval, whereas [E1] is a citation."

        diag = validator.validate(text, sample_context)
        assert len(diag.citations) == 1
        assert diag.citations[0].evidence_id == "E1"
        assert diag.citations[0].start_char > text.find("\\[E1\\]")

    def test_non_citation_math_brackets_not_flagged_as_malformed(self, sample_context):
        validator = CitationValidator()
        text = "The random variable interval is [0, 1] and state is $X[t]$ [E1]."

        diag = validator.validate(text, sample_context)
        assert len(diag.citations) == 1
        assert diag.citations[0].evidence_id == "E1"
        assert len(diag.malformed_citations) == 0


class TestMalformedCitationsAndResolutionSafety:
    """Verify malformed citation detection and non-destructive resolution guarantees."""

    @pytest.mark.parametrize(
        "malformed_text",
        [
            "According to [Lecture 4], stationarity holds.",
            "Refer to [Lecture 4, p. 18] for proofs.",
            "Invalid ID index [E0] cited.",
            "Trailing comma syntax [E1,].",
            "Double comma syntax [E1,,E2].",
            "Bare letter [E] cited.",
            "Lowercase prefix [e1] cited.",
            "Hyphenated index [E-1] cited.",
        ],
    )
    def test_malformed_citation_detection(self, malformed_text, sample_context):
        validator = CitationValidator()
        diag = validator.validate(malformed_text, sample_context)

        assert len(diag.malformed_citations) >= 1
        assert len(diag.valid_cited_ids) == 0

    def test_non_destructive_resolution_invariant(self, sample_context):
        """The original answer string must remain 100% identical and unmodified."""
        validator = CitationValidator()
        original_answer = "ACF measures [E1]. PACF measures [E2]. Unknown is [E99]."
        answer_before_resolution = str(original_answer)

        resolved_answer = validator.resolve_citations(original_answer, sample_context)

        # Invariant check from specification
        assert original_answer == answer_before_resolution
        assert resolved_answer != original_answer
        assert "[Lecture 4, p. 18]" in resolved_answer
        assert "[Lecture 5, pp. 8-9]" in resolved_answer
        assert "[E99]" in resolved_answer  # Unresolved citation kept untouched
