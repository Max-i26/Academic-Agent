
import re


class TextSegmenter:
    """
    Reconstruct wrapped lines while preserving academic text structure.

    Handles:
    - Wrapped paragraphs
    - Wrapped bullet points
    - Headings
    - Standalone equations
    - Mathematical notation inside ordinary sentences
    - Paragraph boundaries
    - Hyphenated words split across lines
    """

    BULLET_PATTERN = re.compile(
        r"^\s*(?:[•●▪◦‣]|\-\s|→)\s*"
    )

    EQUATION_PATTERN = re.compile(
        r"[=≈≠≤≥]"
    )

    MATH_NOTATION_PATTERN = re.compile(
        r"[₀₁₂₃₄₅₆₇₈₉₊₋⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁼|∑∫√∞∂∆]"
    )

    GREEK_VARIABLES = "αβγδεζηθικλμνξοπρστυφχψω"

    SENTENCE_ENDINGS = (".", "?", "!", ";")

    INCOMPLETE_ENDINGS = (
        "=", "≈", "≠", "≤", "≥",
        "+", "-", "*", "/", "×", "÷",
    )

    CONNECTOR_PATTERN = re.compile(
        r"\b(?:and|or|to|of|for|with|from|by|at|in|on|a|an|the|what|that|is|are|was|were)$",
        re.IGNORECASE,
    )

    PROSE_CUES = {
         "a", "an", "and", "are", "as", "calculate",
         "explain", "for", "forecast", "from", "if",
         "in", "is", "it", "means", "of", "on", "or",
         "the", "then", "to", "use", "what", "when",
         "with",
         "current", "previous", "smoothed", "smoothing",
         "parameter", "weight", "past", "level",
         "observation", "horizon",
         }
    def segment(self, text: str) -> str:
        """
        Join likely wrapped lines without damaging meaningful structure.
        """

        if not isinstance(text, str):
            raise TypeError("text must be a string")

        output: list[str] = []
        current = ""

        for raw_line in text.splitlines():
            line = raw_line.strip()

            # Preserve paragraph boundaries.
            if not line:
                self._flush(current, output)
                current = ""

                if output and output[-1] != "":
                    output.append("")

                continue

            # A new bullet starts a new segment.
            if self._is_bullet(line):
                self._flush(current, output)
                current = line
                continue

            # Headings remain independent.
            if self._is_heading(line):
                self._flush(current, output)
                current = ""
                output.append(line)
                continue

            # Start a new segment.
            if not current:
                current = line

            # Join likely continuation lines.
            elif self._should_join(current, line):
                # Do not insert a space after a trailing hyphen.
                separator = (
                    "" if current.rstrip().endswith("-") else " "
                )
                current = f"{current}{separator}{line}"

            # Standalone equations remain independent.
            elif self._is_standalone_equation(line):
                self._flush(current, output)
                current = ""
                output.append(line)

            # Otherwise, finish the previous segment.
            else:
                self._flush(current, output)
                current = line

        self._flush(current, output)

        # Remove unnecessary trailing blank lines.
        while output and output[-1] == "":
            output.pop()

        return "\n".join(output)

    @staticmethod
    def _flush(current: str, output: list[str]) -> None:
        """Append a non-empty segment."""

        if current:
            output.append(current)

    def _is_bullet(self, line: str) -> bool:
        """Identify the beginning of a bullet point."""

        return bool(self.BULLET_PATTERN.match(line))

    @staticmethod
    def _is_heading(line: str) -> bool:
        """Identify headings ending with a colon."""

        return line.endswith(":")

    def _is_standalone_equation(self, line: str) -> bool:
        """
        Identify likely standalone equations.

        Mathematical symbols inside ordinary prose do not make
        the entire line an equation.
        """

        has_equation_operator = bool(
            self.EQUATION_PATTERN.search(line)
        )

        has_math_notation = bool(
            self.MATH_NOTATION_PATTERN.search(line)
        )

        if not has_equation_operator and not has_math_notation:
            return False

        words = re.findall(r"[a-zA-Z]+", line.lower())

        if any(word in self.PROSE_CUES for word in words):
            return False

        return True

    def _should_join(self, previous: str, current: str) -> bool:
        """Determine whether the current line continues the previous one."""

        # A new Greek-letter variable definition starts independently.
        if current[0] in self.GREEK_VARIABLES:
            return False

        # Do not join after completed sentences.
        if previous.endswith(self.SENTENCE_ENDINGS):
            return False

        previous_stripped = previous.rstrip()

        # Join when the previous line ends with an incomplete operator.
        if previous_stripped.endswith(self.INCOMPLETE_ENDINGS):
            return True

        # Join when the previous line ends with an incomplete word connector.
        last_word = previous_stripped.split()[-1].lower()

        if self.CONNECTOR_PATTERN.fullmatch(last_word):
            return True

        # Only ASCII lowercase continuations are treated as prose wrapping.
        return current[0].isascii() and current[0].islower()
