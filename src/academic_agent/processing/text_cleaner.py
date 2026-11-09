
import re


class TextCleaner:
    """Conservatively normalize extracted academic text."""

    def clean(self, text: str) -> str:
        """
        Normalize whitespace while preserving meaningful line breaks
        and Unicode academic content.
        """
        if not isinstance(text, str):
            raise TypeError("text must be a string")

        # Normalize different operating-system line endings.
        text = text.replace("\r\n", "\n").replace("\r", "\n")

        # Normalize non-breaking spaces and remove a leading BOM.
        text = text.replace("\u00a0", " ").lstrip("\ufeff")

        # Normalize horizontal whitespace without removing line breaks.
        lines = []
        for line in text.split("\n"):
            line = re.sub(r"[ \t]+", " ", line)
            lines.append(line.strip())

        text = "\n".join(lines)

        # Collapse excessive blank lines, preserving paragraph boundaries.
        text = re.sub(r"\n{3,}", "\n\n", text)

        return text.strip()
