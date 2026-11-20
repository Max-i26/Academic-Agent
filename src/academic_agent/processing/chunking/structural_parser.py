from __future__ import annotations

import re
from typing import Sequence

from academic_agent.processing.chunking.models import ContentType, ContentUnit
from academic_agent.processing.text_segmenter import TextSegmenter


class StructuralParser:
    """
    Classifies and refines normalized ContentUnits into semantic academic structures.

    Identifies:
    - Headings (using typography font sizes and structural textual patterns)
    - Equations (mathematical expressions and formulas)
    - Lists (bullet points and enumerated items)
    - Tables (tabular structures)
    - Paragraphs (explanatory prose with line-wrapping reconstruction)
    
    Also tracks active section titles across units.
    """

    HEADING_NUMBERED_PATTERN = re.compile(
        r"^(?:(?:Section|Lecture|Chapter|Part)\s+)?(?:\d+(?:\.\d+)*\.?|\d+[\.\)])\s+[A-Za-z]",
        re.IGNORECASE,
    )

    BULLET_PATTERN = re.compile(
        r"^\s*(?:[•●▪◦‣]|\-\s|\*\s|→|\d+[\.\)]\s)\s*"
    )

    LATEX_BLOCK_PATTERN = re.compile(
        r"^(?:\$\$|\\\[)[\s\S]*(?:\$\$|\\\])$"
    )

    def __init__(self, segmenter: TextSegmenter | None = None) -> None:
        self.segmenter = segmenter or TextSegmenter()

    def parse(self, units: Sequence[ContentUnit]) -> list[ContentUnit]:
        """
        Classify content types, assign section titles, and segment prose text.
        """
        if not units:
            return []

        # Calculate baseline body font size across document if available
        body_font_size = self._estimate_body_font_size(units)

        parsed_units: list[ContentUnit] = []
        current_section_title: str | None = None

        for unit in units:
            content_type = self._classify_unit(unit, body_font_size)
            
            # Update active section title if heading is detected
            if content_type == ContentType.HEADING:
                current_section_title = self._clean_heading_title(unit.text)
                section_for_unit = current_section_title
            else:
                section_for_unit = current_section_title

            # Refine text formatting using segmenter for paragraphs and lists
            refined_text = unit.text
            if content_type == ContentType.PARAGRAPH:
                refined_text = self.segmenter.segment(unit.text)
            elif content_type == ContentType.LIST:
                refined_text = self.segmenter.segment(unit.text)

            parsed_units.append(
                ContentUnit(
                    unit_id=unit.unit_id,
                    document_id=unit.document_id,
                    source_file=unit.source_file,
                    page_number=unit.page_number,
                    content_type=content_type,
                    text=refined_text,
                    block_number=unit.block_number,
                    section_title=section_for_unit,
                    metadata=dict(unit.metadata),
                )
            )

        return parsed_units

    def _classify_unit(
        self,
        unit: ContentUnit,
        body_font_size: float | None,
    ) -> ContentType:
        """Determine the precise ContentType for a ContentUnit."""
        # Pre-classified tables retain their type
        if unit.content_type == ContentType.TABLE or self._is_table(unit.text):
            return ContentType.TABLE

        text = unit.text.strip()
        font_size = unit.metadata.get("font_size")

        # 1. Heading check
        if self._is_heading(text, font_size, body_font_size):
            return ContentType.HEADING

        # 2. Equation check
        if self._is_equation(text):
            return ContentType.EQUATION

        # 3. Bullet list check
        if self._is_bullet_list(text):
            return ContentType.LIST

        # 4. Default to Paragraph
        return ContentType.PARAGRAPH

    def _is_heading(
        self,
        text: str,
        font_size: float | None,
        body_font_size: float | None,
    ) -> bool:
        """Identify headings via typography and pattern cues."""
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if not lines:
            return False

        first_line = lines[0]

        # Explicit colon heading (e.g. "Forecasting Models:")
        if len(lines) == 1 and first_line.endswith(":") and len(first_line) < 100:
            return True

        # Numbered headings (e.g. "1.2 Moving Averages" or "Lecture 4")
        if len(lines) == 1 and self.HEADING_NUMBERED_PATTERN.match(first_line) and len(first_line) < 120:
            return True

        # Font-size heuristic from PDF layout
        if font_size is not None and body_font_size is not None:
            # Heading font is at least 15% larger than baseline body font
            if font_size >= body_font_size * 1.15 and len(text) < 150 and not text.endswith((".", ";")):
                return True

        # ALL-CAPS section headers (e.g. "MEASURING ACCURACY")
        if (
            len(lines) == 1
            and first_line.isupper()
            and len(first_line) > 3
            and len(first_line) < 80
            and not first_line.endswith((".", "?", "!"))
            and not self._is_equation(first_line)
        ):
            return True

        return False

    def _is_equation(self, text: str) -> bool:
        """Determine whether text represents a standalone mathematical formula."""
        stripped = text.strip()

        # LaTeX delimiters
        if self.LATEX_BLOCK_PATTERN.match(stripped):
            return True

        # Standalone equation detection using segmenter logic
        lines = [line.strip() for line in stripped.splitlines() if line.strip()]
        if len(lines) == 1:
            line = lines[0]
            if self.segmenter._is_standalone_equation(line):
                return True

        # Multi-line mathematical formula blocks
        if all(self.segmenter._is_standalone_equation(line) for line in lines):
            return True

        return False

    def _is_bullet_list(self, text: str) -> bool:
        """Check if lines represent bulleted or enumerated list items."""
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if not lines:
            return False

        bullet_count = sum(1 for line in lines if self.BULLET_PATTERN.match(line))
        return bullet_count >= 1 and (bullet_count / len(lines) >= 0.5 or self.BULLET_PATTERN.match(lines[0]))

    @staticmethod
    def _is_table(text: str) -> bool:
        """Identify markdown table syntax."""
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if len(lines) >= 2:
            if lines[0].startswith("|") and lines[0].endswith("|"):
                if lines[1].startswith("|") and "---" in lines[1]:
                    return True
        return False

    @staticmethod
    def _clean_heading_title(text: str) -> str:
        """Format a clean section title string."""
        cleaned = text.strip().rstrip(":")
        # Normalize internal whitespace
        return " ".join(cleaned.split())

    @staticmethod
    def _estimate_body_font_size(units: Sequence[ContentUnit]) -> float | None:
        """Estimate median font size across units with typography metadata."""
        font_sizes: list[float] = []
        for unit in units:
            size = unit.metadata.get("font_size")
            if size is not None and isinstance(size, (int, float)) and size > 0:
                font_sizes.append(float(size))

        if not font_sizes:
            return None

        font_sizes.sort()
        mid = len(font_sizes) // 2
        return font_sizes[mid]
