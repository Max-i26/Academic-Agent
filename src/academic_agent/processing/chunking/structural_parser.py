from __future__ import annotations

import re
from typing import Sequence

from academic_agent.processing.chunking.models import ContentType, ContentUnit
from academic_agent.processing.text_segmenter import TextSegmenter


class StructuralParser:
    """
    Classifies and refines normalized ContentUnits into semantic academic structures.

    Identifies:
    - Major section and slide headings (using layout geometry, typography, and textual cues)
    - Distinguishes major headings from prose transitions (e.g. 'Therefore:') and inline labels
    - Equations (mathematical expressions and formulas)
    - Lists (bullet points and enumerated items)
    - Tables (tabular structures)
    - Paragraphs (explanatory prose with line-wrapping reconstruction)
    
    Dynamically tracks active section titles across units.
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

    PROSE_TRANSITIONS = {
        "therefore:",
        "however:",
        "note:",
        "notes:",
        "where:",
        "recall:",
        "hence:",
        "furthermore:",
        "moreover:",
        "thus:",
        "next:",
        "example:",
        "specifically:",
        "in other words:",
        "for example:",
    }

    INLINE_LABELS = {
        "smoothed levels:",
        "parameters:",
        "parameter values:",
        "initial values:",
        "step 1:",
        "step 2:",
        "step 3:",
        "step 4:",
        "solution:",
        "suggested answer:",
    }

    def __init__(self, segmenter: TextSegmenter | None = None) -> None:
        self.segmenter = segmenter or TextSegmenter()

    def parse(self, units: Sequence[ContentUnit]) -> list[ContentUnit]:
        """
        Classify content types, assign section titles, and segment prose text.
        """
        if not units:
            return []

        # 1. Estimate global typography baseline
        body_font_size = self._estimate_body_font_size(units)

        # 2. Map page-level maximum font sizes and detect running header/footer patterns
        page_max_fonts = self._compute_page_max_fonts(units)
        running_patterns = self._detect_running_patterns(units)

        parsed_units: list[ContentUnit] = []
        current_section_title: str | None = None

        for unit in units:
            content_type = self._classify_unit(
                unit=unit,
                body_font_size=body_font_size,
                page_max_font=page_max_fonts.get(unit.page_number),
                running_patterns=running_patterns,
            )

            # Update active section title if major heading is detected
            if content_type == ContentType.HEADING:
                current_section_title = self._clean_heading_title(unit.text)
                section_for_unit = current_section_title
            else:
                section_for_unit = current_section_title

            # Refine text formatting using segmenter for paragraphs and lists
            refined_text = unit.text
            if content_type in (ContentType.PARAGRAPH, ContentType.LIST):
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
        page_max_font: float | None,
        running_patterns: set[str],
    ) -> ContentType:
        """Determine the precise ContentType for a ContentUnit."""
        # Pre-classified tables retain their type
        if unit.content_type == ContentType.TABLE or self._is_table(unit.text):
            return ContentType.TABLE

        text = unit.text.strip()
        lower_text = text.lower()

        # 1. Filter out running headers and footers
        if self._is_running_header_footer(unit, lower_text, running_patterns):
            return ContentType.PARAGRAPH

        # 2. Filter out prose transitions and inline labels (e.g. 'Therefore:', 'Smoothed levels:')
        if self._is_prose_transition_or_label(lower_text):
            return ContentType.PARAGRAPH

        # 3. Check for major section or slide heading
        if self._is_heading(unit, text, body_font_size, page_max_font):
            return ContentType.HEADING

        # 4. Check for equations
        if self._is_equation(text):
            return ContentType.EQUATION

        # 5. Check for bullet lists
        if self._is_bullet_list(text):
            return ContentType.LIST

        # 6. Default to Paragraph
        return ContentType.PARAGRAPH

    def _is_prose_transition_or_label(self, lower_text: str) -> bool:
        """Check if text is a transitional connective or inline label ending in a colon."""
        if lower_text in self.PROSE_TRANSITIONS or lower_text in self.INLINE_LABELS:
            return True
        # If single short word ending with colon (e.g. "where:", "example:")
        if lower_text.endswith(":") and len(lower_text.split()) == 1:
            return True
        return False

    def _is_running_header_footer(
        self,
        unit: ContentUnit,
        lower_text: str,
        running_patterns: set[str],
    ) -> bool:
        """Identify slide footers and recurring course headers."""
        if lower_text in running_patterns:
            return True

        # Footer pattern detection
        if "slide" in lower_text and any(c.isdigit() for c in lower_text):
            if "@" in lower_text or "ds5104" in lower_text:
                return True

        # Check geometry: bottom 12% of slide or page
        bbox = unit.metadata.get("bbox")
        if bbox and len(bbox) == 4:
            y0 = bbox[1]
            page_height = unit.metadata.get("page_height", 540.0)
            if y0 >= (page_height * 0.88):
                if "ds5104" in lower_text or "slide" in lower_text or "@" in lower_text:
                    return True

        return False

    def _is_heading(
        self,
        unit: ContentUnit,
        text: str,
        body_font_size: float | None,
        page_max_font: float | None,
    ) -> bool:
        """Identify major section headings and slide titles."""
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if not lines:
            return False

        first_line = lines[0]
        bbox = unit.metadata.get("bbox")
        font_size = unit.metadata.get("font_size")

        # Layout-rich and typography evidence
        if font_size is not None and body_font_size is not None:
            if bbox is not None and len(bbox) == 4:
                y0 = bbox[1]
                page_height = unit.metadata.get("page_height", 540.0)
                # Relative vertical position: near top of slide or page (top 22% of page height)
                is_near_top = y0 <= (page_height * 0.22)
                is_prominent_font = (
                    page_max_font is not None
                    and font_size >= page_max_font * 0.88
                    and font_size >= body_font_size * 1.05
                )

                if (
                    is_near_top
                    and is_prominent_font
                    and len(text) < 140
                    and not text.startswith(("•", "-", "*"))
                    and not self._is_equation(text)
                ):
                    return True
            else:
                # Synthetic or layout-free record with typography metadata
                if (
                    font_size >= body_font_size * 1.15
                    and len(text) < 120
                    and not text.endswith((".", ";"))
                    and not self._is_equation(text)
                ):
                    return True

        # Textual heuristics (fallback or explicit section numbering)
        if len(lines) == 1 and self.HEADING_NUMBERED_PATTERN.match(first_line) and len(first_line) < 120:
            return True

        # Colon heading (e.g. "Simple Exponential Smoothing:"), but ONLY if title-like and not a transition
        if (
            len(lines) == 1
            and first_line.endswith(":")
            and len(first_line.split()) >= 2
            and len(first_line) < 80
            and not first_line.lower().startswith(("where", "note", "therefore", "e.g"))
        ):
            return True

        # ALL-CAPS section headers (e.g. "MEASURING ACCURACY", "WHY IS IT CALLED EXPONENTIAL?")
        if (
            len(lines) == 1
            and first_line.isupper()
            and len(first_line.split()) >= 1
            and len(first_line) > 3
            and len(first_line) < 70
            and not first_line.endswith((".", ";"))
            and not self._is_equation(first_line)
        ):
            return True

        return False

    def _is_equation(self, text: str) -> bool:
        """Determine whether text represents a standalone mathematical formula."""
        stripped = text.strip()

        if self.LATEX_BLOCK_PATTERN.match(stripped):
            return True

        lines = [line.strip() for line in stripped.splitlines() if line.strip()]
        if len(lines) == 1:
            line = lines[0]
            if self.segmenter._is_standalone_equation(line):
                return True

        if all(self.segmenter._is_standalone_equation(line) for line in lines):
            return True

        return False

    def _is_bullet_list(self, text: str) -> bool:
        """Check if lines represent bulleted or enumerated list items."""
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if not lines:
            return False

        bullet_count = sum(1 for line in lines if self.BULLET_PATTERN.match(line))
        return bullet_count >= 1 and (
            bullet_count / len(lines) >= 0.5 or self.BULLET_PATTERN.match(lines[0])
        )

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
        # Use lower median so body text is prioritized even in small samples
        mid = (len(font_sizes) - 1) // 2
        return font_sizes[mid]

    @staticmethod
    def _compute_page_max_fonts(units: Sequence[ContentUnit]) -> dict[int, float]:
        """Compute the maximum font size for each page."""
        page_max: dict[int, float] = {}
        for unit in units:
            p = unit.page_number
            fsize = unit.metadata.get("font_size")
            if fsize is not None and isinstance(fsize, (int, float)) and fsize > 0:
                page_max[p] = max(page_max.get(p, 0.0), float(fsize))
        return page_max

    @staticmethod
    def _detect_running_patterns(units: Sequence[ContentUnit]) -> set[str]:
        """Detect running text that repeats across 3 or more pages."""
        text_pages: dict[str, set[int]] = {}
        for unit in units:
            cleaned = unit.text.strip().lower()
            if cleaned:
                # Normalize digits (e.g. slide numbers) to group slide footers
                normalized = re.sub(r"\d+", "#", cleaned)
                text_pages.setdefault(normalized, set()).add(unit.page_number)
                text_pages.setdefault(cleaned, set()).add(unit.page_number)

        return {pattern for pattern, pages in text_pages.items() if len(pages) >= 3}
