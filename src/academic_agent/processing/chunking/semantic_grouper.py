from __future__ import annotations

from typing import Sequence

from pydantic import BaseModel, Field

from academic_agent.processing.chunking.models import ContentType, ContentUnit


class SemanticGroup(BaseModel):
    """
    A cohesive cluster of ContentUnits sharing topical and structural context.
    """

    group_id: str
    document_id: str
    source_file: str
    section_title: str | None = None
    units: list[ContentUnit] = Field(default_factory=list)

    @property
    def source_pages(self) -> list[int]:
        """Ascending unique list of all pages contributing to this group."""
        return sorted({u.page_number for u in self.units if u.page_number >= 1})

    @property
    def source_unit_ids(self) -> list[str]:
        """Preserve exact ordered sequence of source unit identifiers."""
        return [u.unit_id for u in self.units]

    @property
    def text(self) -> str:
        """Joined formatted text across all units in the group."""
        return "\n\n".join(u.text.strip() for u in self.units if u.text.strip())

    @property
    def word_count(self) -> int:
        """Total word count across all units in the group."""
        return len(self.text.split()) if self.text else 0


class SemanticGrouper:
    """
    Combines structurally related ContentUnits into coherent groups using deterministic rules.

    Rules:
    - Binds headings with the following explanatory units (never leaves a dangling heading).
    - Preserves equations with their preceding or succeeding variable definitions.
    - Preserves bullet lists together.
    - Preserves table structures.
    - Splits groups when a new section heading is encountered.
    - Prevents mixing content across different source documents.
    """

    def group(self, units: Sequence[ContentUnit]) -> list[SemanticGroup]:
        """Group an ordered sequence of ContentUnits into SemanticGroups."""
        if not units:
            return []

        groups: list[SemanticGroup] = []
        current_units: list[ContentUnit] = []
        current_section: str | None = None
        group_counter = 0

        first_unit = units[0]
        current_doc = first_unit.document_id
        current_file = first_unit.source_file

        def _flush_group() -> None:
            nonlocal group_counter, current_units
            if not current_units:
                return

            group_id = f"{current_doc}_g{group_counter + 1:04d}"
            groups.append(
                SemanticGroup(
                    group_id=group_id,
                    document_id=current_doc,
                    source_file=current_file,
                    section_title=current_section,
                    units=list(current_units),
                )
            )
            group_counter += 1
            current_units = []

        for unit in units:
            # 1. Document boundary strictly forces a new group
            if unit.document_id != current_doc:
                _flush_group()
                current_doc = unit.document_id
                current_file = unit.source_file
                current_section = None

            # 2. Section heading handling
            if unit.content_type == ContentType.HEADING:
                # If current group has content beyond a previous heading, finalize it
                has_non_heading = any(u.content_type != ContentType.HEADING for u in current_units)
                if has_non_heading:
                    _flush_group()

                current_section = unit.section_title or unit.text.strip()
                current_units.append(unit)
                continue

            # 3. Native tables
            if unit.content_type == ContentType.TABLE:
                # If current group already contains body text, isolate the table with its intro
                has_paragraphs = any(u.content_type == ContentType.PARAGRAPH for u in current_units)
                if has_paragraphs:
                    _flush_group()

                current_units.append(unit)
                continue

            # 4. Equations, lists, and paragraphs
            current_units.append(unit)

        _flush_group()
        return groups
