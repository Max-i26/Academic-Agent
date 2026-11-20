from __future__ import annotations

from typing import Any, Sequence

from academic_agent.ingestion.layout_extractor import PageLayoutRecord, TableRecord, TextBlockRecord
from academic_agent.ingestion.pdf_extractor import PageRecord
from academic_agent.processing.chunking.models import ContentType, ContentUnit
from academic_agent.processing.text_cleaner import TextCleaner


class NormalizationError(Exception):
    """Raised when document normalization fails."""


class DocumentNormalizer:
    """
    Standardizes raw page extractions and layout extractions into ordered ContentUnits.

    Preserves typography, bounding boxes, block hierarchy, and source references
    without making premature semantic decisions.
    """

    def __init__(self, text_cleaner: TextCleaner | None = None) -> None:
        self.text_cleaner = text_cleaner or TextCleaner()

    def normalize(
        self,
        records: Sequence[PageRecord | PageLayoutRecord],
    ) -> list[ContentUnit]:
        """Normalize a sequence of PageRecord or PageLayoutRecord instances."""
        if not records:
            return []

        units: list[ContentUnit] = []
        for record in records:
            if isinstance(record, PageLayoutRecord):
                units.extend(self.normalize_layout_record(record))
            elif isinstance(record, PageRecord):
                units.extend(self.normalize_page_record(record))
            else:
                raise NormalizationError(
                    f"Unsupported record type: {type(record).__name__}. "
                    "Expected PageRecord or PageLayoutRecord."
                )

        return units

    def normalize_page_record(self, record: PageRecord) -> list[ContentUnit]:
        """
        Normalize a raw text PageRecord into ContentUnits by paragraph splitting.
        """
        if not record.text or not record.text.strip():
            return []

        cleaned_text = self.text_cleaner.clean(record.text)
        if not cleaned_text:
            return []

        raw_paragraphs = [p.strip() for p in cleaned_text.split("\n\n") if p.strip()]
        units: list[ContentUnit] = []

        for idx, paragraph in enumerate(raw_paragraphs):
            unit_id = f"{record.document_id}_p{record.page_number}_u{idx + 1:04d}"
            units.append(
                ContentUnit(
                    unit_id=unit_id,
                    document_id=record.document_id,
                    source_file=record.source_file,
                    page_number=record.page_number,
                    content_type=ContentType.UNKNOWN,
                    text=paragraph,
                    block_number=idx,
                    metadata={"source": "page_record", "index": idx},
                )
            )

        return units

    def normalize_layout_record(self, record: PageLayoutRecord) -> list[ContentUnit]:
        """
        Normalize a PageLayoutRecord into ContentUnits, preserving block geometry,
        typography metadata, and table structures.
        """
        items: list[tuple[float, float, ContentUnit]] = []

        # 1. Process positioned text blocks
        for block in record.text_blocks:
            unit = self._convert_text_block(block, record)
            if unit:
                top = block.bbox[1]
                left = block.bbox[0]
                items.append((top, left, unit))

        # 2. Process native tables
        for table in record.tables:
            unit = self._convert_table(table, record)
            if unit:
                top = table.bbox[1]
                left = table.bbox[0]
                items.append((top, left, unit))

        # Sort top-to-bottom, left-to-right to maintain natural reading order
        items.sort(key=lambda item: (item[0], item[1]))

        return [item[2] for item in items]

    def _convert_text_block(
        self,
        block: TextBlockRecord,
        record: PageLayoutRecord,
    ) -> ContentUnit | None:
        """Convert a TextBlockRecord into a ContentUnit preserving typography."""
        raw_text = block.text.strip()
        if not raw_text:
            return None

        cleaned_text = self.text_cleaner.clean(raw_text)
        if not cleaned_text:
            return None

        # Extract font sizes and font families from spans
        font_sizes: list[float] = []
        fonts: set[str] = set()
        for line in block.lines:
            for span in line.spans:
                if span.text.strip():
                    font_sizes.append(span.size)
                    if span.font:
                        fonts.add(span.font)

        primary_font_size = max(font_sizes) if font_sizes else None

        unit_id = f"{record.document_id}_p{record.page_number}_b{block.block_number:04d}"
        metadata: dict[str, Any] = {
            "source": "layout_record",
            "bbox": block.bbox,
            "block_number": block.block_number,
            "font_size": primary_font_size,
            "fonts": sorted(fonts),
            "line_count": len(block.lines),
        }

        return ContentUnit(
            unit_id=unit_id,
            document_id=record.document_id,
            source_file=record.source_file,
            page_number=record.page_number,
            content_type=ContentType.UNKNOWN,
            text=cleaned_text,
            block_number=block.block_number,
            metadata=metadata,
        )

    def _convert_table(
        self,
        table: TableRecord,
        record: PageLayoutRecord,
    ) -> ContentUnit | None:
        """Convert a TableRecord into a Markdown-formatted table ContentUnit."""
        if not table.cells:
            return None

        markdown_text = self.format_table_as_markdown(table.cells)
        if not markdown_text.strip():
            return None

        unit_id = f"{record.document_id}_p{record.page_number}_t{table.table_number:04d}"
        metadata: dict[str, Any] = {
            "source": "layout_record",
            "bbox": table.bbox,
            "table_number": table.table_number,
            "row_count": table.row_count,
            "column_count": table.column_count,
            "raw_cells": table.cells,
        }

        return ContentUnit(
            unit_id=unit_id,
            document_id=record.document_id,
            source_file=record.source_file,
            page_number=record.page_number,
            content_type=ContentType.TABLE,
            text=markdown_text,
            block_number=None,
            metadata=metadata,
        )

    @staticmethod
    def format_table_as_markdown(cells: list[list[str | None]]) -> str:
        """Render a 2D matrix of table cells into GitHub-Flavored Markdown."""
        if not cells or not any(cells):
            return ""

        sanitized_rows: list[list[str]] = []
        for row in cells:
            sanitized_rows.append(
                [str(cell).strip().replace("\n", " ") if cell is not None else "" for cell in row]
            )

        # Check if all rows are completely empty
        if not any(any(col for col in row) for row in sanitized_rows):
            return ""

        header = sanitized_rows[0]
        divider = ["---"] * len(header)
        rows = sanitized_rows[1:]

        lines: list[str] = [
            f"| {' | '.join(header)} |",
            f"| {' | '.join(divider)} |",
        ]

        for r in rows:
            # Pad row if columns are missing
            padded_r = r + [""] * max(0, len(header) - len(r))
            lines.append(f"| {' | '.join(padded_r[:len(header)])} |")

        return "\n".join(lines)
