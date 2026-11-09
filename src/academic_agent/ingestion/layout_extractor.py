from pathlib import Path

import pymupdf
from pydantic import BaseModel, Field


class LayoutExtractionError(Exception):
    """Raised when PDF layout extraction fails."""


class TextSpanRecord(BaseModel):
    """A text span with its original typography and position."""

    text: str
    font: str
    size: float = Field(ge=0)
    bbox: tuple[float, float, float, float]
    origin: tuple[float, float]


class TextLineRecord(BaseModel):
    """An ordered collection of text spans belonging to one line."""

    text: str
    bbox: tuple[float, float, float, float]
    spans: list[TextSpanRecord] = Field(default_factory=list)


class TextBlockRecord(BaseModel):
    """A positioned text block containing ordered lines and spans."""

    block_number: int = Field(ge=0)
    text: str
    bbox: tuple[float, float, float, float]
    lines: list[TextLineRecord] = Field(default_factory=list)


class TableRecord(BaseModel):
    """A native table extracted from a PDF page."""

    table_number: int = Field(ge=0)
    bbox: tuple[float, float, float, float]
    row_count: int = Field(ge=0)
    column_count: int = Field(ge=0)
    cells: list[list[str | None]] = Field(default_factory=list)


class PageLayoutRecord(BaseModel):
    """Structured layout information for one PDF page."""

    document_id: str
    source_file: str
    page_number: int = Field(ge=1)
    page_count: int = Field(ge=1)
    page_width: float = Field(gt=0)
    page_height: float = Field(gt=0)
    metadata: dict[str, str | None] = Field(default_factory=dict)
    text_blocks: list[TextBlockRecord] = Field(default_factory=list)
    tables: list[TableRecord] = Field(default_factory=list)


class LayoutExtractor:
    """Extract positioned text and native tables from PDF documents."""

    def extract(self, pdf_path: str | Path) -> list[PageLayoutRecord]:
        """Extract layout records from every page in a PDF."""

        path = self._validate_path(pdf_path)

        try:
            with pymupdf.open(path) as document:
                self._validate_document(document, path)

                metadata = {
                    str(key): str(value) if value is not None else None
                    for key, value in (document.metadata or {}).items()
                }

                page_count = len(document)
                records = []

                for page_index, page in enumerate(document):
                    records.append(
                        self._extract_page(
                            page=page,
                            path=path,
                            page_number=page_index + 1,
                            page_count=page_count,
                            metadata=metadata,
                        )
                    )

                return records

        except LayoutExtractionError:
            raise
        except Exception as exc:
            raise LayoutExtractionError(
                f"Failed to extract layout from {path.name}: {exc}"
            ) from exc

    def extract_page(
        self,
        pdf_path: str | Path,
        page_number: int,
    ) -> PageLayoutRecord:
        """Extract layout information from one 1-based PDF page number."""

        path = self._validate_path(pdf_path)

        if not isinstance(page_number, int) or isinstance(page_number, bool):
            raise TypeError("page_number must be an integer")

        if page_number < 1:
            raise ValueError("page_number must be at least 1")

        try:
            with pymupdf.open(path) as document:
                self._validate_document(document, path)

                if page_number > len(document):
                    raise ValueError(
                        f"Page {page_number} does not exist in {path.name}; "
                        f"document has {len(document)} pages"
                    )

                metadata = {
                    str(key): str(value) if value is not None else None
                    for key, value in (document.metadata or {}).items()
                }

                return self._extract_page(
                    page=document[page_number - 1],
                    path=path,
                    page_number=page_number,
                    page_count=len(document),
                    metadata=metadata,
                )

        except (LayoutExtractionError, TypeError, ValueError):
            raise
        except Exception as exc:
            raise LayoutExtractionError(
                f"Failed to extract page {page_number} from "
                f"{path.name}: {exc}"
            ) from exc

    @staticmethod
    def _validate_path(pdf_path: str | Path) -> Path:
        """Validate the supplied PDF path."""

        path = Path(pdf_path)

        if not path.exists():
            raise FileNotFoundError(f"PDF not found: {path}")

        if not path.is_file():
            raise ValueError(f"Expected a file, got: {path}")

        if path.suffix.lower() != ".pdf":
            raise ValueError(f"Expected a PDF file: {path}")

        return path

    @staticmethod
    def _validate_document(document, path: Path) -> None:
        """Validate an opened PDF document."""

        if document.is_encrypted:
            raise LayoutExtractionError(
                f"Encrypted PDF is not supported: {path.name}"
            )

        if len(document) == 0:
            raise LayoutExtractionError(
                f"PDF contains no pages: {path.name}"
            )

    def _extract_page(
        self,
        page,
        path: Path,
        page_number: int,
        page_count: int,
        metadata: dict[str, str | None],
    ) -> PageLayoutRecord:
        """Extract text blocks and native tables from a single page."""

        page_dict = page.get_text("dict")
        text_blocks = []

        for block_number, block in enumerate(page_dict["blocks"]):
            # Ignore image and non-text blocks.
            if block.get("type") != 0:
                continue

            lines = []

            for raw_line in block.get("lines", []):
                spans = []

                for raw_span in raw_line.get("spans", []):
                    span_text = raw_span.get("text", "")

                    if not span_text:
                        continue

                    spans.append(
                        TextSpanRecord(
                            text=span_text,
                            font=raw_span.get("font", ""),
                            size=raw_span.get("size", 0),
                            bbox=tuple(raw_span["bbox"]),
                            origin=tuple(raw_span["origin"]),
                        )
                    )

                if not spans:
                    continue

                lines.append(
                    TextLineRecord(
                        text="".join(span.text for span in spans),
                        bbox=tuple(raw_line["bbox"]),
                        spans=spans,
                    )
                )

            if not lines:
                continue

            text_blocks.append(
                TextBlockRecord(
                    block_number=block_number,
                    text="\n".join(line.text for line in lines),
                    bbox=tuple(block["bbox"]),
                    lines=lines,
                )
            )

        tables = []

        # Keep native table extraction independent of text-block extraction.
        table_finder = page.find_tables()

        for table_number, table in enumerate(table_finder.tables):
            extracted_rows = table.extract()

            cells = [
                [
                    cell.strip() if isinstance(cell, str) else None
                    for cell in row
                ]
                for row in extracted_rows
            ]

            tables.append(
                TableRecord(
                    table_number=table_number,
                    bbox=tuple(table.bbox),
                    row_count=len(cells),
                    column_count=max(
                        (len(row) for row in cells),
                        default=0,
                    ),
                    cells=cells,
                )
            )

        return PageLayoutRecord(
            document_id=path.stem,
            source_file=path.name,
            page_number=page_number,
            page_count=page_count,
            page_width=page.rect.width,
            page_height=page.rect.height,
            metadata=metadata,
            text_blocks=text_blocks,
            tables=tables,
        )
