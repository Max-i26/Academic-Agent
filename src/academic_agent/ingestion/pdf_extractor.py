
from pathlib import Path

import pymupdf
from pydantic import BaseModel, Field


class PDFExtractionError(Exception):
    """Raised when a PDF cannot be extracted."""


class PageRecord(BaseModel):
    """Structured representation of one PDF page."""

    document_id: str
    source_file: str
    page_number: int = Field(ge=1)
    page_count: int = Field(ge=1)
    text: str
    char_count: int = Field(ge=0)
    has_text: bool
    metadata: dict[str, str | None] = Field(default_factory=dict)


class PDFExtractor:
    """Extract page-level text and metadata from PDF documents."""

    def extract(self, pdf_path: str | Path) -> list[PageRecord]:
        path = Path(pdf_path)

        if not path.exists():
            raise FileNotFoundError(f"PDF not found: {path}")

        if not path.is_file():
            raise ValueError(f"Expected a file, got: {path}")

        if path.suffix.lower() != ".pdf":
            raise ValueError(f"Expected a PDF file: {path}")

        try:
            with pymupdf.open(path) as document:
                if document.is_encrypted:
                    raise PDFExtractionError(
                        f"Encrypted PDF is not supported: {path.name}"
                    )

                metadata = {
                    str(key): str(value) if value is not None else None
                    for key, value in (document.metadata or {}).items()
                }

                page_count = len(document)

                if page_count == 0:
                    raise PDFExtractionError(
                        f"PDF contains no pages: {path.name}"
                    )

                records = []

                for page_index, page in enumerate(document):
                    text = page.get_text("text").strip()

                    records.append(
                        PageRecord(
                            document_id=path.stem,
                            source_file=path.name,
                            page_number=page_index + 1,
                            page_count=page_count,
                            text=text,
                            char_count=len(text),
                            has_text=bool(text),
                            metadata=metadata,
                        )
                    )

                return records

        except PDFExtractionError:
            raise
        except Exception as exc:
            raise PDFExtractionError(
                f"Failed to extract {path.name}: {exc}"
            ) from exc
