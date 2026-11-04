
import pymupdf
import pytest

from academic_agent.ingestion.pdf_extractor import (
    PDFExtractor,
    PDFExtractionError,
)


@pytest.fixture
def extractor():
    return PDFExtractor()


def create_pdf(path, pages):
    """Create a test PDF with the supplied page texts."""
    document = pymupdf.open()

    try:
        for text in pages:
            page = document.new_page()

            if text:
                page.insert_text((72, 72), text)

        document.save(path)

    finally:
        document.close()

    return path


# ---------------------------------------------------------
# PDF extraction error handling tests
# ---------------------------------------------------------

def test_encrypted_pdf_is_rejected(extractor, tmp_path):
    pdf = tmp_path / "encrypted.pdf"

    document = pymupdf.open()
    document.new_page()
    document.save(
        pdf,
        encryption=pymupdf.PDF_ENCRYPT_AES_256,
        owner_pw="owner-password",
        user_pw="user-password",
    )
    document.close()

    with pytest.raises(PDFExtractionError, match="Encrypted PDF"):
        extractor.extract(pdf)


def test_unexpected_exception_is_wrapped(extractor, tmp_path, monkeypatch):
    pdf = tmp_path / "unexpected.pdf"
    pdf.write_bytes(b"%PDF-test")

    def raise_unexpected_error(*args, **kwargs):
        raise RuntimeError("Simulated internal failure")

    monkeypatch.setattr(
        "academic_agent.ingestion.pdf_extractor.pymupdf.open",
        raise_unexpected_error,
    )

    with pytest.raises(PDFExtractionError, match="Simulated internal failure"):
        extractor.extract(pdf)


def test_zero_page_pdf_is_rejected(extractor, tmp_path, monkeypatch):
    pdf = tmp_path / "zero_pages.pdf"
    pdf.write_bytes(b"mock pdf")

    class EmptyPDF:
        is_encrypted = False
        metadata = {}

        def __len__(self):
            return 0

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    monkeypatch.setattr(
        "academic_agent.ingestion.pdf_extractor.pymupdf.open",
        lambda path: EmptyPDF(),
    )

    with pytest.raises(PDFExtractionError, match="no pages"):
        extractor.extract(pdf)


# ---------------------------------------------------------
# Normal PDF extraction tests
# ---------------------------------------------------------

def test_extract_single_page(extractor, tmp_path):
    pdf = create_pdf(
        tmp_path / "single.pdf",
        ["Hello Academic Agent"],
    )

    records = extractor.extract(pdf)

    assert len(records) == 1
    assert records[0].page_number == 1
    assert records[0].page_count == 1
    assert records[0].has_text is True
    assert "Hello Academic Agent" in records[0].text


def test_extract_multiple_pages_in_order(extractor, tmp_path):
    pdf = create_pdf(
        tmp_path / "multiple.pdf",
        ["First page", "Second page", "Third page"],
    )

    records = extractor.extract(pdf)

    assert len(records) == 3
    assert [record.page_number for record in records] == [1, 2, 3]
    assert [record.page_count for record in records] == [3, 3, 3]

    assert "First page" in records[0].text
    assert "Second page" in records[1].text
    assert "Third page" in records[2].text


def test_empty_page_is_detected(extractor, tmp_path):
    pdf = create_pdf(
        tmp_path / "empty_page.pdf",
        ["Page with text", ""],
    )

    records = extractor.extract(pdf)

    assert len(records) == 2
    assert records[0].has_text is True
    assert records[1].has_text is False
    assert records[1].char_count == 0
    assert records[1].text == ""


def test_metadata_is_preserved(extractor, tmp_path):
    pdf = create_pdf(
        tmp_path / "metadata.pdf",
        ["Metadata test"],
    )

    with pymupdf.open(pdf) as document:
        metadata = document.metadata
        metadata["title"] = "Academic Agent Test"
        document.set_metadata(metadata)
        document.saveIncr()

    records = extractor.extract(pdf)

    assert records[0].metadata["title"] == "Academic Agent Test"


# ---------------------------------------------------------
# Input validation tests
# ---------------------------------------------------------

def test_missing_file_raises_error(extractor, tmp_path):
    pdf = tmp_path / "missing.pdf"

    with pytest.raises(FileNotFoundError):
        extractor.extract(pdf)


def test_invalid_pdf_raises_extraction_error(extractor, tmp_path):
    pdf = tmp_path / "invalid.pdf"
    pdf.write_text("This is not a valid PDF.")

    with pytest.raises(PDFExtractionError):
        extractor.extract(pdf)


def test_non_pdf_file_is_rejected(extractor, tmp_path):
    file = tmp_path / "document.txt"
    file.write_text("Not a PDF")

    with pytest.raises(ValueError, match="Expected a PDF"):
        extractor.extract(file)


def test_directory_is_rejected(extractor, tmp_path):
    directory = tmp_path / "documents"
    directory.mkdir()

    with pytest.raises(ValueError, match="Expected a file"):
        extractor.extract(directory)


# ---------------------------------------------------------
# Page record identity test
# ---------------------------------------------------------

def test_page_record_contains_document_identity(extractor, tmp_path):
    pdf = create_pdf(
        tmp_path / "lecture_test.pdf",
        ["Identity test"],
    )

    records = extractor.extract(pdf)

    assert records[0].document_id == "lecture_test"
    assert records[0].source_file == "lecture_test.pdf"
