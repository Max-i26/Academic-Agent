from types import SimpleNamespace

import pymupdf
import pytest

from academic_agent.ingestion.layout_extractor import (
    LayoutExtractionError,
    LayoutExtractor,
)


def create_pdf(path, pages=1):
    """Create a small PDF fixture containing selectable text."""

    document = pymupdf.open()

    for index in range(pages):
        page = document.new_page()
        page.insert_text(
            (72, 72),
            f"Page {index + 1} heading",
        )
        page.insert_text(
            (72, 110),
            "A second line containing academic text.",
        )

    document.save(path)
    document.close()

    return path

def create_fake_document(page=None, encrypted=False, page_count=1):
    """Create a lightweight fake PDF document for controlled tests."""

    class FakeDocument:
        is_encrypted = encrypted
        metadata = {"title": "Test document"}

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def __len__(self):
            return page_count

        def __getitem__(self, index):
            assert index == 0
            return page

        def __iter__(self):
            return iter([page])

    return FakeDocument()

def test_extracts_all_pages_in_order(tmp_path):
    pdf_path = create_pdf(tmp_path / "multiple.pdf", pages=3)

    records = LayoutExtractor().extract(pdf_path)

    assert len(records) == 3
    assert [record.page_number for record in records] == [1, 2, 3]
    assert all(record.page_count == 3 for record in records)
    assert "Page 1 heading" in records[0].text_blocks[0].text
    assert "Page 3 heading" in records[2].text_blocks[0].text


def test_preserves_document_provenance(tmp_path):
    pdf_path = create_pdf(tmp_path / "lecture.pdf")

    record = LayoutExtractor().extract_page(pdf_path, 1)

    assert record.document_id == "lecture"
    assert record.source_file == "lecture.pdf"
    assert record.page_number == 1
    assert record.page_count == 1


def test_preserves_page_dimensions(tmp_path):
    pdf_path = create_pdf(tmp_path / "dimensions.pdf")

    record = LayoutExtractor().extract_page(pdf_path, 1)

    assert record.page_width > 0
    assert record.page_height > 0


def test_preserves_text_block_line_and_span_structure(tmp_path):
    pdf_path = create_pdf(tmp_path / "structure.pdf")

    record = LayoutExtractor().extract_page(pdf_path, 1)

    assert record.text_blocks

    block = record.text_blocks[0]

    assert block.block_number >= 0
    assert "Page 1 heading" in block.text
    assert block.lines
    assert block.lines[0].spans

    span = block.lines[0].spans[0]

    assert span.text
    assert span.font
    assert span.size > 0
    assert len(span.bbox) == 4
    assert len(span.origin) == 2


def test_preserves_text_order_within_spans(tmp_path):
    pdf_path = create_pdf(tmp_path / "order.pdf")

    record = LayoutExtractor().extract_page(pdf_path, 1)

    block = record.text_blocks[0]
    reconstructed = "\n".join(
        line.text for line in block.lines
    )

    assert reconstructed == block.text
    assert "Page 1 heading" in reconstructed


def test_preserves_metadata(tmp_path):
    pdf_path = tmp_path / "metadata.pdf"

    document = pymupdf.open()
    document.new_page()
    document.set_metadata({"title": "Academic Test Document"})
    document.save(pdf_path)
    document.close()

    record = LayoutExtractor().extract_page(pdf_path, 1)

    assert record.metadata["title"] == "Academic Test Document"


def test_extracts_native_table_cells(tmp_path, monkeypatch):
    pdf_path = create_pdf(tmp_path / "table.pdf")

    class FakeTable:
        bbox = (10.0, 20.0, 200.0, 100.0)

        def extract(self):
            return [
                ["Quarter", "Sales"],
                ["Q1", "180"],
                ["Q2", "220"],
            ]

    class FakePage:
        rect = SimpleNamespace(width=900.0, height=550.0)

        def get_text(self, mode):
            assert mode == "dict"
            return {"blocks": []}

        def find_tables(self):
            return SimpleNamespace(tables=[FakeTable()])

    class FakeDocument:
        is_encrypted = False
        metadata = {"title": "Table fixture"}

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def __len__(self):
            return 1

        def __getitem__(self, index):
            assert index == 0
            return FakePage()

        def __iter__(self):
            return iter([FakePage()])

    monkeypatch.setattr(
        "academic_agent.ingestion.layout_extractor.pymupdf.open",
        lambda path: FakeDocument(),
    )

    record = LayoutExtractor().extract_page(pdf_path, 1)

    assert len(record.tables) == 1

    table = record.tables[0]

    assert table.row_count == 3
    assert table.column_count == 2
    assert table.cells == [
        ["Quarter", "Sales"],
        ["Q1", "180"],
        ["Q2", "220"],
    ]
    assert table.bbox == (10.0, 20.0, 200.0, 100.0)


def test_missing_file_raises_error(tmp_path):
    missing_path = tmp_path / "missing.pdf"

    with pytest.raises(FileNotFoundError):
        LayoutExtractor().extract(missing_path)


def test_directory_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="Expected a file"):
        LayoutExtractor().extract(tmp_path)


def test_non_pdf_file_is_rejected(tmp_path):
    file_path = tmp_path / "notes.txt"
    file_path.write_text("Not a PDF")

    with pytest.raises(ValueError, match="Expected a PDF"):
        LayoutExtractor().extract(file_path)


def test_invalid_pdf_is_wrapped(tmp_path):
    pdf_path = tmp_path / "invalid.pdf"
    pdf_path.write_text("This is not a valid PDF")

    with pytest.raises(LayoutExtractionError):
        LayoutExtractor().extract(pdf_path)


def test_extract_page_rejects_invalid_page_numbers(tmp_path):
    pdf_path = create_pdf(tmp_path / "page.pdf")

    with pytest.raises(ValueError, match="at least 1"):
        LayoutExtractor().extract_page(pdf_path, 0)

    with pytest.raises(ValueError, match="does not exist"):
        LayoutExtractor().extract_page(pdf_path, 2)


def test_extract_page_rejects_non_integer_page_number(tmp_path):
    pdf_path = create_pdf(tmp_path / "page_type.pdf")

    with pytest.raises(TypeError, match="must be an integer"):
        LayoutExtractor().extract_page(pdf_path, 1.5)
    with pytest.raises(TypeError, match="must be an integer"):
        LayoutExtractor().extract_page(pdf_path, True)
@pytest.mark.parametrize(
    ("encrypted", "page_count", "expected_message"),
    [
        (True, 1, "Encrypted PDF"),
        (False, 0, "contains no pages"),
    ],
)
def test_extract_rejects_unsupported_documents(
    tmp_path,
    monkeypatch,
    encrypted,
    page_count,
    expected_message,
):
    pdf_path = create_pdf(tmp_path / "unsupported.pdf")

    fake_document = create_fake_document(
        encrypted=encrypted,
        page_count=page_count,
    )

    monkeypatch.setattr(
        "academic_agent.ingestion.layout_extractor.pymupdf.open",
        lambda path: fake_document,
    )

    with pytest.raises(LayoutExtractionError, match=expected_message):
        LayoutExtractor().extract(pdf_path)

def test_extract_page_wraps_unexpected_errors(tmp_path, monkeypatch):
    pdf_path = create_pdf(tmp_path / "unexpected.pdf")

    class FaultyPage:
        def get_text(self, mode):
            raise RuntimeError("Simulated extraction failure")

    fake_document = create_fake_document(page=FaultyPage())

    monkeypatch.setattr(
        "academic_agent.ingestion.layout_extractor.pymupdf.open",
        lambda path: fake_document,
    )

    with pytest.raises(
        LayoutExtractionError,
        match="Simulated extraction failure",
    ):
        LayoutExtractor().extract_page(pdf_path, 1)

def test_ignores_non_text_blocks(tmp_path, monkeypatch):
    pdf_path = create_pdf(tmp_path / "non_text.pdf")

    class FakePage:
        rect = SimpleNamespace(width=600.0, height=800.0)

        def get_text(self, mode):
            assert mode == "dict"

            return {
                "blocks": [
                    {
                        "type": 1,
                        "bbox": (10, 10, 100, 100),
                    }
                ]
            }

        def find_tables(self):
            return SimpleNamespace(tables=[])

    fake_document = create_fake_document(page=FakePage())

    monkeypatch.setattr(
        "academic_agent.ingestion.layout_extractor.pymupdf.open",
        lambda path: fake_document,
    )

    record = LayoutExtractor().extract_page(pdf_path, 1)

    assert record.text_blocks == []
    assert record.tables == []

def test_ignores_empty_spans_and_blocks(tmp_path, monkeypatch):
    pdf_path = create_pdf(tmp_path / "empty_content.pdf")

    class FakePage:
        rect = SimpleNamespace(width=600.0, height=800.0)

        def get_text(self, mode):
            assert mode == "dict"

            return {
                "blocks": [
                    {
                        "type": 0,
                        "bbox": (10, 10, 100, 100),
                        "lines": [
                            {
                                "bbox": (10, 10, 100, 30),
                                "spans": [
                                    {
                                        "text": "",
                                        "font": "",
                                        "size": 0,
                                        "bbox": (10, 10, 10, 10),
                                        "origin": (10, 10),
                                    }
                                ],
                            }
                        ],
                    },
                    {
                        "type": 0,
                        "bbox": (10, 40, 100, 60),
                        "lines": [],
                    },
                ]
            }

        def find_tables(self):
            return SimpleNamespace(tables=[])

    fake_document = create_fake_document(page=FakePage())

    monkeypatch.setattr(
        "academic_agent.ingestion.layout_extractor.pymupdf.open",
        lambda path: fake_document,
    )

    record = LayoutExtractor().extract_page(pdf_path, 1)

    assert record.text_blocks == []
