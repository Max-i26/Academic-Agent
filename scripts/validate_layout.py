from pathlib import Path

from academic_agent.ingestion.layout_extractor import LayoutExtractor

DATA_DIR = Path("data/raw")
extractor = LayoutExtractor()

pdf_files = sorted(DATA_DIR.glob("*.pdf"))

if not pdf_files:
    raise SystemExit("No PDF files found in data/raw")

total_pages = 0
total_tables = 0
empty_pages = 0

print("=" * 80)
print("ACADEMIC-AGENT: REAL PDF LAYOUT VALIDATION")
print("=" * 80)

for pdf_path in pdf_files:
    records = extractor.extract(pdf_path)

    document_tables = sum(len(record.tables) for record in records)
    document_empty_pages = sum(
        not record.text_blocks for record in records
    )

    total_pages += len(records)
    total_tables += document_tables
    empty_pages += document_empty_pages

    print(f"\nDOCUMENT: {pdf_path.name}")
    print(f"Pages: {len(records)}")
    print(f"Pages with no text blocks: {document_empty_pages}")
    print(f"Native tables detected: {document_tables}")

    for record in records:
        for table in record.tables:
            print(
                f"  Page {record.page_number}: "
                f"{table.row_count} rows x "
                f"{table.column_count} columns"
            )

print("\n" + "=" * 80)
print("OVERALL RESULTS")
print("=" * 80)

print(f"PDF documents: {len(pdf_files)}")
print(f"Total pages processed: {total_pages}")
print(f"Pages with no text blocks: {empty_pages}")
print(f"Native tables detected: {total_tables}")
