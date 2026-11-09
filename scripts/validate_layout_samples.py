from pathlib import Path

from academic_agent.ingestion.layout_extractor import LayoutExtractor

DATA = Path("data/raw")
extractor = LayoutExtractor()

samples = [
    (
        "4. DS5104 Lecture4 Simple_Exponential_Smoothing.pdf",
        18,
        ["MAE", "mean(|eₜ|)", "MSE", "mean(eₜ²)", "RMSE", "√MSE"],
    ),
    (
        "3. DS5104 Lecture3 Smoothing_Methods_I.pdf",
        11,
        ["Good uses", "Weak uses", "Sales trend", "Sudden accidents"],
    ),
    (
        "5. DS5104 Lecture5 Trend_Seasonal_Decomposition.pdf",
        6,
        ["T", "Direction", "S", "Calendar repetition",
         "C", "Business/economic wave", "I", "Unplanned disturbance"],
    ),
]

for filename, page_number, expected in samples:
    pdf_path = DATA / filename
    record = extractor.extract_page(pdf_path, page_number)

    extracted_text = "\n".join(
        block.text for block in record.text_blocks
    )

    print("\n" + "=" * 75)
    print(f"DOCUMENT: {filename}")
    print(f"PAGE: {record.page_number}")
    print(f"TEXT BLOCKS: {len(record.text_blocks)}")
    print(f"TABLES: {len(record.tables)}")

    for phrase in expected:
        found = phrase in extracted_text
        print(f"[{'PASS' if found else 'FAIL'}] {phrase}")

    print("\nEXTRACTED BLOCKS:")
    for block in record.text_blocks:
        print(f"  Block {block.block_number}: {block.text!r}")

    assert all(phrase in extracted_text for phrase in expected), (
        f"Content validation failed for {filename}, page {page_number}"
    )

print("\nALL THREE LAYOUT SAMPLES VALIDATED SUCCESSFULLY.")
