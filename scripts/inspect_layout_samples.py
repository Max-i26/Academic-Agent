from pathlib import Path

from academic_agent.ingestion.layout_extractor import LayoutExtractor

DATA_DIR = Path("data/raw")
extractor = LayoutExtractor()

samples = [
    ("4. DS5104 Lecture4 Simple_Exponential_Smoothing.pdf", 18),
    ("3. DS5104 Lecture3 Smoothing_Methods_I.pdf", 11),
    ("5. DS5104 Lecture5 Trend_Seasonal_Decomposition.pdf", 6),
]

for filename, page_number in samples:
    pdf_path = DATA_DIR / filename

    record = extractor.extract_page(pdf_path, page_number)

    print("\n" + "=" * 90)
    print(f"DOCUMENT: {filename}")
    print(f"PAGE: {page_number}")
    print("=" * 90)

    for block in record.text_blocks:
        print(f"\nBLOCK {block.block_number}")
        print(f"BBOX: {block.bbox}")
        print(f"TEXT: {block.text!r}")

        for line in block.lines:
            print(f"  LINE: {line.text!r}")

            for span in line.spans:
                print(
                    f"    SPAN: {span.text!r} | "
                    f"FONT: {span.font} | "
                    f"SIZE: {span.size:.2f} | "
                    f"BBOX: {span.bbox}"
                )
