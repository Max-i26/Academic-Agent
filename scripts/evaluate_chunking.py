from collections import Counter
from pathlib import Path

from academic_agent.ingestion.layout_extractor import LayoutExtractor
from academic_agent.processing.chunking import (
    ChunkingConfig,
    ChunkingEngine,
    EstimateTokenizer,
)

DATA_DIR = Path("data/raw")
pdf_files = sorted(DATA_DIR.glob("*.pdf"))

if not pdf_files:
    raise SystemExit("No PDF files found in data/raw")

layout_extractor = LayoutExtractor()
tokenizer = EstimateTokenizer(factor=1.33)
config = ChunkingConfig(
    target_words=350,
    max_words=500,
    min_words=80,
    overlap_words=50,
    max_overlap_words=75,
)
engine = ChunkingEngine(config=config, tokenizer=tokenizer)

total_docs = len(pdf_files)
total_pages = 0
total_chunks = 0
all_chunks = []
global_content_types = Counter()

print("=" * 80)
print("ACADEMIC-AGENT: REAL-WORLD LECTURE CHUNKING EVALUATION")
print("=" * 80)
print(f"Target size: {config.target_words} words | Max size: {config.max_words} words | Min size: {config.min_words} words")
print(f"Overlap: {config.overlap_words} words (max {config.max_overlap_words} words)\n")

for pdf_path in pdf_files:
    layout_records = layout_extractor.extract(pdf_path)
    total_pages += len(layout_records)

    chunks = engine.chunk(layout_records)
    total_chunks += len(chunks)
    all_chunks.extend(chunks)

    report = engine.get_validation_report(chunks)
    type_counts = Counter(c.content_type.value for c in chunks)
    global_content_types.update(type_counts)

    print(f"DOCUMENT: {pdf_path.name}")
    print(f"  Pages: {len(layout_records)}")
    print(f"  Total Chunks: {report.total_chunks}")
    print(f"  Words: total={report.total_words}, avg={report.avg_words_per_chunk:.1f}, min={report.min_words}, max={report.max_words}")
    print(f"  Overlap Words: {report.total_overlap_words} (ratio={report.overlap_ratio:.2%})")
    print(f"  Forced Splits: {report.forced_splits_count}")
    print(f"  Content Types: {dict(type_counts)}")
    print("-" * 80)

print("\n" + "=" * 80)
print("OVERALL REAL-WORLD EVALUATION SUMMARY")
print("=" * 80)
print(f"Total Documents Processed: {total_docs}")
print(f"Total Pages Processed:     {total_pages}")
print(f"Total Chunks Generated:    {total_chunks}")

if all_chunks:
    overall_report = engine.get_validation_report(all_chunks)
    print(f"Overall Words:             {overall_report.total_words}")
    print(f"Average Words/Chunk:       {overall_report.avg_words_per_chunk:.1f}")
    print(f"Min Words in a Chunk:      {overall_report.min_words}")
    print(f"Max Words in a Chunk:      {overall_report.max_words}")
    print(f"Total Overlap Words:       {overall_report.total_overlap_words}")
    print(f"Overall Overlap Ratio:     {overall_report.overlap_ratio:.2%}")
    print(f"Total Forced Splits:       {overall_report.forced_splits_count}")
    print(f"Overall Content Types:     {dict(global_content_types)}")

    # Sample Inspection
    print("\n" + "=" * 80)
    print("SAMPLE CHUNK INSPECTION (ACADEMIC PROVENANCE & STRUCTURE)")
    print("=" * 80)
    sample_indices = [0, len(all_chunks) // 2, len(all_chunks) - 1]
    for idx in sample_indices:
        c = all_chunks[idx]
        print(f"\n--- Chunk [{c.chunk_id}] (Index {c.chunk_index} of {c.document_id}) ---")
        print(f"Source Document: {c.source_file}")
        print(f"Section Title:   {c.section_title or 'N/A'}")
        print(f"Contributing Pages: {c.source_pages}")
        print(f"Source Unit IDs:    {c.source_unit_ids[:3]}{'...' if len(c.source_unit_ids) > 3 else ''}")
        print(f"Content Type:    {c.content_type.value}")
        print(f"Word Count:      {c.word_count} | Est. Tokens: {c.estimated_token_count} | Overlap Words: {c.overlap_word_count}")
        print(f"Text Preview:\n{c.text[:220]}...")
