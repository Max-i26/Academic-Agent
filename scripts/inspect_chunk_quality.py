from pathlib import Path

from academic_agent.ingestion.layout_extractor import LayoutExtractor
from academic_agent.processing.chunking import (
    ChunkingConfig,
    ChunkingEngine,
    EstimateTokenizer,
)

DATA_DIR = Path("data/raw")
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

# 1. Process all lectures and gather token stats & tiny chunks
pdf_files = sorted(DATA_DIR.glob("*.pdf"))

print("=" * 80)
print("1. TOKEN-BUDGET AUDIT ACROSS ALL LECTURES (Estimated Tokens @ 1.33x words)")
print("=" * 80)

lecture_chunks = {}

for pdf_path in pdf_files:
    layout_records = layout_extractor.extract(pdf_path)
    chunks = engine.chunk(layout_records)
    lecture_chunks[pdf_path.stem] = chunks

    total_c = len(chunks)
    tokens = [c.estimated_token_count for c in chunks]
    words = [c.word_count for c in chunks]
    c_gt_256 = sum(1 for t in tokens if t > 256)
    c_gt_512 = sum(1 for t in tokens if t > 512)
    max_tok = max(tokens) if tokens else 0
    avg_tok = sum(tokens) / len(tokens) if tokens else 0

    print(f"\n{pdf_path.name}")
    print(f"  Chunks: {total_c} | Avg Tokens: {avg_tok:.1f} | Max Tokens: {max_tok}")
    print(f"  > 256 tokens: {c_gt_256} ({c_gt_256/total_c:.1%}) | > 512 tokens: {c_gt_512} ({c_gt_512/total_c:.1%})")

# 2. Detailed inspection of Lecture 5
print("\n" + "=" * 80)
print("2. LECTURE 5 DETAILED CHUNK AUDIT (Trend & Seasonal Decomposition)")
print("=" * 80)
lec5_key = [k for k in lecture_chunks if "Lecture5" in k][0]
lec5_chunks = lecture_chunks[lec5_key]

print(f"Lecture 5 has {len(lec5_chunks)} total chunks generated from 33 pages.")
for c in lec5_chunks:
    print(f"\n--- Chunk {c.chunk_index} [ID: {c.chunk_id}] ---")
    print(f"Section Title: {c.section_title}")
    print(f"Pages: {c.source_pages} (count: {len(c.source_pages)})")
    print(f"Words: {c.word_count} | Tokens: {c.estimated_token_count} | Overlap Words: {c.overlap_word_count}")
    print(f"Unit IDs ({len(c.source_unit_ids)} units): {c.source_unit_ids[:4]} ... {c.source_unit_ids[-2:] if len(c.source_unit_ids) > 4 else ''}")
    print(f"Content Type: {c.content_type.value}")
    if c.metadata.get("has_overlap"):
        print(f"Overlap Text Preview: {c.metadata.get('overlap_text')[:120]}...")
    print(f"Text Preview:\n{c.text[:280]}...")
    print("------------------------------------------------------------------------")

# 3. Detailed inspection of Lecture 4 (tiny chunks analysis)
print("\n" + "=" * 80)
print("3. LECTURE 4 CHUNK AUDIT & TINY CHUNKS ANALYSIS")
print("=" * 80)
lec4_key = [k for k in lecture_chunks if "Lecture4" in k][0]
lec4_chunks = lecture_chunks[lec4_key]

tiny_chunks = [c for c in lec4_chunks if c.word_count <= 20]
print(f"Lecture 4 total chunks: {len(lec4_chunks)} | Chunks <= 20 words: {len(tiny_chunks)} ({len(tiny_chunks)/len(lec4_chunks):.1%})")

print("\n--- Inspecting All Small Chunks (<= 20 words) in Lecture 4 ---")
for c in tiny_chunks:
    print(f"\n[Chunk {c.chunk_index} - ID: {c.chunk_id}]")
    print(f"  Section Title: {c.section_title}")
    print(f"  Pages: {c.source_pages}")
    print(f"  Words: {c.word_count} | Est Tokens: {c.estimated_token_count}")
    print(f"  Content Type: {c.content_type.value}")
    print(f"  Full Text:\n{c.text}")

# 4. Global small chunks distribution across all lectures
print("\n" + "=" * 80)
print("4. GLOBAL TINY CHUNKS (< 20 words) BREAKDOWN BY CONTENT TYPE")
print("=" * 80)
all_tiny = [c for chunks in lecture_chunks.values() for c in chunks if c.word_count < 20]
print(f"Total chunks across all 7 lectures: {sum(len(c) for c in lecture_chunks.values())}")
print(f"Total chunks < 20 words: {len(all_tiny)} ({len(all_tiny)/sum(len(c) for c in lecture_chunks.values()):.1%})")
from collections import Counter
tiny_type_counts = Counter(c.content_type.value for c in all_tiny)
print(f"Tiny Chunks by ContentType: {dict(tiny_type_counts)}")
