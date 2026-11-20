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

# Experimental development token budget proposed by the user
config = ChunkingConfig(
    target_tokens=180,
    max_tokens=220,
    min_tokens=40,
    overlap_tokens=35,
    max_overlap_tokens=45,
    target_words=135,
    max_words=165,
    min_words=30,
    overlap_words=25,
    max_overlap_words=35,
)
engine = ChunkingEngine(config=config, tokenizer=tokenizer)

total_docs = len(pdf_files)
total_pages = 0
total_chunks = 0
all_chunks = []
global_content_types = Counter()

print("=" * 80)
print("ACADEMIC-AGENT: AUTHORITATIVE TOKEN-BUDGET EVALUATION (TARGET 180 / MAX 220 TOKENS)")
print("=" * 80)
print(f"Token Limits: Target={config.target_tokens} | Max={config.max_tokens} | Min={config.min_tokens} | Max Overlap={config.max_overlap_tokens}\n")

for pdf_path in pdf_files:
    layout_records = layout_extractor.extract(pdf_path)
    total_pages += len(layout_records)

    chunks = engine.chunk(layout_records)
    total_chunks += len(chunks)
    all_chunks.extend(chunks)

    report = engine.get_validation_report(chunks)
    type_counts = Counter(c.content_type.value for c in chunks)
    global_content_types.update(type_counts)

    tokens = [c.estimated_token_count for c in chunks]
    c_gt_220 = sum(1 for t in tokens if t > 220)
    c_gt_256 = sum(1 for t in tokens if t > 256)
    max_tok = max(tokens) if tokens else 0
    avg_tok = sum(tokens) / len(tokens) if tokens else 0

    print(f"DOCUMENT: {pdf_path.name}")
    print(f"  Pages: {len(layout_records)} | Total Chunks: {report.total_chunks}")
    print(f"  Tokens: avg={avg_tok:.1f}, min={min(tokens) if tokens else 0}, max={max_tok}")
    print(f"  > 220 tokens: {c_gt_220} | > 256 tokens: {c_gt_256}")
    print(f"  Overlap Words: {report.total_overlap_words} (ratio={report.overlap_ratio:.2%})")
    print(f"  Forced Splits: {report.forced_splits_count}")
    print("-" * 80)

print("\n" + "=" * 80)
print("OVERALL TOKEN-BUDGET REGRESSION SUMMARY")
print("=" * 80)
all_tokens = [c.estimated_token_count for c in all_chunks]
print(f"Total Documents:           {total_docs}")
print(f"Total Pages:               {total_pages}")
print(f"Total Chunks:              {total_chunks}")
print(f"Average Tokens/Chunk:      {sum(all_tokens)/len(all_tokens):.1f}")
print(f"Min Tokens in Chunk:       {min(all_tokens)}")
print(f"Max Tokens in Chunk:       {max(all_tokens)}")
print(f"Chunks Exceeding 220 Toks: {sum(1 for t in all_tokens if t > 220)}")
print(f"Chunks Exceeding 256 Toks: {sum(1 for t in all_tokens if t > 256)}")
print(f"Total Forced Splits:       {sum(1 for c in all_chunks if c.metadata.get('forced_split'))}")
print(f"Content Types Breakdown:   {dict(global_content_types)}")
