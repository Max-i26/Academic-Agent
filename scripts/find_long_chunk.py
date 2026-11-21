from pathlib import Path
from transformers import AutoTokenizer
from academic_agent.ingestion.layout_extractor import LayoutExtractor
from academic_agent.processing.chunking import ChunkingConfig, ChunkingEngine, EstimateTokenizer

extractor = LayoutExtractor()
tokenizer = EstimateTokenizer(factor=1.33)
config = ChunkingConfig(target_tokens=180, max_tokens=220, min_tokens=40, overlap_tokens=35, max_overlap_tokens=45, target_words=135, max_words=165, min_words=30, overlap_words=25, max_overlap_words=35)
engine = ChunkingEngine(config=config, tokenizer=tokenizer)
hf_tok = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

all_chunks = []
for p in sorted(Path("data/raw").glob("*.pdf")):
    recs = extractor.extract(p)
    chunks = engine.chunk(recs)
    all_chunks.extend(chunks)

max_tok = 0
worst_chunk = None
for c in all_chunks:
    toks = len(hf_tok.encode(c.text, add_special_tokens=True))
    if toks > max_tok:
        max_tok = toks
        worst_chunk = c

print(f"Max token count: {max_tok}", flush=True)
if worst_chunk:
    print(f"Lecture: {worst_chunk.source_file}", flush=True)
    print(f"Chunk ID: {worst_chunk.chunk_id}", flush=True)
    print(f"Words: {worst_chunk.word_count} | Estimated tokens: {int(worst_chunk.word_count * 1.33)}", flush=True)
    print(f"Pages: {worst_chunk.source_pages} | Section: {worst_chunk.section_title}", flush=True)
    print("--- FULL TEXT ---", flush=True)
    print(worst_chunk.text, flush=True)
