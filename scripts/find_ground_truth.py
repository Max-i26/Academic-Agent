import json
from pathlib import Path
from academic_agent.ingestion.layout_extractor import LayoutExtractor
from academic_agent.processing.chunking import ChunkingConfig, ChunkingEngine, EstimateTokenizer

extractor = LayoutExtractor()
tokenizer = EstimateTokenizer(factor=1.33)
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

DATA_DIR = Path("data/raw")
pdf_files = sorted(DATA_DIR.glob("*.pdf"))

chunks_by_id = {}
all_chunks = []
for p in pdf_files:
    recs = extractor.extract(p)
    doc_chunks = engine.chunk(recs)
    all_chunks.extend(doc_chunks)
    for c in doc_chunks:
        chunks_by_id[c.chunk_id] = c

print(f"Loaded {len(all_chunks)} chunks across {len(pdf_files)} lectures.")

# Search keywords for each question
targets = {
    "q1": ("mean(|e", "MAE"),
    "q2": ("weighted compromise", "Sₜ = αYₜ"),
    "q3": ("WHEN Is Multiplicative", "Seasonal pattern grows"),
    "q4": ("CMA is needed because", "between periods"),
    "q5": ("confidence bounds", "significant"),
    "q6": ("Stationarity", "constant mean"),
    "q7": ("Moving Average", "smooth"),
    "q8": ("trend", "drops slowly", "ACF"),
}

ground_truth = []
with open("benchmarks/questions.json") as f:
    questions = json.load(f)

for q in questions:
    qid = q["id"]
    source_pdf = q["source_lecture"]
    expected = q["expected_concepts"]
    
    matches = []
    for c in all_chunks:
        if c.source_file == source_pdf:
            score = sum(1 for exp in expected if any(w.lower() in c.text.lower() for w in exp.split()))
            # Special concept check
            if qid == "q1" and "MAE" in c.text and "mean" in c.text.lower():
                matches.append((c, 10))
            elif qid == "q2" and "α" in c.text and ("compromise" in c.text.lower() or "weight" in c.text.lower()):
                matches.append((c, 10))
            elif qid == "q3" and "multiplicative" in c.text.lower() and ("grows" in c.text.lower() or "variation" in c.text.lower() or "suitable" in c.text.lower()):
                matches.append((c, 10))
            elif qid == "q4" and "cma" in c.text.lower() and "even" in c.text.lower():
                matches.append((c, 10))
            elif qid == "q5" and "acf" in c.text.lower() and ("bound" in c.text.lower() or "significant" in c.text.lower()):
                matches.append((c, 10))
            elif qid == "q6" and "stationar" in c.text.lower() and ("mean" in c.text.lower() or "differenc" in c.text.lower()):
                matches.append((c, 10))
            elif qid == "q7" and "moving average" in c.text.lower() and ("smooth" in c.text.lower() or "window" in c.text.lower()):
                matches.append((c, 10))
            elif qid == "q8" and "acf" in c.text.lower() and ("trend" in c.text.lower() or "decay" in c.text.lower() or "white noise" in c.text.lower()):
                matches.append((c, 10))

    matches.sort(key=lambda m: m[1], reverse=True)
    best_chunk = matches[0][0] if matches else None
    
    gt_entry = {
        "question_id": qid,
        "question": q["question"],
        "source_lecture": source_pdf,
        "target_chunk_id": best_chunk.chunk_id if best_chunk else None,
        "target_pages": best_chunk.source_pages if best_chunk else None,
        "section_title": best_chunk.section_title if best_chunk else None,
        "text_preview": best_chunk.text[:200] if best_chunk else None,
    }
    ground_truth.append(gt_entry)
    print(f"[{qid}] -> Chunk ID: {gt_entry['target_chunk_id']} | Pages: {gt_entry['target_pages']} | Title: {gt_entry['section_title']}")

with open("benchmarks/ground_truth.json", "w") as f:
    json.dump(ground_truth, f, indent=2)

print("\nSaved benchmarks/ground_truth.json successfully!")
