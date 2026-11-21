"""Helper to browse lecture chunks and construct the 30-question benchmark."""

import json
from pathlib import Path
from academic_agent.ingestion.layout_extractor import LayoutExtractor
from academic_agent.processing.chunking import ChunkingConfig, ChunkingEngine, EstimateTokenizer
from academic_agent.embeddings.config import KnownEmbeddingModel
from academic_agent.embeddings.fragmenter import StructureAwareFragmenter
from academic_agent.embeddings.tokenizer import HuggingFaceTokenCounter


def main():
    extractor = LayoutExtractor()
    tokenizer = EstimateTokenizer(factor=1.33)
    chunking_config = ChunkingConfig(
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
    chunk_engine = ChunkingEngine(config=chunking_config, tokenizer=tokenizer)

    pdf_files = sorted(Path("data/raw").glob("*.pdf"))
    all_chunks = []
    for pdf in pdf_files:
        layout_records = extractor.extract(pdf)
        all_chunks.extend(chunk_engine.chunk(layout_records))

    token_counter = HuggingFaceTokenCounter(model_name=KnownEmbeddingModel.MINILM.value)
    fragmenter = StructureAwareFragmenter(token_counter=token_counter)

    all_fragments = []
    for chunk in all_chunks:
        all_fragments.extend(fragmenter.fragment_chunk(chunk, max_tokens=256))

    print(f"Total fragments/records: {len(all_fragments)}")

    summary = []
    for f in all_fragments:
        summary.append({
            "record_id": f.fragment_id,
            "chunk_id": f.parent_chunk_id,
            "source_file": f.source_file,
            "source_pages": f.source_pages,
            "section_title": f.section_title,
            "text": f.text[:200].replace("\n", " "),
        })

    with open("data/evaluation/chunks_inventory.json", "w") as out:
        json.dump(summary, out, indent=2)
    print("Saved chunks inventory to data/evaluation/chunks_inventory.json")


if __name__ == "__main__":
    Path("data/evaluation").mkdir(parents=True, exist_ok=True)
    main()
