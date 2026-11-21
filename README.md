# Academic-Agent

A locally hosted, domain-specific AI agent designed to help university students understand academic materials through grounded question answering.

## Project Vision

Academic-Agent aims to transform lecture materials into an interactive academic knowledge assistant that can:

- Answer questions using university lecture materials.
- Retrieve relevant information from source documents.
- Provide page-level references for answers.
- Explain complex academic concepts.
- Generate quizzes and revision exercises.
- Support cross-topic reasoning.

## Architecture

The system is designed with modular components:

- **Document Ingestion**: High-fidelity page extraction and positioned layout extraction (fonts, bounding boxes, tables).
- **Text Processing & Chunking**: Conservative cleaning, wrapped-line reconstruction, and structure-aware recursive chunking.
- **Information Retrieval**: Vector storage and hybrid academic retrieval *(upcoming)*.
- **Language Model Integration**: Local, private LLM orchestration *(upcoming)*.
- **Agent Orchestration**: Retrieval-grounded question answering and citation validation.
- **API and User Interface**: Local student interface.

## Technology Stack

- **Python 3.11+**
- **PyMuPDF 1.28.2** (Fast PDF extraction & layout analysis)
- **Pydantic 2.14.2** (Strict data models & validation)
- **Pytest 9.2.0 & Pytest-Cov** (Automated testing suite)

## Development Status

✅ **Phase 1, Phase 2, Phase 3, Phase 4 & Phase 5 (Stage 5.1 & 5.2) Complete**
- PDF Text Ingestion (`PDFExtractor` & `PageRecord`)
- Layout & Table Extraction (`LayoutExtractor` & `PageLayoutRecord`)
- Text Cleaning (`TextCleaner`) & Line Reconstruction (`TextSegmenter`)
- **Structure-Aware Chunking Engine** (`ChunkingEngine`)
- **Phase 3 Storage & Hybrid Retrieval** (Dense, Sparse BM25, Reciprocal Rank Fusion, Cross-Encoder Reranking)
- **Phase 5.1 Model Generation Layer** (`LLMProviderProtocol`, `MockLLMProvider`, `HuggingFaceTransformersProvider`, `OllamaProvider`)
- **Phase 5.2 Academic Evidence Grounding & Synthesis**:
  - `EvidenceContextBuilder`: Token budgeting, provenance deduplication, sequence-aware 5-gram overlap detection.
  - `GroundingPromptBuilder`: System/user role isolation, untrusted XML evidence wrapping, anti-hallucination citation prompts.
  - `CitationValidator`: Occurrence-level tracking, code masking, grouped citation resolution (`group_span` / `sub_span`).
  - `AnswerPolicyValidator`: 6 policy outcomes (`ANSWERED`, `COMPLIANT_ABSTENTION`, `PARTIAL_ANSWER`, `CONTRADICTORY_REFUSAL`, `UNSUPPORTED_ANSWER`, `INDETERMINATE`).
  - `AcademicSynthesizer`: End-to-end grounded synthesis orchestrator.
- **471 automated tests passing** across unit, boundary, and end-to-end integration suites with zero warnings.

### Architectural Boundary: Citation Integrity vs. Semantic Claim Grounding
> **Note on Verification Scope:** Deterministic validation in Academic-Agent verifies citation identity, provenance mapping, and observable answering-policy compliance (refusal language, contradictory continuations, explicit gap statements). It does not claim full claim-level semantic fact verification. An LLM claim citing a valid ID `[E1]` has verified citation integrity and behavioral policy compliance, but verifying that the specific semantic claim is factually entailed by the evidence passage remains an independent downstream research capability.

## Installation

Clone the repository:

```bash
git clone https://github.com/Max-i26/Academic-Agent.git
cd Academic-Agent

# Create virtual environment
python3.11 -m venv .venv
source .venv/bin/activate

# Install the project in editable mode with development dependencies
python -m pip install -e ".[dev]"
```

## Running Tests & Evaluation

Run the automated test suite:

```bash
python -m pytest -v
```

Run real-world lecture layout extraction validation:

```bash
python scripts/validate_layout.py
```

Run real-world lecture chunking evaluation:

```bash
python scripts/evaluate_chunking.py
```

## Project Structure

```text
Academic-Agent/
├── src/
│   └── academic_agent/
│       ├── ingestion/
│       │   ├── pdf_extractor.py
│       │   └── layout_extractor.py
│       ├── processing/
│       │   ├── text_cleaner.py
│       │   ├── text_segmenter.py
│       │   └── chunking/
│       │       ├── models.py
│       │       ├── tokenizer.py
│       │       ├── normalizer.py
│       │       ├── structural_parser.py
│       │       ├── semantic_grouper.py
│       │       ├── size_controller.py
│       │       ├── overlap_manager.py
│       │       ├── provenance.py
│       │       ├── validator.py
│       │       └── engine.py
│       ├── retrieval/
│       ├── agent/
│       └── llm/
├── tests/
│   ├── test_pdf_extractor.py
│   ├── ingestion/
│   │   └── test_layout_extractor.py
│   └── processing/
│       ├── test_text_cleaner.py
│       ├── test_text_segmenter.py
│       └── chunking/
│           ├── test_models.py
│           ├── test_normalizer.py
│           ├── test_structural_parser.py
│           ├── test_semantic_grouper.py
│           ├── test_size_controller.py
│           ├── test_overlap_manager.py
│           ├── test_provenance.py
│           ├── test_validator.py
│           └── test_engine.py
├── scripts/
│   ├── validate_layout.py
│   ├── inspect_layout_samples.py
│   ├── validate_layout_samples.py
│   └── evaluate_chunking.py
├── data/
│   ├── raw/
│   ├── processed/
│   └── knowledge_base/
├── pyproject.toml
└── README.md
```
