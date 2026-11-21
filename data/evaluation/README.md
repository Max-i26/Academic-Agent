# Academic-Agent Retrieval Benchmark (v1)

This directory contains the canonical evaluation dataset and benchmark results for the Academic-Agent retrieval pipeline.

## Dataset Specification: `retrieval_benchmark_v1.json`

The benchmark consists of **30 verified domain questions** spanning the complete 7-lecture curriculum of DS5104 (Time Series Analysis and Forecasting), strictly evaluated against the frozen corpus of **238 canonical records** (235 intact chunks + 3 fragments from Chunk `e9285418782c3cf7`).

### 1. Categories & Distribution (5 Questions Each)

| Category | Description | Primary Challenges |
| :--- | :--- | :--- |
| **`formulas_and_notation`** | Exact mathematical expressions, equations, and variable symbols. | Lexical precision, Greek letters ($\alpha, \nabla$), subscripts ($Y_t, S_{t-1}$), symbol tokenization. |
| **`definitions_and_terminology`** | Formal definitions of time series concepts and properties. | Terminology matching, high-precision semantic definition retrieval. |
| **`conceptual_explanations`** | In-depth reasons ("Why", "When") behind statistical techniques. | Dense semantic alignment, conceptual abstraction across multi-sentence paragraphs. |
| **`calculations_and_workspaces`** | Step-by-step arithmetic procedures and tabular workspaces. | Preserving sequential steps, table structures, and multi-fragment chunks. |
| **`cross_lecture_and_multi_page`** | Queries synthesizing knowledge across distinct lectures/slides. | High candidate recall, broad evidence aggregation, cross-lecture relevance. |
| **`ambiguous_and_variants`** | Informal student phrasings, paraphrases, and term omissions. | Semantic robustness, vocabulary mismatch resistance, query disambiguation. |

### 2. Relevance Scale (0 – 3)

We use a standard 4-point graded relevance scale compatible with TREC/nDCG evaluation:

- **Grade 3 (Definitive / Exact Match)**: The record directly answers the query with explicit formulas, definitions, or core slides.
- **Grade 2 (Strongly Relevant / Contextual Support)**: The record provides essential context, neighboring steps, or strong explanatory support.
- **Grade 1 (Partially Relevant / Peripheral Mention)**: The record mentions the topic or an application tangentially.
- **Grade 0 (Non-Relevant / Distractor)**: Unrelated or irrelevant content.

### 3. Provenance & Fragment ID Resolution

Every query specifies:
- `relevant_records`: Map of `record_id` (or `chunk_id`) to relevance grade (1–3).
- `canonical_chunk_ids`: List of parent chunk IDs that contain the target content.
- `expected_sources`: Provenance metadata including `source_file`, `pages`, and `section_title`.
- `notes`: Human-readable annotation of expected evidence.

The `RetrievalEvaluator` automatically resolves fragment IDs (e.g. `e9285418782c3cf7#frag_0`) to parent chunk IDs to ensure fair evaluation of fragmented tables and multi-part content.

### 4. Evaluated Metrics

1. **Hit@K ($K \in \{1, 3, 5\}$)**: Proportion of queries with at least one relevant record in the top $K$.
2. **Mean Reciprocal Rank (MRR)**: Average of $1 / \text{rank}$ of the first relevant candidate.
3. **Mean Average Precision (MAP)**: Area under the precision-recall curve across all relevant documents.
4. **Recall@5**: Fraction of all relevant records retrieved in the top 5.
5. **Normalized Discounted Cumulative Gain (nDCG@5)**: Position-weighted metric with graded relevance ($2^{\text{rel}} - 1 / \log_2(i + 1)$).
6. **Latency**: Sub-millisecond timing per stage (Dense retrieval, BM25 sparse search, RRF fusion, Total end-to-end).
