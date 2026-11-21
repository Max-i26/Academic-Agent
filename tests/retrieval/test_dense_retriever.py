"""Unit and integration tests for DenseRetriever."""

import numpy as np
import pytest

from academic_agent.embeddings.config import EmbeddingConfig
from academic_agent.embeddings.engine import EmbeddingEngine
from academic_agent.storage.models import IndexMetadata, VectorRecord, FilterQuery
from academic_agent.storage.mock_store import InMemoryVectorStore
from academic_agent.storage.exceptions import StoreNotInitializedError
from academic_agent.retrieval.models import (
    RetrievalQuery,
    RetrievalCandidate,
    RetrievalStrategy,
    Evidence,
)
from academic_agent.retrieval.protocols import (
    RetrieverProtocol,
    DenseRetrieverProtocol,
)
from academic_agent.retrieval.dense_retriever import DenseRetriever

TEST_DIM = 64


def make_unit_vec(components: dict[int, float], dim: int = TEST_DIM) -> list[float]:
    """Helper to create a unit normalized test vector."""
    arr = np.zeros(dim, dtype=np.float32)
    for idx, val in components.items():
        arr[idx] = val
    norm = np.linalg.norm(arr)
    if norm > 0:
        arr = arr / norm
    return arr.tolist()


class MockEmbeddingProvider:
    """Deterministic mock provider for offline dense retrieval tests."""

    def __init__(self, name: str = "mock-minilm", dim: int = TEST_DIM, device: str = "cpu"):
        self._name = name
        self._dim = dim
        self._device = device
        self.queries_encoded: list[str] = []
        self.fixed_query_vector: np.ndarray | None = None

    @property
    def name(self) -> str:
        return self._name

    @property
    def dimension(self) -> int:
        return self._dim

    @property
    def max_sequence_length(self) -> int:
        return 256

    @property
    def device(self) -> str:
        return self._device

    def count_tokens(self, text: str) -> int:
        return len(text.split())

    def encode_queries(self, queries: list[str], batch_size: int = 32, normalize: bool = True) -> np.ndarray:
        self.queries_encoded.extend(queries)
        if self.fixed_query_vector is not None:
            vec = self.fixed_query_vector.copy().astype(np.float32)
            if normalize and np.linalg.norm(vec) > 0:
                vec = vec / np.linalg.norm(vec)
            return np.tile(vec, (len(queries), 1))
        # Default unit vector along axis 0
        vec = np.zeros((len(queries), self._dim), dtype=np.float32)
        vec[:, 0] = 1.0
        return vec

    def encode_documents(self, documents: list[str], batch_size: int = 32, normalize: bool = True) -> np.ndarray:
        vecs = np.zeros((len(documents), self._dim), dtype=np.float32)
        vecs[:, 0] = 1.0
        return vecs


@pytest.fixture
def mock_retriever_setup():
    """Fixture providing initialized DenseRetriever with mock provider and InMemoryVectorStore."""
    provider = MockEmbeddingProvider(dim=TEST_DIM)
    engine_config = EmbeddingConfig(expected_dimension=TEST_DIM, normalize_embeddings=True)
    engine = EmbeddingEngine(provider=provider, config=engine_config)

    store = InMemoryVectorStore(allow_overwrite=True)
    metadata = IndexMetadata(
        model_name="mock-minilm",
        embedding_dimension=TEST_DIM,
        distance_metric="cosine",
        normalized=True,
    )
    store.initialize(metadata)

    retriever = DenseRetriever(embedding_engine=engine, vector_store=store)
    return retriever, engine, store, provider


def test_dense_retriever_satisfies_protocols(mock_retriever_setup):
    retriever, _, _, _ = mock_retriever_setup
    assert isinstance(retriever, RetrieverProtocol)
    assert isinstance(retriever, DenseRetrieverProtocol)
    assert retriever.name == "dense"


def test_query_encoding_and_candidate_pool_size(mock_retriever_setup):
    retriever, _, store, provider = mock_retriever_setup

    # Insert 5 test records
    unit_v = make_unit_vec({0: 1.0})
    records = []
    for i in range(5):
        records.append(
            VectorRecord(
                record_id=f"rec_{i}",
                chunk_id=f"chunk_{i}",
                vector=unit_v,
                text=f"Sample text {i}",
                source_file="Lecture4.pdf",
                source_pages=[i + 1],
            )
        )
    store.add(records)

    # Request top_k=2 but candidate_pool_size=4
    query = RetrievalQuery(
        text="What is simple exponential smoothing?",
        top_k=2,
        candidate_pool_size=4,
    )
    candidates = retriever.retrieve(query)

    # Verify query text passed to embedding provider
    assert "What is simple exponential smoothing?" in provider.queries_encoded
    # Verify pool size governs returned candidates, not top_k
    assert len(candidates) == 4
    assert candidates[0].rank == 1
    assert candidates[3].rank == 4


def test_filter_forwarding(mock_retriever_setup):
    retriever, _, store, _ = mock_retriever_setup
    unit_v = make_unit_vec({0: 1.0})

    rec1 = VectorRecord(
        record_id="c1",
        chunk_id="c1",
        vector=unit_v,
        text="Exponential smoothing alpha weight",
        source_file="4. DS5104 Lecture4 Simple_Exponential_Smoothing.pdf",
        source_pages=[7],
        section_title="THE SES EQUATION",
    )
    rec2 = VectorRecord(
        record_id="c2",
        chunk_id="c2",
        vector=unit_v,
        text="ACF significance bounds",
        source_file="7. DS5104 Lecture7 Autocorrelation ACF.pdf",
        source_pages=[17],
        section_title="Significance Bounds in ACF",
    )
    store.add([rec1, rec2])

    # Filter strictly for Lecture 7
    query = RetrievalQuery(
        text="smoothing parameter alpha",
        top_k=5,
        filters=FilterQuery(source_file="7. DS5104 Lecture7 Autocorrelation ACF.pdf"),
    )
    candidates = retriever.retrieve(query)

    assert len(candidates) == 1
    assert candidates[0].record_id == "c2"
    assert candidates[0].source_file == "7. DS5104 Lecture7 Autocorrelation ACF.pdf"


def test_candidate_provenance_and_fragment_mapping(mock_retriever_setup):
    retriever, _, store, _ = mock_retriever_setup
    unit_v = make_unit_vec({0: 1.0})

    rec = VectorRecord(
        record_id="chunk_split#frag_1",
        chunk_id="chunk_split",
        vector=unit_v,
        text="CMA Step 2 calculation table",
        source_file="5. DS5104 Lecture5 Trend_Seasonal_Decomposition.pdf",
        source_pages=[13, 14],
        source_unit_ids=["unit_101", "unit_102"],
        section_title="CMA Workspace",
        content_type="TABLE",
        fragment_index=1,
        total_fragments=3,
        metadata={"custom_tag": "calculation"},
    )
    store.add([rec])

    query = RetrievalQuery(text="centered moving average", top_k=1)
    candidates = retriever.retrieve(query)

    assert len(candidates) == 1
    cand = candidates[0]

    # Verify complete field fidelity
    assert cand.record_id == "chunk_split#frag_1"
    assert cand.chunk_id == "chunk_split"
    assert cand.rank == 1
    assert cand.retrieval_method == "dense"
    assert cand.text == "CMA Step 2 calculation table"
    assert cand.source_file == "5. DS5104 Lecture5 Trend_Seasonal_Decomposition.pdf"
    assert cand.source_pages == [13, 14]
    assert cand.source_unit_ids == ["unit_101", "unit_102"]
    assert cand.section_title == "CMA Workspace"
    assert cand.content_type == "TABLE"
    assert cand.fragment_index == 1
    assert cand.total_fragments == 3
    assert cand.is_fragment is True
    assert cand.metadata["custom_tag"] == "calculation"
    assert "dense_similarity" in cand.metadata

    # Verify conversion to Evidence preserves all fields including unit IDs
    ev = cand.to_evidence()
    assert isinstance(ev, Evidence)
    assert ev.source_unit_ids == ["unit_101", "unit_102"]
    assert ev.format_citation() == "[Lecture 5, pp. 13-14]"


def test_similarity_score_preservation_and_negative_cosine(mock_retriever_setup):
    retriever, _, store, provider = mock_retriever_setup

    # Query aligns along [1, 0, 0, 0...]
    provider.fixed_query_vector = np.zeros(TEST_DIM, dtype=np.float32)
    provider.fixed_query_vector[0] = 1.0

    # Record 1: identical vector (cosine = 1.0)
    # Record 2: orthogonal vector (cosine = 0.0)
    # Record 3: opposite vector (cosine = -1.0)
    rec_pos = VectorRecord(
        record_id="pos",
        chunk_id="c_pos",
        vector=make_unit_vec({0: 1.0}),
        text="Positive alignment",
        source_file="Lecture1.pdf",
        source_pages=[1],
    )
    rec_ortho = VectorRecord(
        record_id="ortho",
        chunk_id="c_ortho",
        vector=make_unit_vec({1: 1.0}),
        text="Orthogonal alignment",
        source_file="Lecture1.pdf",
        source_pages=[2],
    )
    rec_neg = VectorRecord(
        record_id="neg",
        chunk_id="c_neg",
        vector=make_unit_vec({0: -1.0}),
        text="Negative alignment",
        source_file="Lecture1.pdf",
        source_pages=[3],
    )
    store.add([rec_pos, rec_ortho, rec_neg])

    query = RetrievalQuery(text="test similarity preservation", top_k=3, candidate_pool_size=3)
    candidates = retriever.retrieve(query)

    assert len(candidates) == 3
    # Check descending order
    assert candidates[0].record_id == "pos"
    assert np.isclose(candidates[0].score, 1.0, atol=1e-3)

    assert candidates[1].record_id == "ortho"
    assert np.isclose(candidates[1].score, 0.0, atol=1e-3)

    # Negative cosine similarity MUST NOT be clipped to zero or scaled
    assert candidates[2].record_id == "neg"
    assert np.isclose(candidates[2].score, -1.0, atol=1e-3)
    assert candidates[2].normalized_score is None


def test_latency_diagnostics(mock_retriever_setup):
    retriever, _, store, _ = mock_retriever_setup
    store.add([
        VectorRecord(
            record_id="r1",
            chunk_id="r1",
            vector=make_unit_vec({0: 1.0}),
            text="Text",
            source_file="L.pdf",
            source_pages=[1],
        )
    ])

    query = RetrievalQuery(text="check diagnostics timing", top_k=1)
    retriever.retrieve(query)

    diag = retriever.last_diagnostics
    assert "embed_latency_ms" in diag
    assert "search_latency_ms" in diag
    assert diag["embed_latency_ms"] >= 0.0
    assert diag["search_latency_ms"] >= 0.0


def test_dimension_mismatch_raises():
    dim = TEST_DIM
    provider = MockEmbeddingProvider(dim=dim)
    engine_config = EmbeddingConfig(expected_dimension=dim, normalize_embeddings=True)
    engine = EmbeddingEngine(provider=provider, config=engine_config)

    store = InMemoryVectorStore()
    # Store expects dimension 128, but engine produces 64
    store.initialize(IndexMetadata(model_name="mock", embedding_dimension=128, normalized=True))

    retriever = DenseRetriever(embedding_engine=engine, vector_store=store)
    query = RetrievalQuery(text="mismatched dimension", top_k=1)

    with pytest.raises(ValueError, match="does not match store index dimension"):
        retriever.retrieve(query)


def test_failure_handling_and_type_check(mock_retriever_setup):
    retriever, _, store, _ = mock_retriever_setup

    # 1. Non-RetrievalQuery argument raises TypeError
    with pytest.raises(TypeError, match="Expected RetrievalQuery instance"):
        retriever.retrieve("raw string query")  # type: ignore

    # 2. Closed store raises StoreNotInitializedError
    store.close()
    query = RetrievalQuery(text="query on closed store", top_k=1)
    with pytest.raises(StoreNotInitializedError):
        retriever.retrieve(query)


def test_end_to_end_dense_retrieval_scenario(mock_retriever_setup):
    """End-to-end integration test validating ranking and provenance across multiple lectures."""
    retriever, _, store, provider = mock_retriever_setup

    # Query vector aligned with axis 0
    provider.fixed_query_vector = np.zeros(TEST_DIM, dtype=np.float32)
    provider.fixed_query_vector[0] = 1.0

    r1 = VectorRecord(
        record_id="c1",
        chunk_id="c1",
        vector=make_unit_vec({0: 0.9, 1: 0.1}),  # Cosine ~ 0.99
        text="The SES equation: S_t = alpha * Y_t + (1-alpha) * S_{t-1}",
        source_file="4. DS5104 Lecture4 Simple_Exponential_Smoothing.pdf",
        source_pages=[7],
        section_title="THE SES EQUATION",
    )
    r2 = VectorRecord(
        record_id="c2",
        chunk_id="c2",
        vector=make_unit_vec({0: 0.2, 1: 0.9}),  # Cosine ~ 0.22
        text="Multiplicative decomposition: Y_t = T_t * S_t * I_t",
        source_file="5. DS5104 Lecture5 Trend_Seasonal_Decomposition.pdf",
        source_pages=[8],
        section_title="Multiplicative Decomposition",
    )
    r3 = VectorRecord(
        record_id="c3",
        chunk_id="c3",
        vector=make_unit_vec({0: 0.7, 1: 0.3}),  # Cosine ~ 0.92
        text="ACF significance bounds: +- 1.96 / sqrt(n)",
        source_file="7. DS5104 Lecture7 Autocorrelation ACF.pdf",
        source_pages=[17],
        section_title="Significance Bounds in ACF",
    )
    store.add([r1, r2, r3])

    query = RetrievalQuery(text="What is the SES formula?", top_k=2, candidate_pool_size=3)
    candidates = retriever.retrieve(query)

    assert len(candidates) == 3
    assert candidates[0].record_id == "c1"
    assert candidates[1].record_id == "c3"
    assert candidates[2].record_id == "c2"

    # Verify ranks
    assert [c.rank for c in candidates] == [1, 2, 3]
