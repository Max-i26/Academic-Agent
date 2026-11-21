"""Academic-Agent Retrieval Subsystem."""

from academic_agent.retrieval.models import (
    RetrievalStrategy,
    RetrievalQuery,
    RetrievalCandidate,
    Evidence,
    RetrievalDiagnostics,
    RetrievalResponse,
)
from academic_agent.retrieval.protocols import (
    RetrieverProtocol,
    DenseRetrieverProtocol,
    SparseRetrieverProtocol,
    FusionStrategyProtocol,
    RerankerProtocol,
    RetrievalPipelineProtocol,
)
from academic_agent.retrieval.dense_retriever import DenseRetriever
from academic_agent.retrieval.sparse.tokenizer import AcademicTokenizer
from academic_agent.retrieval.sparse.bm25 import BM25Index
from academic_agent.retrieval.sparse_retriever import SparseRetriever

__all__ = [
    "RetrievalStrategy",
    "RetrievalQuery",
    "RetrievalCandidate",
    "Evidence",
    "RetrievalDiagnostics",
    "RetrievalResponse",
    "RetrieverProtocol",
    "DenseRetrieverProtocol",
    "SparseRetrieverProtocol",
    "FusionStrategyProtocol",
    "RerankerProtocol",
    "RetrievalPipelineProtocol",
    "DenseRetriever",
    "AcademicTokenizer",
    "BM25Index",
    "SparseRetriever",
]
