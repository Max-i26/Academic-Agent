"""Academic-Agent Retrieval Subsystem."""

from typing import Any

from academic_agent.retrieval.fusion import ReciprocalRankFusion
from academic_agent.retrieval.models import (
    Evidence,
    RetrievalCandidate,
    RetrievalDiagnostics,
    RetrievalQuery,
    RetrievalResponse,
    RetrievalStrategy,
)
from academic_agent.retrieval.protocols import (
    DenseRetrieverProtocol,
    FusionStrategyProtocol,
    RerankerProtocol,
    RetrievalPipelineProtocol,
    RetrieverProtocol,
    SparseRetrieverProtocol,
)
from academic_agent.retrieval.sparse.bm25 import BM25Index
from academic_agent.retrieval.sparse.tokenizer import AcademicTokenizer


def __getattr__(name: str) -> Any:
    """PEP 562 lazy import of heavyweight or hardware-dependent retrievers."""
    if name == "DenseRetriever":
        from academic_agent.retrieval.dense_retriever import DenseRetriever
        return DenseRetriever
    if name == "SparseRetriever":
        from academic_agent.retrieval.sparse_retriever import SparseRetriever
        return SparseRetriever
    if name == "CrossEncoderReranker":
        from academic_agent.retrieval.reranker import CrossEncoderReranker
        return CrossEncoderReranker
    if name == "RetrievalPipeline":
        from academic_agent.retrieval.pipeline import RetrievalPipeline
        return RetrievalPipeline
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


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
    "ReciprocalRankFusion",
    "CrossEncoderReranker",
    "RetrievalPipeline",
]
