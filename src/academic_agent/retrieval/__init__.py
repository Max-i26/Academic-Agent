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
]
