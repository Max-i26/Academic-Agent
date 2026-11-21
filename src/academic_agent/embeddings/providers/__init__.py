"""Embedding providers package."""

from academic_agent.embeddings.providers.minilm import MiniLMProvider, DEFAULT_MINILM_MODEL
from academic_agent.embeddings.providers.bge import BGEProvider, DEFAULT_BGE_MODEL

__all__ = [
    "MiniLMProvider",
    "BGEProvider",
    "DEFAULT_MINILM_MODEL",
    "DEFAULT_BGE_MODEL",
]
