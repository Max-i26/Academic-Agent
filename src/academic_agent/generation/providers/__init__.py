"""LLM Provider implementations for Academic-Agent."""

from academic_agent.generation.providers.base import BaseLLMProvider
from academic_agent.generation.providers.mock import MockLLMProvider
from academic_agent.generation.providers.hf_local import HuggingFaceTransformersProvider
from academic_agent.generation.providers.ollama import OllamaProvider

__all__ = [
    "BaseLLMProvider",
    "MockLLMProvider",
    "HuggingFaceTransformersProvider",
    "OllamaProvider",
]
