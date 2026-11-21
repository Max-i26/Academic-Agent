"""LLM Provider implementations for Academic-Agent."""

from typing import Any
from academic_agent.generation.providers.base import BaseLLMProvider
from academic_agent.generation.providers.mock import MockLLMProvider


def __getattr__(name: str) -> Any:
    """PEP 562 lazy import of heavyweight or hardware-dependent LLM providers."""
    if name == "HuggingFaceTransformersProvider":
        from academic_agent.generation.providers.hf_local import HuggingFaceTransformersProvider
        return HuggingFaceTransformersProvider
    if name == "OllamaProvider":
        from academic_agent.generation.providers.ollama import OllamaProvider
        return OllamaProvider
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


__all__ = [
    "BaseLLMProvider",
    "MockLLMProvider",
    "HuggingFaceTransformersProvider",
    "OllamaProvider",
]
