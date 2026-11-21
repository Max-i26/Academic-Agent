"""Protocols defining contracts for LLM inference providers."""

from typing import Protocol, runtime_checkable
from academic_agent.generation.models import (
    ChatMessage,
    LLMConfig,
    LLMResponse,
)


@runtime_checkable
class LLMProviderProtocol(Protocol):
    """Protocol defining the uniform generation contract for all LLM backends."""

    @property
    def model_name(self) -> str:
        """Identifier or path of the model served by this provider."""
        ...

    @property
    def context_window(self) -> int:
        """Maximum supported context length (in tokens) for this model instance."""
        ...

    @property
    def is_available(self) -> bool:
        """Check whether this provider backend is functional and ready for inference."""
        ...

    def generate(
        self,
        messages: list[ChatMessage],
        config: LLMConfig | None = None,
    ) -> LLMResponse:
        """Generate a model completion from a sequence of chat messages.

        Args:
            messages: List of ChatMessage objects representing dialogue context.
            config: Optional request-level generation parameters.

        Returns:
            LLMResponse containing generated text, usage metrics, and diagnostics.

        Raises:
            ContextLengthExceededError: If input tokens exceed model context budget.
            CudaOutOfMemoryError: If GPU VRAM is exhausted during generation.
            ProviderUnavailableError: If provider is not initialized or backend is unreachable.
            GenerationError: For other execution failures during generation.
        """
        ...

    def load(self) -> None:
        """Explicitly allocate runtime resources and load weights into memory."""
        ...

    def release(self) -> None:
        """Explicitly deallocate weights and clear memory/VRAM caches."""
        ...
