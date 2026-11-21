"""Base class for LLM providers handling shared contracts and validation."""

import time
from abc import ABC, abstractmethod
from typing import Callable
from academic_agent.generation.models import (
    ChatMessage,
    LLMConfig,
    LLMResponse,
    ProviderUnavailableError,
    GenerationError,
)
from academic_agent.generation.protocols import LLMProviderProtocol


class BaseLLMProvider(ABC, LLMProviderProtocol):
    """Abstract base provider handling request validation, invocation lifecycle, and timing.

    Backend-specific loading, inference, memory management, and hardware error
    translation are deliberately encapsulated in concrete subclasses.
    """

    def __init__(
        self,
        on_generation_start: Callable[[list[ChatMessage], LLMConfig], None] | None = None,
        on_generation_end: Callable[[LLMResponse], None] | None = None,
    ) -> None:
        self._on_start = on_generation_start
        self._on_end = on_generation_end

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Identifier or path of the model."""
        ...

    @property
    @abstractmethod
    def context_window(self) -> int:
        """Maximum supported context length in tokens."""
        ...

    @property
    @abstractmethod
    def is_available(self) -> bool:
        """Whether this provider backend is operational and ready to serve."""
        ...

    @abstractmethod
    def load(self) -> None:
        """Explicitly allocate runtime resources and load weights."""
        ...

    @abstractmethod
    def release(self) -> None:
        """Explicitly deallocate weights and clear memory."""
        ...

    @abstractmethod
    def _generate_internal(
        self,
        messages: list[ChatMessage],
        config: LLMConfig,
    ) -> LLMResponse:
        """Backend-specific generation implementation."""
        ...

    def generate(
        self,
        messages: list[ChatMessage],
        config: LLMConfig | None = None,
    ) -> LLMResponse:
        """Validate messages and configuration, dispatch to backend, and record latency.

        Args:
            messages: List of ChatMessage objects forming the dialogue.
            config: Optional generation parameters. If omitted, default LLMConfig is used.

        Returns:
            LLMResponse produced by the concrete backend.

        Raises:
            GenerationError: If messages list is empty or generation fails.
            TypeError: If any item in messages is not a ChatMessage instance.
            ProviderUnavailableError: If provider backend is not available.
        """
        if not messages:
            raise GenerationError("Cannot generate response from empty message list.")

        for idx, msg in enumerate(messages):
            if not isinstance(msg, ChatMessage):
                raise TypeError(f"Message at index {idx} must be a ChatMessage instance, got {type(msg)}.")

        if not self.is_available:
            raise ProviderUnavailableError(
                f"Provider for model '{self.model_name}' is currently unavailable or uninitialized."
            )

        resolved_config = config if config is not None else LLMConfig()

        if self._on_start is not None:
            self._on_start(messages, resolved_config)

        t_start = time.perf_counter()
        response = self._generate_internal(messages, resolved_config)
        elapsed_ms = (time.perf_counter() - t_start) * 1000.0

        if response.latency_ms <= 0.0:
            response.latency_ms = elapsed_ms

        if self._on_end is not None:
            self._on_end(response)

        return response
