"""Mock LLM provider for deterministic offline testing and contract verification."""

import time
from typing import Callable
from academic_agent.generation.models import (
    ChatMessage,
    LLMConfig,
    LLMResponse,
    LLMUsage,
    MessageRole,
    ContextLengthExceededError,
    CudaOutOfMemoryError,
    GenerationError,
    ProviderUnavailableError,
)
from academic_agent.generation.providers.base import BaseLLMProvider


class MockLLMProvider(BaseLLMProvider):
    """Deterministic mock provider satisfying LLMProviderProtocol without hardware dependencies."""

    def __init__(
        self,
        model_name: str = "mock-academic-llm",
        context_window: int = 4096,
        default_response: str = "This is a deterministic mock response for academic synthesis.",
        responses: list[str] | None = None,
        pattern_responses: dict[str, str] | None = None,
        tokens_per_char: float = 0.25,
        simulate_latency_ms: float = 0.0,
        simulate_cuda_oom: bool = False,
        simulate_context_exceeded: bool = False,
        simulate_generic_error: bool = False,
        is_available: bool = True,
        require_loaded: bool = False,
        on_generation_start: Callable[[list[ChatMessage], LLMConfig], None] | None = None,
        on_generation_end: Callable[[LLMResponse], None] | None = None,
    ) -> None:
        super().__init__(
            on_generation_start=on_generation_start,
            on_generation_end=on_generation_end,
        )
        self._model_name = model_name
        self._context_window = context_window
        self.default_response = default_response
        self._responses = list(responses) if responses is not None else []
        self._pattern_responses = dict(pattern_responses) if pattern_responses is not None else {}
        self.tokens_per_char = tokens_per_char
        self.simulate_latency_ms = simulate_latency_ms
        self.simulate_cuda_oom = simulate_cuda_oom
        self.simulate_context_exceeded = simulate_context_exceeded
        self.simulate_generic_error = simulate_generic_error
        self._is_available = is_available
        self.require_loaded = require_loaded
        self._is_loaded = False
        self.calls: list[tuple[list[ChatMessage], LLMConfig]] = []

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def context_window(self) -> int:
        return self._context_window

    @property
    def is_available(self) -> bool:
        if not self._is_available:
            return False
        if self.require_loaded and not self._is_loaded:
            return False
        return True

    @property
    def is_loaded(self) -> bool:
        return self._is_loaded

    def set_available(self, available: bool) -> None:
        self._is_available = available

    def load(self) -> None:
        self._is_loaded = True

    def release(self) -> None:
        self._is_loaded = False

    def reset(self) -> None:
        """Clear call history and reset state."""
        self.calls.clear()
        self._is_loaded = False

    def _generate_internal(
        self,
        messages: list[ChatMessage],
        config: LLMConfig,
    ) -> LLMResponse:
        self.calls.append((messages, config))

        if self.simulate_cuda_oom:
            raise CudaOutOfMemoryError("CUDA out of memory: simulated GPU VRAM allocation failure.")

        if self.simulate_context_exceeded:
            raise ContextLengthExceededError("Context length exceeded: simulated prompt length overflow.")

        if self.simulate_generic_error:
            raise GenerationError("Simulated LLM backend generation error.")

        # Token calculation heuristic
        combined_text = "\n".join(m.content for m in messages)
        prompt_tokens = max(1, int(len(combined_text) * self.tokens_per_char))

        if prompt_tokens > self._context_window:
            raise ContextLengthExceededError(
                f"Prompt tokens ({prompt_tokens}) exceed model context window ({self._context_window})."
            )

        # Select response content
        chosen_content = None
        for pattern, resp_text in self._pattern_responses.items():
            if pattern.lower() in combined_text.lower():
                chosen_content = resp_text
                break

        if chosen_content is None:
            if self._responses:
                chosen_content = self._responses.pop(0)
            else:
                chosen_content = self.default_response

        # Handle stop sequences
        finish_reason = "stop"
        if config.stop_sequences:
            earliest_idx = len(chosen_content)
            matched_seq = None
            for seq in config.stop_sequences:
                pos = chosen_content.find(seq)
                if pos != -1 and pos < earliest_idx:
                    earliest_idx = pos
                    matched_seq = seq
            if matched_seq is not None:
                chosen_content = chosen_content[:earliest_idx]
                finish_reason = "stop"

        # Handle max_new_tokens truncation
        estimated_completion_tokens = max(1, int(len(chosen_content) * self.tokens_per_char))
        if estimated_completion_tokens > config.max_new_tokens:
            max_chars = int(config.max_new_tokens / self.tokens_per_char)
            chosen_content = chosen_content[:max_chars]
            estimated_completion_tokens = config.max_new_tokens
            finish_reason = "length"

        if self.simulate_latency_ms > 0:
            time.sleep(self.simulate_latency_ms / 1000.0)

        usage = LLMUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=estimated_completion_tokens,
            total_tokens=prompt_tokens + estimated_completion_tokens,
        )

        return LLMResponse(
            content=chosen_content,
            role=MessageRole.ASSISTANT,
            finish_reason=finish_reason,
            model_name=self._model_name,
            latency_ms=self.simulate_latency_ms,
            usage=usage,
            metadata={
                "mock": True,
                "call_count": len(self.calls),
                "device": "mock",
            },
        )
