"""Unit tests for MockLLMProvider and BaseLLMProvider contracts."""

import pytest
from academic_agent.generation.models import (
    ChatMessage,
    LLMConfig,
    MessageRole,
    GenerationError,
    ContextLengthExceededError,
    CudaOutOfMemoryError,
    ProviderUnavailableError,
)
from academic_agent.generation.protocols import LLMProviderProtocol
from academic_agent.generation.providers.mock import MockLLMProvider


class TestMockLLMProvider:
    """Contract verification for MockLLMProvider and BaseLLMProvider."""

    def test_protocol_conformance(self):
        provider = MockLLMProvider()
        assert isinstance(provider, LLMProviderProtocol)

    def test_default_generation(self):
        provider = MockLLMProvider()
        messages = [ChatMessage(role=MessageRole.USER, content="Hello, what is AR(1)?")]
        response = provider.generate(messages)

        assert response.role == MessageRole.ASSISTANT
        assert response.finish_reason == "stop"
        assert response.content == provider.default_response
        assert response.model_name == "mock-academic-llm"
        assert response.latency_ms >= 0.0
        assert response.usage.total_tokens > 0
        assert len(provider.calls) == 1

    def test_sequential_responses(self):
        provider = MockLLMProvider(
            responses=["First answer.", "Second answer."],
            default_response="Fallback answer.",
        )
        msg = [ChatMessage(role=MessageRole.USER, content="Query")]

        resp1 = provider.generate(msg)
        assert resp1.content == "First answer."

        resp2 = provider.generate(msg)
        assert resp2.content == "Second answer."

        resp3 = provider.generate(msg)
        assert resp3.content == "Fallback answer."

    def test_pattern_matching_responses(self):
        provider = MockLLMProvider(
            pattern_responses={
                "dickey-fuller": "The Augmented Dickey-Fuller test tests for a unit root.",
                "stationarity": "A process is weakly stationary if mean and autocovariance are constant.",
            }
        )
        msg1 = [ChatMessage(role=MessageRole.USER, content="Tell me about stationarity conditions.")]
        resp1 = provider.generate(msg1)
        assert "weakly stationary" in resp1.content

        msg2 = [ChatMessage(role=MessageRole.USER, content="How does Dickey-Fuller work?")]
        resp2 = provider.generate(msg2)
        assert "Augmented Dickey-Fuller" in resp2.content

    def test_stop_sequence_truncation(self):
        provider = MockLLMProvider(
            default_response="Point 1: First item.\nPoint 2: Second item.\nPoint 3: Third item."
        )
        config = LLMConfig(stop_sequences=["\nPoint 2:"])
        msg = [ChatMessage(role=MessageRole.USER, content="List points")]

        response = provider.generate(msg, config=config)
        assert response.content == "Point 1: First item."
        assert response.finish_reason == "stop"

    def test_max_new_tokens_length_finish_reason(self):
        provider = MockLLMProvider(
            default_response="A" * 200,  # 200 chars * 0.25 = 50 tokens
            tokens_per_char=0.25,
        )
        config = LLMConfig(max_new_tokens=10)  # Allows only 10 tokens (~40 chars)
        msg = [ChatMessage(role=MessageRole.USER, content="Short query")]

        response = provider.generate(msg, config=config)
        assert response.finish_reason == "length"
        assert len(response.content) <= 40
        assert response.usage.completion_tokens == 10

    def test_empty_messages_raises_generation_error(self):
        provider = MockLLMProvider()
        with pytest.raises(GenerationError, match="Cannot generate response from empty message list"):
            provider.generate([])

    def test_invalid_message_type_raises_type_error(self):
        provider = MockLLMProvider()
        with pytest.raises(TypeError, match="must be a ChatMessage instance"):
            provider.generate(["not a chat message"])  # type: ignore

    def test_context_length_exceeded(self):
        provider = MockLLMProvider(context_window=20, tokens_per_char=1.0)
        long_content = "X" * 100  # 100 tokens > 20 tokens
        messages = [ChatMessage(role=MessageRole.USER, content=long_content)]

        with pytest.raises(ContextLengthExceededError):
            provider.generate(messages)

    def test_simulated_cuda_oom(self):
        provider = MockLLMProvider(simulate_cuda_oom=True)
        msg = [ChatMessage(role=MessageRole.USER, content="Test OOM")]

        with pytest.raises(CudaOutOfMemoryError, match="CUDA out of memory"):
            provider.generate(msg)

    def test_simulated_context_exceeded_flag(self):
        provider = MockLLMProvider(simulate_context_exceeded=True)
        msg = [ChatMessage(role=MessageRole.USER, content="Test")]

        with pytest.raises(ContextLengthExceededError):
            provider.generate(msg)

    def test_simulated_generic_error(self):
        provider = MockLLMProvider(simulate_generic_error=True)
        msg = [ChatMessage(role=MessageRole.USER, content="Test")]

        with pytest.raises(GenerationError):
            provider.generate(msg)

    def test_lifecycle_and_availability(self):
        provider = MockLLMProvider(require_loaded=True)
        assert not provider.is_available
        msg = [ChatMessage(role=MessageRole.USER, content="Test")]

        with pytest.raises(ProviderUnavailableError):
            provider.generate(msg)

        provider.load()
        assert provider.is_available
        resp = provider.generate(msg)
        assert resp is not None

        provider.release()
        assert not provider.is_available
        with pytest.raises(ProviderUnavailableError):
            provider.generate(msg)

    def test_explicit_unavailability(self):
        provider = MockLLMProvider()
        provider.set_available(False)
        assert not provider.is_available
        msg = [ChatMessage(role=MessageRole.USER, content="Test")]

        with pytest.raises(ProviderUnavailableError):
            provider.generate(msg)

    def test_callbacks(self):
        started = []
        ended = []

        def on_start(messages, config):
            started.append((messages, config))

        def on_end(response):
            ended.append(response)

        provider = MockLLMProvider(
            on_generation_start=on_start,
            on_generation_end=on_end,
        )
        msg = [ChatMessage(role=MessageRole.USER, content="Query")]
        resp = provider.generate(msg)

        assert len(started) == 1
        assert len(ended) == 1
        assert ended[0] is resp

    def test_reset(self):
        provider = MockLLMProvider()
        provider.load()
        msg = [ChatMessage(role=MessageRole.USER, content="Query")]
        provider.generate(msg)
        assert len(provider.calls) == 1
        assert provider.is_loaded

        provider.reset()
        assert len(provider.calls) == 0
        assert not provider.is_loaded
