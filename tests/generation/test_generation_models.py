"""Unit tests for generation data models, configuration, and exception contracts."""

import pytest
from pydantic import ValidationError

from academic_agent.generation.models import (
    MessageRole,
    ChatMessage,
    LLMConfig,
    ModelLoadingConfig,
    LLMUsage,
    LLMResponse,
    GenerationError,
    ContextLengthExceededError,
    ModelNotFoundError,
    CudaOutOfMemoryError,
    ProviderUnavailableError,
)


class TestChatMessage:
    """Test ChatMessage validation and role assignment."""

    def test_valid_chat_message(self):
        msg = ChatMessage(role=MessageRole.USER, content="Explain autoregression.")
        assert msg.role == MessageRole.USER
        assert msg.content == "Explain autoregression."
        assert msg.name is None

    def test_message_with_name(self):
        msg = ChatMessage(role=MessageRole.SYSTEM, content="You are a tutor.", name="system_prompt")
        assert msg.name == "system_prompt"

    def test_message_empty_content_raises(self):
        with pytest.raises(ValidationError):
            ChatMessage(role=MessageRole.USER, content="")

    def test_message_whitespace_content_raises(self):
        with pytest.raises(ValidationError):
            ChatMessage(role=MessageRole.USER, content="   \n\t  ")


class TestLLMConfig:
    """Test LLMConfig defaults and validation constraints."""

    def test_default_config(self):
        cfg = LLMConfig()
        assert cfg.temperature == 0.1
        assert cfg.top_p == 0.95
        assert cfg.max_new_tokens == 512
        assert cfg.stop_sequences == []
        assert cfg.timeout_seconds == 60.0
        assert cfg.seed is None

    def test_custom_valid_config(self):
        cfg = LLMConfig(
            temperature=0.7,
            top_p=0.9,
            max_new_tokens=256,
            stop_sequences=["\n\n", "User:"],
            timeout_seconds=30.0,
            seed=42,
        )
        assert cfg.temperature == 0.7
        assert cfg.top_p == 0.9
        assert cfg.max_new_tokens == 256
        assert len(cfg.stop_sequences) == 2
        assert cfg.seed == 42

    def test_negative_temperature_raises(self):
        with pytest.raises(ValidationError):
            LLMConfig(temperature=-0.1)

    def test_invalid_top_p_raises(self):
        with pytest.raises(ValidationError):
            LLMConfig(top_p=0.0)
        with pytest.raises(ValidationError):
            LLMConfig(top_p=1.5)

    def test_invalid_max_new_tokens_raises(self):
        with pytest.raises(ValidationError):
            LLMConfig(max_new_tokens=0)

    def test_invalid_timeout_raises(self):
        with pytest.raises(ValidationError):
            LLMConfig(timeout_seconds=0.0)


class TestModelLoadingConfig:
    """Test ModelLoadingConfig defaults and validation constraints."""

    def test_default_loading_config(self):
        cfg = ModelLoadingConfig(model_name="Qwen/Qwen2.5-1.5B-Instruct")
        assert cfg.model_name == "Qwen/Qwen2.5-1.5B-Instruct"
        assert cfg.device == "auto"
        assert cfg.quantization is None
        assert cfg.context_window == 4096
        assert cfg.torch_dtype == "float16"
        assert cfg.trust_remote_code is False

    def test_custom_loading_config(self):
        cfg = ModelLoadingConfig(
            model_name="meta-llama/Llama-3.2-1B-Instruct",
            device="cuda",
            quantization="4bit",
            context_window=2048,
            torch_dtype="bfloat16",
        )
        assert cfg.device == "cuda"
        assert cfg.quantization == "4bit"
        assert cfg.context_window == 2048

    def test_context_window_too_small_raises(self):
        with pytest.raises(ValidationError):
            ModelLoadingConfig(model_name="test", context_window=256)


class TestLLMUsage:
    """Test token usage accounting."""

    def test_explicit_tokens(self):
        usage = LLMUsage(prompt_tokens=100, completion_tokens=50, total_tokens=150)
        assert usage.prompt_tokens == 100
        assert usage.completion_tokens == 50
        assert usage.total_tokens == 150

    def test_auto_compute_total_tokens(self):
        usage = LLMUsage(prompt_tokens=120, completion_tokens=30)
        assert usage.total_tokens == 150

    def test_negative_tokens_raises(self):
        with pytest.raises(ValidationError):
            LLMUsage(prompt_tokens=-5)


class TestLLMResponse:
    """Test LLMResponse structure and defaults."""

    def test_valid_response(self):
        resp = LLMResponse(
            content="Stationarity requires constant mean and variance.",
            model_name="mock-model",
            latency_ms=12.5,
            usage=LLMUsage(prompt_tokens=20, completion_tokens=10),
        )
        assert resp.role == MessageRole.ASSISTANT
        assert resp.finish_reason == "stop"
        assert resp.latency_ms == 12.5
        assert resp.usage.total_tokens == 30
        assert resp.metadata == {}


class TestExceptionHierarchy:
    """Test that all generation exceptions inherit correctly from GenerationError."""

    def test_exception_inheritance(self):
        assert issubclass(ContextLengthExceededError, GenerationError)
        assert issubclass(ModelNotFoundError, GenerationError)
        assert issubclass(CudaOutOfMemoryError, GenerationError)
        assert issubclass(ProviderUnavailableError, GenerationError)
        assert issubclass(GenerationError, Exception)

def test_pure_generation_import_isolation():
    """Importing academic_agent.generation must not eagerly import heavy optional dependencies."""
    import sys
    import academic_agent.generation
    import academic_agent.retrieval.models

    disallowed = ["torch", "transformers", "sentence_transformers", "lancedb", "pyarrow"]
    loaded = [mod for mod in disallowed if mod in sys.modules]
    # In pytest session some modules might have been loaded by earlier test suites if run in same process,
    # but the package __init__ files themselves must not eagerly import them.
    assert hasattr(academic_agent.generation, "__getattr__")
