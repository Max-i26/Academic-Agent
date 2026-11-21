"""Unit tests for HuggingFaceTransformersProvider using isolated mock components."""

from typing import Any
from unittest.mock import MagicMock, patch
import pytest
import torch

from academic_agent.generation.models import (
    ChatMessage,
    ContextLengthExceededError,
    CudaOutOfMemoryError,
    GenerationError,
    LLMConfig,
    MessageRole,
    ModelLoadingConfig,
    ModelNotFoundError,
    ProviderUnavailableError,
)
from academic_agent.generation.protocols import LLMProviderProtocol
from academic_agent.generation.providers.hf_local import HuggingFaceTransformersProvider


class MockTokenizer:
    """Mock tokenizer providing deterministic tokenization and decoding."""

    def __init__(self) -> None:
        self.pad_token_id = 0
        self.eos_token_id = 1
        self.chat_template = "{% for msg in messages %}{{ msg.role }}: {{ msg.content }}\n{% endfor %}"

    def apply_chat_template(
        self,
        conversation: list[dict[str, str]],
        tokenize: bool = False,
        add_generation_prompt: bool = True,
    ) -> str:
        res = "\n".join(f"{m['role']}: {m['content']}" for m in conversation)
        if add_generation_prompt:
            res += "\nassistant: "
        return res

    def __call__(self, text: str, return_tensors: str = "pt") -> dict[str, torch.Tensor]:
        # Simple heuristic: 1 token per 4 chars + 2 tokens
        token_count = max(2, len(text) // 4)
        input_ids = torch.arange(10, 10 + token_count, dtype=torch.long).unsqueeze(0)
        attention_mask = torch.ones_like(input_ids)
        return {"input_ids": input_ids, "attention_mask": attention_mask}

    def decode(self, token_ids: torch.Tensor, skip_special_tokens: bool = True) -> str:
        # Map token values to text strings
        tokens = token_ids.tolist() if hasattr(token_ids, "tolist") else list(token_ids)
        if tokens == [101, 102, 103]:
            return "This is a synthesized academic answer."
        if tokens == [201, 202, 203, 204]:
            return "First section.\n[STOP]\nSecond section."
        return f"Decoded tokens: {tokens}"


class MockModelConfig:
    def __init__(self, max_position_embeddings: int = 2048) -> None:
        self.max_position_embeddings = max_position_embeddings


class MockModel:
    """Mock causal language model simulating generation without weights."""

    def __init__(self, generated_tokens: list[int] | None = None) -> None:
        self.config = MockModelConfig()
        self._generated_tokens = generated_tokens or [101, 102, 103]
        self.eval_called = False
        self.to_device = None

    def eval(self) -> "MockModel":
        self.eval_called = True
        return self

    def to(self, device: str) -> "MockModel":
        self.to_device = device
        return self

    def generate(self, input_ids: torch.Tensor, max_new_tokens: int = 512, **kwargs: Any) -> torch.Tensor:
        gen = torch.tensor([self._generated_tokens], dtype=torch.long, device=input_ids.device)
        return torch.cat([input_ids, gen], dim=-1)


class TestHuggingFaceTransformersProviderMocked:
    """Tests for HuggingFaceTransformersProvider without requiring model downloads or GPU."""

    @pytest.fixture
    def mock_components(self):
        tokenizer = MockTokenizer()
        model = MockModel()
        cfg = ModelLoadingConfig(
            model_name="mock-hf/test-model",
            device="cpu",
            context_window=2048,
        )
        return tokenizer, model, cfg

    def test_protocol_conformance(self, mock_components):
        tokenizer, model, cfg = mock_components
        provider = HuggingFaceTransformersProvider(
            loading_config=cfg,
            tokenizer=tokenizer,
            model=model,
        )
        assert isinstance(provider, LLMProviderProtocol)

    def test_initial_properties(self, mock_components):
        tokenizer, model, cfg = mock_components
        provider = HuggingFaceTransformersProvider(
            loading_config=cfg,
            tokenizer=tokenizer,
            model=model,
        )
        assert provider.model_name == "mock-hf/test-model"
        assert provider.context_window == 2048
        assert provider.is_loaded is True
        assert provider.is_available is True

    def test_generate_standard_response(self, mock_components):
        tokenizer, model, cfg = mock_components
        provider = HuggingFaceTransformersProvider(
            loading_config=cfg,
            tokenizer=tokenizer,
            model=model,
        )

        messages = [
            ChatMessage(role=MessageRole.SYSTEM, content="You are an academic tutor."),
            ChatMessage(role=MessageRole.USER, content="Explain AR(1) processes."),
        ]
        config = LLMConfig(temperature=0.0, max_new_tokens=128)
        response = provider.generate(messages, config=config)

        assert response.role == MessageRole.ASSISTANT
        assert response.finish_reason == "stop"
        assert response.content == "This is a synthesized academic answer."
        assert response.usage.prompt_tokens > 0
        assert response.usage.completion_tokens == 3
        assert response.usage.total_tokens == response.usage.prompt_tokens + 3
        assert response.latency_ms >= 0.0

    def test_stop_sequence_handling(self):
        tokenizer = MockTokenizer()
        model = MockModel(generated_tokens=[201, 202, 203, 204])
        cfg = ModelLoadingConfig(model_name="mock-hf/stop-test", device="cpu", context_window=1024)
        provider = HuggingFaceTransformersProvider(loading_config=cfg, tokenizer=tokenizer, model=model)

        messages = [ChatMessage(role=MessageRole.USER, content="Generate two sections")]
        config = LLMConfig(stop_sequences=["\n[STOP]"])
        response = provider.generate(messages, config=config)

        assert response.finish_reason == "stop"
        assert response.content == "First section."
        assert "[STOP]" not in response.content

    def test_max_new_tokens_finish_reason(self):
        tokenizer = MockTokenizer()
        model = MockModel(generated_tokens=[101, 102, 103, 104])
        cfg = ModelLoadingConfig(model_name="mock-hf/len-test", device="cpu", context_window=1024)
        provider = HuggingFaceTransformersProvider(loading_config=cfg, tokenizer=tokenizer, model=model)

        messages = [ChatMessage(role=MessageRole.USER, content="Generate")]
        # Config limits generation to 4 tokens, matching returned token count exactly
        config = LLMConfig(max_new_tokens=4)
        response = provider.generate(messages, config=config)

        assert response.finish_reason == "length"
        assert response.usage.completion_tokens == 4

    def test_context_length_exceeded_error(self):
        tokenizer = MockTokenizer()
        model = MockModel()
        # Minimum valid context window is 512 tokens
        cfg = ModelLoadingConfig(model_name="mock-hf/overflow", device="cpu", context_window=512)
        provider = HuggingFaceTransformersProvider(loading_config=cfg, tokenizer=tokenizer, model=model)

        # Content producing > 512 tokens (each 4 chars = 1 token, so 3000 chars > 700 tokens)
        huge_text = "word " * 600
        messages = [ChatMessage(role=MessageRole.USER, content=huge_text)]

        with pytest.raises(ContextLengthExceededError):
            provider.generate(messages)

    def test_cuda_oom_during_generation(self, mock_components):
        tokenizer, _, cfg = mock_components

        # Create model that simulates torch.cuda.OutOfMemoryError
        class OOMModel(MockModel):
            def generate(self, *args, **kwargs):
                raise torch.cuda.OutOfMemoryError("CUDA out of memory in generate()")

        oom_model = OOMModel()
        provider = HuggingFaceTransformersProvider(loading_config=cfg, tokenizer=tokenizer, model=oom_model)
        messages = [ChatMessage(role=MessageRole.USER, content="Test OOM")]

        with pytest.raises(CudaOutOfMemoryError, match="CUDA out of memory"):
            provider.generate(messages)

        # Verify provider released memory after OOM
        assert provider.is_loaded is False

    def test_lifecycle_release(self, mock_components):
        tokenizer, model, cfg = mock_components
        provider = HuggingFaceTransformersProvider(
            loading_config=cfg,
            tokenizer=tokenizer,
            model=model,
        )
        assert provider.is_loaded is True

        provider.release()
        assert provider.is_loaded is False
        assert provider._model is None
        assert provider._tokenizer is None

    def test_load_model_not_found(self):
        cfg = ModelLoadingConfig(
            model_name="nonexistent/model-that-does-not-exist-xyz123",
            device="cpu",
        )
        provider = HuggingFaceTransformersProvider(loading_config=cfg)

        with patch("transformers.AutoTokenizer.from_pretrained", side_effect=OSError("Model repo not found")):
            with pytest.raises(ModelNotFoundError):
                provider.load()

    def test_load_cuda_not_available(self):
        cfg = ModelLoadingConfig(
            model_name="mock-hf/cuda-test",
            device="cuda",
        )
        provider = HuggingFaceTransformersProvider(loading_config=cfg)

        with patch("torch.cuda.is_available", return_value=False):
            assert provider.is_available is False
            with pytest.raises(ProviderUnavailableError, match="CUDA device requested but torch.cuda.is_available"):
                provider.load()

    def test_completion_budget_clipped_metadata(self):
        """HF provider sets completion_budget_clipped=True when max_new_tokens is clipped by remaining context."""
        tokenizer = MockTokenizer()
        model = MockModel(generated_tokens=[101, 102])
        # Context window 512, prompt has ~300 tokens (1200 chars // 4)
        cfg = ModelLoadingConfig(model_name="mock-hf/clip-test", device="cpu", context_window=512)
        provider = HuggingFaceTransformersProvider(loading_config=cfg, tokenizer=tokenizer, model=model)

        # 1200 chars -> 300 prompt tokens. Budget remaining = 212 tokens.
        messages = [ChatMessage(role=MessageRole.USER, content="A" * 1200)]
        # Request 400 new tokens, which exceeds 212 remaining
        config = LLMConfig(max_new_tokens=400)
        resp = provider.generate(messages, config=config)

        assert resp.metadata["completion_budget_clipped"] is True
