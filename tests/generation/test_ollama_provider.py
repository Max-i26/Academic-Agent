"""Unit tests for OllamaProvider using mock HTTP responses."""

import io
import json
import urllib.error
from unittest.mock import MagicMock, patch
import pytest

from academic_agent.generation.models import (
    ChatMessage,
    GenerationError,
    LLMConfig,
    MessageRole,
    ModelNotFoundError,
    ProviderUnavailableError,
)
from academic_agent.generation.protocols import LLMProviderProtocol
from academic_agent.generation.providers.ollama import OllamaProvider


def make_mock_http_response(status: int, body_dict: dict) -> MagicMock:
    mock_resp = MagicMock()
    mock_resp.status = status
    mock_resp.read.return_value = json.dumps(body_dict).encode("utf-8")
    mock_resp.__enter__.return_value = mock_resp
    return mock_resp


class TestOllamaProvider:
    """Test suite for OllamaProvider without requiring a running Ollama daemon."""

    def test_protocol_conformance(self):
        provider = OllamaProvider(model_name="qwen2.5:1.5b")
        assert isinstance(provider, LLMProviderProtocol)

    def test_is_available_true(self):
        provider = OllamaProvider(model_name="qwen2.5:1.5b")
        with patch("urllib.request.urlopen", return_value=make_mock_http_response(200, {"models": []})):
            assert provider.is_available is True

    def test_is_available_false_on_connection_error(self):
        provider = OllamaProvider(model_name="qwen2.5:1.5b")
        with patch("urllib.request.urlopen", side_effect=urllib.error.URLError("Connection refused")):
            assert provider.is_available is False

    def test_load_success(self):
        provider = OllamaProvider(model_name="qwen2.5:1.5b")
        mock_tags = {"models": [{"name": "qwen2.5:1.5b"}, {"name": "llama3.2:1b"}]}
        with patch("urllib.request.urlopen", return_value=make_mock_http_response(200, mock_tags)):
            provider.load()
            assert provider.is_loaded is True

    def test_load_model_not_found(self):
        provider = OllamaProvider(model_name="nonexistent:model")
        mock_tags = {"models": [{"name": "qwen2.5:1.5b"}]}
        with patch("urllib.request.urlopen", return_value=make_mock_http_response(200, mock_tags)):
            with pytest.raises(ModelNotFoundError):
                provider.load()

    def test_load_server_unreachable(self):
        provider = OllamaProvider(model_name="qwen2.5:1.5b")
        with patch("urllib.request.urlopen", side_effect=urllib.error.URLError("Connection refused")):
            with pytest.raises(ProviderUnavailableError):
                provider.load()

    def test_generate_success(self):
        provider = OllamaProvider(model_name="qwen2.5:1.5b")
        mock_response = {
            "model": "qwen2.5:1.5b",
            "message": {
                "role": "assistant",
                "content": "Weak stationarity requires constant mean and autocovariance.",
            },
            "done": True,
            "done_reason": "stop",
            "total_duration": 450_000_000,  # 450 ms
            "prompt_eval_count": 25,
            "eval_count": 18,
            "eval_duration": 350_000_000,
        }

        # Mock is_available for BaseLLMProvider.generate() validation check
        with patch.object(OllamaProvider, "is_available", True):
            with patch("urllib.request.urlopen", return_value=make_mock_http_response(200, mock_response)):
                messages = [ChatMessage(role=MessageRole.USER, content="Define weak stationarity.")]
                resp = provider.generate(messages)

                assert resp.role == MessageRole.ASSISTANT
                assert resp.finish_reason == "stop"
                assert "Weak stationarity" in resp.content
                assert resp.usage.prompt_tokens == 25
                assert resp.usage.completion_tokens == 18
                assert resp.usage.total_tokens == 43
                assert resp.latency_ms > 0.0
                assert resp.metadata["backend"] == "ollama"

    def test_generate_length_finish_reason(self):
        provider = OllamaProvider(model_name="qwen2.5:1.5b")
        mock_response = {
            "model": "qwen2.5:1.5b",
            "message": {"role": "assistant", "content": "Partial output..."},
            "done": True,
            "done_reason": "length",
            "total_duration": 200_000_000,
            "prompt_eval_count": 10,
            "eval_count": 50,
        }

        with patch.object(OllamaProvider, "is_available", True):
            with patch("urllib.request.urlopen", return_value=make_mock_http_response(200, mock_response)):
                messages = [ChatMessage(role=MessageRole.USER, content="Query")]
                config = LLMConfig(max_new_tokens=50)
                resp = provider.generate(messages, config=config)

                assert resp.finish_reason == "length"

    def test_release(self):
        provider = OllamaProvider(model_name="qwen2.5:1.5b")
        provider._is_loaded = True

        with patch("urllib.request.urlopen", return_value=make_mock_http_response(200, {})):
            provider.release()
            assert provider.is_loaded is False
