"""Ollama HTTP provider for local LLM inference via the Ollama REST API."""

import json
import logging
import urllib.error
import urllib.request
from typing import Any, Callable

from academic_agent.generation.models import (
    ChatMessage,
    ContextLengthExceededError,
    GenerationError,
    LLMConfig,
    LLMResponse,
    LLMUsage,
    MessageRole,
    ModelNotFoundError,
    ProviderUnavailableError,
)
from academic_agent.generation.providers.base import BaseLLMProvider

logger = logging.getLogger(__name__)


class OllamaProvider(BaseLLMProvider):
    """Local LLM provider interfacing with an Ollama daemon via its HTTP API.

    Provides non-streaming inference, token accounting, and explicit model unloading
    via Ollama's keep_alive parameter.
    """

    def __init__(
        self,
        model_name: str = "qwen2.5:1.5b",
        host: str = "http://localhost:11434",
        context_window: int = 4096,
        timeout_seconds: float = 60.0,
        on_generation_start: Callable[[list[ChatMessage], LLMConfig], None] | None = None,
        on_generation_end: Callable[[LLMResponse], None] | None = None,
    ) -> None:
        super().__init__(
            on_generation_start=on_generation_start,
            on_generation_end=on_generation_end,
        )
        self._model_name = model_name
        self.host = host.rstrip("/")
        self._context_window = context_window
        self.timeout_seconds = timeout_seconds
        self._is_loaded = False

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def context_window(self) -> int:
        return self._context_window

    @property
    def is_available(self) -> bool:
        """Check whether the Ollama server is reachable."""
        try:
            req = urllib.request.Request(f"{self.host}/api/tags", method="GET")
            with urllib.request.urlopen(req, timeout=1.5) as resp:
                return resp.status == 200
        except Exception:
            return False

    @property
    def is_loaded(self) -> bool:
        return self._is_loaded

    def load(self) -> None:
        """Verify server connectivity and check that the target model exists."""
        if self._is_loaded:
            return

        try:
            req = urllib.request.Request(f"{self.host}/api/tags", method="GET")
            with urllib.request.urlopen(req, timeout=3.0) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except Exception as exc:
            raise ProviderUnavailableError(f"Ollama server unreachable at {self.host}: {exc}") from exc

        models_list = [m.get("name", "") for m in data.get("models", [])]
        # Allow matching with or without ':latest' tag
        clean_name = self._model_name if ":" in self._model_name else f"{self._model_name}:latest"
        matched = any(clean_name == m or self._model_name == m.split(":")[0] for m in models_list)

        if not matched and models_list:
            raise ModelNotFoundError(
                f"Model '{self._model_name}' not found on Ollama server. Available models: {models_list}"
            )

        self._is_loaded = True

    def release(self) -> None:
        """Unload the active model from Ollama GPU VRAM by setting keep_alive to 0."""
        if not self._is_loaded:
            return

        try:
            payload = json.dumps({"model": self._model_name, "keep_alive": 0}).encode("utf-8")
            req = urllib.request.Request(
                f"{self.host}/api/generate",
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=3.0):
                pass
        except Exception as exc:
            logger.debug("Failed sending unload keep_alive=0 to Ollama (%s)", exc)

        self._is_loaded = False

    def _generate_internal(
        self,
        messages: list[ChatMessage],
        config: LLMConfig,
    ) -> LLMResponse:
        """Execute chat completion via Ollama /api/chat endpoint."""
        formatted_messages = [{"role": msg.role.value, "content": msg.content} for msg in messages]

        options: dict[str, Any] = {
            "temperature": config.temperature,
            "top_p": config.top_p,
            "num_predict": config.max_new_tokens,
        }
        if config.stop_sequences:
            options["stop"] = config.stop_sequences

        payload_dict = {
            "model": self._model_name,
            "messages": formatted_messages,
            "stream": False,
            "options": options,
        }
        payload_bytes = json.dumps(payload_dict).encode("utf-8")

        req = urllib.request.Request(
            f"{self.host}/api/chat",
            data=payload_bytes,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        timeout = config.timeout_seconds if config.timeout_seconds > 0 else self.timeout_seconds

        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as http_err:
            if http_err.code == 404:
                raise ModelNotFoundError(f"Ollama model '{self._model_name}' not found: {http_err}") from http_err
            raise GenerationError(f"Ollama HTTP {http_err.code} error: {http_err}") from http_err
        except urllib.error.URLError as url_err:
            raise ProviderUnavailableError(f"Connection to Ollama server failed: {url_err}") from url_err
        except TimeoutError as timeout_err:
            raise GenerationError(f"Ollama generation timed out after {timeout} s: {timeout_err}") from timeout_err
        except Exception as exc:
            raise GenerationError(f"Unexpected error communicating with Ollama: {exc}") from exc

        content = data.get("message", {}).get("content", "")
        prompt_tokens = data.get("prompt_eval_count", 0)
        completion_tokens = data.get("eval_count", 0)
        total_duration_ns = data.get("total_duration", 0)
        latency_ms = round(total_duration_ns / 1_000_000.0, 2)

        finish_reason = "stop"
        done_reason = data.get("done_reason", "")
        if done_reason == "length" or (completion_tokens >= config.max_new_tokens):
            finish_reason = "length"

        usage = LLMUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        )

        return LLMResponse(
            content=content,
            role=MessageRole.ASSISTANT,
            finish_reason=finish_reason,
            model_name=self._model_name,
            latency_ms=latency_ms,
            usage=usage,
            metadata={
                "backend": "ollama",
                "done_reason": done_reason,
                "eval_duration_ms": round(data.get("eval_duration", 0) / 1_000_000.0, 2),
            },
        )
