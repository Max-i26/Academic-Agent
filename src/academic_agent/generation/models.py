"""Data models and exception hierarchy for LLM generation in Academic-Agent."""

from enum import Enum
from types import MappingProxyType
from typing import Any, Mapping
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class MessageRole(str, Enum):
    """Role of a message participant in a chat dialogue."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


class ChatMessage(BaseModel):
    """Structured chat dialogue message."""

    role: MessageRole = Field(description="Role of the message author")
    content: str = Field(description="Textual content of the message")
    name: str | None = Field(default=None, description="Optional participant name or tool identifier")

    @model_validator(mode="after")
    def validate_content_not_empty(self) -> "ChatMessage":
        if not self.content or not self.content.strip():
            raise ValueError("ChatMessage content must not be empty or whitespace only.")
        return self


class LLMConfig(BaseModel):
    """Per-request generation parameters."""

    temperature: float = Field(
        default=0.1,
        ge=0.0,
        le=2.0,
        description="Sampling temperature; 0.0 is deterministic greedy decoding",
    )
    top_p: float = Field(
        default=0.95,
        gt=0.0,
        le=1.0,
        description="Nucleus sampling threshold",
    )
    max_new_tokens: int = Field(
        default=512,
        ge=1,
        le=8192,
        description="Maximum number of tokens to generate",
    )
    stop_sequences: list[str] = Field(
        default_factory=list,
        description="List of textual stop sequences that halt generation",
    )
    timeout_seconds: float = Field(
        default=60.0,
        gt=0.0,
        description="Request timeout in seconds",
    )
    seed: int | None = Field(
        default=None,
        description="Optional random seed for reproducible sampling",
    )


class ModelLoadingConfig(BaseModel):
    """Model loading and hardware execution configuration."""

    model_name: str = Field(description="HuggingFace model identifier or local filesystem path")
    device: str = Field(
        default="auto",
        description="Target execution device ('auto', 'cuda', 'cpu')",
    )
    quantization: str | None = Field(
        default=None,
        description="Quantization mode (None, '4bit', '8bit')",
    )
    context_window: int = Field(
        default=4096,
        ge=512,
        description="Total supported context window in tokens",
    )
    torch_dtype: str = Field(
        default="float16",
        description="Precision dtype ('float16', 'bfloat16', 'float32')",
    )
    revision: str | None = Field(
        default=None,
        description="Optional Git revision or commit hash for reproducibility",
    )
    trust_remote_code: bool = Field(
        default=False,
        description="Whether to permit remote code execution when loading model architecture",
    )


class LLMUsage(BaseModel):
    """Token usage accounting for a generation request."""

    model_config = ConfigDict(frozen=True)

    prompt_tokens: int = Field(default=0, ge=0, description="Count of prompt/context tokens")
    completion_tokens: int = Field(default=0, ge=0, description="Count of generated tokens")
    total_tokens: int = Field(default=0, ge=0, description="Sum of prompt and completion tokens")

    @model_validator(mode="before")
    @classmethod
    def compute_total_tokens(cls, data: Any) -> Any:
        if isinstance(data, dict):
            prompt = data.get("prompt_tokens", 0)
            completion = data.get("completion_tokens", 0)
            if data.get("total_tokens", 0) == 0 and (prompt > 0 or completion > 0):
                data["total_tokens"] = prompt + completion
        return data


class LLMResponse(BaseModel):
    """Complete output produced by an LLM provider generation call."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    content: str = Field(description="Generated response text")
    role: MessageRole = Field(
        default=MessageRole.ASSISTANT,
        description="Role assigned to the generated message",
    )
    finish_reason: str = Field(
        default="stop",
        description="Termination reason ('stop', 'length', 'timeout', etc.)",
    )
    model_name: str = Field(description="Model identifier that produced this completion")
    latency_ms: float = Field(
        ge=0.0,
        description="End-to-end execution latency in milliseconds",
    )
    usage: LLMUsage = Field(
        default_factory=LLMUsage,
        description="Token consumption accounting",
    )
    metadata: Mapping[str, Any] = Field(
        default_factory=lambda: MappingProxyType({}),
        description="Provider-specific diagnostics (e.g. peak VRAM, device, timestamps)",
    )

    @field_validator("metadata", mode="before")
    @classmethod
    def _freeze_metadata(cls, v: Any) -> Any:
        if isinstance(v, Mapping) and not isinstance(v, MappingProxyType):
            return MappingProxyType(dict(v))
        return v


# Typed Exception Hierarchy


class GenerationError(Exception):
    """Base exception for all generation and LLM provider failures."""

    pass


class ContextLengthExceededError(GenerationError):
    """Raised when prompt + requested completion tokens exceed the configured context window."""

    pass


class ModelNotFoundError(GenerationError):
    """Raised when a requested model cannot be located on disk or downloaded."""

    pass


class CudaOutOfMemoryError(GenerationError):
    """Raised when GPU VRAM is exhausted during model loading or generation."""

    pass


class ProviderUnavailableError(GenerationError):
    """Raised when an LLM provider backend is not reachable or cannot execute."""

    pass
