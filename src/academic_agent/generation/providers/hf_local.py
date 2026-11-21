"""HuggingFace Transformers local LLM provider supporting CPU, CUDA, FP16, and 4-bit NF4 quantization."""

import gc
import logging
from typing import Any, Callable

import torch
from academic_agent.generation.models import (
    ChatMessage,
    ContextLengthExceededError,
    CudaOutOfMemoryError,
    GenerationError,
    LLMConfig,
    LLMResponse,
    LLMUsage,
    MessageRole,
    ModelLoadingConfig,
    ModelNotFoundError,
    ProviderUnavailableError,
)
from academic_agent.generation.providers.base import BaseLLMProvider

logger = logging.getLogger(__name__)


class _StopSequenceCriteria:
    """Stopping criteria that halts generation when any target string is produced."""

    def __init__(self, stop_sequences: list[str], tokenizer: Any, prompt_length: int) -> None:
        self.stop_sequences = stop_sequences
        self.tokenizer = tokenizer
        self.prompt_length = prompt_length

    def __call__(self, input_ids: torch.LongTensor, scores: Any, **kwargs: Any) -> bool:
        # Check newly generated tokens
        if input_ids.shape[-1] <= self.prompt_length:
            return False
        generated_tokens = input_ids[0, self.prompt_length :]
        decoded_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=False)
        return any(seq in decoded_text for seq in self.stop_sequences)


class HuggingFaceTransformersProvider(BaseLLMProvider):
    """Local LLM provider wrapping HuggingFace Transformers AutoModelForCausalLM.

    Designed for memory-constrained environments (e.g. 4.0 GB VRAM), providing explicit
    lifecycle controls (load/release), FP16 precision, optional 4-bit NF4 quantization,
    and exact token counting.
    """

    def __init__(
        self,
        loading_config: ModelLoadingConfig,
        tokenizer: Any | None = None,
        model: Any | None = None,
        on_generation_start: Callable[[list[ChatMessage], LLMConfig], None] | None = None,
        on_generation_end: Callable[[LLMResponse], None] | None = None,
    ) -> None:
        super().__init__(
            on_generation_start=on_generation_start,
            on_generation_end=on_generation_end,
        )
        self.loading_config = loading_config
        self._tokenizer = tokenizer
        self._model = model
        self._is_loaded = bool(tokenizer is not None and model is not None)
        self._resolved_device: str = "cpu"
        self._effective_context_window: int = loading_config.context_window

        if self._is_loaded:
            self._update_effective_context_window()

    @property
    def model_name(self) -> str:
        return self.loading_config.model_name

    @property
    def context_window(self) -> int:
        return self._effective_context_window

    @property
    def is_available(self) -> bool:
        target_dev = self.loading_config.device
        if target_dev == "cuda" and not torch.cuda.is_available():
            return False
        if self.loading_config.quantization in ("4bit", "8bit") and not torch.cuda.is_available():
            return False
        return True

    @property
    def is_loaded(self) -> bool:
        return self._is_loaded and self._model is not None and self._tokenizer is not None

    @property
    def device(self) -> str:
        return self._resolved_device

    def load(self) -> None:
        """Load tokenizer and model weights into memory according to loading_config."""
        if self.is_loaded:
            return

        # 1. Resolve execution device
        req_device = self.loading_config.device
        if req_device == "auto":
            target_device = "cuda" if torch.cuda.is_available() else "cpu"
        elif req_device == "cuda":
            if not torch.cuda.is_available():
                raise ProviderUnavailableError("CUDA device requested but torch.cuda.is_available() is False.")
            target_device = "cuda"
        elif req_device == "cpu":
            target_device = "cpu"
        else:
            target_device = req_device

        self._resolved_device = target_device

        # 2. Resolve precision dtype
        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        torch_dtype = dtype_map.get(self.loading_config.torch_dtype, torch.float16)

        # 3. Resolve quantization config
        quant_config = None
        device_map = None
        if self.loading_config.quantization is not None:
            if target_device != "cuda":
                raise ProviderUnavailableError(
                    f"Quantization '{self.loading_config.quantization}' requires a CUDA GPU."
                )
            try:
                from transformers import BitsAndBytesConfig
            except ImportError as err:
                raise ProviderUnavailableError(
                    "Quantization requires 'bitsandbytes' and 'transformers'. Install them to use 4-bit/8-bit."
                ) from err

            if self.loading_config.quantization == "4bit":
                quant_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch_dtype,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                )
                device_map = "auto"
            elif self.loading_config.quantization == "8bit":
                quant_config = BitsAndBytesConfig(load_in_8bit=True)
                device_map = "auto"
            else:
                raise ValueError(f"Unsupported quantization mode: '{self.loading_config.quantization}'.")

        # 4. Load Tokenizer & Model if not injected
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer

            if self._tokenizer is None:
                self._tokenizer = AutoTokenizer.from_pretrained(
                    self.loading_config.model_name,
                    revision=self.loading_config.revision,
                    trust_remote_code=self.loading_config.trust_remote_code,
                )
                if self._tokenizer.pad_token_id is None:
                    self._tokenizer.pad_token_id = self._tokenizer.eos_token_id

            if self._model is None:
                if quant_config is not None:
                    self._model = AutoModelForCausalLM.from_pretrained(
                        self.loading_config.model_name,
                        quantization_config=quant_config,
                        torch_dtype=torch_dtype,
                        device_map=device_map,
                        revision=self.loading_config.revision,
                        trust_remote_code=self.loading_config.trust_remote_code,
                    )
                else:
                    self._model = AutoModelForCausalLM.from_pretrained(
                        self.loading_config.model_name,
                        torch_dtype=torch_dtype,
                        revision=self.loading_config.revision,
                        trust_remote_code=self.loading_config.trust_remote_code,
                    )
                    self._model = self._model.to(target_device)

                self._model.eval()

            self._is_loaded = True
            self._update_effective_context_window()

        except torch.cuda.OutOfMemoryError as oom_err:
            self.release()
            raise CudaOutOfMemoryError(
                f"CUDA out of memory while loading model '{self.loading_config.model_name}': {oom_err}"
            ) from oom_err
        except (OSError, EnvironmentError, ValueError) as not_found_err:
            self.release()
            raise ModelNotFoundError(
                f"Failed to load model '{self.loading_config.model_name}': {not_found_err}"
            ) from not_found_err
        except Exception as exc:
            self.release()
            raise GenerationError(f"Unexpected error loading model '{self.loading_config.model_name}': {exc}") from exc

    def release(self) -> None:
        """Deallocate model weights, tokenizer, and clear GPU VRAM cache."""
        if self._model is not None:
            del self._model
            self._model = None
        if self._tokenizer is not None:
            del self._tokenizer
            self._tokenizer = None

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        gc.collect()
        self._is_loaded = False

    def _update_effective_context_window(self) -> None:
        """Derive bounded context window based on model architecture and user configuration."""
        configured = self.loading_config.context_window
        model_limit = None

        if self._model is not None and hasattr(self._model, "config"):
            cfg = self._model.config
            model_limit = getattr(cfg, "max_position_embeddings", None) or getattr(cfg, "seq_length", None)

        if model_limit is not None and isinstance(model_limit, int) and model_limit > 0:
            self._effective_context_window = min(configured, model_limit)
        else:
            self._effective_context_window = configured

    def _format_chat_prompt(self, messages: list[ChatMessage]) -> str:
        """Format dialogue messages into model-appropriate prompt string."""
        if (
            self._tokenizer is not None
            and hasattr(self._tokenizer, "apply_chat_template")
            and getattr(self._tokenizer, "chat_template", None) is not None
        ):
            dialogue = [{"role": msg.role.value, "content": msg.content} for msg in messages]
            try:
                return self._tokenizer.apply_chat_template(
                    dialogue,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except Exception as template_err:
                logger.debug("Failed applying chat template (%s); falling back to default.", template_err)

        # Standard Fallback ChatML-style formatting
        formatted_parts = []
        for msg in messages:
            formatted_parts.append(f"<|im_start|>{msg.role.value}\n{msg.content}<|im_end|>")
        formatted_parts.append("<|im_start|>assistant\n")
        return "\n".join(formatted_parts)

    def _generate_internal(
        self,
        messages: list[ChatMessage],
        config: LLMConfig,
    ) -> LLMResponse:
        """Execute text generation using the resident transformers model."""
        if not self.is_loaded:
            self.load()

        if self._model is None or self._tokenizer is None:
            raise ProviderUnavailableError("Model or tokenizer is uninitialized.")

        # 1. Format and tokenize input
        prompt_text = self._format_chat_prompt(messages)
        inputs = self._tokenizer(prompt_text, return_tensors="pt")

        input_ids = inputs["input_ids"].to(self._resolved_device)
        attention_mask = inputs.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(self._resolved_device)

        prompt_tokens = int(input_ids.shape[-1])

        # 2. Context window enforcement
        if prompt_tokens >= self.context_window:
            raise ContextLengthExceededError(
                f"Prompt tokens ({prompt_tokens}) exceed model context window ({self.context_window})."
            )

        budget_remaining = self.context_window - prompt_tokens
        if budget_remaining < 1:
            raise ContextLengthExceededError(
                f"No token budget remaining for generation (prompt_tokens={prompt_tokens}, context_window={self.context_window})."
            )

        actual_max_new_tokens = min(config.max_new_tokens, budget_remaining)
        completion_clipped = actual_max_new_tokens < config.max_new_tokens
        if completion_clipped:
            logger.warning(
                f"Requested max_new_tokens ({config.max_new_tokens}) exceeds remaining context budget "
                f"({budget_remaining}). Generation ceiling clipped to {actual_max_new_tokens} tokens."
            )

        # 3. Configure stopping criteria
        stopping_criteria = None
        if config.stop_sequences:
            from transformers import StoppingCriteriaList

            criteria = _StopSequenceCriteria(
                stop_sequences=config.stop_sequences,
                tokenizer=self._tokenizer,
                prompt_length=prompt_tokens,
            )
            stopping_criteria = StoppingCriteriaList([criteria])

        # 4. Build generation arguments
        gen_kwargs: dict[str, Any] = {
            "input_ids": input_ids,
            "max_new_tokens": actual_max_new_tokens,
            "pad_token_id": self._tokenizer.pad_token_id or self._tokenizer.eos_token_id,
            "eos_token_id": self._tokenizer.eos_token_id,
        }

        if attention_mask is not None:
            gen_kwargs["attention_mask"] = attention_mask

        if config.temperature > 0.0:
            gen_kwargs["do_sample"] = True
            gen_kwargs["temperature"] = config.temperature
            gen_kwargs["top_p"] = config.top_p
        else:
            gen_kwargs["do_sample"] = False

        if config.seed is not None:
            torch.manual_seed(config.seed)

        if stopping_criteria is not None:
            gen_kwargs["stopping_criteria"] = stopping_criteria

        # 5. Execute generation under no_grad
        try:
            with torch.no_grad():
                output_ids = self._model.generate(**gen_kwargs)
        except torch.cuda.OutOfMemoryError as oom_err:
            self.release()
            raise CudaOutOfMemoryError(
                f"CUDA out of memory during generation with model '{self.model_name}': {oom_err}"
            ) from oom_err

        # 6. Decode output tokens
        generated_ids = output_ids[0, prompt_tokens:]
        completion_tokens = int(len(generated_ids))
        generated_text = self._tokenizer.decode(generated_ids, skip_special_tokens=True)

        # 7. Post-process stop sequences and finish reason
        finish_reason = "stop"
        if config.stop_sequences:
            earliest_idx = len(generated_text)
            matched = False
            for seq in config.stop_sequences:
                pos = generated_text.find(seq)
                if pos != -1 and pos < earliest_idx:
                    earliest_idx = pos
                    matched = True
            if matched:
                generated_text = generated_text[:earliest_idx]
                finish_reason = "stop"
            elif completion_tokens >= actual_max_new_tokens:
                finish_reason = "length"
        elif completion_tokens >= actual_max_new_tokens:
            finish_reason = "length"

        # 8. Token usage accounting and diagnostic metadata
        usage = LLMUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        )

        metadata: dict[str, Any] = {
            "device": self._resolved_device,
            "model_name": self.model_name,
            "quantization": self.loading_config.quantization,
            "torch_dtype": self.loading_config.torch_dtype,
            "completion_budget_clipped": completion_clipped,
        }

        if self._resolved_device == "cuda" and torch.cuda.is_available():
            metadata["peak_vram_mb"] = round(torch.cuda.max_memory_allocated() / (1024 * 1024), 2)

        return LLMResponse(
            content=generated_text,
            role=MessageRole.ASSISTANT,
            finish_reason=finish_reason,
            model_name=self.model_name,
            latency_ms=0.0,  # Will be populated with measured time by BaseLLMProvider.generate()
            usage=usage,
            metadata=metadata,
        )
