"""Exact model-specific tokenizers for sequence validation."""

from typing import Protocol, runtime_checkable
import logging

logger = logging.getLogger(__name__)


@runtime_checkable
class TokenCounter(Protocol):
    """Protocol for counting tokens without requiring full model execution."""

    def count_tokens(self, text: str) -> int:
        ...

    def batch_count_tokens(self, texts: list[str]) -> list[int]:
        ...


class HuggingFaceTokenCounter:
    """Uses Hugging Face AutoTokenizer for exact sub-word token calculation."""

    def __init__(self, model_name: str):
        from transformers import AutoTokenizer

        self.model_name = model_name
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)

    def count_tokens(self, text: str, add_special_tokens: bool = True) -> int:
        """Return exact token count including special tokens ([CLS], [SEP])."""
        tokens = self._tokenizer.encode(text, add_special_tokens=add_special_tokens)
        return len(tokens)

    def batch_count_tokens(self, texts: list[str], add_special_tokens: bool = True) -> list[int]:
        """Return token counts for a list of texts."""
        encodings = self._tokenizer(
            texts,
            add_special_tokens=add_special_tokens,
            padding=False,
            truncation=False,
            return_attention_mask=False,
        )
        return [len(ids) for ids in encodings["input_ids"]]


class WhitespaceTokenCounter:
    """Lightweight token counter for offline testing and fast unit tests."""

    def __init__(self, multiplier: float = 1.33):
        self.multiplier = multiplier

    def count_tokens(self, text: str) -> int:
        words = len(text.split())
        return max(1, int(words * self.multiplier))

    def batch_count_tokens(self, texts: list[str]) -> list[int]:
        return [self.count_tokens(t) for t in texts]
