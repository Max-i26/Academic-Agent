from __future__ import annotations

import re
from typing import Protocol, runtime_checkable


@runtime_checkable
class TokenizerProtocol(Protocol):
    """Protocol for pluggable document and model tokenizers."""

    def count_tokens(self, text: str) -> int:
        """Return the number of tokens in the given text."""
        ...

    def tokenize(self, text: str) -> list[str]:
        """Convert text into a list of token strings."""
        ...


class WhitespaceTokenizer:
    """
    Standard whitespace tokenizer using regex splitting.
    
    Treats whitespace-separated words as tokens.
    """

    WORD_PATTERN = re.compile(r"\S+")

    def count_tokens(self, text: str) -> int:
        if not text:
            return 0
        return len(self.WORD_PATTERN.findall(text))

    def tokenize(self, text: str) -> list[str]:
        if not text:
            return []
        return self.WORD_PATTERN.findall(text)


class EstimateTokenizer:
    """
    Development tokenizer estimating subword tokens for future embedding models.

    Uses a word-to-subword expansion factor (default 1.33x), standard for
    English academic texts prior to selecting a specific model tokenizer.
    """

    WORD_PATTERN = re.compile(r"\S+")

    def __init__(self, factor: float = 1.33) -> None:
        if factor <= 0:
            raise ValueError("Token expansion factor must be positive.")
        self.factor = factor

    def count_tokens(self, text: str) -> int:
        if not text:
            return 0
        words = len(self.WORD_PATTERN.findall(text))
        return int(words * self.factor)

    def tokenize(self, text: str) -> list[str]:
        if not text:
            return []
        return self.WORD_PATTERN.findall(text)
