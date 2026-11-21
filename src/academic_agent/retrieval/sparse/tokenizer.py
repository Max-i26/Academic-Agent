"""Academic Text Analyzer & Tokenizer for Lexical BM25 Retrieval.

Handles domain-specific academic expressions:
- Acronyms (ACF, CMA, SES, MAE, RMSE, ARIMA)
- Subscripts and mathematical variables (Y_t, S_t, e_t, S_{t-1})
- Greek letters and LaTeX expansions (α / alpha, β / beta)
- Model notation (AR(1), MA(2), ARIMA(1,1,1))
- Hyphenated technical terminology (p-value, white-noise)
- Unicode mathematical operators (√ / sqrt, ± / +-, ², ₜ)
"""

import re
from typing import Sequence

# Greek letter mappings to both unicode and latin expansions
GREEK_MAP: dict[str, str] = {
    "α": "alpha",
    "β": "beta",
    "γ": "gamma",
    "δ": "delta",
    "ε": "epsilon",
    "θ": "theta",
    "λ": "lambda",
    "μ": "mu",
    "σ": "sigma",
    "ϕ": "phi",
    "φ": "phi",
    "ω": "omega",
    "ρ": "rho",
    "τ": "tau",
}

# Subscript and superscript character normalizations
UNICODE_CHAR_MAP: dict[str, str] = {
    "₀": "0",
    "₁": "1",
    "₂": "2",
    "₃": "3",
    "₄": "4",
    "₅": "5",
    "₆": "6",
    "₇": "7",
    "₈": "8",
    "₉": "9",
    "ₜ": "t",
    "ᵢ": "i",
    "ⱼ": "j",
    "ₖ": "k",
    "₋": "-",
    "₌": "=",
    "²": "^2",
    "³": "^3",
    "¹": "^1",
    "√": " sqrt ",
    "±": " +- ",
    "−": "-",
    "×": " * ",
    "÷": " / ",
}

# Standard english stop words that carry no discriminative information
# Note: Single-letter variables (e.g. t, k, p, q, y, s) are explicitly NOT filtered out
# because they are critical in time series formulas.
STOP_WORDS: set[str] = {
    "a",
    "about",
    "above",
    "after",
    "again",
    "all",
    "am",
    "an",
    "and",
    "any",
    "are",
    "as",
    "at",
    "be",
    "because",
    "been",
    "before",
    "being",
    "below",
    "between",
    "both",
    "by",
    "can",
    "did",
    "do",
    "does",
    "doing",
    "down",
    "during",
    "each",
    "few",
    "for",
    "from",
    "further",
    "had",
    "has",
    "have",
    "having",
    "he",
    "her",
    "here",
    "hers",
    "herself",
    "him",
    "himself",
    "his",
    "how",
    "i",
    "if",
    "in",
    "into",
    "is",
    "it",
    "its",
    "itself",
    "just",
    "me",
    "more",
    "most",
    "my",
    "myself",
    "no",
    "nor",
    "not",
    "now",
    "of",
    "off",
    "on",
    "once",
    "only",
    "or",
    "other",
    "our",
    "ours",
    "ourselves",
    "out",
    "over",
    "own",
    "same",
    "she",
    "should",
    "so",
    "some",
    "such",
    "than",
    "that",
    "the",
    "their",
    "theirs",
    "them",
    "themselves",
    "then",
    "there",
    "these",
    "they",
    "this",
    "those",
    "through",
    "to",
    "too",
    "under",
    "until",
    "up",
    "very",
    "was",
    "we",
    "were",
    "what",
    "when",
    "where",
    "which",
    "while",
    "who",
    "whom",
    "why",
    "will",
    "with",
    "you",
    "your",
    "yours",
    "yourself",
    "yourselves",
}


class AcademicTokenizer:
    """Specialized tokenizer preserving academic time series terminology, formulas, and symbols."""

    def __init__(self, filter_stop_words: bool = True, expand_greek: bool = True):
        self.filter_stop_words = filter_stop_words
        self.expand_greek = expand_greek

        # Regex for matching distinct academic token patterns
        # 1. Model notation: AR(1), MA(2), ARIMA(1,1,1)
        # 2. Subscript variables: Y_t, S_{t-1}, e_t, s_t
        # 3. Hyphenated technical terms: p-value, white-noise
        # 4. Standard words and acronyms: ACF, MAE, stationarity
        self._pattern = re.compile(
            r"(?P<model>\b(?:AR|MA|ARMA|ARIMA|SARIMA)\s*\(\s*\d+(?:\s*,\s*\d+)*\s*\))|"
            r"(?P<subscript>\b[a-zA-Z]+(?:_\{?[a-zA-Z0-9\-\+]+\}?))|"
            r"(?P<hyphen>\b[a-zA-Z]+(?:-[a-zA-Z0-9]+)+\b)|"
            r"(?P<decimal>\b\d+\.\d+\b)|"
            r"(?P<acronym>\b[A-Z]{2,}\b)|"
            r"(?P<word>\b[a-zA-Z0-9]+\b)|"
            r"(?P<greek>[αβγδεθλμσϕφωρτ])|"
            r"(?P<op>\b(?:sqrt|cma|ses|mae|mse|rmse|acf|pacf)\b)",
            re.IGNORECASE,
        )

    def normalize_academic_text(self, text: str) -> str:
        """Translate unicode subscripts, superscripts, and math operators to ascii representations."""
        # Replace unicode math & subscript characters
        normalized = text
        for char, replacement in UNICODE_CHAR_MAP.items():
            if char in normalized:
                normalized = normalized.replace(char, replacement)
        return normalized

    def tokenize(self, text: str) -> list[str]:
        """Tokenize academic text into a sequence of normalized search tokens."""
        if not text:
            return []

        # 1. Normalize unicode characters
        normalized = self.normalize_academic_text(text)

        tokens: list[str] = []
        for match in self._pattern.finditer(normalized):
            raw = match.group(0).lower()

            # Handle model notations like "ar(1)" -> clean whitespace
            if match.group("model"):
                clean_model = re.sub(r"\s+", "", raw)
                tokens.append(clean_model)
                # Also emit component parts: "ar", "1"
                parts = re.findall(r"[a-z]+|\d+", clean_model)
                tokens.extend(parts)
                continue

            # Handle subscripts like "y_t" or "s_{t-1}"
            if match.group("subscript"):
                clean_sub = raw.replace("{", "").replace("}", "")
                tokens.append(clean_sub)
                # Also emit base and index: "y_t" -> "y", "t"
                parts = clean_sub.split("_")
                if len(parts) == 2:
                    if parts[0]:
                        tokens.append(parts[0])
                    if parts[1]:
                        tokens.append(parts[1])
                continue

            # Handle hyphenated terms like "p-value"
            if match.group("hyphen"):
                tokens.append(raw)
                parts = raw.split("-")
                for p in parts:
                    if p:
                        tokens.append(p)
                continue

            # Handle decimals like "1.96"
            if match.group("decimal"):
                tokens.append(raw)
                continue

            # Handle Greek letters: emit both symbol and name
            if match.group("greek"):
                greek_char = raw
                tokens.append(greek_char)
                if self.expand_greek and greek_char in GREEK_MAP:
                    tokens.append(GREEK_MAP[greek_char])
                continue

            # Standard words or acronyms
            if self.expand_greek and raw in GREEK_MAP.values():
                # If text contains "alpha", also emit "α"
                for g_char, g_name in GREEK_MAP.items():
                    if raw == g_name:
                        tokens.append(g_char)
                        break

            # Filter stop words
            if self.filter_stop_words and raw in STOP_WORDS:
                continue

            tokens.append(raw)

        return tokens
