"""Unit tests for AcademicTokenizer verifying academic symbols, Greek letters, and formulas."""

import pytest
from academic_agent.retrieval.sparse.tokenizer import AcademicTokenizer


class TestAcademicTokenizer:
    """Tests covering domain-specific tokenization rules."""

    @pytest.fixture
    def tokenizer(self):
        return AcademicTokenizer()

    def test_acronym_preservation(self, tokenizer):
        text = "Comparison of ACF, PACF, CMA, and SES models with RMSE and MAE metrics."
        tokens = tokenizer.tokenize(text)
        expected = ["acf", "pacf", "cma", "ses", "models", "rmse", "mae", "metrics"]
        for exp in expected:
            assert exp in tokens, f"Expected '{exp}' in tokens: {tokens}"

    def test_greek_symbol_and_name_expansion(self, tokenizer):
        # α should produce both 'α' and 'alpha'
        text1 = "S_t = α * Y_t + (1 - α) * S_{t-1}"
        tokens1 = tokenizer.tokenize(text1)
        assert "α" in tokens1
        assert "alpha" in tokens1

        # Text with 'alpha' should also match 'α'
        text2 = "smoothing parameter alpha between 0 and 1"
        tokens2 = tokenizer.tokenize(text2)
        assert "alpha" in tokens2
        assert "α" in tokens2

        # Beta / β
        text3 = "Trend slope parameter β"
        tokens3 = tokenizer.tokenize(text3)
        assert "β" in tokens3
        assert "beta" in tokens3

    def test_subscript_and_variable_preservation(self, tokenizer):
        text = "The observation Y_t and error e_t with smoothed level S_{t-1}."
        tokens = tokenizer.tokenize(text)
        # Subscript tokens
        assert "y_t" in tokens
        assert "e_t" in tokens
        assert "s_t-1" in tokens
        # Component parts
        assert "y" in tokens
        assert "t" in tokens

    def test_unicode_subscript_normalization(self, tokenizer):
        # Exact representation extracted from Lecture 4 slide: Sₜ = αYₜ + (1−α)Sₜ₋₁
        text = "THE SES EQUATION: Sₜ = αYₜ + (1−α)Sₜ₋₁"
        tokens = tokenizer.tokenize(text)
        assert "ses" in tokens
        assert "equation" in tokens
        assert "α" in tokens
        assert "alpha" in tokens
        # Normalized subscripts: Sₜ -> s_t or s and t
        assert any("s" in tok for tok in tokens)
        assert any("t" in tok for tok in tokens)

    def test_model_notation(self, tokenizer):
        text = "Fitting an AR(1) and ARIMA(1, 1, 0) process to stationary series."
        tokens = tokenizer.tokenize(text)
        assert "ar(1)" in tokens
        assert "ar" in tokens
        assert "1" in tokens
        assert "arima(1,1,0)" in tokens
        assert "arima" in tokens

    def test_hyphenated_terms(self, tokenizer):
        text = "Reject null hypothesis with small p-value on non-stationary white-noise."
        tokens = tokenizer.tokenize(text)
        assert "p-value" in tokens
        assert "white-noise" in tokens
        assert "non-stationary" in tokens

    def test_unicode_math_operators(self, tokenizer):
        # ± 1.96 / √n and √MSE
        text = "Significance bounds: ± 1.96 / √n, RMSE = √MSE"
        tokens = tokenizer.tokenize(text)
        assert "sqrt" in tokens
        assert "rmse" in tokens
        assert "mse" in tokens
        assert "1.96" in tokens
        assert "n" in tokens

    def test_stop_words_filtering(self, tokenizer):
        # 'the', 'is', 'a' are filtered; mathematical 't' or 'y' are kept
        text = "The value of Y_t is a function of time t."
        tokens = tokenizer.tokenize(text)
        assert "the" not in tokens
        assert "is" not in tokens
        assert "a" not in tokens
        assert "value" in tokens
        assert "y_t" in tokens
        assert "t" in tokens

    def test_empty_and_whitespace_inputs(self, tokenizer):
        assert tokenizer.tokenize("") == []
        assert tokenizer.tokenize("   \n\t  ") == []
        assert tokenizer.tokenize("!@#$%^&*()") == []
