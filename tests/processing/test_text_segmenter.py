
import pytest

from academic_agent.processing.text_segmenter import TextSegmenter


@pytest.fixture
def segmenter():
    return TextSegmenter()


def test_joins_wrapped_sentence_lines(segmenter):
    text = (
        "Can we prepare\n"
        "beds, staff, and\n"
        "medicine for next\n"
        "week?"
    )

    expected = (
        "Can we prepare beds, staff, and "
        "medicine for next week?"
    )

    assert segmenter.segment(text) == expected


def test_preserves_bullet_points(segmenter):
    text = (
        "Where do time series appear?\n"
        "• Daily rainfall\n"
        "• Hourly electricity demand\n"
        "• Weekly dengue cases"
    )

    expected = (
        "Where do time series appear?\n"
        "• Daily rainfall\n"
        "• Hourly electricity demand\n"
        "• Weekly dengue cases"
    )

    assert segmenter.segment(text) == expected


def test_preserves_heading_ending_with_colon(segmenter):
    text = (
        "Question:\n"
        "Can we prepare beds for next week?"
    )

    expected = (
        "Question:\n"
        "Can we prepare beds for next week?"
    )

    assert segmenter.segment(text) == expected


def test_preserves_paragraph_boundaries(segmenter):
    text = (
        "First paragraph\n"
        "continues here.\n"
        "\n"
        "Second paragraph\n"
        "continues here."
    )

    expected = (
        "First paragraph continues here.\n\n"
        "Second paragraph continues here."
    )

    assert segmenter.segment(text) == expected


def test_does_not_join_after_sentence_ending(segmenter):
    text = (
        "First sentence.\n"
        "Second sentence."
    )

    assert segmenter.segment(text) == text


def test_does_not_join_uppercase_continuation(segmenter):
    text = (
        "First line\n"
        "Second line"
    )

    assert segmenter.segment(text) == text


def test_preserves_standalone_equation(segmenter):
    text = (
        "Forecasting model\n"
        "Y = a + bt\n"
        "Next explanation"
    )

    assert segmenter.segment(text) == text


def test_preserves_unicode_mathematical_notation(segmenter):
    text = (
        "Forecasting means using available information up to time t "
        "to predict a future value.\n"
        "Ŷₜ₊ₕ|ₜ\n"
        "h is the forecast horizon."
    )

    expected = (
        "Forecasting means using available information up to time t "
        "to predict a future value.\n"
        "Ŷₜ₊ₕ|ₜ\n"
        "h is the forecast horizon."
    )

    assert segmenter.segment(text) == expected


def test_rejects_non_string_input(segmenter):
    with pytest.raises(TypeError):
        segmenter.segment(None)

def test_handles_consecutive_blank_lines(segmenter):
    text = (
        "First paragraph."
        "\n\n\n"
        "Second paragraph."
    )

    expected = (
        "First paragraph."
        "\n\n"
        "Second paragraph."
    )

    assert segmenter.segment(text) == expected


def test_removes_trailing_blank_lines(segmenter):
    text = "First paragraph.\n\n"

    expected = "First paragraph."

    assert segmenter.segment(text) == expected

def test_joins_wrapped_bullet_continuations():
    segmenter = TextSegmenter()

    text = (
        "• Trend: long-term\n"
        "movement\n"
        "• Seasonality: repeated\n"
        "calendar pattern\n"
        "• Cycle: irregular long\n"
        "wave\n"
        "• Noise: random\n"
        "variation"
    )

    expected = (
        "• Trend: long-term movement\n"
        "• Seasonality: repeated calendar pattern\n"
        "• Cycle: irregular long wave\n"
        "• Noise: random variation"
    )

    assert segmenter.segment(text) == expected


def test_joins_prose_with_incomplete_equation():
    segmenter = TextSegmenter()

    text = "Use α = 0.40 and S₁ =\n50."

    assert segmenter.segment(text) == "Use α = 0.40 and S₁ = 50."


def test_joins_instruction_with_subscript_continuation():
    segmenter = TextSegmenter()

    text = "Calculate S₂ to\nS₆."

    assert segmenter.segment(text) == "Calculate S₂ to S₆."


def test_joins_prose_with_greek_symbol_continuation():
    segmenter = TextSegmenter()

    text = "Explain what α =\n0.40 means."

    assert segmenter.segment(text) == "Explain what α = 0.40 means."


def test_preserves_standalone_equation():
    segmenter = TextSegmenter()

    text = "Sₜ = αYₜ + (1−α)Sₜ₋₁"

    assert segmenter.segment(text) == text


def test_does_not_classify_mathematical_symbols_in_prose_as_equations():
    segmenter = TextSegmenter()

    text = (
        "The model uses log(Yₜ)\n"
        "to transform the observations."
    )

    assert segmenter.segment(text) == (
        "The model uses log(Yₜ) to transform the observations."
    )

def test_does_not_join_new_greek_variable_definition():
    text = (
        "Sₜ₋₁ = previous\n"
        "smoothed level\n"
        "α = smoothing\n"
        "parameter, 0 < α < 1"
    )

    result = TextSegmenter().segment(text)

    assert result == (
        "Sₜ₋₁ = previous smoothed level\n"
        "α = smoothing parameter, 0 < α < 1"
    )


def test_joins_hyphenated_word_without_adding_space():
    text = "Calculate each 4-\nMA."

    result = TextSegmenter().segment(text)

    assert result == "Calculate each 4-MA."
