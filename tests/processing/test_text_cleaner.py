
import pytest

from academic_agent.processing.text_cleaner import TextCleaner


@pytest.fixture
def cleaner():
    return TextCleaner()


def test_normalizes_windows_line_endings(cleaner):
    text = "First line\r\nSecond line\rThird line"

    assert cleaner.clean(text) == (
        "First line\nSecond line\nThird line"
    )


def test_collapses_repeated_spaces(cleaner):
    text = "Time    series\t analysis"

    assert cleaner.clean(text) == "Time series analysis"


def test_preserves_paragraph_boundaries(cleaner):
    text = "First paragraph\n\nSecond paragraph"

    assert cleaner.clean(text) == (
        "First paragraph\n\nSecond paragraph"
    )


def test_collapses_excessive_blank_lines(cleaner):
    text = "First paragraph\n\n\n\nSecond paragraph"

    assert cleaner.clean(text) == (
        "First paragraph\n\nSecond paragraph"
    )


def test_preserves_bullets_and_unicode(cleaner):
    text = "• Daily rainfall\n→ Forecast\nŶₜ₊ₕ|ₜ"

    assert cleaner.clean(text) == text


def test_normalizes_nonbreaking_spaces(cleaner):
    text = "Time\u00a0series analysis"

    assert cleaner.clean(text) == "Time series analysis"


def test_removes_leading_and_trailing_whitespace(cleaner):
    text = "  \n  Time series  \n  "

    assert cleaner.clean(text) == "Time series"


def test_rejects_non_string_input(cleaner):
    with pytest.raises(TypeError):
        cleaner.clean(None)
