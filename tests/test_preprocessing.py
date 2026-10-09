from src.preprocessing import clean_text, preprocess_text


def test_clean_text_removes_noise_and_normalizes():
    raw = "  I LOVE this product!!!  "
    result = clean_text(raw)
    assert "love" in result
    assert "product" in result
    assert "!" not in result


def test_preprocess_text_handles_empty_values():
    assert preprocess_text(None) == ""
    assert preprocess_text("") == ""
    assert preprocess_text("   ") == ""
