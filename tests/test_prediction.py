import pandas as pd

from src.predict import predict_text, summarize_predictions


def test_predict_text_returns_valid_labels():
    result = predict_text("I absolutely love this product and it works perfectly.")
    assert result["label"] in {"positive", "negative", "neutral"}
    assert 0 <= result["confidence"] <= 1


def test_summarize_predictions_counts_examples():
    df = pd.DataFrame(
        {
            "predicted_sentiment": ["positive", "negative", "neutral", "positive"],
            "confidence": [0.8, 0.7, 0.6, 0.9],
        }
    )
    summary = summarize_predictions(df)
    assert summary["total"] == 4
    assert summary["positive"] == 2
    assert summary["negative"] == 1
    assert summary["neutral"] == 1
    assert summary["positive_pct"] == 50.0
