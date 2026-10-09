import pandas as pd

from src.analysis import analyse_dataset, summarize_sentiment_counts


def test_summarize_sentiment_counts():
    df = pd.DataFrame({
        "predicted_sentiment": ["positive", "negative", "positive", "neutral"],
    })
    summary = summarize_sentiment_counts(df)
    assert summary["positive"] == 2
    assert summary["negative"] == 1
    assert summary["neutral"] == 1
    assert summary["total"] == 4
    assert summary["positive_pct"] == 50.0


def test_analyse_dataset_handles_invalid_rows():
    df = pd.DataFrame(
        {
            "review": [
                "I love this product.",
                "",
                "The delivery was terrible.",
                None,
                "Price is okay.",
            ]
        }
    )
    result = analyse_dataset(df, text_column="review", model=None)
    assert result["processed"] == 3
    assert result["skipped"] == 2
    assert result["invalid"] == 0
