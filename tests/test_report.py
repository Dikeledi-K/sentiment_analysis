import pandas as pd

from src.report_generator import generate_insights_report


def test_generate_insights_report_contains_sections():
    df = pd.DataFrame(
        {
            "text": ["I love this product.", "The delivery was terrible.", "The service is okay."],
            "predicted_sentiment": ["positive", "negative", "neutral"],
        }
    )
    report = generate_insights_report(df)
    assert "# SentimentScope AI Insights Report" in report
    assert "Positive" in report
    assert "Negative" in report
    assert "Model limitations" in report
