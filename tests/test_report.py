import pandas as pd

from src.report_generator import generate_insights_report, generate_quality_report


def test_quality_report_marks_missing_return_data_as_not_provided():
    df = pd.DataFrame(
        {
            "text": ["The motor stopped working."],
            "predicted_sentiment": ["negative"],
            "issue_category": ["Faulty components"],
        }
    )
    report = generate_quality_report(df)
    assert "Return rate: not provided" in report


def test_generate_insights_report_contains_sections():
    df = pd.DataFrame(
        {
            "text": ["I love this product.", "The delivery was terrible.", "The service is okay."],
            "predicted_sentiment": ["positive", "negative", "neutral"],
        }
    )
    report = generate_insights_report(df)
    assert "# Quality & Returns Intelligence Report" in report
    assert "Positive" in report
    assert "Negative" in report
    assert "Model limitations" in report
