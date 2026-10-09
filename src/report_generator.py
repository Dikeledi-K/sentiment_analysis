"""Generate a markdown insights report from analysed review data."""

from __future__ import annotations

from datetime import datetime

import pandas as pd


def _percentage(part: int, total: int) -> float:
    if total == 0:
        return 0.0
    return round((part / total) * 100, 2)


def _dominant_sentiment(df: pd.DataFrame) -> str:
    if df.empty or "predicted_sentiment" not in df.columns:
        return "No sentiment data available"
    counts = df["predicted_sentiment"].fillna("neutral").str.lower().value_counts()
    dominant = counts.idxmax() if not counts.empty else "neutral"
    return dominant.title()


def _representative_comments(df: pd.DataFrame, label: str, limit: int = 2) -> list[str]:
    if df.empty or "predicted_sentiment" not in df.columns or "text" not in df.columns:
        return []
    matches = df[df["predicted_sentiment"].str.lower() == label.lower()]
    return matches["text"].dropna().astype(str).head(limit).tolist()


def generate_insights_report(df: pd.DataFrame, analysis_date: str | None = None) -> str:
    """Construct a markdown report with actual observations from the dataset."""
    if df is None or df.empty:
        return "# SentimentScope AI Insights Report\n\nNo analysed data was available to generate a report."

    if analysis_date is None:
        analysis_date = datetime.now().strftime("%Y-%m-%d")

    total = len(df)
    sentiment_counts = df["predicted_sentiment"].fillna("neutral").str.lower().value_counts()
    positive = int(sentiment_counts.get("positive", 0))
    negative = int(sentiment_counts.get("negative", 0))
    neutral = int(sentiment_counts.get("neutral", 0))

    dominant = _dominant_sentiment(df)
    positive_comments = _representative_comments(df, "positive")
    negative_comments = _representative_comments(df, "negative")
    neutral_comments = _representative_comments(df, "neutral")

    negative_keywords = [
        "bad",
        "terrible",
        "slow",
        "poor",
        "expensive",
        "late",
        "broken",
        "worst",
        "disappointed",
        "delay",
        "issue",
    ]

    negative_theme_hits = []
    for comment in df[df["predicted_sentiment"].str.lower() == "negative"]["text"].dropna().astype(str):
        lowered = comment.lower()
        hits = [word for word in negative_keywords if word in lowered]
        if hits:
            negative_theme_hits.append(f"- {comment} (keywords: {', '.join(sorted(set(hits))[:3])})")

    positive_keywords = ["love", "great", "excellent", "easy", "perfect", "fast", "friendly", "quality", "helpful"]
    positive_theme_hits = []
    for comment in df[df["predicted_sentiment"].str.lower() == "positive"]["text"].dropna().astype(str):
        lowered = comment.lower()
        hits = [word for word in positive_keywords if word in lowered]
        if hits:
            positive_theme_hits.append(f"- {comment} (keywords: {', '.join(sorted(set(hits))[:3])})")

    report = f"# SentimentScope AI Insights Report\n\n"
    report += f"- Analysis date: {analysis_date}\n"
    report += f"- Dataset size: {total} reviewed records\n"
    report += f"- Successfully analysed: {total}\n"
    report += f"- Positive: {positive} ({_percentage(positive, total)}%)\n"
    report += f"- Negative: {negative} ({_percentage(negative, total)}%)\n"
    report += f"- Neutral: {neutral} ({_percentage(neutral, total)}%)\n"
    report += f"- Dominant sentiment: {dominant}\n\n"

    report += "## Key observations\n\n"
    report += f"- The dominant sentiment in this dataset is {dominant.lower()}.\n"
    report += f"- Positive comments account for {_percentage(positive, total)}% of the analysed records.\n"
    report += f"- Negative comments account for {_percentage(negative, total)}% of the analysed records.\n"
    report += "- These observations are based on model predictions applied to the dataset and should be reviewed alongside human judgement.\n\n"

    report += "## Representative comments\n\n"
    report += "### Positive\n"
    for comment in positive_comments or ["No positive comments were found."]:
        report += f"- {comment}\n"
    report += "\n### Negative\n"
    for comment in negative_comments or ["No negative comments were found."]:
        report += f"- {comment}\n"
    report += "\n### Neutral\n"
    for comment in neutral_comments or ["No neutral comments were found."]:
        report += f"- {comment}\n\n"

    report += "## Potential customer experience concerns\n\n"
    if negative_theme_hits:
        for item in negative_theme_hits[:5]:
            report += f"{item}\n"
    else:
        report += "- No clear negative keyword themes were identified in the analysed negative comments.\n"
    report += "\n"

    report += "## Positive feedback themes\n\n"
    if positive_theme_hits:
        for item in positive_theme_hits[:5]:
            report += f"{item}\n"
    else:
        report += "- No clear positive keyword themes were identified in the analysed positive comments.\n"
    report += "\n"

    report += "## Data quality limitations\n\n"
    report += "- The sample dataset is synthetic demonstration data and is not a real customer dataset.\n"
    report += "- Empty values, duplicate rows, or malformed records may reduce the number of usable inputs.\n"
    report += "- A small or imbalanced dataset can limit how well the model generalises beyond the provided examples.\n\n"

    report += "## Model limitations\n\n"
    report += "- This is a classic text classification model based on TF-IDF features and logistic regression.\n"
    report += "- The model may misclassify sarcasm, slang, spelling errors, mixed sentiment, or short statements.\n"
    report += "- Predictions should be treated as supportive signals, not as definitive statements about a person's intent or customer sentiment.\n\n"

    report += "## Recommended next steps\n\n"
    report += "- Review the most negative comments with a human analyst before making product or service decisions.\n"
    report += "- Collect more real labelled comments to improve model quality and reduce bias.\n"
    report += "- Expand the dataset to include more categories, time periods, and customer sources.\n"
    report += "- Re-train the model after adding more labelled examples or a larger public dataset.\n\n"

    return report
