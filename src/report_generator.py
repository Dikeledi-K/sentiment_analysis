"""Generate markdown business insights reports from product quality data."""

from __future__ import annotations

from datetime import datetime

import pandas as pd


def _safe_pct(part: int, total: int) -> float:
    if total == 0:
        return 0.0
    return round((part / total) * 100, 2)


def generate_quality_report(df: pd.DataFrame, analysis_date: str | None = None) -> str:
    """Generate a product quality insights report from actual analysis results."""
    if df is None or df.empty:
        return "# Product Quality Insights Report\n\nNo analysed data was available to generate a report."

    if analysis_date is None:
        analysis_date = datetime.now().strftime("%Y-%m-%d")

    total = len(df)
    sentiment_counts = df["predicted_sentiment"].fillna("neutral").str.lower().value_counts()
    positive = int(sentiment_counts.get("positive", 0))
    negative = int(sentiment_counts.get("negative", 0))
    neutral = int(sentiment_counts.get("neutral", 0))

    issue_counts = df["issue_category"].fillna("Needs Review").astype(str).value_counts()
    issue_top = issue_counts.head(5).to_dict()

    if "product_name" in df.columns:
        product_negative = (
            df.groupby("product_name")["predicted_sentiment"]
            .apply(lambda s: (s.fillna("neutral").str.lower() == "negative").mean() * 100)
            .sort_values(ascending=False)
            .head(5)
        )
    else:
        product_negative = pd.Series(dtype=float)

    if {"units_sold", "units_returned"}.intersection(df.columns):
        sold = pd.to_numeric(df.get("units_sold", pd.Series(0, index=df.index)), errors="coerce").fillna(0)
        returned = pd.to_numeric(df.get("units_returned", pd.Series(0, index=df.index)), errors="coerce").fillna(0)
        rate = (returned.sum() / sold.sum()) * 100 if sold.sum() else 0.0
    else:
        rate = 0.0

    negative_comments = df[df["predicted_sentiment"].fillna("neutral").str.lower() == "negative"]["text"].dropna().astype(str).head(3).tolist()
    positive_comments = df[df["predicted_sentiment"].fillna("neutral").str.lower() == "positive"]["text"].dropna().astype(str).head(3).tolist()

    report = "# Product Quality and Returns Intelligence Report\n\n"
    report += f"- Analysis date: {analysis_date}\n"
    report += f"- Total records analysed: {total}\n"
    report += f"- Positive: {positive} ({_safe_pct(positive, total)}%)\n"
    report += f"- Negative: {negative} ({_safe_pct(negative, total)}%)\n"
    report += f"- Neutral: {neutral} ({_safe_pct(neutral, total)}%)\n"
    report += f"- Return rate: {rate:.2f}% (returned units / sold units)\n\n"

    report += "## Issue overview\n"
    for issue_name, count in issue_top.items():
        report += f"- {issue_name}: {count} records\n"
    report += "\n"

    report += "## Product categories with highest negative feedback\n"
    if "category" in df.columns:
        category_negative = (
            df.groupby("category")["predicted_sentiment"]
            .apply(lambda s: (s.fillna("neutral").str.lower() == "negative").mean() * 100)
            .sort_values(ascending=False)
            .head(5)
        )
        for category, pct in category_negative.items():
            report += f"- {category}: {pct:.2f}% negative sentiment\n"
    else:
        report += "- No category column was available for this breakdown.\n"
    report += "\n"

    report += "## Representative feedback\n\n"
    report += "### Positive\n"
    for comment in positive_comments or ["No positive comments were available."]:
        report += f"- {comment}\n"
    report += "\n### Negative\n"
    for comment in negative_comments or ["No negative comments were available."]:
        report += f"- {comment}\n"
    report += "\n"

    report += "## Suggested quality investigation areas\n\n"
    if not product_negative.empty:
        for product, pct in product_negative.items():
            report += f"- {product}: {pct:.2f}% negative sentiment in the sampled records; review for recurring quality signals.\n"
    else:
        report += "- No clear product-level negative pattern was identified in the provided data.\n"
    report += "- Investigate repeated issue categories reported by customers before making a manufacturing conclusion.\n"
    report += "- Validate negative observations with a human analyst and supporting operational data.\n\n"

    report += "## Data limitations\n\n"
    report += "- This project uses demonstration data or locally uploaded records and should not be treated as a live operational dataset.\n"
    report += "- Keyword-based issue classification is a transparent baseline and may miss subtle or mixed problems.\n"
    report += "- Sentiment predictions are supportive signals, not proof of product fault or customer intent.\n"
    report += "- Personal identifiers should be removed before uploading files.\n\n"

    return report


def generate_insights_report(df: pd.DataFrame, analysis_date: str | None = None) -> str:
    """Backward-compatible summary report expected by the existing tests and app."""
    if df is None or df.empty:
        return "# Quality & Returns Intelligence Report\n\nNo analysed data was available to generate a report."

    if analysis_date is None:
        analysis_date = datetime.now().strftime("%Y-%m-%d")

    total = len(df)
    sentiment_counts = df["predicted_sentiment"].fillna("neutral").str.lower().value_counts()
    positive = int(sentiment_counts.get("positive", 0))
    negative = int(sentiment_counts.get("negative", 0))
    neutral = int(sentiment_counts.get("neutral", 0))

    report = "# Quality & Returns Intelligence Report\n\n"
    report += f"- Analysis date: {analysis_date}\n"
    report += f"- Dataset size: {total} reviewed records\n"
    report += f"- Positive: {positive} ({_safe_pct(positive, total)}%)\n"
    report += f"- Negative: {negative} ({_safe_pct(negative, total)}%)\n"
    report += f"- Neutral: {neutral} ({_safe_pct(neutral, total)}%)\n\n"
    report += "## Key observations\n\n"
    report += "- The dataset has been classified into positive, negative, and neutral sentiment groups.\n"
    report += "- The highest-volume issue category should be reviewed by operations and quality teams.\n"
    report += "- Model outputs are supportive signals and should be checked against real-world context.\n\n"
    report += "## Representative comments\n\n"
    if "text" in df.columns:
        for label in ["positive", "negative", "neutral"]:
            matches = df[df["predicted_sentiment"].fillna("neutral").str.lower() == label]["text"].dropna().astype(str).head(2)
            report += f"### {label.title()}\n"
            for comment in matches.tolist() or ["No comments available."]:
                report += f"- {comment}\n"
            report += "\n"
    report += "## Model limitations\n\n"
    report += "- The algorithm may misclassify sarcasm, slang, mixed sentiment, or short, incomplete messages.\n"
    report += "- Human judgement is still needed before making product decisions or customer escalations.\n"
    report += "- Dataset balance and quality directly affect model reliability.\n"

    return report
