"""Dataset analysis utilities for reviews and CSV uploads."""

from __future__ import annotations

from typing import Any

import pandas as pd

from src.business_metrics import calculate_product_return_rate, flag_products_for_review, has_unit_totals
from src.issue_classifier import classify_issue
from src.predict import load_model, predict_texts, summarize_predictions
from src.preprocessing import preprocess_text


COLUMN_HINTS = {
    "text": ["text", "review_text", "customer_feedback", "feedback", "comment", "comments", "review", "message"],
    "product": ["product_name", "product", "item_name", "item"],
    "category": ["category", "product_category", "item_category"],
    "date": ["date", "review_date", "feedback_date", "order_date", "created_at"],
}


def guess_column(df: pd.DataFrame, role: str) -> str | None:
    """Suggest the column that best fits a role ("text", "product", "category" or "date"), or None."""
    by_lower_name = {str(column).strip().lower(): column for column in df.columns}
    for hint in COLUMN_HINTS[role]:
        if hint in by_lower_name:
            return by_lower_name[hint]

    if role == "text":
        # Fall back to the column with the longest average string, which is usually the free-text field.
        lengths = {
            column: df[column].dropna().astype(str).str.len().mean()
            for column in df.select_dtypes(include=["object", "string"]).columns
        }
        lengths = {column: length for column, length in lengths.items() if pd.notna(length)}
        if lengths:
            return max(lengths, key=lengths.get)
    return None


def read_csv_file(source: Any) -> pd.DataFrame:
    """Read a CSV path or upload, falling back to Windows-1252 for files exported from Excel."""
    try:
        return pd.read_csv(source, encoding="utf-8-sig")
    except UnicodeDecodeError:
        if hasattr(source, "seek"):
            source.seek(0)
        return pd.read_csv(source, encoding="cp1252")


def summarize_sentiment_counts(df: pd.DataFrame) -> dict:
    """Return total counts and percentages for sentiment labels."""
    if "predicted_sentiment" not in df.columns:
        raise ValueError("The DataFrame does not contain a predicted_sentiment column.")
    return summarize_predictions(df)


def analyse_dataset(
    df: pd.DataFrame,
    text_column: str,
    model: Any | None = None,
    date_column: str | None = None,
    source_column: str | None = None,
    category_column: str | None = None,
    product_column: str | None = None,
) -> dict:
    """Clean text rows, skip empty ones, and add sentiment, confidence, and issue columns."""
    if df is None or df.empty:
        return {
            "processed": 0,
            "skipped": 0,
            "invalid": 0,
            "dataframe": pd.DataFrame(),
            "summary": summarize_predictions(pd.DataFrame()),
        }

    if text_column not in df.columns:
        raise ValueError(f"Text column '{text_column}' was not found in the uploaded CSV file.")

    if model is None:
        model = load_model()

    cleaned_text = df[text_column].map(preprocess_text)
    valid = cleaned_text != ""
    skipped_count = int((~valid).sum())

    result_df = df.loc[valid].copy()
    if text_column != "text" and "text" in result_df.columns:
        result_df = result_df.rename(columns={"text": "text_original"})
    # Keep the customer's original wording for tables and reports; the model gets the cleaned text.
    result_df["text"] = df.loc[valid, text_column].astype(str).str.strip()
    predictions = predict_texts(cleaned_text[valid].tolist(), model=model)
    result_df["predicted_sentiment"] = [prediction["label"] for prediction in predictions]
    result_df["confidence"] = [prediction["confidence"] for prediction in predictions]
    result_df["issue_category"] = result_df["text"].map(classify_issue)

    # Copy user-selected columns to the standard names the dashboard and report expect.
    # An uploaded column that already uses a standard name but was not selected for it is kept
    # under "<name>_original" rather than silently overwritten.
    for standard_name, selected in (
        ("date", date_column),
        ("source", source_column),
        ("category", category_column),
        ("product_name", product_column),
    ):
        if selected and selected in result_df.columns and selected != standard_name and standard_name in result_df.columns:
            result_df = result_df.rename(columns={standard_name: f"{standard_name}_original"})
    if date_column and date_column in result_df.columns:
        result_df["date"] = pd.to_datetime(result_df[date_column], errors="coerce")
    if source_column and source_column in result_df.columns:
        result_df["source"] = result_df[source_column].fillna("Unknown")
    if category_column and category_column in result_df.columns:
        result_df["category"] = result_df[category_column].fillna("Unspecified")
    if product_column and product_column in result_df.columns:
        result_df["product_name"] = result_df[product_column].fillna("Unknown product")
    result_df = result_df.reset_index(drop=True)

    summary = summarize_predictions(result_df)
    summary["issue_counts"] = result_df["issue_category"].value_counts().to_dict()
    summary["return_rate"] = calculate_product_return_rate(result_df)["percentage"] if has_unit_totals(result_df) else None
    summary["flags"] = flag_products_for_review(result_df)

    return {
        "processed": len(result_df),
        "skipped": skipped_count,
        "invalid": 0,
        "dataframe": result_df,
        "summary": summary,
    }
