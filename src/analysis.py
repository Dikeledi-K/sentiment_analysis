"""Dataset analysis utilities for reviews and CSV uploads."""

from __future__ import annotations

from typing import Any

import pandas as pd

from src.business_metrics import calculate_return_rate, flag_products_for_review
from src.issue_classifier import classify_issue
from src.predict import predict_text, summarize_predictions
from src.preprocessing import preprocess_text


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
    """Process a dataset by validating text rows and predicting sentiment."""
    if df is None or df.empty:
        return {
            "processed": 0,
            "skipped": 0,
            "invalid": 0,
            "dataframe": pd.DataFrame(),
            "summary": summarize_predictions(pd.DataFrame()),
        }

    if model is None:
        from src.predict import load_model
        model = load_model()

    if text_column not in df.columns:
        raise ValueError(f"Text column '{text_column}' was not found in the uploaded CSV file.")

    cleaned_df = df.copy()
    processed_rows = []
    processed_count = 0
    skipped_count = 0
    invalid_count = 0

    for _, row in cleaned_df.iterrows():
        raw_value = row.get(text_column)
        cleaned_text = preprocess_text(raw_value)

        if not cleaned_text:
            skipped_count += 1
            continue

        try:
            prediction = predict_text(cleaned_text, model=model)
            issue_category = classify_issue(cleaned_text)
            record = row.to_dict()
            record["text"] = cleaned_text
            record["predicted_sentiment"] = prediction["label"]
            record["confidence"] = prediction["confidence"]
            record["issue_category"] = issue_category
            if product_column and product_column in record:
                record["product_name"] = record.get(product_column)
            if category_column and category_column in record:
                record["category"] = record.get(category_column)
            processed_rows.append(record)
            processed_count += 1
        except Exception:
            invalid_count += 1

    result_df = pd.DataFrame(processed_rows)
    if date_column and date_column in result_df.columns:
        result_df[date_column] = pd.to_datetime(result_df[date_column], errors="coerce")
    if source_column and source_column in result_df.columns:
        result_df[source_column] = result_df[source_column].fillna("Unknown")
    if category_column and category_column in result_df.columns:
        result_df["category"] = result_df[category_column].fillna("Unspecified")
    if product_column and product_column in result_df.columns:
        result_df["product_name"] = result_df[product_column].fillna("Unknown product")
    if "issue_category" not in result_df.columns:
        result_df["issue_category"] = "Needs Review"

    summary = summarize_predictions(result_df)
    summary["issue_counts"] = result_df["issue_category"].fillna("Needs Review").value_counts().to_dict()
    summary["return_rate"] = calculate_return_rate(result_df)["percentage"] if {"units_sold", "units_returned"}.intersection(result_df.columns) else 0.0
    summary["flags"] = flag_products_for_review(result_df)

    return {
        "processed": processed_count,
        "skipped": skipped_count,
        "invalid": invalid_count,
        "dataframe": result_df,
        "summary": summary,
    }
