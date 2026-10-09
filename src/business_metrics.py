"""Business-quality metrics for product returns and quality intelligence."""

from __future__ import annotations

import pandas as pd


def calculate_return_rate(df: pd.DataFrame, units_sold_col: str = "units_sold", units_returned_col: str = "units_returned") -> dict:
    """Calculate return rate using returned units divided by sold units."""
    if df.empty:
        return {"rate": 0.0, "percentage": 0.0, "returned_units": 0.0, "sold_units": 0.0, "records_used": 0}

    sold = pd.to_numeric(df.get(units_sold_col, pd.Series(0, index=df.index)), errors="coerce")
    returned = pd.to_numeric(df.get(units_returned_col, pd.Series(0, index=df.index)), errors="coerce")

    valid = sold.notna() & returned.notna() & (sold > 0)
    sold_total = float(sold[valid].sum())
    returned_total = float(returned[valid].sum())

    if sold_total == 0:
        return {"rate": 0.0, "percentage": 0.0, "returned_units": returned_total, "sold_units": sold_total, "records_used": int(valid.sum())}

    rate = returned_total / sold_total
    return {
        "rate": round(rate, 6),
        "percentage": round(rate * 100, 2),
        "returned_units": returned_total,
        "sold_units": sold_total,
        "records_used": int(valid.sum()),
    }


def summarize_quality_metrics(df: pd.DataFrame) -> dict:
    """Compute a compact quality summary from a dataset with sentiment and issue columns."""
    if df.empty:
        return {
            "total_records": 0,
            "negative_count": 0,
            "positive_count": 0,
            "neutral_count": 0,
            "issue_counts": {},
            "return_rate": 0.0,
        }

    sentiment = df["predicted_sentiment"].fillna("neutral").str.lower()
    issue_counts = df["issue_category"].fillna("Needs Review").astype(str).str.lower().value_counts().to_dict()
    summary = {
        "total_records": int(len(df)),
        "negative_count": int((sentiment == "negative").sum()),
        "positive_count": int((sentiment == "positive").sum()),
        "neutral_count": int((sentiment == "neutral").sum()),
        "issue_counts": {str(k): v for k, v in issue_counts.items()},
        "return_rate": calculate_return_rate(df)["percentage"],
    }
    return summary


def flag_products_for_review(df: pd.DataFrame, negative_threshold: float = 0.4, min_records: int = 2) -> list[dict]:
    """Flag products or categories needing review using negative-sentiment share and issue frequency."""
    if df.empty:
        return []

    if "product_name" not in df.columns and "category" not in df.columns:
        return []

    group_cols = []
    if "product_name" in df.columns:
        group_cols.append("product_name")
    if "category" in df.columns:
        group_cols.append("category")

    by_group = df.groupby(group_cols, dropna=False)
    results = []
    for group_key, group in by_group:
        if isinstance(group_key, tuple):
            product = group_key[0] if len(group_key) > 0 else "Unknown product"
            category = group_key[1] if len(group_key) > 1 else "Unspecified"
        else:
            product = group_key
            category = "Unspecified"
        total = len(group)
        if total < min_records:
            continue
        negative_share = (group["predicted_sentiment"].fillna("neutral").str.lower() == "negative").mean()
        issue_rate = (group["issue_category"].fillna("Needs Review").str.lower() != "other").mean()
        if negative_share >= negative_threshold or issue_rate >= negative_threshold:
            results.append(
                {
                    "product_name": product,
                    "category": category,
                    "records": total,
                    "negative_share": round(float(negative_share), 3),
                    "issue_rate": round(float(issue_rate), 3),
                    "reason": "Negative sentiment and recurring issue signals exceed the review threshold.",
                }
            )
    return results
