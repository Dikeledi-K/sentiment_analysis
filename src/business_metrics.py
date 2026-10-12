"""Business-quality metrics for product returns and quality intelligence."""

from __future__ import annotations

import pandas as pd

RETURNED_STATUSES = {"returned", "refunded", "return complete"}
NON_ISSUE_CATEGORIES = {"other", "needs review"}


def returned_mask(df: pd.DataFrame, status_col: str = "return_status") -> pd.Series:
    """Return a boolean mask of rows whose return status marks a completed return."""
    if status_col not in df.columns:
        return pd.Series(False, index=df.index)
    status = df[status_col].fillna("").astype(str).str.strip().str.lower()
    return status.isin(RETURNED_STATUSES)


def has_unit_totals(df: pd.DataFrame) -> bool:
    """True when both unit columns needed for a return rate are present."""
    return {"units_sold", "units_returned"}.issubset(df.columns)


def calculate_product_return_rate(df: pd.DataFrame) -> dict:
    """Return rate using one unit-total row per product, so repeated product totals are not double counted."""
    volume_data = df.drop_duplicates("product_name") if "product_name" in df.columns else df
    return calculate_return_rate(volume_data)


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

    group_cols = [col for col in ("product_name", "category") if col in df.columns]
    if not group_cols:
        return []

    results = []
    for group_key, group in df.groupby(group_cols, dropna=False):
        keys = dict(zip(group_cols, group_key if isinstance(group_key, tuple) else (group_key,)))
        total = len(group)
        if total < min_records:
            continue
        negative_share = (group["predicted_sentiment"].fillna("neutral").str.lower() == "negative").mean()
        issue_rate = (~group["issue_category"].fillna("Needs Review").str.lower().isin(NON_ISSUE_CATEGORIES)).mean()
        reasons = []
        if negative_share >= negative_threshold:
            reasons.append(f"{negative_share:.0%} negative feedback")
        if issue_rate >= negative_threshold:
            reasons.append(f"{issue_rate:.0%} of records report a classified issue")
        if reasons:
            results.append(
                {
                    "product_name": keys.get("product_name", "Not provided"),
                    "category": keys.get("category", "Unspecified"),
                    "records": total,
                    "negative_share": round(float(negative_share), 3),
                    "issue_rate": round(float(issue_rate), 3),
                    "reason": "; ".join(reasons) + f" (threshold {negative_threshold:.0%}).",
                }
            )
    return sorted(results, key=lambda row: (row["negative_share"], row["issue_rate"]), reverse=True)
