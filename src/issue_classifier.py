"""Transparent keyword-based issue classification for product quality analysis."""

from __future__ import annotations

from typing import Any

from src.preprocessing import preprocess_text

ISSUE_KEYWORDS = {
    "Manufacturing defects": [
        "defect",
        "faulty",
        "manufacturing",
        "malfunction",
        "broken",
        "flawed",
        "defective",
        "not working",
    ],
    "Damaged in transit": [
        "damaged",
        "crushed",
        "bent",
        "delivery damage",
        "arrived broken",
        "box damaged",
        "in transit",
    ],
    "Faulty components": [
        "battery",
        "charger",
        "motor",
        "screen",
        "button",
        "component",
        "wire",
        "connector",
        "sensor",
    ],
    "Poor durability": [
        "broke after",
        "fell apart",
        "not durable",
        "wear out",
        "cheap material",
        "deteriorated",
        "short lifespan",
        "worn out",
    ],
    "Missing parts or accessories": [
        "missing",
        "no parts",
        "accessories missing",
        "without cable",
        "missing screws",
        "missing manual",
    ],
    "Packaging problems": [
        "packaging",
        "poor packaging",
        "box damaged",
        "unpacked badly",
        "wrapping",
        "padding",
    ],
    "Sizing or fit issues": [
        "too small",
        "too large",
        "fits poorly",
        "size issue",
        "wrong size",
        "not fit",
        "tight fit",
    ],
    "Product performance problems": [
        "slow",
        "laggy",
        "not powerful",
        "poor performance",
        "does not work",
        "weak performance",
        "stopped working",
    ],
    "Refund or return dissatisfaction": [
        "refund",
        "return",
        "refunded",
        "return policy",
        "refund delay",
        "return process",
        "refund issue",
    ],
}


def classify_issue(text: Any) -> str:
    """Classify text into a transparent product issue category using keyword matching."""
    cleaned = preprocess_text(text)
    if not cleaned:
        return "Needs Review"

    for issue_name, keywords in ISSUE_KEYWORDS.items():
        if any(keyword in cleaned for keyword in keywords):
            return issue_name
    return "Other"


def classify_issue_series(values) -> list[str]:
    """Apply classification across a list-like input."""
    return [classify_issue(value) for value in values]
