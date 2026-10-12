"""Transparent keyword-based issue classification for product quality analysis."""

from __future__ import annotations

import re
from typing import Any

from src.preprocessing import preprocess_text

# Categories are checked in order, so specific phrases (missing parts, transit damage)
# come before broad terms such as "broken" or "battery".
ISSUE_KEYWORDS = {
    "Missing parts or accessories": [
        "missing",
        "absent",
        "no parts",
        "accessories missing",
        "without cable",
        "missing screws",
        "missing manual",
    ],
    "Damaged in transit": [
        "damaged",
        "crushed",
        "dented",
        "bent",
        "cracked",
        "scratched",
        "delivery damage",
        "arrived broken",
        "box damaged",
        "in transit",
    ],
    "Packaging problems": [
        "packaging",
        "packaged poorly",
        "poor packaging",
        "unpacked badly",
        "wrapping",
        "padding",
    ],
    "Sizing or fit issues": [
        "too small",
        "too large",
        "too big",
        "fits poorly",
        "size issue",
        "wrong size",
        "not fit",
        "tight fit",
    ],
    "Poor durability": [
        "broke after",
        "fell apart",
        "not durable",
        "wear out",
        "wears out",
        "cheap material",
        "deteriorated",
        "short lifespan",
        "worn out",
        "ripped",
        "fray",
        "frays",
        "sheds",
        "flimsy",
    ],
    "Manufacturing defects": [
        "defect",
        "defects",
        "faulty",
        "manufacturing",
        "malfunction",
        "broken",
        "flawed",
        "defective",
        "not working",
        "leaks",
        "leaking",
        "does not seal",
    ],
    "Faulty components": [
        "battery",
        "charger",
        "motor",
        "screen",
        "button",
        "buttons",
        "switch",
        "component",
        "wire",
        "connector",
        "sensor",
    ],
    "Product performance problems": [
        "slow",
        "slows down",
        "laggy",
        "not powerful",
        "poor performance",
        "does not work",
        "weak performance",
        "stopped working",
        "disconnecting",
        "disconnects",
        "cuts out",
        "flickers",
        "overheats",
    ],
    "Refund or return dissatisfaction": [
        "refund",
        "return",
        "returned",
        "returning",
        "refunded",
        "return policy",
        "refund delay",
        "return process",
        "refund issue",
    ],
}

# Whole-word patterns so short keywords such as "bent" do not match inside "absent".
_ISSUE_PATTERNS = {
    issue_name: re.compile(r"\b(?:" + "|".join(re.escape(keyword) for keyword in keywords) + r")\b")
    for issue_name, keywords in ISSUE_KEYWORDS.items()
}


def classify_issue(text: Any) -> str:
    """Classify text into a transparent product issue category using keyword matching."""
    cleaned = preprocess_text(text)
    if not cleaned:
        return "Needs Review"

    for issue_name, pattern in _ISSUE_PATTERNS.items():
        if pattern.search(cleaned):
            return issue_name
    return "Other"


def classify_issue_series(values) -> list[str]:
    """Apply classification across a list-like input."""
    return [classify_issue(value) for value in values]
