"""Text cleaning helpers for the SentimentScope AI project."""

from __future__ import annotations

import re
from typing import Any

import pandas as pd


def clean_text(text: Any) -> str:
    """Normalize review text for downstream modeling and analysis."""
    if text is None or pd.isna(text):
        return ""

    cleaned = str(text).strip()
    if not cleaned:
        return ""

    cleaned = cleaned.lower()
    cleaned = re.sub(r"https?://\S+|www\.\S+", " ", cleaned)
    cleaned = re.sub(r"[^a-z0-9\s]", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def preprocess_text(text: Any) -> str:
    """Public helper used by the application and training pipeline."""
    return clean_text(text)
