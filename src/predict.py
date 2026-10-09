"""Prediction helpers for sentiment analysis."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from src.preprocessing import preprocess_text
from train_model import train_and_evaluate_model

ROOT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_PATH = ROOT_DIR / "models" / "sentiment_model.joblib"


def load_model(model_path: str | Path = DEFAULT_MODEL_PATH):
    """Load the trained sklearn pipeline from disk."""
    path = Path(model_path)
    if not path.exists():
        if path != DEFAULT_MODEL_PATH:
            raise FileNotFoundError(f"Model file not found at: {path}")
        fallback_data = Path(__file__).resolve().parent.parent / "data" / "product_quality_demo.csv"
        fallback_metrics = Path(__file__).resolve().parent.parent / "reports" / "model_metrics.json"
        train_and_evaluate_model(fallback_data, path, fallback_metrics)
    return joblib.load(path)


def _normalize_label(label: Any) -> str:
    if label is None:
        return "neutral"
    label_text = str(label).strip().lower()
    mapping = {
        "positive": "positive",
        "neg": "negative",
        "negative": "negative",
        "neutral": "neutral",
        "pos": "positive",
    }
    return mapping.get(label_text, label_text)


def predict_text(text: Any, model: Any | None = None) -> dict:
    """Return a prediction label and confidence score for one text record."""
    cleaned = preprocess_text(text)
    if not cleaned:
        raise ValueError("Text is empty. Please enter a valid review before analysing.")

    model_to_use = model or load_model()
    prediction = model_to_use.predict([cleaned])[0]
    label = _normalize_label(prediction)

    probabilities = None
    if hasattr(model_to_use, "predict_proba"):
        probabilities = model_to_use.predict_proba([cleaned])[0]
        confidence = float(max(probabilities)) if len(probabilities) else 0.0
    else:
        confidence = 0.0

    return {
        "label": label,
        "confidence": round(confidence, 4),
        "raw_prediction": prediction,
    }


def summarize_predictions(df: pd.DataFrame) -> dict:
    """Calculate counts and percentages from a sentiment DataFrame."""
    if df.empty:
        return {
            "total": 0,
            "positive": 0,
            "negative": 0,
            "neutral": 0,
            "positive_pct": 0.0,
            "negative_pct": 0.0,
            "neutral_pct": 0.0,
        }

    sentiment_series = df["predicted_sentiment"].fillna("neutral").astype(str).str.lower()
    total = len(sentiment_series)
    positive = int((sentiment_series == "positive").sum())
    negative = int((sentiment_series == "negative").sum())
    neutral = int((sentiment_series == "neutral").sum())

    return {
        "total": total,
        "positive": positive,
        "negative": negative,
        "neutral": neutral,
        "positive_pct": round((positive / total) * 100, 2) if total else 0.0,
        "negative_pct": round((negative / total) * 100, 2) if total else 0.0,
        "neutral_pct": round((neutral / total) * 100, 2) if total else 0.0,
    }
