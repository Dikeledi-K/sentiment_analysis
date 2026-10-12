"""Prediction helpers for sentiment analysis."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from src.preprocessing import preprocess_text

ROOT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_PATH = ROOT_DIR / "models" / "sentiment_model.joblib"


def load_model(model_path: str | Path = DEFAULT_MODEL_PATH):
    """Load the trained sklearn pipeline from disk."""
    path = Path(model_path)
    if not path.exists():
        if path != DEFAULT_MODEL_PATH:
            raise FileNotFoundError(f"Model file not found at: {path}")
        from train_model import train_and_evaluate_model

        train_and_evaluate_model(model_path=path)
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
    return predict_texts([cleaned], model=model)[0]


def predict_texts(texts: list[str], model: Any | None = None) -> list[dict]:
    """Predict labels and confidence scores for many cleaned texts in one model call."""
    if not texts:
        return []

    model_to_use = load_model() if model is None else model
    if hasattr(model_to_use, "predict_proba"):
        probabilities = model_to_use.predict_proba(texts)
        classes = model_to_use.classes_
        predictions = [classes[row.argmax()] for row in probabilities]
        confidences = [float(row.max()) for row in probabilities]
    else:
        predictions = list(model_to_use.predict(texts))
        confidences = [0.0] * len(predictions)

    return [
        {
            "label": _normalize_label(prediction),
            "confidence": round(confidence, 4),
            "raw_prediction": prediction,
        }
        for prediction, confidence in zip(predictions, confidences)
    ]


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
