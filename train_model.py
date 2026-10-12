"""Train and evaluate the sentiment-analysis pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline

from src.preprocessing import clean_text

ROOT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_PATH = ROOT_DIR / "data" / "training_reviews.csv"
DEFAULT_MODEL_PATH = ROOT_DIR / "models" / "sentiment_model.joblib"
DEFAULT_METRICS_PATH = ROOT_DIR / "reports" / "model_metrics.json"
LABELS = ["positive", "negative", "neutral"]


def train_and_evaluate_model(
    data_path: str | Path = DEFAULT_DATA_PATH,
    model_path: str | Path = DEFAULT_MODEL_PATH,
    metrics_path: str | Path = DEFAULT_METRICS_PATH,
) -> dict:
    """Train a reusable TF-IDF + Logistic Regression classifier."""
    dataset_path = Path(data_path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Training dataset not found: {dataset_path}")

    df = pd.read_csv(dataset_path)
    required_columns = {"text", "sentiment"}
    missing = sorted(required_columns - set(df.columns))
    if missing:
        raise ValueError(f"Training CSV is missing required columns: {missing}")

    df = df.dropna(subset=["text", "sentiment"])
    text_data = df["text"].astype(str)
    labels = df["sentiment"].astype(str).str.strip().str.lower()
    unknown = sorted(set(labels) - set(LABELS))
    if unknown:
        raise ValueError(f"Unsupported sentiment labels {unknown}; expected one of {LABELS}.")

    X_train, X_test, y_train, y_test = train_test_split(
        text_data,
        labels,
        test_size=0.25,
        random_state=42,
        stratify=labels,
    )

    pipeline = build_pipeline()
    # Cross-validation on the training split gives a steadier estimate than one small test split.
    folds = StratifiedKFold(n_splits=min(5, int(y_train.value_counts().min())), shuffle=True, random_state=42)
    cv_scores = cross_val_score(build_pipeline(), X_train, y_train, cv=folds, scoring="f1_macro")

    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)

    metrics = {
        "accuracy": round(float(accuracy_score(y_test, predictions)), 4),
        "precision": round(float(precision_score(y_test, predictions, average="macro", zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, predictions, average="macro", zero_division=0)), 4),
        "f1": round(float(f1_score(y_test, predictions, average="macro", zero_division=0)), 4),
        "cv_f1_mean": round(float(cv_scores.mean()), 4),
        "cv_f1_std": round(float(cv_scores.std()), 4),
        "train_size": int(len(X_train)),
        "test_size": int(len(X_test)),
        "sklearn_version": sklearn.__version__,
        "classification_report": classification_report(y_test, predictions, digits=4, zero_division=0),
        "confusion_matrix": confusion_matrix(y_test, predictions, labels=LABELS).tolist(),
        "labels": LABELS,
    }

    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, model_path)

    metrics_path = Path(metrics_path)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    return metrics


def build_pipeline() -> Pipeline:
    """Create an untrained TF-IDF + Logistic Regression pipeline."""
    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    preprocessor=clean_text,
                    lowercase=False,
                    ngram_range=(1, 2),
                    min_df=1,
                    sublinear_tf=True,
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a sentiment analysis model.")
    parser.add_argument("--data", default=str(DEFAULT_DATA_PATH), help="Path to the labelled CSV dataset.")
    parser.add_argument("--model", default=str(DEFAULT_MODEL_PATH), help="Where to save the trained model.")
    parser.add_argument("--metrics", default=str(DEFAULT_METRICS_PATH), help="Where to save training metrics.")
    args = parser.parse_args()

    metrics = train_and_evaluate_model(args.data, args.model, args.metrics)
    print("Model training complete.")
    print(f"Accuracy: {metrics['accuracy']}")
    print(f"Precision: {metrics['precision']}")
    print(f"Recall: {metrics['recall']}")
    print(f"F1 score: {metrics['f1']}")
    print(f"Cross-validated F1: {metrics['cv_f1_mean']} (+/- {metrics['cv_f1_std']})")
