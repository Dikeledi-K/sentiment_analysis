import json

import pytest

from train_model import DEFAULT_DATA_PATH, train_and_evaluate_model


def test_training_writes_model_and_metrics(tmp_path):
    model_path = tmp_path / "model.joblib"
    metrics_path = tmp_path / "metrics.json"
    metrics = train_and_evaluate_model(DEFAULT_DATA_PATH, model_path, metrics_path)

    assert model_path.exists()
    saved = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert saved["labels"] == ["positive", "negative", "neutral"]
    assert 0 <= metrics["cv_f1_mean"] <= 1
    # Guard against regressing to the near-random model trained on the 24-row demo file.
    assert metrics["f1"] > 0.5


def test_training_rejects_unknown_labels(tmp_path):
    data_path = tmp_path / "bad.csv"
    data_path.write_text("text,sentiment\ngood,positive\nbad,angry\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Unsupported sentiment labels"):
        train_and_evaluate_model(data_path, tmp_path / "m.joblib", tmp_path / "m.json")
