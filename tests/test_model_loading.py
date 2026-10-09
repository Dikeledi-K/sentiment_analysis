from pathlib import Path

import pytest

from src.predict import load_model


def test_load_model_raises_when_missing_file(tmp_path):
    fake_path = tmp_path / "missing_model.joblib"
    with pytest.raises(FileNotFoundError):
        load_model(fake_path)
