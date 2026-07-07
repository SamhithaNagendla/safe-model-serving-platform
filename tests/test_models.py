from pathlib import Path

import pytest

from model_serving.models import LogisticModel


def test_model_load_and_dimension_validation(tmp_path: Path) -> None:
    path = tmp_path / "model.json"
    path.write_text(
        '{"version":"x","weights":[1,2],"bias":0,"training":{"samples":10}}',
        encoding="utf-8",
    )
    model = LogisticModel.load(path)
    assert 0 < model.predict([1, 1]) < 1
    assert model.metadata["samples"] == 10
    with pytest.raises(ValueError, match="expected 2"):
        model.predict([1])
