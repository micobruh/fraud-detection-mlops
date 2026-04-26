import json
from pathlib import Path


def test_required_artifacts_exist():
    path = Path("artifacts/selected_v_columns.json")
    assert path.exists()

    with path.open() as f:
        data = json.load(f)

    assert isinstance(data, dict)
    assert isinstance(data["columns"], list)
    assert len(data["columns"]) > 0
    assert isinstance(data["available_v_columns"], list)
    assert isinstance(data["threshold"], float)
