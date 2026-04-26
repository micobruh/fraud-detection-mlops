import json
from pathlib import Path


def test_required_artifacts_exist():
    candidate_paths = [
        Path("artifacts/selected_v_columns.json"),
        Path("artifacts/old_stuffs/selected_v_columns.json"),
    ]
    path = next((candidate for candidate in candidate_paths if candidate.exists()), None)
    assert path is not None

    with path.open() as f:
        data = json.load(f)

    assert isinstance(data, dict)
    assert isinstance(data["columns"], list)
    assert len(data["columns"]) > 0
    assert isinstance(data["available_v_columns"], list)
    assert isinstance(data["threshold"], float)
