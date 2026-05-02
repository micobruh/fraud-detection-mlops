from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.pipelines import batch_scoring_flow as flow


def test_champion_classification_threshold_falls_back_for_null_metadata():
    assert flow.champion_classification_threshold({"classification_threshold": None}) == 0.5
    assert flow.champion_classification_threshold({}) == 0.5


def test_champion_v_selection_threshold_uses_metadata_or_training_default():
    assert flow.champion_v_selection_threshold({"v_selection_threshold": 0.7}) == 0.7
    assert flow.champion_v_selection_threshold({"threshold": 0.6}) == 0.6
    assert flow.champion_v_selection_threshold({}) == flow.DEFAULT_V_SELECTION_THRESHOLD


def test_predict_champion_test_input_path_uses_v_selection_threshold(monkeypatch, tmp_path):
    metadata = {
        "classification_threshold": 0.4,
        "feature_set_name": "base_selected_v",
        "model_uri": "models:/test@champion",
    }
    input_df = pd.DataFrame(
        {
            "TransactionID": [101, 102],
            "TransactionDT": [2, 1],
            "feature": [0.2, 0.8],
        }
    )
    calls = {}

    monkeypatch.setattr(flow, "load_champion_metadata", lambda metadata_path: metadata)
    monkeypatch.setattr(flow, "load_model_from_mlflow", lambda metadata_path: object())
    monkeypatch.setattr(flow, "load_prediction_input_data", lambda input_path: input_df)

    def fake_determine_columns(df, feature_set_name, threshold, cache_path):
        calls["feature_set_name"] = feature_set_name
        calls["threshold"] = threshold
        calls["cache_path"] = cache_path
        return ["feature"]

    monkeypatch.setattr(flow, "determine_columns", fake_determine_columns)
    monkeypatch.setattr(
        flow,
        "score_champion_predictions",
        lambda champion_model, X_test, metadata: (
            X_test,
            np.array([0.2, 0.8]),
            np.array([0, 1]),
        ),
    )

    def fake_save_test_prediction_outputs(
        X_test,
        y_scores,
        y_preds,
        metadata,
        submission_path,
        production_path,
    ):
        return Path(submission_path), Path(production_path)

    monkeypatch.setattr(flow, "save_test_prediction_outputs", fake_save_test_prediction_outputs)

    result = flow.predict_champion_test(
        data_dir="unused",
        input_path="unused.csv",
        prediction_path=tmp_path / "predictions.csv",
        production_prediction_path=tmp_path / "production.csv",
        v_columns_cache_path=tmp_path / "selected_v_columns.json",
    )

    assert calls == {
        "feature_set_name": "base_selected_v",
        "threshold": flow.DEFAULT_V_SELECTION_THRESHOLD,
        "cache_path": tmp_path / "selected_v_columns.json",
    }
    assert result["classification_threshold"] == 0.4


class IdScoreModel:
    def predict_proba(self, X):
        probabilities = X["TransactionID"].to_numpy(dtype=float) / 100
        return np.column_stack([1 - probabilities, probabilities])


class IdScorePipeline:
    steps = [("model", IdScoreModel())]


def test_streaming_champion_prediction_outputs_keep_ids_aligned_after_sort(tmp_path):
    X_test = pd.DataFrame(
        {
            "TransactionID": [90, 10],
            "TransactionDT": [2, 1],
        }
    )
    metadata = {
        "alias": "champion",
        "classification_threshold": 0.5,
        "feature_set_name": "base",
        "feature_state_policy": "not_applicable",
        "model_name": "xgboost",
        "model_uri": "models:/test@champion",
        "registered_model_name": "TestModel",
        "registered_model_version": 1,
        "streaming_batch_size": 1,
        "test_mode": "streaming",
        "training_data_scope": "train_plus_validation",
    }

    X_scored, y_scores, y_preds = flow.score_champion_predictions(
        IdScorePipeline(),
        X_test,
        metadata,
    )
    submission_path, production_path = flow.save_test_prediction_outputs(
        X_test=X_scored,
        y_scores=y_scores,
        y_preds=y_preds,
        metadata=metadata,
        submission_path=tmp_path / "submission.csv",
        production_path=tmp_path / "production.csv",
    )

    submission = pd.read_csv(submission_path)
    production = pd.read_csv(production_path)

    assert submission["TransactionID"].tolist() == [10, 90]
    assert submission["isFraud"].tolist() == [0, 1]
    assert production["TransactionID"].tolist() == [10, 90]
    assert production["fraud_score"].tolist() == pytest.approx([0.1, 0.9])


def test_predict_champion_test_input_path_raises_for_missing_required_columns(monkeypatch):
    metadata = {
        "classification_threshold": 0.4,
        "feature_set_name": "base",
        "model_uri": "models:/test@champion",
    }
    input_df = pd.DataFrame({"TransactionID": [1], "TransactionDT": [1]})

    monkeypatch.setattr(flow, "load_champion_metadata", lambda metadata_path: metadata)
    monkeypatch.setattr(flow, "load_model_from_mlflow", lambda metadata_path: object())
    monkeypatch.setattr(flow, "load_prediction_input_data", lambda input_path: input_df)
    monkeypatch.setattr(
        flow,
        "determine_columns",
        lambda df, feature_set_name, threshold, cache_path: ["missing_feature"],
    )

    with pytest.raises(KeyError, match="missing_feature"):
        flow.predict_champion_test(data_dir="unused", input_path="unused.csv")
