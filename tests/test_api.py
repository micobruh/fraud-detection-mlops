from types import SimpleNamespace

import numpy as np

from src.api.main import (
    ChampionState,
    PredictionRequest,
    create_app,
    health,
    model_info,
    predict,
)


class AmountPipeline:
    def predict_proba(self, X):
        probabilities = X["TransactionAmt"].to_numpy(dtype=float) / 100
        return np.column_stack([1 - probabilities, probabilities])


def request_for_app(app):
    return SimpleNamespace(app=app)


def test_create_app_registers_expected_routes_without_startup_loader():
    app = create_app(load_model_on_startup=False)

    paths = {route.path for route in app.routes}

    assert "/" in paths
    assert "/health" in paths
    assert "/model-info" in paths
    assert "/predict" in paths


def test_health_reports_model_not_loaded_without_champion_state():
    app = create_app(load_model_on_startup=False)

    assert health(request_for_app(app)) == {"status": "model_not_loaded"}


def test_predict_scores_records_from_loaded_champion_state():
    state = ChampionState(
        model=AmountPipeline(),
        metadata={
            "model_uri": "models:/test@champion",
            "classification_threshold": 0.5,
            "streaming_batch_size": None,
        },
        classification_threshold=0.5,
    )

    response = predict(
        PredictionRequest(
            records=[
                {"TransactionID": 101, "TransactionAmt": 25.0},
                {"TransactionID": 102, "TransactionAmt": 75.0},
            ]
        ),
        state=state,
    )

    assert response.model_dump() == {
        "model_uri": "models:/test@champion",
        "classification_threshold": 0.5,
        "predictions": [
            {"transaction_id": 101, "fraud_score": 0.25, "is_fraud": 0},
            {"transaction_id": 102, "fraud_score": 0.75, "is_fraud": 1},
        ],
    }


def test_model_info_returns_champion_metadata():
    state = ChampionState(
        model=AmountPipeline(),
        metadata={
            "model_uri": "models:/test@champion",
            "registered_model_name": "TestModel",
            "registered_model_version": 7,
            "feature_set_name": "base",
            "test_mode": "offline_full_batch",
            "streaming_batch_size": None,
        },
        classification_threshold=0.5,
    )

    assert model_info(state) == {
        "model_uri": "models:/test@champion",
        "registered_model_name": "TestModel",
        "registered_model_version": 7,
        "feature_set_name": "base",
        "classification_threshold": 0.5,
        "test_mode": "offline_full_batch",
        "streaming_batch_size": None,
    }
