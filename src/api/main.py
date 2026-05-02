from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

import pandas as pd
from fastapi import APIRouter, Depends, FastAPI, HTTPException, Request, status
from pydantic import BaseModel

from ..models import (
    load_champion_metadata,
    load_model_from_mlflow,
)
from ..pipelines import (
    champion_classification_threshold,
    score_champion_predictions,
)


DEFAULT_METADATA_PATH = "artifacts/champion_model.json"


class PredictionRequest(BaseModel):
    records: list[dict[str, Any]]


class PredictionRecord(BaseModel):
    transaction_id: Any | None
    fraud_score: float
    is_fraud: int


class PredictionResponse(BaseModel):
    model_uri: str
    classification_threshold: float
    predictions: list[PredictionRecord]


@dataclass(frozen=True)
class ChampionState:
    model: Any
    metadata: dict[str, Any]
    classification_threshold: float


def load_champion_state(metadata_path: str = DEFAULT_METADATA_PATH) -> ChampionState:
    metadata = load_champion_metadata(metadata_path)
    model = load_model_from_mlflow(metadata_path=metadata_path)
    classification_threshold = champion_classification_threshold(metadata)
    return ChampionState(
        model=model,
        metadata=metadata,
        classification_threshold=classification_threshold,
    )


def build_lifespan(metadata_path: str):
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.champion = load_champion_state(metadata_path)
        yield

    return lifespan


def get_champion_state(request: Request) -> ChampionState:
    champion = getattr(request.app.state, "champion", None)
    if champion is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Champion model is not loaded.",
        )
    return champion


router = APIRouter()


@router.get("/")
@router.get("/health")
def health(request: Request) -> dict[str, str]:
    if not hasattr(request.app.state, "champion"):
        return {"status": "model_not_loaded"}
    return {"status": "ok"}


@router.get("/model-info")
def model_info(state: ChampionState = Depends(get_champion_state)) -> dict[str, Any]:
    metadata = state.metadata
    return {
        "model_uri": metadata["model_uri"],
        "registered_model_name": metadata.get("registered_model_name"),
        "registered_model_version": metadata.get("registered_model_version"),
        "feature_set_name": metadata.get("feature_set_name"),
        "classification_threshold": state.classification_threshold,
        "test_mode": metadata.get("test_mode"),
        "streaming_batch_size": metadata.get("streaming_batch_size"),
    }


@router.post("/predict", response_model=PredictionResponse)
def predict(
    request: PredictionRequest,
    state: ChampionState = Depends(get_champion_state),
) -> PredictionResponse:
    if not request.records:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="At least one prediction record is required.",
        )

    X = pd.DataFrame(request.records)
    X_scored, y_scores, y_preds = score_champion_predictions(
        state.model,
        X,
        state.metadata,
    )

    predictions = [
        PredictionRecord(
            transaction_id=row.get("TransactionID"),
            fraud_score=float(score),
            is_fraud=int(pred),
        )
        for row, score, pred in zip(
            X_scored.to_dict(orient="records"),
            y_scores,
            y_preds,
        )
    ]

    return PredictionResponse(
        model_uri=state.metadata["model_uri"],
        classification_threshold=state.classification_threshold,
        predictions=predictions,
    )


def create_app(
    metadata_path: str = DEFAULT_METADATA_PATH,
    load_model_on_startup: bool = True,
) -> FastAPI:
    app = FastAPI(
        title="Fraud Detection AI",
        description="Classifies streaming transactions instantly.",
        version="1.0.0",
        lifespan=build_lifespan(metadata_path) if load_model_on_startup else None,
    )
    app.include_router(router)
    return app


app = create_app()
