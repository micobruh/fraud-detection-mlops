import logging
import ast
from pathlib import Path
from time import perf_counter
from typing import Any

import mlflow
import pandas as pd
from pydantic import BaseModel, Field

from ..data import (
    load_interim_data,
    temporal_train_val_test_split
)
from ..models import (
    build_pipeline_from_config
)
from ..utils import (
    MLFLOW_VALIDATION_EXPERIMENT_NAME,
    resolve_project_path,
)


logger = logging.getLogger(__name__)


class ValidationConfig(BaseModel):
    data_dir: str | Path
    comparison_path: str | Path = "artifacts/model_comparison_incremental.csv"
    top_k: int = Field(default=3, ge=1)
    min_k: int = Field(default=3, ge=1)
    max_per_model_sampler: int = Field(default=1, ge=1)
    save_incremental: bool = True
    v_columns_cache_path: str | Path | None = "artifacts/selected_v_columns.json"


class CandidateRow(BaseModel):
    rank: int = Field(ge=1)
    selection_rank: int | None = Field(default=None, ge=1)
    model_name: str
    feature_set_name: str
    search_smote: bool
    best_cv_score: float = Field(ge=0.0, le=1.0)
    best_cv_roc_auc: float = Field(ge=0.0, le=1.0)
    best_cv_average_precision: float = Field(ge=0.0, le=1.0)
    best_cv_f1: float = Field(ge=0.0, le=1.0)
    best_cv_accuracy: float = Field(ge=0.0, le=1.0)
    best_cv_recall: float = Field(ge=0.0, le=1.0)
    best_cv_precision: float = Field(ge=0.0, le=1.0)
    num_cv_folds_used: int = Field(ge=1)
    best_params: dict[str, Any]
    best_sampler: str


class CandidateFitResult(BaseModel):
    rank: int = Field(ge=1)
    selection_rank: int = Field(ge=1)
    model_name: str
    feature_set_name: str
    best_sampler: str
    best_cv_score: float = Field(ge=0.0, le=1.0)
    fit_elapsed_seconds: float = Field(ge=0.0)


def parse_best_params(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if pd.isna(value):
        return {}
    return ast.literal_eval(value)


def select_top_candidates(
    comparison_path: str | Path = "artifacts/model_comparison_incremental.csv",
    top_k: int = 3,
    min_k: int = 3,
    max_per_model_sampler: int = 1,
) -> list[CandidateRow]:
    df_models = pd.read_csv(resolve_project_path(comparison_path))
    df_models["best_params"] = df_models["best_params"].apply(parse_best_params)
    df_models = df_models.sort_values(
        by=["best_cv_score"],
        ascending=False,
    ).reset_index(drop=True)

    selected = []
    selected_indices = set()
    seen_model_sampler = {}

    for idx, row in df_models.iterrows():
        key = (row["model_name"], row["feature_set_name"], row["best_sampler"])

        if seen_model_sampler.get(key, 0) >= max_per_model_sampler:
            continue

        selected.append(row)
        selected_indices.add(idx)
        seen_model_sampler[key] = seen_model_sampler.get(key, 0) + 1

        if len(selected) >= top_k:
            break

    df_selected_models = pd.DataFrame(selected)

    if len(df_selected_models) < min_k:
        # Fallback: fill remaining slots by pure score, avoiding exact duplicates.
        for idx, row in df_models.iterrows():
            if idx in selected_indices:
                continue
            df_selected_models = pd.concat([df_selected_models, row.to_frame().T], ignore_index=True)
            selected_indices.add(idx)
            if len(df_selected_models) >= min_k:
                break

    df_selected_models = df_selected_models.reset_index(drop=True)
    df_selected_models.insert(0, "selection_rank", df_selected_models.index + 1)

    return [
        CandidateRow.model_validate(record)
        for record in df_selected_models.to_dict(orient="records")
    ]


def streaming_validation(
    data_dir: str, 
    save_incremental: bool = True,
    v_columns_cache_path: str | None = "artifacts/selected_v_columns.json",
    top_k: int = 3,
    min_k: int = 3,    
) -> None:
    validation_config = ValidationConfig(
        data_dir=data_dir,
        save_incremental=save_incremental,
        v_columns_cache_path=v_columns_cache_path,
        top_k=top_k,
        min_k=min_k,        
    )   
    candidates = select_top_candidates(
        comparison_path=validation_config.comparison_path,
        top_k=validation_config.top_k,
        min_k=validation_config.min_k,
        max_per_model_sampler=validation_config.max_per_model_sampler,
    )

    mlflow.set_experiment(MLFLOW_VALIDATION_EXPERIMENT_NAME)

    with mlflow.start_run(run_name="online-streaming-validation"):  
        df = load_interim_data(validation_config.data_dir)
        split_cache = {}
        fit_results: list[CandidateFitResult] = []

        for candidate in candidates:
            feature_set = candidate.feature_set_name
            model_name = candidate.model_name
            best_params = candidate.best_params

            logger.info("Preparing feature set: %s", feature_set)

            if feature_set not in split_cache:
                split_cache[feature_set] = temporal_train_val_test_split(
                    df,
                    feature_set,
                    v_columns_cache_path=validation_config.v_columns_cache_path,
                )

            _, selected_columns, X_train, X_val, _, y_train, y_val, _ = split_cache[feature_set]

            pipeline = build_pipeline_from_config(
                selected_columns,
                feature_set,
                model_name,
                best_params,
            )
            
            start_time = perf_counter()
            pipeline.fit(X_train, y_train)
            fit_elapsed_seconds = perf_counter() - start_time

            selection_rank = candidate.selection_rank or candidate.rank
            fit_result = CandidateFitResult(
                rank=candidate.rank,
                selection_rank=selection_rank,
                model_name=candidate.model_name,
                feature_set_name=candidate.feature_set_name,
                best_sampler=candidate.best_sampler,
                best_cv_score=candidate.best_cv_score,
                fit_elapsed_seconds=fit_elapsed_seconds,
            )
            fit_results.append(fit_result)

            mlflow.log_metric(
                f"candidate_{fit_result.selection_rank}_final_fit_seconds",
                fit_result.fit_elapsed_seconds,
            )        
            mlflow.sklearn.log_model(
                pipeline,
                artifact_path=f"candidate_{fit_result.selection_rank}",
            )                           

        mlflow.log_dict(
            {
                "candidates": [
                    fit_result.model_dump()
                    for fit_result in fit_results
                ]
            },
            artifact_file="candidate_fit_results.json",
        )
