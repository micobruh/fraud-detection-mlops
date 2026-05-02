import logging
import ast
import json
from copy import deepcopy
from pathlib import Path
from time import perf_counter
from typing import Any

import mlflow
import pandas as pd
from pydantic import BaseModel, Field, field_validator

from ..data import (
    load_interim_data,
    temporal_train_val_test_split
)
from ..models import (
    build_pipeline_from_config,
    streaming_predict_scores,
    offline_predict_scores,
    sort_y_labels,
    compute_classification_metric,
    select_threshold_by_f1,
)
from ..utils import (
    FEATURE_SETS,
    MLFLOW_VALIDATION_EXPERIMENT_NAME,
    MLFLOW_TEST_EXPERIMENT_NAME,
    resolve_project_path,
)


logger = logging.getLogger(__name__)


VALIDATION_RESULT_COLUMNS = [
    "validation_rank",
    "rank",
    "selection_rank",
    "model_name",
    "feature_set_name",
    "search_smote",
    "best_sampler",
    "best_cv_score",
    "best_params",
    "model_uri",
    "validation_mode",
    "feature_state_policy",
    "streaming_batch_size",
    "roc_auc",
    "f1",
    "recall",
    "precision",
    "average_precision",
    "accuracy",
    "classification_threshold",
    "threshold_selection_metric",
    "threshold_validation_f1",
    "threshold_validation_precision",
    "threshold_validation_recall",
    "fit_elapsed_seconds",
    "predict_elapsed_seconds",
]

TEST_RESULT_COLUMNS = [
    "test_rank",
    "training_data_scope",
    "selected_validation_rank",
    "selected_validation_roc_auc",
    "selected_feature_state_policy",
    "model_name",
    "feature_set_name",
    "search_smote",
    "best_sampler",
    "best_params",
    "test_mode",
    "streaming_batch_size",
    "classification_threshold",
    "roc_auc",
    "f1",
    "recall",
    "precision",
    "average_precision",
    "accuracy",
    "fit_elapsed_seconds",
    "predict_elapsed_seconds",
]


class ValidationConfig(BaseModel):
    data_dir: str | Path
    comparison_path: str | Path = "artifacts/model_comparison_incremental.csv"
    validation_results_path: str | Path = "artifacts/model_validation_incremental.csv"
    selected_model_path: str | Path = "artifacts/selected_model.json"
    top_k: int = Field(default=3, ge=1)
    min_k: int = Field(default=3, ge=1)
    max_per_model_sampler: int = Field(default=1, ge=1)
    v_columns_cache_path: str | Path | None = "artifacts/selected_v_columns.json"
    streaming_batch_size: int | None = Field(default=1)

    @field_validator("streaming_batch_size")
    @classmethod
    def validate_streaming_batch_size(cls, value: int | None) -> int | None:
        if value is not None and value < 1:
            raise ValueError("streaming_batch_size must be None or >= 1")
        return value    
    

class TestConfig(BaseModel):
    data_dir: str | Path
    comparison_path: str | Path = "artifacts/model_validation_incremental.csv"
    test_results_path: str | Path = "artifacts/model_test_incremental.csv"
    test_performance_path: str | Path = "artifacts/model_test_performance.csv"
    max_per_model_sampler: int = Field(default=1, ge=1)
    v_columns_cache_path: str | Path | None = "artifacts/selected_v_columns.json"
    streaming_batch_size: int | None = Field(default=1)

    @field_validator("streaming_batch_size")
    @classmethod
    def validate_streaming_batch_size(cls, value: int | None) -> int | None:
        if value is not None and value < 1:
            raise ValueError("streaming_batch_size must be None or >= 1")
        return value       


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


class CandidateValidationResult(BaseModel):
    rank: int = Field(ge=1)
    selection_rank: int = Field(ge=1)
    model_name: str
    feature_set_name: str
    search_smote: bool
    best_sampler: str
    best_cv_score: float = Field(ge=0.0, le=1.0)
    best_params: dict[str, Any]
    model_uri: str
    validation_mode: str
    feature_state_policy: str
    streaming_batch_size: int | None
    roc_auc: float
    f1: float
    recall: float
    precision: float
    average_precision: float
    accuracy: float
    fit_elapsed_seconds: float = Field(ge=0.0)
    predict_elapsed_seconds: float = Field(ge=0.0)


class TestResult(BaseModel):
    training_data_scope: str
    model_name: str
    feature_set_name: str
    search_smote: bool
    best_sampler: str
    selected_validation_rank: int = Field(ge=1)
    selected_validation_roc_auc: float
    best_params: dict[str, Any]
    selected_feature_state_policy: str
    test_mode: str
    streaming_batch_size: int | None
    classification_threshold: float
    roc_auc: float
    f1: float
    recall: float
    precision: float
    average_precision: float
    accuracy: float
    fit_elapsed_seconds: float = Field(ge=0.0)
    predict_elapsed_seconds: float = Field(ge=0.0)    


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


class ValidatedCandidateRow(BaseModel):
    validation_rank: int = Field(ge=1)
    model_name: str
    feature_set_name: str
    search_smote: bool
    best_sampler: str
    best_params: dict[str, Any]
    model_uri: str
    roc_auc: float
    feature_state_policy: str
    classification_threshold: float = 0.5

    @field_validator("classification_threshold", mode="before")
    @classmethod
    def default_missing_threshold(cls, value: Any) -> float:
        if value is None or pd.isna(value):
            return 0.5
        return float(value)


def select_best_validated_candidate(
    comparison_path: str | Path = "artifacts/model_validation_incremental.csv",
) -> ValidatedCandidateRow:
    df_models = pd.read_csv(resolve_project_path(comparison_path))
    df_models["best_params"] = df_models["best_params"].apply(parse_best_params)
    df_models = df_models.sort_values(
        by=["roc_auc"],
        ascending=False,
        na_position="last",
    ).reset_index(drop=True)

    if df_models.empty:
        raise ValueError(f"No validated candidates found in {comparison_path}.")

    return ValidatedCandidateRow.model_validate(df_models.iloc[0].to_dict())


def save_validation_results(
    results_df: pd.DataFrame,
    output_path: str | Path = "artifacts/model_validation_incremental.csv",
) -> None:
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    ordered_columns = [
        col for col in VALIDATION_RESULT_COLUMNS if col in results_df.columns
    ]
    remaining_columns = [
        col for col in results_df.columns if col not in ordered_columns
    ]
    results_df = results_df[ordered_columns + remaining_columns]
    results_df.to_csv(output_file, index=False)


def save_selected_model(
    results_df: pd.DataFrame,
    output_path: str | Path = "artifacts/selected_model.json",
) -> dict[str, Any]:
    if results_df.empty:
        raise ValueError("Cannot save selected model from empty validation results.")

    selected = (
        results_df
        .sort_values("roc_auc", ascending=False, na_position="last")
        .iloc[0]
        .to_dict()
    )
    selected_model = {
        "model_name": selected["model_name"],
        "feature_set_name": selected["feature_set_name"],
        "best_params": parse_best_params(selected["best_params"]),
        "validation_rank": int(selected["validation_rank"]),
        "validation_roc_auc": float(selected["roc_auc"]),
        "validation_average_precision": float(selected["average_precision"]),
        "feature_state_policy": selected["feature_state_policy"],
        "validation_mode": selected["validation_mode"],
        "streaming_batch_size": (
            None
            if pd.isna(selected["streaming_batch_size"])
            else int(selected["streaming_batch_size"])
        ),
        "model_uri": selected["model_uri"],
    }
    if "classification_threshold" in selected and not pd.isna(selected["classification_threshold"]):
        selected_model["classification_threshold"] = float(selected["classification_threshold"])
        selected_model["threshold_selection_metric"] = selected.get("threshold_selection_metric")
        selected_model["threshold_validation_f1"] = float(selected["threshold_validation_f1"])
        selected_model["threshold_validation_precision"] = float(selected["threshold_validation_precision"])
        selected_model["threshold_validation_recall"] = float(selected["threshold_validation_recall"])

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(json.dumps(selected_model, indent=2, sort_keys=True))
    return selected_model


def uses_uid_aggregation(feature_set_name: str) -> bool:
    feature_config = FEATURE_SETS.get(feature_set_name)
    if feature_config is None:
        raise ValueError(f"Unknown feature set: {feature_set_name}")
    return bool(feature_config["use_uid_features"])


def validation_feature_state_policies(
    feature_set_name: str,
    streaming_batch_size: int | None,
) -> list[tuple[str, bool]]:
    if streaming_batch_size is None:
        return [("offline_full_batch", False)]
    if uses_uid_aggregation(feature_set_name):
        return [
            ("frozen", False),
            ("updated_after_prediction", True),
        ]
    return [("not_applicable", False)]


def stream_update_for_policy(feature_state_policy: str) -> bool:
    return feature_state_policy == "updated_after_prediction"


def predict_labels_at_threshold(y_scores: Any, threshold: float) -> Any:
    return (pd.Series(y_scores) >= threshold).astype(int).to_numpy()


def safe_mlflow_key_part(value: str) -> str:
    return value.replace(" ", "_").replace("-", "_")


def select_threshold_for_validated_candidate(
    data_dir: str,
    comparison_path: str | Path = "artifacts/model_validation_incremental.csv",
    selected_model_path: str | Path = "artifacts/selected_model.json",
    v_columns_cache_path: str | None = "artifacts/selected_v_columns.json",
    streaming_batch_size: int | None = None,
) -> dict[str, Any]:
    """
    Tune the operating threshold for the already selected validation candidate.

    This reruns only the best validated candidate from `comparison_path`; it does
    not repeat the full top-k validation search or final test evaluation.
    """
    comparison_file = resolve_project_path(comparison_path)
    results_df = pd.read_csv(comparison_file)
    if results_df.empty:
        raise ValueError(f"No validated candidates found in {comparison_file}.")

    results_df["best_params"] = results_df["best_params"].apply(parse_best_params)
    selected_idx = results_df["roc_auc"].astype(float).idxmax()
    selected = results_df.loc[selected_idx]

    batch_size = (
        None
        if pd.isna(selected.get("streaming_batch_size"))
        else int(selected["streaming_batch_size"])
    )
    if streaming_batch_size is not None:
        batch_size = streaming_batch_size

    logger.info(
        "Selecting threshold for validation_rank=%s model=%s feature_set=%s.",
        selected["validation_rank"],
        selected["model_name"],
        selected["feature_set_name"],
    )

    df = load_interim_data(data_dir)
    _, selected_columns, X_train, X_stream_val, _, y_train, y_stream_val, _ = temporal_train_val_test_split(
        df,
        selected["feature_set_name"],
        v_columns_cache_path=v_columns_cache_path,
    )
    pipeline = build_pipeline_from_config(
        selected_columns,
        selected["feature_set_name"],
        selected["model_name"],
        selected["best_params"],
    )
    pipeline.fit(X_train, y_train)

    if batch_size is None:
        y_scores, _ = offline_predict_scores(pipeline, X_stream_val)
        y_eval = y_stream_val
    else:
        y_scores, _ = streaming_predict_scores(
            pipeline,
            X_stream_val,
            batch_size=batch_size,
            stream_update=stream_update_for_policy(selected["feature_state_policy"]),
        )
        _, y_eval = sort_y_labels(X_stream_val, y_stream_val)

    threshold_result = select_threshold_by_f1(y_eval, y_scores)
    threshold = threshold_result["threshold"]
    y_preds = predict_labels_at_threshold(y_scores, threshold)
    threshold_metrics = compute_classification_metric(
        selected["model_name"],
        "Validation selected threshold",
        y_eval,
        y_scores,
        y_preds,
    )

    results_df.loc[selected_idx, "classification_threshold"] = threshold
    results_df.loc[selected_idx, "threshold_selection_metric"] = threshold_result["selection_metric"]
    results_df.loc[selected_idx, "threshold_validation_f1"] = threshold_result["f1"]
    results_df.loc[selected_idx, "threshold_validation_precision"] = threshold_result["precision"]
    results_df.loc[selected_idx, "threshold_validation_recall"] = threshold_result["recall"]
    save_validation_results(results_df, comparison_path)
    save_selected_model(results_df, selected_model_path)

    output = {
        **threshold_result,
        "validation_rank": int(selected["validation_rank"]),
        "model_name": selected["model_name"],
        "feature_set_name": selected["feature_set_name"],
        "streaming_batch_size": batch_size,
        "metrics_at_threshold": threshold_metrics,
    }
    logger.info(
        "Selected classification threshold %.6f by %s.",
        threshold,
        threshold_result["selection_metric"],
    )
    return output


def validation(
    data_dir: str, 
    v_columns_cache_path: str | None = "artifacts/selected_v_columns.json",
    top_k: int = 3,
    min_k: int = 3,  
    streaming_batch_size: int | None = 1,
) -> None:
    validation_config = ValidationConfig(
        data_dir=data_dir,
        v_columns_cache_path=v_columns_cache_path,
        top_k=top_k,
        min_k=min_k,    
        streaming_batch_size=streaming_batch_size    
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
        all_results: list[dict[str, Any]] = []

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

            _, selected_columns, X_train, X_stream_val, _, y_train, y_stream_val, _ = split_cache[feature_set]

            pipeline = build_pipeline_from_config(
                selected_columns,
                feature_set,
                model_name,
                best_params,
            )
            
            fit_start_time = perf_counter()
            pipeline.fit(X_train, y_train)
            fit_elapsed_seconds = perf_counter() - fit_start_time

            selection_rank = candidate.selection_rank or candidate.rank
            model_artifact_path = f"candidate_{selection_rank}"
            run_id = mlflow.active_run().info.run_id
            model_uri = f"runs:/{run_id}/{model_artifact_path}"
            mlflow.sklearn.log_model(
                pipeline,
                artifact_path=model_artifact_path,
            )

            policies = validation_feature_state_policies(
                feature_set,
                validation_config.streaming_batch_size,
            )
            for feature_state_policy, stream_update in policies:
                scoring_pipeline = deepcopy(pipeline)
                predict_start_time = perf_counter()
                if validation_config.streaming_batch_size is None:
                    y_scores, y_preds = offline_predict_scores(
                        scoring_pipeline,
                        X_stream_val,
                    )
                    y_eval = y_stream_val
                    validation_mode = "offline_full_batch"
                else:
                    y_scores, y_preds = streaming_predict_scores(
                        scoring_pipeline,
                        X_stream_val,
                        batch_size=validation_config.streaming_batch_size,
                        stream_update=stream_update,
                    )
                    _, y_eval = sort_y_labels(X_stream_val, y_stream_val)
                    validation_mode = "streaming"
                predict_elapsed_seconds = perf_counter() - predict_start_time

                metrics = compute_classification_metric(
                    model_name,
                    f"Validation {feature_state_policy}",
                    y_eval,
                    y_scores,
                    y_preds,
                )

                result = CandidateValidationResult(
                    rank=candidate.rank,
                    selection_rank=selection_rank,
                    model_name=candidate.model_name,
                    feature_set_name=candidate.feature_set_name,
                    search_smote=candidate.search_smote,
                    best_sampler=candidate.best_sampler,
                    best_cv_score=candidate.best_cv_score,
                    best_params=candidate.best_params,
                    model_uri=model_uri,
                    validation_mode=validation_mode,
                    feature_state_policy=feature_state_policy,
                    streaming_batch_size=validation_config.streaming_batch_size,
                    fit_elapsed_seconds=fit_elapsed_seconds,
                    predict_elapsed_seconds=predict_elapsed_seconds,
                    **metrics,
                )
                all_results.append(result.model_dump())

                incremental_results_df = (
                    pd.DataFrame(all_results)
                    .sort_values("roc_auc", ascending=False, na_position="last")
                    .reset_index(drop=True)
                )
                incremental_results_df.insert(0, "validation_rank", incremental_results_df.index + 1)
                save_validation_results(
                    incremental_results_df,
                    validation_config.validation_results_path,
                )
                save_selected_model(
                    incremental_results_df,
                    validation_config.selected_model_path,
                )
                mlflow.log_artifact(
                    str(resolve_project_path(validation_config.selected_model_path)),
                    artifact_path="selected_model",
                )

                policy_key = safe_mlflow_key_part(feature_state_policy)
                metric_prefix = f"candidate_{result.selection_rank}_{policy_key}_validation"
                mlflow.log_metrics({
                    f"{metric_prefix}_roc_auc": result.roc_auc,
                    f"{metric_prefix}_f1": result.f1,
                    f"{metric_prefix}_recall": result.recall,
                    f"{metric_prefix}_precision": result.precision,
                    f"{metric_prefix}_average_precision": result.average_precision,
                    f"{metric_prefix}_accuracy": result.accuracy,
                    f"{metric_prefix}_fit_elapsed_seconds": result.fit_elapsed_seconds,
                    f"{metric_prefix}_predict_elapsed_seconds": result.predict_elapsed_seconds,
                })
                mlflow.log_params({
                    f"{metric_prefix}_model_name": result.model_name,
                    f"{metric_prefix}_feature_set_name": result.feature_set_name,
                    f"{metric_prefix}_validation_mode": result.validation_mode,
                    f"{metric_prefix}_feature_state_policy": result.feature_state_policy,
                    f"{metric_prefix}_streaming_batch_size": str(result.streaming_batch_size),
                })


def save_test_results(
    results_df: pd.DataFrame,
    output_path: str | Path = "artifacts/model_test_incremental.csv",
) -> None:
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    ordered_columns = [
        col for col in TEST_RESULT_COLUMNS if col in results_df.columns
    ]
    remaining_columns = [
        col for col in results_df.columns if col not in ordered_columns
    ]
    results_df = results_df[ordered_columns + remaining_columns]
    results_df.to_csv(output_file, index=False)


def evaluate_test_candidate(
    X_fit: pd.DataFrame,
    y_fit: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    selected_columns: list[str],
    candidate: ValidatedCandidateRow,
    test_config: TestConfig,
    training_data_scope: str,
) -> tuple[TestResult, Any]:
    pipeline = build_pipeline_from_config(
        selected_columns,
        candidate.feature_set_name,
        candidate.model_name,
        candidate.best_params,
    )

    fit_start_time = perf_counter()
    pipeline.fit(X_fit, y_fit)
    fit_elapsed_seconds = perf_counter() - fit_start_time

    y_test_for_metrics = y_test
    predict_start_time = perf_counter()
    if test_config.streaming_batch_size is None:
        y_scores, _ = offline_predict_scores(
            pipeline,
            X_test,
        )
        test_mode = "offline_full_batch"
    else:
        y_scores, _ = streaming_predict_scores(
            pipeline,
            X_test,
            batch_size=test_config.streaming_batch_size,
            stream_update=stream_update_for_policy(candidate.feature_state_policy),
        )
        _, y_test_for_metrics = sort_y_labels(X_test, y_test)
        test_mode = "streaming"
    y_preds = predict_labels_at_threshold(y_scores, candidate.classification_threshold)
    predict_elapsed_seconds = perf_counter() - predict_start_time

    metrics = compute_classification_metric(
        candidate.model_name,
        f"Test {training_data_scope}",
        y_test_for_metrics,
        y_scores,
        y_preds,
    )

    result = TestResult(
        training_data_scope=training_data_scope,
        model_name=candidate.model_name,
        feature_set_name=candidate.feature_set_name,
        search_smote=candidate.search_smote,
        best_sampler=candidate.best_sampler,
        selected_validation_rank=candidate.validation_rank,
        selected_validation_roc_auc=candidate.roc_auc,
        selected_feature_state_policy=candidate.feature_state_policy,
        best_params=candidate.best_params,
        test_mode=test_mode,
        streaming_batch_size=test_config.streaming_batch_size,
        classification_threshold=candidate.classification_threshold,
        fit_elapsed_seconds=fit_elapsed_seconds,
        predict_elapsed_seconds=predict_elapsed_seconds,
        **metrics,
    )
    return result, pipeline


def test(
    data_dir: str, 
    v_columns_cache_path: str | None = "artifacts/selected_v_columns.json",
    streaming_batch_size: int | None = 1,
) -> None:
    test_config = TestConfig(
        data_dir=data_dir,
        v_columns_cache_path=v_columns_cache_path,
        streaming_batch_size=streaming_batch_size    
    )   
    candidate = select_best_validated_candidate(test_config.comparison_path)

    mlflow.set_experiment(MLFLOW_TEST_EXPERIMENT_NAME)

    with mlflow.start_run(run_name="online-streaming-test"):  
        df = load_interim_data(test_config.data_dir)

        logger.info("Preparing feature set: %s", candidate.feature_set_name)

        _, selected_columns, X_train, X_stream_val, X_test, y_train, y_stream_val, y_test = temporal_train_val_test_split(
            df,
            candidate.feature_set_name,
            v_columns_cache_path=test_config.v_columns_cache_path,
        )

        training_sets = [
            ("train_only", X_train, y_train),
            (
                "train_plus_validation",
                pd.concat([X_train, X_stream_val]),
                pd.concat([y_train, y_stream_val]),
            ),
        ]
        performance_results = []
        final_result = None
        final_pipeline = None

        for training_data_scope, X_fit, y_fit in training_sets:
            result, pipeline = evaluate_test_candidate(
                X_fit=X_fit,
                y_fit=y_fit,
                X_test=X_test,
                y_test=y_test,
                selected_columns=selected_columns,
                candidate=candidate,
                test_config=test_config,
                training_data_scope=training_data_scope,
            )
            performance_results.append(result.model_dump())

            performance_df = pd.DataFrame(performance_results)
            performance_df.insert(0, "test_rank", range(1, len(performance_df) + 1))
            save_test_results(
                performance_df,
                test_config.test_performance_path,
            )

            prefix = f"{training_data_scope}_test"
            mlflow.log_metrics({
                f"{prefix}_roc_auc": result.roc_auc,
                f"{prefix}_f1": result.f1,
                f"{prefix}_recall": result.recall,
                f"{prefix}_precision": result.precision,
                f"{prefix}_average_precision": result.average_precision,
                f"{prefix}_accuracy": result.accuracy,
                f"{prefix}_fit_elapsed_seconds": result.fit_elapsed_seconds,
                f"{prefix}_predict_elapsed_seconds": result.predict_elapsed_seconds,
            })

            if training_data_scope == "train_plus_validation":
                final_result = result
                final_pipeline = pipeline

        if final_result is None or final_pipeline is None:
            raise RuntimeError("Final train_plus_validation test result was not produced.")

        final_results_df = pd.DataFrame([final_result.model_dump()])
        final_results_df.insert(0, "test_rank", 1)
        save_test_results(
            final_results_df,
            test_config.test_results_path,
        )

        mlflow.log_artifact(
            str(resolve_project_path(test_config.test_performance_path)),
            artifact_path="test_results",
        )

        mlflow.log_metrics({
            "final_test_roc_auc": final_result.roc_auc,
            "final_test_f1": final_result.f1,
            "final_test_recall": final_result.recall,
            "final_test_precision": final_result.precision,
            "final_test_average_precision": final_result.average_precision,
            "final_test_accuracy": final_result.accuracy,
            "final_fit_elapsed_seconds": final_result.fit_elapsed_seconds,
            "final_predict_elapsed_seconds": final_result.predict_elapsed_seconds,
        })
        mlflow.log_params({
            "final_model_name": final_result.model_name,
            "final_feature_set_name": final_result.feature_set_name,
            "final_training_data_scope": final_result.training_data_scope,
            "final_test_mode": final_result.test_mode,
            "final_streaming_batch_size": str(final_result.streaming_batch_size),
            "final_classification_threshold": str(final_result.classification_threshold),
        })
        mlflow.sklearn.log_model(
            final_pipeline,
            artifact_path="final_candidate",
        )
