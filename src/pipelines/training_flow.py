import logging
import pandas as pd
from time import perf_counter
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import make_scorer, f1_score, accuracy_score, recall_score, precision_score
from typing import Any
from pathlib import Path
from pydantic import BaseModel, Field, ConfigDict
import mlflow

from ..data import (
    load_interim_data,
    temporal_train_val_test_split
)
from ..models import (
    get_candidate_configs,
)
from ..utils import (
    RANDOM_STATE, 
    DEFAULT_FEATURE_SET,
    DEFAULT_SEARCH_SMOTE,
    DEFAULT_SEARCH_N_JOBS,
    FEATURE_SETS,
    MLFLOW_TRAINING_EXPERIMENT_NAME,
)


logger = logging.getLogger(__name__)

CV_SCORING = {
    "roc_auc": "roc_auc",
    "average_precision": "average_precision",
    "f1": make_scorer(f1_score, zero_division=0),
    "accuracy": make_scorer(accuracy_score),
    "recall": make_scorer(recall_score, zero_division=0),
    "precision": make_scorer(precision_score, zero_division=0),
}


class TrainingConfig(BaseModel):
    data_dir: str | Path
    feature_set_names: list[str] = Field(default_factory=lambda: list(FEATURE_SETS))
    threshold: float = Field(default=0.65, ge=0.5, le=1.0)
    search_smote_options: list[bool] = Field(default_factory=lambda: [False, True])
    use_successive_halving: bool = False
    search_n_jobs: int = Field(default=1, ge=1)
    v_columns_cache_path: str | Path | None = "artifacts/selected_v_columns.json"


class ModelSearchResult(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

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
    rebuild_params: dict[str, Any]
    best_sampler: str


def normalize_feature_set_names(feature_set_names: list[str] | None) -> list[str]:
    names = feature_set_names or list(FEATURE_SETS)
    unknown_names = sorted(set(names) - set(FEATURE_SETS))
    if unknown_names:
        raise ValueError(f"Unknown feature set(s): {unknown_names}")
    return list(dict.fromkeys(names))


def normalize_search_smote_options(search_smote_options: list[bool] | None) -> list[bool]:
    options = search_smote_options or [False, True]
    return list(dict.fromkeys(bool(option) for option in options))


def filter_valid_cv_splits(cv_splits, y_train):
    valid_splits = []

    for fold_idx, (train_idx, val_idx) in enumerate(cv_splits, start=1):
        y_fold_train = y_train.iloc[train_idx]
        y_fold_val = y_train.iloc[val_idx]

        if y_fold_train.nunique() < 2:
            logger.warning(
                "Skipping CV fold %s because the training fold has only one class.",
                fold_idx,
            )
            continue

        if y_fold_val.nunique() < 2:
            logger.warning(
                "Skipping CV fold %s because the validation fold has only one class.",
                fold_idx,
            )
            continue

        valid_splits.append((train_idx, val_idx))

    if not valid_splits:
        raise ValueError("No valid CV folds remain after filtering single-class folds.")

    return valid_splits


def serialize_search_params(best_params):
    serialized = {}

    for key, value in best_params.items():
        if key == "sampler":
            if value == "passthrough":
                serialized[key] = "passthrough"
            else:
                serialized[key] = {
                    "type": value.__class__.__name__,
                    "params": value.get_params(deep=False) if hasattr(value, "get_params") else str(value),
                }
            continue

        if isinstance(value, (str, int, float, bool)) or value is None:
            serialized[key] = value
        else:
            serialized[key] = str(value)

    return serialized


def save_model_comparison(
    results_df: pd.DataFrame,
    output_path: str = "artifacts/model_comparison.csv",
) -> None:
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    results_df.drop(columns=["rebuild_params"], errors="ignore").to_csv(output_file, index=False)


def run_model_search(
    X_train,
    y_train,
    cv_splits,
    selected_columns,
    feature_set_name: str = DEFAULT_FEATURE_SET,
    search_smote: bool = DEFAULT_SEARCH_SMOTE,
    use_successive_halving: bool = False,
    search_n_jobs: int = DEFAULT_SEARCH_N_JOBS,
    incremental_save_path: str = "artifacts/model_comparison_incremental.csv",
    existing_results: list[dict[str, Any]] | None = None,
):
    all_results = []
    existing_results = existing_results or []
    valid_cv_splits = filter_valid_cv_splits(cv_splits, y_train)

    candidate_configs = get_candidate_configs(
        selected_columns,
        feature_set_name,
        search_smote=search_smote,
    )
    if use_successive_halving:
        logger.warning(
            "use_successive_halving=True is ignored because multi-metric CV requires RandomizedSearchCV."
        )

    mlflow.sklearn.autolog(log_models=False)
    for config_idx, config in enumerate(candidate_configs, start=1):
        smote_label = "with-smote" if search_smote else "without-smote"
        with mlflow.start_run(
            run_name=f"search-{feature_set_name}-{smote_label}-{config['name']}",
            nested=True,
        ):
            mlflow.log_params({
                "model_name": config["name"],
                "feature_set_name": feature_set_name,
                "search_smote": search_smote,
                "n_iter": config.get("n_iter", 20),
                "num_cv_folds_used": len(valid_cv_splits),
            })

            search_kwargs = {
                "estimator": config["pipeline"],
                "param_distributions": config["param_distributions"],
                "scoring": CV_SCORING,
                "cv": valid_cv_splits,
                "n_jobs": search_n_jobs,
                "refit": "roc_auc",
                "random_state": RANDOM_STATE,
                "verbose": 2,
                "error_score": "raise",
            }
            search_kwargs.update({
                "n_iter": config.get("n_iter", 20),
                "pre_dispatch": "n_jobs",
            })

            logger.info(
                "Training model %d/%d for feature_set=%s search_smote=%s: %s",
                config_idx,
                len(candidate_configs),
                feature_set_name,
                search_smote,
                config["name"],
            )

            start_time = perf_counter()
            search = RandomizedSearchCV(**search_kwargs)
            search.fit(X_train, y_train)
            elapsed_seconds = perf_counter() - start_time

            best_index = search.best_index_
            cv_results = search.cv_results_
            best_params = search.best_params_
            serialized_best_params = serialize_search_params(best_params)
            sampler_used = best_params.get("sampler", "passthrough")

            result = ModelSearchResult(
                model_name=config["name"],
                feature_set_name=feature_set_name,
                search_smote=search_smote,
                best_cv_score=search.best_score_,
                best_cv_roc_auc=cv_results["mean_test_roc_auc"][best_index],
                best_cv_average_precision=cv_results["mean_test_average_precision"][best_index],
                best_cv_f1=cv_results["mean_test_f1"][best_index],
                best_cv_accuracy=cv_results["mean_test_accuracy"][best_index],
                best_cv_recall=cv_results["mean_test_recall"][best_index],
                best_cv_precision=cv_results["mean_test_precision"][best_index],
                num_cv_folds_used=len(valid_cv_splits),
                best_params=serialized_best_params,
                rebuild_params=best_params,
                best_sampler=str(sampler_used),
            )
            all_results.append(result.model_dump())

            incremental_results_df = (
                pd.DataFrame(existing_results + all_results)
                .sort_values("best_cv_score", ascending=False)
                .reset_index(drop=True)
            )
            incremental_results_df.insert(0, "rank", incremental_results_df.index + 1)
            save_model_comparison(incremental_results_df, incremental_save_path)

            n_iter = config.get("n_iter", 20)
            n_cv_folds = len(valid_cv_splits)
            num_fit_attempts = n_iter * n_cv_folds

            mlflow.log_metrics({
                "best_cv_roc_auc": result.best_cv_roc_auc,
                "best_cv_average_precision": result.best_cv_average_precision,
                "best_cv_f1": result.best_cv_f1,
                "best_cv_accuracy": result.best_cv_accuracy,
                "best_cv_recall": result.best_cv_recall,
                "best_cv_precision": result.best_cv_precision,
                "search_elapsed_seconds": elapsed_seconds,
                "avg_seconds_per_cv_fit": elapsed_seconds / num_fit_attempts,                
            })

            mlflow.log_params({
                "num_search_iterations": n_iter,
                "num_fit_attempts": num_fit_attempts,
            })            

            mlflow.log_dict(
                serialized_best_params,
                artifact_file="best_params.json",
            )      

            cv_results_df = pd.DataFrame(search.cv_results_)
            cv_results_path = (
                f"artifacts/cv_results_{feature_set_name}_{smote_label}_{config['name']}.csv"
            )
            cv_results_df.to_csv(cv_results_path, index=False)
            mlflow.log_artifact(cv_results_path, artifact_path="cv_results")
            del search                  

    results_df = (
        pd.DataFrame(all_results)
        .sort_values("best_cv_score", ascending=False)
        .reset_index(drop=True)
    )
    results_df.insert(0, "rank", results_df.index + 1)
    return results_df


def training(
    data_dir: str, 
    feature_set_name: str = DEFAULT_FEATURE_SET,
    feature_set_names: list[str] | None = None,
    threshold: float = 0.65,
    search_smote: bool = DEFAULT_SEARCH_SMOTE,
    search_smote_options: list[bool] | None = None,
    use_successive_halving: bool = False,
    search_n_jobs: int = DEFAULT_SEARCH_N_JOBS,
    v_columns_cache_path: str | None = "artifacts/selected_v_columns.json",
) -> None:
    requested_feature_set_names = feature_set_names
    if requested_feature_set_names is None and feature_set_name != DEFAULT_FEATURE_SET:
        requested_feature_set_names = [feature_set_name]

    requested_search_smote_options = search_smote_options
    if requested_search_smote_options is None and search_smote != DEFAULT_SEARCH_SMOTE:
        requested_search_smote_options = [search_smote]

    resolved_feature_set_names = normalize_feature_set_names(requested_feature_set_names)
    resolved_search_smote_options = normalize_search_smote_options(requested_search_smote_options)

    training_config = TrainingConfig(
        data_dir=data_dir,
        feature_set_names=resolved_feature_set_names,
        threshold=threshold,
        search_smote_options=resolved_search_smote_options,
        use_successive_halving=use_successive_halving,
        search_n_jobs=search_n_jobs,
        v_columns_cache_path=v_columns_cache_path,
    )   

    mlflow.set_experiment(MLFLOW_TRAINING_EXPERIMENT_NAME)

    with mlflow.start_run(run_name="offline-full-batch-training"):  
        mlflow.log_params({
            "feature_set_names": ",".join(training_config.feature_set_names),
            "threshold": training_config.threshold,
            "search_smote_options": ",".join(
                str(option) for option in training_config.search_smote_options
            ),
            "use_successive_halving": training_config.use_successive_halving,
            "search_n_jobs": training_config.search_n_jobs,
        })        

        df = load_interim_data(training_config.data_dir)
        all_results = []
        for feature_set in training_config.feature_set_names:
            logger.info("Preparing feature set: %s", feature_set)
            cv_splits, selected_columns, X_train, _, _, y_train, _, _ = \
                temporal_train_val_test_split(
                    df,
                    feature_set,
                    training_config.threshold,
                    training_config.v_columns_cache_path,
                )

            for smote_enabled in training_config.search_smote_options:
                smote_label = "with-smote" if smote_enabled else "without-smote"
                logger.info(
                    "Starting model search for feature_set=%s search_smote=%s",
                    feature_set,
                    smote_enabled,
                )
                with mlflow.start_run(
                    run_name=f"feature-set-{feature_set}-{smote_label}",
                    nested=True,
                ):
                    mlflow.log_params({
                        "feature_set_name": feature_set,
                        "threshold": training_config.threshold,
                        "search_smote": smote_enabled,
                        "num_train_rows": len(X_train),
                        "num_input_columns": X_train.shape[1],
                        "num_selected_columns": len(selected_columns),
                        "num_cv_splits": len(cv_splits),
                    })
                    mlflow.log_metric("train_positive_rate", float(y_train.mean()))

                    results_df = run_model_search(
                        X_train,
                        y_train,
                        cv_splits,
                        selected_columns,
                        feature_set_name=feature_set,
                        search_smote=smote_enabled,
                        use_successive_halving=training_config.use_successive_halving,
                        search_n_jobs=training_config.search_n_jobs,
                        existing_results=all_results,
                    )
                    all_results.extend(results_df.drop(columns=["rank"]).to_dict(orient="records"))

                    incremental_results_df = (
                        pd.DataFrame(all_results)
                        .sort_values("best_cv_score", ascending=False)
                        .reset_index(drop=True)
                    )
                    incremental_results_df.insert(0, "rank", incremental_results_df.index + 1)
                    save_model_comparison(
                        incremental_results_df,
                        "artifacts/model_comparison_incremental.csv",
                    )
