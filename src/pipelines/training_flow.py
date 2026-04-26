import logging
import pandas as pd
from time import perf_counter
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import make_scorer, f1_score, accuracy_score, recall_score, precision_score
from typing import List, Any, Literal
from pathlib import Path
from pydantic import BaseModel, Field, ConfigDict
import mlflow

from ..data import (
    load_interim_data,
    temporal_train_val_test_split
)
from .cv_logging import (
    log_top_cv_candidates,
)
from ..models import (
    get_candidate_configs,
    build_pipeline_from_config,
)
from ..utils import (
    RANDOM_STATE, 
    DEFAULT_FEATURE_SET,
    DEFAULT_SEARCH_SMOTE,
    DEFAULT_SEARCH_N_JOBS,
    MLFLOW_EXPERIMENT_NAME,
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
    feature_set_name: str = "base_selected_v"
    threshold: float = Field(default=0.65, ge=0.5, le=1.0)
    search_smote: bool = True
    use_successive_halving: bool = False
    search_n_jobs: int = Field(default=1, ge=1)
    top_k: int = Field(default=5, ge=1)
    shortlist_roc_auc_tolerance: float = Field(default=0.005, ge=0.0)
    save_incremental: bool = True
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


class TopCandidatesMetadata(BaseModel):
    feature_set_name: str
    threshold: float = Field(ge=0.5, le=1.0)
    search_smote: bool
    shortlist_strategy: Literal["performance_filtered_diverse_shortlist"]
    roc_auc_tolerance: float = Field(ge=0.0)
    selected_columns: list[str]
    top_k: int = Field(ge=0)
    candidates: list[dict[str, Any]]


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
    save_incremental: bool = True,
    incremental_save_path: str = "artifacts/model_comparison_incremental.csv",
):
    all_results = []
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
    
    # Initialize incremental save file
    if save_incremental:
        Path(incremental_save_path).parent.mkdir(parents=True, exist_ok=True)

    mlflow.sklearn.autolog(log_models=False)
    for config_idx, config in enumerate(candidate_configs, start=1):
        with mlflow.start_run(run_name=f"search-{config['name']}", nested=True):
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
                "Training model %d/%d: %s",
                config_idx,
                len(candidate_configs),
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

            if save_incremental:
                incremental_results_df = (
                    pd.DataFrame(all_results)
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
            cv_results_path = f"artifacts/cv_results_{config['name']}.csv"
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


def shortlist_candidates(
    results_df: pd.DataFrame,
    max_candidates: int = 5,
    roc_auc_tolerance: float = 0.005,
    max_per_feature_set: int = 5,
    max_per_model_name: int = 1,
    max_per_sampler: int = 5,
) -> pd.DataFrame:
    if results_df.empty:
        return results_df.copy()

    sorted_results = results_df.sort_values(
        by=[
            "best_cv_roc_auc",
            "best_cv_average_precision",
            "best_cv_recall",
            "best_cv_f1",
        ],
        ascending=False,
    ).reset_index(drop=True)

    best_roc_auc = float(sorted_results.iloc[0]["best_cv_roc_auc"])
    eligible = sorted_results[
        sorted_results["best_cv_roc_auc"] >= best_roc_auc - roc_auc_tolerance
    ].copy()
    if eligible.empty:
        eligible = sorted_results.head(1).copy()

    selected_rows = []
    selected_indices = set()
    feature_set_counts: dict[str, int] = {}
    model_name_counts: dict[str, int] = {}
    sampler_counts: dict[str, int] = {}

    def can_select(candidate: pd.Series) -> bool:
        feature_set_name = candidate["feature_set_name"]
        model_name = candidate["model_name"]
        sampler_name = candidate["best_sampler"]
        return (
            feature_set_counts.get(feature_set_name, 0) < max_per_feature_set
            and model_name_counts.get(model_name, 0) < max_per_model_name
            and sampler_counts.get(sampler_name, 0) < max_per_sampler
        )

    def add_candidate(candidate: pd.Series) -> None:
        feature_set_name = candidate["feature_set_name"]
        model_name = candidate["model_name"]
        sampler_name = candidate["best_sampler"]
        selected_rows.append(candidate.to_dict())
        selected_indices.add(int(candidate.name))
        feature_set_counts[feature_set_name] = feature_set_counts.get(feature_set_name, 0) + 1
        model_name_counts[model_name] = model_name_counts.get(model_name, 0) + 1
        sampler_counts[sampler_name] = sampler_counts.get(sampler_name, 0) + 1

    add_candidate(sorted_results.iloc[0])

    for _, candidate in eligible.iterrows():
        if len(selected_rows) >= max_candidates:
            break
        if int(candidate.name) in selected_indices:
            continue
        if can_select(candidate):
            add_candidate(candidate)

    for _, candidate in sorted_results.iterrows():
        if len(selected_rows) >= max_candidates:
            break
        if int(candidate.name) in selected_indices:
            continue
        if can_select(candidate):
            add_candidate(candidate)

    shortlist_df = pd.DataFrame(selected_rows)
    shortlist_df = shortlist_df.sort_values(
        by=[
            "best_cv_roc_auc",
            "best_cv_average_precision",
            "best_cv_recall",
            "best_cv_f1",
        ],
        ascending=False,
    ).reset_index(drop=True)
    shortlist_df["rank"] = range(1, len(shortlist_df) + 1)
    return shortlist_df


def save_top_candidates_metadata(
    top_candidates_df: pd.DataFrame,
    selected_columns: List[str],
    feature_set_name: str,
    threshold: float,
    search_smote: bool,
    roc_auc_tolerance: float,
    output_path: str = "artifacts/top_candidates.json",
) -> None:
    metadata = TopCandidatesMetadata(
        feature_set_name=feature_set_name,
        threshold=threshold,
        search_smote=search_smote,
        shortlist_strategy="performance_filtered_diverse_shortlist",
        roc_auc_tolerance=roc_auc_tolerance,
        selected_columns=selected_columns,
        top_k=len(top_candidates_df),
        candidates=top_candidates_df.drop(columns=["rebuild_params"], errors="ignore").to_dict(orient="records"),
    )

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(metadata.model_dump_json(indent=2))


def fit_and_save_top_candidates(
    top_candidates_df: pd.DataFrame,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    selected_columns: List[str],
    feature_set_name: str,
    search_smote: bool,
) -> None:
    for candidate in top_candidates_df.to_dict(orient="records"):
        pipeline = build_pipeline_from_config(
            selected_columns,
            feature_set_name,
            candidate["model_name"],
            candidate["rebuild_params"],
            search_smote=search_smote,
        )
        start_time = perf_counter()
        pipeline.fit(X_train, y_train)
        fit_elapsed_seconds = perf_counter() - start_time

        mlflow.log_metric(
            f"candidate_{candidate['rank']}_final_fit_seconds",
            fit_elapsed_seconds,
        )        
        mlflow.sklearn.log_model(
            pipeline,
            artifact_path=f"candidate_{candidate['rank']}",
        )        


def training(
    data_dir: str, 
    feature_set_name: str = DEFAULT_FEATURE_SET,
    threshold: float = 0.65,
    search_smote: bool = DEFAULT_SEARCH_SMOTE,
    use_successive_halving: bool = False,
    search_n_jobs: int = DEFAULT_SEARCH_N_JOBS,
    top_k: int = 5,
    shortlist_roc_auc_tolerance: float = 0.005,
    save_incremental: bool = True,
    v_columns_cache_path: str | None = "artifacts/selected_v_columns.json",
) -> None:
    training_config = TrainingConfig(
        data_dir=data_dir,
        feature_set_name=feature_set_name,
        threshold=threshold,
        search_smote=search_smote,
        use_successive_halving=use_successive_halving,
        search_n_jobs=search_n_jobs,
        top_k=top_k,
        shortlist_roc_auc_tolerance=shortlist_roc_auc_tolerance,
        save_incremental=save_incremental,
        v_columns_cache_path=v_columns_cache_path,
    )   

    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    with mlflow.start_run(run_name="offline-full-batch-training"):  
        mlflow.log_params({
            "feature_set_name": training_config.feature_set_name,
            "threshold": training_config.threshold,
            "search_smote": training_config.search_smote,
            "use_successive_halving": training_config.use_successive_halving,
            "search_n_jobs": training_config.search_n_jobs,
            "top_k": training_config.top_k,
            "shortlist_roc_auc_tolerance": training_config.shortlist_roc_auc_tolerance,
        })        

        df = load_interim_data(training_config.data_dir)     
        cv_splits, selected_columns, X_train, _, _, y_train, _, _ = \
            temporal_train_val_test_split(
                df,
                training_config.feature_set_name,
                training_config.threshold,
                training_config.v_columns_cache_path,
            )
        
        mlflow.log_params({
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
            feature_set_name=training_config.feature_set_name, 
            search_smote=training_config.search_smote, 
            use_successive_halving=training_config.use_successive_halving,
            search_n_jobs=training_config.search_n_jobs,
            save_incremental=training_config.save_incremental,
        )
        save_model_comparison(results_df)

        top_candidates_df = shortlist_candidates(
            results_df,
            max_candidates=training_config.top_k,
            roc_auc_tolerance=training_config.shortlist_roc_auc_tolerance,
        )
        save_top_candidates_metadata(
            top_candidates_df,
            selected_columns,
            training_config.feature_set_name,
            training_config.threshold,
            training_config.search_smote,
            training_config.shortlist_roc_auc_tolerance,
        )
        log_top_cv_candidates(logger, top_candidates_df, training_config.top_k)

        if not top_candidates_df.empty:
            best_candidate = top_candidates_df.iloc[0]
            mlflow.log_params({
                "best_model_name": best_candidate["model_name"],
                "best_sampler": best_candidate["best_sampler"],
            })
            mlflow.log_metrics({
                "best_cv_roc_auc": best_candidate["best_cv_roc_auc"],
                "best_cv_average_precision": best_candidate["best_cv_average_precision"],
                "best_cv_f1": best_candidate["best_cv_f1"],
                "best_cv_recall": best_candidate["best_cv_recall"],
                "best_cv_precision": best_candidate["best_cv_precision"],
                "best_cv_accuracy": best_candidate["best_cv_accuracy"],
            })      
            fit_and_save_top_candidates(
                top_candidates_df,
                X_train,
                y_train,
                selected_columns,
                training_config.feature_set_name,
                training_config.search_smote,
            )

        mlflow.log_artifact("artifacts/model_comparison.csv")
        mlflow.log_artifact("artifacts/top_candidates.json")
