import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.experimental import enable_halving_search_cv  # noqa: F401
from sklearn.model_selection import HalvingRandomSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import make_scorer, f1_score, accuracy_score, recall_score, precision_score
from typing import List
import joblib
import json
from pathlib import Path
from ..data import (
    load_interim_data,
    temporal_train_val_test_split
)
from .cv_logging import (
    log_top_cv_candidates,
)
from ..features import (
    NumericShiftFillTransformer, 
    DataFrameOrdinalEncoder, 
    DColumnNormalizer, 
    DropColumnsTransformer,
    UIDAggregationAppendTransformer,
)
from ..models import (
    get_candidate_configs,
    build_pipeline_from_config,
)
from ..utils import (
    RANDOM_STATE, 
    DEFAULT_FEATURE_SET,
    DEFAULT_SEARCH_SMOTE,
    FEATURE_SETS,
    CATEGORICAL_COLUMNS, 
    NUMERICAL_COLUMNS, 
    DROP_COLUMNS,
    UID_AGGREGATION_MAIN_COLUMNS,
    UID_AGGREGATION_UID_COLUMNS,
    UID_AGGREGATION_FUNCTIONS,
)
import logging

logger = logging.getLogger(__name__)

CV_SCORING = {
    "roc_auc": "roc_auc",
    "average_precision": "average_precision",
    "f1": make_scorer(f1_score, zero_division=0),
    "accuracy": make_scorer(accuracy_score),
    "recall": make_scorer(recall_score, zero_division=0),
    "precision": make_scorer(precision_score, zero_division=0),
}


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


def build_feature_pipeline(selected_columns: List[str], feature_set_name: str = DEFAULT_FEATURE_SET) -> Pipeline:
    feature_config = FEATURE_SETS.get(feature_set_name)
    if feature_config is None:
        raise ValueError(f"Unknown feature set: {feature_set_name}")

    numeric_columns = [col for col in NUMERICAL_COLUMNS if col in selected_columns + DROP_COLUMNS]
    categorical_columns = [col for col in CATEGORICAL_COLUMNS if col in selected_columns and col not in DROP_COLUMNS]
    steps = [
        # Keep the feature pipeline row-local and stateless at inference time.
        ("numerical_shift_fill", NumericShiftFillTransformer(numeric_columns)),
        ("normalize_D_columns", DColumnNormalizer()),
    ]

    if feature_config["use_uid_features"]:
        steps.append((
            "append_uid_aggregates",
            UIDAggregationAppendTransformer(
                main_columns=UID_AGGREGATION_MAIN_COLUMNS,
                uid_columns=UID_AGGREGATION_UID_COLUMNS,
                aggregations=UID_AGGREGATION_FUNCTIONS,
                fill_value=-1.0,
                dtype="float32",
            ),
        ))

    steps.extend([
        ("drop_columns", DropColumnsTransformer(DROP_COLUMNS, copy=False)),
        ("ordinal_encode", DataFrameOrdinalEncoder(categorical_columns, handle_unknown="use_encoded_value", unknown_value=-1)),
    ])

    return Pipeline(steps, verbose=True)


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


def run_model_search(
    X_train,
    y_train,
    cv_splits,
    selected_columns,
    feature_set_name: str = DEFAULT_FEATURE_SET,
    search_smote: bool = DEFAULT_SEARCH_SMOTE,
    use_successive_halving: bool = True,
):
    all_results = []
    valid_cv_splits = filter_valid_cv_splits(cv_splits, y_train)

    candidate_configs = get_candidate_configs(
        selected_columns,
        feature_set_name,
        search_smote=search_smote,
    )
    search_cls = HalvingRandomSearchCV if use_successive_halving else RandomizedSearchCV

    for config in candidate_configs:
        search_kwargs = {
            "estimator": config["pipeline"],
            "param_distributions": config["param_distributions"],
            "scoring": CV_SCORING,
            "cv": valid_cv_splits,
            "n_jobs": -1,
            "refit": "roc_auc",
            "random_state": RANDOM_STATE,
            "verbose": 2,
            "error_score": "raise",
        }
        if use_successive_halving:
            search_kwargs.update({
                "n_candidates": config.get("n_candidates", "exhaust"),
                "factor": config.get("factor", 3),
                "aggressive_elimination": True,
            })
        else:
            search_kwargs.update({
                "n_iter": config.get("n_iter", 20),
                "pre_dispatch": "n_jobs",
            })

        search = search_cls(**search_kwargs)

        search.fit(X_train, y_train)

        best_index = search.best_index_
        cv_results = search.cv_results_
        best_params = search.best_params_
        serialized_best_params = serialize_search_params(best_params)
        sampler_used = best_params.get("sampler", "passthrough")

        all_results.append({
            "model_name": config["name"],
            "feature_set_name": feature_set_name,
            "search_smote": search_smote,
            "best_cv_score": search.best_score_,
            "best_cv_roc_auc": cv_results["mean_test_roc_auc"][best_index],
            "best_cv_average_precision": cv_results["mean_test_average_precision"][best_index],
            "best_cv_f1": cv_results["mean_test_f1"][best_index],
            "best_cv_accuracy": cv_results["mean_test_accuracy"][best_index],
            "best_cv_recall": cv_results["mean_test_recall"][best_index],
            "best_cv_precision": cv_results["mean_test_precision"][best_index],
            "num_cv_folds_used": len(valid_cv_splits),
            "best_params": serialized_best_params,
            "rebuild_params": best_params,
            "best_sampler": str(sampler_used),
        })

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
    max_per_feature_set: int = 2,
    max_per_model_name: int = 1,
    max_per_sampler: int = 2,
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
    output = {
        "feature_set_name": feature_set_name,
        "threshold": threshold,
        "search_smote": search_smote,
        "shortlist_strategy": "performance_filtered_diverse_shortlist",
        "roc_auc_tolerance": roc_auc_tolerance,
        "selected_columns": selected_columns,
        "top_k": len(top_candidates_df),
        "candidates": top_candidates_df.drop(columns=["rebuild_params"], errors="ignore").to_dict(orient="records"),
    }

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(json.dumps(output, indent=2, default=str))


def fit_and_save_top_candidates(
    top_candidates_df: pd.DataFrame,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    selected_columns: List[str],
    feature_set_name: str,
    search_smote: bool,
    output_dir: str = "artifacts/candidates",
) -> None:
    candidates_dir = Path(output_dir)
    candidates_dir.mkdir(parents=True, exist_ok=True)

    for candidate in top_candidates_df.to_dict(orient="records"):
        pipeline = build_pipeline_from_config(
            selected_columns,
            feature_set_name,
            candidate["model_name"],
            candidate["rebuild_params"],
            search_smote=search_smote,
        )
        pipeline.fit(X_train, y_train)
        joblib.dump(pipeline, candidates_dir / f"candidate_{candidate['rank']}.joblib")


def training(
    data_dir: str, 
    feature_set_name: str = DEFAULT_FEATURE_SET,
    threshold: float = 0.65,
    search_smote: bool = DEFAULT_SEARCH_SMOTE,
    use_successive_halving: bool = True,
    top_k: int = 5,
    shortlist_roc_auc_tolerance: float = 0.005,
    save_fitted_pipelines: bool = True,
    v_columns_cache_path: str | None = "artifacts/selected_v_columns.json",
) -> None:
    df = load_interim_data(data_dir)     
    cv_splits, selected_columns, X_train, _, _, y_train, _, _ = \
        temporal_train_val_test_split(
            df,
            feature_set_name,
            threshold,
            v_columns_cache_path,
        )

    results_df = run_model_search(
        X_train, y_train, cv_splits, selected_columns, feature_set_name, search_smote, use_successive_halving
    )
    top_candidates_df = shortlist_candidates(
        results_df,
        max_candidates=top_k,
        roc_auc_tolerance=shortlist_roc_auc_tolerance,
    )

    if not top_candidates_df.empty:
        best_candidate = top_candidates_df.iloc[0]
        logger.info(
            "Top CV candidate: %s with ROC-AUC %.3f",
            best_candidate["model_name"],
            best_candidate["best_cv_roc_auc"],
        )
    log_top_cv_candidates(logger, top_candidates_df, top_k=top_k)

    Path("artifacts").mkdir(parents=True, exist_ok=True)
    results_df.drop(columns=["rebuild_params"], errors="ignore").to_csv("artifacts/model_comparison.csv", index=False)
    save_top_candidates_metadata(
        top_candidates_df,
        selected_columns,
        feature_set_name,
        threshold,
        search_smote,
        shortlist_roc_auc_tolerance,
    )

    if save_fitted_pipelines:
        fit_and_save_top_candidates(
            top_candidates_df,
            X_train,
            y_train,
            selected_columns,
            feature_set_name,
            search_smote,
        )
