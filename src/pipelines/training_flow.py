import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.experimental import enable_halving_search_cv  # noqa: F401
from sklearn.model_selection import HalvingRandomSearchCV
from sklearn.model_selection import RandomizedSearchCV
from typing import List
import joblib
import json
from ..data import (
    load_interim_data,
    temporal_train_val_test_split
)
from ..features import (
    NumericShiftFillTransformer, 
    DataFrameOrdinalEncoder, 
    DColumnNormalizer, 
    DropColumnsTransformer
)
from ..models import (
    test_evaluation,
    get_candidate_configs,
    build_pipeline_from_config,
)
from ..utils import (
    RANDOM_STATE, 
    CATEGORICAL_COLUMNS, 
    NUMERICAL_COLUMNS, 
    DROP_COLUMNS
)
import logging

logger = logging.getLogger(__name__)


def build_feature_pipeline(selected_columns: List[str]) -> Pipeline:
    numeric_columns = [col for col in NUMERICAL_COLUMNS if col in selected_columns + DROP_COLUMNS]
    categorical_columns = [col for col in CATEGORICAL_COLUMNS if col in selected_columns and col not in DROP_COLUMNS]

    return Pipeline([
        # Keep the feature pipeline row-local and stateless at inference time.
        ("numerical_shift_fill", NumericShiftFillTransformer(numeric_columns)),
        ("normalize_D_columns", DColumnNormalizer()),
        ("drop_columns", DropColumnsTransformer(DROP_COLUMNS, copy=False)),
        ("ordinal_encode", DataFrameOrdinalEncoder(categorical_columns, handle_unknown="use_encoded_value", unknown_value=-1)),
    ], verbose=True)


def run_model_search(
    X_train,
    y_train,
    cv_splits,
    selected_columns,
    use_successive_halving: bool = True,
):
    all_results = []
    best_model_name = None
    best_params = None
    best_score = float("-inf")

    candidate_configs = get_candidate_configs(selected_columns)
    search_cls = HalvingRandomSearchCV if use_successive_halving else RandomizedSearchCV

    for config in candidate_configs:
        search_kwargs = {
            "estimator": config["pipeline"],
            "param_distributions": config["param_distributions"],
            "scoring": "roc_auc",
            "cv": cv_splits,
            "n_jobs": -1,
            "refit": True,
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

        best_params = search.best_params_
        sampler_used = best_params.get("sampler", "passthrough")

        all_results.append({
            "model_name": config["name"],
            "best_cv_score": search.best_score_,
            "best_params": str(best_params),
            "best_sampler": str(sampler_used),
        })

        if search.best_score_ > best_score:
            best_score = search.best_score_
            best_model_name = config["name"]
            best_params = search.best_params_

        del search

    results_df = pd.DataFrame(all_results).sort_values("best_cv_score", ascending=False)
    return best_model_name, best_score, best_params, results_df


def training(
    data_dir: str, 
    extract_V_columns_needed: bool = False,
    threshold: float = 0.65,
    use_successive_halving: bool = True,
    v_columns_cache_path: str | None = "artifacts/selected_v_columns.json",
) -> None:
    df = load_interim_data(data_dir)     
    cv_splits, selected_columns, X_train, X_stream_val, X_stream_test, y_train, y_stream_val, y_stream_test = \
        temporal_train_val_test_split(
            df,
            extract_V_columns_needed,
            threshold,
            v_columns_cache_path,
        )

    best_model_name, cv_roc_auc_score, best_params, results_df = run_model_search(
        X_train, y_train, cv_splits, selected_columns, use_successive_halving
    )
    best_pipeline = build_pipeline_from_config(selected_columns, best_model_name, best_params)
    best_pipeline.fit(X_train, y_train)
    logger.info(f"Best Model CV ROC-AUC: {round(cv_roc_auc_score, 3)}")

    y_stream_val_score = best_pipeline.predict_proba(X_stream_val)[:, 1]
    y_stream_val_pred = np.where(y_stream_val_score >= 0.5, 1, 0)
    val_roc_auc_score, val_f1_score, val_recall, val_precision, val_ap_score, val_accuracy = \
        test_evaluation("Best Model", "Streaming Validation", y_stream_val, y_stream_val_score, y_stream_val_pred)   
    joblib.dump(best_pipeline, "artifacts/best_pipeline.joblib")

    with open("artifacts/best_params.json", "w") as f:
        json.dump(
            {"model_name": best_model_name, "best_params": best_params},
            f,
            indent=2,
            default=str,
        )

    results_df.to_csv("artifacts/model_comparison.csv", index=False)
