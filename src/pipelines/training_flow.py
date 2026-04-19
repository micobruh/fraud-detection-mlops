import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV
from typing import List
import joblib
import json
import gc
from ..data import (
    load_interim_data, 
    temporal_balanced_train_test_split, 
    temporal_train_val_split
)
from ..features import (
    NumericShiftFillTransformer, 
    DataFrameOrdinalEncoder, 
    DColumnNormalizer, 
    FrequencyEncoder, 
    CombineColumnsTransformer, 
    UIDAggregationTransformer, 
    DropColumnsTransformer, 
    extract_relevant_V_columns
)
from ..models import (
    test_evaluation
)
from ..utils import (
    RANDOM_STATE, 
    TARGET_COLUMN, 
    BASE_COLUMNS, 
    V_COLUMNS, 
    CATEGORICAL_COLUMNS, 
    NUMERICAL_COLUMNS, 
    DROP_COLUMNS
)
import logging

logger = logging.getLogger(__name__)


def build_feature_pipeline() -> Pipeline:
    return Pipeline([
        ("numerical_shift_fill", NumericShiftFillTransformer(NUMERICAL_COLUMNS)),
        ("ordinal_encode", DataFrameOrdinalEncoder(CATEGORICAL_COLUMNS, handle_unknown="use_encoded_value", unknown_value=-1)),
        ("normalize_D_columns", DColumnNormalizer()),
        ("frequency_encode_og_features", FrequencyEncoder(["addr1", "card1", "card2", "card3", "P_emaildomain"])),
        ("combine_card1_addr1", CombineColumnsTransformer(["card1", "addr1"])),
        ("combine_card1_addr1_P_emaildomain", CombineColumnsTransformer(["card1_addr1", "P_emaildomain"])),
        ("frequency_encode_new_features", FrequencyEncoder(["card1_addr1", "card1_addr1_P_emaildomain"])),
        ("aggregate_UID_columns", UIDAggregationTransformer(["TransactionAmt", "D9", "D11"], 
                                                            ["card1", "card1_addr1", "card1_addr1_P_emaildomain"], 
                                                            ["mean", "std"], 
                                                            use_na_sentinel=True)),
        ("drop_columns", DropColumnsTransformer(DROP_COLUMNS))                                                    
    ], verbose=True)


def run_model_search(X_train, y_train, cv_splits, candidate_configs):
    all_results = []
    best_search = None
    best_score = float("-inf")

    for config in candidate_configs:
        search = RandomizedSearchCV(
            estimator=config["pipeline"],
            param_distributions=config["param_distributions"],
            n_iter=config.get("n_iter", 20),
            scoring="roc_auc",
            cv=cv_splits,
            n_jobs=-1,
            refit=True,
            random_state=RANDOM_STATE,
            verbose=2,
            error_score="raise",
        )

        search.fit(X_train, y_train)

        best_params = search.best_params_
        sampler_used = best_params.get("sampler", "passthrough")

        all_results.append({
            "model_name": config["name"],
            "best_cv_score": search.best_score_,
            "best_params": best_params,
            "best_estimator": search.best_estimator_,
            "best_sampler": str(sampler_used),
        })

        if search.best_score_ > best_score:
            best_score = search.best_score_
            best_search = search

    results_df = pd.DataFrame([
        {
            "model_name": r["model_name"],
            "best_cv_score": r["best_cv_score"],
            "best_sampler": r["best_sampler"],
            "best_params": str(r["best_params"]),
        }
        for r in all_results
    ]).sort_values("best_cv_score", ascending=False)

    return best_search, results_df, all_results


def main(
    data_dir: str, 
    target_column: str = TARGET_COLUMN, 
    base_columns: List[str] | None = None, 
    v_columns: List[str] | None = None, 
    extract_V_columns_needed: bool = False,
    threshold: float = 0.65
) -> None:
    df = load_interim_data(data_dir)      
    df_main, df_local_test = temporal_balanced_train_test_split(df) 
    
    if base_columns is None:
        base_columns = BASE_COLUMNS
    if v_columns is None:
        if extract_V_columns_needed:
            v_columns = extract_relevant_V_columns(df_main, target_column, v_columns, threshold)
        else:
            v_columns = V_COLUMNS       

    X_train, X_local_test = df_main[base_columns + v_columns].copy(), df_local_test[base_columns + v_columns].copy()
    y_train, y_local_test = df_main[target_column].copy(), df_local_test[target_column].copy()
    cv_splits = temporal_train_val_split(df_main)
    del df_main, df_local_test
    gc.collect()

    cv_splits = temporal_train_val_split(df_main)

    best_search, results_df, all_results = run_model_search(
        X_train=X_train,
        y_train=y_train,
        cv_splits=cv_splits,
    )

    best_pipeline = best_search.best_estimator_
    cv_roc_auc_score = best_search.best_score_
    y_test_score = best_pipeline.predict_proba(X_local_test)[:, 1]
    y_test_pred = np.where(y_test_score >= 0.5, 1, 0)
    final_roc_auc_score, final_f1_score, final_recall, final_precision, final_ap_score, final_accuracy = test_evaluation(y_local_test, y_test_score, y_test_pred)   

    logger.info(f"Best Model CV ROC-AUC: {round(cv_roc_auc_score, 3)}")
    logger.info(f"Best Model Local Test ROC-AUC: {round(final_roc_auc_score, 3)}")
    logger.info(f"Best Model Local Test F1 Score: {round(final_f1_score, 3)}")
    logger.info(f"Best Model Local Test Accuracy: {round(final_accuracy, 3)}")
    logger.info(f"Best Model Local Test Recall: {round(final_recall, 3)}")
    logger.info(f"Best Model Local Test Precision: {round(final_precision, 3)}")
    logger.info(f"Best Model Local Test Average Precision Score: {round(final_ap_score, 3)}")    
    joblib.dump(best_pipeline, "artifacts/best_pipeline.joblib")

    with open("artifacts/best_params.json", "w") as f:
        json.dump(best_search.best_params_, f, indent=2, default=str)

    results_df.to_csv("artifacts/model_comparison.csv", index=False)    