from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score
from Typing import List
import gc
from ..data import load_interim_data, temporal_balanced_train_test_split, temporal_train_val_split
from ..features import NumericShiftFillTransformer, DataFrameOrdinalEncoder, DColumnNormalizer, FrequencyEncoder, CombineColumnsTransformer, UIDAggregationTransformer, DropColumnsTransformer, extract_relevant_V_columns
from ..models import build_full_pipeline
from ..utils import TARGET_COLUMN, BASE_COLUMNS, V_COLUMNS_USED, CATEGORICAL_COLUMNS, NUMERICAL_COLUMNS, DROP_COLUMNS

import logging

logger = logging.getLogger(__name__)

def build_feature_pipeline() -> Pipeline:
    return Pipeline([
        ("numerical_shift_fill", NumericShiftFillTransformer(NUMERICAL_COLUMNS))
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
            v_columns = V_COLUMNS_USED        

    X_train, X_local_test = df_main[base_columns + v_columns].copy(), df_local_test[base_columns + v_columns].copy()
    y_train, y_local_test = df_main[target_column].copy(), df_local_test[target_column].copy()
    train_val_id_pairs = temporal_train_val_split(df_main)
    del df_main, df_local_test
    gc.collect()

    avg_score = 0
    for idx, (train_ids, val_ids) in enumerate(train_val_id_pairs):
        X_tr, X_val = X_train.loc[train_ids], X_train.loc[val_ids]
        y_tr, y_val = y_train.loc[train_ids], y_train.loc[val_ids]
        
        feature_pipeline = build_feature_pipeline()
        pipeline = build_full_pipeline(feature_pipeline)
        pipeline.fit(X_tr, y_tr)

        y_val_pred = pipeline.predict_proba(X_val)[:, 1]
        fold_score = roc_auc_score(y_val, y_val_pred)
        avg_score += fold_score
        logger.info(f"CV Fold {idx} Score: {round(fold_score, 2)}")

        del X_tr, X_val, y_tr, y_val, y_val_pred
        gc.collect()

    avg_score /= len(train_val_id_pairs)
    logger.info(f"Average CV Score: {round(avg_score, 2)}")

    final_feature_pipeline = build_feature_pipeline()
    final_pipeline = build_full_pipeline(final_feature_pipeline)
    final_pipeline.fit(X_train, y_train)

    y_test_pred = final_pipeline.predict_proba(X_local_test)[:, 1]    
    final_score = roc_auc_score(y_local_test, y_test_pred)
    logger.info(f"Final Test Score: {round(final_score, 2)}")