import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
import gc
from ..data import (
    load_interim_data, 
    temporal_balanced_train_test_split, 
    temporal_train_val_split
)
from ..models import (
    test_evaluation
)
from ..utils import (
    TARGET_COLUMN
)
import logging

logger = logging.getLogger(__name__)

def baseline_evaluation(
    data_dir: str, 
    target_column: str = TARGET_COLUMN, 
) -> None:
    df = load_interim_data(data_dir)      
    df_main, df_local_test = temporal_balanced_train_test_split(df) 

    y_train, y_local_test = df_main[target_column].copy(), df_local_test[target_column].copy()
    cv_splits = temporal_train_val_split(df_main)
    del df_main, df_local_test
    gc.collect()

    cv_roc_auc_score = 0
    for split in cv_splits:
        y_val = y_train.loc[split[1]]
        y_val_score = np.zeros(len(y_val), dtype=float)
        # ROC AUC is undefined if y_val has only one class
        if y_val.nunique() < 2:
            logger.warning("Skipping CV fold for ROC-AUC because y_val has only one class.")
            continue
        cv_roc_auc_score += roc_auc_score(y_val, y_val_score)
    cv_roc_auc_score /= len(cv_splits)

    y_test_score = np.zeros(len(y_local_test), dtype=float)
    y_test_pred = np.zeros(len(y_local_test), dtype=int)
    final_roc_auc_score, final_f1_score, final_recall, final_precision, final_ap_score, final_accuracy = test_evaluation(y_local_test, y_test_score, y_test_pred)   

    logger.info(f"Baseline CV ROC-AUC: {round(cv_roc_auc_score, 3)}")
    logger.info(f"Baseline Local Test ROC-AUC: {round(final_roc_auc_score, 3)}")
    logger.info(f"Baseline Local Test F1 Score: {round(final_f1_score, 3)}")
    logger.info(f"Baseline Local Test Accuracy: {round(final_accuracy, 3)}")
    logger.info(f"Baseline Local Test Recall: {round(final_recall, 3)}")
    logger.info(f"Baseline Local Test Precision: {round(final_precision, 3)}")
    logger.info(f"Baseline Local Test Average Precision Score: {round(final_ap_score, 3)}")    
    # Baseline CV ROC-AUC: 0.5
    # Baseline Local Test ROC-AUC: 0.5
    # Baseline Local Test F1 Score: 0.0
    # Baseline Local Test Recall: 0.0
    # Baseline Local Test Precision: 0.0
    # Baseline Local Test Average Precision Score: 0.035    
    # Baseline Local Test Accuracy: 0.965    