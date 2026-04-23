import numpy as np
from sklearn.metrics import roc_auc_score
from ..data import (
    load_interim_data, 
    temporal_train_val_test_split
)
from ..models import (
    test_evaluation
)
from ..utils import (
    TARGET_COLUMN,
    TIME_COLUMN
)
import logging

logger = logging.getLogger(__name__)

def baseline_evaluation(data_dir: str) -> None:
    df = load_interim_data(data_dir)      
    cv_splits, _, _, _, _, y_train, y_stream_val, y_stream_test = \
    temporal_train_val_test_split(df) 

    fold_scores = []

    for _, val_idx in cv_splits:
        y_val = y_train.iloc[val_idx]
        y_val_score = np.zeros(len(y_val), dtype=float)

        if y_val.nunique() < 2:
            logger.warning("Skipping CV fold for ROC-AUC because y_val has only one class.")
            continue

        fold_scores.append(roc_auc_score(y_val, y_val_score))

    cv_roc_auc_score = float("nan") if not fold_scores else sum(fold_scores) / len(fold_scores)
    logger.info(f"Baseline Model CV ROC-AUC: {round(cv_roc_auc_score, 3)}")

    y_stream_val_score = np.zeros(len(y_stream_val), dtype=float)
    y_stream_val_pred = np.zeros(len(y_stream_val), dtype=int)
    val_roc_auc_score, val_f1_score, val_recall, val_precision, val_ap_score, val_accuracy = \
        test_evaluation("Baseline Model", "Streaming Validation", y_stream_val, y_stream_val_score, y_stream_val_pred)   
  
    # Baseline CV ROC-AUC: 0.5
    # Baseline Local Test ROC-AUC: 0.5
    # Baseline Local Test F1 Score: 0.0
    # Baseline Local Test Recall: 0.0
    # Baseline Local Test Precision: 0.0
    # Baseline Local Test Average Precision Score: 0.035    
    # Baseline Local Test Accuracy: 0.965    