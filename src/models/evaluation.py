import pandas as pd
import numpy as np
from sklearn.metrics import (
    roc_auc_score, 
    f1_score, 
    accuracy_score, 
    average_precision_score, 
    recall_score, 
    precision_score,
    precision_recall_curve
)
import logging


logger = logging.getLogger(__name__)


def compute_classification_metric(
    model_name: str,
    test_stage: str,
    y: pd.Series | np.ndarray,
    y_scores: np.ndarray,
    y_preds: np.ndarray
) -> dict[str, float]:
    y = np.asarray(y)
    y_scores = np.asarray(y_scores)
    y_preds = np.asarray(y_preds)    
    
    if np.unique(y).size < 2:
        final_roc_auc_score = float("nan")
    else:
        final_roc_auc_score = roc_auc_score(y, y_scores)
    
    final_f1_score = f1_score(y, y_preds, pos_label=1, zero_division=0)
    final_recall = recall_score(y, y_preds, pos_label=1, zero_division=0)
    final_precision = precision_score(y, y_preds, pos_label=1, zero_division=0)
    final_ap_score = average_precision_score(y, y_scores)  
    final_accuracy = accuracy_score(y, y_preds)    

    logger.info(f"{model_name} {test_stage} ROC-AUC: {round(final_roc_auc_score, 3)}")
    logger.info(f"{model_name} {test_stage} F1 Score: {round(final_f1_score, 3)}")
    logger.info(f"{model_name} {test_stage} Accuracy: {round(final_accuracy, 3)}")
    logger.info(f"{model_name} {test_stage} Recall: {round(final_recall, 3)}")
    logger.info(f"{model_name} {test_stage} Precision: {round(final_precision, 3)}")
    logger.info(f"{model_name} {test_stage} Average Precision Score: {round(final_ap_score, 3)}")       

    return {
        "roc_auc": final_roc_auc_score,
        "f1": final_f1_score,
        "recall": final_recall,
        "precision": final_precision,
        "average_precision": final_ap_score,
        "accuracy": final_accuracy,
    }


def select_threshold_by_f1(y_true: np.ndarray, y_score: np.ndarray) -> dict:
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)

    f1 = 2 * precision * recall / (precision + recall + 1e-12)

    best_idx = f1[:-1].argmax()

    return {
        "threshold": float(thresholds[best_idx]),
        "f1": float(f1[best_idx]),
        "precision": float(precision[best_idx]),
        "recall": float(recall[best_idx]),
        "selection_metric": "max_validation_f1",
    }
