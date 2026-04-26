import pandas as pd
import numpy as np
from sklearn.metrics import (
    roc_auc_score, 
    f1_score, 
    accuracy_score, 
    average_precision_score, 
    recall_score, 
    precision_score
)
from typing import Tuple
import logging


logger = logging.getLogger(__name__)


def test_evaluation(
    model_name: str,
    test_stage: str,
    y_stream: pd.Series | np.ndarray,
    y_stream_score: np.ndarray,
    y_stream_pred: np.ndarray
) -> Tuple[float, float, float, float, float, float]:
    y_stream = np.asarray(y_stream)
    y_stream_score = np.asarray(y_stream_score)
    y_stream_pred = np.asarray(y_stream_pred)    
    
    if np.unique(y_stream).size < 2:
        final_roc_auc_score = float("nan")
    else:
        final_roc_auc_score = roc_auc_score(y_stream, y_stream_score)
    
    final_f1_score = f1_score(y_stream, y_stream_pred, pos_label=1, zero_division=0)
    final_recall = recall_score(y_stream, y_stream_pred, pos_label=1, zero_division=0)
    final_precision = precision_score(y_stream, y_stream_pred, pos_label=1, zero_division=0)
    final_ap_score = average_precision_score(y_stream, y_stream_score)  
    final_accuracy = accuracy_score(y_stream, y_stream_pred)    

    logger.info(f"{model_name} {test_stage} ROC-AUC: {round(final_roc_auc_score, 3)}")
    logger.info(f"{model_name} {test_stage} F1 Score: {round(final_f1_score, 3)}")
    logger.info(f"{model_name} {test_stage} Accuracy: {round(final_accuracy, 3)}")
    logger.info(f"{model_name} {test_stage} Recall: {round(final_recall, 3)}")
    logger.info(f"{model_name} {test_stage} Precision: {round(final_precision, 3)}")
    logger.info(f"{model_name} {test_stage} Average Precision Score: {round(final_ap_score, 3)}")       
    
    return (
        final_roc_auc_score, 
        final_f1_score, 
        final_recall, 
        final_precision, 
        final_ap_score, 
        final_accuracy
    )