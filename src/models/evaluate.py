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

def test_evaluation(
    y_local_test: pd.Series | np.ndarray,
    y_test_score: np.ndarray,
    y_test_pred: np.ndarray
) -> Tuple[float, float, float, float, float, float]:
    y_local_test = np.asarray(y_local_test)
    y_test_score = np.asarray(y_test_score)
    y_test_pred = np.asarray(y_test_pred)    
    
    if y_local_test.nunique() < 2:
        final_roc_auc_score = float("nan")
    else:
        final_roc_auc_score = roc_auc_score(y_local_test, y_test_score)
    
    final_f1_score = f1_score(y_local_test, y_test_pred, pos_label=1, zero_division=0)
    final_recall = recall_score(y_local_test, y_test_pred, pos_label=1, zero_division=0)
    final_precision = precision_score(y_local_test, y_test_pred, pos_label=1, zero_division=0)
    final_ap_score = average_precision_score(y_local_test, y_test_score)  
    final_accuracy = accuracy_score(y_local_test, y_test_pred)    
    
    return (
        final_roc_auc_score, 
        final_f1_score, 
        final_recall, 
        final_precision, 
        final_ap_score, 
        final_accuracy
    )