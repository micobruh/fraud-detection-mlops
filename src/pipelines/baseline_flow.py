import logging
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from ..data import (
    load_interim_data,
    temporal_train_val_test_split,
)
from ..models import (
    test_evaluation,
)
from .cv_logging import (
    log_cv_metrics,
)

logger = logging.getLogger(__name__)


def _baseline_metrics(y_true: pd.Series | np.ndarray):
    y_true = np.asarray(y_true)
    y_score = np.zeros(len(y_true), dtype=float)
    y_pred = np.zeros(len(y_true), dtype=int)

    roc_auc = float("nan") if np.unique(y_true).size < 2 else roc_auc_score(y_true, y_score)
    f1 = f1_score(y_true, y_pred, pos_label=1, zero_division=0)
    recall = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
    precision = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
    average_precision = average_precision_score(y_true, y_score)
    accuracy = accuracy_score(y_true, y_pred)

    return {
        "roc_auc": roc_auc,
        "f1": f1,
        "recall": recall,
        "precision": precision,
        "average_precision": average_precision,
        "accuracy": accuracy,
    }


def baseline_training(data_dir: str) -> None:
    df = load_interim_data(data_dir)
    cv_splits, _, _, _, _, y_train, _, _ = temporal_train_val_test_split(df)

    fold_metrics = []
    for fold_idx, (_, val_idx) in enumerate(cv_splits, start=1):
        y_val = y_train.iloc[val_idx]

        if y_val.nunique() < 2:
            logger.warning(
                "Skipping baseline CV fold %s because the validation fold has only one class.",
                fold_idx,
            )
            continue

        fold_metrics.append(_baseline_metrics(y_val))

    if not fold_metrics:
        logger.warning("No valid baseline CV folds were available for training metrics.")
        return

    averaged_metrics = {
        "roc_auc": float(np.nanmean([metrics["roc_auc"] for metrics in fold_metrics])),
        "f1": float(np.nanmean([metrics["f1"] for metrics in fold_metrics])),
        "accuracy": float(np.nanmean([metrics["accuracy"] for metrics in fold_metrics])),
        "recall": float(np.nanmean([metrics["recall"] for metrics in fold_metrics])),
        "precision": float(np.nanmean([metrics["precision"] for metrics in fold_metrics])),
        "average_precision": float(np.nanmean([metrics["average_precision"] for metrics in fold_metrics])),
    }
    log_cv_metrics(logger, "Baseline Model", averaged_metrics)


def baseline_streaming_validation(data_dir: str) -> None:
    df = load_interim_data(data_dir)
    _, _, _, _, _, _, y_stream_val, _ = temporal_train_val_test_split(df)

    y_stream_val_score = np.zeros(len(y_stream_val), dtype=float)
    y_stream_val_pred = np.zeros(len(y_stream_val), dtype=int)
    test_evaluation(
        "Baseline Model",
        "Streaming Validation",
        y_stream_val,
        y_stream_val_score,
        y_stream_val_pred,
    )


def baseline_streaming_test(data_dir: str) -> None:
    df = load_interim_data(data_dir)
    _, _, _, _, _, _, _, y_stream_test = temporal_train_val_test_split(df)

    y_stream_test_score = np.zeros(len(y_stream_test), dtype=float)
    y_stream_test_pred = np.zeros(len(y_stream_test), dtype=int)
    test_evaluation(
        "Baseline Model",
        "Streaming Test",
        y_stream_test,
        y_stream_test_score,
        y_stream_test_pred,
    )
