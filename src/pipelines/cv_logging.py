import logging
import pandas as pd


def log_cv_metrics(logger: logging.Logger, prefix: str, metrics: dict[str, float]) -> None:
    logger.info("%s CV ROC-AUC: %.3f", prefix, metrics["roc_auc"])
    logger.info("%s CV F1 Score: %.3f", prefix, metrics["f1"])
    logger.info("%s CV Accuracy: %.3f", prefix, metrics["accuracy"])
    logger.info("%s CV Recall: %.3f", prefix, metrics["recall"])
    logger.info("%s CV Precision: %.3f", prefix, metrics["precision"])
    logger.info("%s CV Average Precision Score: %.3f", prefix, metrics["average_precision"])


def log_top_cv_candidates(
    logger: logging.Logger,
    candidates_df: pd.DataFrame,
    top_k: int = 5,
) -> None:
    if candidates_df.empty:
        logger.warning("No CV candidates available to log.")
        return

    logger.info("Top %s CV candidates:", min(top_k, len(candidates_df)))

    for candidate in candidates_df.head(top_k).to_dict(orient="records"):
        logger.info(
            "Rank %s | Model: %s | ROC-AUC: %.3f | AP: %.3f | F1: %.3f | "
            "Accuracy: %.3f | Recall: %.3f | Precision: %.3f | Folds: %s | Sampler: %s",
            candidate["rank"],
            candidate["model_name"],
            candidate["best_cv_roc_auc"],
            candidate["best_cv_average_precision"],
            candidate["best_cv_f1"],
            candidate["best_cv_accuracy"],
            candidate["best_cv_recall"],
            candidate["best_cv_precision"],
            candidate["num_cv_folds_used"],
            candidate["best_sampler"],
        )
