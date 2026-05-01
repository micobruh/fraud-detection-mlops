import logging


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


def log_cv_metrics(
    logger: logging.Logger,
    prefix: str,
    metrics: dict[str, float],
) -> None:
    logger.info("%s CV ROC-AUC: %.3f", prefix, metrics["roc_auc"])
    logger.info("%s CV F1 Score: %.3f", prefix, metrics["f1"])
    logger.info("%s CV Accuracy: %.3f", prefix, metrics["accuracy"])
    logger.info("%s CV Recall: %.3f", prefix, metrics["recall"])
    logger.info("%s CV Precision: %.3f", prefix, metrics["precision"])
    logger.info("%s CV Average Precision Score: %.3f", prefix, metrics["average_precision"])
