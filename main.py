import argparse
import logging
from pathlib import Path

from src.pipelines import (
    baseline_training,
    baseline_validation,
    baseline_test,    
    training,
    validation,
    test
)
from src.utils import (
    MLFLOW_TRAINING_EXPERIMENT_NAME,
    MLFLOW_VALIDATION_EXPERIMENT_NAME,
    MLFLOW_TEST_EXPERIMENT_NAME,    
    setup_logging,
)


logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run fraud detection ML pipelines.")
    parser.add_argument(
        "stage",
        nargs="?",
        choices=["all", "training", "validation", "test"],
        default="all",
        help="Pipeline stage to run. Defaults to all.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Optional path to the interim IEEE fraud detection data directory.",
    )
    return parser.parse_args()


def get_interim_data_dir() -> Path:
    return Path(__file__).resolve().parent / "data" / "interim" / "ieee-fraud-detection"


def log_mlflow_hint(experiment_name: str) -> None:
    logger.info(
        "MLflow tracking is enabled for experiment '%s'. You can start the UI with `mlflow ui` and open http://127.0.0.1:5000",
        experiment_name,
    )


def run_training(data_dir: str | Path | None = None) -> None:
    setup_logging()
    log_mlflow_hint(MLFLOW_TRAINING_EXPERIMENT_NAME)

    interim_data_dir = data_dir or get_interim_data_dir()
    baseline_training(interim_data_dir)
    training(interim_data_dir)
    logger.info("Training complete. MLflow runs are available in experiment '%s'.", MLFLOW_TRAINING_EXPERIMENT_NAME)


def run_validation(data_dir: str | Path | None = None) -> None:
    setup_logging()
    log_mlflow_hint(MLFLOW_VALIDATION_EXPERIMENT_NAME)

    interim_data_dir = data_dir or get_interim_data_dir()
    baseline_validation(interim_data_dir)
    validation(interim_data_dir)
    logger.info("Validation complete. MLflow runs are available in experiment '%s'.", MLFLOW_VALIDATION_EXPERIMENT_NAME)


def run_test(data_dir: str | Path | None = None) -> None:
    setup_logging()
    log_mlflow_hint(MLFLOW_TEST_EXPERIMENT_NAME)

    interim_data_dir = data_dir or get_interim_data_dir()
    baseline_test(interim_data_dir)
    test(interim_data_dir)
    logger.info("Test complete. MLflow runs are available in experiment '%s'.", MLFLOW_TEST_EXPERIMENT_NAME)


def main() -> None:
    setup_logging()

    args = parse_args()
    interim_data_dir = args.data_dir or get_interim_data_dir()

    if args.stage in {"all", "training"}:
        log_mlflow_hint(MLFLOW_TRAINING_EXPERIMENT_NAME)
        baseline_training(interim_data_dir)
        training(interim_data_dir)
        logger.info("Training complete. MLflow runs are available in experiment '%s'.", MLFLOW_TRAINING_EXPERIMENT_NAME)

    if args.stage in {"all", "validation"}:
        log_mlflow_hint(MLFLOW_VALIDATION_EXPERIMENT_NAME)
        baseline_validation(interim_data_dir)
        validation(interim_data_dir)
        logger.info("Validation complete. MLflow runs are available in experiment '%s'.", MLFLOW_VALIDATION_EXPERIMENT_NAME)

    if args.stage in {"all", "test"}:
        log_mlflow_hint(MLFLOW_TEST_EXPERIMENT_NAME)
        baseline_test(interim_data_dir)
        test(interim_data_dir)
        logger.info("Test complete. MLflow runs are available in experiment '%s'.", MLFLOW_TEST_EXPERIMENT_NAME)



if __name__ == "__main__":
    main()
