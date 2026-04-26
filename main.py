import logging
from pathlib import Path

from src.pipelines import baseline_training, training
from src.utils import MLFLOW_EXPERIMENT_NAME, setup_logging


logger = logging.getLogger(__name__)


def main() -> None:
    setup_logging()

    project_root = Path(__file__).resolve().parent
    interim_data_dir = project_root / "data" / "interim" / "ieee-fraud-detection"

    logger.info(
        "MLflow tracking is enabled for experiment '%s'. You can start the UI with `mlflow ui` and open http://127.0.0.1:5000",
        MLFLOW_EXPERIMENT_NAME,
    )
    baseline_training(interim_data_dir)
    training(interim_data_dir)
    logger.info("Training complete. MLflow runs are available in experiment '%s'.", MLFLOW_EXPERIMENT_NAME)



if __name__ == "__main__":
    main()
