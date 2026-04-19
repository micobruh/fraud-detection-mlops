from pathlib import Path
from src.pipelines import baseline_evaluation
from src.utils import setup_logging

setup_logging()

PROJECT_ROOT = Path(__file__).resolve().parent
INTERIM_DATA_DIR = PROJECT_ROOT / "data" / "interim" / "ieee-fraud-detection"
baseline_evaluation(INTERIM_DATA_DIR)