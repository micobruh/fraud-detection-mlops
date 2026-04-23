from pathlib import Path
from src.pipelines import baseline_training, training
from src.utils import setup_logging

def main() -> None:
    setup_logging()

    project_root = Path(__file__).resolve().parent
    interim_data_dir = project_root / "data" / "interim" / "ieee-fraud-detection"

    baseline_training(interim_data_dir)
    training(interim_data_dir)


if __name__ == "__main__":
    main()
