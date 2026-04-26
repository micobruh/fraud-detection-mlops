from pathlib import Path
import pandas as pd
import numpy as np

from ..utils import setup_logging


ROOT_DIR = Path(__file__).resolve().parents[2]
INTERIM_DATA_DIR = ROOT_DIR / "data" / "interim" / "ieee-fraud-detection"
setup_logging()


def load_interim_data(data_dir: Path | None = None) -> pd.DataFrame:
    """Load the training parquet files."""
    base_dir = data_dir or INTERIM_DATA_DIR
    df = pd.read_parquet(base_dir / "train.parquet")
    return df