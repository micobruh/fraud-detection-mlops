from pathlib import Path
import pandas as pd

from ..utils import resolve_project_path


INTERIM_DATA_DIR = resolve_project_path("data/interim/ieee-fraud-detection")


def load_interim_data(data_dir: str | Path | None = None) -> pd.DataFrame:
    """Load the training parquet files."""
    base_dir = resolve_project_path(data_dir) if data_dir is not None else INTERIM_DATA_DIR
    df = pd.read_parquet(base_dir / "train.parquet")
    return df
