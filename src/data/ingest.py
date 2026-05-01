import logging
import pandas as pd
import numpy as np
from tqdm import tqdm

from ..utils import ID_COLUMN, resolve_project_path, setup_logging


setup_logging()
logger = logging.getLogger(__name__)


def load_csv(csv_file: str) -> pd.DataFrame:
    """Load a CSV file in chunks and return a concatenated DataFrame."""
    csv_path = resolve_project_path(csv_file)
    try:
        chunks = pd.read_csv(csv_path, index_col=ID_COLUMN, chunksize=100000)
    except Exception:
        logger.exception(f"CSV File not found: {csv_path}")
        raise FileNotFoundError(f"CSV File not found: {csv_path}")
    df = pd.concat(chunks) 
    return df


def reduce_memory_usage(df: pd.DataFrame) -> pd.DataFrame:
    for col in tqdm(df.columns):
        col_type = df[col].dtype

        if pd.api.types.is_object_dtype(col_type):
            num_unique = df[col].nunique(dropna=False)
            num_total = len(df[col])
            if num_unique / num_total < 0.3:
                df[col] = df[col].astype("category")

        elif pd.api.types.is_integer_dtype(col_type):
            df[col] = pd.to_numeric(df[col], downcast="integer")

        elif pd.api.types.is_float_dtype(col_type):
            df[col] = pd.to_numeric(df[col], downcast="float")

    return df


def convert_to_parquet(init_relative_path: str, interim_relative_path: str) -> None:
    """Convert the raw CSV file to Parquet format for efficient storage and faster loading."""
    logger.debug("Data loading started")
    transaction_csv = init_relative_path + "_transaction.csv"
    identity_csv = init_relative_path + "_identity.csv"

    df_transaction = load_csv(transaction_csv)
    logger.debug("Transaction data loaded") 

    df_identity = load_csv(identity_csv)
    logger.debug("Identity data loaded")   

    df_identity = df_identity.rename(columns=lambda col: str(col).replace("-", "_"))
    df = df_transaction.join(df_identity, how="left")
    logger.debug("Data merged")

    df = reduce_memory_usage(df)
    logger.debug("Data types changed for memory efficiency")

    interim_parquet = resolve_project_path(interim_relative_path + ".parquet")
    interim_parquet.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(interim_parquet)


def conversion() -> None:
    links = [f"data/{item}/ieee-fraud-detection/" for item in ["raw", "interim"]]
    convert_to_parquet(f"{links[0]}train", f"{links[1]}train")  
    logger.info("Training data conversion from CSV to Parquet completed") 
     
    convert_to_parquet(f"{links[0]}test", f"{links[1]}test")
    logger.info("Test data conversion from CSV to Parquet completed")


if __name__ == "__main__":
    conversion()
# python -m src.data.ingest
