import pandas as pd
from pathlib import Path
import logging
from ..utils import ID_COLUMN, setup_logging

# setup_logging()
logger = logging.getLogger(__name__)

def load_csv(csv_file: str) -> pd.DataFrame:
    """Load a CSV file in chunks and return a concatenated DataFrame."""
    root_path = Path(__file__).resolve().parents[2]
    csv_path = root_path / Path(csv_file)
    try:
        chunks = pd.read_csv(str(csv_path), index_col=ID_COLUMN, chunksize=100000)
    except Exception:
        logger.exception(f"CSV File not found: {csv_path}")
        raise FileNotFoundError(f"CSV File not found: {csv_path}")
    df = pd.concat(chunks) 
    return df

def convert_to_parquet(init_relative_path: str, interim_relative_path: str) -> None:
    """Convert the raw CSV file to Parquet format for efficient storage and faster loading."""
    transaction_csv = init_relative_path + "_transaction.csv"
    identity_csv = init_relative_path + "_identity.csv"

    df_transaction = load_csv(transaction_csv)
    logger.debug("Transaction data loaded") 

    df_identity = load_csv(identity_csv)
    logger.debug("Identity data loaded")   

    df_identity = df_identity.rename(columns=lambda col: str(col).replace("-", "_"))
    df = df_transaction.join(df_identity, how="left")
    logger.debug("Data merged")

    interim_parquet = interim_relative_path + ".parquet"
    df.to_parquet(interim_parquet)

def conversion() -> None:
    links = [f"data/{item}/ieee-fraud-detection/" for item in ["raw", "interim"]]
    convert_to_parquet(f"{links[0]}train", f"{links[1]}train")  
    logger.info("Training data conversion from CSV to Parquet completed") 
     
    convert_to_parquet(f"{links[0]}test", f"{links[1]}test")
    logger.info("Test data conversion from CSV to Parquet completed")

# if __name__ == "__main__":
#     conversion()
# python -m src.data.ingest