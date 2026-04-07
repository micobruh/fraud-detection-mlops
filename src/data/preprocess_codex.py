from __future__ import annotations

from pathlib import Path
from typing import Tuple

import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[2]
INTERIM_DATA_DIR = ROOT_DIR / "data" / "interim" / "ieee-fraud-detection"
TARGET_COLUMN = "isFraud"
ID_COLUMN = "TransactionID"


def normalize_identity_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize identity feature names across train/test files."""
    return df.rename(columns=lambda col: str(col).replace("-", "_"))


def load_interim_data(data_dir: Path | None = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load the training transaction and identity parquet files."""
    base_dir = data_dir or INTERIM_DATA_DIR
    transaction_df = pd.read_parquet(base_dir / "train_transaction.parquet")
    identity_df = pd.read_parquet(base_dir / "train_identity.parquet")
    return transaction_df, normalize_identity_columns(identity_df)


def merge_transaction_and_identity(
    transaction_df: pd.DataFrame,
    identity_df: pd.DataFrame,
) -> pd.DataFrame:
    """Left join identity data onto the transaction table."""
    identity_df = normalize_identity_columns(identity_df)
    return transaction_df.merge(identity_df, on=ID_COLUMN, how="left")


def add_missing_indicators(
    df: pd.DataFrame,
    columns: list[str],
    *,
    threshold: float = 0.2,
) -> pd.DataFrame:
    """
    Add binary indicators for columns with meaningful missingness.

    Missingness was informative in the EDA, so we only add indicators
    where the null rate exceeds the threshold.
    """
    result = df.copy()
    for column in columns:
        if column not in result.columns:
            continue
        missing_rate = result[column].isna().mean()
        if missing_rate >= threshold:
            result[f"{column}_missing"] = result[column].isna().astype("int8")
    return result


def reduce_memory_footprint(df: pd.DataFrame) -> pd.DataFrame:
    """Downcast numeric columns and coerce object columns to category."""
    result = df.copy()

    for column in result.select_dtypes(include=["int64", "int32"]).columns:
        result[column] = pd.to_numeric(result[column], downcast="integer")

    for column in result.select_dtypes(include=["float64"]).columns:
        result[column] = pd.to_numeric(result[column], downcast="float")

    for column in result.select_dtypes(include=["object"]).columns:
        result[column] = result[column].astype("category")

    return result
